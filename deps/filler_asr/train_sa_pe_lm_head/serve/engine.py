"""Load-once inference engine for a trained SA + PE + lm_head filler-ASR head.

No FastAPI here — this module can be used standalone (see __main__ smoke test).
Model construction and decode are the SAME code path as ../infer.py:

    * num_sa_layers counted from the `sa.layers.<i>.*` keys in the checkpoint
    * tokenizer (33-tok char vocab incl <fill>) loaded from the run dir
    * pred = logits.argmax(-1); decode with skip_special_tokens=True, group_tokens=False
    * hyp          = decode capped at the FIRST </s> frame  (the "real" output)
    * hyp_uncapped = decode of the full frame axis          (pure unstripped hypothesis)

Extra per-utterance geometry returned for the API:
    duration_s   : audio length in seconds
    total_frames : encoder frame count T (~50 fps; logits length)
    n_keep       : duration budget ceil(duration_s * frames_per_sec) — same rule as the
                   training collator (frames_per_sec recovered from ckpt args, default 25)
"""
import os
import sys
import math
import time

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_TRAIN_DIR = os.path.dirname(_HERE)                       # train_sa_pe_lm_head/
_ROOT = os.path.dirname(_TRAIN_DIR)                       # filler_asr/
for p in (_TRAIN_DIR, _ROOT):
    if p not in sys.path:
        sys.path.insert(0, p)

from transformers import Wav2Vec2FeatureExtractor                                  # noqa: E402
from filler_sa_reference import FILL_TOKEN, SAMPLING_RATE, build_tokenizer, build_model  # noqa: E402
from infer import count_sa_layers, DEFAULT_CKPT                                    # noqa: E402


class Engine:
    """One loaded model + tokenizer. Build with Engine.load(); call .transcribe(wave)."""

    def __init__(self, model, tok, fe, device, fps, info):
        self.model = model
        self.tok = tok
        self.fe = fe
        self.device = device
        self.fps = fps
        self.info = info
        self.eos_id = tok.eos_token_id

    @classmethod
    def load(cls, run_dir, ckpt="", model_checkpoint="", device=None):
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        ckpt = ckpt or os.path.join(run_dir, "best.pt")

        ck = torch.load(ckpt, map_location="cpu")
        head_sd = ck["head"]
        cargs = ck.get("args", {}) or {}
        n_sa = count_sa_layers(head_sd)
        enc = model_checkpoint or cargs.get("model_checkpoint") or DEFAULT_CKPT
        use_pe = bool(cargs.get("pe", True))
        fps = int(cargs.get("frames_per_sec", 25))

        tok = build_tokenizer(run_dir)
        fe = Wav2Vec2FeatureExtractor(feature_size=1, sampling_rate=SAMPLING_RATE,
                                      padding_value=0.0, do_normalize=True,
                                      return_attention_mask=True)

        model = build_model(enc, tok, num_sa_layers=n_sa, use_sinusoidal_pe=use_pe, device=device)
        missing, unexpected = model.load_state_dict(head_sd, strict=False)
        bad = list(unexpected) + [k for k in missing if k.startswith(("sa.", "lm_head."))]
        assert not bad, f"head load mismatch: {bad[:6]}"
        model.eval()

        info = {
            "run_dir": run_dir,
            "ckpt": ckpt,
            "num_sa_layers": n_sa,
            "use_pe": use_pe,
            "encoder": enc,
            "frames_per_sec": fps,
            "trained_step": ck.get("step"),
            "best_wer": ck.get("best_wer"),
            "device": device,
            "sampling_rate": SAMPLING_RATE,
        }
        print(f"[engine] {ckpt}\n[engine] num_sa_layers={n_sa} pe={use_pe} fps={fps} "
              f"encoder={enc} device={device} "
              f"(trained step={ck.get('step')} best_wer={ck.get('best_wer')})", flush=True)
        return cls(model, tok, fe, device, fps, info)

    def warmup(self, seconds=1.0):
        """One throwaway forward so the first real request doesn't pay CUDA init/alloc cost."""
        self.transcribe(np.zeros(int(seconds * SAMPLING_RATE), dtype=np.float32))

    @torch.no_grad()
    def transcribe(self, wave, include_frame_tokens=False):
        """wave: mono float32 numpy array at 16 kHz. Returns the result dict."""
        t0 = time.time()
        duration_s = len(wave) / SAMPLING_RATE

        # feature-extract + pad exactly like DataCollatorFillerASR (batch of 1)
        iv = self.fe(wave, sampling_rate=SAMPLING_RATE).input_values[0]
        batch = self.fe.pad([{"input_values": iv}], padding=True, return_tensors="pt")
        batch = {k: v.to(self.device) for k, v in batch.items()}

        if self.device == "cuda":
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits = self.model(**batch).logits.float()
        else:
            logits = self.model(**batch).logits.float()

        pred = logits.argmax(-1)[0]
        total_frames = int(pred.shape[0])
        n_keep = min(math.ceil(duration_s * self.fps), total_frames)

        hit = (pred == self.eos_id).nonzero(as_tuple=True)[0]        # cap at first </s>
        cap = pred[:hit[0]] if hit.numel() else pred
        hyp = self.tok.decode(cap.tolist(), skip_special_tokens=True, group_tokens=False).strip()
        hyp_uncapped = self.tok.decode(pred.tolist(), skip_special_tokens=True,
                                       group_tokens=False).strip()
        frame_toks = self.tok.convert_ids_to_tokens(pred.tolist())   # one token per frame, nothing removed
        hyp_raw = "".join(frame_toks)

        out = {
            "hyp": hyp,
            "hyp_uncapped": hyp_uncapped,
            "hyp_raw": hyp_raw,
            "leaked": hyp != hyp_uncapped,
            "duration_s": round(duration_s, 3),
            "total_frames": total_frames,
            "n_keep": n_keep,
            "eos_frame": int(hit[0]) if hit.numel() else None,
            "inference_ms": round((time.time() - t0) * 1000, 1),
        }
        out["rtf"] = round((out["inference_ms"] / 1000) / max(duration_s, 1e-6), 4)
        if include_frame_tokens:
            out["frame_tokens"] = frame_toks
        return out


if __name__ == "__main__":
    # smoke: transcribe one wav and print the result dict
    import json
    import argparse
    import soundfile as sf

    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--ckpt", default="")
    ap.add_argument("--wav", required=True)
    args = ap.parse_args()

    eng = Engine.load(args.run_dir, args.ckpt)
    arr, sr = sf.read(args.wav, dtype="float32")
    if arr.ndim > 1:
        arr = arr.mean(axis=1)
    assert sr == SAMPLING_RATE, f"expected 16kHz, got {sr}"
    print(json.dumps(eng.transcribe(arr), indent=2))

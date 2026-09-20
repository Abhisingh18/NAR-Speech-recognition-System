"""Decode the SA checkpoint (FillerHubertSAModel) on test-clean.

For every utterance it PRINTS (to stdout):
  * Ref  -- normalized reference transcript
  * Hyp  -- decoded transcript (special tokens stripped, no CTC repeat-merge)
  * Full -- the raw per-frame argmax, ONE token per ~20 ms frame, with NOTHING
            stripped: <pad>, <s>, </s>, <fill> and the word-delimiter | are all
            kept exactly as emitted. (Single-utterance forward -> the only <pad>
            frames you see are ones the model itself predicted, not batch padding.)

At the end it prints corpus WER, SER and CER:
  * WER -- word edit distance / total ref words   (corpus-aggregated)
  * SER -- fraction of utterances whose Hyp != Ref exactly
  * CER -- char edit distance / total ref chars    (corpus-aggregated)

Model class, tokenizer build, wav load, ref normalization and Levenshtein are
reused unchanged from infer_sa_test_clean.py / infer_test_clean_local.py -- only
the reporting (stdout + SER) differs.

Run (pin a free GPU yourself; checkpoint defaults to the latest under the lazy run):
  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=2 \
      conda run -n filler_asr python decode_sa_test_clean.py --limit 5      # smoke
  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=2 \
      conda run -n filler_asr python decode_sa_test_clean.py               # full 2620
  ... --checkpoint experiments/Hubert_SA_2gpu_lazy_lr2e3/checkpoint-4000   # pick a ckpt
"""
import os
import json
import argparse

import torch
from tqdm import tqdm
from transformers import Wav2Vec2FeatureExtractor, HubertConfig

from model_sa import FillerHubertSAModel
from infer_test_clean_local import (
    build_tokenizer,
    load_wav,
    normalize_ref,
    levenshtein,
)

HERE = os.path.dirname(os.path.abspath(__file__))
FILL_TOKEN = "<fill>"
DEFAULT_CKPT = os.path.join(HERE, "experiments", "Hubert_SA_2gpu_lazy_lr2e3", "checkpoint-4000")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=DEFAULT_CKPT,
                    help="path to a FillerHubertSAModel checkpoint dir")
    ap.add_argument("--manifest", default=os.path.join(HERE, "data", "test_clean.jsonl"))
    ap.add_argument("--out_txt", default=None,
                    help="also tee the per-utt Ref/Hyp/Full + summary to this file")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--limit", type=int, default=0, help="0 = all; else first N utterances")
    args = ap.parse_args()

    print(f"checkpoint: {args.checkpoint}")
    print(f"manifest:   {args.manifest}")
    print(f"device:     {args.device}\n")

    tokenizer = build_tokenizer(args.checkpoint)
    feature_extractor = Wav2Vec2FeatureExtractor(
        feature_size=1, sampling_rate=16000, padding_value=0.0,
        do_normalize=True, return_attention_mask=True,
    )

    cfg = HubertConfig.from_pretrained(args.checkpoint)  # carries num_sa_layers, sa_nhead, ...
    cfg.apply_spec_augment = False                       # deterministic; eval() also disables masking
    model = FillerHubertSAModel.from_pretrained(args.checkpoint, config=cfg).eval().to(args.device)

    rows = [json.loads(l) for l in open(args.manifest) if l.strip()]
    if args.limit > 0:
        rows = rows[: args.limit]
    print(f"utterances: {len(rows)}\n")

    ftxt = None
    if args.out_txt:
        os.makedirs(os.path.dirname(args.out_txt), exist_ok=True)
        ftxt = open(args.out_txt, "w", encoding="utf-8")

    def emit(line=""):
        print(line)
        if ftxt:
            ftxt.write(line + "\n")

    w_err = w_tot = c_err = c_tot = 0
    s_err = s_tot = 0
    with torch.no_grad():
        for d in tqdm(rows, desc="decoding"):
            audio = load_wav(d["audio"])
            inputs = feature_extractor(audio, sampling_rate=16000, return_tensors="pt")
            input_values = inputs["input_values"].to(args.device)
            logits = model(input_values=input_values).logits     # (1, T, 33)
            ids = torch.argmax(logits, dim=-1)[0]                 # (T,)

            # FULL per-frame output -- raw tokens, nothing stripped (pad/fill/bos/eos/| kept).
            frame_tokens = tokenizer.convert_ids_to_tokens(ids.tolist())
            full = " ".join(frame_tokens)
            # Truncate at the first </s> before stripping, so any stray char the model
            # emits in the post-eos tail cannot leak into the hypothesis.
            eos_hits = (ids == tokenizer.eos_token_id).nonzero(as_tuple=True)[0]
            cut = eos_hits[0].item() if eos_hits.numel() else ids.shape[0]
            # Stripped transcript for WER/SER/CER (no CTC repeat-merge, matching training decode).
            hyp = tokenizer.batch_decode(ids[:cut].unsqueeze(0), skip_special_tokens=True,
                                         group_tokens=False)[0].strip()
            ref = normalize_ref(d["text"])

            rw, hw = ref.split(), hyp.split()
            w_err += levenshtein(rw, hw); w_tot += len(rw)
            c_err += levenshtein(ref, hyp); c_tot += len(ref)
            s_err += int(hyp != ref); s_tot += 1

            num_frames = len(frame_tokens)
            num_fill = sum(1 for t in frame_tokens if t == FILL_TOKEN)
            num_pad = sum(1 for t in frame_tokens if t == "<pad>")
            content_frac = (num_frames - num_fill) / max(1, num_frames)

            emit(f"Key:  {d['key']}  | frames={num_frames} fill={num_fill} "
                 f"pad={num_pad} content_frac={content_frac:.3f}")
            emit(f"Ref:  {ref}")
            emit(f"Hyp:  {hyp}")
            emit(f"Full: {full}")
            emit()

    wer = w_err / max(1, w_tot)
    ser = s_err / max(1, s_tot)
    cer = c_err / max(1, c_tot)
    emit("===== SUMMARY =====")
    emit(f"Test-Clean WER: {wer:.4f}  ({w_err}/{w_tot} words)")
    emit(f"Test-Clean SER: {ser:.4f}  ({s_err}/{s_tot} utts)")
    emit(f"Test-Clean CER: {cer:.4f}  ({c_err}/{c_tot} chars)")

    if ftxt:
        ftxt.close()
        print(f"\n(teed to {args.out_txt})")


if __name__ == "__main__":
    main()

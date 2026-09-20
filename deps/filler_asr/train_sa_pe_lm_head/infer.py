"""Inference / decode for a trained SA + PE + lm_head filler-ASR head on a FROZEN HuBERT-xlarge.

One engine for every depth (sa2/sa4/sa6/sa8).  Everything needed to rebuild the model is
recovered from the checkpoint itself:

    * num_sa_layers : COUNTED from the `sa.layers.<i>.*` keys in the saved head
                      (robust even for old ckpts whose args don't store it, e.g. full960_sa_pe_lmhead).
    * model_checkpoint / use_pe : read from ck["args"] when present, else sane defaults.
    * tokenizer (33-tok char vocab incl <fill>) : loaded from --run_dir (train.py saved it there).

Decode is bit-identical to train.py:evaluate() so WER here matches the training-time eval curve:
    pred = logits.argmax(-1)
    hyp  = tok.batch_decode(pred, skip_special_tokens=True, group_tokens=False)   # NO CTC collapse
References come from the RAW manifest transcript (normalized), so this is a true test-set WER.

Writes per-utterance {audio, ref, hyp, wer} to --out (jsonl) and prints a summary line.

Launch: see infer.sh
"""
import os
import re
import sys
import json
import time
import argparse

import torch
from torch.utils.data import DataLoader
from transformers import Wav2Vec2FeatureExtractor

from data import load_rows, ManifestDataset, DataCollatorFillerASR
from filler_sa_reference import FILL_TOKEN, SAMPLING_RATE, build_tokenizer, build_model

DEFAULT_CKPT = "/speech/tomson/filler_asr/models/hubert-xlarge-ls960-ft"

# ---- text normalization + Levenshtein (identical to train.py) ----
CHARS_TO_IGNORE = r'[\,\?\.\!\-\;\:\"]'
def norm_ref(t): return re.sub(CHARS_TO_IGNORE, "", t).lower().strip()
def lev(ref, hyp):
    n, m = len(ref), len(hyp)
    if n == 0: return m
    prev = list(range(m + 1))
    for i in range(1, n + 1):
        cur = [i] + [0]*m; ri = ref[i-1]
        for j in range(1, m+1): cur[j] = min(prev[j]+1, cur[j-1]+1, prev[j-1]+(0 if ri == hyp[j-1] else 1))
        prev = cur
    return prev[m]


def count_sa_layers(head_sd):
    """Number of SA layers = (max index in `sa.layers.<i>.*` keys) + 1."""
    idx = {int(m.group(1)) for k in head_sd
           for m in [re.match(r"sa\.layers\.(\d+)\.", k)] if m}
    return (max(idx) + 1) if idx else 0


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True, help="dir with vocab.json/added_tokens.json (tokenizer) + ckpt")
    ap.add_argument("--ckpt", default="", help="path to .pt (default: <run_dir>/best.pt)")
    ap.add_argument("--manifest", default="/speech/tomson/exps/speech-recog/data/librispeech/librispeech_test_clean.jsonl")
    ap.add_argument("--model_checkpoint", default="", help="override frozen encoder path (else from ckpt args)")
    ap.add_argument("--out", default="", help="per-utt jsonl out (default: <run_dir>/infer_<manifest>.jsonl)")
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--dev_n", type=int, default=0, help="0 = full manifest; else first N utts")
    ap.add_argument("--num_workers", type=int, default=6)
    args = ap.parse_args()

    ckpt = args.ckpt or os.path.join(args.run_dir, "best.pt")
    tag = os.path.splitext(os.path.basename(args.manifest))[0]
    out = args.out or os.path.join(args.run_dir, f"infer_{tag}.jsonl")
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    # --- load checkpoint, recover architecture ---
    ck = torch.load(ckpt, map_location="cpu")
    head_sd = ck["head"]
    cargs = ck.get("args", {}) or {}
    n_sa = count_sa_layers(head_sd)
    enc = args.model_checkpoint or cargs.get("model_checkpoint") or DEFAULT_CKPT
    use_pe = bool(cargs.get("pe", True))
    # mask-trained ckpts (train_sa8_mask_*) carry a mask_embed param; masking itself is
    # train-only (gated on self.training) but the param must exist for load_state_dict.
    has_mask = "mask_embed" in head_sd
    mask_prob = float(cargs.get("mask_prob") or 0.0) or (1.0 if has_mask else 0.0)
    print(f"[ckpt] {ckpt}\n[arch] num_sa_layers={n_sa} pe={use_pe} mask_embed={has_mask} encoder={enc} "
          f"(trained step={ck.get('step')} best_wer={ck.get('best_wer')})", flush=True)

    # --- tokenizer (from the run dir train.py saved) + feature extractor ---
    tok = build_tokenizer(args.run_dir)
    fill_id = tok.convert_tokens_to_ids(FILL_TOKEN)
    fe = Wav2Vec2FeatureExtractor(feature_size=1, sampling_rate=16000, padding_value=0.0,
                                  do_normalize=True, return_attention_mask=True)
    base_collator = DataCollatorFillerASR(fe, tok)               # fps only affects loss mask; irrelevant for decode
    def collate(feats):
        b = base_collator(feats); b["_texts"] = [f["text"] for f in feats]; return b

    # --- model: frozen HuBERT + PE + n_sa SA + lm_head, then load the trained head ---
    model = build_model(enc, tok, num_sa_layers=n_sa, use_sinusoidal_pe=use_pe, device=dev,
                        mask_prob=mask_prob,
                        mask_prob_max=float(cargs.get("mask_prob_max") or 0.0),
                        mask_length=int(cargs.get("mask_length") or 10))
    missing, unexpected = model.load_state_dict(head_sd, strict=False)   # only head keys present → hubert keys "missing" OK
    bad = [k for k in unexpected] + [k for k in missing if k.startswith(("sa.", "lm_head.")) or k == "mask_embed"]
    assert not bad, f"head load mismatch: {bad[:6]}"
    model.eval()

    rows = load_rows(args.manifest, args.dev_n or None)
    loader = DataLoader(ManifestDataset(rows), batch_size=args.batch_size, shuffle=False,
                        collate_fn=collate, num_workers=args.num_workers, pin_memory=True)

    w_err = w_tot = c_err = c_tot = s_err = s_tot = 0
    wf_err = cf_err = n_leak = 0                     # uncapped (old) errors + #utts changed by capping
    eos_id = tok.eos_token_id
    t0 = time.time(); n = 0
    with open(out, "w") as fout:
        for batch in loader:
            texts = batch.pop("_texts"); batch.pop("labels", None)
            inp = {k: v.to(dev) for k, v in batch.items()}
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits = model(**inp).logits.float()
            pred = logits.argmax(-1)
            for i, raw in enumerate(texts):
                ref = norm_ref(raw)
                p = pred[i]
                hit = (p == eos_id).nonzero(as_tuple=True)[0]                 # cap at first </s> = duration ceiling
                cap = p[:hit[0]] if hit.numel() else p
                hyp = tok.decode(cap.tolist(), skip_special_tokens=True, group_tokens=False).strip()        # TRUE capped
                hyp_full = tok.decode(p.tolist(), skip_special_tokens=True, group_tokens=False).strip()     # old uncapped
                we = lev(ref.split(), hyp.split()); ce = lev(ref, hyp)
                wef = lev(ref.split(), hyp_full.split()); cef = lev(ref, hyp_full)
                w_err += we; w_tot += len(ref.split()); c_err += ce; c_tot += len(ref)
                wf_err += wef; cf_err += cef
                s_err += int(hyp != ref); s_tot += 1; n += 1
                n_leak += int(hyp != hyp_full)
                fout.write(json.dumps({"ref": ref, "hyp": hyp, "hyp_uncapped": hyp_full,
                                       "wer": we / max(1, len(ref.split())),
                                       "leaked": hyp != hyp_full}) + "\n")
            if n and (n // args.batch_size) % 10 == 0:
                print(f"  ..{n} utts  running WER={w_err/max(1,w_tot):.4f}  {n/(time.time()-t0):.1f} utt/s", flush=True)

    WER, CER, SER = w_err/max(1, w_tot), c_err/max(1, c_tot), s_err/max(1, s_tot)
    WER_unc, CER_unc = wf_err/max(1, w_tot), cf_err/max(1, c_tot)
    print(f"\n[done] {os.path.basename(args.run_dir)}  {tag}  n={n}  "
          f"WER={WER:.4f}  CER={CER:.4f}  SER={SER:.4f}  ({time.time()-t0:.0f}s)", flush=True)
    print(f"[cap ] capped@</s> vs uncapped:  WER {WER:.4f} vs {WER_unc:.4f}   "
          f"CER {CER:.4f} vs {CER_unc:.4f}   leaked_utts={n_leak}/{n}", flush=True)
    print(f"[out ] {out}", flush=True)
    with open(os.path.join(args.run_dir, f"infer_{tag}.summary"), "w") as f:
        json.dump({"run": os.path.basename(args.run_dir), "set": tag, "n": n,
                   "WER": WER, "CER": CER, "SER": SER,
                   "WER_uncapped": WER_unc, "CER_uncapped": CER_unc, "leaked_utts": n_leak,
                   "ckpt": ckpt}, f, indent=2)


if __name__ == "__main__":
    main()

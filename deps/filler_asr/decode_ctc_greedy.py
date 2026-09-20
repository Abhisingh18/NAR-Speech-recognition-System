#!/usr/bin/env python3
"""Standard greedy CTC decode of facebook/hubert-xlarge-ls960-ft on one testset
jsonl, scored with jiwer. This is the canonical HF model-card inference for a
fine-tuned CTC model:

    logits = model(input_values).logits
    pred_ids = logits.argmax(-1)
    text = processor.batch_decode(pred_ids)

No beam search, no language model, no custom edit-distance scorer. WER/CER come
from jiwer (process_words / process_characters). The model emits UPPERCASE A-Z,
space and apostrophe only, so both sides get the standard normalization
(uppercase + strip punctuation the model cannot produce) before scoring.

Input jsonl lines: {"key","source"(16k wav path),"target"}.
Outputs into --out_dir:
  decode_<set>_greedy_pred    key \t hypothesis
  decode_<set>_greedy_gt      key \t reference
  decode_<set>_greedy_wer     jiwer WER/CER/SER + counts (json + text)

usage:
  python decode_ctc_greedy.py --jsonl <path> --set <name> --out_dir <dir> \
      [--batch 8] [--model <dir>]
GPU: pin with CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=<n>.
"""
import argparse
import json
import os
import sys
import time

import jiwer
import soundfile as sf
import torch
from transformers import Wav2Vec2Processor, HubertForCTC

HUB_ID = "facebook/hubert-xlarge-ls960-ft"    # tokenizer/vocab source (32 chars)

# Standard eval normalization for an uppercase-only CTC model: uppercase both
# sides and drop punctuation the model can't emit. (jiwer built-in transforms.)
_WORD = jiwer.Compose([
    jiwer.ToUpperCase(),
    jiwer.RemovePunctuation(),
    jiwer.RemoveMultipleSpaces(),
    jiwer.Strip(),
    jiwer.ReduceToListOfListOfWords(),
])
_CHAR = jiwer.Compose([
    jiwer.ToUpperCase(),
    jiwer.RemovePunctuation(),
    jiwer.RemoveMultipleSpaces(),
    jiwer.Strip(),
    jiwer.ReduceToListOfListOfChars(),
])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl", required=True)
    ap.add_argument("--set", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--model", default="/speech/tomson/filler_asr/models/hubert-xlarge-ls960-ft")
    ap.add_argument("--batch", type=int, default=8)
    args = ap.parse_args()

    if not torch.cuda.is_available():
        sys.exit("no CUDA visible — pin a GPU with CUDA_VISIBLE_DEVICES")
    os.makedirs(args.out_dir, exist_ok=True)

    proc = Wav2Vec2Processor.from_pretrained(HUB_ID)
    model = HubertForCTC.from_pretrained(args.model).eval().half().cuda()

    rows = [json.loads(l) for l in open(args.jsonl, encoding="utf-8") if l.strip()]
    rows.sort(key=lambda r: os.path.getsize(r["source"]), reverse=True)

    base = os.path.join(args.out_dir, f"decode_{args.set}_greedy")
    keys, hyps, refs = [], [], []
    t0 = time.time()
    for i in range(0, len(rows), args.batch):
        batch = rows[i:i + args.batch]
        wavs = []
        for r in batch:
            wav, sr = sf.read(r["source"])
            if sr != 16000:
                sys.exit(f"{r['source']} is {sr}Hz, expected 16000")
            wavs.append(wav)
        feats = proc(wavs, sampling_rate=16000, return_tensors="pt", padding=True)
        with torch.no_grad():
            logits = model(feats.input_values.half().cuda(),
                           attention_mask=feats.attention_mask.cuda()).logits
        pred_ids = logits.argmax(dim=-1)
        texts = proc.batch_decode(pred_ids)        # canonical CTC collapse + decode
        for r, t in zip(batch, texts):
            keys.append(r["key"]); hyps.append(t); refs.append(r["target"])
        if len(keys) % 400 < args.batch:
            print(f"  {len(keys)}/{len(rows)}  {len(keys)/(time.time()-t0):.1f} utt/s", flush=True)

    with open(base + "_pred", "w", encoding="utf-8") as fp, \
         open(base + "_gt", "w", encoding="utf-8") as fg:
        for k, h, r in zip(keys, hyps, refs):
            fp.write(f"{k}\t{h}\n"); fg.write(f"{k}\t{r}\n")
    print(f"decoded {len(keys)} utts in {time.time()-t0:.0f}s -> {base}_pred", flush=True)

    w = jiwer.process_words(refs, hyps, reference_transform=_WORD, hypothesis_transform=_WORD)
    c = jiwer.process_characters(refs, hyps, reference_transform=_CHAR, hypothesis_transform=_CHAR)
    n = len(refs)
    sent_err = sum(1 for a, b in zip(w.references, w.hypotheses) if a != b)
    summary = dict(
        model=HUB_ID, set=args.set, n=n, decode="greedy_argmax",
        scorer="jiwer-4.0.0 (uppercase+strip-punct)",
        WER=w.wer, CER=c.cer, SER=sent_err / n,
        ins=w.insertions, dele=w.deletions, sub=w.substitutions, hits=w.hits,
    )
    with open(base + "_wer", "w") as f:
        json.dump(summary, f, indent=2)
        f.write("\n\n")
        f.write(f"%WER {100*w.wer:.2f} [ {w.insertions+w.deletions+w.substitutions} "
                f"/ {w.hits+w.deletions+w.substitutions}, {w.insertions} ins, "
                f"{w.deletions} del, {w.substitutions} sub ]\n")
        f.write(f"%CER {100*c.cer:.2f}\n")
        f.write(f"%SER {100*sent_err/n:.2f} [ {sent_err} / {n} ]\n")
    print("=== %s ===" % args.set, flush=True)
    print(open(base + "_wer").read(), flush=True)


if __name__ == "__main__":
    main()

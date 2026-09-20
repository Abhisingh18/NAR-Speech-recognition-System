"""Is 50 fps enough frame budget for Hindi char targets?

The A-CMLM lays the label sequence  [<s>] + chars(with | word-seps) + [</s>]*eos_repeat  on the
frame axis, 1 token per frame. data2vec gives 50 frames/s, and at FPS=50 the loss budget is
n_keep = T = ceil(dur*50). So a clip is TRUNCATED iff:
      required = Ntokens + 1 + eos_repeat  >  ceil(dur * FPS)
i.e. the char rate exceeds the frame rate. This measures the char/second distribution and the
truncation rate at FPS=50 (and 25 for contrast).
"""
import os
import sys
import json
import math
import random
import argparse

import numpy as np
import soundfile as sf

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from text_norm_hi import normalize_hi                                          # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl", default="/speech/tomson/exps/speech-recog/data/smear-more-hi-ta-te-ma/train_hi.jsonl")
    ap.add_argument("--field", default="target")
    ap.add_argument("--sample", type=int, default=5000)
    ap.add_argument("--eos_repeat", type=int, default=3)
    ap.add_argument("--fps_list", type=int, nargs="+", default=[25, 50])
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(args.jsonl) if l.strip()]
    random.seed(0)
    if args.sample and len(rows) > args.sample:
        rows = random.sample(rows, args.sample)

    tps = []          # tokens per second
    durs = []
    ntoks = []
    for r in rows:
        try:
            info = sf.info(r[args.field == "x" and "x" or "source"])
        except Exception:
            continue
        dur = info.frames / info.samplerate
        if dur <= 0:
            continue
        norm = normalize_hi(r[args.field])
        ntok = len(norm.replace(" ", "|"))          # 1 token per char, spaces -> | (same length)
        if ntok == 0:
            continue
        durs.append(dur); ntoks.append(ntok); tps.append(ntok / dur)

    tps = np.array(tps); durs = np.array(durs); ntoks = np.array(ntoks)
    print(f"clips measured: {len(tps)}   dur: mean={durs.mean():.1f}s p50={np.percentile(durs,50):.1f} "
          f"p99={np.percentile(durs,99):.1f} max={durs.max():.1f}")
    print(f"tokens/clip   : mean={ntoks.mean():.0f} p50={np.percentile(ntoks,50):.0f} "
          f"p99={np.percentile(ntoks,99):.0f} max={ntoks.max()}")
    print("\ntokens/second (char rate incl | word-seps):")
    for q in [50, 90, 95, 99, 99.9, 100]:
        print(f"   p{q:<5} = {np.percentile(tps, q):.2f}")

    print("\ntruncation (required = Ntokens + 1 + eos_repeat > ceil(dur*FPS)):")
    req = ntoks + 1 + args.eos_repeat
    for fps in args.fps_list:
        budget = np.ceil(durs * fps)
        trunc = req > budget
        # by how much, for the ones that overflow
        over = (req - budget)[trunc]
        headroom = (budget - req)                     # +ve = spare frames
        print(f"   FPS={fps:<3}: truncated {trunc.sum()}/{len(req)} = {100*trunc.mean():.2f}%   "
              f"| median headroom = {np.median(headroom):.0f} frames   "
              f"| worst overflow = {int(over.max()) if trunc.any() else 0} tokens")

    print("\nverdict: 50 fps is fine if truncation ~0% and headroom is large; "
          "25 fps shown for contrast (filler_asr's early runs used 25).")


if __name__ == "__main__":
    main()

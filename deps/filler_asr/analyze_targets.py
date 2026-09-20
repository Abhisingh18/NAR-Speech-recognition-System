"""Analyze the positional <fill>/</s> targets over train_960h at fps=25 grading.

Replicates DataCollatorFillerASR._labels_for + the two loss masks EXACTLY:
    labels = [<s>] + char_ids + [</s>] + <fill>*(T-len)   (T = encoder frames ~50fps)
    graded region = first n_keep = ceil(dur*FPS) frames   (rest -> -100)

len(char_ids) == len(normalized_string) was verified, so we count chars directly.
"""
import sys, os, re, json, math, argparse
import numpy as np
sys.path.insert(0, '/speech/tomson/filler_asr')
import soundfile as sf
from filler_sa_reference import compute_output_length, CHARS_TO_IGNORE_REGEX

FPS = 25
SR  = 16000
_rx = re.compile(CHARS_TO_IGNORE_REGEX)

def norm_len(text):
    # drop punctuation, lowercase, spaces->'|' ; len == len(tok ids)
    return len(_rx.sub('', text).lower().replace(' ', '|'))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--manifest', default='/speech/tomson/filler_asr/data/train_960h.jsonl')
    ap.add_argument('--fps', type=int, default=FPS)
    ap.add_argument('--limit', type=int, default=0)
    args = ap.parse_args()

    n = 0
    eos_not_graded = 0          # </s> falls outside the 25*dur budget -> never learned
    truncated = 0               # transcript longer than T frames (chars pinned 1:1 overflow)
    content_clipped = 0         # some chars (not just eos) fall outside budget
    dur_l, T_l, nkeep_l, Lids_l = [], [], [], []
    gfill_l, gtot_l, fillfrac_l, cps_l = [], [], [], []

    with open(args.manifest) as f:
        for line in f:
            line = line.strip()
            if not line: continue
            d = json.loads(line)
            path = d.get('audio') or d.get('source')
            text = d.get('text') or d.get('target') or ''
            info = sf.info(path)
            ns = info.frames                       # == len(input_values)
            dur = ns / SR
            T = compute_output_length(ns)          # encoder frames (~50 fps)
            L = norm_len(text)                      # len(char_ids)

            if L + 2 > T:                           # collator truncates ids to T-2
                truncated += 1
                prefix = T                          # [<s>]+ids[:T-2]+[</s>]
            else:
                prefix = L + 2                      # [<s>] + L chars + [</s>]
            eos_pos = prefix - 1                    # index of </s>

            n_keep = math.ceil(dur * args.fps)
            graded_total = min(n_keep, T)
            eos_graded = eos_pos < graded_total
            graded_content = min(prefix, graded_total)   # specials+chars graded
            graded_fill = graded_total - graded_content  # contiguous <fill> run graded

            if not eos_graded: eos_not_graded += 1
            if graded_content < prefix: content_clipped += 1   # chars themselves clipped

            n += 1
            dur_l.append(dur); T_l.append(T); nkeep_l.append(n_keep); Lids_l.append(L)
            gfill_l.append(graded_fill); gtot_l.append(graded_total)
            fillfrac_l.append(graded_fill / graded_total if graded_total else 0.0)
            cps_l.append(L / dur if dur else 0.0)
            if args.limit and n >= args.limit: break

    A = lambda x: np.array(x, dtype=np.float64)
    dur, T, nkeep, Lids = A(dur_l), A(T_l), A(nkeep_l), A(Lids_l)
    gfill, gtot, fillfrac, cps = A(gfill_l), A(gtot_l), A(fillfrac_l), A(cps_l)

    def stats(x): return (f"min={x.min():.2f} p1={np.percentile(x,1):.2f} med={np.median(x):.2f} "
                          f"mean={x.mean():.2f} p99={np.percentile(x,99):.2f} max={x.max():.2f}")

    print(f"=== {n} utterances | fps(grade)={args.fps} ===\n")
    print("[Q1] </s> presence")
    print(f"  literal </s> in label            : {n}/{n} (100%) -- always appended")
    print(f"  transcript truncated (L+2 > T)   : {truncated} ({truncated/n*100:.3f}%)")
    print(f"  </s> NOT graded (outside budget) : {eos_not_graded} ({eos_not_graded/n*100:.3f}%)")
    print(f"  content chars clipped by budget  : {content_clipped} ({content_clipped/n*100:.3f}%)\n")

    print("[ctx] durations (s)          :", stats(dur))
    print("[ctx] chars/sec (L/dur)      :", stats(cps))
    print("[ctx] encoder frames T(~50fps):", stats(T))
    print("[ctx] n_keep=ceil(25*dur)    :", stats(nkeep))
    print("[ctx] transcript len L(chars):", stats(Lids), "\n")

    print("[Q2/Q3] graded <fill> run per utterance (contiguous tail of graded region)")
    print("  frames :", stats(gfill))
    print(f"  utts with 0 graded <fill>    : {(gfill==0).sum()} ({(gfill==0).mean()*100:.3f}%)")
    print("  graded frames total /utt     :", stats(gtot))
    print("  <fill> fraction of graded    :", stats(fillfrac))
    print(f"  overall graded-frame <fill>% : {gfill.sum()/gtot.sum()*100:.2f}%  "
          f"(sum fill {int(gfill.sum())}/{int(gtot.sum())})\n")

    # histogram of fill fraction
    edges = [0,0.001,0.1,0.2,0.3,0.4,0.5,0.6,1.0]
    h,_ = np.histogram(fillfrac, bins=edges)
    print("  <fill>-fraction histogram:")
    for i in range(len(edges)-1):
        print(f"    [{edges[i]:.3f},{edges[i+1]:.3f}) : {h[i]:7d} ({h[i]/n*100:5.2f}%)")

if __name__ == '__main__':
    main()

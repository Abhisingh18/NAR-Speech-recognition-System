"""Analysis of the >=10 s subset of a decode, sorted by per-utt WER (increasing).

Reads all_utts.jsonl, keeps clips with dur >= THRESH, sorts by utt_wer ascending
(0.00 -> max), and writes: a WER-distribution summary + a per-utt list with the
audio path, duration, WER, GT and PRED.

Usage: python analyze_ge10_by_wer.py --decode_dir <dir> [--thresh 10.0]
"""
import os, re, json, argparse

clean = lambda h: " ".join(re.sub(r'(<fill>)+', ' ', h).split())

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--decode_dir", required=True)
    ap.add_argument("--thresh", type=float, default=10.0)
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(os.path.join(args.decode_dir, "all_utts.jsonl"))]
    rows = [r for r in rows if float(r['dur']) >= args.thresh]
    # utt_wer is stored as a FRACTION (0.143 = 14.3%); expose a percent value once.
    for r in rows:
        r['wer_pct'] = float(r['utt_wer']) * 100.0
    # sort by WER ascending, tie-break shorter-first then key for a stable order
    rows.sort(key=lambda r: (r['wer_pct'], float(r['dur']), r['key']))

    # WER distribution buckets
    buckets = [(0, 0), (0, 5), (5, 10), (10, 20), (20, 50), (50, 1e9)]
    def bucket_label(lo, hi):
        if lo == 0 and hi == 0: return "WER == 0.00%"
        if hi >= 1e9: return f"WER > {lo:g}%"
        return f"{lo:g}% < WER <= {hi:g}%"
    counts = []
    for lo, hi in buckets:
        if lo == 0 and hi == 0:
            n = sum(r['wer_pct'] == 0.0 for r in rows)
        else:
            n = sum(lo < r['wer_pct'] <= hi for r in rows)
        counts.append(n)

    N = len(rows)
    we = sum(int(r['w_err']) for r in rows); wt = sum(int(r['w_tot']) for r in rows)
    out = os.path.join(args.decode_dir, f"ANALYSIS_ge{args.thresh:g}s_by_wer.txt")
    with open(out, "w") as f:
        f.write("A-CMLM iterative decode — LibriSpeech test-clean  [epoch 66, 16 steps, temp=1.0]\n")
        f.write("=" * 78 + "\n")
        f.write(f"subset       : clips with audio duration >= {args.thresh:g} s\n")
        f.write(f"utterances   : {N}\n")
        f.write(f"aggregate WER: {we/max(1,wt)*100:.2f}%  ({we}/{wt} words)\n")
        f.write(f"WER range    : {rows[0]['wer_pct']:.2f}%  ->  {rows[-1]['wer_pct']:.2f}%\n")
        f.write("-" * 78 + "\n")
        f.write("WER distribution:\n")
        for (lo, hi), n in zip(buckets, counts):
            f.write(f"    {bucket_label(lo, hi):<22} : {n:>4}  ({n/max(1,N)*100:5.1f}%)\n")
        f.write("-" * 78 + "\n")
        f.write("per-utt, sorted by WER increasing (0.00% first):\n")
        f.write("    key  dur  WER | audio | GT / PRED\n\n")
        for i, r in enumerate(rows, 1):
            f.write(f"[{i:>3}] {r['key']}  dur={float(r['dur']):.2f}s  WER={r['wer_pct']:.2f}%  "
                    f"({r['w_err']}/{r['w_tot']} words)\n")
            f.write(f"      audio: {r['audio']}\n")
            f.write(f"      GT   : {r['ref']}\n")
            f.write(f"      PRED : {clean(r['hyp'])}\n\n")

    # console summary
    print(f"subset >= {args.thresh:g}s : {N} utts, aggregate WER {we/max(1,wt)*100:.2f}%, "
          f"range {rows[0]['wer_pct']:.2f}% -> {rows[-1]['wer_pct']:.2f}%")
    print("WER distribution:")
    for (lo, hi), n in zip(buckets, counts):
        print(f"    {bucket_label(lo, hi):<22} : {n:>4}  ({n/max(1,N)*100:5.1f}%)")
    print(f"\nwrote: {out}")

if __name__ == "__main__":
    main()

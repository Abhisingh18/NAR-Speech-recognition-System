"""Split a decode's per-utt records (all_utts.jsonl) into two REPORT-style files by
audio duration: clips >= THRESH seconds and clips < THRESH seconds.

Metrics per group match REPORT.txt:
    WER        = sum(w_err)/sum(w_tot)              on the clean hyp (cut@</s>, specials dropped)
    CER[clean] = char-Levenshtein(ref, clean(hyp)) / sum(len(ref))   (<fill>-leak removed)
    CER[raw]   = sum(c_err)/sum(c_tot)              (raw hyp, includes <fill> leak)
    SER        = sum(s_err)/N ; exact-match ; utts with no </s> emitted

Usage: python split_by_duration.py --decode_dir <dir with all_utts.jsonl> [--thresh 10.0]
"""
import os, re, json, argparse

def char_lev(a, b):
    n, m = len(a), len(b)
    if n == 0: return m
    prev = list(range(m + 1))
    for i in range(1, n + 1):
        cur = [i] + [0] * m; ri = a[i - 1]
        for j in range(1, m + 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (0 if ri == b[j - 1] else 1))
        prev = cur
    return prev[m]

clean = lambda h: " ".join(re.sub(r'(<fill>)+', ' ', h).split())

def summarize(rows):
    we = wt = crawE = crawT = se = ne = 0
    cclean_e = cclean_t = exact = 0
    for r in rows:
        we += int(r['w_err']); wt += int(r['w_tot'])
        crawE += int(r['c_err']); crawT += int(r['c_tot'])
        se += int(r['s_err']); exact += (int(r['s_err']) == 0)
        hc = clean(r['hyp'])
        cclean_e += char_lev(r['ref'], hc); cclean_t += len(r['ref'])
        ne += ('</s>' not in r['full'].split()[:int(r['n_keep'])])
    N = len(rows)
    return dict(N=N, WER=we / max(1, wt) * 100, wnum=(we, wt),
                CERc=cclean_e / max(1, cclean_t) * 100, cnum=(cclean_e, cclean_t),
                CERraw=crawE / max(1, crawT) * 100, rnum=(crawE, crawT),
                SER=se / max(1, N) * 100, snum=(se, N), exact=exact, noeos=ne)

def write_report(path, title, rows, s, thresh):
    with open(path, "w") as f:
        f.write("A-CMLM iterative decode — LibriSpeech test-clean  [epoch 66, 16 steps, temp=1.0]\n")
        f.write("=" * 72 + "\n")
        f.write(f"subset     : {title}\n")
        f.write(f"utterances : {s['N']}  (of 2620)\n")
        f.write(f"WER        = {s['WER']:.2f}%   ({s['wnum'][0]}/{s['wnum'][1]} words)\n")
        f.write(f"CER[clean] = {s['CERc']:.2f}%   ({s['cnum'][0]}/{s['cnum'][1]} chars)   [<fill>-leak removed]\n")
        f.write(f"CER[raw]   = {s['CERraw']:.2f}%   ({s['rnum'][0]}/{s['rnum'][1]} chars)   [raw hyp]\n")
        f.write(f"SER        = {s['SER']:.2f}%   ({s['snum'][0]}/{s['snum'][1]} sentences)\n")
        f.write(f"Exact-match: {s['exact']}/{s['N']} ({s['exact']/max(1,s['N'])*100:.1f}%)\n")
        f.write(f"no </s>    : {s['noeos']}\n")
        f.write("-" * 72 + "\n")
        f.write("per-utt (sorted by duration, longest first): key  dur  uWER\n\n")
        for r in sorted(rows, key=lambda r: -float(r['dur'])):
            f.write(f"[{r['key']}]  dur={float(r['dur']):.2f}s  WER={float(r['utt_wer']):.1f}%\n")
            f.write(f"  GT   : {r['ref']}\n")
            f.write(f"  PRED : {clean(r['hyp'])}\n\n")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--decode_dir", required=True)
    ap.add_argument("--thresh", type=float, default=10.0)
    args = ap.parse_args()
    rows = [json.loads(l) for l in open(os.path.join(args.decode_dir, "all_utts.jsonl"))]
    long_rows = [r for r in rows if float(r['dur']) >= args.thresh]
    short_rows = [r for r in rows if float(r['dur']) < args.thresh]
    sl, ss = summarize(long_rows), summarize(short_rows)

    p_long = os.path.join(args.decode_dir, f"REPORT_ge{args.thresh:g}s.txt")
    p_short = os.path.join(args.decode_dir, f"REPORT_lt{args.thresh:g}s.txt")
    write_report(p_long, f"clips with audio duration >= {args.thresh:g} s", long_rows, sl, args.thresh)
    write_report(p_short, f"clips with audio duration < {args.thresh:g} s", short_rows, ss, args.thresh)

    print(f"threshold = {args.thresh:g} s   (from {os.path.join(args.decode_dir,'all_utts.jsonl')})")
    print(f"{'subset':<12}{'N':>6}{'WER':>9}{'CERclean':>10}{'CERraw':>9}{'SER':>8}{'exact':>7}{'no</s>':>7}")
    for name, s in [(f">= {args.thresh:g}s", sl), (f"<  {args.thresh:g}s", ss)]:
        print(f"{name:<12}{s['N']:>6}{s['WER']:>8.2f}%{s['CERc']:>9.2f}%{s['CERraw']:>8.2f}%"
              f"{s['SER']:>7.1f}%{s['exact']:>7}{s['noeos']:>7}")
    print(f"\nwrote:\n  {p_long}\n  {p_short}")

if __name__ == "__main__":
    main()

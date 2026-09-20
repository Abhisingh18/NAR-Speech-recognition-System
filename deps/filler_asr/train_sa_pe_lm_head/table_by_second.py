"""Per-second duration table for a decode's per-utt records (all_utts.jsonl).

Bins clips by floor(dur) starting at --start seconds (default 10) up to max, one row
per second. Columns:
    dur bin | #clips | avg dur | ref words | ref chars | err words | Sub Ins Del | CER% | WER%
WER = (S+D+I)/ref_words ; CER = char-Levenshtein(ref, clean(hyp)) / ref_chars  (<fill> removed).

Usage: python table_by_second.py --decode_dir <dir> [--start 10]
"""
import os, re, json, argparse

clean = lambda h: " ".join(re.sub(r'(<fill>)+', ' ', h).split())

def sdi(a, b):
    """word-level Sub/Del/Ins between ref list a and hyp list b (via edit backtrace)."""
    n, m = len(a), len(b)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1): dp[i][0] = i
    for j in range(m + 1): dp[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            c = 0 if a[i - 1] == b[j - 1] else 1
            dp[i][j] = min(dp[i - 1][j] + 1, dp[i][j - 1] + 1, dp[i - 1][j - 1] + c)
    i, j, S, D, I = n, m, 0, 0, 0
    while i > 0 or j > 0:
        if i > 0 and j > 0 and a[i - 1] == b[j - 1] and dp[i][j] == dp[i - 1][j - 1]:
            i -= 1; j -= 1
        elif i > 0 and j > 0 and dp[i][j] == dp[i - 1][j - 1] + 1:
            S += 1; i -= 1; j -= 1
        elif i > 0 and dp[i][j] == dp[i - 1][j] + 1:
            D += 1; i -= 1
        else:
            I += 1; j -= 1
    return S, D, I

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

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--decode_dir", required=True)
    ap.add_argument("--start", type=int, default=10)
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(os.path.join(args.decode_dir, "all_utts.jsonl"))]
    rows = [r for r in rows if float(r['dur']) >= args.start]
    bins = {}
    for r in rows:
        b = int(float(r['dur']))                       # floor -> bin "b s" = [b, b+1)
        d = bins.setdefault(b, dict(n=0, dur=0.0, w=0, ch=0, S=0, D=0, I=0, ce=0, ct=0))
        ref_w = r['ref'].split(); hyp = clean(r['hyp'])
        S, D, I = sdi(ref_w, hyp.split())
        d['n'] += 1; d['dur'] += float(r['dur'])
        d['w'] += len(ref_w); d['ch'] += len(r['ref'])
        d['S'] += S; d['D'] += D; d['I'] += I
        d['ce'] += char_lev(r['ref'], hyp); d['ct'] += len(r['ref'])

    hdr = (f"{'dur':>5} {'clips':>6} {'avgdur':>7} {'words':>7} {'chars':>7} "
           f"{'errW':>6} {'Sub':>6} {'Ins':>5} {'Del':>5} {'CER%':>7} {'WER%':>7}")
    lines = [hdr, "-" * len(hdr)]
    tot = dict(n=0, dur=0.0, w=0, ch=0, S=0, D=0, I=0, ce=0, ct=0)
    for b in sorted(bins):
        d = bins[b]; E = d['S'] + d['D'] + d['I']
        cer = d['ce'] / max(1, d['ct']) * 100; wer = E / max(1, d['w']) * 100
        lines.append(f"{str(b)+'s':>5} {d['n']:>6} {d['dur']/d['n']:>7.2f} {d['w']:>7} {d['ch']:>7} "
                     f"{E:>6} {d['S']:>6} {d['I']:>5} {d['D']:>5} {cer:>6.2f}% {wer:>6.2f}%")
        for k in tot: tot[k] += d[k]
    E = tot['S'] + tot['D'] + tot['I']
    cer = tot['ce'] / max(1, tot['ct']) * 100; wer = E / max(1, tot['w']) * 100
    lines.append("-" * len(hdr))
    lines.append(f"{('>='+str(args.start)+'s'):>5} {tot['n']:>6} {tot['dur']/max(1,tot['n']):>7.2f} "
                 f"{tot['w']:>7} {tot['ch']:>7} {E:>6} {tot['S']:>6} {tot['I']:>5} {tot['D']:>5} "
                 f"{cer:>6.2f}% {wer:>6.2f}%")

    table = "\n".join(lines)
    out = os.path.join(args.decode_dir, f"TABLE_by_second_ge{args.start}s.txt")
    with open(out, "w") as f:
        f.write("A-CMLM iterative decode — LibriSpeech test-clean  [epoch 66, 16 steps, temp=1.0]\n")
        f.write(f"per-second duration bins (bin 'Ns' = [N, N+1) s), start = {args.start}s\n\n")
        f.write(table + "\n")
    print(table)
    print(f"\nwrote: {out}")

if __name__ == "__main__":
    main()

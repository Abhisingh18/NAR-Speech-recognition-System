"""Duration-bucketed WER + budget-truncation check from a dump_decodes decodes.txt.

Reads the per-utt blocks (dur / n_keep header, HYP/REF lines) and reports:
  * word-weighted WER per duration bucket  (tests: do errors concentrate in long clips?)
  * how many utts have label len (chars+<s></s>) > n_keep  (budget truncates transcript)
  * label chars/sec distribution vs the 25/s grading budget
"""
import re, sys

path = sys.argv[1] if len(sys.argv) > 1 else \
    "/speech/tomson/filler_asr/train_sa_pe_lm_head/eval_analysis/strip_variants/test_clean.decodes.txt"
hdr = re.compile(r"=+ (\S+)\s+dur=([\d.]+)s\s+T=(\d+)\s+n_keep=(\d+)\s+WER=([\d.]+) =+")
utts, cur = [], None
for line in open(path):
    m = hdr.match(line)
    if m:
        cur = {"dur": float(m.group(2)), "T": int(m.group(3)), "n_keep": int(m.group(4))}
        utts.append(cur)
    elif cur is not None:
        if line.startswith("HYP   : "):
            cur["hyp"] = line[8:].strip()
        elif line.startswith("REF   : "):
            cur["ref"] = line[8:].strip()

def lev(a, b):
    n, m = len(a), len(b)
    if n == 0:
        return m
    prev = list(range(m + 1))
    for i in range(1, n + 1):
        curr = [i] + [0] * m
        for j in range(1, m + 1):
            curr[j] = min(prev[j] + 1, curr[j - 1] + 1, prev[j - 1] + (a[i - 1] != b[j - 1]))
        prev = curr
    return prev[m]

buckets = [(0, 5), (5, 10), (10, 15), (15, 99)]
agg = {b: [0, 0, 0] for b in buckets}          # word_err, word_tot, n_utts
trunc = trunc_we = trunc_wt = 0
for u in utts:
    ref_w, hyp_w = u["ref"].split(), u["hyp"].split()
    we = lev(ref_w, hyp_w)
    lab_len = len(u["ref"].replace(" ", "|")) + 2          # <s> + chars + </s>
    if lab_len > u["n_keep"]:
        trunc += 1; trunc_we += we; trunc_wt += len(ref_w)
    for b in buckets:
        if b[0] <= u["dur"] < b[1]:
            agg[b][0] += we; agg[b][1] += len(ref_w); agg[b][2] += 1
            break

print(f"n_utts={len(utts)}")
print("\n--- duration-bucketed WER (capped decode) ---")
print(f"{'bucket':10s} {'n_utts':>7s} {'words':>8s} {'WER':>8s}")
for b in buckets:
    we, wt, n = agg[b]
    if n:
        print(f"{b[0]:>2d}-{b[1]:<2d}s    {n:7d} {wt:8d} {we / max(1, wt):8.4f}")

print("\n--- transcript longer than budget (label_len > n_keep) ---")
if trunc:
    print(f"count={trunc}/{len(utts)}  WER on those={trunc_we / max(1, trunc_wt):.4f}")
else:
    print(f"count=0/{len(utts)} — budget never truncates the transcript")

rates = sorted((len(u["ref"].replace(" ", "|")) + 2) / max(u["dur"], 0.1) for u in utts)
print(f"\nlabel chars/sec: median={rates[len(rates)//2]:.1f}  "
      f"p95={rates[int(0.95*len(rates))]:.1f}  max={rates[-1]:.1f}   (grading budget = 25/s)")

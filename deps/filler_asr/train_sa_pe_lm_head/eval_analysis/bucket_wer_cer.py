"""Duration-bucketed WER + CER for two decode sources, side-by-side.

Buckets (upper-inclusive): <=5s, >5<=10, >10<=20, >20<=25, >25<=30, >30.
Source A (new run): a dump_decodes decodes.txt (dur in header, HYP/REF lines).
Source B (old run): an all_utts.jsonl with per-row {dur, ref, hyp}.
WER = word edit dist / ref words.  CER = char edit dist / ref chars (spaces included).
"""
import re, sys, json

def clean_fill(s):
    """Strip the <fill> readout-leak (matches ger_acmlm / REPORT 'cleaned' numbers)."""
    return " ".join(re.sub(r"(<fill>)+", " ", s).split())

def lev(a, b):
    n, m = len(a), len(b)
    if n == 0: return m
    prev = list(range(m + 1))
    for i in range(1, n + 1):
        curr = [i] + [0] * m
        ai = a[i - 1]
        for j in range(1, m + 1):
            curr[j] = min(prev[j] + 1, curr[j - 1] + 1, prev[j - 1] + (ai != b[j - 1]))
        prev = curr
    return prev[m]

def load_decodes(path):
    hdr = re.compile(r"=+ (\S+)\s+dur=([\d.]+)s\s+T=(\d+)\s+n_keep=(\d+)\s+WER=([\d.]+) =+")
    utts, cur = [], None
    for line in open(path):
        m = hdr.match(line)
        if m:
            cur = {"dur": float(m.group(2))}; utts.append(cur)
        elif cur is not None:
            if line.startswith("HYP   : "): cur["hyp"] = clean_fill(line[8:].rstrip("\n"))
            elif line.startswith("REF   : "): cur["ref"] = line[8:].rstrip("\n")
    return utts

def load_jsonl(path):
    out = []
    for l in open(path):
        d = json.loads(l)
        out.append({"dur": float(d["dur"]), "ref": d["ref"], "hyp": clean_fill(d["hyp"])})
    return out

# (lo, hi]  — lo exclusive, hi inclusive; first bucket lo=-inf effectively (<=5)
BUCKETS = [(0, 5), (5, 10), (10, 20), (20, 25), (25, 30), (30, 1e9)]
def label(b):
    lo, hi = b
    if lo == 0: return "<=5s"
    if hi >= 1e9: return ">30s"
    return f">{lo}<={hi}s"

def analyze(utts):
    agg = {b: [0, 0, 0, 0, 0] for b in BUCKETS}  # w_err,w_tot,c_err,c_tot,n
    for u in utts:
        rw, hw = u["ref"].split(), u["hyp"].split()
        we = lev(rw, hw)
        ce = lev(u["ref"], u["hyp"])
        d = u["dur"]
        for b in BUCKETS:
            lo, hi = b
            if (lo < d <= hi) or (lo == 0 and d <= hi):
                a = agg[b]
                a[0] += we; a[1] += len(rw); a[2] += ce; a[3] += len(u["ref"]); a[4] += 1
                break
    return agg

def report(name, utts):
    agg = analyze(utts)
    print(f"\n=== {name}  (n={len(utts)}) ===")
    print(f"{'bucket':10s} {'n':>6s} {'words':>8s} {'WER%':>8s} {'chars':>8s} {'CER%':>8s}")
    tw_e = tw_t = tc_e = tc_t = 0
    for b in BUCKETS:
        we, wt, ce, ct, n = agg[b]
        tw_e += we; tw_t += wt; tc_e += ce; tc_t += ct
        wer = 100 * we / wt if wt else 0.0
        cer = 100 * ce / ct if ct else 0.0
        print(f"{label(b):10s} {n:6d} {wt:8d} {wer:8.2f} {ct:8d} {cer:8.2f}")
    print(f"{'ALL':10s} {len(utts):6d} {tw_t:8d} {100*tw_e/tw_t:8.2f} {tc_t:8d} {100*tc_e/tc_t:8.2f}")
    return agg

if __name__ == "__main__":
    new_dec = sys.argv[1]
    old_jsonl = sys.argv[2]
    a_new = report("NEW  fps50 v2 iter_steps32_best", load_decodes(new_dec))
    a_old = report("PREV fps25 fw0.1 epoch66 steps32", load_jsonl(old_jsonl))

    print("\n=== side-by-side WER% / CER%  (new vs prev, delta = new-prev) ===")
    print(f"{'bucket':10s} {'n_new':>6s} {'n_prev':>7s} | {'WERnew':>7s} {'WERprv':>7s} {'dWER':>7s} | {'CERnew':>7s} {'CERprv':>7s} {'dCER':>7s}")
    for b in BUCKETS:
        wn, wtn, cn, ctn, nn = a_new[b]
        wo, wto, co, cto, no = a_old[b]
        wern = 100*wn/wtn if wtn else 0; werp = 100*wo/wto if wto else 0
        cern = 100*cn/ctn if ctn else 0; cerp = 100*co/cto if cto else 0
        print(f"{label(b):10s} {nn:6d} {no:7d} | {wern:7.2f} {werp:7.2f} {wern-werp:+7.2f} | {cern:7.2f} {cerp:7.2f} {cern-cerp:+7.2f}")

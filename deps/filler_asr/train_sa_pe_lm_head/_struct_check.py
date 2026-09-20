"""Strict structural audit of a decode's RAW frame readouts.
For every clip checks: does it terminate (>=1 </s>)? how many </s>? is the tail after </s>
pure <fill> (no leak)? any <fill> holes in the body? Reports per-set aggregates.
Usage: python _struct_check.py <decodes.txt> [label]"""
import re, sys
from collections import Counter

hdr = re.compile(r"=+\s+(\S+)\s+dur=([\d.]+)s\s+T=(\d+)\s+n_keep=(\d+)\s+WER=([\d.]+)")
def is_content(t): return not t.startswith("<")          # a-z, '|', etc.  (specials all start '<')

path = sys.argv[1]; label = sys.argv[2] if len(sys.argv) > 2 else path
recs = []; cur = None
for line in open(path):
    m = hdr.search(line)
    if m:
        cur = dict(dur=float(m.group(2)), nkeep=int(m.group(4))); recs.append(cur)
    elif cur is not None and line.startswith("RAW"):
        cur["toks"] = line.split(":", 1)[1].split()[: cur["nkeep"]]

n = len(recs)
term = 0                    # has >=1 </s>
eos_hist = Counter()        # total number of </s> tokens
after_eos_contig = Counter()# contiguous </s> run length right after content
leak = 0                    # content token appears AFTER the first </s>
fill_after = 0              # >=1 <fill> after the first </s>
body_fill = 0              # >=1 <fill> BEFORE the first </s> (hole in body)
fully_clean = 0            # terminate + no leak + no body-fill hole + fill tail present
for r in recs:
    t = r.get("toks", [])
    if "</s>" not in t:
        eos_hist[0] += 1; continue
    term += 1
    e = t.index("</s>")
    total_eos = sum(1 for x in t if x == "</s>")
    eos_hist[total_eos] += 1
    # contiguous </s> run starting at e
    c = 0
    while e + c < len(t) and t[e + c] == "</s>": c += 1
    after_eos_contig[c] += 1
    tail = t[e + 1:]
    lk = any(is_content(x) for x in tail)
    bf = any(x == "<fill>" for x in t[:e])
    fa = any(x == "<fill>" for x in tail)
    leak += lk; body_fill += bf; fill_after += fa
    if (not lk) and (not bf) and fa:
        fully_clean += 1

pct = lambda x: f"{100*x/max(1,n):5.1f}%"
print(f"\n===== {label}   (n={n}) =====")
print(f"  terminates (has </s>)        : {term:5d}  {pct(term)}")
print(f"  NO </s> (never terminates)   : {eos_hist[0]:5d}  {pct(eos_hist[0])}")
print(f"  # of </s> tokens  -> count   : " + "  ".join(f"{k}:{eos_hist[k]}({pct(eos_hist[k]).strip()})" for k in sorted(eos_hist) if k>0))
print(f"  contiguous </s> run after ct : " + "  ".join(f"{k}:{after_eos_contig[k]}" for k in sorted(after_eos_contig)))
print(f"  <fill> present after </s>    : {fill_after:5d}  {pct(fill_after)}")
print(f"  LEAK (content after </s>)    : {leak:5d}  {pct(leak)}")
print(f"  <fill> hole in body          : {body_fill:5d}  {pct(body_fill)}")
print(f"  FULLY CLEAN (term+fill+noleak): {fully_clean:5d}  {pct(fully_clean)}")

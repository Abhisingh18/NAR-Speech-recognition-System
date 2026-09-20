"""Long-utterance reliability analysis from a decode_iterative.py decodes.txt.

Parses each record's header (dur, n_keep, WER) + the RAW frame readout, and reports:
  - WER binned by duration
  - leakage taxonomy per clip (same idea as diagnose_long.py):
        NO_EOS      : no </s> emitted in the graded region -> tail can leak
        POST_EOS    : non-<fill> tokens AFTER the first </s> (the leakage you asked about)
        FILL_INTRUDE: >=5 <fill> frames INSIDE the body (before </s>)
        OK/DRIFT    : </s> fine; errors are substitution/spelling only
Usage: python _longutt_analysis.py <decodes.txt> [label]
"""
import re, sys, statistics as st

path = sys.argv[1]; label = sys.argv[2] if len(sys.argv) > 2 else path

hdr = re.compile(r"dur=([\d.]+)s\s+T=(\d+)\s+n_keep=(\d+)\s+WER=([\d.]+)")
rows = []
cur = None
for line in open(path):
    m = hdr.search(line)
    if m:
        cur = dict(dur=float(m.group(1)), T=int(m.group(2)), nkeep=int(m.group(3)), wer=float(m.group(4)))
        rows.append(cur)
    elif line.startswith("RAW") and cur is not None and "raw" not in cur:
        cur["raw"] = line.split(":", 1)[1].split()

def classify(r):
    toks = r["raw"][: r["nkeep"]]            # only the graded/decoded region
    try:
        eos = toks.index("</s>")
    except ValueError:
        eos = None
    if eos is None:
        return "NO_EOS"
    post = [t for t in toks[eos+1:] if t not in ("</s>", "<fill>")]   # content after first </s>
    if len(post) >= 1:
        return "POST_EOS"
    body_fill = sum(1 for t in toks[:eos] if t == "<fill>")
    if body_fill >= 5:
        return "FILL_INTRUDE"
    return "OK/DRIFT"

for r in rows:
    r["cls"] = classify(r)

def report(subset, name):
    if not subset:
        print(f"  [{name}] no clips"); return
    n = len(subset)
    wers = [r["wer"] for r in subset]
    from collections import Counter
    c = Counter(r["cls"] for r in subset)
    macro = sum(wers)/n
    print(f"  [{name:>10}] n={n:4d}  meanWER(macro)={macro:.3f}  "
          f"NO_EOS={c['NO_EOS']:3d}  POST_EOS={c['POST_EOS']:3d}  FILL_INTRUDE={c['FILL_INTRUDE']:3d}  OK/DRIFT={c['OK/DRIFT']:4d}")

print(f"\n===== {label}  (n={len(rows)}) =====")
print("duration bins:")
bins = [(0,5),(5,10),(10,15),(15,100)]
for lo,hi in bins:
    report([r for r in rows if lo <= r["dur"] < hi], f"{lo}-{hi}s")
print("overall long vs short:")
report([r for r in rows if r["dur"] < 10], "<10s")
report([r for r in rows if r["dur"] >= 10], ">=10s")
# show a couple of the longest clips' tail behaviour
longest = sorted(rows, key=lambda r: -r["dur"])[:3]
print("longest clips (tail sanity):")
for r in longest:
    toks = r["raw"][:r["nkeep"]]
    eos = toks.index("</s>") if "</s>" in toks else None
    tail = toks[eos: eos+8] if eos is not None else toks[-8:]
    print(f"   dur={r['dur']:.1f}s nkeep={r['nkeep']} wer={r['wer']:.2f} cls={r['cls']}  eos@{eos}  tail={' '.join(tail)} ...")

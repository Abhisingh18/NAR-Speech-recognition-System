"""Compare EXPECTED vs PREDICTED first-</s> frame position.
Target layout: <s> + chars(+'|' for spaces) + </s>...  => expected_eos = 1 + len(REF_normalized).
predicted_eos = index of first </s> in the RAW frame readout.
offset = predicted - expected  (>0 model runs long/late; <0 early termination).
Reports offset stats binned by EXPECTED eos position (i.e. transcript length ~ utterance length).
Usage: python _eos_pos.py <decodes.txt> [label]"""
import re, sys, statistics as st

hdr = re.compile(r"=+\s+(\S+)\s+dur=([\d.]+)s\s+T=(\d+)\s+n_keep=(\d+)\s+WER=([\d.]+)")
path = sys.argv[1]; label = sys.argv[2] if len(sys.argv) > 2 else path
recs = []; cur = None
for line in open(path):
    m = hdr.search(line)
    if m:
        cur = dict(dur=float(m.group(2)), nkeep=int(m.group(4)), wer=float(m.group(5))); recs.append(cur)
    elif cur is not None:
        if line.startswith("RAW"): cur["raw"] = line.split(":",1)[1].split()
        elif line.startswith("REF"): cur["ref"] = line.split(":",1)[1].rstrip("\n").strip()

rows = []
for r in recs:
    if "raw" not in r or "ref" not in r: continue
    toks = r["raw"][: r["nkeep"]]
    if "</s>" not in toks:  # never terminates
        rows.append((r, 1+len(r["ref"]), None)); continue
    exp = 1 + len(r["ref"])          # <s> + one token per REF character (spaces already counted)
    pred = toks.index("</s>")
    rows.append((r, exp, pred))

def stats(sub, name):
    off = [p-e for (_,e,p) in sub if p is not None]
    noeos = sum(1 for (_,_,p) in sub if p is None)
    if not off:
        print(f"  [{name:>9}] n={len(sub):4d}  (no eos in all)"); return
    ab = [abs(x) for x in off]
    within = lambda k: 100*sum(a<=k for a in ab)/len(ab)
    print(f"  [{name:>9}] n={len(sub):4d}  medExp={st.median([e for _,e,_ in sub]):5.0f}fr  "
          f"medOffset={st.median(off):+5.0f}fr  med|off|={st.median(ab):4.0f}fr  "
          f"|off|<=5:{within(5):4.0f}%  <=20:{within(20):4.0f}%  noEOS={noeos}")

print(f"\n===== {label}  (n={len(rows)}) =====")
print("  binned by EXPECTED eos frame (≈ transcript length; 50 fr = 1 s of speech):")
for lo,hi in [(0,150),(150,300),(300,500),(500,900),(900,9999)]:
    stats([x for x in rows if lo <= x[1] < hi], f"{lo}-{hi}")
print("  offset sign (all clips):")
off = [p-e for (_,e,p) in rows if p is not None]
print(f"    early (<-5fr): {sum(o<-5 for o in off)}   on-time (±5fr): {sum(-5<=o<=5 for o in off)}   late (>+5fr): {sum(o>5 for o in off)}")

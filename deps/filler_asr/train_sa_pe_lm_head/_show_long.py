"""Show the longest-clip predictions from two decodes.txt (ALiBi vs baseline) on the SAME keys.
Prints the per-frame RAW readout with the <fill> run collapsed, so you can see where </s>
lands and that the tail is clean fill. Usage: python _show_long.py <alibi.decodes> <base.decodes> [k]"""
import re, sys

hdr = re.compile(r"=+\s+(\S+)\s+dur=([\d.]+)s\s+T=(\d+)\s+n_keep=(\d+)\s+WER=([\d.]+)")

def parse(path):
    recs = {}; cur = None
    for line in open(path):
        m = hdr.search(line)
        if m:
            cur = dict(key=m.group(1), dur=float(m.group(2)), T=int(m.group(3)),
                       nkeep=int(m.group(4)), wer=float(m.group(5)))
            recs[cur["key"]] = cur
        elif cur is None:
            continue
        elif line.startswith("RAW"):   cur["raw"] = line.split(":",1)[1].strip()
        elif line.startswith("HYP"):   cur["hyp"] = line.split(":",1)[1].strip()
        elif line.startswith("REF"):   cur["ref"] = line.split(":",1)[1].strip()
    return recs

def collapse_fill(raw):
    """Collapse the trailing <fill> run into '<fill> x N' and show content + a few fill."""
    toks = raw.split()
    # find first </s>
    try: e = toks.index("</s>")
    except ValueError: e = len(toks)
    # keep everything up to a few tokens past the </s> block, then count remaining <fill>
    head = toks[:e]
    tailblock = toks[e:]
    # split tailblock into leading (</s>/content) vs pure fill run
    j = 0
    while j < len(tailblock) and tailblock[j] != "<fill>":
        j += 1
    shown = tailblock[:j]                 # </s> </s> </s> (+ any leaked content)
    nfill = sum(1 for t in tailblock[j:] if t == "<fill>")
    leaked = [t for t in tailblock[j:] if t not in ("<fill>",)]
    s = " ".join(head + shown)
    s += f"   <fill>x{nfill}"
    if leaked:
        s += f"   [LEAKED AFTER FILL: {' '.join(leaked[:20])}]"
    return s

A = parse(sys.argv[1]); B = parse(sys.argv[2])
k = int(sys.argv[3]) if len(sys.argv) > 3 else 3
longest = sorted(A.values(), key=lambda r: -r["dur"])[:k]
for r in longest:
    key = r["key"]; b = B.get(key)
    print("="*100)
    print(f"KEY {key}   dur={r['dur']:.1f}s   T={r['T']}   n_keep={r['nkeep']}")
    print(f"REF                : {r['ref']}")
    print("-"*100)
    print(f"ALiBi HYP  (WER {r['wer']:.2f}): {r['hyp']}")
    print(f"ALiBi RAW  : {collapse_fill(r['raw'])}")
    if b:
        print("-"*100)
        print(f"BASE  HYP  (WER {b['wer']:.2f}): {b['hyp']}")
        print(f"BASE  RAW  : {collapse_fill(b['raw'])}")
    print()

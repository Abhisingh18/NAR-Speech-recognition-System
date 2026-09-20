"""Find every 'leak' clip (content token AFTER the first </s>) across given decodes.txt files
and print: REF, RAW hyp (special symbols shown, fill run collapsed), and cleaned hyp.
Usage: python _show_leaks.py label1=path1 label2=path2 ..."""
import re, sys

hdr = re.compile(r"=+\s+(\S+)\s+dur=([\d.]+)s\s+T=(\d+)\s+n_keep=(\d+)\s+WER=([\d.]+)")
def is_content(t): return not t.startswith("<")

def clean(tokens):
    """char tokens -> words: drop specials, '|' -> space."""
    out = []
    for t in tokens:
        if t.startswith("<"): continue
        out.append(" " if t == "|" else t)
    return re.sub(r"\s+", " ", "".join(out)).strip()

def collapse(tokens):
    """render token list with the long <fill> run collapsed to <fill>xN."""
    s, i = [], 0
    while i < len(tokens):
        if tokens[i] == "<fill>":
            j = i
            while j < len(tokens) and tokens[j] == "<fill>": j += 1
            s.append(f"<fill>x{j-i}"); i = j
        else:
            s.append(tokens[i]); i += 1
    return " ".join(s)

def parse(path):
    recs = {}; cur = None
    for line in open(path):
        m = hdr.search(line)
        if m:
            cur = dict(key=m.group(1), dur=float(m.group(2)), nkeep=int(m.group(4)), wer=float(m.group(5)))
            recs[cur["key"]] = cur
        elif cur is not None:
            if line.startswith("RAW"): cur["raw"] = line.split(":",1)[1].split()
            elif line.startswith("HYP"): cur["hyp"] = line.split(":",1)[1].strip()
            elif line.startswith("REF"): cur["ref"] = line.split(":",1)[1].strip()
    return recs

total = 0
for arg in sys.argv[1:]:
    label, path = arg.split("=", 1)
    recs = parse(path)
    leaks = []
    for r in recs.values():
        toks = r["raw"][: r["nkeep"]]
        if "</s>" not in toks: continue
        e = toks.index("</s>")
        after = toks[e+1:]
        if any(is_content(x) for x in after):
            leaks.append((r, e, after))
    for r, e, after in leaks:
        total += 1
        # tokens from just before </s> through the leaked tail (collapse fill)
        window = r["raw"][max(0, e-1): r["nkeep"]]
        leaked_chars = clean([x for x in after])          # what the post-</s> tail decodes to
        print("="*104)
        print(f"[{label}]  {r['key']}   dur={r['dur']:.1f}s   WER={r['wer']:.2f}   (</s> at frame {e})")
        print(f"  REF                     : {r['ref']}")
        print(f"  HYP (specials removed)  : {r['hyp']}")
        print(f"  RAW tail w/ symbols     : … {collapse(window)}")
        print(f"  leaked-tail decoded     : '{leaked_chars}'")
        print()
print(f"##### TOTAL LEAK CLIPS: {total} #####")

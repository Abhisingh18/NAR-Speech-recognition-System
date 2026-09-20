"""Sort a decode_iterative.py *.decodes.txt by audio length and annotate each clip with the
expected vs predicted </s> position (and the related structural fields).

Per record it inserts one [eos] line right under the header:
    expected </s>@E  = 1 + transcript length (chars, spaces->|)  -> where </s> *should* land
    predicted </s>@P = first </s> frame in the model's RAW prediction (None if never emitted)
    got/exp          = P/E   (1.0 = perfect; <1 fires early -> deletions; >1 fires late)
    tail_leak        = # non-<fill> frames AFTER </s> (content leaking into the tail)
    fill_in_body     = # <fill> frames BEFORE </s> (mid-sentence gaps)
    letters          = # real content tokens (not <s>/</s>/<fill>/|) in the body

Usage: python sort_decodes_by_duration.py --decodes <file.decodes.txt> [--order desc|asc] [--out <file>]
"""
import re, argparse


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--decodes", required=True)
    ap.add_argument("--order", choices=["desc", "asc"], default="desc")
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    out = args.out or args.decodes.replace(".txt", f".by_duration_{args.order}.txt")

    # --- split into records at each header line ---
    lines = open(args.decodes).read().splitlines()
    recs = []; cur = None
    for ln in lines:
        if ln.startswith("===="):
            if cur is not None: recs.append(cur)
            cur = {"header": ln, "body": []}
        elif cur is not None:
            cur["body"].append(ln)
    if cur is not None: recs.append(cur)

    def field(body, prefix):
        for ln in body:
            if ln.startswith(prefix):
                return ln.split(":", 1)[1].split()
        return []

    for r in recs:
        m = re.search(r"dur=([\d.]+)s", r["header"])
        r["dur"] = float(m.group(1)) if m else 0.0
        raw = field(r["body"], "RAW")
        ref_toks = field(r["body"], "REF")
        ref = " ".join(ref_toks)
        eos = next((i for i, t in enumerate(raw) if t == "</s>"), None)
        exp = 1 + len(re.sub(r"[^a-z' ]", "", ref.lower()).replace(" ", "|"))
        end = eos if eos is not None else len(raw)
        leak = sum(1 for t in raw[end:] if t not in ("<fill>", "</s>"))
        fillbody = sum(1 for t in raw[:end] if t == "<fill>")
        letters = sum(1 for t in raw[:end] if t not in ("<s>", "</s>", "<fill>", "|"))
        ratio = f"{eos/exp:.2f}" if (eos is not None and exp > 0) else "n/a"
        r["eos_line"] = (f"[eos]  expected </s>@{exp}   predicted </s>@{eos if eos is not None else 'NONE'}"
                         f"   got/exp={ratio}   tail_leak={leak}   fill_in_body={fillbody}   letters={letters}")

    recs.sort(key=lambda r: r["dur"], reverse=(args.order == "desc"))

    with open(out, "w") as f:
        f.write(f"# sorted by audio length ({args.order}); n={len(recs)}\n")
        f.write("# [eos] expected@=1+transcript_len ; predicted@=first </s> frame ; "
                "got/exp<1 fires early ; tail_leak=content after </s> ; fill_in_body=gaps before </s>\n\n")
        for r in recs:
            f.write(r["header"] + "\n")
            f.write(r["eos_line"] + "\n")
            f.write("\n".join(r["body"]) + "\n\n")
    print(f"wrote {out}  ({len(recs)} clips, {args.order} by duration)")
    print("longest 3:")
    for r in recs[:3]:
        print("  " + re.sub(r"^=+ | =+$", "", r["header"]).strip() + "  ||  " + r["eos_line"])


if __name__ == "__main__":
    main()

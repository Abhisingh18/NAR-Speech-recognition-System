"""Dump the utterances whose decode emits REAL CHARACTERS (a-z / ') after the
first <fill>, in a readable run-length-encoded form.

The per-frame sequence is shown as run-length tokens (e.g. `<fill>x40`) and split
at the FIRST <fill> with a marker, so you can see at a glance what content leaks
into the fill region. For each utt we also list the (frame_index, token) of every
real char that occurs after the first <fill>, plus Hyp / Ref.

Usage:
  python dump_chars_after_fill.py [--jsonl ...] [--out results/chars_after_fill.txt]
                                  [--preview 8] [--sort count|order]
"""
import os
import json
import argparse

HERE = os.path.dirname(os.path.abspath(__file__))
FILL, PAD, BOS, EOS, UNK = "<fill>", "<pad>", "<s>", "</s>", "<unk>"
DELIM = "|"
SPECIAL = {FILL, PAD, BOS, EOS, UNK}


def is_char(t):  # real transcript character: a-z and apostrophe
    return t not in SPECIAL and t != DELIM


def rle(tokens):
    """Run-length-encode a token list -> 'tok' or 'tokxN' joined by spaces."""
    out = []
    for t in tokens:
        if out and out[-1][0] == t:
            out[-1][1] += 1
        else:
            out.append([t, 1])
    return " ".join(f"{t}x{c}" if c > 1 else t for t, c in out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl", default=os.path.join(
        HERE, "results", "sa_Hubert_SA_finetuning_checkpoint-11000_test_clean.jsonl"))
    ap.add_argument("--out", default=os.path.join(HERE, "results", "chars_after_fill.txt"))
    ap.add_argument("--preview", type=int, default=8, help="also print N examples to stdout")
    ap.add_argument("--sort", choices=["count", "order"], default="count",
                    help="count = most chars-after-fill first; order = manifest order")
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(args.jsonl) if l.strip()]

    selected = []
    for d in rows:
        ft = d["frame_tokens"]
        T = len(ft)
        ff = next((i for i, t in enumerate(ft) if t == FILL), None)
        if ff is None:
            continue
        after = [(i, t) for i, t in enumerate(ft[ff + 1:], start=ff + 1) if is_char(t)]
        if not after:
            continue
        selected.append((d, ff, T, after))

    if args.sort == "count":
        selected.sort(key=lambda x: len(x[3]), reverse=True)

    def render(d, ff, T, after):
        ft = d["frame_tokens"]
        prefix_rle = rle(ft[:ff])
        suffix_rle = rle(ft[ff:])  # begins at the first <fill>
        chars = " ".join(f"{i}:{t}" for i, t in after)
        return (
            f"Key: {d['key']}   frames={T}  first_fill@{ff} ({ff/T:.2f})  "
            f"real_chars_after_fill={len(after)}\n"
            f"Seq:  {prefix_rle}  >>>FIRST<fill>@{ff}>>>  {suffix_rle}\n"
            f"Chars_after_fill (frame:tok): {chars}\n"
            f"Hyp:  {d['hyp']}\n"
            f"Ref:  {d['ref']}\n"
        )

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        f.write(f"utterances with >=1 REAL CHAR after first <fill>: {len(selected)}/{len(rows)}\n")
        f.write(f"sorted by: {args.sort}\n")
        f.write("(Seq is run-length encoded: tokxN means tok repeated N times)\n\n")
        for d, ff, T, after in selected:
            f.write(render(d, ff, T, after) + "\n")

    print(f"{len(selected)}/{len(rows)} utts have a real char after the first <fill>")
    print(f"written -> {args.out}\n")
    print(f"===== first {min(args.preview, len(selected))} examples "
          f"({'most chars-after-fill first' if args.sort=='count' else 'manifest order'}) =====\n")
    for d, ff, T, after in selected[:args.preview]:
        print(render(d, ff, T, after))


if __name__ == "__main__":
    main()

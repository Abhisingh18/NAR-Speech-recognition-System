"""In-depth analysis of the SA per-frame decode dump (results/*_fulldump-style jsonl).

Probes the NATURE of the emissions, not just WER/CER. The training target was
front-loaded:  [<s>] + chars/| + [</s>] + [<fill>]*(T-len)  -- all content first,
then pure <fill>. So the central questions are:
  * Does the model ever emit CONTENT (chars / |) AFTER the first <fill>?  (target says never)
  * Where does the <fill> region begin, and how big is the trailing pure-fill run?
  * What is the content actually made of (real chars a-z vs the | delimiter)?
  * Does it open with <s> / ever emit </s>, and where?

Usage:
  python analyze_sa_decodes.py [--jsonl results/sa_..._test_clean.jsonl] [--examples N]
"""
import os
import json
import argparse
import collections
import statistics as st

HERE = os.path.dirname(os.path.abspath(__file__))
FILL, PAD, BOS, EOS, UNK = "<fill>", "<pad>", "<s>", "</s>", "<unk>"
DELIM = "|"
SPECIAL = {FILL, PAD, BOS, EOS, UNK}


def is_char(t):       # real transcript character: a-z and apostrophe
    return t not in SPECIAL and t != DELIM


def is_content(t):    # anything that is NOT padding-like fill/pad
    return t not in (FILL, PAD)


def pct(xs, p):
    if not xs:
        return float("nan")
    xs = sorted(xs)
    k = max(0, min(len(xs) - 1, int(round(p / 100 * (len(xs) - 1)))))
    return xs[k]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl", default=os.path.join(
        HERE, "results", "sa_Hubert_SA_finetuning_checkpoint-11000_test_clean.jsonl"))
    ap.add_argument("--examples", type=int, default=6)
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(args.jsonl) if l.strip()]
    N = len(rows)

    n_with_fill = 0
    n_content_after_fill = 0     # any non-fill/non-pad token after first <fill>
    n_char_after_fill = 0        # any REAL CHAR (a-z/') after first <fill>
    content_after_counts = []    # per-utt count of content tokens after first fill
    char_after_counts = []       # per-utt count of real chars after first fill
    first_fill_frac = []         # index of first <fill> / T
    trailing_fill_frac = []      # length of final contiguous <fill> run / T
    fill_runs = []               # number of separate <fill> runs per utt
    starts_with_bos = 0
    has_eos = 0
    eos_before_first_fill = 0
    no_fill_at_all = 0

    after_fill_tok = collections.Counter()   # token composition AFTER first fill
    prefix_tok = collections.Counter()       # token composition BEFORE first fill
    global_tok = collections.Counter()
    examples = []

    for d in rows:
        ft = d["frame_tokens"]
        T = len(ft)
        global_tok.update(ft)

        if ft and ft[0] == BOS:
            starts_with_bos += 1
        if EOS in ft:
            has_eos += 1

        ff = next((i for i, t in enumerate(ft) if t == FILL), None)
        if ff is None:
            no_fill_at_all += 1
            prefix_tok.update(ft)
            # count fill runs = 0; first_fill_frac = 1.0 (all prefix)
            first_fill_frac.append(1.0)
            trailing_fill_frac.append(0.0)
            fill_runs.append(0)
            continue

        n_with_fill += 1
        first_fill_frac.append(ff / T)
        prefix_tok.update(ft[:ff])

        # trailing contiguous fill run
        k = T
        while k > 0 and ft[k - 1] == FILL:
            k -= 1
        trailing_fill_frac.append((T - k) / T)

        # number of separate <fill> runs
        runs = sum(1 for i, t in enumerate(ft)
                   if t == FILL and (i == 0 or ft[i - 1] != FILL))
        fill_runs.append(runs)

        after = ft[ff + 1:]
        content_after = [t for t in after if is_content(t)]
        char_after = [t for t in after if is_char(t)]
        after_fill_tok.update(content_after)
        content_after_counts.append(len(content_after))
        char_after_counts.append(len(char_after))
        if content_after:
            n_content_after_fill += 1
        if char_after:
            n_char_after_fill += 1

        if EOS in ft and ft.index(EOS) < ff:
            eos_before_first_fill += 1

        if content_after and len(examples) < args.examples:
            examples.append((d["key"], ff, T, len(content_after), len(char_after),
                             content_after[:40]))

    def line(label, val):
        print(f"  {label:<42} {val}")

    print(f"\n================ SA decode nature analysis ================")
    print(f"file: {args.jsonl}")
    print(f"utterances: {N}\n")

    print("--- <fill> region structure (target = content prefix, then pure fill) ---")
    line("utts with at least one <fill>:", f"{n_with_fill}/{N} ({100*n_with_fill/N:.1f}%)")
    line("utts with NO <fill> at all:", f"{no_fill_at_all}/{N}")
    line("first-<fill> position / T  (p10/p50/p90):",
         f"{pct(first_fill_frac,10):.3f} / {pct(first_fill_frac,50):.3f} / {pct(first_fill_frac,90):.3f}")
    line("trailing pure-<fill> run / T (p10/p50/p90):",
         f"{pct(trailing_fill_frac,10):.3f} / {pct(trailing_fill_frac,50):.3f} / {pct(trailing_fill_frac,90):.3f}")
    line("separate <fill> runs per utt (mean/median/max):",
         f"{st.mean(fill_runs):.2f} / {int(st.median(fill_runs))} / {max(fill_runs)}")

    print("\n--- CONTENT AFTER <fill>  (the key question) ---")
    line("utts with ANY content token after 1st <fill>:",
         f"{n_content_after_fill}/{N} ({100*n_content_after_fill/N:.1f}%)")
    line("utts with a REAL CHAR (a-z/') after 1st <fill>:",
         f"{n_char_after_fill}/{N} ({100*n_char_after_fill/N:.1f}%)")
    line("content tokens after fill / utt (mean/p50/p90/max):",
         f"{st.mean(content_after_counts):.2f} / {pct(content_after_counts,50)} / "
         f"{pct(content_after_counts,90)} / {max(content_after_counts)}")
    line("real chars after fill / utt   (mean/p50/p90/max):",
         f"{st.mean(char_after_counts):.2f} / {pct(char_after_counts,50)} / "
         f"{pct(char_after_counts,90)} / {max(char_after_counts)}")
    tot_after = sum(after_fill_tok.values())
    n_delim_after = after_fill_tok.get(DELIM, 0)
    n_char_after_tot = sum(v for t, v in after_fill_tok.items() if is_char(t))
    line("total content tokens after fill:", tot_after)
    if tot_after:
        line("  ... of which | (delimiter):", f"{n_delim_after} ({100*n_delim_after/tot_after:.1f}%)")
        line("  ... of which real chars:", f"{n_char_after_tot} ({100*n_char_after_tot/tot_after:.1f}%)")
    print("  top tokens emitted after the first <fill>:")
    for t, c in after_fill_tok.most_common(12):
        print(f"      {t!r:>8}: {c}")

    print("\n--- <s> / </s> behavior ---")
    line("utts starting with <s> at frame 0:", f"{starts_with_bos}/{N} ({100*starts_with_bos/N:.1f}%)")
    line("utts emitting </s> anywhere:", f"{has_eos}/{N} ({100*has_eos/N:.1f}%)")
    line("  ... </s> before the first <fill>:", f"{eos_before_first_fill}/{has_eos if has_eos else 1}")

    print("\n--- token composition ---")
    g_tot = sum(global_tok.values())
    p_tot = sum(prefix_tok.values())
    print(f"  GLOBAL (all {g_tot} frames):")
    for t, c in global_tok.most_common(10):
        print(f"      {t!r:>8}: {c:>9}  ({100*c/g_tot:5.1f}%)")
    print(f"  PREFIX only (the {p_tot} pre-fill frames the transcript comes from):")
    for t, c in prefix_tok.most_common(10):
        print(f"      {t!r:>8}: {c:>9}  ({100*c/p_tot:5.1f}%)")

    print(f"\n--- {len(examples)} example utts WITH content after <fill> ---")
    for key, ff, T, nc, nch, toks in examples:
        print(f"  {key}: first_fill@{ff}/{T}, {nc} content ({nch} real chars) after -> {toks}")
    print()


if __name__ == "__main__":
    main()

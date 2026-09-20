"""Replicate akshaya's fairseq CTC letter-tokenization on the slam IndicASR (AI4B) data,
to SEE the character tokens before committing to an A-CMLM vocab.

The char split is byte-for-byte from akshaya's espnet_to_fairseq_convert.py:extract_letter_and_word:
    ltr_line = ' '.join(list('|'.join(words))) + ' |'      # every codepoint = 1 token, word sep = |
    dict.ltr.txt = per-codepoint frequency table (first-appearance order); '|' forced to the top.

Usage:
    python prep_ltr_chars.py --jsonl /speech/tomson/exps/speech-recog/data/smear-more-hi-ta-te-ma/train_hi.jsonl \
        --field target --examples 3 --out_dir chars_hi
"""
import os
import re
import json
import argparse
import unicodedata
from collections import Counter


def akshaya_ltr(words):
    """Exactly akshaya's .ltr line: words -> '|'-joined -> per-codepoint -> space-separated + ' |'."""
    return ' '.join(list('|'.join(words))) + ' |'


def build_letter_dict(lines_words):
    """Exactly akshaya's letter_dict: '|' first (=1), then every non-space codepoint by first
    appearance, with frequency. (fairseq later prepends <s></s><pad><unk> and reads this order.)"""
    d = {}
    for words in lines_words:
        d['|'] = 1
        for ch in ' '.join(words).strip():
            if ch == ' ':
                continue
            d[ch] = d.get(ch, 0) + 1
    return d


def describe(ch):
    """Human-readable label for a codepoint: U+xxxx NAME (category)."""
    try:
        name = unicodedata.name(ch)
    except ValueError:
        name = "<no name>"
    return f"U+{ord(ch):04X} {name} [{unicodedata.category(ch)}]"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl", default="/speech/tomson/exps/speech-recog/data/smear-more-hi-ta-te-ma/train_hi.jsonl")
    ap.add_argument("--field", default="target")
    ap.add_argument("--limit", type=int, default=0, help="0=all lines; else first N")
    ap.add_argument("--examples", type=int, default=3)
    ap.add_argument("--rare_thresh", type=int, default=20, help="flag codepoints rarer than this")
    ap.add_argument("--out_dir", default="chars_hi")
    args = ap.parse_args()

    # --- read transcripts ---
    lines_words = []
    with open(args.jsonl) as f:
        for i, line in enumerate(f):
            if args.limit and i >= args.limit:
                break
            line = line.strip()
            if not line:
                continue
            txt = json.loads(line)[args.field].strip()
            words = txt.split()          # whitespace tokenization into words (like the espnet 'text')
            if words:
                lines_words.append(words)
    print(f"read {len(lines_words)} utterances from {args.jsonl} [field={args.field}]\n")

    # --- example .ltr lines (word form -> letter form) ---
    print("=" * 90)
    print(f"{args.examples} EXAMPLE UTTERANCES  (word form  ->  akshaya .ltr char split)")
    print("=" * 90)
    for words in lines_words[:args.examples]:
        print(f"  WRD : {' '.join(words)}")
        print(f"  LTR : {akshaya_ltr(words)}\n")

    # --- letter dict (exactly akshaya's build) ---
    ld = build_letter_dict(lines_words)
    tokens = [k for k in ld if k != '|']                       # the real char tokens (| is word sep)
    print("=" * 90)
    print(f"CHARACTER INVENTORY  —  {len(ld)} entries incl '|'  ({len(tokens)} real char tokens)")
    print("=" * 90)
    print("(sorted by frequency; fairseq dict.ltr.txt keeps FIRST-APPEARANCE order instead)\n")
    print(f"  {'tok':>4}  {'count':>10}   description")
    print(f"  {'|':>4}  {'(wordsep)':>10}   word boundary token")
    for ch, c in sorted(((k, v) for k, v in ld.items() if k != '|'), key=lambda x: -x[1]):
        print(f"  {ch:>4}  {c:>10}   {describe(ch)}")

    # --- categorize / flag ---
    def cat(ch):
        return unicodedata.category(ch)
    deva   = [c for c in tokens if 'ऀ' <= c <= 'ॿ']
    latin  = [c for c in tokens if ('a' <= c.lower() <= 'z')]
    digits = [c for c in tokens if cat(c) == 'Nd']
    punct  = [c for c in tokens if cat(c).startswith('P') or cat(c).startswith('S')]
    zerowidth = [c for c in tokens if c in ('​', '‌', '‍', '﻿', ' ')]
    rare   = [c for c in tokens if ld[c] < args.rare_thresh]

    print("\n" + "=" * 90)
    print("BREAKDOWN")
    print("=" * 90)
    print(f"  Devanagari-block codepoints : {len(deva)}")
    print(f"  Latin letters (foreign)     : {len(latin)}  -> {latin}")
    print(f"  digits (Nd)                 : {len(digits)} -> {digits}")
    print(f"  punctuation/symbol (P*/S*)  : {len(punct)} -> {punct}")
    print(f"  zero-width / nbsp (invisible): {len(zerowidth)} -> {[hex(ord(c)) for c in zerowidth]}")
    print(f"  rare (< {args.rare_thresh} occurrences)      : {len(rare)} -> {rare}")

    # --- NFC gotcha: precomposed vs decomposed nukta (ड़=U+095C vs ड+़) ---
    nfc_tokens = set()
    for ch in tokens:
        for c2 in unicodedata.normalize("NFC", ch):
            nfc_tokens.add(c2)
    print("\n" + "=" * 90)
    print("UNICODE NORMALIZATION (NFC)")
    print("=" * 90)
    print(f"  raw distinct codepoints         : {len(tokens)}")
    print(f"  after NFC normalize             : {len(nfc_tokens)}")
    nukta_pre = [c for c in tokens if c in 'क़ख़ग़ज़ड़ढ़फ़य़ऱ']
    print(f"  precomposed nukta forms present : {nukta_pre}")
    print(f"  standalone nukta U+093C present : {'़' in tokens}  "
          f"(if BOTH appear, the corpus mixes NFC/NFD -> normalize before building the vocab)")

    # --- write a fairseq-style dict.ltr.txt (first-appearance order, akshaya-identical) ---
    os.makedirs(args.out_dir, exist_ok=True)
    dict_path = os.path.join(args.out_dir, "dict.ltr.txt")
    with open(dict_path, "w") as f:
        for k, v in ld.items():
            if k == ' ':
                continue
            f.write(f"{k} {v}\n")
    print(f"\nwrote fairseq-style dict -> {dict_path}  ({len(ld)} lines)")
    print("(next: turn this inventory into the A-CMLM vocab.json — after NFC + dropping junk)")


if __name__ == "__main__":
    main()

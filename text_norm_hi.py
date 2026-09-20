"""Hindi text normalization + character vocab for the data2vec A-CMLM model.

Mirrors akshaya's fairseq letter tokenization (every Devanagari codepoint = 1 token, word sep
= '|') but adds the cleanup we decided on:
  * NFC-normalize  (collapses precomposed nukta ज़/ड़/... -> base + U+093C so encoding is consistent)
  * DROP punctuation / symbols / control-format chars (categories P*, S*, C*  -> incl. । , ? : ; $
    and the zero-width ZWSP/ZWNJ/ZWJ noise)
  * DROP anything outside the Devanagari block (foreign Latin code-switch etc.)
  * keep single spaces (become the '|' word-delimiter token downstream)

vocab.json layout matches filler_asr's canonical file (chars FIRST, then | <unk> <pad> <s> </s>);
build_acmlm_tokenizer_hi() then adds <fill> and <mask> as the last two ids, exactly like the
English build_acmlm_tokenizer — but WITHOUT its hardcoded len==33 assert.
"""
import os
import sys
import json
import unicodedata
from collections import Counter

FILLER_ROOT = os.environ.get("FILLER_ROOT",                                   # local copy
                             os.path.join(os.path.dirname(os.path.abspath(__file__)), "deps", "filler_asr"))
if FILLER_ROOT not in sys.path:
    sys.path.insert(0, FILLER_ROOT)
from filler_sa_reference import FILL_TOKEN, MASK_TOKEN                         # noqa: E402
from transformers import Wav2Vec2CTCTokenizer                                 # noqa: E402

DEVA_LO, DEVA_HI = "ऀ", "ॿ"        # Devanagari Unicode block


def normalize_hi(text):
    """NFC + keep only Devanagari-block codepoints and single spaces; drop everything else."""
    text = unicodedata.normalize("NFC", text)
    out = []
    prev_space = False
    for ch in text:
        if ch.isspace():
            if not prev_space:
                out.append(" ")
                prev_space = True
            continue
        prev_space = False
        # keep Devanagari letters + marks (matras/virama/nukta); drop danda ।/॥/॰ and any other
        # punctuation/symbol EVEN inside the block, and everything outside the block (foreign etc.)
        if DEVA_LO <= ch <= DEVA_HI and not unicodedata.category(ch)[0] in ("P", "S"):
            out.append(ch)
    return "".join(out).strip()


def char_frequencies(jsonl_paths, field="target"):
    """Count Devanagari char frequencies over the normalized transcripts (spaces excluded)."""
    freq = Counter()
    n = 0
    for p in jsonl_paths:
        with open(p) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                norm = normalize_hi(json.loads(line)[field])
                if not norm:
                    continue
                n += 1
                for ch in norm:
                    if ch != " ":
                        freq[ch] += 1
    return freq, n


def build_vocab(jsonl_paths, out_dir, field="target"):
    """Write vocab.json (chars-first, then | <unk> <pad> <s> </s>) from the corpus char set."""
    freq, n = char_frequencies(jsonl_paths, field)
    chars = [c for c, _ in freq.most_common()]                # most frequent first (order is cosmetic)
    vocab = {}
    for i, c in enumerate(chars):
        vocab[c] = i
    k = len(chars)
    for j, t in enumerate(["|", "<unk>", "<pad>", "<s>", "</s>"]):
        vocab[t] = k + j
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "vocab.json"), "w", encoding="utf-8") as f:
        json.dump(vocab, f, ensure_ascii=False, indent=0)
    # human-readable inventory
    with open(os.path.join(out_dir, "chars_report.txt"), "w", encoding="utf-8") as f:
        f.write(f"utterances used: {n}\nreal char tokens (excl |,specials): {k}\n\n")
        for c in chars:
            f.write(f"{c}\t{freq[c]}\tU+{ord(c):04X} {unicodedata.name(c,'?')}\n")
    return vocab, freq, n


def build_acmlm_tokenizer_hi(vocab_dir):
    """Load vocab.json + add <fill> then <mask> as the last two ids (Hindi-size-agnostic)."""
    tok = Wav2Vec2CTCTokenizer(
        os.path.join(vocab_dir, "vocab.json"),
        unk_token="<unk>", pad_token="<pad>", word_delimiter_token="|",
        bos_token="<s>", eos_token="</s>",
    )
    tok.add_special_tokens({"additional_special_tokens": [FILL_TOKEN]})
    assert tok.convert_tokens_to_ids(FILL_TOKEN) == len(tok) - 1, "<fill> must be the last real class"
    tok.add_special_tokens({"additional_special_tokens": [FILL_TOKEN, MASK_TOKEN]})
    assert tok.convert_tokens_to_ids(MASK_TOKEN) == len(tok) - 1, "<mask> must be the very last id"
    return tok


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonls", nargs="+",
                    default=["/speech/tomson/exps/speech-recog/data/smear-more-hi-ta-te-ma/train_hi.jsonl"])
    ap.add_argument("--field", default="target")
    ap.add_argument("--out_dir", default="vocab_hi")
    args = ap.parse_args()

    vocab, freq, n = build_vocab(args.jsonls, args.out_dir, args.field)
    tok = build_acmlm_tokenizer_hi(args.out_dir)
    k = len(freq)
    print(f"utterances                 : {n}")
    print(f"real char tokens (Devanagari): {k}")
    print(f"vocab.json entries          : {len(vocab)}  (chars + | + <unk><pad><s></s>)")
    print(f"tokenizer len (+<fill>+<mask>): {len(tok)}   <fill>={tok.convert_tokens_to_ids(FILL_TOKEN)}  <mask>={tok.convert_tokens_to_ids(MASK_TOKEN)}")
    print(f"cfg.vocab_size would be     : {len(tok) - 1}  (real classes incl <fill>, excl <mask>)")
    print(f"\nchar set (freq desc):\n  " + " ".join(c for c, _ in freq.most_common()))
    print(f"\nwrote: {args.out_dir}/vocab.json  and  {args.out_dir}/chars_report.txt")

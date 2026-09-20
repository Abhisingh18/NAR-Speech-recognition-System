"""Kannada + English (bilingual, code-switched) text normalization for the data2vec A-CMLM model.

Kannada twin of text_norm_hi.py. Every codepoint = 1 token, word sep = '|', but the kept
character set is Kannada block + lowercase Latin (so code-switched words like "sir", "doctor"
and the English-only utterances survive as Latin letters):
  * NFC-normalize
  * '-', '|', '~', '_', '/' and any whitespace -> single space ('|' is our word-delimiter token,
    so a literal '|' in a transcript must not survive as a character)
  * KEEP Kannada-block codepoints U+0C80-U+0CFF (letters, matras, virama, anusvara, ...),
    except punctuation/symbols and Kannada digits
  * KEEP ASCII letters, lowercased
  * DROP punctuation (. , ? : ' ’ ...), zero-width ZWSP/ZWNJ/ZWJ, stray Devanagari, everything else
  * digits (ASCII or Kannada) are dropped from the text; has_digit() lets the manifest prep
    discard such utterances for TRAINING, because the audio says the number as words

No transformers import at module level, so the manifest prep can run under plain python3.
"""
import os
import json
import unicodedata
from collections import Counter

KN_LO, KN_HI = 0x0C80, 0x0CFF          # Kannada Unicode block
SPACE_LIKE = set("-|~_/")


def _is_kannada_char(ch):
    o = ord(ch)
    return KN_LO <= o <= KN_HI and unicodedata.category(ch)[0] not in ("P", "S", "N")


def has_digit(text):
    """True if the RAW transcript contains any digit (ASCII 0-9, Kannada ೦-೯, or other Nd)."""
    return any(unicodedata.category(ch) == "Nd" for ch in text)


def normalize_kn(text):
    """NFC + keep Kannada letters/marks and lowercase Latin letters + single spaces."""
    text = unicodedata.normalize("NFC", text)
    out = []
    prev_space = True                    # also strips leading space
    for ch in text:
        if ch.isspace() or ch in SPACE_LIKE:
            if not prev_space:
                out.append(" ")
                prev_space = True
            continue
        if _is_kannada_char(ch):
            out.append(ch)
        elif ch.isascii() and ch.isalpha():
            out.append(ch.lower())
        else:
            continue                     # dropped char: do not reset prev_space
        prev_space = False
    return "".join(out).strip()


def build_vocab(texts, out_dir):
    """Write vocab.json (chars-first by frequency, then | <unk> <pad> <s> </s>) from NORMALIZED
    texts, plus a human-readable chars_report.txt. Same layout as vocab_hi/."""
    freq = Counter()
    n = 0
    for t in texts:
        if not t:
            continue
        n += 1
        for ch in t:
            if ch != " ":
                freq[ch] += 1
    chars = [c for c, _ in freq.most_common()]
    vocab = {c: i for i, c in enumerate(chars)}
    k = len(chars)
    for j, t in enumerate(["|", "<unk>", "<pad>", "<s>", "</s>"]):
        vocab[t] = k + j
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "vocab.json"), "w", encoding="utf-8") as f:
        json.dump(vocab, f, ensure_ascii=False, indent=0)
    n_kn = sum(1 for c in chars if KN_LO <= ord(c) <= KN_HI)
    with open(os.path.join(out_dir, "chars_report.txt"), "w", encoding="utf-8") as f:
        f.write(f"utterances used: {n}\nreal char tokens (excl |,specials): {k}  "
                f"(kannada={n_kn}, latin={k - n_kn})\n\n")
        for c in chars:
            f.write(f"{c}\t{freq[c]}\tU+{ord(c):04X} {unicodedata.name(c, '?')}\n")
    return vocab, freq, n


def build_acmlm_tokenizer_kn(vocab_dir):
    """Same tokenizer recipe as Hindi (vocab.json + <fill> + <mask> as the last two ids)."""
    from text_norm_hi import build_acmlm_tokenizer_hi     # language-agnostic builder
    return build_acmlm_tokenizer_hi(vocab_dir)

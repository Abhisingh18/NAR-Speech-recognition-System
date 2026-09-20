"""Build clean Kannada+English (bilingual) manifests + vocab for the data2vec A-CMLM model.

Inputs (all local on gpu17):
  train : /speech/abhishek/kannada_data/merged/train_kn_en_merged.jsonl   (kn + kn-en + en)
  dev   : /speech/abhishek/kannada_data/merged/dev_kn_en_merged.jsonl
  test  : /speech/abhishek/kannada_data/testsets/jsonl/*_kannada_test.jsonl

Per train/dev utterance:
  1. read wav header -> #samples, sr (must be 16 kHz)
  2. target_norm = normalize_kn(target)       (NFC, Kannada + lowercase Latin, no punct)
  3. DROP if: digits in raw text | empty after norm | dur < MIN_DUR | dur > MAX_DUR
             | len(target_norm) + 1 (<s>) + EOS_REPEAT > T   (would be truncated at 50 fps)
Test sets are NOT filtered (scored as-is); target_norm is added for WER.

Outputs (--out_dir, default data_kn/):
  train.jsonl, dev.jsonl, test_<name>.jsonl   {key, source, target(=normalized), target_raw,
                                              language, duration}
  prep_report.txt                              drop counts per reason / per language, hours
and vocab (--vocab_dir, default vocab_kn/) built from the kept TRAIN targets.

  python3 prep_kn_manifests.py --workers 32
"""
import os
import sys
import json
import glob
import wave
import argparse
from collections import Counter, defaultdict
from multiprocessing import Pool

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from text_norm_kn import normalize_kn, has_digit, build_vocab       # noqa: E402

SR = 16000


def compute_output_length(n):
    """Same 320x conv geometry as filler_sa_reference.compute_output_length (no torch import)."""
    for k, s in zip([10, 3, 3, 3, 3, 2, 2], [5, 2, 2, 2, 2, 2, 2]):
        n = (n - k) // s + 1
    return n


def wav_info(path):
    try:
        with wave.open(path) as w:
            return w.getnframes(), w.getframerate()
    except Exception:
        return None, None


def load(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def with_info(rows, workers):
    with Pool(workers) as p:
        infos = p.map(wav_info, [r["source"] for r in rows], chunksize=256)
    return infos


def clean_split(rows, infos, args, name, report):
    kept, drops = [], Counter()
    hours_in = hours_out = 0.0
    per_lang = defaultdict(Counter)
    for r, (n, sr) in zip(rows, infos):
        lang = r.get("language", "?")
        per_lang[lang]["in"] += 1
        if n is None:
            drops["unreadable_wav"] += 1; per_lang[lang]["drop"] += 1; continue
        if sr != SR:
            drops[f"sr_{sr}"] += 1; per_lang[lang]["drop"] += 1; continue
        dur = n / SR
        hours_in += dur / 3600
        raw = r.get("target", "")
        norm = normalize_kn(raw)
        T = compute_output_length(n)
        reason = None
        if has_digit(raw):
            reason = "has_digit"
        elif not norm:
            reason = "empty_after_norm"
        elif dur < args.min_dur:
            reason = f"dur<{args.min_dur}"
        elif dur > args.max_dur:
            reason = f"dur>{args.max_dur}"
        elif len(norm) + 1 + args.eos_repeat > T:
            reason = "too_many_chars_for_50fps"
        if reason:
            drops[reason] += 1; per_lang[lang]["drop"] += 1; continue
        hours_out += dur / 3600
        per_lang[lang]["kept"] += 1
        kept.append({"key": r.get("key"), "source": r["source"], "target": norm,
                     "target_raw": raw, "language": lang, "duration": round(dur, 3)})
    report.append(f"== {name}: in={len(rows)} ({hours_in:.1f} h)  kept={len(kept)} ({hours_out:.1f} h)  "
                  f"dropped={len(rows) - len(kept)}")
    for k, v in drops.most_common():
        report.append(f"     drop {k:<28} {v}")
    for lang, c in sorted(per_lang.items()):
        report.append(f"     lang {lang:<6} in={c['in']} kept={c['kept']} drop={c['drop']}")
    return kept


def write_jsonl(rows, path):
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default="/speech/abhishek/kannada_data/merged/train_kn_en_merged.jsonl")
    ap.add_argument("--dev", default="/speech/abhishek/kannada_data/merged/dev_kn_en_merged.jsonl")
    ap.add_argument("--test_glob", default="/speech/abhishek/kannada_data/testsets/jsonl/*_kannada_test.jsonl")
    ap.add_argument("--out_dir", default=os.path.join(HERE, "data_kn"))
    ap.add_argument("--vocab_dir", default=os.path.join(HERE, "vocab_kn"))
    ap.add_argument("--min_dur", type=float, default=0.5)
    ap.add_argument("--max_dur", type=float, default=20.0)
    ap.add_argument("--eos_repeat", type=int, default=3)
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--limit", type=int, default=0, help="0 = all; else first N rows per split (quick test)")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    report = [f"args: {vars(args)}"]

    for split, path in [("dev", args.dev), ("train", args.train)]:
        rows = load(path)
        if args.limit:
            rows = rows[:args.limit]
        print(f"[{split}] reading {len(rows)} wav headers ...", flush=True)
        start = len(report)
        kept = clean_split(rows, with_info(rows, args.workers), args, split, report)
        write_jsonl(kept, os.path.join(args.out_dir, f"{split}.jsonl"))
        print("\n".join(report[start:]), flush=True)
        if split == "train":
            vocab, freq, n = build_vocab([r["target"] for r in kept], args.vocab_dir)
            report.append(f"== vocab: {len(vocab)} entries ({len(freq)} chars + | <unk> <pad> <s> </s>) "
                          f"from {n} train utts -> {args.vocab_dir}; tokenizer adds <fill>,<mask>")
            chars = set(freq)
            dev_oov = Counter(c for r in load(os.path.join(args.out_dir, "dev.jsonl"))
                              for c in r["target"] if c != " " and c not in chars)
            report.append(f"     dev chars not in train vocab (-> <unk>): {dict(dev_oov.most_common(10))}")

    chars = set(json.load(open(os.path.join(args.vocab_dir, "vocab.json"), encoding="utf-8")))
    for tpath in sorted(glob.glob(args.test_glob)):
        name = os.path.basename(tpath).replace("_kannada_test.jsonl", "")
        rows = load(tpath)
        infos = with_info(rows, args.workers)
        out, oov, dig = [], Counter(), 0
        for r, (n, sr) in zip(rows, infos):
            norm = normalize_kn(r.get("target", ""))
            dig += has_digit(r.get("target", ""))
            oov.update(c for c in norm if c != " " and c not in chars)
            out.append({"key": r.get("key"), "source": r["source"], "target": norm,
                        "target_raw": r.get("target", ""), "language": r.get("language", "kn"),
                        "duration": round((n or 0) / SR, 3)})
        write_jsonl(out, os.path.join(args.out_dir, f"test_{name}.jsonl"))
        report.append(f"== test_{name}: {len(out)} utts, {sum(r['duration'] for r in out)/3600:.2f} h, "
                      f"refs_with_digits={dig}, oov_chars={dict(oov.most_common(8))}")

    with open(os.path.join(args.out_dir, "prep_report.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(report) + "\n")
    print("\n".join(report), flush=True)


if __name__ == "__main__":
    main()

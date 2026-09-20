"""Prepare a LibriSpeech *test-clean* manifest for filler_asr inference.

Source of truth: an existing jsonl under
    /speech/tomson/exps/speech-recog/data/librispeech
whose lines look like:
    {"key": "...", "source": "/abs/path/to.wav", "target": "UPPERCASE TEXT"}

We DO NOT copy the wavs (they are large and already on disk); we only build a
self-contained manifest inside the filler_asr directory that references them by
absolute path. Output line schema:
    {"key": "...", "audio": "/abs/path/to.wav", "text": "UPPERCASE TEXT"}

Reference-text normalization (lowercase + strip punctuation) is applied at
scoring time in infer_test_clean_local.py, matching the training-time regex,
so the manifest keeps the original `target` verbatim.
"""
import os
import json
import argparse

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_SRC = "/speech/tomson/exps/speech-recog/data/librispeech/librispeech_test_clean.jsonl"
DEFAULT_OUT = os.path.join(HERE, "data", "test_clean.jsonl")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=DEFAULT_SRC, help="source librispeech test-clean jsonl")
    ap.add_argument("--out", default=DEFAULT_OUT, help="output manifest inside filler_asr")
    ap.add_argument("--check-wavs", action="store_true", help="assert every referenced wav exists")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    n, missing = 0, 0
    with open(args.src) as fin, open(args.out, "w") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            key = d.get("key")
            audio = d.get("source") or d.get("audio")
            text = d.get("target") or d.get("text") or ""
            if args.check_wavs and not os.path.exists(audio):
                missing += 1
                continue
            fout.write(json.dumps({"key": key, "audio": audio, "text": text}) + "\n")
            n += 1

    print(f"wrote {n} utterances -> {args.out}")
    if args.check_wavs:
        print(f"missing wavs skipped: {missing}")


if __name__ == "__main__":
    main()

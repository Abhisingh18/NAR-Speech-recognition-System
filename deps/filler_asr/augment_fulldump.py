"""Augment an existing test_clean_fulldump.{jsonl,txt} with audio-length fields.

Adds, per utterance:
  * audio_len_sec : true waveform duration (samples / 16000)
  * len_hyp / len_ref : char length of the plain transcript (no <fill>)
  * audio_per_hyp / audio_per_ref : audio_len_sec / text_len  (seconds per char)

This re-uses the per-frame predictions already stored in the JSONL, so it does
NOT re-run the model -- it only reads the wav headers (via soundfile.info) for
duration and recomputes the length ratios. Both the .jsonl and the human-readable
.txt are rewritten in place.

Run:
  python augment_fulldump.py
"""
import os
import json
import argparse

import soundfile as sf

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default=os.path.join(HERE, "data", "test_clean.jsonl"))
    ap.add_argument("--jsonl", default=os.path.join(HERE, "results", "test_clean_fulldump.jsonl"))
    ap.add_argument("--txt", default=os.path.join(HERE, "results", "test_clean_fulldump.txt"))
    args = ap.parse_args()

    # key -> audio path, from the manifest used to produce the dump.
    audio_by_key = {}
    for line in open(args.manifest):
        if line.strip():
            r = json.loads(line)
            audio_by_key[r["key"]] = r["audio"]

    rows = [json.loads(l) for l in open(args.jsonl) if l.strip()]
    print(f"utterances: {len(rows)}")

    dur_cache = {}
    for d in rows:
        path = audio_by_key[d["key"]]
        if path not in dur_cache:
            info = sf.info(path)
            dur_cache[path] = info.frames / info.samplerate
        audio_len = dur_cache[path]
        len_hyp = len(d["hyp"])
        len_ref = len(d["ref"])
        d["audio_len_sec"] = round(audio_len, 4)
        d["len_hyp"] = len_hyp
        d["len_ref"] = len_ref
        d["audio_per_hyp"] = round(audio_len / max(1, len_hyp), 6)
        d["audio_per_ref"] = round(audio_len / max(1, len_ref), 6)

    # Rewrite JSONL with the new keys placed next to content_frac, for readability.
    order = ["key", "num_frames", "num_fill", "content_frac",
             "audio_len_sec", "len_hyp", "len_ref", "audio_per_hyp", "audio_per_ref",
             "frame_tokens", "full", "hyp", "ref"]
    with open(args.jsonl, "w", encoding="utf-8") as f:
        for d in rows:
            ordered = {k: d[k] for k in order if k in d}
            ordered.update({k: v for k, v in d.items() if k not in ordered})
            f.write(json.dumps(ordered) + "\n")

    # Rewrite the human-readable txt with the extra Audio line.
    with open(args.txt, "w", encoding="utf-8") as f:
        for d in rows:
            f.write(
                f"Key: {d['key']}  | frames={d['num_frames']} fill={d['num_fill']} "
                f"content_frac={d['content_frac']:.3f}\n"
                f"Audio: {d['audio_len_sec']:.2f}s | len(hyp)={d['len_hyp']} "
                f"len(ref)={d['len_ref']} | audio/hyp={d['audio_per_hyp']:.4f} "
                f"audio/ref={d['audio_per_ref']:.4f}\n"
                f"Full: {d['full']}\n"
                f"Hyp:  {d['hyp']}\n"
                f"Ref:  {d['ref']}\n\n"
            )

    print(f"updated -> {args.jsonl}")
    print(f"updated -> {args.txt}")


if __name__ == "__main__":
    main()

"""
Gate 1 -- verify the duration frame-budget loss mask in DataCollatorForFillerASR.

Builds a small batch of synthetic utterances (known durations + transcripts),
runs them through the REAL collator, and asserts per utterance that:
  * n_keep             == ceil(n_samples/16000 * FRAMES_PER_SEC)   (from collator)
  * graded positions    are exactly indices [0, min(n_keep, T))
  * positions >= n_keep  are -100
  * cross-batch <pad>    is -100
  * the graded label ids match the original (un-masked) target tokens

Run on CPU:  CUDA_VISIBLE_DEVICES="" python verify_mask.py
"""
import json
import math
import numpy as np
import torch
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2CTCTokenizer

from dataset import compute_output_length
from collator import DataCollatorForFillerASR, SAMPLING_RATE, FRAMES_PER_SEC

CFG = "config_xlarge_sa_2gpu.json"

# (duration_seconds, transcript) -- mix of fits / clips / exact
SAMPLES = [
    (2.5, "hello my name is tomson"),
    (1.3, "to all the people who care about me i am doing good"),  # dense -> may clip
    (0.8, "i recently"),
    (2.7, "sir whats your command"),
]


def build_feature(dur, text, tok):
    n_samples = int(round(dur * SAMPLING_RATE))
    T = compute_output_length(n_samples)
    ids = tok(text.replace(" ", "|")).input_ids
    if len(ids) + 2 > T:                       # mirror data_cached truncation
        ids = ids[: T - 2]
    labels = [tok.bos_token_id] + ids + [tok.eos_token_id]
    labels = labels + [tok.convert_tokens_to_ids("<fill>")] * (T - len(labels))
    return {
        "input_values": np.zeros(n_samples, dtype=np.float32),
        "labels": labels,
        "_dur": dur, "_T": T, "_content": len(ids) + 2,
    }


def main():
    cfg = json.load(open(CFG))
    cache_dir = cfg["cache_dir"]
    import os
    if os.path.exists(os.path.join(cache_dir, "preprocessor_config.json")):
        fe = Wav2Vec2FeatureExtractor.from_pretrained(cache_dir)
        tok = Wav2Vec2CTCTokenizer.from_pretrained(cache_dir)
        print(f"(loaded fe/tokenizer from cache: {cache_dir})")
    else:
        # cache not prepared yet -- the mask logic is index-based and doesn't need
        # the real data, so fall back to the repo vocab + pipeline feature extractor.
        fe = Wav2Vec2FeatureExtractor(feature_size=1, sampling_rate=SAMPLING_RATE,
                                      padding_value=0.0, do_normalize=True,
                                      return_attention_mask=True)
        tok = Wav2Vec2CTCTokenizer("vocab.json", unk_token="<unk>", pad_token="<pad>",
                                   word_delimiter_token="|", bos_token="<s>", eos_token="</s>")
        print("(cache not prepared -- using repo vocab.json + default feature extractor)")
    print(f"FRAMES_PER_SEC={FRAMES_PER_SEC}  SAMPLING_RATE={SAMPLING_RATE}\n")

    feats = [build_feature(d, t, tok) for d, t in SAMPLES]
    meta = [(f.pop("_dur"), f.pop("_T"), f.pop("_content")) for f in feats]
    # keep originals (collator pops nothing, but be safe)
    orig_labels = [list(f["labels"]) for f in feats]

    collator = DataCollatorForFillerASR(feature_extractor=fe, tokenizer=tok)
    batch = collator([dict(input_values=f["input_values"], labels=f["labels"]) for f in feats])
    L = batch["labels"]  # (B, Lmax)

    ok = True
    hdr = f"{'#':>2} {'dur':>5} {'T':>5} {'content':>8} {'n_keep':>7} {'graded':>7} {'clip?':>6} {'status':>7}"
    print(hdr); print("-" * len(hdr))
    for i, (dur, T, content) in enumerate(meta):
        n_keep = math.ceil((dur * SAMPLING_RATE) / SAMPLING_RATE * FRAMES_PER_SEC)
        row = L[i]
        graded = (row != -100)
        graded_idx = graded.nonzero(as_tuple=True)[0].tolist()
        expect_graded = min(n_keep, T)

        c1 = graded_idx == list(range(expect_graded))          # exactly [0, expect_graded)
        c2 = bool((row[n_keep:] == -100).all().item())          # tail masked
        c3 = all(L[i, j].item() == orig_labels[i][j]            # graded ids unchanged
                 for j in graded_idx)
        good = c1 and c2 and c3
        ok = ok and good
        print(f"{i:>2} {dur:>5} {T:>5} {content:>8} {n_keep:>7} {len(graded_idx):>7} "
              f"{('YES' if content > n_keep else 'no'):>6} {('PASS' if good else 'FAIL'):>7}")
        if not good:
            print(f"     c1(range)={c1} c2(tail-100)={c2} c3(ids-match)={c3}")

    # cross-batch pad check: shortest example must have -100 well past its own T
    short = min(range(len(meta)), key=lambda i: meta[i][1])
    pad_ok = bool((L[short, meta[short][1]:] == -100).all().item())
    print(f"\ncross-batch <pad> -> -100 (shortest utt, idx {short}): "
          f"{'PASS' if pad_ok else 'FAIL'}")
    ok = ok and pad_ok

    print("\n" + ("ALL CHECKS PASS" if ok else "*** SOME CHECKS FAILED ***"))
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()

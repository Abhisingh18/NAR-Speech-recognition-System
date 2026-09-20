"""Filler-HuBERT inference that DUMPS THE FULL per-frame output, INCLUDING <fill>.

The normal decode (infer_test_clean_local.py) calls
    tokenizer.batch_decode(ids, skip_special_tokens=True, group_tokens=False)
which throws away <fill>/<s>/</s>/<pad> and maps | -> space. That gives the clean
transcript but hides what every frame actually predicted.

Here we instead take the per-frame argmax ids and map EACH frame to its raw token
via `convert_ids_to_tokens`, so the output keeps <fill>, <s>, </s>, <pad> and the
word-delimiter | exactly as emitted -- one token per ~20 ms HuBERT frame, in order.

For every utterance we record:
  * frame_tokens : the full list, length T (= number of HuBERT frames)
  * full         : those tokens space-joined (human-readable, <fill> visible)
  * hyp          : the stripped transcript (for comparison)
  * ref          : normalized reference
  * num_frames / num_fill / content_frac

Outputs (default under results/):
  * test_clean_fulldump.txt   -- human-readable, per utterance
  * test_clean_fulldump.jsonl -- one JSON object per utterance (programmatic use)

Run:
  CUDA_VISIBLE_DEVICES=6 python infer_test_clean_fulldump.py --limit 5   # smoke
  CUDA_VISIBLE_DEVICES=6 python infer_test_clean_fulldump.py             # full 2620
"""
import os
import json
import argparse

import torch
from tqdm import tqdm
from transformers import Wav2Vec2FeatureExtractor, HubertConfig

# Reuse the verified pieces from the sibling script (no duplication).
from infer_test_clean_local import (
    FillerHubertModel,
    build_tokenizer,
    load_wav,
    normalize_ref,
    DEFAULT_CKPT,
)

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=DEFAULT_CKPT)
    ap.add_argument("--manifest", default=os.path.join(HERE, "data", "test_clean.jsonl"))
    ap.add_argument("--out_txt", default=os.path.join(HERE, "results", "test_clean_fulldump.txt"))
    ap.add_argument("--out_jsonl", default=os.path.join(HERE, "results", "test_clean_fulldump.jsonl"))
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--limit", type=int, default=0, help="0 = all; else first N utterances")
    args = ap.parse_args()

    print(f"checkpoint: {args.checkpoint}")
    print(f"manifest:   {args.manifest}")
    print(f"device:     {args.device}")

    tokenizer = build_tokenizer(args.checkpoint)
    fill_token = "<fill>"
    feature_extractor = Wav2Vec2FeatureExtractor(
        feature_size=1, sampling_rate=16000, padding_value=0.0,
        do_normalize=True, return_attention_mask=True,
    )

    cfg = HubertConfig.from_pretrained(args.checkpoint)
    cfg.apply_spec_augment = False
    model = FillerHubertModel.from_pretrained(args.checkpoint, config=cfg).eval().to(args.device)

    rows = [json.loads(l) for l in open(args.manifest) if l.strip()]
    if args.limit > 0:
        rows = rows[: args.limit]
    print(f"utterances: {len(rows)}")

    os.makedirs(os.path.dirname(args.out_txt), exist_ok=True)
    with torch.no_grad(), \
         open(args.out_txt, "w", encoding="utf-8") as ftxt, \
         open(args.out_jsonl, "w", encoding="utf-8") as fjs:
        for d in tqdm(rows):
            audio = load_wav(d["audio"])
            inputs = feature_extractor(audio, sampling_rate=16000, return_tensors="pt")
            input_values = inputs["input_values"].to(args.device)
            logits = model(input_values=input_values).logits        # (1, T, 33)
            ids = torch.argmax(logits, dim=-1)[0]                    # (T,)

            # FULL per-frame output -- one raw token per frame, <fill> kept.
            frame_tokens = tokenizer.convert_ids_to_tokens(ids.tolist())
            full = " ".join(frame_tokens)
            # Stripped transcript, for side-by-side comparison.
            hyp = tokenizer.batch_decode(ids.unsqueeze(0), skip_special_tokens=True,
                                         group_tokens=False)[0].strip()
            ref = normalize_ref(d["text"])

            num_frames = len(frame_tokens)
            num_fill = sum(1 for t in frame_tokens if t == fill_token)
            content_frac = (num_frames - num_fill) / max(1, num_frames)

            # Audio length in seconds (true waveform duration), and audio-length
            # to text-length ratios. Text length is the plain transcript char
            # count (hyp/ref never contain <fill>), so these are seconds/char.
            audio_len = len(audio) / 16000.0
            len_hyp = len(hyp)
            len_ref = len(ref)
            audio_per_hyp = audio_len / max(1, len_hyp)
            audio_per_ref = audio_len / max(1, len_ref)

            ftxt.write(
                f"Key: {d['key']}  | frames={num_frames} fill={num_fill} "
                f"content_frac={content_frac:.3f}\n"
                f"Audio: {audio_len:.2f}s | len(hyp)={len_hyp} len(ref)={len_ref} "
                f"| audio/hyp={audio_per_hyp:.4f} audio/ref={audio_per_ref:.4f}\n"
                f"Full: {full}\n"
                f"Hyp:  {hyp}\n"
                f"Ref:  {ref}\n\n"
            )
            fjs.write(json.dumps({
                "key": d["key"],
                "num_frames": num_frames,
                "num_fill": num_fill,
                "content_frac": round(content_frac, 4),
                "audio_len_sec": round(audio_len, 4),
                "len_hyp": len_hyp,
                "len_ref": len_ref,
                "audio_per_hyp": round(audio_per_hyp, 6),
                "audio_per_ref": round(audio_per_ref, 6),
                "frame_tokens": frame_tokens,
                "full": full,
                "hyp": hyp,
                "ref": ref,
            }) + "\n")

    print(f"full dump (txt)   -> {args.out_txt}")
    print(f"full dump (jsonl) -> {args.out_jsonl}")


if __name__ == "__main__":
    main()

"""Decode the SA checkpoint (FillerHubertSAModel) on test-clean: WER/CER + FULL dump.

The checkpoints under experiments/Hubert_SA_finetuning are FillerHubertSAModel
(frozen HuBERT-xlarge encoder + 2 new self-attention layers + new lm_head), NOT
the plain FillerHubertModel that infer_test_clean_local.py / infer_test_clean_fulldump.py
load. Loading this checkpoint with the plain model would silently drop the SA layers,
so we use model_sa.FillerHubertSAModel here.

This single pass produces BOTH:
  * the stripped transcript + corpus WER/CER  (skip_special_tokens, group_tokens=False)
  * the FULL per-frame output  -- one raw token per ~20 ms frame, <fill>/<s>/</s>/<pad>
    and the word-delimiter | kept exactly as emitted (nothing stripped).

All decode helpers (tokenizer build, wav load, ref normalization, Levenshtein) are
reused unchanged from infer_test_clean_local.py -- only the model class differs.

Outputs (default under results/):
  * sa_<ckpt>_test_clean.txt    -- header WER/CER, then per-utt Full / Hyp / Ref
  * sa_<ckpt>_test_clean.jsonl  -- one JSON object per utterance (programmatic)

Run:
  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=2 \
      conda run -n filler_asr python infer_sa_test_clean.py \
      --checkpoint experiments/Hubert_SA_finetuning/checkpoint-11000 --limit 5   # smoke
  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=2 \
      conda run -n filler_asr python infer_sa_test_clean.py \
      --checkpoint experiments/Hubert_SA_finetuning/checkpoint-11000            # full 2620
"""
import os
import json
import argparse

import torch
from tqdm import tqdm
from transformers import Wav2Vec2FeatureExtractor, HubertConfig

from model_sa import FillerHubertSAModel
from infer_test_clean_local import (
    build_tokenizer,
    load_wav,
    normalize_ref,
    levenshtein,
)

HERE = os.path.dirname(os.path.abspath(__file__))
FILL_TOKEN = "<fill>"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True,
                    help="path to a FillerHubertSAModel checkpoint dir")
    ap.add_argument("--manifest", default=os.path.join(HERE, "data", "test_clean.jsonl"))
    ap.add_argument("--out_txt", default=None)
    ap.add_argument("--out_jsonl", default=None)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--limit", type=int, default=0, help="0 = all; else first N utterances")
    args = ap.parse_args()

    # Default output names tagged with the experiment + checkpoint step.
    ckpt = os.path.normpath(args.checkpoint)
    tag = f"{os.path.basename(os.path.dirname(ckpt))}_{os.path.basename(ckpt)}"  # Hubert_SA_finetuning_checkpoint-11000
    if args.out_txt is None:
        args.out_txt = os.path.join(HERE, "results", f"sa_{tag}_test_clean.txt")
    if args.out_jsonl is None:
        args.out_jsonl = os.path.join(HERE, "results", f"sa_{tag}_test_clean.jsonl")

    print(f"checkpoint: {args.checkpoint}")
    print(f"manifest:   {args.manifest}")
    print(f"device:     {args.device}")

    tokenizer = build_tokenizer(args.checkpoint)
    feature_extractor = Wav2Vec2FeatureExtractor(
        feature_size=1, sampling_rate=16000, padding_value=0.0,
        do_normalize=True, return_attention_mask=True,
    )

    cfg = HubertConfig.from_pretrained(args.checkpoint)  # carries num_sa_layers, sa_nhead, ...
    cfg.apply_spec_augment = False  # deterministic; eval() also disables masking
    model = FillerHubertSAModel.from_pretrained(args.checkpoint, config=cfg).eval().to(args.device)

    rows = [json.loads(l) for l in open(args.manifest) if l.strip()]
    if args.limit > 0:
        rows = rows[: args.limit]
    print(f"utterances: {len(rows)}")

    w_err = w_tot = c_err = c_tot = 0
    os.makedirs(os.path.dirname(args.out_txt), exist_ok=True)
    with torch.no_grad(), \
         open(args.out_txt, "w", encoding="utf-8") as ftxt, \
         open(args.out_jsonl, "w", encoding="utf-8") as fjs:
        ftxt.write(f"Checkpoint: {args.checkpoint}\n")
        ftxt.write(f"Manifest:   {args.manifest}  ({len(rows)} utts)\n")
        ftxt.write("(WER/CER summary appended at end of run; per-utt records below)\n\n")

        for d in tqdm(rows):
            audio = load_wav(d["audio"])
            inputs = feature_extractor(audio, sampling_rate=16000, return_tensors="pt")
            input_values = inputs["input_values"].to(args.device)
            logits = model(input_values=input_values).logits     # (1, T, 33)
            ids = torch.argmax(logits, dim=-1)[0]                 # (T,)

            # FULL per-frame output (raw tokens, <fill> kept).
            frame_tokens = tokenizer.convert_ids_to_tokens(ids.tolist())
            full = " ".join(frame_tokens)
            # Stripped transcript for WER/CER.
            hyp = tokenizer.batch_decode(ids.unsqueeze(0), skip_special_tokens=True,
                                         group_tokens=False)[0].strip()
            ref = normalize_ref(d["text"])

            rw, hw = ref.split(), hyp.split()
            w_err += levenshtein(rw, hw); w_tot += len(rw)
            c_err += levenshtein(ref, hyp); c_tot += len(ref)

            num_frames = len(frame_tokens)
            num_fill = sum(1 for t in frame_tokens if t == FILL_TOKEN)
            content_frac = (num_frames - num_fill) / max(1, num_frames)

            ftxt.write(
                f"Key: {d['key']}  | frames={num_frames} fill={num_fill} "
                f"content_frac={content_frac:.3f}\n"
                f"Full: {full}\n"
                f"Hyp:  {hyp}\n"
                f"Ref:  {ref}\n\n"
            )
            fjs.write(json.dumps({
                "key": d["key"],
                "num_frames": num_frames,
                "num_fill": num_fill,
                "content_frac": round(content_frac, 4),
                "frame_tokens": frame_tokens,
                "full": full,
                "hyp": hyp,
                "ref": ref,
            }) + "\n")

        wer = w_err / max(1, w_tot)
        cer = c_err / max(1, c_tot)
        summary = (f"\n===== SUMMARY =====\n"
                   f"Test-Clean WER: {wer:.4f}  ({w_err}/{w_tot} words)\n"
                   f"Test-Clean CER: {cer:.4f}  ({c_err}/{c_tot} chars)\n")
        ftxt.write(summary)

    print(f"\nTest-Clean WER: {wer:.4f}  ({w_err}/{w_tot} words)")
    print(f"Test-Clean CER: {cer:.4f}  ({c_err}/{c_tot} chars)")
    print(f"full dump (txt)   -> {args.out_txt}")
    print(f"full dump (jsonl) -> {args.out_jsonl}")


if __name__ == "__main__":
    main()

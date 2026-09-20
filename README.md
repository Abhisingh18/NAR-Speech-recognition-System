# indic-nar-filler_asr

Non-autoregressive **A-CMLM** ASR head (the 8-self-attention-block + text-embedding + lm_head
recipe from `filler_asr`) on a **frozen data2vec-aqc (SPRING-INX)** encoder instead of frozen
HuBERT-xlarge. The tap is taken **before the CTC head** — `Data2VecAQCEncoder` instantiates only
the inner `data2vec_audio` body, so the CTC / AQC pretraining heads (`contr_proj`, `project_q`,
`quantizer`, `_ema`) are physically absent.

Everything downstream of the tap is **reused unchanged** from the verified filler_asr pipeline
(span masking, ALiBi/PE, ADDITION text fusion, iterative decode, loss). Only the encoder and the
feature width change: **1280 (HuBERT-xlarge) → 1024 (data2vec-aqc)**.

## Env
`indic-nar-filler_asr` — a clone of `smear-moe-gemma3` (already has the working fairseq 0.12.2 +
data2vec-aqc stack **and** the transformers-5.x stack the A-CMLM head needs). `filler_asr` itself
can't host this: fairseq 0.12.2 won't install on its torch 2.7.

## Layout
- `model_d2v_acmlm.py` — `Data2VecEncoderAdapter` (makes the fairseq encoder quack like HF
  `HubertModel`) + `build_d2v_acmlm(...)` (builds the A-CMLM head at 1024 and swaps in the frozen
  data2vec body) + `build_collator(...)`.
- `smoke.py` / `run_smoke.sh` — 6-check smoke on LibriSpeech.

## Smoke
```bash
GPU=6 bash run_smoke.sh
```
Verifies: (1) encoder frozen / head trainable, (2) **frame geometry** data2vec T ==
`compute_output_length` (frame↔label alignment), (3) forward → logits + finite loss ~ln(V),
(4) backward → head grads flow, frozen body gets none, (5) deterministic tap in eval,
(6) 30-step single-batch overfit → loss falls (head learns through the tap).

Last verified run (GPU 6): encoder 313.3M frozen, head 100.8M trainable, all clips aligned,
init loss 3.72 (≈ln33), overfit 3.70 → 1.38. ✅

## Key facts
- data2vec-aqc ckpt: `/speech/tomson/exps/speech-recog/models/data2vec-aqc/SPRING_INX_data2vec_aqc_SSL.pt` (1024-d, 50 fps, same 320× conv stack as HuBERT).
- Input normalization: `Wav2Vec2FeatureExtractor(do_normalize=True)` — per-utterance zero-mean/unit-var, numerically == fairseq `normalize=true` / `F.layer_norm(x, x.shape)`.
- The reused `filler_asr` code is imported live via `FILLER_ROOT` (default `/speech/tomson/filler_asr`); the data2vec loader via `SLAM_SRC` (default `/speech/tomson/SMEAR-MoE-ASR/src`).

## Next steps (not done yet)
- **Full training**: port `filler_asr/train_sa_pe_lm_head/train.py` to call `build_d2v_acmlm`
  (it already references `raw_model.hubert` and min-crops logits/labels, so it's a small swap).
- **Hindi**: build a Devanagari char `vocab.json` (same normalization as the collator) and point
  `--vocab_dir` / manifests at the Hindi data. WER scoring must strip punctuation via
  `unicodedata` category-P, **not** `\w` (which shatters Devanagari).

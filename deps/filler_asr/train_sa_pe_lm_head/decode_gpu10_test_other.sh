#!/usr/bin/env bash
# Iterative decode (steps=32) of the fps50 v2 best.pt on GPU 10:
#   librispeech test_other.
# Params match the test_clean iter_steps32_best run: steps 32, tau 0.1, temp 1.0,
# fill_penalty 0.0, no remask, fps auto-read from ckpt (=50).
set -euo pipefail

PY=/speech/tomson/miniconda3/envs/filler_asr/bin/python
CODE=/speech/tomson/filler_asr/train_sa_pe_lm_head
RUN=$CODE/runs/full960_sa8_ACMLM_pe_fps50_mask20-30x10_eos3_fw0.03_exp_150ep_tapcache_v2
export CUDA_VISIBLE_DEVICES=0
cd "$CODE"

MAN=/speech/tomson/exps/speech-recog/data/librispeech/librispeech_test_other.jsonl
OUT=$RUN/iter_steps32_best_test_other
mkdir -p "$OUT"

echo "==== decoding $MAN -> $OUT ($(date)) ===="
"$PY" decode_iterative.py \
  --run_dir "$RUN" \
  --manifest "$MAN" \
  --out_dir "$OUT" \
  --steps 32 --tau 0.1 --temp 1.0 --fill_penalty 0.0 \
  2>&1 | tee "$OUT/decode.log"

echo "==== GPU10 test_other DECODE DONE ($(date)) ===="

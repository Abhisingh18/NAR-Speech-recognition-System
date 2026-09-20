#!/usr/bin/env bash
# Iterative decode (steps=32) of the fps50 v2 best.pt on GPU 0:
#   l2arctic_test, seedtts_en_pairs, seedtts_en_prompts  (run sequentially)
# Params match the test_clean iter_steps32_best run: steps 32, tau 0.1, temp 1.0,
# fill_penalty 0.0, no remask, fps auto-read from ckpt (=50).
set -euo pipefail

PY=/speech/tomson/miniconda3/envs/filler_asr/bin/python
CODE=/speech/tomson/filler_asr/train_sa_pe_lm_head
RUN=$CODE/runs/full960_sa8_ACMLM_pe_fps50_mask20-30x10_eos3_fw0.03_exp_150ep_tapcache_v2
export CUDA_VISIBLE_DEVICES=0
cd "$CODE"

decode () {  # $1=manifest  $2=out_subdir
  local man="$1" out="$RUN/$2"
  mkdir -p "$out"
  echo "==== decoding $man -> $out ($(date)) ===="
  "$PY" decode_iterative.py \
    --run_dir "$RUN" \
    --manifest "$man" \
    --out_dir "$out" \
    --steps 32 --tau 0.1 --temp 1.0 --fill_penalty 0.0 \
    2>&1 | tee "$out/decode.log"
}

decode /speech/tomson/filler_asr/data/testsets/l2arctic_data/l2arctic_test.jsonl   iter_steps32_best_l2arctic
decode /speech/tomson/filler_asr/data/testsets/en/seedtts_en_pairs.jsonl           iter_steps32_best_seedtts_pairs
decode /speech/tomson/filler_asr/data/testsets/en/seedtts_en_prompts.jsonl         iter_steps32_best_seedtts_prompts

echo "==== ALL GPU0 DECODES DONE ($(date)) ===="

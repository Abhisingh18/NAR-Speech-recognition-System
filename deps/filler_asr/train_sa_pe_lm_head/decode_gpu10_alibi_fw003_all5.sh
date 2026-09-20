#!/usr/bin/env bash
# Iterative decode (steps=32, NO REMASK) of the ALiBi fw0.03 run's best.pt on GPU 10.
# best.pt = argmin eval/iter_WER (step 75000, ep69; best_val iter_WER=0.1066).
# Order: test_clean, test_other, then the other 3 (l2arctic, seedtts_pairs, seedtts_prompts).
# Params match the sibling fps50 v2 decodes: steps 32, tau 0.1, temp 1.0, fill_penalty 0.0.
set -euo pipefail

PY=/speech/tomson/miniconda3/envs/filler_asr/bin/python
CODE=/speech/tomson/filler_asr/train_sa_pe_lm_head
RUN=$CODE/runs/full960_sa8_ACMLM_alibi_fps50_mask20-30x10_eos3_fw0.03_exp_150ep
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=10
export PYTHONUNBUFFERED=1
cd "$CODE"

decode () {  # $1=manifest  $2=out_subdir
  local man="$1" out="$RUN/$2"
  mkdir -p "$out"
  echo "==== decoding $man -> $out ($(date)) ===="
  "$PY" decode_iterative.py \
    --run_dir "$RUN" \
    --ckpt "$RUN/best.pt" \
    --manifest "$man" \
    --out_dir "$out" \
    --steps 32 --tau 0.1 --temp 1.0 --fill_penalty 0.0 \
    2>&1 | tee "$out/decode.log"
}

LS=/speech/tomson/exps/speech-recog/data/librispeech
TS=/speech/tomson/filler_asr/data/testsets

decode $LS/librispeech_test_clean.jsonl             iter_steps32_best_test_clean
decode $LS/librispeech_test_other.jsonl             iter_steps32_best_test_other
decode $TS/l2arctic_data/l2arctic_test.jsonl        iter_steps32_best_l2arctic
decode $TS/en/seedtts_en_pairs.jsonl                iter_steps32_best_seedtts_pairs
decode $TS/en/seedtts_en_prompts.jsonl              iter_steps32_best_seedtts_prompts

echo "==== ALL GPU10 alibi-fw0.03 DECODES DONE ($(date)) ===="

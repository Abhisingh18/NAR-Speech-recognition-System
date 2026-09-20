#!/usr/bin/env bash
# =============================================================================
# Decode the ALiBi fw0.2 run's 150-ep FINAL checkpoint (decode_final_ep150.pt,
# step 164700) on all 5 testsets, using the SAME harness/params as the fw0.03
# all-5 decode (decode_iterative.py, steps 32, tau 0.1, temp 1.0, fill_penalty 0.0).
# This makes the fw0.2-vs-fw0.03 comparison apples-to-apples across every set.
# (test-clean was 6.74% in the milestone report; re-decoded here for harness parity.)
# =============================================================================
set -euo pipefail

PY=/speech/tomson/miniconda3/envs/filler_asr/bin/python
CODE=/speech/tomson/filler_asr/train_sa_pe_lm_head
RUN=$CODE/runs/full960_sa8_ACMLM_alibi_fps50_mask20-30x10_eos3_fw0.2_exp_150ep
CKPT=$RUN/decode_final_ep150.pt
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="${GPU:-6}"
export PYTHONUNBUFFERED=1
cd "$CODE"

decode () {  # $1=manifest  $2=out_subdir
  local man="$1" out="$RUN/$2"
  mkdir -p "$out"
  echo "==== decoding $man -> $out ($(date)) ===="
  "$PY" decode_iterative.py \
    --run_dir "$RUN" \
    --ckpt "$CKPT" \
    --manifest "$man" \
    --out_dir "$out" \
    --steps 32 --tau 0.1 --temp 1.0 --fill_penalty 0.0 \
    2>&1 | tee "$out/decode.log"
}

LS=/speech/tomson/exps/speech-recog/data/librispeech
TS=/speech/tomson/filler_asr/data/testsets

decode $LS/librispeech_test_clean.jsonl             iter_steps32_final_test_clean
decode $LS/librispeech_test_other.jsonl             iter_steps32_final_test_other
decode $TS/l2arctic_data/l2arctic_test.jsonl        iter_steps32_final_l2arctic
decode $TS/en/seedtts_en_pairs.jsonl                iter_steps32_final_seedtts_pairs
decode $TS/en/seedtts_en_prompts.jsonl              iter_steps32_final_seedtts_prompts

echo "==== ALL fw0.2 FINAL(150ep) DECODES DONE ($(date)) ===="

#!/usr/bin/env bash
# 32-step iterative decode (NO remask, defaults) of the ALiBi fw0.03 150ep run.
#   snap_latest_for_decode.pt = step 130000 / ep119
#   snap_best_for_decode.pt   = step 110000 / ep101 (dev iter_WER 0.1025)
# Params identical to the 2026-09-08 best.pt all5 decode: steps 32, tau 0.1, temp 1.0, fill_penalty 0.0, seed 0.
set -euo pipefail
PY=/speech/tomson/miniconda3/envs/filler_asr/bin/python
CODE=/speech/tomson/filler_asr/train_sa_pe_lm_head
RUN=$CODE/runs/full960_sa8_ACMLM_alibi_fps50_mask20-30x10_eos3_fw0.03_exp_150ep
export CUDA_DEVICE_ORDER=PCI_BUS_ID PYTHONUNBUFFERED=1
cd "$CODE"

GPU="$1"; CKPT="$2"; TAG="$3"; shift 3
export CUDA_VISIBLE_DEVICES="$GPU"

LS=/speech/tomson/exps/speech-recog/data/librispeech
TS=/speech/tomson/filler_asr/data/testsets
declare -A MAN=(
  [test_clean]=$LS/librispeech_test_clean.jsonl
  [test_other]=$LS/librispeech_test_other.jsonl
  [l2arctic]=$TS/l2arctic_data/l2arctic_test.jsonl
  [seedtts_pairs]=$TS/en/seedtts_en_pairs.jsonl
  [seedtts_prompts]=$TS/en/seedtts_en_prompts.jsonl
)

for name in "$@"; do
  out="$RUN/iter32_${TAG}_${name}"
  mkdir -p "$out"
  echo "==== [gpu$GPU][$TAG] $name -> $out ($(date)) ===="
  "$PY" decode_iterative.py \
    --run_dir "$RUN" --ckpt "$RUN/$CKPT" \
    --manifest "${MAN[$name]}" --out_dir "$out" \
    --steps 32 --tau 0.1 --temp 1.0 --fill_penalty 0.0 --seed 0 \
    2>&1 | tee "$out/decode.log"
done
echo "==== [gpu$GPU][$TAG] DONE ($(date)) ===="

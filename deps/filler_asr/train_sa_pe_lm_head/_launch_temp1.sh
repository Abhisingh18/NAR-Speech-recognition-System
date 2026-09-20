#!/usr/bin/env bash
set -u
cd /speech/tomson/filler_asr/train_sa_pe_lm_head
source /speech/tomson/miniconda3/etc/profile.d/conda.sh && conda activate filler_asr

RUN=runs/full960_sa8_ACMLM_pe_fps50_mask20-30x10_eos3_fw0.03_exp_150ep_tapcache_v2
CKPT=$RUN/frozen_step56000.pt
MAN=/speech/tomson/exps/speech-recog/data/librispeech/librispeech_test_clean.jsonl

run_one () {  # $1=gpu  $2=steps
  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="$1" PYTHONUNBUFFERED=1 \
    python -u decode_iterative.py --run_dir "$RUN" --ckpt "$CKPT" --manifest "$MAN" \
      --out_dir "$RUN/iter_steps$2_temp1" --steps "$2" --fps 0 --n 0 --tau 0.1 --temp 1 --seed 0
}

run_one 0  32 > "$RUN/iter_gpu0_steps32_temp1.log"  2>&1 &  echo "gpu0  steps 32 temp1  pid=$!"
run_one 10 64 > "$RUN/iter_gpu10_steps64_temp1.log" 2>&1 &  echo "gpu10 steps 64 temp1  pid=$!"

wait
echo "ALL DONE"

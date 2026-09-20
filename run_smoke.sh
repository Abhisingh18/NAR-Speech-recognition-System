#!/usr/bin/env bash
# Smoke the data2vec-aqc A-CMLM model on a few LibriSpeech clips.
#   GPU=6 bash run_smoke.sh            # pick a free GPU (nvidia-smi)
#   GPU=6 N=8 bash run_smoke.sh
set -euo pipefail
cd "$(dirname "$0")"

GPU="${GPU:-6}"
N="${N:-6}"
ENV="${ENV:-indic-nar-filler_asr}"
MANIFEST="${MANIFEST:-/speech/tomson/exps/speech-recog/data/librispeech/librispeech_dev_clean.jsonl}"
VOCAB_DIR="${VOCAB_DIR:-/speech/tomson/filler_asr/train_sa_pe_lm_head/runs/full960_sa8_pe_fps25_lin_100ep_mask20-30x10_eos3_fw0.1}"

used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$GPU" 2>/dev/null | tr -d ' ')
[ -n "$used" ] || { echo "GPU $GPU not found"; exit 1; }
[ "$used" -gt 2000 ] && echo "WARN: GPU $GPU has ${used}MiB used (still trying)."

CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="$GPU" PYTHONUNBUFFERED=1 \
  conda run --no-capture-output -n "$ENV" \
  python smoke.py --n "$N" --device cuda --manifest "$MANIFEST" --vocab_dir "$VOCAB_DIR"

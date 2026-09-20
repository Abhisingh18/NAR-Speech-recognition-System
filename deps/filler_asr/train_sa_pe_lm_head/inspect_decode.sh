#!/usr/bin/env bash
# Stage-by-stage decode inspection for a trained SA+PE+lm_head filler head.
#   RUN=full960_sa8_pe_fps25_lin GPU=1 N=4 bash inspect_decode.sh
#   RUN=full960_sa6_pe_fps25_lin GPU=1 N=6 MAX_SEC=6 bash inspect_decode.sh
#   RUN=full960_sa4_pe_fps25_lin GPU=1 MANIFEST=.../librispeech_test_other.jsonl bash inspect_decode.sh
# GPU = single nvidia-smi index (PCI_BUS_ID order). GPU 1 free; GPU 0 flaky; never the busy training cards.
set -euo pipefail
cd "$(dirname "$0")"

RUN="${RUN:?set RUN=<run dir name under runs/ or absolute path>}"
GPU="${GPU:-1}"
MANIFEST="${MANIFEST:-/speech/tomson/exps/speech-recog/data/librispeech/librispeech_test_clean.jsonl}"
CKPT_NAME="${CKPT:-best.pt}"
N="${N:-4}"
MAX_SEC="${MAX_SEC:-8.0}"
SHOW_FRAMES="${SHOW_FRAMES:-0}"     # 0 = full frame dump; else truncate to first N tokens

case "$RUN" in
  /*) RUN_DIR="$RUN" ;;
  *)  RUN_DIR="/speech/tomson/filler_asr/train_sa_pe_lm_head/runs/$RUN" ;;
esac
[ -f "$RUN_DIR/$CKPT_NAME" ] || { echo "no checkpoint at $RUN_DIR/$CKPT_NAME" >&2; exit 1; }

echo ">> RUN_DIR=$RUN_DIR GPU=$GPU set=$(basename "$MANIFEST" .jsonl) ckpt=$CKPT_NAME n=$N max_sec=$MAX_SEC"
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="$GPU" PYTHONUNBUFFERED=1 \
  conda run --no-capture-output -n filler_asr \
  python -u inspect_decode.py \
    --run_dir "$RUN_DIR" --ckpt "$RUN_DIR/$CKPT_NAME" \
    --manifest "$MANIFEST" --n "$N" --max_sec "$MAX_SEC" --show_frames "$SHOW_FRAMES"

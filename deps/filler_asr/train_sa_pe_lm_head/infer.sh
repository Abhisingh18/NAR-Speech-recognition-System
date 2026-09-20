#!/usr/bin/env bash
# Decode a trained SA+PE+lm_head filler head on a frozen HuBERT-xlarge.
# Architecture (num_sa_layers/pe/encoder) is auto-recovered from the checkpoint, so this
# one wrapper drives every depth (sa2/sa4/sa6/sa8).
#
#   RUN=full960_sa8_pe_fps25_lin GPU=1 bash infer.sh
#   RUN=full960_sa4_pe_fps25_lin GPU=1 MANIFEST=.../librispeech_test_other.jsonl bash infer.sh
#   RUN=full960_sa6_pe_fps25_lin GPU=1 NMAX=50 bash infer.sh        # quick 50-utt smoke
#
# GPU is a single nvidia-smi index (PCI_BUS_ID order). GPU 1 is the free card on this box;
# GPU 0 is the flaky illegal-instruction one — avoid. Never touch the busy training GPUs.
set -euo pipefail
cd "$(dirname "$0")"

RUN="${RUN:?set RUN=<run dir name under runs/ or an absolute path>}"
GPU="${GPU:-1}"
MANIFEST="${MANIFEST:-/speech/tomson/exps/speech-recog/data/librispeech/librispeech_test_clean.jsonl}"
CKPT_NAME="${CKPT:-best.pt}"
BATCH="${BATCH:-16}"
NMAX="${NMAX:-0}"                                # 0 = full set

case "$RUN" in
  /*) RUN_DIR="$RUN" ;;
  *)  RUN_DIR="/speech/tomson/filler_asr/train_sa_pe_lm_head/runs/$RUN" ;;
esac
[ -f "$RUN_DIR/$CKPT_NAME" ] || { echo "no checkpoint at $RUN_DIR/$CKPT_NAME" >&2; exit 1; }
TAG=$(basename "$MANIFEST" .jsonl)

echo ">> RUN_DIR=$RUN_DIR  GPU=$GPU  set=$TAG  ckpt=$CKPT_NAME  batch=$BATCH  nmax=$NMAX"
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="$GPU" PYTHONUNBUFFERED=1 \
  conda run --no-capture-output -n filler_asr \
  python -u infer.py \
    --run_dir "$RUN_DIR" --ckpt "$RUN_DIR/$CKPT_NAME" \
    --manifest "$MANIFEST" --batch_size "$BATCH" --dev_n "$NMAX" \
    --out "$RUN_DIR/infer_${TAG}.jsonl"

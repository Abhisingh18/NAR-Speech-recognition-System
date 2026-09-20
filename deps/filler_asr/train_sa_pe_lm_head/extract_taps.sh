#!/usr/bin/env bash
# =============================================================================
# Precompute the frozen HuBERT tap cache (train + dev) on 4 GPUs via torchrun DDP.
# The 4 ranks write DISJOINT clip ranges of ONE shared ragged memmap per split
# (precompute_taps.py). One-time (~1h); reusable across ALL later A-CMLM sweeps.
#
#   full run  : bash extract_taps.sh
#   smoke     : CACHE=/speech/tomson/filler_asr/data/tapcache_smoke LIMIT=512 bash extract_taps.sh
#   resume    : re-run -- a split with a PREPARED marker is skipped.
# =============================================================================
set -euo pipefail
cd "$(dirname "$0")"
export PYTHONPATH="/speech/tomson/filler_asr${PYTHONPATH:+:$PYTHONPATH}"

GPUS="${GPUS:-1,2,3,4}"                                 # 4 free GPUs (PCI-BUS ids)
CACHE="${CACHE:-/speech/tomson/filler_asr/data/tapcache}"
TRAIN_MANIFEST="${TRAIN_MANIFEST:-/speech/tomson/exps/speech-recog/data/librispeech/librispeech_train_960h.jsonl}"
DEV_MANIFEST="${DEV_MANIFEST:-/speech/tomson/exps/speech-recog/data/librispeech/librispeech_dev_clean.jsonl}"
BATCH="${BATCH:-8}"                                     # encoder-inference batch per GPU
LIMIT="${LIMIT:-0}"                                     # 0 = all clips; else first N (smoke)
MASTER_PORT="${MASTER_PORT:-29581}"
NG=$(awk -F, '{print NF}' <<<"$GPUS")
mkdir -p logs "$CACHE"

command -v conda >/dev/null || { echo "ERROR: conda not on PATH -> conda activate filler_asr first"; exit 1; }

extract () {   # $1 split  $2 manifest  $3 master_port
  local split=$1 manifest=$2 port=$3 outdir="$CACHE/$1"
  if [ -f "$outdir/PREPARED" ]; then echo "[$split] already PREPARED ($(cat "$outdir/PREPARED")) -> skip"; return; fi
  mkdir -p "$outdir"
  echo "== extract $split -> $outdir   (GPUs=$GPUS, per-GPU batch=$BATCH, limit=$LIMIT) =="
  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="$GPUS" NCCL_SOCKET_IFNAME=lo PYTHONUNBUFFERED=1 \
    conda run --no-capture-output -n filler_asr \
    torchrun --nproc_per_node="$NG" --master_port="$port" precompute_taps.py \
      --manifest "$manifest" --cache_dir "$outdir" --batch_size "$BATCH" --limit "$LIMIT"
  [ -f "$outdir/PREPARED" ] && echo "[$split] DONE -> $(cat "$outdir/PREPARED")" || { echo "[$split] FAILED (no PREPARED)"; exit 1; }
}

extract dev   "$DEV_MANIFEST"   "$MASTER_PORT"
extract train "$TRAIN_MANIFEST" "$((MASTER_PORT + 1))"
echo "ALL EXTRACTION DONE -> $CACHE   (train/ + dev/)"

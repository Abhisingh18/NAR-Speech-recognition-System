#!/usr/bin/env bash
# =============================================================================
# Precompute the FROZEN data2vec-aqc tap cache (train + dev) once. Reusable across ALL later
# sweeps (only invalidated if the encoder .pt changes). Then train with TAP_CACHE=<dir> bash run.sh.
#
#   full (4 GPU) : GPUS=1,2,3,4 bash extract_taps_d2v.sh
#   smoke (CPU)  : DEVICE=cpu LIMIT=8 CACHE=.../tapcache_smoke bash extract_taps_d2v.sh
#   resume       : a split with a PREPARED marker is skipped.
# =============================================================================
set -euo pipefail
cd "$(dirname "$0")"
FILLER_ROOT="${FILLER_ROOT:-/speech/tomson/filler_asr}"
SLAM_SRC="${SLAM_SRC:-/speech/tomson/SMEAR-MoE-ASR/src}"
export FILLER_ROOT SLAM_SRC
export PYTHONPATH="${FILLER_ROOT}:${FILLER_ROOT}/train_sa_pe_lm_head:${SLAM_SRC}:$(pwd)${PYTHONPATH:+:$PYTHONPATH}"
export TOKENIZERS_PARALLELISM=false

ENV="${ENV:-indic-nar-filler_asr}"
GPUS="${GPUS:-1,2,3,4}"
DEVICE="${DEVICE:-}"                       # set DEVICE=cpu to run without GPUs (smoke)
CACHE="${CACHE:-/speech/tomson/indic-nar-filler_asr/data/tapcache_d2v_hi}"
# MUST match run.sh's ENCODER_PT: the cached tap is only valid for the exact encoder it was built
# from. Keep this default identical to run.sh (SPRING-INX SSL). Override BOTH together to switch.
ENCODER_PT="${ENCODER_PT:-/speech/tomson/exps/speech-recog/models/data2vec-aqc/SPRING_INX_data2vec_aqc_SSL.pt}"
VOCAB_DIR="${VOCAB_DIR:-/speech/tomson/indic-nar-filler_asr/vocab_hi}"
TRAIN_MANIFEST="${TRAIN_MANIFEST:-/speech/tomson/exps/speech-recog/data/smear-more-hi-ta-te-ma/train_hi.jsonl}"
DEV_MANIFEST="${DEV_MANIFEST:-/speech/tomson/exps/speech-recog/data/smear-more-hi-ta-te-ma/val_hi.jsonl}"
BATCH="${BATCH:-8}"; LIMIT="${LIMIT:-0}"; MASTER_PORT="${MASTER_PORT:-29601}"
mkdir -p logs "$CACHE"

extract () {   # $1 split  $2 manifest  $3 port
  local split=$1 manifest=$2 port=$3 outdir="$CACHE/$1"
  if [ -f "$outdir/PREPARED" ]; then echo "[$split] already PREPARED -> skip"; return; fi
  mkdir -p "$outdir"
  echo "== extract $split -> $outdir  (device=${DEVICE:-cuda:$GPUS}, batch=$BATCH, limit=$LIMIT) =="
  local common=(--manifest "$manifest" --cache_dir "$outdir" --encoder_pt "$ENCODER_PT" \
                --vocab_dir "$VOCAB_DIR" --batch_size "$BATCH" --limit "$LIMIT")
  if [ "$DEVICE" = "cpu" ]; then
    conda run --no-capture-output -n "$ENV" python precompute_taps_d2v.py "${common[@]}" --device cpu
  else
    local NG; NG=$(awk -F, '{print NF}' <<<"$GPUS")
    CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="$GPUS" NCCL_SOCKET_IFNAME=lo PYTHONUNBUFFERED=1 \
      conda run --no-capture-output -n "$ENV" \
      torchrun --nproc_per_node="$NG" --master_port="$port" precompute_taps_d2v.py "${common[@]}"
  fi
  [ -f "$outdir/PREPARED" ] && echo "[$split] DONE -> $(cat "$outdir/PREPARED")" || { echo "[$split] FAILED"; exit 1; }
}

extract dev   "$DEV_MANIFEST"   "$MASTER_PORT"
extract train "$TRAIN_MANIFEST" "$((MASTER_PORT + 1))"
echo "ALL DONE -> $CACHE   (then: TAP_CACHE=$CACHE GPU=6,7,8 bash launch_real.sh)"

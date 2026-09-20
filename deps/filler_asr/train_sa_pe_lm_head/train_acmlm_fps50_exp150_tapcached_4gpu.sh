#!/usr/bin/env bash
# =============================================================================
# A-CMLM SA-8 + PE + lm_head, LS-960 -- 4-GPU DDP, CACHED-TAP, DETACHED (setsid).
#
# Same recipe as train_acmlm_fps50_exp150_nohup_4gpu.sh, but trains straight from the
# PRECOMPUTED HuBERT tap cache (extract_taps.sh / precompute_taps.py) -> NO encoder forward
# every step (~3x faster). Uses length-bucketed batches (no padding waste). Carries all the
# reviewed fixes (train.py / filler_sa_reference.py). Verified: cached tap corr 0.99995 vs live.
#
#   prereq : bash extract_taps.sh          (builds $TAP_CACHE/{train,dev} once, ~45 min)
#   run    : bash train_acmlm_fps50_exp150_tapcached_4gpu.sh
#   watch  : tail -f <printed log>
#   resume : RESUME="$OUT/latest.pt" bash train_acmlm_fps50_exp150_tapcached_4gpu.sh
#   stop   : kill -- -<printed pid>
# =============================================================================
set -euo pipefail
cd "$(dirname "$0")"; RUN_DIR="$(pwd)"

# ============================ EDIT VALUES HERE ===============================
GPUS="${GPUS:-1,2,3,4}"
TAP_CACHE="${TAP_CACHE:-/speech/tomson/filler_asr/data/tapcache}"   # holds train/ + dev/
EPOCHS="${EPOCHS:-150}"
BATCH="${BATCH:-64}"               # per-GPU (cache frees encoder memory -> larger batch is fine)
ACCUM="${ACCUM:-1}"                # eff batch = BATCH*ACCUM*nGPU = 64*1*4 = 256 (== the audio run)
MASK_PROB="${MASK_PROB:-0.20}"; MASK_PROB_MAX="${MASK_PROB_MAX:-0.30}"; MASK_LEN="${MASK_LEN:-10}"
EOS_REPEAT="${EOS_REPEAT:-3}"
FILL_WEIGHT="${FILL_WEIGHT:-0.03}"
FPS="${FPS:-50}"
LR="${LR:-2e-4}"
WARMUP="${WARMUP:-5000}"
LR_SCHEDULE="${LR_SCHEDULE:-exponential}"; LR_EXP_FINAL_RATIO="${LR_EXP_FINAL_RATIO:-0.05}"
CLIP="${CLIP:-1.0}"
SA_LAYERS="${SA_LAYERS:-8}"
MASTER_PORT="${MASTER_PORT:-29576}"
OUT="${OUT:-/speech/tomson/filler_asr/train_sa_pe_lm_head/runs/full960_sa8_ACMLM_pe_fps50_mask20-30x10_eos3_fw0.03_exp_150ep_tapcache}"
WANDB="${WANDB:-1}"
DEBUG_STEPS="${DEBUG_STEPS:-5}"
RESUME="${RESUME:-}"
WANDB_RUN_ID="${WANDB_RUN_ID:-}"; WANDB_RESUME="${WANDB_RESUME:-}"
FORCE="${FORCE:-0}"
# =============================================================================

command -v conda >/dev/null || { echo "ERROR: conda not on PATH -> conda activate filler_asr first"; exit 1; }
[ -f "$TAP_CACHE/train/PREPARED" ] && [ -f "$TAP_CACHE/dev/PREPARED" ] || {
  echo "ERROR: tap cache not ready at $TAP_CACHE (need train/PREPARED + dev/PREPARED). Run extract_taps.sh first."; exit 1; }
STAMP="$(date +%Y%m%d_%H%M%S)"
LOG="$RUN_DIR/logs/train_acmlm_fps50_exp150_tapcache_${STAMP}.log"
PIDF="$RUN_DIR/logs/train_acmlm_fps50_exp150_tapcache.pid"
mkdir -p "$RUN_DIR/logs"

if [ "$FORCE" != "1" ]; then
  busy=""
  for g in ${GPUS//,/ }; do
    used=$(CUDA_DEVICE_ORDER=PCI_BUS_ID nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$g" 2>/dev/null | tr -d ' ')
    [ -z "$used" ] && { echo "ERROR: GPU $g not found"; exit 1; }
    [ "$used" -gt 1000 ] && busy="$busy $g(${used}MiB)"
  done
  [ -n "$busy" ] && { echo "ABORT: busy GPUs:$busy (GPUS=.. or FORCE=1)"; exit 1; }
fi
if [ -z "$RESUME" ] && [ "$FORCE" != "1" ] && ls "$OUT"/*.pt >/dev/null 2>&1; then
  echo "ABORT: $OUT already has checkpoints. set OUT=.. , RESUME=$OUT/latest.pt , or FORCE=1."; exit 1
fi
mkdir -p "$OUT"
[ -n "$WANDB_RUN_ID" ] && export WANDB_RUN_ID
[ -n "$WANDB_RESUME" ] && export WANDB_RESUME
NG=$(awk -F, '{print NF}' <<<"$GPUS")

setsid env \
  ACMLM=1 GPU="$GPUS" SA_LAYERS=$SA_LAYERS EPOCHS=$EPOCHS BATCH=$BATCH ACCUM=$ACCUM \
  MASK_PROB=$MASK_PROB MASK_PROB_MAX=$MASK_PROB_MAX MASK_LEN=$MASK_LEN \
  EOS_REPEAT=$EOS_REPEAT FILL_WEIGHT=$FILL_WEIGHT \
  FPS=$FPS LR=$LR WARMUP=$WARMUP CLIP=$CLIP \
  LR_SCHEDULE=$LR_SCHEDULE LR_EXP_FINAL_RATIO=$LR_EXP_FINAL_RATIO \
  DEBUG_STEPS=$DEBUG_STEPS WANDB=$WANDB RESUME="$RESUME" TAP_CACHE="$TAP_CACHE" \
  OUT="$OUT" MASTER_PORT=$MASTER_PORT \
  bash "$RUN_DIR/run.sh" > "$LOG" 2>&1 < /dev/null &
PID=$!; echo "$PID" > "$PIDF"

echo "launched CACHED A-CMLM fps50/exp/${EPOCHS}ep (setsid, detached)"
echo "  GPUs   : $GPUS   eff batch = $BATCH*$ACCUM*$NG = $((BATCH*ACCUM*NG))   tap_cache=$TAP_CACHE"
echo "  out    : $OUT"
echo "  log    : $LOG"
echo "  pid    : $PID   ($PIDF)"
echo "  watch  : tail -f $LOG"
echo "  stop   : kill -- -$PID"

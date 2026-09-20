#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Grad-norm smoke for the A-CMLM fps50/fw0.03/exp run  (SINGLE GPU 0).
#
# Purpose: watch train/grad_norm vs the clip threshold (clip=1.0) to see whether
# the norm sits PINNED at the clip for a long stretch (clip binding) or falls
# below 1.0 after the initial transient (LR schedule in control). See autopsy note.
#
# Why these numbers:
#   * eff batch kept at 256 == the real run (BATCH 16 * ACCUM 16 * 1 GPU), because
#     grad-norm magnitude/variance depends on effective batch. "less batch, more
#     accum" vs the real 32*2 -> same eff batch, lower per-GPU memory.
#   * every optimizer step logged (LOG_STEPS=1) -> full gnorm trace captured.
#   * WARMUP shortened so LR reaches peak fast and we observe the peak-LR regime
#     (where the real run spends ~97% of its life) within a short smoke.
#   * eval/save pushed past MAX_STEPS -> no checkpoints written, no eval stalls.
#   * mask 20-30x10, eos3, fw0.03, fps50, exp schedule, lr 2e-4, clip 1.0 MIRROR
#     train_acmlm_..._fps50_exp_150ep_4gpu.sh so the trace is representative.
#
# Run (detached; analyzer works on the partial log, so you can kill it early):
#   cd /speech/tomson/filler_asr/train_sa_pe_lm_head/smoke
#   bash smoke_gradnorm.sh
#
# Overridable: GPU BATCH ACCUM WARMUP MAX_STEPS CLIP FORCE
#   longer trace     : MAX_STEPS=300 bash smoke_gradnorm.sh
#   test looser clip : CLIP=5.0     bash smoke_gradnorm.sh
# ---------------------------------------------------------------------------
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
TRAIN_DIR="$(dirname "$HERE")"          # .../train_sa_pe_lm_head  (holds run.sh)

# NOTE: GPU 0 has a history of CUDA "illegal instruction" crashes on rank 0
# (memory: gpu0-illegal-instruction-crashes). If the smoke dies with that error
# it is the hardware, not this code -- rerun with e.g. GPU=1.
GPU="${GPU:-0}"
BATCH="${BATCH:-16}"                     # less batch ...
ACCUM="${ACCUM:-16}"                     # ... more accum -> 16*16*1 = eff batch 256 (== real run)
WARMUP="${WARMUP:-30}"                    # short: reach peak LR fast, then observe post-warmup gnorm
MAX_STEPS="${MAX_STEPS:-150}"             # optimizer steps to trace
CLIP="${CLIP:-1.0}"                       # clip threshold under study
FORCE="${FORCE:-0}"                       # 1 = launch even if the GPU looks busy

STAMP="$(date +%Y%m%d_%H%M%S)"
OUT="$HERE/_scratch_out"                  # scratch run dir (no ckpts: SAVE_STEPS >> MAX_STEPS)
LOG="$HERE/logs/gradnorm_${STAMP}.log"
mkdir -p "$HERE/logs" "$OUT"

# ---- preflight: is GPU $GPU actually free? (shared box; do not stomp others) ----
echo "== GPU preflight (need $GPU idle) =="
CUDA_DEVICE_ORDER=PCI_BUS_ID nvidia-smi \
  --query-gpu=index,name,memory.used,memory.total,utilization.gpu \
  --format=csv,noheader,nounits -i "$GPU"
used=$(CUDA_DEVICE_ORDER=PCI_BUS_ID nvidia-smi --query-gpu=memory.used \
  --format=csv,noheader,nounits -i "$GPU" | tr -d ' ')
if [ "${used:-0}" -gt 2000 ] && [ "$FORCE" != "1" ]; then
  echo "ABORT: GPU $GPU has ${used} MiB in use (someone may be running). FORCE=1 to override."
  exit 1
fi

echo "== launching grad-norm smoke (bf16, log every step) =="
echo "  GPU     : $GPU     eff batch = $BATCH*$ACCUM*1 = $((BATCH*ACCUM))"
echo "  steps   : $MAX_STEPS   warmup=$WARMUP   clip=$CLIP"
echo "  log     : $LOG"

# setsid: fully detach so it survives logout (nohup alone is not enough when a run
# re-arms SIGHUP -- memory: filler-asr-full960-resume). Single GPU -> run.sh takes
# the plain `python train.py` path (no torchrun / NCCL).
setsid env \
  ACMLM=1 \
  GPU="$GPU" \
  SA_LAYERS=8 \
  EPOCHS=999 \
  BATCH="$BATCH" ACCUM="$ACCUM" \
  MASK_PROB=0.20 MASK_PROB_MAX=0.30 MASK_LEN=10 \
  EOS_REPEAT=3 FILL_WEIGHT=0.03 \
  FPS=50 LR=2e-4 LR_SCHEDULE=exponential LR_EXP_FINAL_RATIO=0.01 \
  WARMUP="$WARMUP" CLIP="$CLIP" MAX_STEPS="$MAX_STEPS" \
  TRAIN_N=0 DEV_N=8 \
  LOG_STEPS=1 EVAL_STEPS=100000000 SAVE_STEPS=100000000 \
  WANDB=0 \
  OUT="$OUT" \
  bash "$TRAIN_DIR/run.sh" > "$LOG" 2>&1 < /dev/null &

PID=$!
echo "$PID" > "$HERE/logs/gradnorm.pid"
echo
echo "  pid     : $PID"
echo "  watch   : tail -f $LOG"
echo "  analyze : python $HERE/analyze_gradnorm.py $LOG   (safe on a partial log)"
echo "  stop    : kill -- -$PID    # kills the whole process group"

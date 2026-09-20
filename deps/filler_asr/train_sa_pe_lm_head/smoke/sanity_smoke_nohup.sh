#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# SANITY smoke (detached via setsid; survives logout, logs to file).
#
# Runs the REAL A-CMLM fps50 config for a few optimizer steps on ONE GPU to confirm
# the pipeline TRAINS end-to-end before committing 4 GPUs for weeks:
#   * setup audits print ([frozen-audit]/[ckpt-audit]/[select]/[data-audit])
#   * [eval @ 0] loss is FINITE (NaN gate: text_embed init is sound)
#   * a few finite training steps (loss falling, gnorm sane) + first [grad-audit] lines
#   * a final eval + best.pt/latest.pt save (checkpoint path works)
#
# This only checks that it RUNS. For the grad-norm-vs-clip study at the true eff batch
# (256), use smoke_gradnorm.sh instead.
#
#   run   : GPU=<free id> bash sanity_smoke_nohup.sh
#   watch : tail -f <printed log>
# Overridable: GPU BATCH ACCUM MAX_STEPS WARMUP FORCE
# ---------------------------------------------------------------------------
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"; TRAIN_DIR="$(dirname "$HERE")"   # .../train_sa_pe_lm_head (holds run.sh)

GPU="${GPU:-0}"                   # <-- one FREE GPU (PCI-BUS id). check: nvidia-smi
BATCH="${BATCH:-8}"
ACCUM="${ACCUM:-8}"               # eff batch = BATCH*ACCUM = 64 (sanity: small & fast; only checks it RUNS)
MAX_STEPS="${MAX_STEPS:-30}"
WARMUP="${WARMUP:-10}"
FORCE="${FORCE:-0}"               # 1 = co-tenant a busy GPU

command -v conda >/dev/null || { echo "ERROR: 'conda' not on PATH -> run 'conda activate filler_asr' (or source conda.sh) first"; exit 1; }
used=$(CUDA_DEVICE_ORDER=PCI_BUS_ID nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$GPU" 2>/dev/null | tr -d ' ')
[ -z "$used" ] && { echo "ERROR: GPU $GPU not found by nvidia-smi"; exit 1; }
if [ "${used:-0}" -gt 2000 ] && [ "$FORCE" != "1" ]; then
  echo "ABORT: GPU $GPU busy (${used}MiB). pick another (GPU=..) or FORCE=1 to co-tenant."; exit 1
fi

STAMP="$(date +%Y%m%d_%H%M%S)"
LOG="$HERE/logs/sanity_${STAMP}.log"; OUT="$HERE/_sanity_out"
mkdir -p "$HERE/logs"; rm -rf "$OUT"; mkdir -p "$OUT"

# setsid: new session -> the terminal's SIGHUP can't reach it (nohup alone is not enough
# for torchrun, which re-arms SIGHUP). Single GPU here -> run.sh takes the plain python path.
setsid env \
  ACMLM=1 GPU="$GPU" SA_LAYERS=8 EPOCHS=999 BATCH="$BATCH" ACCUM="$ACCUM" \
  MASK_PROB=0.20 MASK_PROB_MAX=0.30 MASK_LEN=10 EOS_REPEAT=3 FILL_WEIGHT=0.03 \
  FPS=50 LR=2e-4 WARMUP="$WARMUP" CLIP=1.0 LR_SCHEDULE=exponential LR_EXP_FINAL_RATIO=0.01 \
  MAX_STEPS="$MAX_STEPS" TRAIN_N=0 DEV_N=8 \
  LOG_STEPS=1 EVAL_STEPS=100000000 SAVE_STEPS=100000000 DEBUG_STEPS=5 WANDB=0 \
  OUT="$OUT" \
  bash "$TRAIN_DIR/run.sh" > "$LOG" 2>&1 < /dev/null &
PID=$!; echo "$PID" > "$HERE/logs/sanity.pid"

echo "sanity smoke launched (setsid) on GPU $GPU"
echo "  steps=$MAX_STEPS  eff_batch=$((BATCH*ACCUM))  pid=$PID"
echo "  log   : $LOG"
echo "  watch : tail -f $LOG"
echo "  PASS? : grep -E 'frozen-audit|grad-audit|eval @|DONE' $LOG   (finite loss, ends in DONE, no Traceback)"
echo "  stop  : kill -- -$PID     # kills the whole process group"
echo "  clean : rm -rf $OUT       # removes sanity checkpoints (~few GB)"

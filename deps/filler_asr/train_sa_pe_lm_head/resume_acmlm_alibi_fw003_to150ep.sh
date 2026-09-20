#!/usr/bin/env bash
# =============================================================================
# RESUME the ALiBi fw0.03 A-CMLM run that was SIGTERM-killed at step 130000 (ep119)
# on 2026-09-09, so the fw0.2-vs-fw0.03 ablation can finish at a matched 150 epochs.
#
# latest.pt: step=130000 epoch=119 best_val(iter_WER)=0.1025  opt+sched intact.
# Target   : step 164700 (150 ep) -> ~34.7k steps left (~25 epochs) on 4 GPUs.
#
# Resume is step-accurate (train.py --resume restores head+opt+sched+step+best_val
# and skips already-trained micro-batches; the exponential LR decay continues from
# where it left off). wandb curve wn4rfb26 is APPENDED (WANDB_RESUME=allow).
#
# Launched via run.sh DIRECTLY (the fw003 launcher does not forward RESUME/WANDB_*),
# under setsid so torchrun's re-armed SIGHUP can't take the job down on logout.
# =============================================================================
set -euo pipefail
cd "$(dirname "$0")"
RUN_DIR="$(pwd)"

GPUS="${GPUS:-1,2,3,4}"                 # 4 free full GPUs (avoid flaky gpu0, small gpu5, busy gpu10)
OUT="$RUN_DIR/runs/full960_sa8_ACMLM_alibi_fps50_mask20-30x10_eos3_fw0.03_exp_150ep"
RESUME="$OUT/latest.pt"
MASTER_PORT="${MASTER_PORT:-29590}"

[ -f "$RESUME" ] || { echo "ERROR: no $RESUME to resume from"; exit 1; }

# preflight: requested GPUs actually free
busy=""
for g in ${GPUS//,/ }; do
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$g" 2>/dev/null | tr -d ' ')
  [ -z "$used" ] && { echo "ERROR: GPU $g not found"; exit 1; }
  [ "$used" -gt 1000 ] && busy="$busy $g(${used}MiB)"
done
[ -n "$busy" ] && { echo "ABORT: requested GPUs busy:$busy"; exit 1; }

STAMP="$(date +%Y%m%d_%H%M%S)"
LOG="$RUN_DIR/logs/resume_acmlm_alibi_fw003_to150ep_${STAMP}.log"
PIDFILE="$OUT/run_resume.pid"

setsid env \
  ACMLM=1 GPU="$GPUS" SA_LAYERS=8 EPOCHS=150 BATCH=32 ACCUM=2 \
  MASK_PROB=0.20 MASK_PROB_MAX=0.30 MASK_LEN=10 \
  EOS_REPEAT=3 FILL_WEIGHT=0.03 \
  FPS=50 LR=2e-4 WARMUP=5000 LR_SCHEDULE=exponential LR_EXP_FINAL_RATIO=0.01 \
  POS_MODE=alibi EVAL_DECODE_STEPS=32 EVAL_DECODE_N=200 \
  SELECT_METRIC=iter_wer EVAL_STEPS=5000 SAVE_STEPS=5000 \
  TAP_CACHE=/speech/tomson/filler_asr/data/tapcache \
  VOCAB_DIR="$RUN_DIR/runs/full960_sa8_pe_fps25_lin_100ep_mask20-30x10_eos3_fw0.1" \
  OUT="$OUT" MASTER_PORT="$MASTER_PORT" \
  RESUME="$RESUME" WANDB=1 WANDB_RUN_ID=wn4rfb26 WANDB_RESUME=allow \
  bash run.sh > "$LOG" 2>&1 &
echo $! > "$PIDFILE"

echo "RESUMED fw0.03 -> 150ep  (pid=$(cat "$PIDFILE"))"
echo "  resume from : $RESUME  (step 130000 / ep119)"
echo "  gpus        : $GPUS   port $MASTER_PORT   wandb wn4rfb26 (append)"
echo "  log         : $LOG"

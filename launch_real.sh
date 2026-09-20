#!/usr/bin/env bash
# =============================================================================
# Launch the REAL 150-epoch A-CMLM run on a FROZEN data2vec-aqc-CTC (Hindi) encoder,
# detached (setsid: torchrun re-arms SIGHUP, so nohup alone would not survive logout).
# Recipe = filler_asr ALiBi / fw0.03 / fps50 / 8 SA / mask20-30x10 / eos3 / exp-LR / iter_wer.
#
#   GPU=6,7,8 bash launch_real.sh
#   stop     : kill -TERM -<PGID>   (PGID printed below / saved in $OUT/run.pgid)
# =============================================================================
set -euo pipefail
cd "$(dirname "$0")"

GPU="${GPU:-6,7,8}"
ENV="${ENV:-indic-nar-filler_asr}"
ENCODER_PT="${ENCODER_PT:-/speech/akshaya/fairseq_expxx/hindi/24kup_model/checkpoint_best.pt}"
OUT="${OUT:-/speech/tomson/indic-nar-filler_asr/runs/hi_d2vaqcCTC_sa8_ACMLM_alibi_fps50_fw003_exp150}"
MASTER_PORT="${MASTER_PORT:-29594}"
EVAL_STEPS="${EVAL_STEPS:-1000}"; SAVE_STEPS="${SAVE_STEPS:-1000}"   # Hindi ~225 steps/epoch -> denser than 5000
# batch 16 x accum 4 x 3 GPU = eff 192 (was 32x2: OOM'd -- one 28s clip pads the whole batch to T=1400).
BATCH="${BATCH:-16}"; ACCUM="${ACCUM:-4}"
FORCE="${FORCE:-0}"

# --- GPU-free guard (shared box: never grab a GPU someone else is using) ---
for g in ${GPU//,/ }; do
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$g" 2>/dev/null | tr -d ' ')
  [ -n "$used" ] || { echo "ERROR: GPU $g not found"; exit 1; }
  [ "$used" -gt 2000 ] && { echo "ABORT: GPU $g busy (${used}MiB). pick others via GPU=... or FORCE=1"; [ "$FORCE" = 1 ] || exit 1; }
done
# --- don't clobber an existing run ---
if [ "$FORCE" != 1 ] && ls "$OUT"/*.pt >/dev/null 2>&1; then
  echo "ABORT: $OUT already has checkpoints. set OUT=... or FORCE=1 (or RESUME via run.sh)."; exit 1
fi

mkdir -p logs "$OUT"
STAMP=$(date +%Y%m%d_%H%M%S)
LOG="logs/train_ctc_${STAMP}.log"
NG=$(awk -F, '{print NF}' <<< "$GPU")

setsid env GPU="$GPU" ENV="$ENV" ENCODER_PT="$ENCODER_PT" OUT="$OUT" MASTER_PORT="$MASTER_PORT" \
  EVAL_STEPS="$EVAL_STEPS" SAVE_STEPS="$SAVE_STEPS" BATCH="$BATCH" ACCUM="$ACCUM" WANDB=1 \
  bash run.sh > "$LOG" 2>&1 < /dev/null &
PID=$!
echo "$PID" > "$OUT/run.pid"
sleep 1; PGID=$(ps -o pgid= -p "$PID" 2>/dev/null | tr -d ' ' || echo "$PID")
echo "$PGID" > "$OUT/run.pgid"

echo "launched data2vec-aqc-CTC A-CMLM (Hindi)  pid=$PID  pgid=$PGID  gpus=$GPU (n=$NG, eff_batch=$((32*2*NG)))"
echo "  encoder : $ENCODER_PT"
echo "  out     : $OUT"
echo "  log     : $LOG"
echo "  watch   : tail -f $LOG"
echo "  stop    : kill -TERM -$PGID"

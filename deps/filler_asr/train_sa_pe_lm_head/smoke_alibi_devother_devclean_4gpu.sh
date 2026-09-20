#!/usr/bin/env bash
# =============================================================================
# SMOKE TEST for the ALiBi + fill_weight=0.2 + 32-step iterative-eval pipeline.
# Small + real: trains A-CMLM (AUDIO path, live frozen HuBERT — dev_other is NOT
# in the tap cache) on dev_other, validates on dev_clean, 10 epochs, warmup = 1
# epoch, wandb on, launched with nohup.  Purpose = prove the ALiBi code trains &
# decodes end-to-end (loss goes down, no NaN, iter WER prints), NOT to get SOTA.
#
#   run   : GPUS=1,2,3,4 bash smoke_alibi_devother_devclean_4gpu.sh
#   watch : tail -f <printed log>     |     wandb: project 'filler_asr' (name has _alibi)
#   stop  : kill $(cat <printed pidfile>)
# =============================================================================
set -euo pipefail
cd "$(dirname "$0")"
RUN_DIR="$(pwd)"

# ---- data (override run.sh defaults) -----------------------------------------
TRAIN_MANIFEST="${TRAIN_MANIFEST:-/speech/tomson/filler_asr/data/dev_other.jsonl}"   # training  = dev_other
DEV_MANIFEST="${DEV_MANIFEST:-/speech/tomson/filler_asr/data/dev_clean.jsonl}"       # validation = dev_clean

# ---- the three things under test ---------------------------------------------
POS_MODE="${POS_MODE:-alibi}"                       # <-- exercise symmetric ALiBi
FILL_WEIGHT="${FILL_WEIGHT:-0.2}"                   # <-- fill_weight 0.2
EVAL_DECODE_STEPS="${EVAL_DECODE_STEPS:-32}"        # <-- 32-step iterative mask-predict eval
EVAL_DECODE_N="${EVAL_DECODE_N:-30}"               # decode 30 dev clips/eval (keeps eval snappy)

# ---- run shape ---------------------------------------------------------------
GPUS="${GPUS:-1,2,3,4}"
SA_LAYERS="${SA_LAYERS:-8}"
EPOCHS="${EPOCHS:-10}"
BATCH="${BATCH:-8}"                                 # small: live HuBERT-xlarge fwd every step
ACCUM="${ACCUM:-1}"                                 # eff batch = 8*1*4 = 32
MASK_PROB="${MASK_PROB:-0.20}"; MASK_PROB_MAX="${MASK_PROB_MAX:-0.30}"; MASK_LEN="${MASK_LEN:-10}"
EOS_REPEAT="${EOS_REPEAT:-3}"
FPS="${FPS:-50}"
LR="${LR:-2e-4}"
LR_SCHEDULE="${LR_SCHEDULE:-exponential}"; LR_EXP_FINAL_RATIO="${LR_EXP_FINAL_RATIO:-0.01}"
DEV_N="${DEV_N:-200}"                               # cap one-shot eval to 200 dev_clean clips (speed)
MASTER_PORT="${MASTER_PORT:-29583}"
VOCAB_DIR="${VOCAB_DIR:-/speech/tomson/filler_asr/train_sa_pe_lm_head/runs/full960_sa8_pe_fps25_lin_100ep_mask20-30x10_eos3_fw0.1}"
OUT="${OUT:-/speech/tomson/filler_asr/train_sa_pe_lm_head/runs/smoke_alibi_devother_fw0.2_10ep}"
MIN_FREE_MIB="${MIN_FREE_MIB:-12000}"               # need ~12GB free/GPU for HuBERT-xlarge audio fwd
FORCE="${FORCE:-0}"

NG=$(awk -F, '{print NF}' <<<"$GPUS")

# ---- warmup = ~1 epoch of optimizer steps (matches train.py's steps_per_epoch) ----
N_TRAIN=$(wc -l < "$TRAIN_MANIFEST")
STEPS_PER_EPOCH=$(( (N_TRAIN / NG / BATCH) / ACCUM )); [ "$STEPS_PER_EPOCH" -lt 1 ] && STEPS_PER_EPOCH=1
WARMUP="${WARMUP:-$STEPS_PER_EPOCH}"                # 1 epoch of warmup
EVAL_STEPS="${EVAL_STEPS:-$STEPS_PER_EPOCH}"        # eval once per epoch
SAVE_STEPS="${SAVE_STEPS:-$STEPS_PER_EPOCH}"
LOG_STEPS="${LOG_STEPS:-10}"

STAMP="$(date +%Y%m%d_%H%M%S)"
mkdir -p "$RUN_DIR/logs" "$OUT"
LOG="$RUN_DIR/logs/smoke_alibi_devother_${STAMP}.log"
PIDFILE="$OUT/run.pid"

# ---- preflight: manifests + first wav exist ----------------------------------
for m in "$TRAIN_MANIFEST" "$DEV_MANIFEST"; do [ -s "$m" ] || { echo "ERROR: manifest missing/empty: $m"; exit 1; }; done
w=$(python -c "import json,sys;print(json.loads(open('$TRAIN_MANIFEST').readline())['audio'])" 2>/dev/null || true)
[ -n "$w" ] && [ ! -f "$w" ] && { echo "ERROR: first train wav not found: $w"; exit 1; }

# ---- preflight: enough FREE memory on each requested GPU ----------------------
if [ "$FORCE" != "1" ]; then
  tight=""
  for g in ${GPUS//,/ }; do
    free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "$g" 2>/dev/null | tr -d ' ')
    [ -z "$free" ] && { echo "ERROR: GPU $g not found"; exit 1; }
    [ "$free" -lt "$MIN_FREE_MIB" ] && tight="$tight $g(${free}MiB free)"
  done
  if [ -n "$tight" ]; then
    echo "ABORT: these GPUs have < ${MIN_FREE_MIB}MiB free:$tight"
    echo "       (this smoke needs ~12GB/GPU). Wait for them to free up, pick others via GPUS=...,"
    echo "       lower BATCH, or FORCE=1 to override (risks CUDA OOM)."; exit 1
  fi
fi

# ---- preflight: don't clobber ------------------------------------------------
if [ "$FORCE" != "1" ] && ls "$OUT"/*.pt >/dev/null 2>&1; then
  echo "ABORT: $OUT already has checkpoints. set OUT=... or FORCE=1."; exit 1
fi

echo "smoke config: pos_mode=$POS_MODE fill_weight=$FILL_WEIGHT eval_decode_steps=$EVAL_DECODE_STEPS"
echo "  train=$N_TRAIN utts (dev_other)  val=dev_clean(<=$DEV_N)  epochs=$EPOCHS  steps/epoch~$STEPS_PER_EPOCH  warmup=$WARMUP"
echo "  GPUs=$GPUS  eff_batch=$((BATCH*ACCUM*NG))  wandb=on"

# ---- launch (nohup) : wandb ON, no TAP_CACHE (audio path) --------------------
nohup env \
  ACMLM=1 GPU="$GPUS" WANDB=1 \
  SA_LAYERS="$SA_LAYERS" EPOCHS="$EPOCHS" BATCH="$BATCH" ACCUM="$ACCUM" \
  MASK_PROB="$MASK_PROB" MASK_PROB_MAX="$MASK_PROB_MAX" MASK_LEN="$MASK_LEN" \
  EOS_REPEAT="$EOS_REPEAT" FILL_WEIGHT="$FILL_WEIGHT" FPS="$FPS" \
  LR="$LR" WARMUP="$WARMUP" LR_SCHEDULE="$LR_SCHEDULE" LR_EXP_FINAL_RATIO="$LR_EXP_FINAL_RATIO" \
  POS_MODE="$POS_MODE" EVAL_DECODE_STEPS="$EVAL_DECODE_STEPS" EVAL_DECODE_N="$EVAL_DECODE_N" \
  EVAL_STEPS="$EVAL_STEPS" SAVE_STEPS="$SAVE_STEPS" LOG_STEPS="$LOG_STEPS" DEV_N="$DEV_N" \
  TRAIN_MANIFEST="$TRAIN_MANIFEST" DEV_MANIFEST="$DEV_MANIFEST" \
  VOCAB_DIR="$VOCAB_DIR" OUT="$OUT" MASTER_PORT="$MASTER_PORT" \
  bash run.sh > "$LOG" 2>&1 &
echo $! > "$PIDFILE"

echo "launched (pid=$(cat "$PIDFILE"))"
echo "  log    : $LOG"
echo "  pidfile: $PIDFILE"
echo "  watch  : tail -f \"$LOG\"     (look for 'pos_mode=alibi', decreasing loss, and '[eval @ N] iter32 WER=...')"
echo "  wandb  : project 'filler_asr'  (run name contains _alibi_fw0.2)"
echo "  stop   : kill \$(cat \"$PIDFILE\")"

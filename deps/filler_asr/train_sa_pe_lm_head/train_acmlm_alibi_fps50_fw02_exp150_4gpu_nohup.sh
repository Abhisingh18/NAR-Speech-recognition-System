#!/usr/bin/env bash
# =============================================================================
# A-CMLM SA-8 + lm_head, LS-960.  DERIVED from the CURRENTLY-RUNNING baseline
#   runs/full960_sa8_ACMLM_pe_fps50_mask20-30x10_eos3_fw0.03_exp_150ep_tapcache_v2
# (launcher: train_acmlm_mask20-30_eos3_fw003_fps50_exp_150ep_4gpu.sh) with ONLY
# the three requested changes:
#     * POSITION   : sinusoidal PE  ->  symmetric ALiBi   (POS_MODE=alibi; no absolute PE)
#     * FILL_WEIGHT: 0.03           ->  0.2               (stop starving the <fill> tail)
#     * VALIDATION : one-shot p=1.0 ->  ALSO 32-step iterative mask-predict WER at eval
# KEPT identical: A-CMLM text masking + addition fusion, frozen HuBERT tap, 20-30%
#     audio-tap masking x10, 8 SA layers, eos_repeat=3, fps=50 (n_keep=T),
#     lr 2e-4, warmup 5000, exponential decay to 1% peak, 150 epochs, batch 32,
#     accum 2, 4-way DDP.  Runs via the SAME run.sh (new flags are opt-in passthroughs).
#
#   run   : GPUS=1,2,3,4 bash train_acmlm_alibi_fps50_fw02_exp150_4gpu_nohup.sh
#   watch : tail -f <printed log>
#   stop  : kill $(cat <printed pidfile>)
#   decode: python filler_asr_omni_style_decoding.py --run_dir <OUT> --manifest <test.jsonl>
#           (pos_mode is read back from the checkpoint args automatically)
# =============================================================================
set -euo pipefail
cd "$(dirname "$0")"
RUN_DIR="$(pwd)"

# ---- knobs (only the 3 changes differ from the fw0.03 baseline) --------------
GPUS="${GPUS:-1,2,3,4}"                             # <-- pick FREE gpus (baseline uses 5,6,7,8)
POS_MODE="${POS_MODE:-alibi}"                       # CHANGED: symmetric ALiBi
FILL_WEIGHT="${FILL_WEIGHT:-0.2}"                   # CHANGED: 0.03 -> 0.2
EVAL_DECODE_STEPS="${EVAL_DECODE_STEPS:-32}"        # CHANGED: 32-step iterative eval
EVAL_DECODE_N="${EVAL_DECODE_N:-200}"              # cap dev clips iteratively decoded per eval
EVAL_STEPS="${EVAL_STEPS:-5000}"                    # CHANGED: validate every 5k steps
SAVE_STEPS="${SAVE_STEPS:-5000}"                    # CHANGED: checkpoint latest.pt every 5k steps

SA_LAYERS="${SA_LAYERS:-8}"
EPOCHS="${EPOCHS:-150}"
BATCH="${BATCH:-32}"
ACCUM="${ACCUM:-2}"                                 # eff batch = 32*2*4 = 256
MASK_PROB="${MASK_PROB:-0.20}"                      # AUDIO-tap masking (KEPT): 20-30%
MASK_PROB_MAX="${MASK_PROB_MAX:-0.30}"
MASK_LEN="${MASK_LEN:-10}"
EOS_REPEAT="${EOS_REPEAT:-3}"
FPS="${FPS:-50}"                                    # n_keep = T (grade all frames)
LR="${LR:-2e-4}"
WARMUP="${WARMUP:-5000}"
LR_SCHEDULE="${LR_SCHEDULE:-exponential}"
LR_EXP_FINAL_RATIO="${LR_EXP_FINAL_RATIO:-0.01}"
MASTER_PORT="${MASTER_PORT:-29581}"                # fresh port (baseline used 29574)
TAP_CACHE="${TAP_CACHE:-/speech/tomson/filler_asr/data/tapcache}"  # same cache as baseline (train/ + dev/ + PREPARED)
VOCAB_DIR="${VOCAB_DIR:-/speech/tomson/filler_asr/train_sa_pe_lm_head/runs/full960_sa8_pe_fps25_lin_100ep_mask20-30x10_eos3_fw0.1}"
OUT="${OUT:-/speech/tomson/filler_asr/train_sa_pe_lm_head/runs/full960_sa8_ACMLM_alibi_fps50_mask20-30x10_eos3_fw0.2_exp_150ep}"
FORCE="${FORCE:-0}"

STAMP="$(date +%Y%m%d_%H%M%S)"
mkdir -p "$RUN_DIR/logs" "$OUT"
LOG="$RUN_DIR/logs/train_acmlm_alibi_fw02_exp150_${STAMP}.log"
PIDFILE="$OUT/run.pid"

# ---- preflight: TAP_CACHE ready (this run trains from cached taps) ------------
if [ ! -f "$TAP_CACHE/train/PREPARED" ] || [ ! -f "$TAP_CACHE/dev/PREPARED" ]; then
  echo "ERROR: tap cache not ready at $TAP_CACHE (need train/PREPARED + dev/PREPARED)."
  echo "       Run extract_taps.sh first, or set TAP_CACHE=<prepared cache dir>."; exit 1
fi

# ---- preflight: requested GPUs actually free (don't fight the baseline run) ---
if [ "$FORCE" != "1" ]; then
  busy=""
  for g in ${GPUS//,/ }; do
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$g" 2>/dev/null | tr -d ' ')
    if [ -z "$used" ]; then echo "ERROR: GPU $g not found by nvidia-smi"; exit 1; fi
    if [ "$used" -gt 1000 ]; then busy="$busy $g(${used}MiB)"; fi
  done
  [ -n "$busy" ] && { echo "ABORT: requested GPUs busy:$busy  (pick others via GPUS=... or FORCE=1)"; exit 1; }
fi

# ---- preflight: don't clobber an existing run --------------------------------
if [ "$FORCE" != "1" ] && ls "$OUT"/*.pt >/dev/null 2>&1; then
  echo "ABORT: $OUT already has checkpoints. set OUT=... or FORCE=1."; exit 1
fi

# ---- launch detached with nohup ----------------------------------------------
nohup env \
  ACMLM=1 GPU="$GPUS" SA_LAYERS="$SA_LAYERS" EPOCHS="$EPOCHS" BATCH="$BATCH" ACCUM="$ACCUM" \
  MASK_PROB="$MASK_PROB" MASK_PROB_MAX="$MASK_PROB_MAX" MASK_LEN="$MASK_LEN" \
  EOS_REPEAT="$EOS_REPEAT" FILL_WEIGHT="$FILL_WEIGHT" \
  FPS="$FPS" LR="$LR" WARMUP="$WARMUP" LR_SCHEDULE="$LR_SCHEDULE" LR_EXP_FINAL_RATIO="$LR_EXP_FINAL_RATIO" \
  POS_MODE="$POS_MODE" EVAL_DECODE_STEPS="$EVAL_DECODE_STEPS" EVAL_DECODE_N="$EVAL_DECODE_N" \
  EVAL_STEPS="$EVAL_STEPS" SAVE_STEPS="$SAVE_STEPS" \
  TAP_CACHE="$TAP_CACHE" VOCAB_DIR="$VOCAB_DIR" OUT="$OUT" MASTER_PORT="$MASTER_PORT" \
  bash run.sh > "$LOG" 2>&1 &
echo $! > "$PIDFILE"

NG=$(awk -F, '{print NF}' <<<"$GPUS")
echo "launched A-CMLM ALiBi fw0.2 run with nohup  (pid=$(cat "$PIDFILE"))"
echo "  changed     : pos_mode=$POS_MODE  fill_weight=$FILL_WEIGHT  eval_decode_steps=$EVAL_DECODE_STEPS"
echo "  eval/save   : every $EVAL_STEPS / $SAVE_STEPS steps"
echo "  GPUs        : $GPUS   (eff batch = $BATCH*$ACCUM*$NG = $((BATCH*ACCUM*NG)))"
echo "  schedule    : $LR_SCHEDULE (final=$LR_EXP_FINAL_RATIO of peak), epochs=$EPOCHS, fps=$FPS (n_keep=T)"
echo "  out dir     : $OUT"
echo "  log         : $LOG"
echo "  pidfile     : $PIDFILE"
echo "  watch  : tail -f \"$LOG\""
echo "  stop   : kill \$(cat \"$PIDFILE\")"
echo "  decode : python filler_asr_omni_style_decoding.py --run_dir \"$OUT\" --manifest <test.jsonl>"

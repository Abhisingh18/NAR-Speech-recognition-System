#!/usr/bin/env bash
# =============================================================================
# A-CMLM SA-8 + lm_head, LS-960.  DERIVED from the ALiBi fw0.2 run
#   runs/full960_sa8_ACMLM_alibi_fps50_mask20-30x10_eos3_fw0.2_exp_150ep
# with ONLY two changes (fill_weight ablation + checkpoint-selection fix):
#     * FILL_WEIGHT   : 0.2  ->  0.03   (test whether pulling loss budget back to CONTENT
#                                        cuts the char-doubling substitution tax; ALiBi -- not
#                                        fill_weight -- is what holds the </s>/tail structure,
#                                        so we expect to keep the structural fix while sharpening spelling)
#     * SELECT_METRIC : loss ->  iter_wer  (best.pt = argmin realistic 32-step iterative-decode
#                                        dev WER, NOT one-shot MLM loss.  On the fw0.2 run best.pt-by-loss
#                                        saturated at ~ep69 while decode WER kept falling to ep150 -- so
#                                        loss-selected best.pt was ~55% worse than the final ckpt.)
# KEPT identical to the fw0.2 ALiBi run: pos_mode=alibi, A-CMLM text masking + addition fusion,
#     frozen HuBERT tap, 20-30% tap masking x10, 8 SA layers, eos_repeat=3, fps=50 (n_keep=T),
#     lr 2e-4, warmup 5000, exponential decay to 1% peak, 150 epochs, eff batch 256, 4-way DDP,
#     32-step iterative-decode eval, eval/save every 5k steps.
#
#   run : GPUS=1,2,3,4 bash train_acmlm_alibi_fps50_fw003_exp150_4gpu_nohup.sh
# =============================================================================
set -euo pipefail
cd "$(dirname "$0")"
RUN_DIR="$(pwd)"

# ---- knobs (differences from fw0.2 ALiBi run: FILL_WEIGHT + SELECT_METRIC) ---
GPUS="${GPUS:-1,2,3,4}"
POS_MODE="${POS_MODE:-alibi}"                       # KEPT: symmetric ALiBi
FILL_WEIGHT="${FILL_WEIGHT:-0.03}"                  # CHANGED: 0.2 -> 0.03
SELECT_METRIC="${SELECT_METRIC:-iter_wer}"          # CHANGED: best.pt on iterative-decode WER
EVAL_DECODE_STEPS="${EVAL_DECODE_STEPS:-32}"        # 32-step iterative eval (drives iter_wer selection)
EVAL_DECODE_N="${EVAL_DECODE_N:-200}"              # dev clips iteratively decoded per eval
EVAL_STEPS="${EVAL_STEPS:-5000}"
SAVE_STEPS="${SAVE_STEPS:-5000}"

SA_LAYERS="${SA_LAYERS:-8}"
EPOCHS="${EPOCHS:-150}"
BATCH="${BATCH:-32}"
ACCUM="${ACCUM:-2}"                                 # eff batch = 32*2*4 = 256
MASK_PROB="${MASK_PROB:-0.20}"
MASK_PROB_MAX="${MASK_PROB_MAX:-0.30}"
MASK_LEN="${MASK_LEN:-10}"
EOS_REPEAT="${EOS_REPEAT:-3}"
FPS="${FPS:-50}"
LR="${LR:-2e-4}"
WARMUP="${WARMUP:-5000}"
LR_SCHEDULE="${LR_SCHEDULE:-exponential}"
LR_EXP_FINAL_RATIO="${LR_EXP_FINAL_RATIO:-0.01}"
MASTER_PORT="${MASTER_PORT:-29582}"                # fresh port
TAP_CACHE="${TAP_CACHE:-/speech/tomson/filler_asr/data/tapcache}"
VOCAB_DIR="${VOCAB_DIR:-/speech/tomson/filler_asr/train_sa_pe_lm_head/runs/full960_sa8_pe_fps25_lin_100ep_mask20-30x10_eos3_fw0.1}"
OUT="${OUT:-/speech/tomson/filler_asr/train_sa_pe_lm_head/runs/full960_sa8_ACMLM_alibi_fps50_mask20-30x10_eos3_fw0.03_exp_150ep}"
FORCE="${FORCE:-0}"

STAMP="$(date +%Y%m%d_%H%M%S)"
mkdir -p "$RUN_DIR/logs" "$OUT"
LOG="$RUN_DIR/logs/train_acmlm_alibi_fw003_exp150_${STAMP}.log"
PIDFILE="$OUT/run.pid"

if [ ! -f "$TAP_CACHE/train/PREPARED" ] || [ ! -f "$TAP_CACHE/dev/PREPARED" ]; then
  echo "ERROR: tap cache not ready at $TAP_CACHE (need train/PREPARED + dev/PREPARED)."; exit 1
fi

if [ "$FORCE" != "1" ]; then
  busy=""
  for g in ${GPUS//,/ }; do
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$g" 2>/dev/null | tr -d ' ')
    if [ -z "$used" ]; then echo "ERROR: GPU $g not found by nvidia-smi"; exit 1; fi
    if [ "$used" -gt 1000 ]; then busy="$busy $g(${used}MiB)"; fi
  done
  [ -n "$busy" ] && { echo "ABORT: requested GPUs busy:$busy  (pick others via GPUS=... or FORCE=1)"; exit 1; }
fi

if [ "$FORCE" != "1" ] && ls "$OUT"/*.pt >/dev/null 2>&1; then
  echo "ABORT: $OUT already has checkpoints. set OUT=... or FORCE=1."; exit 1
fi

nohup env \
  ACMLM=1 GPU="$GPUS" SA_LAYERS="$SA_LAYERS" EPOCHS="$EPOCHS" BATCH="$BATCH" ACCUM="$ACCUM" \
  MASK_PROB="$MASK_PROB" MASK_PROB_MAX="$MASK_PROB_MAX" MASK_LEN="$MASK_LEN" \
  EOS_REPEAT="$EOS_REPEAT" FILL_WEIGHT="$FILL_WEIGHT" \
  FPS="$FPS" LR="$LR" WARMUP="$WARMUP" LR_SCHEDULE="$LR_SCHEDULE" LR_EXP_FINAL_RATIO="$LR_EXP_FINAL_RATIO" \
  POS_MODE="$POS_MODE" EVAL_DECODE_STEPS="$EVAL_DECODE_STEPS" EVAL_DECODE_N="$EVAL_DECODE_N" \
  SELECT_METRIC="$SELECT_METRIC" EVAL_STEPS="$EVAL_STEPS" SAVE_STEPS="$SAVE_STEPS" \
  TAP_CACHE="$TAP_CACHE" VOCAB_DIR="$VOCAB_DIR" OUT="$OUT" MASTER_PORT="$MASTER_PORT" \
  bash run.sh > "$LOG" 2>&1 &
echo $! > "$PIDFILE"

NG=$(awk -F, '{print NF}' <<<"$GPUS")
echo "launched A-CMLM ALiBi fw0.03 run with nohup  (pid=$(cat "$PIDFILE"))"
echo "  changed     : fill_weight=$FILL_WEIGHT (was 0.2)   select_metric=$SELECT_METRIC (was loss)"
echo "  kept        : pos_mode=$POS_MODE  eval_decode_steps=$EVAL_DECODE_STEPS  eff batch=$((BATCH*ACCUM*NG))"
echo "  out dir     : $OUT"
echo "  log         : $LOG"
echo "  watch  : tail -f \"$LOG\""
echo "  stop   : kill \$(cat \"$PIDFILE\")"

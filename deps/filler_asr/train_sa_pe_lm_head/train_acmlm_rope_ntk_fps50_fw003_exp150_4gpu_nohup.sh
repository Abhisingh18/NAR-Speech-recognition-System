#!/usr/bin/env bash
# =============================================================================
# A-CMLM SA-8 + lm_head, LS-960.  RoPE + dynamic-NTK arm of the positional-scheme
# comparison. DERIVED from the ALiBi fw0.03 run
#   runs/full960_sa8_ACMLM_alibi_fps50_mask20-30x10_eos3_fw0.03_exp_150ep
# (train_acmlm_alibi_fps50_fw003_exp150_4gpu_nohup.sh) with ONLY the positional
# scheme changed, so ALiBi vs RoPE is the single controlled variable:
#     * POS_MODE : alibi -> rope   (rotary PE: rotates q,k inside attention so the
#                                   SA logits depend on RELATIVE frame offset; a
#                                   custom post-norm encoder replaces nn.Transformer-
#                                   Encoder only in this mode, same post-norm/GELU math)
#     * RoPE base θ=10000, DYNAMIC NTK-aware length scaling on: sequences up to
#       ROPE_ORIG_LEN=1024 frames (~20s @50fps) are plain RoPE; longer ones get
#       base'=base*s^(d/(d-2)), s=max(1,T/1024) — no in-distribution penalty,
#       graceful long-utterance extrapolation with no fine-tuning.
# KEPT identical to the fw0.03 ALiBi run: A-CMLM text masking + addition fusion,
#     frozen HuBERT tap, 20-30% tap masking x10, 8 SA layers, eos_repeat=3, fps=50,
#     fill_weight=0.03, select_metric=iter_wer, 32-step iterative-decode eval,
#     lr 2e-4, warmup 5000, exponential decay to 1% peak, 150 epochs, eff batch 256,
#     4-way DDP, eval/save every 5k steps.
#
#   run : GPUS=1,2,3,4 bash train_acmlm_rope_ntk_fps50_fw003_exp150_4gpu_nohup.sh
#   (Llama-3-style long-context variant:  ROPE_THETA=500000 GPUS=1,2,3,4 bash ...)
# =============================================================================
set -euo pipefail
cd "$(dirname "$0")"
RUN_DIR="$(pwd)"

# ---- knobs (only difference from the fw0.03 ALiBi run: POS_MODE + RoPE/NTK) ---
GPUS="${GPUS:-1,2,3,4}"
POS_MODE="${POS_MODE:-rope}"                        # CHANGED: rotary PE (was alibi)
ROPE_THETA="${ROPE_THETA:-10000}"                  # RoPE base θ (500000 = Llama-3-style)
ROPE_ORIG_LEN="${ROPE_ORIG_LEN:-1024}"             # dynamic-NTK reference length (frames)
ROPE_NTK_FACTOR="${ROPE_NTK_FACTOR:-1.0}"          # static NTK stretch (only if ROPE_NTK_DYNAMIC=0)
ROPE_NTK_DYNAMIC="${ROPE_NTK_DYNAMIC:-1}"          # 1 = dynamic NTK (default); 0 = static

FILL_WEIGHT="${FILL_WEIGHT:-0.03}"                 # KEPT: 0.03 (as in the fw0.03 ALiBi run)
SELECT_METRIC="${SELECT_METRIC:-iter_wer}"         # KEPT: best.pt on iterative-decode WER
EVAL_DECODE_STEPS="${EVAL_DECODE_STEPS:-32}"       # 32-step iterative eval (drives iter_wer selection)
EVAL_DECODE_N="${EVAL_DECODE_N:-200}"             # dev clips iteratively decoded per eval
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
MASTER_PORT="${MASTER_PORT:-29584}"                # fresh port (alibi run used 29582)
TAP_CACHE="${TAP_CACHE:-/speech/tomson/filler_asr/data/tapcache}"
VOCAB_DIR="${VOCAB_DIR:-/speech/tomson/filler_asr/train_sa_pe_lm_head/runs/full960_sa8_pe_fps25_lin_100ep_mask20-30x10_eos3_fw0.1}"
OUT="${OUT:-/speech/tomson/filler_asr/train_sa_pe_lm_head/runs/full960_sa8_ACMLM_rope_ntk_fps50_mask20-30x10_eos3_fw0.03_exp_150ep}"
FORCE="${FORCE:-0}"

STAMP="$(date +%Y%m%d_%H%M%S)"
mkdir -p "$RUN_DIR/logs" "$OUT"
LOG="$RUN_DIR/logs/train_acmlm_rope_ntk_fw003_exp150_${STAMP}.log"
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
  POS_MODE="$POS_MODE" ROPE_THETA="$ROPE_THETA" ROPE_ORIG_LEN="$ROPE_ORIG_LEN" \
  ROPE_NTK_FACTOR="$ROPE_NTK_FACTOR" ROPE_NTK_DYNAMIC="$ROPE_NTK_DYNAMIC" \
  EVAL_DECODE_STEPS="$EVAL_DECODE_STEPS" EVAL_DECODE_N="$EVAL_DECODE_N" \
  SELECT_METRIC="$SELECT_METRIC" EVAL_STEPS="$EVAL_STEPS" SAVE_STEPS="$SAVE_STEPS" \
  TAP_CACHE="$TAP_CACHE" VOCAB_DIR="$VOCAB_DIR" OUT="$OUT" MASTER_PORT="$MASTER_PORT" \
  bash run.sh > "$LOG" 2>&1 &
echo $! > "$PIDFILE"

NG=$(awk -F, '{print NF}' <<<"$GPUS")
echo "launched A-CMLM RoPE+NTK fw0.03 run with nohup  (pid=$(cat "$PIDFILE"))"
echo "  changed     : pos_mode=$POS_MODE (was alibi)   rope_theta=$ROPE_THETA  ntk_dynamic=$ROPE_NTK_DYNAMIC (orig_len=$ROPE_ORIG_LEN)"
echo "  kept        : fill_weight=$FILL_WEIGHT  select_metric=$SELECT_METRIC  eval_decode_steps=$EVAL_DECODE_STEPS  eff batch=$((BATCH*ACCUM*NG))"
echo "  out dir     : $OUT"
echo "  log         : $LOG"
echo "  watch  : tail -f \"$LOG\""
echo "  stop   : kill \$(cat \"$PIDFILE\")"

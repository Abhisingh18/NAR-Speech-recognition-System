#!/usr/bin/env bash
# =============================================================================
# A-CMLM SA-8 + PE + lm_head, LS-960.  DERIVED from the fps25/linear/100ep parent
#   runs/full960_sa8_ACMLM_pe_fps25_mask20-30x10_eos3_fw0.1
# with EVERYTHING the same EXCEPT the three requested changes:
#     * frames_per_sec 25 -> 50   (n_keep = min(ceil(dur*50),T) = T -> grade ALL
#                                  HuBERT frames; in A-CMLM this widens the CMLM
#                                  reconstruction window to the full grid.)
#     * fill_weight 0.1 -> 0.03    (fps=50 already ~3x'd the <fill> loss share;
#                                  0.03 restores the parent's ~6% <fill> share)
#     * lr_schedule linear -> exponential (warmup 5000 -> geometric decay to 1% peak)
#     * epochs 100 -> 150
# KEPT identical: A-CMLM text masking + addition fusion, frozen HuBERT tap,
#     20-30% audio-tap masking x10, sinusoidal PE, 8 SA, eos_repeat=3,
#     lr 2e-4, warmup 5000, batch 32, accum 2, 4-way DDP.
#
#   GPUs   : physical 5,6,7,8  (4-way DDP)     accum 2 -> eff batch 32*2*4 = 256
#   run    : bash train_acmlm_mask20-30_eos3_fw003_fps50_exp_150ep_4gpu.sh
#   watch  : tmux attach -t acmlm_fps50_fw003_exp150
#   decode : python filler_asr_omni_style_decoding.py --run_dir <OUT> --manifest <test.jsonl>
# =============================================================================
set -euo pipefail
cd "$(dirname "$0")"

SESSION="${SESSION:-acmlm_fps50_fw003_exp150}"
GPUS="${GPUS:-5,6,7,8}"
SA_LAYERS="${SA_LAYERS:-8}"
EPOCHS="${EPOCHS:-150}"                             # was 100
BATCH="${BATCH:-32}"
ACCUM="${ACCUM:-2}"                                 # 32*2*4 = eff batch 256
MASK_PROB="${MASK_PROB:-0.20}"                      # AUDIO-tap masking (KEPT): 20-30%
MASK_PROB_MAX="${MASK_PROB_MAX:-0.30}"
MASK_LEN="${MASK_LEN:-10}"
EOS_REPEAT="${EOS_REPEAT:-3}"
FILL_WEIGHT="${FILL_WEIGHT:-0.03}"
FPS="${FPS:-50}"                                    # was 25 -> grade all frames (remove the ceil(dur*25) budget)
LR_SCHEDULE="${LR_SCHEDULE:-exponential}"           # was linear
LR_EXP_FINAL_RATIO="${LR_EXP_FINAL_RATIO:-0.01}"    # exponential floor: 1% of peak
MASTER_PORT="${MASTER_PORT:-29574}"                 # free port (parent used 29573)
OUT="${OUT:-/speech/tomson/filler_asr/train_sa_pe_lm_head/runs/full960_sa8_ACMLM_pe_fps50_mask20-30x10_eos3_fw0.03_exp_150ep}"
FORCE="${FORCE:-0}"

RUN_DIR="$(pwd)"
STAMP="$(date +%Y%m%d_%H%M%S)"
LOG="$RUN_DIR/logs/train_acmlm_fps50_exp150_${STAMP}.log"
mkdir -p "$RUN_DIR/logs"

# ---- preflight: requested GPUs free? -----------------------------------------
if [ "$FORCE" != "1" ]; then
  busy=""
  for g in ${GPUS//,/ }; do
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$g" 2>/dev/null | tr -d ' ')
    if [ -z "$used" ]; then echo "ERROR: GPU $g not found by nvidia-smi"; exit 1; fi
    if [ "$used" -gt 1000 ]; then busy="$busy $g(${used}MiB)"; fi
  done
  if [ -n "$busy" ]; then
    echo "ABORT: these requested GPUs are busy:$busy  (pick others via GPUS=... or FORCE=1)"; exit 1
  fi
fi

# ---- preflight: don't clobber an existing run --------------------------------
if [ "$FORCE" != "1" ] && ls "$OUT"/*.pt >/dev/null 2>&1; then
  echo "ABORT: $OUT already has checkpoints. set OUT=... or FORCE=1."; exit 1
fi
mkdir -p "$OUT"

if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "tmux session '$SESSION' exists -> tmux attach -t $SESSION"; exit 1
fi

# ---- launch detached in tmux -------------------------------------------------
tmux new-session -d -s "$SESSION" "bash -lc '
  source /speech/tomson/miniconda3/etc/profile.d/conda.sh
  conda activate filler_asr
  cd \"$RUN_DIR\"
  ACMLM=1 GPU=\"$GPUS\" SA_LAYERS=$SA_LAYERS EPOCHS=$EPOCHS BATCH=$BATCH ACCUM=$ACCUM \
    MASK_PROB=$MASK_PROB MASK_PROB_MAX=$MASK_PROB_MAX MASK_LEN=$MASK_LEN \
    EOS_REPEAT=$EOS_REPEAT FILL_WEIGHT=$FILL_WEIGHT \
    FPS=$FPS LR_SCHEDULE=$LR_SCHEDULE LR_EXP_FINAL_RATIO=$LR_EXP_FINAL_RATIO \
    OUT=\"$OUT\" MASTER_PORT=$MASTER_PORT \
    bash run.sh 2>&1 | tee \"$LOG\"
  echo \"=== training exited (rc=\${PIPESTATUS[0]}) ; pane kept open ===\"
  exec bash
'"

NG=$(awk -F, '{print NF}' <<<"$GPUS")
echo "launched A-CMLM fps50/exp/150ep run in tmux session : $SESSION"
echo "  GPUs        : $GPUS   (eff batch = $BATCH*$ACCUM*$NG = $((BATCH*ACCUM*NG)))"
echo "  budget      : frames_per_sec=$FPS  (n_keep=T -> grade all frames)"
echo "  schedule    : $LR_SCHEDULE (final=$LR_EXP_FINAL_RATIO of peak), epochs=$EPOCHS"
echo "  audio mask  : U[$MASK_PROB,$MASK_PROB_MAX] x $MASK_LEN   (KEPT)"
echo "  out dir     : $OUT"
echo "  log         : $LOG"
echo "  attach : tmux attach -t $SESSION      (detach: Ctrl-b then d)"
echo "  decode : python filler_asr_omni_style_decoding.py --run_dir \"$OUT\" --manifest <test.jsonl>"

#!/usr/bin/env bash
# =============================================================================
# A-CMLM (audio-conditioned masked-LM) SA-8 + PE + lm_head, LS-960, COLD 100ep.
# Same recipe as the mask20-30 / eos3 / fw0.1 parent run, PLUS:
#     * text embedding (34x1280, <mask>=33) fused by ADDITION (no gamma)
#     * CMLM text masking p~U(0,1) (+15% forced p=1.0); loss on masked slots only
#     * iterative OmniVoice-style decoding at inference (filler_asr_omni_style_decoding.py)
# KEPT from parent: frozen HuBERT tap, 20-30% AUDIO-tap masking, sinusoidal PE,
# 8 SA, eos_repeat=3, duration budget n_keep=ceil(dur*25), fill_weight=0.1, lr 2e-4.
#
#   GPUs   : physical 5,6,7,8  (4-way DDP)     accum 2 -> eff batch 32*2*4 = 256
#   run    : bash train_acmlm_mask20-30_eos3_fw01_100ep_4gpu.sh
#   watch  : tmux attach -t acmlm_m2030_eos3fw01
#   decode : python filler_asr_omni_style_decoding.py --run_dir <OUT> --manifest <test.jsonl>
# =============================================================================
set -euo pipefail
cd "$(dirname "$0")"

SESSION="${SESSION:-acmlm_m2030_eos3fw01}"
GPUS="${GPUS:-5,6,7,8}"
SA_LAYERS="${SA_LAYERS:-8}"
EPOCHS="${EPOCHS:-100}"
BATCH="${BATCH:-32}"
ACCUM="${ACCUM:-2}"                                 # 32*2*4 = eff batch 256
MASK_PROB="${MASK_PROB:-0.20}"                      # AUDIO-tap masking (KEPT): 20-30%
MASK_PROB_MAX="${MASK_PROB_MAX:-0.30}"
MASK_LEN="${MASK_LEN:-10}"
EOS_REPEAT="${EOS_REPEAT:-3}"
FILL_WEIGHT="${FILL_WEIGHT:-0.01}"
LR_EXP_FINAL_RATIO="${LR_EXP_FINAL_RATIO:-0.03}"
MASTER_PORT="${MASTER_PORT:-29573}"                 # free port (parent used 29572)
OUT="${OUT:-/speech/tomson/filler_asr/train_sa_pe_lm_head/runs/full960_sa8_ACMLM_pe_fps25_mask20-30x10_eos3_fw0.1}"
FORCE="${FORCE:-0}"

RUN_DIR="$(pwd)"
STAMP="$(date +%Y%m%d_%H%M%S)"
LOG="$RUN_DIR/logs/train_acmlm_${STAMP}.log"
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
    OUT=\"$OUT\" MASTER_PORT=$MASTER_PORT \
    bash run.sh 2>&1 | tee \"$LOG\"
  echo \"=== training exited (rc=\${PIPESTATUS[0]}) ; pane kept open ===\"
  exec bash
'"

NG=$(awk -F, '{print NF}' <<<"$GPUS")
echo "launched A-CMLM run in tmux session : $SESSION"
echo "  GPUs        : $GPUS   (eff batch = $BATCH*$ACCUM*$NG = $((BATCH*ACCUM*NG)))"
echo "  audio mask  : U[$MASK_PROB,$MASK_PROB_MAX] x $MASK_LEN   (KEPT)"
echo "  text mask   : CMLM p~U(0,1) + iterative decode           (NEW)"
echo "  out dir     : $OUT"
echo "  log         : $LOG"
echo "  attach : tmux attach -t $SESSION      (detach: Ctrl-b then d)"
echo "  decode : python filler_asr_omni_style_decoding.py --run_dir \"$OUT\" --manifest <test.jsonl>"

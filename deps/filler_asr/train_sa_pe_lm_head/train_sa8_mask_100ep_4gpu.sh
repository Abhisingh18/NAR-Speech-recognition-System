#!/usr/bin/env bash
# =============================================================================
# SA-8 + PE + lm_head with TRAIN-TIME INPUT MASKING of the HuBERT tap, LS-960.
# Same recipe as train_sa8_100ep_6gpu.sh (the 13.5%-dev baseline) EXCEPT:
#   - masking : 10-20% of valid frames (per-batch rate ~U[0.10,0.20]),
#               spans of 10 frames (200ms), learned mask_embed, BEFORE the PE add.
#               Eval is always unmasked (model.eval() gates it off).
#   - GPUs    : physical 7,8,9,10  (4-way DDP via torchrun)
#   - accum   : 3  ->  effective batch = 32*3*4 = 384  (matches baseline's 384,
#               so steps/epoch (~732) and the LR schedule line up run-to-run)
#
# Baseline for A/B: runs/full960_sa8_pe_fps25_lin_100ep_bs32acc2 (best dev WER 0.135)
#
#   run   : bash train_sa8_mask_100ep_4gpu.sh
#   watch : tmux attach -t sa8_mask          (detach again with Ctrl-b d)
#   log   : tail -f <printed log path>
#
# Optional warm-start from the baseline head (fresh optimizer/schedule):
#   INIT_FROM=/speech/tomson/filler_asr/train_sa_pe_lm_head/runs/full960_sa8_pe_fps25_lin_100ep_bs32acc2/best.pt \
#     EPOCHS=30 bash train_sa8_mask_100ep_4gpu.sh
#
# Safety: refuses to start if any requested GPU is busy, or if OUT already has
# checkpoints. Override either with FORCE=1 (only if you know what you're doing).
# =============================================================================
set -euo pipefail
cd "$(dirname "$0")"

# ---- config (override via env if needed) ------------------------------------
SESSION="${SESSION:-sa8_mask}"
GPUS="${GPUS:-7,8,9,10}"                          # physical PCI-BUS indices (run.sh sets PCI_BUS_ID)
SA_LAYERS="${SA_LAYERS:-8}"
EPOCHS="${EPOCHS:-100}"
BATCH="${BATCH:-32}"
ACCUM="${ACCUM:-2}"                               # 32*3*4 GPUs = eff batch 384 (baseline parity)
MASK_PROB="${MASK_PROB:-0.10}"
MASK_PROB_MAX="${MASK_PROB_MAX:-0.20}"
MASK_LEN="${MASK_LEN:-10}"
INIT_FROM="${INIT_FROM:-}"                        # optional warm-start head ckpt
MASTER_PORT="${MASTER_PORT:-29561}"               # free; 29551=sa8 baseline, 29525=SLAM
OUT="${OUT:-/speech/tomson/filler_asr/train_sa_pe_lm_head/runs/full960_sa8_pe_fps25_lin_100ep_mask10-20x10_bs32acc3}"
FORCE="${FORCE:-0}"
# -----------------------------------------------------------------------------

RUN_DIR="$(pwd)"
STAMP="$(date +%Y%m%d_%H%M%S)"
LOG="$RUN_DIR/logs/train_sa8_mask_100ep_${STAMP}.log"
mkdir -p "$RUN_DIR/logs"

# ---- preflight: are the requested GPUs actually free? -----------------------
if [ "$FORCE" != "1" ]; then
  busy=""
  for g in ${GPUS//,/ }; do
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$g" 2>/dev/null | tr -d ' ')
    if [ -z "$used" ]; then echo "ERROR: GPU $g not found by nvidia-smi"; exit 1; fi
    if [ "$used" -gt 1000 ]; then busy="$busy $g(${used}MiB)"; fi
  done
  if [ -n "$busy" ]; then
    echo "ABORT: these requested GPUs are busy:$busy"
    echo "       wait for them to free up, pick other GPUs (GPUS=...), or override with FORCE=1."
    exit 1
  fi
fi

# ---- preflight: don't clobber an existing run -------------------------------
if [ "$FORCE" != "1" ] && ls "$OUT"/*.pt >/dev/null 2>&1; then
  echo "ABORT: $OUT already contains checkpoints (fresh run would overwrite)."
  echo "       set a new OUT=... or FORCE=1 to proceed."
  exit 1
fi
mkdir -p "$OUT"

# ---- already running? -------------------------------------------------------
if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "tmux session '$SESSION' already exists -> attach with: tmux attach -t $SESSION"
  exit 1
fi

# ---- launch detached in tmux: activate env, set GPUs, run, tee to log --------
tmux new-session -d -s "$SESSION" "bash -lc '
  source /speech/tomson/miniconda3/etc/profile.d/conda.sh
  conda activate filler_asr
  cd \"$RUN_DIR\"
  GPU=\"$GPUS\" SA_LAYERS=$SA_LAYERS EPOCHS=$EPOCHS BATCH=$BATCH ACCUM=$ACCUM \
    MASK_PROB=$MASK_PROB MASK_PROB_MAX=$MASK_PROB_MAX MASK_LEN=$MASK_LEN \
    INIT_FROM=\"$INIT_FROM\" \
    OUT=\"$OUT\" MASTER_PORT=$MASTER_PORT \
    bash run.sh 2>&1 | tee \"$LOG\"
  echo \"=== training exited (rc=\${PIPESTATUS[0]}) ; pane kept open ===\"
  exec bash
'"

echo "launched detached in tmux session : $SESSION"
echo "  GPUs        : $GPUS   (effective batch = $BATCH*$ACCUM*$(awk -F, '{print NF}' <<<"$GPUS")) "
echo "  masking     : U[$MASK_PROB,$MASK_PROB_MAX] x ${MASK_LEN}-frame spans (train only)"
echo "  out dir     : $OUT"
echo "  log         : $LOG"
echo
echo "  attach : tmux attach -t $SESSION      (detach: Ctrl-b then d)"
echo "  tail   : tail -f \"$LOG\""

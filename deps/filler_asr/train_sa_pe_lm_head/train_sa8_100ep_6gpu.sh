#!/usr/bin/env bash
# =============================================================================
# Fresh SA-8 + PE + lm_head training on a FROZEN HuBERT-xlarge, full LibriSpeech-960.
#   GPUs        : physical 1,4,6,7,8,9   (6-way DDP via torchrun)
#   epochs      : 100
#   SA blocks   : 8
#   batch/accum : 32 per-GPU x 2 accum  ->  effective batch = 32*2*6 = 384
#
# Launches DETACHED inside a tmux session with full logging, so it survives
# logout. It wraps run.sh (which handles env + torchrun + NCCL loopback).
#
#   run   : bash train_sa8_100ep_6gpu.sh
#   watch : tmux attach -t sa8_100ep        (detach again with Ctrl-b d)
#   log   : tail -f <printed log path>
#
# Safety: refuses to start if any requested GPU is busy, or if OUT already has
# checkpoints. Override either with FORCE=1 (only if you know what you're doing).
# =============================================================================
set -euo pipefail
cd "$(dirname "$0")"

# ---- config (override via env if needed) ------------------------------------
SESSION="${SESSION:-sa8_100ep}"
GPUS="${GPUS:-1,4,6,7,8,9}"                       # physical PCI-BUS indices (run.sh sets PCI_BUS_ID)
SA_LAYERS="${SA_LAYERS:-8}"
EPOCHS="${EPOCHS:-100}"
BATCH="${BATCH:-32}"
ACCUM="${ACCUM:-2}"
MASTER_PORT="${MASTER_PORT:-29551}"              # verified free; SLAM run uses 29525
# Fresh run -> its OWN out dir. DO NOT reuse full960_sa8_pe_fps25_lin: that best.pt
# is loaded live by the SLAM training + decode and would be clobbered mid-run.
OUT="${OUT:-/speech/tomson/filler_asr/train_sa_pe_lm_head/runs/full960_sa8_pe_fps25_lin_100ep_bs32acc2}"
FORCE="${FORCE:-0}"
# -----------------------------------------------------------------------------

RUN_DIR="$(pwd)"
STAMP="$(date +%Y%m%d_%H%M%S)"
LOG="$RUN_DIR/logs/train_sa8_100ep_${STAMP}.log"
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
    OUT=\"$OUT\" MASTER_PORT=$MASTER_PORT \
    bash run.sh 2>&1 | tee \"$LOG\"
  echo \"=== training exited (rc=\${PIPESTATUS[0]}) ; pane kept open ===\"
  exec bash
'"

echo "launched detached in tmux session : $SESSION"
echo "  GPUs        : $GPUS   (effective batch = $BATCH*$ACCUM*$(awk -F, '{print NF}' <<<"$GPUS")) "
echo "  out dir     : $OUT"
echo "  log         : $LOG"
echo
echo "  attach : tmux attach -t $SESSION      (detach: Ctrl-b then d)"
echo "  tail   : tail -f \"$LOG\""

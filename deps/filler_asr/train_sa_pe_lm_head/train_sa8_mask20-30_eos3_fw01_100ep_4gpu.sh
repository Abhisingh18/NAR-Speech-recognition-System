#!/usr/bin/env bash
# =============================================================================
# SA-8 + PE + lm_head, LS-960, COLD 100-epoch run (from scratch, NO warm-start —
# same regime as the mask10-20x10 baseline), with three deltas vs that baseline:
#     * masking     : 20-30% of frames  (was 10-20%)   -> stronger regularization
#     * eos_repeat  : 3   -> </s> on 3 consecutive end frames (was 1)
#     * fill_weight : 0.1 -> <fill> down-weighted in framewise CE (rest = 1)
# Everything else identical: frozen HuBERT-xlarge tap, sinusoidal PE (scale 1.0),
# 8 SA layers, 25-frame duration budget, lr 2e-4 (UNCHANGED), eff batch 256.
#
#   GPUs   : physical 1,2,3,4  (4-way DDP via torchrun)   [0 flaky; 5=T1000; 7-10 busy]
#   accum  : 2 -> effective batch = 32*2*4 = 256  (matches the mask baseline)
#   start  : COLD (random head) — NO --init_from, full 100-epoch schedule
#
#   run    : bash train_sa8_mask20-30_eos3_fw01_100ep_4gpu.sh
#   watch  : tmux attach -t sa8_m2030_eos3fw01     (detach: Ctrl-b then d)
#   log    : tail -f <printed log path>            (+ wandb project filler_asr)
#
# Safety: refuses to start if any requested GPU is busy or if OUT already has
# checkpoints. Override with FORCE=1.
# =============================================================================
set -euo pipefail
cd "$(dirname "$0")"

# ---- config (override via env if needed) ------------------------------------
SESSION="${SESSION:-sa8_m2030_eos3fw01}"
GPUS="${GPUS:-1,2,3,4}"
SA_LAYERS="${SA_LAYERS:-8}"
EPOCHS="${EPOCHS:-100}"                             # actual full run
BATCH="${BATCH:-32}"
ACCUM="${ACCUM:-2}"                                # 32*2*4 = eff batch 256 (baseline parity)
MASK_PROB="${MASK_PROB:-0.20}"                     # CHANGE: 20-30% (was 10-20%)
MASK_PROB_MAX="${MASK_PROB_MAX:-0.30}"
MASK_LEN="${MASK_LEN:-10}"
EOS_REPEAT="${EOS_REPEAT:-3}"                       # CHANGE 1
FILL_WEIGHT="${FILL_WEIGHT:-0.1}"                 # CHANGE 2
INIT_FROM="${INIT_FROM:-}"                          # EMPTY = cold start (no warm-start)
MASTER_PORT="${MASTER_PORT:-29572}"                # free; 29551/29561/29571 used, 29525=SLAM
OUT="${OUT:-/speech/tomson/filler_asr/train_sa_pe_lm_head/runs/full960_sa8_pe_fps25_lin_100ep_mask20-30x10_eos3_fw0.1}"
FORCE="${FORCE:-0}"
# -----------------------------------------------------------------------------

RUN_DIR="$(pwd)"
STAMP="$(date +%Y%m%d_%H%M%S)"
LOG="$RUN_DIR/logs/train_sa8_mask20-30_eos3_fw01_${STAMP}.log"
mkdir -p "$RUN_DIR/logs"

# ---- preflight: warm-start ckpt (only if INIT_FROM set) ---------------------
if [ -n "$INIT_FROM" ] && [ "$FORCE" != "1" ] && [ ! -f "$INIT_FROM" ]; then
  echo "ABORT: INIT_FROM set but not found: $INIT_FROM"; exit 1
fi

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
    echo "       wait, pick other GPUs (GPUS=...), or override with FORCE=1."
    exit 1
  fi
fi

# ---- preflight: don't clobber an existing run -------------------------------
if [ "$FORCE" != "1" ] && ls "$OUT"/*.pt >/dev/null 2>&1; then
  echo "ABORT: $OUT already contains checkpoints (fresh run would overwrite)."
  echo "       set OUT=... or FORCE=1 to proceed."
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
    EOS_REPEAT=$EOS_REPEAT FILL_WEIGHT=$FILL_WEIGHT \
    INIT_FROM=\"$INIT_FROM\" \
    OUT=\"$OUT\" MASTER_PORT=$MASTER_PORT \
    bash run.sh 2>&1 | tee \"$LOG\"
  echo \"=== training exited (rc=\${PIPESTATUS[0]}) ; pane kept open ===\"
  exec bash
'"

NG=$(awk -F, '{print NF}' <<<"$GPUS")
echo "launched detached in tmux session : $SESSION"
echo "  GPUs        : $GPUS   (effective batch = $BATCH*$ACCUM*$NG = $((BATCH*ACCUM*NG)))"
echo "  start       : $([ -z "$INIT_FROM" ] && echo COLD\ \(from\ scratch\) || echo warm:$INIT_FROM)"
echo "  epochs      : $EPOCHS"
echo "  changes     : mask U[$MASK_PROB,$MASK_PROB_MAX]x$MASK_LEN  +  </s>x$EOS_REPEAT  +  fill_weight=$FILL_WEIGHT   (lr 2e-4)"
echo "  out dir     : $OUT"
echo "  log         : $LOG"
echo
echo "  attach : tmux attach -t $SESSION      (detach: Ctrl-b then d)"
echo "  tail   : tail -f \"$LOG\""
echo "  wandb  : project 'filler_asr', run name ...mask0.2-0.3x10_eos3_fw0.1..."

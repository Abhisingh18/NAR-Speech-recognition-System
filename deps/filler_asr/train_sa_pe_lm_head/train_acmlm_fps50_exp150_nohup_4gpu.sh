#!/usr/bin/env bash
# =============================================================================
# A-CMLM SA-8 + PE + lm_head, LS-960 -- 4-GPU DDP, DETACHED (setsid; survives logout).
#
# Same config as the tmux launcher (train_acmlm_mask20-30_eos3_fw003_fps50_exp_150ep_4gpu.sh)
# but detaches with setsid instead of tmux and writes a pid file + log. Carries the reviewed
# fixes automatically -- they live in train.py / filler_sa_reference.py, not here:
#   #1 frozen HuBERT pinned to eval (no layerdrop/dropout)   #3 token-exact loss averaging
#   #4 step-accurate resume   #5 checkpoint saves all trainable params   eval: best.pt = min loss
#
#   run    : bash train_acmlm_fps50_exp150_nohup_4gpu.sh
#   watch  : tail -f <printed log>
#   resume : RESUME="$OUT/latest.pt" bash train_acmlm_fps50_exp150_nohup_4gpu.sh
#   stop   : kill -- -<printed pid>          (kills the whole process group)
#   decode : python filler_asr_omni_style_decoding.py --run_dir <OUT> --manifest <test.jsonl>
# =============================================================================
set -euo pipefail
cd "$(dirname "$0")"; RUN_DIR="$(pwd)"

# ============================ EDIT VALUES HERE ===============================
GPUS="${GPUS:-1,2,3,4}"            # <-- 4 FREE GPUs, PCI-BUS ids (check: nvidia-smi). DDP.
EPOCHS="${EPOCHS:-150}"
BATCH="${BATCH:-32}"               # per-GPU
ACCUM="${ACCUM:-2}"                # eff batch = BATCH*ACCUM*nGPU = 32*2*4 = 256
MASK_PROB="${MASK_PROB:-0.20}"     # audio-tap span masking, per-batch rate ~U[prob,max]
MASK_PROB_MAX="${MASK_PROB_MAX:-0.30}"
MASK_LEN="${MASK_LEN:-10}"         # mask span length in frames (10 = 200ms @ 50fps)
EOS_REPEAT="${EOS_REPEAT:-3}"      # emit </s> on N end frames
FILL_WEIGHT="${FILL_WEIGHT:-0.03}" # CE down-weight for <fill>
FPS="${FPS:-50}"                   # duration budget (50 -> n_keep=T -> grade all frames)
LR="${LR:-2e-4}"                   # peak LR
WARMUP="${WARMUP:-5000}"           # warmup steps
LR_SCHEDULE="${LR_SCHEDULE:-exponential}"    # linear | cosine | exponential
LR_EXP_FINAL_RATIO="${LR_EXP_FINAL_RATIO:-0.05}"  # exp floor = 5% of peak
CLIP="${CLIP:-1.0}"                # grad-norm clip (smoke confirmed non-binding past warmup)
SA_LAYERS="${SA_LAYERS:-8}"
MASTER_PORT="${MASTER_PORT:-29574}"          # any FREE port (see gpu17 loopback note)
OUT="${OUT:-/speech/tomson/filler_asr/train_sa_pe_lm_head/runs/full960_sa8_ACMLM_pe_fps50_mask20-30x10_eos3_fw0.03_exp_150ep}"
WANDB="${WANDB:-1}"                # 1 = log to wandb (needs `wandb login`); 0 = offline
DEBUG_STEPS="${DEBUG_STEPS:-5}"    # print the change-audits for the first N steps (0 = off)
RESUME="${RESUME:-}"              # set to $OUT/latest.pt to resume (uses the fixed step-accurate resume)
WANDB_RUN_ID="${WANDB_RUN_ID:-}"  # + WANDB_RESUME=allow to append to the same wandb curve on resume
WANDB_RESUME="${WANDB_RESUME:-}"
FORCE="${FORCE:-0}"                # 1 = skip the free-GPU / no-clobber preflights
# =============================================================================
# Paths/knobs that rarely change live in run.sh (edit THERE):
#   CKPT (hubert-xlarge ckpt), VOCAB_DIR, TRAIN_MANIFEST, DEV_MANIFEST, DEV_N (eval subset),
#   num_workers. Model-internal fixes live in train.py + filler_sa_reference.py.

command -v conda >/dev/null || { echo "ERROR: 'conda' not on PATH -> run 'conda activate filler_asr' (or source conda.sh) first"; exit 1; }
STAMP="$(date +%Y%m%d_%H%M%S)"
LOG="$RUN_DIR/logs/train_acmlm_fps50_exp150_nohup_${STAMP}.log"
PIDF="$RUN_DIR/logs/train_acmlm_fps50_exp150_nohup.pid"
mkdir -p "$RUN_DIR/logs"

# ---- preflight: requested GPUs free? (foreground, so an abort is VISIBLE) ----
if [ "$FORCE" != "1" ]; then
  busy=""
  for g in ${GPUS//,/ }; do
    used=$(CUDA_DEVICE_ORDER=PCI_BUS_ID nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$g" 2>/dev/null | tr -d ' ')
    [ -z "$used" ] && { echo "ERROR: GPU $g not found by nvidia-smi"; exit 1; }
    [ "$used" -gt 1000 ] && busy="$busy $g(${used}MiB)"
  done
  [ -n "$busy" ] && { echo "ABORT: requested GPUs busy:$busy   (pick others: GPUS=.. or FORCE=1)"; exit 1; }
fi
# ---- preflight: don't clobber an existing run (unless resuming) ----
if [ -z "$RESUME" ] && [ "$FORCE" != "1" ] && ls "$OUT"/*.pt >/dev/null 2>&1; then
  echo "ABORT: $OUT already has checkpoints. set OUT=.. , or RESUME=$OUT/latest.pt , or FORCE=1."; exit 1
fi
mkdir -p "$OUT"

[ -n "$WANDB_RUN_ID" ] && export WANDB_RUN_ID
[ -n "$WANDB_RESUME" ] && export WANDB_RESUME
NG=$(awk -F, '{print NF}' <<<"$GPUS")

# ---- launch detached: setsid -> new session, terminal SIGHUP can't reach it ----
# (nohup ALONE is not enough here: torchrun re-arms SIGHUP. setsid is the reliable tool.)
setsid env \
  ACMLM=1 GPU="$GPUS" SA_LAYERS=$SA_LAYERS EPOCHS=$EPOCHS BATCH=$BATCH ACCUM=$ACCUM \
  MASK_PROB=$MASK_PROB MASK_PROB_MAX=$MASK_PROB_MAX MASK_LEN=$MASK_LEN \
  EOS_REPEAT=$EOS_REPEAT FILL_WEIGHT=$FILL_WEIGHT \
  FPS=$FPS LR=$LR WARMUP=$WARMUP CLIP=$CLIP \
  LR_SCHEDULE=$LR_SCHEDULE LR_EXP_FINAL_RATIO=$LR_EXP_FINAL_RATIO \
  DEBUG_STEPS=$DEBUG_STEPS WANDB=$WANDB RESUME="$RESUME" \
  OUT="$OUT" MASTER_PORT=$MASTER_PORT \
  bash "$RUN_DIR/run.sh" > "$LOG" 2>&1 < /dev/null &
PID=$!; echo "$PID" > "$PIDF"

echo "launched A-CMLM fps50/exp/${EPOCHS}ep (setsid, detached)"
echo "  GPUs   : $GPUS   eff batch = $BATCH*$ACCUM*$NG = $((BATCH*ACCUM*NG))"
echo "  sched  : $LR_SCHEDULE (peak=$LR, final=$LR_EXP_FINAL_RATIO of peak, warmup=$WARMUP)"
echo "  out    : $OUT"
echo "  log    : $LOG"
echo "  pid    : $PID   ($PIDF)"
echo "  watch  : tail -f $LOG"
echo "  sanity : grep -E 'frozen-audit|ckpt-audit|select|grad-audit|eval @' $LOG   (first ~2 min)"
echo "  stop   : kill -- -$PID     # kills the whole process group"

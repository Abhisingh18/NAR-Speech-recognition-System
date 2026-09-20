#!/usr/bin/env bash
# =============================================================================
# SA-8 fine-tune, phase 2: FILL-WEIGHTED CE + GRADUAL UNFREEZING of the top
# HuBERT layer, warm-started from a converged head, with warmup->cosine-decay LR.
#
# What differs from train_sa8_100ep_6gpu.sh (phase-1, frozen backbone):
#   * --fill_weight 0.2         : down-weight the dominant <fill> class so the
#                                 gradient focuses on CONTENT frames.
#   * --unfreeze_top_hubert 1   : also train the top HuBERT encoder layer(s)
#                                 (+ final LayerNorm); conv extractor stays frozen.
#   * --hubert_lr 1e-5          : DISCRIMINATIVE LR -- gentle on the pretrained
#                                 backbone, normal (--lr 1e-4) on the SA head.
#   * --lr_schedule cosine      : warmup(1000) -> cosine decay.
#   * --init_from <best.pt>      : warm-start SA+lm_head weights, FRESH optimizer
#                                 (NOT --resume: old opt state is head-only).
#
# Isolated from the live job: invokes torchrun train.py DIRECTLY (not run.sh),
# writes to its OWN out dir, and refuses to start on busy GPUs or a non-empty out.
#
#   run   : GPUS=<free,gpus> bash train_sa8_fillw_unfreezetop.sh
#   watch : tmux attach -t sa8_fillw_unfreeze     (detach: Ctrl-b d)
# =============================================================================
set -euo pipefail
cd "$(dirname "$0")"
RUN_DIR="$(pwd)"

# ---- config (override via env) ----------------------------------------------
SESSION="${SESSION:-sa8_fillw_unfreeze}"
GPUS="${GPUS:-}"                                  # REQUIRED: comma list of FREE physical GPU indices
SA_LAYERS="${SA_LAYERS:-8}"
EPOCHS="${EPOCHS:-15}"                            # warm-start -> fewer epochs than a fresh run
BATCH="${BATCH:-16}"                             # smaller: unfrozen HuBERT layer retains activations
ACCUM="${ACCUM:-2}"
LR="${LR:-1e-4}"                                 # SA head / lm_head LR
HUBERT_LR="${HUBERT_LR:-1e-5}"                   # unfrozen HuBERT layer LR (gentle)
UNFREEZE_TOP="${UNFREEZE_TOP:-1}"               # # top HuBERT encoder layers to unfreeze
FILL_WEIGHT="${FILL_WEIGHT:-0.2}"
WARMUP="${WARMUP:-1000}"
LR_SCHEDULE="${LR_SCHEDULE:-cosine}"
INIT_FROM="${INIT_FROM:-$RUN_DIR/runs/full960_sa8_pe_fps25_lin/best.pt}"   # converged phase-1 head
CKPT="${CKPT:-/speech/tomson/filler_asr/models/hubert-xlarge-ls960-ft}"
OUT="${OUT:-$RUN_DIR/runs/full960_sa8_fillw${FILL_WEIGHT}_unfreezetop${UNFREEZE_TOP}}"
MASTER_PORT="${MASTER_PORT:-29561}"             # 29525=SLAM, 29551=live sa8 100ep -> avoid both
FORCE="${FORCE:-0}"
# -----------------------------------------------------------------------------

if [ -z "$GPUS" ]; then
  echo "ERROR: set GPUS to a comma list of FREE physical GPU indices, e.g. GPUS=1,4 bash $0"
  echo "       (check with: nvidia-smi --query-gpu=index,memory.used --format=csv)"
  exit 1
fi
if [ ! -f "$INIT_FROM" ]; then
  echo "ERROR: INIT_FROM checkpoint not found: $INIT_FROM"; exit 1
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
    echo "ABORT: requested GPUs busy:$busy  (pick others via GPUS=... or FORCE=1)"; exit 1
  fi
fi
if [ "$FORCE" != "1" ] && ls "$OUT"/*.pt >/dev/null 2>&1; then
  echo "ABORT: $OUT already has checkpoints (would overwrite). Set OUT=... or FORCE=1."; exit 1
fi
if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "tmux session '$SESSION' exists -> tmux attach -t $SESSION"; exit 1
fi
mkdir -p "$OUT" "$RUN_DIR/logs"
STAMP="$(date +%Y%m%d_%H%M%S)"
LOG="$RUN_DIR/logs/train_sa8_fillw_unfreeze_${STAMP}.log"
NGPU=$(awk -F, '{print NF}' <<< "$GPUS")

TRAIN_ARGS=(
  --model_checkpoint "$CKPT"
  --train_manifest /speech/tomson/exps/speech-recog/data/librispeech/librispeech_train_960h.jsonl
  --dev_manifest   /speech/tomson/exps/speech-recog/data/librispeech/librispeech_dev_clean.jsonl
  --out_dir "$OUT"
  --epochs "$EPOCHS" --batch_size "$BATCH" --grad_accum "$ACCUM"
  --lr "$LR" --hubert_lr "$HUBERT_LR" --unfreeze_top_hubert "$UNFREEZE_TOP"
  --fill_weight "$FILL_WEIGHT"
  --warmup "$WARMUP" --lr_schedule "$LR_SCHEDULE" --clip 1.0
  --num_sa_layers "$SA_LAYERS" --frames_per_sec 25
  --init_from "$INIT_FROM"
  --num_workers 8 --dev_n 500
  --eval_steps 2000 --save_steps 2000 --log_steps 50
  --dump_k 10 --sample_max_sec 10
  --pe --wandb
)

echo "launching phase-2 (fill-weight + unfreeze-top) detached in tmux: $SESSION"
echo "  GPUs=$GPUS  eff_batch=$((BATCH*ACCUM*NGPU))  init_from=$INIT_FROM"
echo "  out=$OUT"
echo "  log=$LOG    (attach: tmux attach -t $SESSION)"

tmux new-session -d -s "$SESSION" "bash -lc '
  source /speech/tomson/miniconda3/etc/profile.d/conda.sh
  conda activate filler_asr
  cd \"$RUN_DIR\"
  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=\"$GPUS\" NCCL_SOCKET_IFNAME=lo PYTHONUNBUFFERED=1 \
    torchrun --nproc_per_node=$NGPU --master_port=$MASTER_PORT train.py ${TRAIN_ARGS[*]} 2>&1 | tee \"$LOG\"
  echo \"=== training exited (rc=\${PIPESTATUS[0]}) ; pane kept open ===\"
  exec bash
'"

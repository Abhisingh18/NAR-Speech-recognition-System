#!/usr/bin/env bash
# Launch the filler-ASR FastAPI server (one model per process, chosen by RUN).
#
#   RUN=full960_sa8_pe_fps25_lin GPU=1 bash run_server.sh
#   RUN=full960_sa4_pe_fps25_lin GPU=1 PORT=8678 bash run_server.sh     # second depth on another port
#
# Background it with:  setsid bash run_server.sh > serve.log 2>&1 &     (setsid, not nohup —
# same SIGHUP re-arm gotcha as training). Kill by the PID printed below, never pkill -f.
#
# GPU is a single nvidia-smi index (PCI_BUS_ID order). GPU 1 is the free card on this box;
# GPU 0 is the flaky illegal-instruction one — avoid. Never touch the busy training GPUs.
set -euo pipefail
cd "$(dirname "$0")"

RUN="${RUN:?set RUN=<run dir name under ../runs/ or an absolute path>}"
GPU="${GPU:-6}"
PORT="${PORT:-8677}"
HOST="${HOST:-0.0.0.0}"
CKPT_NAME="${CKPT:-best.pt}"
MAX_SECONDS="${MAX_SECONDS:-60}"

# fail fast if the port is taken (BEFORE the ~1 min model load), and show the holder.
# A `Stat=T` holder means a Ctrl+Z-suspended server: kill <pid>; kill -CONT <pid>
if ss -tln 2>/dev/null | grep -q ":$PORT "; then
  echo "!! port $PORT already in use:" >&2
  ss -tlnp 2>/dev/null | grep ":$PORT " >&2
  echo "!! kill that PID (and 'kill -CONT <pid>' if it was Ctrl+Z-suspended), or relaunch with PORT=<other>" >&2
  exit 1
fi

case "$RUN" in
  /*) RUN_DIR="$RUN" ;;
  *)  RUN_DIR="/speech/tomson/filler_asr/train_sa_pe_lm_head/runs/$RUN" ;;
esac
[ -f "$RUN_DIR/$CKPT_NAME" ] || { echo "no checkpoint at $RUN_DIR/$CKPT_NAME" >&2; exit 1; }

echo ">> RUN_DIR=$RUN_DIR  GPU=$GPU  port=$PORT  ckpt=$CKPT_NAME  max=${MAX_SECONDS}s  pid=$$"

# --workers must stay 1: each worker would load its own copy of the model.
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="$GPU" PYTHONUNBUFFERED=1 \
RUN_DIR="$RUN_DIR" CKPT="$RUN_DIR/$CKPT_NAME" MAX_SECONDS="$MAX_SECONDS" \
  exec conda run --no-capture-output -n filler_asr \
  python -m uvicorn app:app --host "$HOST" --port "$PORT" --workers 1

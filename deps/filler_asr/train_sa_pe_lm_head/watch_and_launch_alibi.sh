#!/usr/bin/env bash
# Autonomous watcher: every INTERVAL seconds, try to launch the full ALiBi run on GPUs 1,2,3,4.
# The launcher (train_acmlm_alibi_fps50_fw02_exp150_4gpu_nohup.sh) has its OWN preflight that
# aborts cleanly if any of 1-4 is busy, so THAT is the authoritative "are they free?" gate here.
# We simply retry until it reports "launched", then exit. Safe to call repeatedly while busy
# (an aborted launch only touches empty dirs). Checks IMMEDIATELY first, then every 30 min.
set -u
cd /speech/tomson/filler_asr/train_sa_pe_lm_head
LAUNCHER="train_acmlm_alibi_fps50_fw02_exp150_4gpu_nohup.sh"
GPUS_WANT="${GPUS_WANT:-1,2,3,4}"
INTERVAL="${INTERVAL:-1800}"                     # 30 min

echo "[$(date '+%F %T')] watcher up: will launch $LAUNCHER on GPUs $GPUS_WANT as soon as they are free"
echo "[$(date '+%F %T')] poll interval = ${INTERVAL}s; first check is immediate"

while true; do
  out=$(GPUS="$GPUS_WANT" bash "$LAUNCHER" 2>&1)
  if echo "$out" | grep -q "launched A-CMLM ALiBi"; then
    echo "[$(date '+%F %T')] GPUs $GPUS_WANT FREE -> TRAINING LAUNCHED:"
    echo "$out" | sed 's/^/    /'
    echo "[$(date '+%F %T')] watcher finished (training now running detached)."
    break
  fi
  reason=$(echo "$out" | grep -E "ABORT|ERROR" | head -1)
  echo "[$(date '+%F %T')] not launched: ${reason:-<unknown>}  -> retry in ${INTERVAL}s"
  sleep "$INTERVAL"
done

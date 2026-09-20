#!/usr/bin/env bash
# =============================================================================
# Kannada + English (bilingual) A-CMLM on a FROZEN data2vec-aqc KANNADA-finetuned encoder.
# Thin wrapper over run.sh: only sets the Kannada defaults; the recipe (ALiBi / fw0.03 / fps50 /
# 8 SA / mask20-30x10 / eos3 / exp-LR / iter_wer) is unchanged from Tomson's Hindi run.
#
# Data must be prepared first (writes data_kn/ + vocab_kn/):
#   python3 prep_kn_manifests.py --workers 32
#
#   smoke  : SMOKE=1 GPU=6 bash run_kn.sh
#   real   : GPU=6,7,8 bash run_kn.sh 2>&1 | tee logs/train_kn_$(date +%Y%m%d_%H%M).log
#   resume : RESUME=runs/kn_.../latest.pt GPU=6,7,8 bash run_kn.sh
# Any run.sh knob can still be overridden from the env (EPOCHS, BATCH, LR, ...).
# =============================================================================
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"

export FILLER_ROOT="${FILLER_ROOT:-$HERE/deps/filler_asr}"
export SLAM_SRC="${SLAM_SRC:-$HERE/deps/SMEAR-MoE-ASR/src}"
export ENV="${ENV:-/speech/abhishek/miniconda3/envs/indic-nar-filler_asr}"   # abhishek's own env copy
export PYTHONNOUSERSITE=1                                                    # ignore ~/.local packages
export ENCODER_PT="${ENCODER_PT:-/speech/abhishek/kannada_data/encoders/SPRING_INX_data2vec_aqc_Kannada.pt}"
export VOCAB_DIR="${VOCAB_DIR:-$HERE/vocab_kn}"
export TRAIN_MANIFEST="${TRAIN_MANIFEST:-$HERE/data_kn/train.jsonl}"
export DEV_MANIFEST="${DEV_MANIFEST:-$HERE/data_kn/dev.jsonl}"
export OUT="${OUT:-$HERE/runs/kn_en_d2vaqcKN_sa8_ACMLM_alibi_fps50_fw003_exp}"
export TEXT_LANG=kn

# ~10x Hindi's data (~900 h): 150 epochs is infeasible live. ~2700 steps/epoch at eff batch 192
# -> 15 epochs ~ 40k optimizer steps (Hindi total was 33.7k). Dev is 37k clips -> eval a subset.
export EPOCHS="${EPOCHS:-15}"
export BATCH="${BATCH:-16}"; export ACCUM="${ACCUM:-4}"
export EVAL_STEPS="${EVAL_STEPS:-2000}"; export SAVE_STEPS="${SAVE_STEPS:-1000}"
export DEV_N="${DEV_N:-500}"; export EVAL_DECODE_N="${EVAL_DECODE_N:-200}"
export MASTER_PORT="${MASTER_PORT:-29630}"
export WANDB_PROJECT="${WANDB_PROJECT:-indic_nar_kn_en}"          # all Kannada runs + decodes + plots go here
export WANDB_NAME="${WANDB_NAME:-$(basename "$OUT")}"

for f in "$ENCODER_PT" "$VOCAB_DIR/vocab.json" "$TRAIN_MANIFEST" "$DEV_MANIFEST"; do
  [ -e "$f" ] || { echo "ERROR: missing $f  (run: python3 prep_kn_manifests.py)"; exit 1; }
done
# GPUs 6,7,8,9 by default (4 cards -> eff batch 16*4*4 = 256). Never grab a GPU someone else is using.
export GPU="${GPU:-6,7,8,9}"
for g in ${GPU//,/ }; do
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$g" 2>/dev/null | tr -d ' ')
  [ -n "$used" ] || { echo "ERROR: GPU $g not found"; exit 1; }
  [ "$used" -gt 2000 ] && { echo "ABORT: GPU $g busy (${used}MiB); pick another via GPU=..."; [ "${FORCE:-0}" = 1 ] || exit 1; }
done
if [ -z "${RESUME:-}" ] && [ "${SMOKE:-0}" != 1 ] && ls "$OUT"/*.pt >/dev/null 2>&1; then
  echo "ABORT: $OUT already has checkpoints. Use RESUME=$OUT/latest.pt or a new OUT=..."; exit 1
fi
# every run is logged to logs/ (plot_kn_run.py parses this file for the training/eval curves)
mkdir -p "$HERE/logs"
LOG="${LOG:-$HERE/logs/train_kn$([ "${SMOKE:-0}" = 1 ] && echo _smoke)_$(date +%Y%m%d_%H%M%S).log}"
echo "[run_kn] log -> $LOG"
bash "$HERE/run.sh" 2>&1 | tee "$LOG"

#!/usr/bin/env bash
# Training-inference decode on a FIXED (frozen) checkpoint, then sweep the hyp-stripping
# variants (eos / fillrun / collapse) and report WER/CER/SER per variant.
#
# This is the ONE-SHOT (single forward pass, frame-argmax, cut@first </s>) decode that
# train.py::evaluate() uses for the wandb hyp/ref table + eval/loss -- i.e. the N=1
# baseline for the omni-style step sweep (sweep_steps.sh). No step axis: training
# inference is single-pass, so the swept axis here is the stripping variant.
#
#   GPU=10 LIMIT=600 bash sweep_train_infer.sh
#   GPU=10 LIMIT=0   bash sweep_train_infer.sh              # LIMIT=0 -> full 2620
#   FILLRUN_K=3 RUN=runs/<other_acmlm_run> bash sweep_train_infer.sh
set -euo pipefail
cd "$(dirname "$0")"

RUN="${RUN:-runs/full960_sa8_ACMLM_pe_fps50_mask20-30x10_eos3_fw0.03_exp_150ep_tapcache_v2}"
CKPT="${CKPT:-$RUN/best.pt}"
MAN="${MAN:-/speech/tomson/exps/speech-recog/data/librispeech/librispeech_test_clean.jsonl}"
GPU="${GPU:-10}"; LIMIT="${LIMIT:-600}"; FILLRUN_K="${FILLRUN_K:-3}"
STAMP="$(date +%Y%m%d_%H%M%S)"
OUT="$RUN/train_infer_${STAMP}"; mkdir -p "$OUT"

# --- freeze the checkpoint so a concurrent training run can't change it under us ---
FROZEN="$OUT/frozen_ckpt.pt"; cp "$CKPT" "$FROZEN"
echo "[freeze] $CKPT -> $FROZEN   (LIMIT=$LIMIT utts, GPU=$GPU, fillrun_k=$FILLRUN_K)"

source /speech/tomson/miniconda3/etc/profile.d/conda.sh; conda activate filler_asr
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="$GPU" PYTHONUNBUFFERED=1 \
  python -u decode_train_infer.py --run_dir "$RUN" --ckpt "$FROZEN" --manifest "$MAN" \
    --out_dir "$OUT" --n "$LIMIT" --fillrun_k "$FILLRUN_K"

echo; echo "output dir: $OUT"

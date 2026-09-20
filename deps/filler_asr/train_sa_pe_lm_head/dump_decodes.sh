#!/usr/bin/env bash
# Dump per-utterance decode chains for a trained SA+PE+lm_head filler head, KEEPING the
# raw frame argmax before any special-token/<fill> stripping (the RAW line), e.g.
#   RAW : <s> m i s t e r | q u i l t e r | ... </s> <fill> <fill> <fill> ...
# Also writes NKEEP (raw cut at the duration ceiling), the cleaned HYP, REF and WER.
# Outputs land in the run dir: <set>.decodes.txt / <set>.hyp.txt / <set>.ref.txt / <set>.dump.summary
#
#   bash dump_decodes.sh                                            # full test-clean on GPU 2
#   N=10 bash dump_decodes.sh                                       # 10-utt smoke
#   MANIFEST=.../librispeech_test_other.jsonl bash dump_decodes.sh
#   RUN=full960_sa6_pe_fps25_lin GPU=3 bash dump_decodes.sh
#   OUT_DIR=<dir> bash dump_decodes.sh                               # outputs elsewhere (default: run dir)
#
# GPU is a single nvidia-smi index (PCI_BUS_ID order). GPU 0 is the flaky
# illegal-instruction one — avoid. Never touch the busy training GPUs.
set -euo pipefail
cd "$(dirname "$0")"

RUN="${RUN:-full960_sa8_pe_fps25_lin_100ep_bs32acc2}"
GPU="${GPU:-2}"
MANIFEST="${MANIFEST:-/speech/tomson/exps/speech-recog/data/librispeech/librispeech_test_clean.jsonl}"
CKPT_NAME="${CKPT:-best.pt}"
N="${N:-0}"                                      # 0 = whole manifest
FPS="${FPS:-25}"

case "$RUN" in
  /*) RUN_DIR="$RUN" ;;
  *)  RUN_DIR="/speech/tomson/filler_asr/train_sa_pe_lm_head/runs/$RUN" ;;
esac
[ -f "$RUN_DIR/$CKPT_NAME" ] || { echo "no checkpoint at $RUN_DIR/$CKPT_NAME" >&2; exit 1; }
OUT_DIR="${OUT_DIR:-$RUN_DIR}"
mkdir -p "$OUT_DIR"
TAG=$(basename "$MANIFEST" .jsonl)

echo ">> RUN_DIR=$RUN_DIR  GPU=$GPU  set=$TAG  ckpt=$CKPT_NAME  n=$N  fps=$FPS  out=$OUT_DIR"
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="$GPU" PYTHONUNBUFFERED=1 \
  conda run --no-capture-output -n filler_asr \
  python -u dump_decodes.py \
    --run_dir "$RUN_DIR" --ckpt "$RUN_DIR/$CKPT_NAME" \
    --manifest "$MANIFEST" --n "$N" --fps "$FPS" \
    --out_dir "$OUT_DIR"

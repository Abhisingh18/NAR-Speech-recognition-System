#!/bin/bash
# Standard GREEDY CTC decode of the off-the-shelf facebook/hubert-xlarge-ls960-ft
# baseline on the 3 English testsets (test_clean, seedtts prompts+pairs,
# l2arctic_full), sequentially on ONE GPU. This is the canonical HF model-card
# inference (argmax + processor.batch_decode); WER/CER/SER via jiwer. Per-set
# files land in $OUT_DIR as decode_<set>_greedy_{pred,gt,wer}.
#
# This is the RAW CTC baseline (model's own 32-char head), a reference point
# below the sa8_pe stage-2 encoder and the vicuna SLAM model.
#
# Run (detached, survives logout; test_clean+seedtts are quick, l2arctic is the
# bulk):
#   cd /speech/tomson/filler_asr
#   setsid bash run_ctc_beam_testsets.sh >> /speech/tomson/filler_asr/ctc_beam_testsets.log 2>&1 &
#
# Override GPU / batch / sets via env, e.g.:
#   CUDA_VISIBLE_DEVICES=4 SETS="test_clean" bash run_ctc_beam_testsets.sh

set -u
source /speech/tomson/miniconda3/etc/profile.d/conda.sh
conda activate filler_asr

export CUDA_DEVICE_ORDER=PCI_BUS_ID          # CUDA index == nvidia-smi index
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-6}
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4

cd /speech/tomson/filler_asr
BATCH=${BATCH:-8}
OUT_DIR=${OUT_DIR:-/speech/tomson/exps/speech-recog/exps/decode-hubert-xlarge-ctc-baseline-greedy-$(date +%Y%m%d_%H%M%S)}
mkdir -p "$OUT_DIR"

# set name -> jsonl
declare -A data=(
    [test_clean]=/speech/tomson/exps/speech-recog/data/librispeech/librispeech_test_clean.jsonl
    [seedtts_en_prompts]=/speech/tomson/filler_asr/data/testsets/en/seedtts_en_prompts.jsonl
    [seedtts_en_pairs]=/speech/tomson/filler_asr/data/testsets/en/seedtts_en_pairs.jsonl
    [l2arctic_full]=/speech/tomson/filler_asr/data/testsets/l2arctic_data/l2arctic_full.jsonl
)
sets=${SETS:-"test_clean seedtts_en_prompts seedtts_en_pairs l2arctic_full"}

echo "=== hubert-xlarge-ls960-ft CTC greedy baseline (jiwer) ==="
echo "out_dir: $OUT_DIR"
echo "gpu:     CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES   batch=$BATCH"
echo "sets:    $sets"

fail=0
for s in $sets; do
    jsonl=${data[$s]}
    if [ -z "${jsonl:-}" ] || [ ! -f "$jsonl" ]; then
        echo "unknown/missing set: $s ($jsonl) — skipping" >&2; fail=1; continue
    fi
    echo "=== decoding $s ($(wc -l < "$jsonl") utts, $(date +%H:%M:%S)) ==="
    python decode_ctc_greedy.py --jsonl "$jsonl" --set "$s" --out_dir "$OUT_DIR" \
        --batch "$BATCH" || { echo "FAILED $s" >&2; fail=1; }
done

echo "=== summary ($(date +%H:%M:%S)) ==="
for s in $sets; do
    f="$OUT_DIR/decode_${s}_greedy_wer"
    [ -s "$f" ] && echo "$s: $(grep -m1 '^%WER' "$f")" || echo "$s: NO RESULT"
done
exit $fail

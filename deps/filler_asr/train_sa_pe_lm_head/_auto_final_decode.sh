#!/usr/bin/env bash
# When the ALiBi training finishes, decode the FINAL (epoch-150) checkpoint on all 5 test sets with
# BOTH decoders, so we get a clean side-by-side:
#   (1) "real" 32-step no-remask   -> decode_iterative.py               (the default/production decode)
#   (2) "new" fillban remask       -> decode_iterative_refined.py       (--refine_rounds 4 --frac 0.15
#                                       --select conf --no_accept)       (the max-quality method)
# IMPORTANT: decodes latest.pt (= epoch 150), NOT best.pt (best.pt tracks the one-shot masked-LM val
# loss, which saturates ~epoch 69 and does NOT track iterative-decode WER).
# Robust: every number lands in files even if the interactive session ends. One free GPU, sequential.
set -uo pipefail
cd /speech/tomson/filler_asr/train_sa_pe_lm_head
OUT=runs/full960_sa8_ACMLM_alibi_fps50_mask20-30x10_eos3_fw0.2_exp_150ep
PID="${1:-3515902}"
WLOG="$OUT/AUTO_final_decode.log"
SNAP="$OUT/decode_final_ep150.pt"
SUMMARY="$OUT/FINAL_ep150_results.txt"

echo "[watch $(date)] waiting for training pid $PID (decodes latest.pt=ep150, both methods) ..." >> "$WLOG"
while kill -0 "$PID" 2>/dev/null; do sleep 300; done
echo "[watch $(date)] training finished. flushing 60s then decoding." >> "$WLOG"
sleep 60

cp -f "$OUT/latest.pt" "$SNAP"
STEP=$(conda run -n filler_asr python -c "import torch;ck=torch.load('$SNAP',map_location='cpu');print('step',ck.get('step'),'epoch',ck.get('epoch'))" 2>/dev/null)
echo "[watch $(date)] FINAL checkpoint (latest.pt) = $STEP" >> "$WLOG"

pick_gpu() { nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader,nounits 2>/dev/null \
  | awk -F, '{f=$3-$2; if(f>20000){print $1; exit}}'; }
GPU=$(pick_gpu); [ -z "$GPU" ] && GPU=0
echo "[watch $(date)] decoding on gpu $GPU" >> "$WLOG"

: > "$SUMMARY"
{
  echo "===== FINAL epoch-150 decode  (latest.pt = $STEP)  32 steps  $(date) ====="
  echo "## checkpoint: $SNAP"
  echo "## method A = real no-remask (decode_iterative.py)"
  echo "## method B = fillban remask (decode_iterative_refined.py  --refine_rounds 4 --frac 0.15 --select conf --no_accept)"
  echo
} >> "$SUMMARY"

run_one () {  # $1=script  $2=manifest  $3=outdir  $4=logfile  (extra decode args follow as $5..)
  local SCRIPT="$1" MAN="$2" ODIR="$3" LOG="$4"; shift 4
  : > "$LOG"
  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="$GPU" PYTHONUNBUFFERED=1 conda run --no-capture-output -n filler_asr \
    python -u "$SCRIPT" --run_dir "$OUT" --ckpt "$SNAP" --manifest "$MAN" --out_dir "$ODIR" "$@" >> "$LOG" 2>&1
}

decode_both () {  # $1=manifest  $2=setname
  local MAN="$1" NAME="$2"
  echo "----- $NAME  A:no-remask  $(date +%H:%M:%S) -----" >> "$WLOG"
  run_one decode_iterative.py "$MAN" "$OUT/finalep150_${NAME}_noremask" "$OUT/finaldec_${NAME}_A_noremask.log" \
    --steps 32 --n 0 --fps 0 --tau 0.1 --temp 1 --seed 0
  local A_DONE A_ERRS
  A_DONE=$(grep -hE "\[done\]" "$OUT/finaldec_${NAME}_A_noremask.log" | tail -1)
  A_ERRS=$(grep -hE "\[errs\]" "$OUT/finaldec_${NAME}_A_noremask.log" | tail -1)

  echo "----- $NAME  B:fillban    $(date +%H:%M:%S) -----" >> "$WLOG"
  run_one decode_iterative_refined.py "$MAN" "$OUT/finalep150_${NAME}_fillban" "$OUT/finaldec_${NAME}_B_fillban.log" \
    --steps 32 --refine_rounds 4 --frac_hi 0.15 --frac_lo 0.15 --select conf --no_accept --seed 0
  local B_DONE B_ERRS
  B_DONE=$(grep -hE "\[done\]" "$OUT/finaldec_${NAME}_B_fillban.log" | tail -1)
  B_ERRS=$(grep -hE "\[errs\]" "$OUT/finaldec_${NAME}_B_fillban.log" | tail -1)

  { echo "### $NAME"
    echo "  A no-remask : $A_DONE"
    echo "              : $A_ERRS"
    echo "  B fillban   : $B_DONE"
    echo "              : $B_ERRS"
    echo
  } >> "$SUMMARY"
  echo "[watch $(date)] $NAME done" >> "$WLOG"
}

decode_both /speech/tomson/filler_asr/data/test_clean.jsonl                                 test_clean
decode_both /speech/tomson/exps/speech-recog/data/librispeech/librispeech_test_other.jsonl  test_other
decode_both /speech/tomson/filler_asr/data/testsets/l2arctic_data/l2arctic_test.jsonl       l2arctic
decode_both /speech/tomson/filler_asr/data/testsets/en/seedtts_en_pairs.jsonl               seedtts_pairs
decode_both /speech/tomson/filler_asr/data/testsets/en/seedtts_en_prompts.jsonl             seedtts_prompts
echo "ALL FINAL DECODES DONE $(date)" >> "$SUMMARY"
echo "[watch $(date)] ALL DONE -> $SUMMARY" >> "$WLOG"

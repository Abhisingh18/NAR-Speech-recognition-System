#!/usr/bin/env bash
# Auto decode+by-duration-analysis of the ALiBi run at epoch milestones (50/75/100/150).
# Reads current step from the training log, waits for each milestone, snapshots latest.pt
# (so a mid-decode checkpoint overwrite can't corrupt the read), decodes test_clean (32 steps)
# on a genuinely-free GPU, and appends the report to logs/alibi_milestone_reports.txt.
set -u
cd /speech/tomson/filler_asr/train_sa_pe_lm_head
D=runs/full960_sa8_ACMLM_alibi_fps50_mask20-30x10_eos3_fw0.2_exp_150ep
MAN=/speech/tomson/exps/speech-recog/data/librispeech/librispeech_test_clean.jsonl
REPORT=logs/alibi_milestone_reports.txt
CONDA="conda run --no-capture-output -n filler_asr"
MILES=("54900:50ep" "82350:75ep" "109800:100ep" "164700:150ep")   # step_threshold:label

tl(){ ls -t logs/train_acmlm_alibi_fw02_exp150_*.log 2>/dev/null | head -1; }
cur_step(){ grep -aoE "step[0-9]+/164700" "$(tl)" 2>/dev/null | tail -1 | sed -E 's|step([0-9]+)/.*|\1|'; }
train_alive(){ pgrep -f "train.py.*pos_mode alibi" >/dev/null; }
free_gpu(){ nvidia-smi --query-gpu=index,memory.free,compute_mode --format=csv,noheader,nounits 2>/dev/null \
            | awk -F, '$2+0>16000 && $3 ~ /Default/ {print $1; exit}'; }

echo "[$(date '+%F %T')] milestone watcher up; targets: ${MILES[*]}" | tee -a "$REPORT"
for m in "${MILES[@]}"; do
  thr=${m%%:*}; label=${m##*:}
  # 1) wait for the milestone step (or training end)
  while :; do
    s=$(cur_step); s=${s:-0}
    [ "$s" -ge "$thr" ] && { echo "[$(date '+%F %T')] $label reached (step $s)" | tee -a "$REPORT"; break; }
    train_alive || { echo "[$(date '+%F %T')] training ended (step $s) before $label; decoding final state." | tee -a "$REPORT"; break; }
    sleep 600
  done
  # 2) grab a free GPU (retry up to ~3h; skip milestone if none)
  g=""; for t in $(seq 1 36); do g=$(free_gpu); [ -n "$g" ] && break; sleep 300; done
  [ -z "$g" ] && { echo "[$(date '+%F %T')] no free GPU for $label after ~3h; skipping." | tee -a "$REPORT"; continue; }
  # 3) snapshot latest.pt, decode, analyze
  snap="$D/_snap_${label}.pt"; cp -f "$D/latest.pt" "$snap"
  bs=$($CONDA python -c "import torch;print(torch.load('$snap',map_location='cpu').get('step'))" 2>/dev/null)
  odir="$D/testclean_alibi_${label}_step${bs}_steps32"
  echo "[$(date '+%F %T')] $label -> decoding latest.pt(step=$bs) on gpu$g -> $odir" | tee -a "$REPORT"
  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$g PYTHONUNBUFFERED=1 $CONDA \
    python -u decode_iterative.py --run_dir "$D" --ckpt "$snap" --manifest "$MAN" \
    --out_dir "$odir" --steps 32 --fps 0 --n 0 --tau 0.1 --temp 1 --seed 0 >>"$REPORT" 2>&1
  { echo "===================== $label (step $bs) BY-DURATION ====================="; \
    $CONDA python analyze_eos_by_duration.py --decode_dir "$odir"; \
    echo; } >>"$REPORT" 2>&1
  rm -f "$snap"
  echo "[$(date '+%F %T')] $label DONE" | tee -a "$REPORT"
done
echo "[$(date '+%F %T')] ALL MILESTONE DECODES COMPLETE" | tee -a "$REPORT"

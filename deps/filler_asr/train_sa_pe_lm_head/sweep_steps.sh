#!/usr/bin/env bash
# Sweep the iterative-decode step count on a FIXED (frozen) checkpoint and report WER/CER/SER.
# Freezes the ckpt once (so training saving a new best.pt mid-sweep can't confound the comparison),
# decodes the same utts at each step count, builds a REPORT per step, and prints a comparison table.
#
#   GPU=10 LIMIT=600 STEPS="16 32 50 64" bash sweep_steps.sh
#   GPU=10 LIMIT=0   STEPS="16 32 50 64" bash sweep_steps.sh     # LIMIT=0 -> full 2620
#   REMASK=1 ...                                                  # add Mask-Predict refinement
set -euo pipefail
cd "$(dirname "$0")"

RUN="${RUN:-runs/full960_sa8_ACMLM_pe_fps25_mask20-30x10_eos3_fw0.1}"
CKPT="${CKPT:-$RUN/best.pt}"
MAN="${MAN:-/speech/tomson/exps/speech-recog/data/librispeech/librispeech_test_clean.jsonl}"
GPU="${GPU:-10}"; LIMIT="${LIMIT:-600}"; STEPS="${STEPS:-16 32 50 64}"
TAU="${TAU:-0.1}"; TEMP="${TEMP:-5}"; SEED="${SEED:-0}"; REMASK="${REMASK:-0}"
STAMP="$(date +%Y%m%d_%H%M%S)"
SWEEP="$RUN/omni_sweep_${STAMP}"; mkdir -p "$SWEEP"
REMASK_FLAG=""; [ "$REMASK" = "1" ] && REMASK_FLAG="--remask"

# --- freeze the checkpoint so it can't change under us ---
FROZEN="$SWEEP/frozen_ckpt.pt"; cp "$CKPT" "$FROZEN"
echo "[freeze] $CKPT -> $FROZEN   (LIMIT=$LIMIT utts, STEPS=$STEPS, GPU=$GPU, remask=$REMASK)"

source /speech/tomson/miniconda3/etc/profile.d/conda.sh; conda activate filler_asr
for S in $STEPS; do
  OUT="$SWEEP/steps${S}"; mkdir -p "$OUT"
  echo "===== decoding steps=$S -> $OUT ====="
  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="$GPU" PYTHONUNBUFFERED=1 \
    python -u decode_report_test_clean.py --run_dir "$RUN" --ckpt "$FROZEN" --manifest "$MAN" \
      --out_dir "$OUT" --rank 0 --world 1 --limit "$LIMIT" \
      --steps "$S" --tau "$TAU" --temp "$TEMP" --seed "$SEED" $REMASK_FLAG
  python make_report.py --decode_dir "$OUT" --steps "$S" --tau "$TAU" --temp "$TEMP" --ckpt_name "$(basename "$CKPT")"
done

# --- comparison table across step counts ---
echo; echo "================ STEP SWEEP SUMMARY ================"
python - "$SWEEP" $STEPS <<'PY'
import sys, os, re, json, glob
sweep = sys.argv[1]; steps = sys.argv[2:]
def sdi(a,b):
    n,m=len(a),len(b); dp=[[0]*(m+1) for _ in range(n+1)]
    for i in range(n+1): dp[i][0]=i
    for j in range(m+1): dp[0][j]=j
    for i in range(1,n+1):
        for j in range(1,m+1):
            c=0 if a[i-1]==b[j-1] else 1
            dp[i][j]=min(dp[i-1][j]+1,dp[i][j-1]+1,dp[i-1][j-1]+c)
    i,j=n,m;S=D=I=0
    while i>0 or j>0:
        if i>0 and j>0 and a[i-1]==b[j-1] and dp[i][j]==dp[i-1][j-1]: i-=1;j-=1
        elif i>0 and j>0 and dp[i][j]==dp[i-1][j-1]+1: S+=1;i-=1;j-=1
        elif i>0 and dp[i][j]==dp[i-1][j]+1: D+=1;i-=1
        else: I+=1;j-=1
    return S,D,I
clean=lambda h:" ".join(re.sub(r'(<fill>)+',' ',h).split())
print(f"{'steps':>6} {'WER':>7} {'CERclean':>9} {'SER':>7} {'S/D/I':>10} {'no</s>':>6} {'utts':>6}")
for S in steps:
    rows=[json.loads(l) for l in open(os.path.join(sweep,f"steps{S}","all_utts.jsonl"))]
    we=wt=cc=ct=se=ne=0; wS=wD=wI=0
    for r in rows:
        we+=r['w_err']; wt+=r['w_tot']; se+=r['s_err']
        ne+= '</s>' not in r['full'].split()[:r['n_keep']]
        hc=clean(r['hyp'])
        s,d,i=sdi(r['ref'].split(),hc.split()); wS+=s;wD+=d;wI+=i
        # char err (cleaned) via simple DP
        a,b=r['ref'],hc; n,m=len(a),len(b); prev=list(range(m+1))
        for x in range(1,n+1):
            cur=[x]+[0]*m; ri=a[x-1]
            for y in range(1,m+1): cur[y]=min(prev[y]+1,cur[y-1]+1,prev[y-1]+(0 if ri==b[y-1] else 1))
            prev=cur
        cc+=prev[m]; ct+=len(a)
    E=wS+wD+wI
    print(f"{S:>6} {we/wt*100:6.2f}% {cc/ct*100:8.2f}% {se/len(rows)*100:6.1f}% "
          f"{wS/E*100:.0f}/{wD/E*100:.0f}/{wI/E*100:.0f}".rjust(0).ljust(0)
          + f"  {ne:>5}  {len(rows):>5}")
print(f"\nsweep dir: {sweep}")
PY

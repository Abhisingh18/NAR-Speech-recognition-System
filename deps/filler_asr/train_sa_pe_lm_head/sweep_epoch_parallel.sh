#!/usr/bin/env bash
# Iterative-decode step sweep for ONE checkpoint, run 3-at-once, fairly.
#
# "Fair"  -> the checkpoint is frozen once up front, so training saving a new best.pt
#            mid-sweep can't confound the comparison; every step count decodes the exact
#            same weights over the exact same utts.
# "3 at once" -> each step count runs on its own GPU in parallel (positional pairing with GPUS).
#
# Layout produced under the run dir:
#     <RUN>/epoch<N>/
#         frozen_ckpt.pt          <- the frozen weights all three share
#         steps16/  steps32/  steps64/   (REPORT.txt, all_utts.jsonl, gt_vs_pred.txt, shard_*.jsonl)
#         SUMMARY.txt             <- comparison table across the three step counts
# <N> is the epoch stored inside the checkpoint.
#
#   bash sweep_epoch_parallel.sh                                # all three on GPU 10 (default)
#   GPUS="10 0 2" bash sweep_epoch_parallel.sh                   # one GPU per step instead
#   STEPS="16 32 64" LIMIT=0 bash sweep_epoch_parallel.sh        # LIMIT=0 -> full 2620
#   CKPT=latest.pt bash sweep_epoch_parallel.sh                  # sweep latest instead of best
#   REMASK=1 bash sweep_epoch_parallel.sh                        # add Mask-Predict refinement
set -euo pipefail
cd "$(dirname "$0")"

RUN="${RUN:-runs/full960_sa8_ACMLM_pe_fps25_mask20-30x10_eos3_fw0.1}"
CKPT="${CKPT:-best.pt}"                                   # name under $RUN, or an absolute path
MAN="${MAN:-/speech/tomson/exps/speech-recog/data/librispeech/librispeech_test_clean.jsonl}"
STEPS="${STEPS:-16 32 64}"
GPUS="${GPUS:-10}"                                       # one index -> all steps share it; list 3 -> one each
LIMIT="${LIMIT:-0}"                                       # 0 = full test-clean (2620)
TAU="${TAU:-0.1}"; TEMP="${TEMP:-5}"; SEED="${SEED:-0}"; REMASK="${REMASK:-0}"
REMASK_FLAG=""; [ "$REMASK" = "1" ] && REMASK_FLAG="--remask"

case "$CKPT" in /*) CKPT_PATH="$CKPT" ;; *) CKPT_PATH="$RUN/$CKPT" ;; esac
[ -f "$CKPT_PATH" ] || { echo "no checkpoint at $CKPT_PATH" >&2; exit 1; }

source /speech/tomson/miniconda3/etc/profile.d/conda.sh; conda activate filler_asr

# --- epoch number straight from the checkpoint ---
EPOCH="$(python - "$CKPT_PATH" <<'PY'
import sys, torch
ck = torch.load(sys.argv[1], map_location="cpu")
print(int(ck.get("epoch", -1)))
PY
)"
EPDIR="$RUN/epoch${EPOCH}"; mkdir -p "$EPDIR"

# --- freeze the checkpoint so it can't change under us ---
FROZEN="$EPDIR/frozen_ckpt.pt"; cp "$CKPT_PATH" "$FROZEN"

read -r -a STEP_ARR <<< "$STEPS"
read -r -a GPU_ARR  <<< "$GPUS"
echo "[sweep] ckpt=$CKPT_PATH  epoch=$EPOCH  frozen=$FROZEN"
echo "[sweep] LIMIT=$LIMIT  tau=$TAU  temp=$TEMP  seed=$SEED  remask=$REMASK"
echo "[sweep] out=$EPDIR"

# --- launch every step count in parallel, one per GPU (cycles GPUS if fewer than STEPS) ---
declare -a PIDS OUTS
for i in "${!STEP_ARR[@]}"; do
  S="${STEP_ARR[$i]}"
  G="${GPU_ARR[$(( i % ${#GPU_ARR[@]} ))]}"
  OUT="$EPDIR/steps${S}"; mkdir -p "$OUT"
  echo "  -> steps=$S on GPU $G  ($OUT)"
  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="$G" PYTHONUNBUFFERED=1 \
    python -u decode_report_test_clean.py --run_dir "$RUN" --ckpt "$FROZEN" --manifest "$MAN" \
      --out_dir "$OUT" --rank 0 --world 1 --limit "$LIMIT" \
      --steps "$S" --tau "$TAU" --temp "$TEMP" --seed "$SEED" $REMASK_FLAG \
      > "$OUT/decode.log" 2>&1 &
  PIDS[$i]=$!; OUTS[$i]="$OUT"
done

# --- wait for all three; report any failure but let the rest finish ---
FAIL=0
for i in "${!PIDS[@]}"; do
  if wait "${PIDS[$i]}"; then
    echo "[done] steps=${STEP_ARR[$i]}  (${OUTS[$i]})"
  else
    echo "[FAIL] steps=${STEP_ARR[$i]}  — see ${OUTS[$i]}/decode.log" >&2; FAIL=1
  fi
done

# --- per-step reports (fast, sequential) ---
for i in "${!STEP_ARR[@]}"; do
  OUT="${OUTS[$i]}"
  [ -n "$(ls "$OUT"/shard_*.jsonl 2>/dev/null)" ] || continue
  python make_report.py --decode_dir "$OUT" --steps "${STEP_ARR[$i]}" \
    --tau "$TAU" --temp "$TEMP" --ckpt_name "$(basename "$CKPT_PATH") (epoch $EPOCH)"
done

# --- comparison table across step counts ---
echo; echo "================ EPOCH $EPOCH STEP SWEEP SUMMARY ================" | tee "$EPDIR/SUMMARY.txt"
python - "$EPDIR" "${STEP_ARR[@]}" <<'PY' | tee -a "$EPDIR/SUMMARY.txt"
import sys, os, re, json
epdir = sys.argv[1]; steps = sys.argv[2:]
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
    p=os.path.join(epdir,f"steps{S}","all_utts.jsonl")
    if not os.path.exists(p): print(f"{S:>6}   (no all_utts.jsonl — decode failed?)"); continue
    rows=[json.loads(l) for l in open(p)]
    we=wt=cc=ct=se=ne=0; wS=wD=wI=0
    for r in rows:
        we+=r['w_err']; wt+=r['w_tot']; se+=r['s_err']
        ne+= '</s>' not in r['full'].split()[:r['n_keep']]
        hc=clean(r['hyp'])
        s,d,i=sdi(r['ref'].split(),hc.split()); wS+=s;wD+=d;wI+=i
        a,b=r['ref'],hc; n,m=len(a),len(b); prev=list(range(m+1))
        for x in range(1,n+1):
            cur=[x]+[0]*m; ri=a[x-1]
            for y in range(1,m+1): cur[y]=min(prev[y]+1,cur[y-1]+1,prev[y-1]+(0 if ri==b[y-1] else 1))
            prev=cur
        cc+=prev[m]; ct+=len(a)
    E=max(wS+wD+wI,1)
    print(f"{S:>6} {we/wt*100:6.2f}% {cc/ct*100:8.2f}% {se/len(rows)*100:6.1f}% "
          f"{wS/E*100:.0f}/{wD/E*100:.0f}/{wI/E*100:.0f}".ljust(11)
          + f"  {ne:>5}  {len(rows):>5}")
print(f"\nepoch dir: {epdir}")
PY
[ "$FAIL" = "0" ] || { echo "one or more decodes failed — check the steps*/decode.log files" >&2; exit 1; }

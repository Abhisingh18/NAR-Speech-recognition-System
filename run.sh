#!/usr/bin/env bash
# =============================================================================
# indic-nar-filler_asr : A-CMLM head on a FROZEN data2vec-aqc, Hindi.
# Recipe = filler_asr's ALiBi / fill_weight=0.03 / fps=50 / 8 SA / mask20-30x10 /
#          eos_repeat=3 / exp-decay LR / iter_wer selection.  LIVE encoder (no tap cache).
#
#   1 GPU : GPU=6 bash run.sh
#   4 GPU : GPU=1,2,3,4 bash run.sh              (eff batch = BATCH*ACCUM*nGPU)
#   smoke : SMOKE=1 GPU=6 bash run.sh            (tiny subset, few steps, no wandb)
# Override any knob via env (ENCODER_PT, TRAIN_MANIFEST, OUT, EPOCHS, BATCH, ...).
# =============================================================================
set -euo pipefail
# INDIC_ROOT = where train.py + the python modules live (default: this script's dir = tomson's).
# Another user can run THIS script in-place and only override OUT/data/env — nothing is written here.
INDIC_ROOT="${INDIC_ROOT:-$(cd "$(dirname "$0")" && pwd)}"
cd "$INDIC_ROOT"
# shared deps resolve from the filler_asr trees; slam_llm from SMEAR; this dir last.
# Portable: point these at your copies (defaults = tomson's trees). model_d2v_acmlm.py /
# text_norm_hi.py read FILLER_ROOT / SLAM_SRC from the env too, so setting them here is enough.
FILLER_ROOT="${FILLER_ROOT:-$INDIC_ROOT/deps/filler_asr}"        # local copy: filler_sa_reference.py (+ train_sa_pe_lm_head/)
SLAM_SRC="${SLAM_SRC:-$INDIC_ROOT/deps/SMEAR-MoE-ASR/src}"        # local copy: slam_llm/models/encoder.py (data2vec loader)
export FILLER_ROOT SLAM_SRC
export PYTHONPATH="${FILLER_ROOT}:${FILLER_ROOT}/train_sa_pe_lm_head:${SLAM_SRC}:$(pwd)${PYTHONPATH:+:$PYTHONPATH}"
export TOKENIZERS_PARALLELISM=false
# reduce allocator fragmentation (one long clip pads the whole batch -> big transient spikes)
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

GPU="${GPU:-6}"
ENV="${ENV:-/speech/abhishek/miniconda3/envs/indic-nar-filler_asr}"   # conda env PREFIX path (abhishek's own copy) or NAME
CONDA_SEL=$([ "${ENV:0:1}" = "/" ] && echo "-p" || echo "-n")   # -p for a prefix path, -n for a name
export PYTHONNOUSERSITE=1                                      # ignore ~/.local site-packages (conflicting torchvision)
# defaults = abhishek's copy, Kannada+English (run_kn.sh sets the same values explicitly)
ENCODER_PT="${ENCODER_PT:-/speech/abhishek/kannada_data/encoders/SPRING_INX_data2vec_aqc_Kannada.pt}"
VOCAB_DIR="${VOCAB_DIR:-$INDIC_ROOT/vocab_kn}"
TRAIN_MANIFEST="${TRAIN_MANIFEST:-$INDIC_ROOT/data_kn/train.jsonl}"
DEV_MANIFEST="${DEV_MANIFEST:-$INDIC_ROOT/data_kn/dev.jsonl}"
OUT="${OUT:-$INDIC_ROOT/runs/kn_en_d2vaqcKN_sa8_ACMLM_alibi_fps50_fw003_exp}"
TEXT_LANG="${TEXT_LANG:-kn}"

# ---- recipe knobs (match train_acmlm_alibi_fps50_fw003_exp150) ----
SA_LAYERS="${SA_LAYERS:-8}"; EPOCHS="${EPOCHS:-150}"; BATCH="${BATCH:-32}"; ACCUM="${ACCUM:-2}"
MASK_PROB="${MASK_PROB:-0.20}"; MASK_PROB_MAX="${MASK_PROB_MAX:-0.30}"; MASK_LEN="${MASK_LEN:-10}"
EOS_REPEAT="${EOS_REPEAT:-3}"; FILL_WEIGHT="${FILL_WEIGHT:-0.03}"; FPS="${FPS:-50}"
LR="${LR:-2e-4}"; WARMUP="${WARMUP:-5000}"; LR_SCHEDULE="${LR_SCHEDULE:-exponential}"; LR_EXP_FINAL_RATIO="${LR_EXP_FINAL_RATIO:-0.01}"
POS_MODE="${POS_MODE:-alibi}"; SELECT_METRIC="${SELECT_METRIC:-iter_wer}"
EVAL_DECODE_STEPS="${EVAL_DECODE_STEPS:-32}"; EVAL_DECODE_N="${EVAL_DECODE_N:-200}"
EVAL_STEPS="${EVAL_STEPS:-5000}"; SAVE_STEPS="${SAVE_STEPS:-5000}"; DEV_N="${DEV_N:-355}"
WANDB="${WANDB:-1}"; MASTER_PORT="${MASTER_PORT:-29590}"; TRAIN_N="${TRAIN_N:-0}"; MAX_STEPS="${MAX_STEPS:-0}"
TAP_CACHE="${TAP_CACHE:-}"   # set to a precomputed cache dir (train/ + dev/) to skip the live encoder
# TEXT_LANG (set above, default kn): target normalizer  hi = normalize_hi (Devanagari) | kn = normalize_kn (Kannada+English)
RESUME="${RESUME:-}"         # path to latest.pt to resume (optimizer/scheduler/step restored)

# ---- smoke overrides: tiny + fast + offline (FORCE-assign; the vars are already set above,
#      so ${VAR:-x} would be a no-op -- use plain '=' to actually cap the smoke) ----
if [ "${SMOKE:-0}" = "1" ]; then
  OUT="${OUT}_smoke"; EPOCHS=3; TRAIN_N=256; DEV_N=32; BATCH=4; ACCUM=1
  EVAL_STEPS=20; SAVE_STEPS=20; MAX_STEPS=40; EVAL_DECODE_N=8; WANDB=0; WARMUP=10
  echo "[smoke] tiny run (256 train / 40 steps / no wandb) -> $OUT"
fi

EXTRA=()
[ "$FILL_WEIGHT" != "" ] && EXTRA+=(--fill_weight "$FILL_WEIGHT")
[ "$MASK_PROB" != "0" ] && EXTRA+=(--mask_prob "$MASK_PROB" --mask_prob_max "$MASK_PROB_MAX" --mask_length "$MASK_LEN")
[ "$LR_SCHEDULE" = "exponential" ] && EXTRA+=(--lr_exp_final_ratio "$LR_EXP_FINAL_RATIO")
[ "$POS_MODE" != "sinusoidal" ] && EXTRA+=(--pos_mode "$POS_MODE")
[ "$EVAL_DECODE_STEPS" != "0" ] && EXTRA+=(--eval_decode_steps "$EVAL_DECODE_STEPS" --eval_decode_n "$EVAL_DECODE_N")
[ -n "$SELECT_METRIC" ] && EXTRA+=(--select_metric "$SELECT_METRIC")
[ "$WANDB" = "1" ] && EXTRA+=(--wandb)
[ "$TRAIN_N" != "0" ] && EXTRA+=(--train_n "$TRAIN_N")
[ "$MAX_STEPS" != "0" ] && EXTRA+=(--max_steps "$MAX_STEPS")
[ -n "$TAP_CACHE" ] && EXTRA+=(--tap_cache "$TAP_CACHE")   # cached data2vec taps (no live encoder)
EXTRA+=(--lang "$TEXT_LANG")
if [ -n "$RESUME" ]; then
  [ -f "$RESUME" ] || { echo "ERROR: RESUME=$RESUME not found"; exit 1; }
  EXTRA+=(--resume "$RESUME")
fi

ARGS=(
  --acmlm --pe
  --model_checkpoint "$ENCODER_PT" --vocab_dir "$VOCAB_DIR"
  --train_manifest "$TRAIN_MANIFEST" --dev_manifest "$DEV_MANIFEST" --out_dir "$OUT"
  --epochs "$EPOCHS" --batch_size "$BATCH" --grad_accum "$ACCUM"
  --lr "$LR" --warmup "$WARMUP" --clip "${CLIP:-1.0}" --lr_schedule "$LR_SCHEDULE"
  --num_sa_layers "$SA_LAYERS" --frames_per_sec "$FPS" --eos_repeat "$EOS_REPEAT"
  --num_workers "${NUM_WORKERS:-8}" --dev_n "$DEV_N"
  --eval_steps "$EVAL_STEPS" --save_steps "$SAVE_STEPS" --log_steps "${LOG_STEPS:-25}"
  --dump_k 6 --sample_max_sec 10
  "${EXTRA[@]}"
)

NGPU=$(awk -F, '{print NF}' <<< "$GPU")
echo "[run] env=$ENV GPUs=$GPU (n=$NGPU)  eff_batch=$((BATCH*ACCUM*NGPU))  out=$OUT"
echo "[run] encoder=$ENCODER_PT"
# ENV as a PREFIX path -> call that env's binaries directly (no `conda` needed on PATH; e.g. user
# abhishek has no conda in non-interactive shells). ENV as a NAME -> `conda run` as before.
if [ "${ENV:0:1}" = "/" ]; then
  [ -x "$ENV/bin/python" ] || { echo "ERROR: $ENV/bin/python not found"; exit 1; }
  export PATH="$ENV/bin:$PATH"
  RUNNER=()
else
  RUNNER=(conda run --no-capture-output $CONDA_SEL "$ENV")
fi
if [ "${DRY_RUN:-0}" = "1" ]; then            # print exactly what would run, launch nothing
  printf '[dry-run] CUDA_VISIBLE_DEVICES=%s %s ' "$GPU" "${RUNNER[*]:-}"
  [ "$NGPU" -gt 1 ] && printf 'torchrun --nproc_per_node=%s --master_port=%s ' "$NGPU" "$MASTER_PORT" || printf 'python -u '
  printf '%q ' train.py "${ARGS[@]}"; echo
  echo "[dry-run] python=$(command -v python)  PYTHONPATH=$PYTHONPATH  WANDB_PROJECT=${WANDB_PROJECT:-} WANDB_NAME=${WANDB_NAME:-}"
  exit 0
fi
if [ "$NGPU" -gt 1 ]; then
  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="$GPU" NCCL_SOCKET_IFNAME=lo PYTHONUNBUFFERED=1 \
    "${RUNNER[@]}" torchrun --nproc_per_node="$NGPU" --master_port="$MASTER_PORT" train.py "${ARGS[@]}"
else
  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="$GPU" PYTHONUNBUFFERED=1 \
    "${RUNNER[@]}" python -u train.py "${ARGS[@]}"
fi

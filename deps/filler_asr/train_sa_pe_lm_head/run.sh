#!/usr/bin/env bash
# Full-LibriSpeech SA + PE + lm_head training on a FROZEN HuBERT-xlarge.
# No caching: the encoder runs a forward pass every step.
#
# Single OR multi-GPU. GPU is a comma list of PCI-BUS indices; >1 GPU → DDP via torchrun.
#   1 GPU : GPU=6 bash run.sh
#   2 GPU : GPU=4,6 bash run.sh           (effective batch = BATCH*ACCUM*nGPU)
#   other ckpt: CKPT=/speech/tomson/filler_asr/models/hubert-xlarge-ll60k bash run.sh
#
# To run detached and keep a log + pid (preferred on this shared box):
#   cd /speech/tomson/filler_asr/train_sa_pe_lm_head
#   nohup bash run.sh > train_full960.log 2>&1 &
#   echo $! > run.pid ; tail -f train_full960.log
set -euo pipefail
cd "$(dirname "$0")"
# filler_sa_reference.py lives in the parent dir; put it on the import path explicitly so
# train.py resolves it no matter how this is launched (local modules still win: script dir is first).
export PYTHONPATH="/speech/tomson/filler_asr${PYTHONPATH:+:$PYTHONPATH}"

GPU="${GPU:-4,6}"                                  # PCI-BUS index(es); comma list → multi-GPU DDP
CKPT="${CKPT:-/speech/tomson/filler_asr/models/hubert-xlarge-ls960-ft}"
# dir holding vocab.json for the char tokenizer (train.py's old default path was deleted)
VOCAB_DIR="${VOCAB_DIR:-/speech/tomson/filler_asr/train_sa_pe_lm_head/runs/full960_sa8_pe_fps25_lin_100ep_mask20-30x10_eos3_fw0.1}"
# data manifests (jsonl: {"audio"|"source", "text"|"target"} per line) — override on a new server
TRAIN_MANIFEST="${TRAIN_MANIFEST:-/speech/tomson/exps/speech-recog/data/librispeech/librispeech_train_960h.jsonl}"
DEV_MANIFEST="${DEV_MANIFEST:-/speech/tomson/exps/speech-recog/data/librispeech/librispeech_dev_clean.jsonl}"
SA_LAYERS="${SA_LAYERS:-4}"                        # # self-attention head layers; sweep e.g. 4/6/8
# OUT defaults to a per-SA_LAYERS dir so different depths don't clobber each other's checkpoints.
OUT="${OUT:-/speech/tomson/filler_asr/train_sa_pe_lm_head/runs/full960_sa${SA_LAYERS}_pe_fps25_lin}"
EPOCHS="${EPOCHS:-30}"
BATCH="${BATCH:-32}"
ACCUM="${ACCUM:-4}"                              # per-GPU; eff batch = BATCH*ACCUM*nGPU
RESUME="${RESUME:-}"                             # path to latest.pt to resume
MASK_PROB="${MASK_PROB:-0}"                      # train-time tap masking before SA (0 = off)
MASK_PROB_MAX="${MASK_PROB_MAX:-0}"              # > MASK_PROB → per-batch rate U[prob,max]
MASK_LEN="${MASK_LEN:-10}"                       # masked span length in frames (10 = 200ms)
INIT_FROM="${INIT_FROM:-}"                       # warm-start head weights (fresh optimizer/schedule)
EOS_REPEAT="${EOS_REPEAT:-1}"                    # emit </s> on N consecutive end frames (1 = original; 3 = boundary-reinforced)
FILL_WEIGHT="${FILL_WEIGHT:-}"                    # CE class weight for <fill> (empty = unweighted; e.g. 0.1 down-weights <fill>)
FPS="${FPS:-25}"                                 # duration budget frames/sec: 25 = ceil(dur*25); 50 = grade all HuBERT frames (n_keep=T)
LR="${LR:-2e-4}"                                 # peak LR
WARMUP="${WARMUP:-5000}"                          # warmup steps
LR_SCHEDULE="${LR_SCHEDULE:-linear}"             # linear | cosine | exponential
LR_EXP_FINAL_RATIO="${LR_EXP_FINAL_RATIO:-0.01}" # exponential only: final LR as fraction of peak
MASTER_PORT="${MASTER_PORT:-29532}"             # pick a free port (gpu17 multi-proc note)
DEV_N="${DEV_N:-500}"                            # # dev utts for eval (0 = full dev set)
TAP_CACHE="${TAP_CACHE:-}"                        # dir of precomputed HuBERT taps (train/ + dev/); skips encoder
POS_MODE="${POS_MODE:-sinusoidal}"               # SA-head positional scheme: sinusoidal (default) | alibi | rope
ROPE_THETA="${ROPE_THETA:-10000}"                # rope only: base θ (500000 = Llama-3-style long-context)
ROPE_ORIG_LEN="${ROPE_ORIG_LEN:-1024}"           # rope only: dynamic-NTK reference length (frames; ~20s at 50fps)
ROPE_NTK_FACTOR="${ROPE_NTK_FACTOR:-1.0}"        # rope only: STATIC NTK stretch (used only if ROPE_NTK_DYNAMIC=0)
ROPE_NTK_DYNAMIC="${ROPE_NTK_DYNAMIC:-1}"        # rope only: 1 = dynamic NTK (default); 0 = static (uses factor)
EVAL_DECODE_STEPS="${EVAL_DECODE_STEPS:-0}"       # >0 = also run N-step iterative mask-predict decode at eval (A-CMLM)
EVAL_DECODE_N="${EVAL_DECODE_N:-200}"             # cap on #dev clips iteratively decoded per eval (0 = all)
MAX_STEPS="${MAX_STEPS:-0}"                       # 0 = run full EPOCHS; else cap optimizer steps (smoke)
TRAIN_N="${TRAIN_N:-0}"                           # 0 = full train manifest; else first N utts (smoke)
WANDB="${WANDB:-1}"                              # 1 = log to wandb (needs login); 0 = no wandb (smoke/offline)
WANDB_RUN_ID="${WANDB_RUN_ID:-}"                # set to an existing wandb run id to APPEND to its curve
WANDB_RESUME="${WANDB_RESUME:-}"               # 'must' (fail if missing) / 'allow' — resume that run's history
# wandb reads these from the env at wandb.init(); export only when set so a fresh run still gets a new id.
[ -n "$WANDB_RUN_ID" ] && export WANDB_RUN_ID
[ -n "$WANDB_RESUME" ] && export WANDB_RESUME

EXTRA=()
[ -n "$RESUME" ] && EXTRA+=(--resume "$RESUME")
[ -n "$INIT_FROM" ] && EXTRA+=(--init_from "$INIT_FROM")
[ "$MASK_PROB" != "0" ] && EXTRA+=(--mask_prob "$MASK_PROB" --mask_prob_max "$MASK_PROB_MAX" --mask_length "$MASK_LEN")
[ -n "$FILL_WEIGHT" ] && EXTRA+=(--fill_weight "$FILL_WEIGHT")
[ "${ACMLM:-0}" = "1" ] && EXTRA+=(--acmlm)     # audio-conditioned masked-LM (A-CMLM) mode
[ "$LR_SCHEDULE" = "exponential" ] && EXTRA+=(--lr_exp_final_ratio "$LR_EXP_FINAL_RATIO")
[ "$MAX_STEPS" != "0" ] && EXTRA+=(--max_steps "$MAX_STEPS")   # smoke: cap steps
[ "$TRAIN_N"  != "0" ] && EXTRA+=(--train_n "$TRAIN_N")        # smoke: subset train
[ "$WANDB" = "1" ] && EXTRA+=(--wandb)                          # off -> no wandb login needed
[ -n "$TAP_CACHE" ] && EXTRA+=(--tap_cache "$TAP_CACHE")        # cached-tap training (no encoder forward)
[ "$POS_MODE" != "sinusoidal" ] && EXTRA+=(--pos_mode "$POS_MODE")               # opt-in ALiBi/RoPE (default = unchanged)
if [ "$POS_MODE" = "rope" ]; then                                                 # rope-only NTK knobs
  EXTRA+=(--rope_theta "$ROPE_THETA" --rope_orig_len "$ROPE_ORIG_LEN" --rope_ntk_factor "$ROPE_NTK_FACTOR")
  [ "$ROPE_NTK_DYNAMIC" = "0" ] && EXTRA+=(--no_rope_ntk_dynamic)                 # else dynamic (train.py default)
fi
[ "$EVAL_DECODE_STEPS" != "0" ] && EXTRA+=(--eval_decode_steps "$EVAL_DECODE_STEPS" --eval_decode_n "$EVAL_DECODE_N")
[ -n "${SELECT_METRIC:-}" ] && EXTRA+=(--select_metric "$SELECT_METRIC")   # loss (default) | iter_wer

ARGS=(
  --model_checkpoint "$CKPT"
  --vocab_dir "$VOCAB_DIR"
  --train_manifest "$TRAIN_MANIFEST"
  --dev_manifest   "$DEV_MANIFEST"
  --out_dir "$OUT"
  --epochs "$EPOCHS" --batch_size "$BATCH" --grad_accum "$ACCUM"
  --lr "$LR" --warmup "$WARMUP" --clip "${CLIP:-1.0}" --lr_schedule "$LR_SCHEDULE"
  --num_sa_layers "$SA_LAYERS" --frames_per_sec "$FPS" --eos_repeat "$EOS_REPEAT"
  --num_workers 8
  --dev_n "$DEV_N"
  --eval_steps "${EVAL_STEPS:-2000}" --save_steps "${SAVE_STEPS:-2000}" --log_steps "${LOG_STEPS:-50}"
  --debug_steps "${DEBUG_STEPS:-5}"
  --dump_k 10 --sample_max_sec 10
  --pe
  "${EXTRA[@]}"
)

NGPU=$(awk -F, '{print NF}' <<< "$GPU")

if [ "$NGPU" -gt 1 ]; then
  # multi-GPU: DDP via torchrun. NCCL_SOCKET_IFNAME=lo is required on gpu17 (loopback note).
  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="$GPU" NCCL_SOCKET_IFNAME=lo PYTHONUNBUFFERED=1 \
    conda run --no-capture-output -n filler_asr \
    torchrun --nproc_per_node="$NGPU" --master_port="$MASTER_PORT" train.py "${ARGS[@]}"
else
  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="$GPU" \
    conda run --no-capture-output -n filler_asr \
    python -u train.py "${ARGS[@]}"
fi

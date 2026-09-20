# Command reference — current run

**Run:** `full960_sa8_ACMLM_pe_fps50_mask20-30x10_eos3_fw0.03_exp_150ep_tapcache_v2`
Tapcached A-CMLM, SA-8 + PE + lm_head on frozen HuBERT-xlarge, LS-960, fps50,
mask 20–30% × 10, eos_repeat 3, fill_weight 0.03, exponential LR, 150 ep.

Training launches on 4-GPU DDP (default GPUs `1,2,3,4`, port 29576).
Checkpoints: `runs/.../tapcache_v2/{best.pt, latest.pt}`.

---

## 0. Setup (once per shell)

```bash
cd /speech/tomson/filler_asr/train_sa_pe_lm_head
source /speech/tomson/miniconda3/etc/profile.d/conda.sh && conda activate filler_asr

CODE=/speech/tomson/filler_asr/train_sa_pe_lm_head
PY=/speech/tomson/miniconda3/envs/filler_asr/bin/python
RUN=$CODE/runs/full960_sa8_ACMLM_pe_fps50_mask20-30x10_eos3_fw0.03_exp_150ep_tapcache_v2
TAP_CACHE=/speech/tomson/filler_asr/data/tapcache
LS=/speech/tomson/exps/speech-recog/data/librispeech
TS=/speech/tomson/filler_asr/data/testsets
```

### Data manifests (drop into any `--manifest` / `MAN=`)

| name | path |
|---|---|
| train 960h | `$LS/librispeech_train_960h.jsonl` |
| dev-clean | `$LS/librispeech_dev_clean.jsonl` |
| dev-other | `$LS/librispeech_dev_other.jsonl` |
| test-clean | `$LS/librispeech_test_clean.jsonl` |
| test-other | `$LS/librispeech_test_other.jsonl` |
| test-clean smoke 50 / 5 | `$LS/librispeech_test_clean_50.jsonl` · `$LS/librispeech_test_clean_5.jsonl` |
| l2arctic test | `$TS/l2arctic_data/l2arctic_test.jsonl` |
| l2arctic full | `$TS/l2arctic_data/l2arctic_full.jsonl` |
| seedtts pairs | `$TS/en/seedtts_en_pairs.jsonl` |
| seedtts prompts | `$TS/en/seedtts_en_prompts.jsonl` |

---

## 1. Data prep — HuBERT tap cache (one-time; needed for tapcached training)

```bash
# full build (train/ + dev/) on 4 GPUs, ~1h. A split with a PREPARED marker is skipped.
GPUS=1,2,3,4 bash extract_taps.sh

# smoke (512 clips, separate cache dir)
CACHE=/speech/tomson/filler_asr/data/tapcache_smoke LIMIT=512 GPUS=0 bash extract_taps.sh

# other GPUs / cache location
GPUS=6,7,8,9 CACHE=$TAP_CACHE bash extract_taps.sh
```

---

## 2. Training

> The `train_acmlm_*` wrappers **self-detach** via `setsid` (survives logout) and write
> their own `logs/*.log` + `.pid`; `nohup` is not strictly needed on those. Raw-`nohup`
> forms via `run.sh` are in §2c. Wrappers abort if the GPUs are busy or `OUT` already has
> checkpoints — override with `FORCE=1` (careful: clobbers) or point `OUT`/`RESUME` elsewhere.

### 2a. Tapcached A-CMLM (current recipe) — fast, no encoder forward per step
```bash
# 4-GPU DDP (the exact current run)
GPUS=1,2,3,4 bash train_acmlm_fps50_exp150_tapcached_4gpu.sh

# other free GPUs + free port + fresh OUT
GPUS=6,7,8,9 MASTER_PORT=29586 \
  OUT=$CODE/runs/full960_sa8_ACMLM_tapcache_v3 \
  bash train_acmlm_fps50_exp150_tapcached_4gpu.sh

# 1-GPU (bump batch — cache frees encoder memory)
GPUS=0 BATCH=64 ACCUM=4 MASTER_PORT=29590 \
  OUT=$CODE/runs/full960_sa8_ACMLM_tapcache_1gpu \
  bash train_acmlm_fps50_exp150_tapcached_4gpu.sh

# 2-GPU
GPUS=0,5 ACCUM=2 MASTER_PORT=29591 \
  OUT=$CODE/runs/full960_sa8_ACMLM_tapcache_2gpu \
  bash train_acmlm_fps50_exp150_tapcached_4gpu.sh

# resume (step-accurate) + append to same wandb curve
RESUME=$RUN/latest.pt WANDB_RUN_ID=<id> WANDB_RESUME=allow \
  GPUS=1,2,3,4 bash train_acmlm_fps50_exp150_tapcached_4gpu.sh
```

### 2b. Live-encoder A-CMLM (no tap cache — HuBERT forward every step)
```bash
# 4-GPU DDP
GPUS=1,2,3,4 bash train_acmlm_fps50_exp150_nohup_4gpu.sh

# 1-GPU
GPUS=0 BATCH=16 ACCUM=8 MASTER_PORT=29592 \
  OUT=$CODE/runs/full960_sa8_ACMLM_live_1gpu \
  bash train_acmlm_fps50_exp150_nohup_4gpu.sh
```

### 2c. Raw `nohup` form (generic `run.sh` — full env control, any GPU count)
`GPU` is a comma list of PCI-BUS ids; >1 GPU → torchrun DDP automatically.
```bash
# 1-GPU
GPU=0 ACMLM=1 SA_LAYERS=8 FPS=50 EPOCHS=150 BATCH=64 ACCUM=4 \
  MASK_PROB=0.20 MASK_PROB_MAX=0.30 MASK_LEN=10 EOS_REPEAT=3 FILL_WEIGHT=0.03 \
  LR=2e-4 WARMUP=5000 LR_SCHEDULE=exponential LR_EXP_FINAL_RATIO=0.05 \
  TAP_CACHE=$TAP_CACHE OUT=$CODE/runs/my_1gpu_run \
  nohup bash run.sh > $CODE/logs/train_1gpu_$(date +%m%d_%H%M).log 2>&1 &
echo $! > $CODE/logs/train_1gpu.pid

# multi-GPU DDP
GPU=6,7,8,9 MASTER_PORT=29588 ACMLM=1 SA_LAYERS=8 FPS=50 EPOCHS=150 \
  BATCH=64 ACCUM=1 TAP_CACHE=$TAP_CACHE OUT=$CODE/runs/my_4gpu_run \
  nohup bash run.sh > $CODE/logs/train_4gpu_$(date +%m%d_%H%M).log 2>&1 &
echo $! > $CODE/logs/train_4gpu.pid
```

### 2d. Watch / stop
```bash
tail -f $(ls -t $CODE/logs/train_*.log | head -1)
grep -E 'frozen-audit|ckpt-audit|eval @' $CODE/logs/train_*.log   # sanity, first ~2 min
kill -- -$(cat $CODE/logs/train_acmlm_fps50_exp150_tapcache.pid)  # kills whole process group
```

---

## 3. Decoding — iterative (steps=32), the main decode

Params matched to the best `iter_steps32_best` runs: `--steps 32 --tau 0.1 --temp 1.0
--fill_penalty 0.0`; fps auto-read from ckpt (=50). One GPU each via `CUDA_VISIBLE_DEVICES`.
Uses `best.pt` by default.

```bash
# generic one-liner: iter-decode any manifest on any GPU, with nohup
dec () {  # usage: dec <GPU> <manifest> <out_subdir>
  local g=$1 man=$2 sub=$3 out=$RUN/$sub
  mkdir -p "$out"
  CUDA_VISIBLE_DEVICES=$g CUDA_DEVICE_ORDER=PCI_BUS_ID \
    nohup "$PY" decode_iterative.py --run_dir "$RUN" --manifest "$man" --out_dir "$out" \
      --steps 32 --tau 0.1 --temp 1.0 --fill_penalty 0.0 \
      > "$out/decode.nohup" 2>&1 &
  echo "$! -> $out/decode.nohup"
}
```

All datasets (change the leading GPU id as GPUs free up):
```bash
dec 0 $LS/librispeech_test_clean.jsonl        iter_steps32_best_test_clean
dec 0 $LS/librispeech_test_other.jsonl        iter_steps32_best_test_other
dec 0 $LS/librispeech_dev_clean.jsonl         iter_steps32_best_dev_clean
dec 0 $LS/librispeech_dev_other.jsonl         iter_steps32_best_dev_other
dec 0 $TS/l2arctic_data/l2arctic_test.jsonl   iter_steps32_best_l2arctic
dec 0 $TS/l2arctic_data/l2arctic_full.jsonl   iter_steps32_best_l2arctic_full
dec 0 $TS/en/seedtts_en_pairs.jsonl           iter_steps32_best_seedtts_pairs
dec 0 $TS/en/seedtts_en_prompts.jsonl         iter_steps32_best_seedtts_prompts
```

Prebuilt bundle scripts (each self-`cd`s and `tee`s its own log):
```bash
# GPU 0: l2arctic + both seedtts, sequential
nohup bash decode_gpu0_arctic_seedtts.sh > gpu0_arctic_seedtts.nohup 2>&1 &
# librispeech test_other (edit CUDA_VISIBLE_DEVICES inside the script for the GPU)
nohup bash decode_gpu10_test_other.sh    > gpu10_test_other.nohup    2>&1 &
```

Variants (append to any `decode_iterative.py` call):
```bash
--ckpt $RUN/latest.pt     # decode latest instead of best
--steps 1                 # one-shot (== training inference)
--steps 8|16|64           # step-count sweep
--temp 5.0                # more exploratory position sampling
--fill_penalty 0.3        # suppress <fill> over-emission
--remask                  # extra Mask-Predict refinement rounds
--n 600                   # subset (0 = whole manifest)
--fps 25                  # override; 0 = read from ckpt
```

---

## 4. Decoding — one-shot "training inference" sweep (N=1 baseline + hyp-strip variants)

Freezes the ckpt first (safe while training writes checkpoints), then reports
WER/CER/SER per stripping variant (eos / fillrun / collapse).

```bash
GPU=0 LIMIT=600 bash sweep_train_infer.sh                                   # test-clean subset
GPU=0 LIMIT=0   bash sweep_train_infer.sh                                   # full test-clean (2620)
GPU=0 LIMIT=0   MAN=$LS/librispeech_test_other.jsonl bash sweep_train_infer.sh
GPU=0 FILLRUN_K=3 RUN=runs/<other_acmlm_run>         bash sweep_train_infer.sh
```

---

## Notes / gotchas

- **Don't** launch new training onto GPUs `1,2,3,4` — that's the live `tapcache_v2` run.
- Pick a **free `MASTER_PORT`** per concurrent DDP job (29576 is in use).
- `NCCL_SOCKET_IFNAME=lo` is set by the wrappers (required on this box's loopback).
- Check free GPUs before launching:
  ```bash
  CUDA_DEVICE_ORDER=PCI_BUS_ID nvidia-smi \
    --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
  ```

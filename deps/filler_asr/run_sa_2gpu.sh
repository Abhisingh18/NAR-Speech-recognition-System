#!/bin/bash
# SA (FillerHubertSAModel) training on the PRECOMPUTED Arrow cache, 2 GPUs.
# LR = 2e-3, linear warmup(1000)->decay on ALL trainable params (SA + lm_head;
# classifier_only_train_ratio=0, no head-only curriculum). HuBERT-xlarge frozen.
# GPUs: the two RTX 6000 Ada at nvidia-smi indices 4 and 6 (T1000 at 5 is skipped).
# Run run_prepare.sh first so the cache exists.
#
# Launch (background, logged):
#   nohup bash run_sa_2gpu.sh > SA_NEW_LOGS.txt 2>&1 &
cd /speech/tomson/filler_asr
export CUDA_DEVICE_ORDER=PCI_BUS_ID         # CUDA indices == nvidia-smi indices
export CUDA_VISIBLE_DEVICES=4,6             # two RTX 6000 Ada (skip T1000 at 5)
export TOKENIZERS_PARALLELISM=false
export HF_DATASETS_CACHE=/speech/tomson/filler_asr/data/hf_datasets_cache
export WANDB_PROJECT=filler_asr
exec /speech/tomson/miniconda3/envs/filler_asr/bin/torchrun \
  --nproc_per_node=2 --master_port=29718 \
  train_cached.py --config_path config_xlarge_sa_2gpu.json

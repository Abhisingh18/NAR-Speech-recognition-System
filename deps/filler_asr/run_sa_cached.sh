#!/bin/bash
# Launch the SA (FillerHubertSAModel) training on the PRECOMPUTED Arrow cache.
# Run run_prepare.sh first. Inside tmux:
#   tmux new -s sa_cached  ->  bash run_sa_cached.sh 2>&1 | tee train_sa_cached.log
cd /speech/tomson/filler_asr
export CUDA_DEVICE_ORDER=PCI_BUS_ID        # make CUDA indices match nvidia-smi (T1000 offset)
export CUDA_VISIBLE_DEVICES=2,3,4,6        # 4 RTX cards (skip the T1000 at slot 5)
export TOKENIZERS_PARALLELISM=false
export HF_DATASETS_CACHE=/speech/tomson/filler_asr/data/hf_datasets_cache
export WANDB_PROJECT=filler_asr
exec /speech/tomson/miniconda3/envs/filler_asr/bin/torchrun \
  --nproc_per_node=4 --master_port=29717 \
  train_cached.py --config_path config_xlarge_sa_cached.json

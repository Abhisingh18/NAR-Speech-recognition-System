#!/bin/bash
# SA (FillerHubertSAModel) training, fresh experiment:
#   * frames_per_sec=50  -> grade ~all HuBERT frames (n_keep >= T, only pad is -100)
#   * fill_weight=0.1    -> down-weight the <fill> majority class in framewise CE
#   * 150 epochs, exponential LR decay (warmup 1000 -> geometric decay to 1% of peak)
# Reuses the existing PREPARED Arrow cache (frames_per_sec is applied in the collator,
# not baked into the cache, so no re-prepare needed). Run run_prepare.sh only if the
# cache is missing. Inside tmux:
#   tmux new -s sa_fill50  ->  bash run_sa_fill50_exp.sh 2>&1 | tee train_sa_fill50_exp.log
cd /speech/tomson/filler_asr
export CUDA_DEVICE_ORDER=PCI_BUS_ID        # make CUDA indices match nvidia-smi (T1000 offset)
export CUDA_VISIBLE_DEVICES=2,3,4,6        # 4 RTX cards (skip the T1000 at slot 5)
export TOKENIZERS_PARALLELISM=false
export HF_DATASETS_CACHE=/speech/tomson/filler_asr/data/hf_datasets_cache
export WANDB_PROJECT=filler_asr
exec /speech/tomson/miniconda3/envs/filler_asr/bin/torchrun \
  --nproc_per_node=4 --master_port=29719 \
  train_cached.py --config_path config_xlarge_sa_fill50_exp.json

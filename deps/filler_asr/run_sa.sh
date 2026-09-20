#!/bin/bash
# Launch the SA (FillerHubertSAModel) 20-epoch training on GPU 2,3,4,6.
# Run inside tmux:  tmux new -s sa  ->  bash run_sa.sh 2>&1 | tee train_sa.log
cd /speech/tomson/filler_asr
export CUDA_DEVICE_ORDER=PCI_BUS_ID        # make CUDA indices match nvidia-smi (T1000 offset)
export CUDA_VISIBLE_DEVICES=2,3,4,6        # 4 RTX cards (skip the T1000 at slot 5)
export TOKENIZERS_PARALLELISM=false
export WANDB_PROJECT=filler_asr
exec /speech/tomson/miniconda3/envs/filler_asr/bin/torchrun \
  --nproc_per_node=4 --master_port=29716 \
  train_local.py --config_path config_xlarge_sa_full.json

#!/bin/bash
# ONE-TIME data prep: precompute feature-extracted input_values + framewise <fill>
# labels into the Arrow cache (~220GB) from LOCAL wavs. Heavy/CPU-bound; run in tmux.
#   tmux new -s prep  ->  bash run_prepare.sh 2>&1 | tee prepare.log
# No GPU needed. Safe to run alongside an existing GPU training (CPU/disk only).
cd /speech/tomson/filler_asr
export TOKENIZERS_PARALLELISM=false
# Keep any HF datasets temp/cache on the big /speech FS, never on a small $HOME.
export HF_DATASETS_CACHE=/speech/tomson/filler_asr/data/hf_datasets_cache
exec /speech/tomson/miniconda3/envs/filler_asr/bin/python \
  prepare_data_local.py --config_path config_xlarge_sa_cached.json

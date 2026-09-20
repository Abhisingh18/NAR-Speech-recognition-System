#!/bin/bash

# Activate the correct conda environment
# Adjust the path to conda.sh if it's installed differently on your system
if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
    . "$HOME/miniconda3/etc/profile.d/conda.sh"
elif [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
    . "$HOME/anaconda3/etc/profile.d/conda.sh"
else
    echo "Warning: Could not automatically source conda.sh. Ensure you are running this in the asr_filler environment."
fi

conda activate asr_filler

# Parse config to get number of processes (fallback to 1 if not found or jq not installed)
NUM_PROC=1
if command -v jq &> /dev/null; then
    NUM_PROC=$(jq '.num_processes // 1' config.json)
fi

echo "Starting training with $NUM_PROC processes using torchrun..."

PORT=29501
torchrun \
    --nproc_per_node=$NUM_PROC \
    --master_port=$PORT \
    train.py --config_path config.json

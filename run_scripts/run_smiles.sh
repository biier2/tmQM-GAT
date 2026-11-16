#!/bin/bash
# Run tmQM_Fe.ipynb in the background (SMILES GAT, 5070 Ti / Linux)
# Usage: bash run_smiles.sh
cd "$(dirname "$0")"

# Linux GPU optimization environment variables
export CUDA_LAUNCH_BLOCKING=0
export OMP_NUM_THREADS=8
export TORCH_CUDNN_V8_API_ENABLED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

nohup jupyter nbconvert --to notebook --execute --inplace \
  --ExecutePreprocessor.timeout=-1 \
  tmQM_Fe.ipynb > train_smiles.log 2>&1 &
echo "Started SMILES GAT training (PID=$!). tail -f train_smiles.log to check progress"

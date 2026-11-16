#!/bin/bash
# Run tmQM_Fe_XYZ.ipynb in the background (5070 Ti / Linux)
# Usage: bash run.sh
cd "$(dirname "$0")"

# Linux GPU optimization environment variables
export CUDA_LAUNCH_BLOCKING=0
export OMP_NUM_THREADS=8
export TORCH_CUDNN_V8_API_ENABLED=1

nohup jupyter nbconvert --to notebook --execute --inplace \
  --ExecutePreprocessor.timeout=-1 \
  tmQM_Fe_XYZ.ipynb > train.log 2>&1 &
echo "Started XYZ training (PID=$!). tail -f train.log to check progress"

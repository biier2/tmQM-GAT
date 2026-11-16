#!/bin/bash
# Run scorer.ipynb in the background (5070 Ti / Linux)
# Usage: bash run_scorer.sh
cd "$(dirname "$0")"

# Linux GPU optimization environment variables
export CUDA_LAUNCH_BLOCKING=0

nohup jupyter nbconvert --to notebook --execute --inplace \
  --ExecutePreprocessor.timeout=-1 \
  scorer.ipynb > score.log 2>&1 &
echo "Started scoring (PID=$!). tail -f score.log to check results"

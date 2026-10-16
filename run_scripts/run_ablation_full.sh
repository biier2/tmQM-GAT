#!/bin/bash
cd "$(dirname "$0")"
nohup jupyter nbconvert --to notebook --execute --inplace \
  --ExecutePreprocessor.timeout=-1 \
  ablation_full.ipynb > ablation_full.log 2>&1 &
echo "Started full ablation (PID=$!). tail -f ablation_full.log"

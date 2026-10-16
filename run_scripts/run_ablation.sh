#!/bin/bash
cd "$(dirname "$0")"
nohup jupyter nbconvert --to notebook --execute --inplace \
  --ExecutePreprocessor.timeout=-1 \
  ablation.ipynb > ablation.log 2>&1 &
echo "Started ablation (PID=$!). tail -f ablation.log"

#!/bin/bash
cd "$(dirname "$0")"
nohup jupyter nbconvert --to notebook --execute --inplace \
  --ExecutePreprocessor.timeout=-1 \
  benchmark_comparison_v3.ipynb > benchmark_v3.log 2>&1 &
echo "Started benchmark v3 (PID=$!). tail -f benchmark_v3.log"

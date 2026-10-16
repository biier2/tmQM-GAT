#!/bin/bash
cd "$(dirname "$0")"
nohup jupyter nbconvert --to notebook --execute --inplace \
  --ExecutePreprocessor.timeout=-1 \
  benchmark_comparison.ipynb > benchmark.log 2>&1 &
echo "Started benchmark (PID=$!). tail -f benchmark.log"

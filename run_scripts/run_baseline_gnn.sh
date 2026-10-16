#!/bin/bash
cd "$(dirname "$0")"
nohup jupyter nbconvert --to notebook --execute --inplace \
  --ExecutePreprocessor.timeout=-1 \
  baseline_gnn.ipynb > baseline_gnn.log 2>&1 &
echo "Started GNN baselines (PID=$!). tail -f baseline_gnn.log"

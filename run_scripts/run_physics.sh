#!/bin/bash
cd "$(dirname "$0")"
nohup jupyter nbconvert --to notebook --execute --inplace \
  --ExecutePreprocessor.timeout=-1 \
  physics_consistency.ipynb > physics.log 2>&1 &
echo "Started Physics Consistency (PID=$!). tail -f physics.log"

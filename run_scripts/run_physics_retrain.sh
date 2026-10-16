#!/bin/bash
cd "$(dirname "$0")"
nohup jupyter nbconvert --to notebook --execute --inplace \
  --ExecutePreprocessor.timeout=-1 \
  physics_retrain.ipynb > physics_retrain.log 2>&1 &
echo "Started Physics Retraining (PID=$!). tail -f physics_retrain.log"

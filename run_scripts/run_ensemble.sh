#!/bin/bash
cd "$(dirname "$0")"
nohup jupyter nbconvert --to notebook --execute --inplace \
  --ExecutePreprocessor.timeout=-1 \
  ensemble_cv.ipynb > ensemble.log 2>&1 &
echo "Started Ensemble CV (PID=$!). tail -f ensemble.log"

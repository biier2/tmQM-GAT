#!/bin/bash
cd "$(dirname "$0")"
nohup jupyter nbconvert --to notebook --execute --inplace \
  --ExecutePreprocessor.timeout=-1 \
  ensemble_cv_weighted.ipynb > ensemble_cv_weighted.log 2>&1 &
echo "Started ensemble CV weighted (PID=$!). tail -f ensemble_cv_weighted.log"

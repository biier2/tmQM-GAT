#!/bin/bash
cd "$(dirname "$0")"
nohup jupyter nbconvert --to notebook --execute --inplace \
  --ExecutePreprocessor.timeout=-1 \
  external_validation.ipynb > external_val.log 2>&1 &
echo "Started External Validation (PID=$!). tail -f external_val.log"

#!/bin/bash
cd "$(dirname "$0")"
nohup jupyter nbconvert --to notebook --execute --inplace \
  --ExecutePreprocessor.timeout=-1 \
  tmQM_SchNet.ipynb > schnet.log 2>&1 &
echo "Started SchNet training (PID=$!). tail -f schnet.log"

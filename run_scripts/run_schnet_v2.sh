#!/bin/bash
cd "$(dirname "$0")"
nohup jupyter nbconvert --to notebook --execute --inplace \
  --ExecutePreprocessor.timeout=-1 \
  tmQM_SchNet_v2.ipynb > schnet_v2.log 2>&1 &
echo "Started SchNet v2 (PID=$!). tail -f schnet_v2.log"

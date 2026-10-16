#!/bin/bash
cd "$(dirname "$0")"
nohup jupyter nbconvert --to notebook --execute --inplace \
  --ExecutePreprocessor.timeout=-1 \
  CN_Control.ipynb > cn_control.log 2>&1 &
echo "Started CN control (PID=$!). tail -f cn_control.log"

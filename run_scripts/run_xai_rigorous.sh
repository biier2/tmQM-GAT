#!/bin/bash
cd "$(dirname "$0")"
nohup jupyter nbconvert --to notebook --execute --inplace \
  --ExecutePreprocessor.timeout=-1 \
  xai_rigorous.ipynb > xai_rigorous.log 2>&1 &
echo "Started XAI Rigorous (PID=$!). tail -f xai_rigorous.log"

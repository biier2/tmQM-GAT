#!/bin/bash
cd "$(dirname "$0")"
nohup jupyter nbconvert --to notebook --execute --inplace \
  --ExecutePreprocessor.timeout=-1 \
  xai_attention.ipynb > xai.log 2>&1 &
echo "Started XAI analysis (PID=$!). tail -f xai.log"

#!/bin/bash
cd "$(dirname "$0")"
nohup jupyter nbconvert --to notebook --execute --inplace \
  --ExecutePreprocessor.timeout=-1 \
  scorer_3d.ipynb > scorer_3d.log 2>&1 &
echo "Started scorer 3D (PID=$!). tail -f scorer_3d.log"

#!/bin/bash
cd "$(dirname "$0")"
nohup jupyter nbconvert --to notebook --execute --inplace \
  --ExecutePreprocessor.timeout=-1 \
  3D_GCN.ipynb > 3dgcn.log 2>&1 &
echo "Started 3D GCN (PID=$!). tail -f 3dgcn.log"

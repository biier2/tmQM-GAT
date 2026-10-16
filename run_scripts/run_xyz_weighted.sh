#!/bin/bash
cd "$(dirname "$0")"
nohup jupyter nbconvert --to notebook --execute --inplace \
  --ExecutePreprocessor.timeout=-1 \
  tmQM_Fe_XYZ_weighted.ipynb > xyz_weighted.log 2>&1 &
echo "Started XYZ GAT weighted training (PID=$!). tail -f xyz_weighted.log"

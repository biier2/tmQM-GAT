#!/bin/bash
cd "$(dirname "$0")"
nohup jupyter nbconvert --to notebook --execute --inplace \
  --ExecutePreprocessor.timeout=-1 \
  tmQM_Fe_smiles_cv.ipynb > smiles_cv.log 2>&1 &
echo "Started SMILES GAT CV training (PID=$!). tail -f smiles_cv.log"

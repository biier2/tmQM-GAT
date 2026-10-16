#!/bin/bash
cd "$(dirname "$0")"
nohup jupyter nbconvert --to notebook --execute --inplace \
  --ExecutePreprocessor.timeout=-1 \
  bostmc_validation.ipynb > bostmc.log 2>&1 &
echo "Started BOSTMC validation (PID=$!). tail -f bostmc.log"

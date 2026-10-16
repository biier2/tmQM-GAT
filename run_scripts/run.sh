#!/bin/bash
# 后台跑 tmQM_Fe_XYZ.ipynb (5070 Ti / Linux)
# 用法: bash run.sh
cd "$(dirname "$0")"

# Linux GPU 优化环境变量
export CUDA_LAUNCH_BLOCKING=0
export OMP_NUM_THREADS=8
export TORCH_CUDNN_V8_API_ENABLED=1

nohup jupyter nbconvert --to notebook --execute --inplace \
  --ExecutePreprocessor.timeout=-1 \
  tmQM_Fe_XYZ.ipynb > train.log 2>&1 &
echo "Started XYZ training (PID=$!). tail -f train.log 查看进度"

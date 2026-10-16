#!/bin/bash
# 后台跑 tmQM_Fe.ipynb (SMILES GAT, 5070 Ti / Linux)
# 用法: bash run_smiles.sh
cd "$(dirname "$0")"

# Linux GPU 优化环境变量
export CUDA_LAUNCH_BLOCKING=0
export OMP_NUM_THREADS=8
export TORCH_CUDNN_V8_API_ENABLED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

nohup jupyter nbconvert --to notebook --execute --inplace \
  --ExecutePreprocessor.timeout=-1 \
  tmQM_Fe.ipynb > train_smiles.log 2>&1 &
echo "Started SMILES GAT training (PID=$!). tail -f train_smiles.log 查看进度"

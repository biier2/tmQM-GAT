#!/bin/bash
# 后台跑 scorer.ipynb (5070 Ti / Linux)
# 用法: bash run_scorer.sh
cd "$(dirname "$0")"

# Linux GPU 优化环境变量
export CUDA_LAUNCH_BLOCKING=0

nohup jupyter nbconvert --to notebook --execute --inplace \
  --ExecutePreprocessor.timeout=-1 \
  scorer.ipynb > score.log 2>&1 &
echo "Started scoring (PID=$!). tail -f score.log 查看结果"

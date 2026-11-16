#!/bin/bash
# R2-3e driver: 4 lambda values x 3 seeds, two parallel streams.
# lambda is in sigma_gap^2 units (see train_fig4_physL1.py header).
set -u
cd "$HOME/ml/chem_ml5/_r2_3e_20261115" || exit 1
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 PYTHONIOENCODING=utf-8
PY="$HOME/anaconda3/envs/tmqm/bin/python"
mkdir -p out logs

run_one() {  # $1 = lambda token, $2 = numeric lambda, $3 = seed
  local tag="physL1_lam$1_s$2"
  echo "[$(date +%H:%M:%S)] START $tag"
  $PY train_fig4_physL1.py --lam-sigma "$1" --seed "$2" \
      --out-dir out --tag "$tag" > "logs/$tag.log" 2>&1
  echo "[$(date +%H:%M:%S)] DONE  $tag (exit $?)"
}

# two parallel streams, 6 runs each
(
  run_one 0    42  ; run_one 0    123 ; run_one 0    456
  run_one 1    42  ; run_one 1    123 ; run_one 1    456
) > logs/stream_a.log 2>&1 &
(
  run_one 100  42  ; run_one 100  123 ; run_one 100  456
  run_one 10000 42 ; run_one 10000 123 ; run_one 10000 456
) > logs/stream_b.log 2>&1 &
wait
echo "ALL 12 RUNS FINISHED at $(date)"

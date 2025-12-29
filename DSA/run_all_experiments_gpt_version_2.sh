#!/usr/bin/env bash
set -euo pipefail

OUTDIR="runs_gpt_version_3"
SEEDS=(0 1 2)

for SEED in "${SEEDS[@]}"; do
  EXP="B_termhead_v2_s${SEED}"
  echo "=============================="
  echo "Running ${EXP}"
  echo "=============================="
  python train_rl_dsa_B_termhead_v2.py \
    --exp_name "${EXP}" \
    --seed "${SEED}"
done

echo "=============================="
echo "All runs finished ✅"
echo "TensorBoard:"
echo "tensorboard --logdir ${OUTDIR}"

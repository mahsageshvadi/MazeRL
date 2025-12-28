#!/usr/bin/env bash
set -euo pipefail

OUT="runs_gpt_version_2"
DEVICE="cuda"
TOTAL=1200000
ROLLOUT=4096
SEEDS=(0 1 2)

mkdir -p "${OUT}"

for S in "${SEEDS[@]}"; do
  EXP="B_termhead_s${S}"
  echo "=============================="
  echo "Running ${EXP}"
  echo "=============================="

  python train_rl_dsa_GPT_version_2.py \
    --exp_name "${EXP}" \
    --out_dir "${OUT}" \
    --seed "${S}" \
    --device "${DEVICE}" \
    --total_steps "${TOTAL}" \
    --rollout_steps "${ROLLOUT}" \
    --w_turn 0.25 \
    --w_align 0.25 \
    --stop_threshold 0.7 \
    --stop_coef 1.5
done

echo "================================"
echo "All runs finished ✅"
echo "TensorBoard:"
echo "tensorboard --logdir ${OUT}"
echo "================================"

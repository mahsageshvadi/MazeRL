#!/usr/bin/env bash
set -euo pipefail

OUT=runs
DEVICE=cuda
TOTAL=1200000
ROLLOUT=4096
SEEDS=(0 1 2)

for S in "${SEEDS[@]}"; do
  python train_rl_dsa_GPT_version_2.py\
    --exp_name "B_termhead_s${S}" \
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

echo "Done. TensorBoard:"
echo "tensorboard --logdir ${OUT}"

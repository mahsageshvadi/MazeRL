#!/usr/bin/env bash
set -euo pipefail

########################################
# Global config
########################################
OUTDIR="runs_gpt_version"
TOTAL_STEPS=1200000
ROLLOUT=4096
DEVICE="cuda"

SEEDS=(0 1 2)
WTURN=(0.20 0.35)
WALIGN=(0.20 0.35)

mkdir -p "${OUTDIR}"

########################################
# Version A — STOP as explicit action
########################################
echo "=============================="
echo "Running Version A (STOP action)"
echo "=============================="

for SEED in "${SEEDS[@]}"; do
  for TURN in "${WTURN[@]}"; do
    for ALIGN in "${WALIGN[@]}"; do

      EXP="A_stopaction_s${SEED}_turn${TURN}_align${ALIGN}"
      echo ">> ${EXP}"

      python train_rl_dsa_GPT_version.py \
        --exp_name "${EXP}" \
        --out_dir "${OUTDIR}" \
        --seed "${SEED}" \
        --device "${DEVICE}" \
        --step_alpha 1.0 \
        --w_turn "${TURN}" \
        --w_align "${ALIGN}" \
        --total_steps "${TOTAL_STEPS}" \
        --rollout_steps "${ROLLOUT}" \
        --use_stop_action 1

    done
  done
done

########################################
# Version B — NO STOP (implicit termination)
########################################
echo "================================"
echo "Running Version B (No STOP action)"
echo "================================"

for SEED in "${SEEDS[@]}"; do
  for TURN in "${WTURN[@]}"; do
    for ALIGN in "${WALIGN[@]}"; do

      EXP="B_nostop_s${SEED}_turn${TURN}_align${ALIGN}"
      echo ">> ${EXP}"

      python train_rl_dsa_GPT_version.py \
        --exp_name "${EXP}" \
        --out_dir "${OUTDIR}" \
        --seed "${SEED}" \
        --device "${DEVICE}" \
        --step_alpha 1.0 \
        --w_turn "${TURN}" \
        --w_align "${ALIGN}" \
        --total_steps "${TOTAL_STEPS}" \
        --rollout_steps "${ROLLOUT}" \
        --use_stop_action 0

    done
  done
done

echo "================================"
echo "All experiments finished ✅"
echo "TensorBoard:"
echo "  tensorboard --logdir ${OUTDIR}"
echo "================================"

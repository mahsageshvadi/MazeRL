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
# Trainer: train_rl_dsa_A_stopaction.py
########################################
echo "=============================="
echo "Running Version A (STOP action)"
echo "=============================="

for SEED in "${SEEDS[@]}"; do
  for TURN in "${WTURN[@]}"; do
    for ALIGN in "${WALIGN[@]}"; do

      EXP="A_stopaction_s${SEED}_turn${TURN}_align${ALIGN}"

      echo ">> ${EXP}"

      python train_rl_dsa_A_stopaction.py \
        --exp_name "${EXP}" \
        --out_dir "${OUTDIR}" \
        --seed "${SEED}" \
        --device "${DEVICE}" \
        --step_alpha 1.0 \
        --w_turn "${TURN}" \
        --w_align "${ALIGN}" \
        --total_steps "${TOTAL_STEPS}" \
        --rollout_steps "${ROLLOUT}"

    done
  done
done


########################################
# Version B — Termination head
# Trainer: train_rl_dsa_B_termhead.py
########################################
echo "================================"
echo "Running Version B (Term head)"
echo "================================"

for SEED in "${SEEDS[@]}"; do
  for TURN in "${WTURN[@]}"; do
    for ALIGN in "${WALIGN[@]}"; do

      EXP="B_termhead_s${SEED}_turn${TURN}_align${ALIGN}"

      echo ">> ${EXP}"

      python train_rl_dsa_B_termhead.py \
        --exp_name "${EXP}" \
        --out_dir "${OUTDIR}" \
        --seed "${SEED}" \
        --device "${DEVICE}" \
        --step_alpha 1.0 \
        --w_turn "${TURN}" \
        --w_align "${ALIGN}" \
        --total_steps "${TOTAL_STEPS}" \
        --rollout_steps "${ROLLOUT}"

    done
  done
done

echo "================================"
echo "All experiments finished ✅"
echo "TensorBoard:"
echo "  tensorboard --logdir ${OUTDIR}"
echo "================================"

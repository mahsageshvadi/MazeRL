#!/usr/bin/env bash
set -euo pipefail

OUTDIR="runs"
TOTAL_STEPS=1200000
ROLLOUT=4096
DEVICE="cuda"

SEEDS=(0 1 2)
STEP_ALPHA=(1.0)
WTURN=(0.20 0.35)
WALIGN=(0.20 0.35)

########################################
# Version A — STOP as action
########################################
for SEED in "${SEEDS[@]}"; do
  for TURN in "${WTURN[@]}"; do
    for ALIGN in "${WALIGN[@]}"; do
      EXP="A_stopaction_s${SEED}_turn${TURN}_align${ALIGN}"
      python train_rl_dsa_GPT_version.py \
        --exp_name "${EXP}" \
        --out_dir "${OUTDIR}" \
        --seed "${SEED}" \
        --device "${DEVICE}" \
        --use_distractors 1 \
        --use_stop_action 1 \
        --use_termination_head 0 \
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
########################################
for SEED in "${SEEDS[@]}"; do
  for TURN in "${WTURN[@]}"; do
    for ALIGN in "${WALIGN[@]}"; do
      EXP="B_termhead_s${SEED}_turn${TURN}_align${ALIGN}"
      python train_rl_dsa_GPT_version.py \
        --exp_name "${EXP}" \
        --out_dir "${OUTDIR}" \
        --seed "${SEED}" \
        --device "${DEVICE}" \
        --use_distractors 1 \
        --use_stop_action 0 \
        --use_termination_head 1 \
        --step_alpha 1.0 \
        --w_turn "${TURN}" \
        --w_align "${ALIGN}" \
        --total_steps "${TOTAL_STEPS}" \
        --rollout_steps "${ROLLOUT}"
    done
  done
done

echo "All experiments finished."
echo "Analyze with:"
echo "  tensorboard --logdir ${OUTDIR}"

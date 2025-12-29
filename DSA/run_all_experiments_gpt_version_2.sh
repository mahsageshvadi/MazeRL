#!/usr/bin/env bash
set -euo pipefail

PY="train_rl_dsa_GPT_version_2.py"
OUTDIR="runs_gpt_version_2"
TOTAL_STEPS=1200000
ROLLOUT=4096
DEVICE="cuda"

SEEDS=(0 1 2)
WTURN=(0.20 0.35)
WALIGN=(0.20 0.35)

########################################
# Version A — STOP as action
########################################
for SEED in "${SEEDS[@]}"; do
  for TURN in "${WTURN[@]}"; do
    for ALIGN in "${WALIGN[@]}"; do
      EXP="A_stopaction_v2_s${SEED}_turn${TURN}_align${ALIGN}"
      echo "=============================="
      echo "Running ${EXP}"
      echo "=============================="
      python "${PY}" \
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
        --w_precision 2.0 \
        --w_progress 0.25 \
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
      EXP="B_termhead_v2_s${SEED}_turn${TURN}_align${ALIGN}"
      echo "=============================="
      echo "Running ${EXP}"
      echo "=============================="
      python "${PY}" \
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
        --w_precision 2.0 \
        --w_progress 0.25 \
        --stop_threshold 0.7 \
        --stop_coef 1.5 \
        --stop_radius 5.0 \
        --total_steps "${TOTAL_STEPS}" \
        --rollout_steps "${ROLLOUT}"
    done
  done
done

echo "================================"
echo "All runs finished ✅"
echo "TensorBoard:"
echo "  tensorboard --logdir ${OUTDIR}"

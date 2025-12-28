#!/bin/bash

# Create logs directory if it doesn't exist
mkdir -p runs

echo "Starting Multi-Variation Training on GPU..."

# --- EXPERIMENT 1: BASELINE ---
# Standard settings from my previous answer.
# Tests if the core logic works.
nohup python3 train_rl_DSA_gemini_version.py \
  --exp_name "Baseline_Align2.0_Smooth0.2" \
  --align_weight 2.0 \
  --smooth_weight 0.2 \
  --hidden_size 128 \
  --gpu_id 0 \
  > runs/baseline.log 2>&1 &

echo "Launched Experiment 1: Baseline (PID $!)"

# --- EXPERIMENT 2: HIGH ALIGNMENT (FORCE SMOOTHNESS) ---
# Increases alignment reward to 4.0. 
# HYPOTHESIS: Reduces zig-zagging significantly, but might make it harder to turn sharp corners.
nohup python3 train_rl_DSA_gemini_version.py\
  --exp_name "HighAlign_Align4.0_Smooth0.2" \
  --align_weight 4.0 \
  --smooth_weight 0.2 \
  --hidden_size 128 \
  --gpu_id 0 \
  > runs/high_align.log 2>&1 &

echo "Launched Experiment 2: High Alignment (PID $!)"

# --- EXPERIMENT 3: HIGH CAPACITY (COMPLEX DISTRACTORS) ---
# Doubles the LSTM hidden size to 256.
# HYPOTHESIS: Helps with "Object Permanence" (remembering momentum) during crossing distractors.
nohup python3 train_rl_DSA_gemini_version.py \
  --exp_name "HighCap_Hidden256" \
  --align_weight 2.0 \
  --smooth_weight 0.2 \
  --hidden_size 256 \
  --gpu_id 0 \
  > runs/high_capacity.log 2>&1 &

echo "Launched Experiment 3: High Capacity (PID $!)"

# --- EXPERIMENT 4: STRICT SMOOTHNESS ---
# Increases penalty for changing actions.
# HYPOTHESIS: Forces the agent to pick a direction and commit, preventing jitter.
nohup python3 train_rl_DSA_gemini_version.py \
  --exp_name "StrictSmooth_Smooth0.5" \
  --align_weight 2.0 \
  --smooth_weight 0.5 \
  --hidden_size 128 \
  --gpu_id 0 \
  > runs/strict_smooth.log 2>&1 &

echo "Launched Experiment 4: Strict Smoothness (PID $!)"

# --- WAIT COMMAND ---
echo "All experiments launched in background."
echo "Tail the logs to see progress: tail -f runs/baseline.log"
echo "To visualize, run: tensorboard --logdir=runs"

# Optional: Wait for all background processes to finish
wait
echo "All training runs completed."
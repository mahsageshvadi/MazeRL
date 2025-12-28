#!/bin/bash

# --- CONFIGURATION ---
# PASTE THE OUTPUT OF 'which python' HERE
PYTHON_EXEC="/home/mahsa.geshvadi001/anaconda3/envs/RL/bin/python"

# If the path above is wrong, try to uncomment the line below to dynamically find it (if conda is active)
# PYTHON_EXEC=$(which python)

# Create logs directory
mkdir -p runs_gemini_version

echo "Using Python: $PYTHON_EXEC"
echo "Starting Experiments..."

# --- EXPERIMENT 1: BASELINE ---
nohup $PYTHON_EXEC -u train_rl_DSA_gemini_version.py \
  --exp_name "Baseline_Align2.0" \
  --align_weight 2.0 \
  --smooth_weight 0.2 \
  --gpu_id 0 \
  > runs_gemini_version/baseline.log 2>&1 &

echo "Launched Baseline (PID $!)"

# --- EXPERIMENT 2: HIGH ALIGNMENT ---
nohup $PYTHON_EXEC -u train_rl_DSA_gemini_version.py \
  --exp_name "HighAlign_Align4.0" \
  --align_weight 4.0 \
  --smooth_weight 0.2 \
  --gpu_id 0 \
  > runs_gemini_version/high_align.log 2>&1 &

echo "Launched High Alignment (PID $!)"

# --- EXPERIMENT 3: HIGH CAPACITY ---
nohup $PYTHON_EXEC -u train_rl_DSA_gemini_version.py \
  --exp_name "HighCap_Hidden256" \
  --align_weight 2.0 \
  --smooth_weight 0.2 \
  --hidden_size 256 \
  --gpu_id 0 \
  > runs_gemini_version/high_capacity.log 2>&1 &

echo "Launched High Capacity (PID $!)"

wait
echo "All done!"







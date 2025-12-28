#!/bin/bash

# --- CONFIGURATION ---

# 1. Automatically find the python executable from the active environment
PYTHON_EXEC=$(which python)

# 2. Check if Python was found
if [ -z "$PYTHON_EXEC" ]; then
    echo "CRITICAL ERROR: Could not find 'python'."
    echo "Please activate your environment first: conda activate RL"
    exit 1
fi

# 3. Define the python script name
SCRIPT_NAME="train_rl_DSA_gemini_version.py"

# 4. Check if the python script exists in the current folder
if [ ! -f "$SCRIPT_NAME" ]; then
    echo "CRITICAL ERROR: Could not find '$SCRIPT_NAME' in this directory."
    echo "Please make sure you are in the correct folder."
    exit 1
fi

# Create logs directory
LOG_DIR="runs_gemini_version"
mkdir -p "$LOG_DIR"

echo "------------------------------------------------"
echo "Using Python: $PYTHON_EXEC"
echo "Script:       $SCRIPT_NAME"
echo "Logs Dir:     $LOG_DIR"
echo "------------------------------------------------"
echo "Starting Experiments..."

# --- EXPERIMENT 1: BASELINE ---
nohup $PYTHON_EXEC -u "$SCRIPT_NAME" \
  --exp_name "Baseline_Align2.0" \
  --align_weight 2.0 \
  --smooth_weight 0.2 \
  --gpu_id 0 \
  > "$LOG_DIR/baseline.log" 2>&1 &

echo "Launched Experiment 1: Baseline (PID $!)"

# --- EXPERIMENT 2: HIGH ALIGNMENT ---
# Tests if stronger alignment reward fixes zig-zagging
nohup $PYTHON_EXEC -u "$SCRIPT_NAME" \
  --exp_name "HighAlign_Align4.0" \
  --align_weight 4.0 \
  --smooth_weight 0.2 \
  --gpu_id 0 \
  > "$LOG_DIR/high_align.log" 2>&1 &

echo "Launched Experiment 2: High Alignment (PID $!)"

# --- EXPERIMENT 3: HIGH CAPACITY ---
# Tests if a larger brain helps with complex distractors
nohup $PYTHON_EXEC -u "$SCRIPT_NAME" \
  --exp_name "HighCap_Hidden256" \
  --align_weight 2.0 \
  --smooth_weight 0.2 \
  --hidden_size 256 \
  --gpu_id 0 \
  > "$LOG_DIR/high_capacity.log" 2>&1 &

echo "Launched Experiment 3: High Capacity (PID $!)"

# --- WAIT INSTRUCTION ---
echo "------------------------------------------------"
echo "All experiments running in background."
echo "To monitor progress, run: tail -f $LOG_DIR/baseline.log"
echo "To visualize, run: tensorboard --logdir=$LOG_DIR"
echo "------------------------------------------------"

wait
echo "All training runs completed."
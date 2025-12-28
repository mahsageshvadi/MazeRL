#!/bin/bash

# --- 1. ROBUST PYTHON DETECTION ---
# We ask Python to tell us its own path. This is safer than 'which'.
PYTHON_EXEC=$(python -c "import sys; print(sys.executable)" 2>/dev/null)

# Fallback: If the above failed, try 'command -v'
if [ -z "$PYTHON_EXEC" ]; then
    PYTHON_EXEC=$(command -v python)
fi

# Final Check
if [ -z "$PYTHON_EXEC" ] || [ ! -x "$PYTHON_EXEC" ]; then
    echo "CRITICAL ERROR: Python executable not found."
    echo "Please activate your conda environment (e.g., 'conda activate RL') and try again."
    exit 1
fi

# --- 2. SCRIPT SETUP ---
SCRIPT_NAME="train_rl_DSA_gemini_version.py"
LOG_DIR="runs_gemini_version"

# Check if the python script exists
if [ ! -f "$SCRIPT_NAME" ]; then
    echo "CRITICAL ERROR: '$SCRIPT_NAME' not found in this directory."
    echo "Current directory is: $(pwd)"
    exit 1
fi

mkdir -p "$LOG_DIR"

echo "=================================================="
echo "Python Path:   $PYTHON_EXEC"
echo "Training Script: $SCRIPT_NAME"
echo "Logs Directory:  $LOG_DIR"
echo "=================================================="

# --- 3. RUN EXPERIMENTS ---

echo "Launching Experiment 1: Baseline..."
nohup "$PYTHON_EXEC" -u "$SCRIPT_NAME" \
  --exp_name "Baseline_Align2.0" \
  --align_weight 2.0 \
  --smooth_weight 0.2 \
  --gpu_id 0 \
  > "$LOG_DIR/baseline.log" 2>&1 &
echo "PID: $!"

echo "Launching Experiment 2: High Alignment..."
nohup "$PYTHON_EXEC" -u "$SCRIPT_NAME" \
  --exp_name "HighAlign_Align4.0" \
  --align_weight 4.0 \
  --smooth_weight 0.2 \
  --gpu_id 0 \
  > "$LOG_DIR/high_align.log" 2>&1 &
echo "PID: $!"

echo "Launching Experiment 3: High Capacity..."
nohup "$PYTHON_EXEC" -u "$SCRIPT_NAME" \
  --exp_name "HighCap_Hidden256" \
  --align_weight 2.0 \
  --smooth_weight 0.2 \
  --hidden_size 256 \
  --gpu_id 0 \
  > "$LOG_DIR/high_capacity.log" 2>&1 &
echo "PID: $!"

echo "=================================================="
echo "All experiments started."
echo "View output with: tail -f $LOG_DIR/baseline.log"
wait
echo "All runs finished."



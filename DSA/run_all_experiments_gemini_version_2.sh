#!/bin/bash

# --- 1. ROBUST PYTHON DETECTION ---
PYTHON_EXEC=$(python -c "import sys; print(sys.executable)" 2>/dev/null)
if [ -z "$PYTHON_EXEC" ]; then PYTHON_EXEC=$(command -v python); fi
if [ -z "$PYTHON_EXEC" ]; then
    echo "CRITICAL ERROR: Python not found. Activate your env!"
    exit 1
fi

SCRIPT_NAME="train_rl_DSA_gemini_version_2.py"
LOG_DIR="runs_long_gemini"
mkdir -p "$LOG_DIR"

echo "=================================================="
echo "Starting LONG Training Run (160k Episodes)"
echo "Python: $PYTHON_EXEC"
echo "Logs:   $LOG_DIR"
echo "=================================================="

# --- EXPERIMENT 1: BALANCED LONG RUN ---
# Scale 10.0 = 160,000 Episodes total
nohup "$PYTHON_EXEC" -u "$SCRIPT_NAME" \
  --exp_name "Long_Baseline" \
  --scale 10.0 \
  --align_weight 2.0 \
  --smooth_weight 0.2 \
  --gpu_id 0 \
  > "$LOG_DIR/long_baseline.log" 2>&1 &
echo "Launched Long Baseline (PID $!)"

# --- EXPERIMENT 2: HIGH SMOOTHNESS ---
# Emphasizes tracking path smoothly
nohup "$PYTHON_EXEC" -u "$SCRIPT_NAME" \
  --exp_name "Long_HighAlign" \
  --scale 10.0 \
  --align_weight 4.0 \
  --smooth_weight 0.2 \
  --gpu_id 0 \
  > "$LOG_DIR/long_smooth.log" 2>&1 &
echo "Launched Long Smoothness (PID $!)"

# --- EXPERIMENT 3: HIGH CAPACITY BRAIN ---
# Larger brain to handle complex stopping/distractors
nohup "$PYTHON_EXEC" -u "$SCRIPT_NAME" \
  --exp_name "Long_HighCap" \
  --scale 10.0 \
  --hidden_size 256 \
  --align_weight 2.0 \
  --gpu_id 0 \
  > "$LOG_DIR/long_capacity.log" 2>&1 &
echo "Launched Long Capacity (PID $!)"

echo "=================================================="
echo "Monitor with: tail -f $LOG_DIR/long_baseline.log"
wait
echo "Done."
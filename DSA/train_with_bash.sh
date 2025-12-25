#!/bin/bash

# Create a master folder for this session
SESSION_DIR="PathStudy_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$SESSION_DIR"

# CONFIGURATION GRID
# Format: "Name Episodes_Per_Stage LR StopThreshold NoiseMult WidthMax IntensityMin"
CONFIGS=(
    "01_Baseline 4000 0.0001 0.8 1.0 15.0 0.3"
    "02_HeavyNoise 5000 0.0001 0.8 2.5 15.0 0.3"
    "03_UltraWide 5000 0.0001 0.8 1.0 40.0 0.3"
    "04_FaintVessel 6000 0.00005 0.85 1.2 12.0 0.05"
    "05_SlowPrecise 8000 0.00002 0.8 1.5 20.0 0.2"
    "06_Generalist 10000 0.00005 0.9 2.0 30.0 0.1"
)

echo "Starting Grand Experiment Suite..."

for CONF in "${CONFIGS[@]}"; do
    read -r NAME EPS LR STOP NOISE WIDTH INTEN <<< "$CONF"
    
    RUN_DIR="$SESSION_DIR/$NAME"
    mkdir -p "$RUN_DIR"
    
    echo "===================================================="
    echo "RUNNING: $NAME"
    echo "Params: Eps:$EPS, LR:$LR, Stop:$STOP, Noise:$NOISE, Width:$WIDTH, Inten:$INTEN"
    echo "===================================================="
    
    # Run trainer and save terminal output to log file
    python3 train_generalist.py \
        --run_name "$NAME" \
        --out_dir "$RUN_DIR" \
        --eps_per_stage "$EPS" \
        --lr "$LR" \
        --stop_thresh "$STOP" \
        --noise_mult "$NOISE" \
        --width_max "$WIDTH" \
        --intensity_min "$INTEN" > "$RUN_DIR/console.log" 2>&1

    echo "Completed $NAME. Results in $RUN_DIR"
done

echo "ALL EXPERIMENTS FINISHED."
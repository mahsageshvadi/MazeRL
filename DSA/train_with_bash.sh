#!/bin/bash

# Configuration
SCRIPT="train_final.py"
ROOT_DIR="Results_$(date +%m%d_%H%M)"
mkdir -p "$ROOT_DIR"

# List of Experiment Settings
# Name | Episodes | LR | StopThresh | WidthMax | NoiseMult | IntensityMin
EXPERIMENTS=(
    "Baseline 3000 0.0001 0.8 15.0 1.0 0.3"
    "UltraWide 4000 0.0001 0.8 40.0 1.0 0.3"
    "ExtremeNoise 4000 0.0001 0.8 15.0 2.5 0.3"
    "GhostVessel 5000 0.00005 0.85 10.0 1.2 0.05"
)

for EXP in "${EXPERIMENTS[@]}"; do
    read -r NAME EPS LR STOP WIDTH NOISE INTEN <<< "$EXP"
    
    RUN_DIR="$ROOT_DIR/$NAME"
    mkdir -p "$RUN_DIR"
    
    echo "=========================================================="
    echo "STARTING: $NAME"
    echo "=========================================================="
    
    python3 "$SCRIPT" \
        --run_name "$NAME" \
        --out_dir "$RUN_DIR" \
        --eps_per_stage "$EPS" \
        --lr "$LR" \
        --stop_thresh "$STOP" \
        --width_max "$WIDTH" \
        --noise_mult "$NOISE" \
        --intensity_min "$INTEN"
done
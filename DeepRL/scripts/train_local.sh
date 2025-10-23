#!/bin/bash

# Local training script (no SLURM)

# Create directories
mkdir -p models logs results

# Activate environment
source ~/miniconda3/bin/activate maze_drl

# Choose algorithm
ALGORITHM="ddqn"  # or "ppo"

# Training parameters
EPISODES=100000
MIN_SIZE=10
MAX_SIZE=25
OBSTACLE_DENSITY=0.2

echo "Starting training: $ALGORITHM"
echo "Episodes: $EPISODES"
echo "Maze size: $MIN_SIZE to $MAX_SIZE"

# Run training
python train_drl.py \
    --algorithm $ALGORITHM \
    --episodes $EPISODES \
    --min_size $MIN_SIZE \
    --max_size $MAX_SIZE \
    --obstacle_density $OBSTACLE_DENSITY \
    --save_interval 1000 \
    --log_interval 100 \
    --save_dir ./models \
    --seed 42

echo "Training completed!"

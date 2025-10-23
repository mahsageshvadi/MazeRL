#!/bin/bash

# Evaluation script for trained models

# Parameters
MODEL_PATH="./models/best_model_ddqn.pth"
ALGORITHM="ddqn"
NUM_EPISODES=50
MIN_SIZE=10
MAX_SIZE=30

echo "Evaluating model: $MODEL_PATH"
echo "Algorithm: $ALGORITHM"
echo "Test episodes: $NUM_EPISODES"

# Run evaluation
python visualize_model.py \
    --model_path $MODEL_PATH \
    --algorithm $ALGORITHM \
    --mode evaluate \
    --num_episodes $NUM_EPISODES \
    --min_size $MIN_SIZE \
    --max_size $MAX_SIZE \
    --save_dir ./results/evaluation_$(date +%Y%m%d_%H%M%S)

echo "Evaluation completed!"
echo "Results saved in: ./results/"

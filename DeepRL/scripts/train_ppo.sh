#!/bin/bash
#SBATCH --job-name=ppo_maze
#SBATCH --output=logs/ppo_%j.out
#SBATCH --error=logs/ppo_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32GB
#SBATCH --time=48:00:00
#SBATCH --partition=gpu

# Load modules
module purge
module load cuda/11.8
module load python/3.9

# Activate environment
source ~/miniconda3/bin/activate maze_drl

# Create directories
mkdir -p models logs results

# Training parameters
ALGORITHM="ppo"
EPISODES=150000
MIN_SIZE=10
MAX_SIZE=30
OBSTACLE_DENSITY=0.2
SAVE_INTERVAL=2000
LOG_INTERVAL=100

# Run training
python train_drl.py \
    --algorithm $ALGORITHM \
    --episodes $EPISODES \
    --min_size $MIN_SIZE \
    --max_size $MAX_SIZE \
    --obstacle_density $OBSTACLE_DENSITY \
    --save_interval $SAVE_INTERVAL \
    --log_interval $LOG_INTERVAL \
    --save_dir ./models \
    --seed $SLURM_JOB_ID

echo "Training completed!"
echo "Job ID: $SLURM_JOB_ID"
echo "Model saved in: ./models"

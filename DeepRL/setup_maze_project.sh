#!/bin/bash
# =============================================================================
# Deep RL Maze Navigation - Cluster Training Scripts
# =============================================================================

# -----------------------------------------------------------------------------
# File: README.md
# -----------------------------------------------------------------------------
cat > README.md << 'EOF'
# Deep Reinforcement Learning for Maze Navigation

Implementation of Double DQN and PPO algorithms for autonomous navigation in randomly generated mazes, based on the research paper.

## Features

- **Fully randomized environments**: Random maze sizes, wall positions, start/goal locations
- **Two state-of-the-art algorithms**: Double DQN and PPO (Actor-Critic)
- **Distributed training**: Optimized for cluster/GPU environments
- **Model persistence**: Save/load trained models
- **Comprehensive evaluation**: Test on unseen mazes with visualizations
- **Generalization testing**: Evaluate model performance on various maze configurations

## Installation

```bash
# Create conda environment
conda create -n maze_drl python=3.9
conda activate maze_drl

# Install dependencies
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install numpy matplotlib pillow

# Or use pip
pip install -r requirements.txt
```

## Project Structure

```
maze_drl/
├── train_drl.py              # Main training script
├── visualize_model.py        # Visualization and evaluation script
├── requirements.txt          # Python dependencies
├── scripts/
│   ├── train_ddqn.sh        # SLURM script for Double DQN
│   ├── train_ppo.sh         # SLURM script for PPO
│   └── train_local.sh       # Local training script
├── models/                   # Saved model checkpoints
└── results/                  # Evaluation results and visualizations
```

## Training

### Local Training

```bash
# Train Double DQN
python train_drl.py --algorithm ddqn --episodes 100000 --min_size 10 --max_size 30

# Train PPO
python train_drl.py --algorithm ppo --episodes 100000 --min_size 10 --max_size 30

# Resume from checkpoint
python train_drl.py --algorithm ddqn --load_checkpoint ./models/checkpoint_ddqn_ep50000.pth

# Custom parameters
python train_drl.py \
    --algorithm ppo \
    --episodes 200000 \
    --min_size 15 \
    --max_size 40 \
    --obstacle_density 0.25 \
    --save_interval 5000 \
    --log_interval 100
```

### Cluster Training (SLURM)

```bash
# Submit Double DQN job
sbatch scripts/train_ddqn.sh

# Submit PPO job
sbatch scripts/train_ppo.sh

# Check job status
squeue -u $USER

# View output
tail -f slurm-JOBID.out
```

## Evaluation & Visualization

### Quick Demo

```bash
# Single episode demo with static visualization
python visualize_model.py \
    --model_path ./models/best_model_ddqn.pth \
    --algorithm ddqn \
    --mode demo

# Animated visualization
python visualize_model.py \
    --model_path ./models/best_model_ppo.pth \
    --algorithm ppo \
    --mode animate
```

### Comprehensive Evaluation

```bash
# Evaluate on 50 random mazes
python visualize_model.py \
    --model_path ./models/final_model_ddqn.pth \
    --algorithm ddqn \
    --mode evaluate \
    --num_episodes 50 \
    --min_size 10 \
    --max_size 30 \
    --save_dir ./results/ddqn_eval

# Test generalization on larger mazes
python visualize_model.py \
    --model_path ./models/best_model_ppo.pth \
    --algorithm ppo \
    --mode evaluate \
    --num_episodes 100 \
    --min_size 30 \
    --max_size 50 \
    --save_dir ./results/ppo_large_mazes
```

## Training Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--algorithm` | Algorithm: ddqn or ppo | ddqn |
| `--episodes` | Number of training episodes | 100000 |
| `--min_size` | Minimum maze size | 10 |
| `--max_size` | Maximum maze size | 30 |
| `--obstacle_density` | Wall density (0-1) | 0.2 |
| `--seed` | Random seed | 42 |
| `--save_dir` | Model save directory | ./models |
| `--save_interval` | Checkpoint interval | 1000 |
| `--log_interval` | Logging interval | 100 |
| `--load_checkpoint` | Resume from checkpoint | None |

## Algorithm Details

### Double DQN
- **State representation**: Local observation window + relative goal position
- **Action space**: 4 discrete actions (up, down, left, right)
- **Network**: 4-layer feedforward neural network (256 hidden units)
- **Training**: Experience replay, target network, epsilon-greedy exploration
- **Convergence**: ~50k-100k episodes

### PPO (Proximal Policy Optimization)
- **Architecture**: Actor-Critic with shared layers
- **Policy**: Stochastic policy with categorical distribution
- **Training**: Clipped surrogate objective, GAE advantages
- **Convergence**: ~30k-80k episodes

## Expected Results

After training, you should see:
- **Success rate**: 80-95% on test mazes
- **Path optimality**: 1.1-1.3x optimal path length
- **Generalization**: 70-85% success on unseen maze sizes

## Monitoring Training

Training statistics are saved in `training_stats_{algorithm}.json`:
```json
{
  "episode": 50000,
  "episode_rewards": [...],
  "episode_lengths": [...],
  "best_reward": 9.85
}
```

Plot training curves:
```python
import json
import matplotlib.pyplot as plt

with open('models/training_stats_ddqn.json') as f:
    stats = json.load(f)

plt.plot(stats['episode_rewards'])
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.title('Training Progress')
plt.savefig('training_curve.png')
```

## Troubleshooting

**Out of Memory (OOM)**
- Reduce batch size in the code
- Use smaller mazes (reduce max_size)
- Use CPU instead of GPU for very large mazes

**Low Success Rate**
- Train for more episodes
- Adjust reward function
- Check epsilon decay (for DQN)
- Reduce maze complexity

**Model Not Learning**
- Verify maze is solvable
- Check learning rate
- Ensure sufficient exploration
- Monitor loss values

## Citation

If you use this code, please cite the original paper:
```
Isabella Jacob, Felix Williams, James Alex (2025). 
"Deep Reinforcement Learning for Autonomous Navigation in Complex Maze Environments"
```

## License

MIT License - See LICENSE file for details
EOF

# -----------------------------------------------------------------------------
# File: requirements.txt
# -----------------------------------------------------------------------------
cat > requirements.txt << 'EOF'
torch>=2.0.0
numpy>=1.24.0
matplotlib>=3.7.0
pillow>=10.0.0
EOF

# -----------------------------------------------------------------------------
# File: train_ddqn.sh (SLURM script for Double DQN)
# -----------------------------------------------------------------------------
cat > train_ddqn.sh << 'EOF'
#!/bin/bash
#SBATCH --job-name=ddqn_maze
#SBATCH --output=logs/ddqn_%j.out
#SBATCH --error=logs/ddqn_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32GB
#SBATCH --time=48:00:00
#SBATCH --partition=gpu

# Load modules (adjust for your cluster)
module purge
module load cuda/11.8
module load python/3.9

# Activate environment
source ~/miniconda3/bin/activate maze_drl

# Create directories
mkdir -p models logs results

# Training parameters
ALGORITHM="ddqn"
EPISODES=200000
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
EOF

# -----------------------------------------------------------------------------
# File: train_ppo.sh (SLURM script for PPO)
# -----------------------------------------------------------------------------
cat > train_ppo.sh << 'EOF'
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
EOF

# -----------------------------------------------------------------------------
# File: train_local.sh (Local training script)
# -----------------------------------------------------------------------------
cat > train_local.sh << 'EOF'
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
EOF

# -----------------------------------------------------------------------------
# File: evaluate_model.sh (Evaluation script)
# -----------------------------------------------------------------------------
cat > evaluate_model.sh << 'EOF'
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
EOF

# -----------------------------------------------------------------------------
# File: compare_models.py (Compare different models)
# -----------------------------------------------------------------------------
cat > compare_models.py << 'EOF'
import json
import matplotlib.pyplot as plt
import numpy as np
import glob

def plot_training_comparison(model_dirs):
    """Compare training curves from multiple models"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    for model_dir in model_dirs:
        stats_files = glob.glob(f"{model_dir}/training_stats_*.json")
        
        for stats_file in stats_files:
            with open(stats_file, 'r') as f:
                stats = json.load(f)
            
            algorithm = stats_file.split('_')[-1].replace('.json', '')
            
            # Plot rewards
            window = 100
            rewards = stats['episode_rewards']
            smoothed = np.convolve(rewards, np.ones(window)/window, mode='valid')
            axes[0, 0].plot(smoothed, label=f"{algorithm} - {model_dir}")
            
            # Plot episode lengths
            lengths = stats['episode_lengths']
            smoothed_len = np.convolve(lengths, np.ones(window)/window, mode='valid')
            axes[0, 1].plot(smoothed_len, label=f"{algorithm} - {model_dir}")
    
    axes[0, 0].set_xlabel('Episode')
    axes[0, 0].set_ylabel('Reward (smoothed)')
    axes[0, 0].set_title('Training Rewards')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    axes[0, 1].set_xlabel('Episode')
    axes[0, 1].set_ylabel('Steps (smoothed)')
    axes[0, 1].set_title('Episode Lengths')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    plt.tight_layout()
    plt.savefig('training_comparison.png', dpi=150)
    print("Comparison plot saved: training_comparison.png")

if __name__ == "__main__":
    # Compare models from different runs
    model_dirs = ['./models', './models_run2']
    plot_training_comparison(model_dirs)
EOF

chmod +x *.sh

echo "All files created successfully!"
echo ""
echo "File structure:"
echo "  README.md - Complete documentation"
echo "  requirements.txt - Python dependencies"
echo "  train_ddqn.sh - SLURM script for Double DQN"
echo "  train_ppo.sh - SLURM script for PPO"
echo "  train_local.sh - Local training script"
echo "  evaluate_model.sh - Model evaluation script"
echo "  compare_models.py - Training comparison tool"
echo ""
echo "Next steps:"
echo "  1. Install dependencies: pip install -r requirements.txt"
echo "  2. For cluster: sbatch train_ddqn.sh or sbatch train_ppo.sh"
echo "  3. For local: bash train_local.sh"
echo "  4. Evaluate: bash evaluate_model.sh"
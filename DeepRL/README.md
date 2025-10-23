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

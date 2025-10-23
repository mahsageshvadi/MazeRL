"""
Visualize trained DRL model on random unseen mazes
Tests generalization capability and creates visual demonstrations
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Rectangle
import argparse
import os
import json

# Import the classes from training script
# (In practice, you'd import from the training module)
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical


class MazeEnvironment:
    """Same as training environment"""
    def __init__(self, min_size=10, max_size=30, obstacle_density=0.2):
        self.min_size = min_size
        self.max_size = max_size
        self.obstacle_density = obstacle_density
        self.reset()
    
    def reset(self):
        self.size = np.random.randint(self.min_size, self.max_size + 1)
        self.maze = np.zeros((self.size, self.size), dtype=np.float32)
        
        self.maze[0, :] = 1
        self.maze[-1, :] = 1
        self.maze[:, 0] = 1
        self.maze[:, -1] = 1
        
        num_obstacles = int(self.size * self.size * self.obstacle_density)
        for _ in range(num_obstacles):
            x = np.random.randint(1, self.size - 1)
            y = np.random.randint(1, self.size - 1)
            self.maze[x, y] = 1
        
        while True:
            self.start_pos = [np.random.randint(1, self.size - 1), 
                             np.random.randint(1, self.size - 1)]
            if self.maze[self.start_pos[0], self.start_pos[1]] == 0:
                break
        
        while True:
            self.goal_pos = [np.random.randint(1, self.size - 1), 
                            np.random.randint(1, self.size - 1)]
            distance = abs(self.goal_pos[0] - self.start_pos[0]) + \
                      abs(self.goal_pos[1] - self.start_pos[1])
            if (self.maze[self.goal_pos[0], self.goal_pos[1]] == 0 and 
                distance > self.size // 2):
                break
        
        self.agent_pos = self.start_pos.copy()
        self.visited = set()
        self.visited.add(tuple(self.agent_pos))
        self.steps = 0
        self.max_steps = self.size * self.size
        
        return self._get_state()
    
    def _get_state(self):
        window_size = 5
        padded_maze = np.pad(self.maze, window_size, constant_values=1)
        
        x, y = self.agent_pos[0] + window_size, self.agent_pos[1] + window_size
        local_view = padded_maze[x-window_size:x+window_size+1, 
                                 y-window_size:y+window_size+1]
        
        rel_goal = [
            (self.goal_pos[0] - self.agent_pos[0]) / self.size,
            (self.goal_pos[1] - self.agent_pos[1]) / self.size
        ]
        
        norm_pos = [
            self.agent_pos[0] / self.size,
            self.agent_pos[1] / self.size
        ]
        
        state = np.concatenate([
            local_view.flatten(),
            rel_goal,
            norm_pos,
            [self.steps / self.max_steps]
        ])
        
        return state.astype(np.float32)
    
    def step(self, action):
        self.steps += 1
        moves = [[-1, 0], [1, 0], [0, -1], [0, 1]]
        dx, dy = moves[action]
        
        new_pos = [self.agent_pos[0] + dx, self.agent_pos[1] + dy]
        
        if self.maze[new_pos[0], new_pos[1]] == 1:
            reward = -1.0
            done = False
            return self._get_state(), reward, done, {}
        
        self.agent_pos = new_pos
        reward = -0.01
        
        if tuple(self.agent_pos) in self.visited:
            reward -= 0.05
        else:
            self.visited.add(tuple(self.agent_pos))
        
        done = False
        if self.agent_pos == self.goal_pos:
            reward = 10.0
            done = True
        
        if self.steps >= self.max_steps:
            reward -= 5.0
            done = True
        
        return self._get_state(), reward, done, {}
    
    def get_state_size(self):
        window_size = 5
        return (2 * window_size + 1) ** 2 + 5


class DQNNetwork(nn.Module):
    def __init__(self, state_size, action_size, hidden_size=256):
        super(DQNNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.fc4 = nn.Linear(hidden_size, action_size)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return self.fc4(x)


class PPONetwork(nn.Module):
    def __init__(self, state_size, action_size, hidden_size=256):
        super(PPONetwork, self).__init__()
        
        self.shared = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU()
        )
        
        self.actor = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, action_size)
        )
        
        self.critic = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )
    
    def forward(self, x):
        shared = self.shared(x)
        return self.actor(shared), self.critic(shared)
    
    def get_action(self, x):
        logits, value = self.forward(x)
        dist = Categorical(logits=logits)
        action = dist.sample()
        return action, dist.log_prob(action), value


def load_model(model_path, algorithm, device='cpu'):
    """Load trained model"""
    checkpoint = torch.load(model_path, map_location=device)
    
    state_size = checkpoint['state_size']
    action_size = checkpoint['action_size']
    
    if algorithm == 'ddqn':
        model = DQNNetwork(state_size, action_size).to(device)
        model.load_state_dict(checkpoint['q_network_state_dict'])
    else:  # ppo
        model = PPONetwork(state_size, action_size).to(device)
        model.load_state_dict(checkpoint['network_state_dict'])
    
    model.eval()
    print(f"Model loaded: {model_path}")
    print(f"State size: {state_size}, Action size: {action_size}")
    
    return model, state_size, action_size


def test_model(model, algorithm, env, device='cpu', max_steps=500):
    """Test model on a single episode and record trajectory"""
    state = env.reset()
    trajectory = [env.agent_pos.copy()]
    actions_taken = []
    rewards = []
    done = False
    steps = 0
    
    while not done and steps < max_steps:
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
        
        with torch.no_grad():
            if algorithm == 'ddqn':
                q_values = model(state_tensor)
                action = q_values.argmax().item()
            else:  # ppo
                logits, _ = model(state_tensor)
                action = logits.argmax().item()
        
        state, reward, done, _ = env.step(action)
        
        trajectory.append(env.agent_pos.copy())
        actions_taken.append(action)
        rewards.append(reward)
        steps += 1
    
    success = env.agent_pos == env.goal_pos
    total_reward = sum(rewards)
    
    return {
        'trajectory': trajectory,
        'actions': actions_taken,
        'rewards': rewards,
        'steps': steps,
        'success': success,
        'total_reward': total_reward,
        'maze': env.maze.copy(),
        'start_pos': env.start_pos,
        'goal_pos': env.goal_pos
    }


def visualize_episode(result, save_path=None, show=True):
    """Create static visualization of an episode"""
    maze = result['maze']
    trajectory = result['trajectory']
    start_pos = result['start_pos']
    goal_pos = result['goal_pos']
    
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Draw maze
    ax.imshow(maze, cmap='binary', alpha=0.3)
    
    # Draw walls
    for i in range(maze.shape[0]):
        for j in range(maze.shape[1]):
            if maze[i, j] == 1:
                ax.add_patch(Rectangle((j-0.5, i-0.5), 1, 1, 
                                      facecolor='black', edgecolor='gray'))
    
    # Draw trajectory
    if len(trajectory) > 1:
        traj_array = np.array(trajectory)
        ax.plot(traj_array[:, 1], traj_array[:, 0], 
               'b-', linewidth=2, alpha=0.6, label='Path')
        
        # Color gradient for trajectory
        for i in range(len(trajectory)-1):
            alpha = (i + 1) / len(trajectory)
            ax.plot([trajectory[i][1], trajectory[i+1][1]], 
                   [trajectory[i][0], trajectory[i+1][0]], 
                   'cyan', linewidth=2, alpha=alpha)
    
    # Draw start and goal
    ax.plot(start_pos[1], start_pos[0], 'go', markersize=15, 
           label='Start', markeredgecolor='white', markeredgewidth=2)
    ax.plot(goal_pos[1], goal_pos[0], 'r*', markersize=20, 
           label='Goal', markeredgecolor='white', markeredgewidth=2)
    
    # Draw final position
    if len(trajectory) > 0:
        final_pos = trajectory[-1]
        ax.plot(final_pos[1], final_pos[0], 'bs', markersize=12, 
               label='Final Position', markeredgecolor='white', markeredgewidth=2)
    
    ax.set_xlim(-0.5, maze.shape[1]-0.5)
    ax.set_ylim(maze.shape[0]-0.5, -0.5)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right')
    
    # Add title with stats
    success_text = "SUCCESS" if result['success'] else "FAILED"
    color = 'green' if result['success'] else 'red'
    ax.set_title(f"Episode Result: {success_text}\n"
                f"Steps: {result['steps']} | "
                f"Reward: {result['total_reward']:.2f} | "
                f"Maze: {maze.shape[0]}x{maze.shape[1]}",
                fontsize=14, fontweight='bold', color=color)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Visualization saved to {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()
    
    return fig


def create_animation(result, save_path=None):
    """Create animated visualization of episode"""
    maze = result['maze']
    trajectory = result['trajectory']
    start_pos = result['start_pos']
    goal_pos = result['goal_pos']
    
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Draw maze background
    ax.imshow(maze, cmap='binary', alpha=0.3)
    
    # Draw walls
    for i in range(maze.shape[0]):
        for j in range(maze.shape[1]):
            if maze[i, j] == 1:
                ax.add_patch(Rectangle((j-0.5, i-0.5), 1, 1, 
                                      facecolor='black', edgecolor='gray'))
    
    # Draw start and goal
    ax.plot(start_pos[1], start_pos[0], 'go', markersize=15, 
           label='Start', markeredgecolor='white', markeredgewidth=2)
    ax.plot(goal_pos[1], goal_pos[0], 'r*', markersize=20, 
           label='Goal', markeredgecolor='white', markeredgewidth=2)
    
    ax.set_xlim(-0.5, maze.shape[1]-0.5)
    ax.set_ylim(maze.shape[0]-0.5, -0.5)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right')
    
    # Initialize agent and trail
    agent, = ax.plot([], [], 'bs', markersize=12, 
                    markeredgecolor='white', markeredgewidth=2)
    trail, = ax.plot([], [], 'cyan', linewidth=2, alpha=0.6)
    
    title = ax.text(0.5, 1.05, '', transform=ax.transAxes, 
                   ha='center', fontsize=12, fontweight='bold')
    
    def init():
        agent.set_data([], [])
        trail.set_data([], [])
        return agent, trail, title
    
    def animate(frame):
        if frame < len(trajectory):
            pos = trajectory[frame]
            agent.set_data([pos[1]], [pos[0]])
            
            # Update trail
            trail_data = np.array(trajectory[:frame+1])
            if len(trail_data) > 1:
                trail.set_data(trail_data[:, 1], trail_data[:, 0])
            
            # Update title
            title.set_text(f"Step: {frame}/{len(trajectory)-1} | "
                         f"Reward: {sum(result['rewards'][:frame]):.2f}")
        
        return agent, trail, title
    
    anim = animation.FuncAnimation(fig, animate, init_func=init,
                                  frames=len(trajectory), interval=100,
                                  blit=True, repeat=True)
    
    if save_path:
        anim.save(save_path, writer='pillow', fps=10)
        print(f"Animation saved to {save_path}")
    
    plt.show()
    return anim


def run_evaluation(model_path, algorithm, num_episodes=10, 
                   min_size=10, max_size=30, save_dir='./results'):
    """Run comprehensive evaluation on multiple random mazes"""
    
    os.makedirs(save_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load model
    model, state_size, action_size = load_model(model_path, algorithm, device)
    
    # Create environment
    env = MazeEnvironment(min_size, max_size, obstacle_density=0.2)
    
    # Run tests
    results = []
    success_count = 0
    total_steps = 0
    total_reward = 0
    
    print(f"\nRunning evaluation on {num_episodes} random mazes...")
    print("-" * 60)
    
    for i in range(num_episodes):
        result = test_model(model, algorithm, env, device)
        results.append(result)
        
        if result['success']:
            success_count += 1
        
        total_steps += result['steps']
        total_reward += result['total_reward']
        
        print(f"Episode {i+1}/{num_episodes} | "
              f"Maze: {result['maze'].shape[0]}x{result['maze'].shape[1]} | "
              f"Steps: {result['steps']} | "
              f"Reward: {result['total_reward']:.2f} | "
              f"{'SUCCESS' if result['success'] else 'FAILED'}")
        
        # Save visualization for each episode
        fig = visualize_episode(result, 
                               save_path=f"{save_dir}/episode_{i+1}.png",
                               show=False)
        plt.close(fig)
    
    # Print summary statistics
    print("\n" + "="*60)
    print("EVALUATION SUMMARY")
    print("="*60)
    print(f"Success Rate: {success_count}/{num_episodes} ({success_count/num_episodes*100:.1f}%)")
    print(f"Average Steps: {total_steps/num_episodes:.1f}")
    print(f"Average Reward: {total_reward/num_episodes:.2f}")
    print(f"Results saved to: {save_dir}")
    
    # Save summary statistics
    summary = {
        'model_path': model_path,
        'algorithm': algorithm,
        'num_episodes': num_episodes,
        'success_rate': success_count / num_episodes,
        'avg_steps': total_steps / num_episodes,
        'avg_reward': total_reward / num_episodes,
        'episodes': [
            {
                'episode': i+1,
                'maze_size': result['maze'].shape[0],
                'steps': result['steps'],
                'reward': result['total_reward'],
                'success': result['success']
            }
            for i, result in enumerate(results)
        ]
    }
    
    with open(f"{save_dir}/evaluation_summary.json", 'w') as f:
        json.dump(summary, f, indent=2)
    
    return results, summary


def demo_single_episode(model_path, algorithm, animate=False):
    """Demo a single episode with visualization"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load model
    model, state_size, action_size = load_model(model_path, algorithm, device)
    
    # Create environment
    env = MazeEnvironment(min_size=15, max_size=25, obstacle_density=0.2)
    
    # Run test
    print("\nRunning demo episode...")
    result = test_model(model, algorithm, env, device)
    
    print(f"\nResults:")
    print(f"  Maze Size: {result['maze'].shape[0]}x{result['maze'].shape[1]}")
    print(f"  Steps: {result['steps']}")
    print(f"  Total Reward: {result['total_reward']:.2f}")
    print(f"  Status: {'SUCCESS' if result['success'] else 'FAILED'}")
    
    # Visualize
    if animate:
        create_animation(result, save_path='demo_animation.gif')
    else:
        visualize_episode(result, save_path='demo_episode.png')
    
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Visualize trained DRL maze navigation model')
    
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to trained model checkpoint')
    parser.add_argument('--algorithm', type=str, required=True, 
                       choices=['ddqn', 'ppo'],
                       help='Algorithm used for training')
    parser.add_argument('--mode', type=str, default='evaluate',
                       choices=['demo', 'evaluate', 'animate'],
                       help='Visualization mode')
    parser.add_argument('--num_episodes', type=int, default=10,
                       help='Number of episodes for evaluation')
    parser.add_argument('--min_size', type=int, default=10,
                       help='Minimum maze size for testing')
    parser.add_argument('--max_size', type=int, default=30,
                       help='Maximum maze size for testing')
    parser.add_argument('--save_dir', type=str, default='./results',
                       help='Directory to save results')
    
    args = parser.parse_args()
    
    if args.mode == 'demo':
        demo_single_episode(args.model_path, args.algorithm, animate=False)
    elif args.mode == 'animate':
        demo_single_episode(args.model_path, args.algorithm, animate=True)
    elif args.mode == 'evaluate':
        run_evaluation(args.model_path, args.algorithm, 
                      args.num_episodes, args.min_size, args.max_size,
                      args.save_dir)
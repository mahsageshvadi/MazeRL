

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np
import random
from collections import deque
import pickle
import os
import argparse
from datetime import datetime
import json

# Set random seeds for reproducibility
def set_seed(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# Maze Environment with full randomization
class MazeEnvironment:
    def __init__(self, min_size=10, max_size=30, obstacle_density=0.2):
        self.min_size = min_size
        self.max_size = max_size
        self.obstacle_density = obstacle_density
        self.reset()
    
    def reset(self):
        """Generate a new random maze"""
        # Random maze size
        self.size = np.random.randint(self.min_size, self.max_size + 1)
        
        # Initialize empty maze
        self.maze = np.zeros((self.size, self.size), dtype=np.float32)
        
        # Add borders
        self.maze[0, :] = 1
        self.maze[-1, :] = 1
        self.maze[:, 0] = 1
        self.maze[:, -1] = 1
        
        # Add random obstacles
        num_obstacles = int(self.size * self.size * self.obstacle_density)
        for _ in range(num_obstacles):
            x = np.random.randint(1, self.size - 1)
            y = np.random.randint(1, self.size - 1)
            self.maze[x, y] = 1
        
        # Random start position
        while True:
            self.start_pos = [np.random.randint(1, self.size - 1), 
                             np.random.randint(1, self.size - 1)]
            if self.maze[self.start_pos[0], self.start_pos[1]] == 0:
                break
        
        # Random goal position (ensure it's far from start)
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
        """Get state representation with local observation window"""
        window_size = 5
        padded_maze = np.pad(self.maze, window_size, constant_values=1)
        
        # Agent position in padded maze
        x, y = self.agent_pos[0] + window_size, self.agent_pos[1] + window_size
        
        # Local observation
        local_view = padded_maze[x-window_size:x+window_size+1, 
                                 y-window_size:y+window_size+1]
        
        # Relative goal position (normalized)
        rel_goal = [
            (self.goal_pos[0] - self.agent_pos[0]) / self.size,
            (self.goal_pos[1] - self.agent_pos[1]) / self.size
        ]
        
        # Current position (normalized)
        norm_pos = [
            self.agent_pos[0] / self.size,
            self.agent_pos[1] / self.size
        ]
        
        # Flatten and concatenate
        state = np.concatenate([
            local_view.flatten(),
            rel_goal,
            norm_pos,
            [self.steps / self.max_steps]
        ])
        
        return state.astype(np.float32)
    
    def step(self, action):
        """Take action: 0=up, 1=down, 2=left, 3=right"""
        self.steps += 1
        moves = [[-1, 0], [1, 0], [0, -1], [0, 1]]
        dx, dy = moves[action]
        
        new_pos = [self.agent_pos[0] + dx, self.agent_pos[1] + dy]
        
        # Check collision
        if self.maze[new_pos[0], new_pos[1]] == 1:
            reward = -1.0
            done = False
            return self._get_state(), reward, done, {}
        
        # Update position
        self.agent_pos = new_pos
        
        # Calculate reward
        reward = -0.01  # Small step penalty
        
        # Check if revisiting
        if tuple(self.agent_pos) in self.visited:
            reward -= 0.05
        else:
            self.visited.add(tuple(self.agent_pos))
        
        # Check goal
        done = False
        if self.agent_pos == self.goal_pos:
            reward = 10.0
            done = True
        
        # Check timeout
        if self.steps >= self.max_steps:
            reward -= 5.0
            done = True
        
        return self._get_state(), reward, done, {}
    
    def get_state_size(self):
        """Calculate state size"""
        window_size = 5
        return (2 * window_size + 1) ** 2 + 5  # local view + goal + pos + steps


# Double DQN Network
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


# Double DQN Agent
class DoubleDQNAgent:
    def __init__(self, state_size, action_size, device='cuda'):
        self.state_size = state_size
        self.action_size = action_size
        self.device = device
        
        # Hyperparameters
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.9995
        self.learning_rate = 0.0001
        self.batch_size = 64
        self.update_target_freq = 1000
        
        # Networks
        self.q_network = DQNNetwork(state_size, action_size).to(device)
        self.target_network = DQNNetwork(state_size, action_size).to(device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=self.learning_rate)
        
        # Replay buffer
        self.memory = deque(maxlen=100000)
        self.steps = 0
    
    def act(self, state, training=True):
        """Select action using epsilon-greedy policy"""
        if training and random.random() < self.epsilon:
            return random.randint(0, self.action_size - 1)
        
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.q_network(state_tensor)
            return q_values.argmax().item()
    
    def remember(self, state, action, reward, next_state, done):
        """Store experience in replay buffer"""
        self.memory.append((state, action, reward, next_state, done))
    
    def train(self):
        """Train the network using experience replay"""
        if len(self.memory) < self.batch_size:
            return 0.0
        
        # Sample batch
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        states = torch.FloatTensor(np.array(states)).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(np.array(next_states)).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        
        # Current Q values
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
        
        # Double DQN: use online network to select action, target network to evaluate
        with torch.no_grad():
            next_actions = self.q_network(next_states).argmax(1)
            next_q_values = self.target_network(next_states).gather(1, next_actions.unsqueeze(1))
            target_q_values = rewards.unsqueeze(1) + self.gamma * next_q_values * (1 - dones.unsqueeze(1))
        
        # Compute loss
        loss = F.mse_loss(current_q_values, target_q_values)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 1.0)
        self.optimizer.step()
        
        # Update target network
        self.steps += 1
        if self.steps % self.update_target_freq == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())
        
        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        
        return loss.item()
    
    def save(self, filepath):
        """Save model weights and parameters"""
        torch.save({
            'q_network_state_dict': self.q_network.state_dict(),
            'target_network_state_dict': self.target_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'steps': self.steps,
            'state_size': self.state_size,
            'action_size': self.action_size
        }, filepath)
        print(f"Model saved to {filepath}")
    
    def load(self, filepath):
        """Load model weights and parameters"""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
        self.target_network.load_state_dict(checkpoint['target_network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']
        self.steps = checkpoint['steps']
        print(f"Model loaded from {filepath}")


# PPO Network
class PPONetwork(nn.Module):
    def __init__(self, state_size, action_size, hidden_size=256):
        super(PPONetwork, self).__init__()
        
        # Shared layers
        self.shared = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU()
        )
        
        # Actor head
        self.actor = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, action_size)
        )
        
        # Critic head
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


# PPO Agent
class PPOAgent:
    def __init__(self, state_size, action_size, device='cuda'):
        self.state_size = state_size
        self.action_size = action_size
        self.device = device
        
        # Hyperparameters
        self.gamma = 0.99
        self.gae_lambda = 0.95
        self.clip_epsilon = 0.2
        self.learning_rate = 0.0003
        self.epochs = 4
        self.batch_size = 64
        self.value_coef = 0.5
        self.entropy_coef = 0.01
        
        # Network
        self.network = PPONetwork(state_size, action_size).to(device)
        self.optimizer = optim.Adam(self.network.parameters(), lr=self.learning_rate)
        
        # Storage
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.log_probs = []
        self.dones = []
    
    def act(self, state, training=True):
        """Select action using policy"""
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            action, log_prob, value = self.network.get_action(state_tensor)
        
        if training:
            self.states.append(state)
            self.actions.append(action.item())
            self.log_probs.append(log_prob.item())
            self.values.append(value.item())
        
        return action.item()
    
    def remember(self, reward, done):
        """Store reward and done flag"""
        self.rewards.append(reward)
        self.dones.append(done)
    
    def train(self):
        """Train using PPO algorithm"""
        if len(self.states) == 0:
            return 0.0
        
        # Convert to tensors
        states = torch.FloatTensor(np.array(self.states)).to(self.device)
        actions = torch.LongTensor(self.actions).to(self.device)
        old_log_probs = torch.FloatTensor(self.log_probs).to(self.device)
        
        # Calculate returns and advantages
        returns = []
        advantages = []
        gae = 0
        
        for t in reversed(range(len(self.rewards))):
            if t == len(self.rewards) - 1:
                next_value = 0
            else:
                next_value = self.values[t + 1]
            
            delta = self.rewards[t] + self.gamma * next_value * (1 - self.dones[t]) - self.values[t]
            gae = delta + self.gamma * self.gae_lambda * (1 - self.dones[t]) * gae
            
            advantages.insert(0, gae)
            returns.insert(0, gae + self.values[t])
        
        advantages = torch.FloatTensor(advantages).to(self.device)
        returns = torch.FloatTensor(returns).to(self.device)
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # PPO update
        total_loss = 0
        for _ in range(self.epochs):
            # Get current policy outputs
            logits, values = self.network(states)
            dist = Categorical(logits=logits)
            log_probs = dist.log_prob(actions)
            entropy = dist.entropy().mean()
            
            # Calculate ratios
            ratios = torch.exp(log_probs - old_log_probs)
            
            # Calculate surrogate losses
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages
            
            # Calculate losses
            actor_loss = -torch.min(surr1, surr2).mean()
            critic_loss = F.mse_loss(values.squeeze(), returns)
            loss = actor_loss + self.value_coef * critic_loss - self.entropy_coef * entropy
            
            # Optimize
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 0.5)
            self.optimizer.step()
            
            total_loss += loss.item()
        
        # Clear storage
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.log_probs = []
        self.dones = []
        
        return total_loss / self.epochs
    
    def save(self, filepath):
        """Save model weights"""
        torch.save({
            'network_state_dict': self.network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'state_size': self.state_size,
            'action_size': self.action_size
        }, filepath)
        print(f"Model saved to {filepath}")
    
    def load(self, filepath):
        """Load model weights"""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.network.load_state_dict(checkpoint['network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print(f"Model loaded from {filepath}")


# Training function
def train_agent(args):
    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create environment
    env = MazeEnvironment(args.min_size, args.max_size, args.obstacle_density)
    state_size = env.get_state_size()
    action_size = 4
    
    # Create agent
    if args.algorithm == 'ddqn':
        agent = DoubleDQNAgent(state_size, action_size, device)
    else:
        agent = PPOAgent(state_size, action_size, device)
    
    # Load checkpoint if exists
    if args.load_checkpoint and os.path.exists(args.load_checkpoint):
        agent.load(args.load_checkpoint)
    
    # Training loop
    best_reward = -float('inf')
    episode_rewards = []
    episode_lengths = []
    success_count = 0
    
    print(f"\nTraining {args.algorithm.upper()} on mazes size {args.min_size}-{args.max_size}")
    print(f"Target episodes: {args.episodes}\n")
    
    for episode in range(args.episodes):
        state = env.reset()
        episode_reward = 0
        episode_length = 0
        done = False
        
        while not done:
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            
            if args.algorithm == 'ddqn':
                agent.remember(state, action, reward, next_state, done)
                loss = agent.train()
            else:
                agent.remember(reward, done)
            
            state = next_state
            episode_reward += reward
            episode_length += 1
        
        # Train PPO at end of episode
        if args.algorithm == 'ppo':
            loss = agent.train()
        
        # Track statistics
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        if episode_reward > 5:  # Successful episode
            success_count += 1
        
        # Logging
        if (episode + 1) % args.log_interval == 0:
            avg_reward = np.mean(episode_rewards[-args.log_interval:])
            avg_length = np.mean(episode_lengths[-args.log_interval:])
            success_rate = success_count / args.log_interval * 100
            
            print(f"Episode {episode + 1}/{args.episodes} | "
                  f"Avg Reward: {avg_reward:.2f} | "
                  f"Avg Length: {avg_length:.1f} | "
                  f"Success Rate: {success_rate:.1f}% | "
                  f"Epsilon: {agent.epsilon if args.algorithm == 'ddqn' else 'N/A'}")
            
            success_count = 0
        
        # Save best model
        if episode_reward > best_reward:
            best_reward = episode_reward
            agent.save(f"{args.save_dir}/best_model_{args.algorithm}.pth")
        
        # Save checkpoint
        if (episode + 1) % args.save_interval == 0:
            agent.save(f"{args.save_dir}/checkpoint_{args.algorithm}_ep{episode+1}.pth")
            
            # Save training stats
            stats = {
                'episode': episode + 1,
                'episode_rewards': episode_rewards,
                'episode_lengths': episode_lengths,
                'best_reward': best_reward
            }
            with open(f"{args.save_dir}/training_stats_{args.algorithm}.json", 'w') as f:
                json.dump(stats, f)
    
    # Save final model
    agent.save(f"{args.save_dir}/final_model_{args.algorithm}.pth")
    print(f"\nTraining completed! Final model saved.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train DRL agent for maze navigation')
    
    # Training parameters
    parser.add_argument('--algorithm', type=str, default='ddqn', choices=['ddqn', 'ppo'],
                        help='Algorithm to use: ddqn or ppo')
    parser.add_argument('--episodes', type=int, default=100000,
                        help='Number of training episodes')
    parser.add_argument('--min_size', type=int, default=10,
                        help='Minimum maze size')
    parser.add_argument('--max_size', type=int, default=30,
                        help='Maximum maze size')
    parser.add_argument('--obstacle_density', type=float, default=0.2,
                        help='Obstacle density (0-1)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    
    # Logging and saving
    parser.add_argument('--save_dir', type=str, default='./models',
                        help='Directory to save models')
    parser.add_argument('--save_interval', type=int, default=1000,
                        help='Save checkpoint every N episodes')
    parser.add_argument('--log_interval', type=int, default=100,
                        help='Log statistics every N episodes')
    parser.add_argument('--load_checkpoint', type=str, default=None,
                        help='Path to checkpoint to resume training')
    
    args = parser.parse_args()
    
    # Create save directory
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Start training
    train_agent(args)
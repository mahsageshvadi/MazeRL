#!/usr/bin/env python3
import numpy as np
import random
from collections import deque, namedtuple
import torch
import torch.nn as nn
import torch.nn.functional as F

# =========================
# Utils: reproducibility
# =========================
def set_seeds(seed=123):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

# =========================
# Maze generation (solvable)
# =========================
def generate_random_maze(rows, cols, wall_frac=0.25, max_tries=50):
    """
    Generate a random maze with start on left edge and goal on right edge.
    Ensure solvable via BFS; retry up to max_tries.
    """
    assert rows >= 5 and cols >= 5
    for _ in range(max_tries):
        maze = np.zeros((rows, cols), dtype=np.int32)

        # Random interior walls
        n_cells = rows * cols
        n_walls = int(n_cells * wall_frac)
        wall_indices = np.random.choice(n_cells, size=n_walls, replace=False)
        wr, wc = np.unravel_index(wall_indices, (rows, cols))
        maze[wr, wc] = 1

        # Carve borders a bit to avoid fully blocking edges
        maze[:, 0] = 0
        maze[:, -1] = 0

        # Random start on left column; random goal on right column
        start = (np.random.randint(rows), 0)
        goal  = (np.random.randint(rows), cols - 1)
        maze[start] = 2
        maze[goal] = 3

        # Make immediate neighbors free to reduce trivial traps
        if start[1] + 1 < cols: maze[start[0], start[1] + 1] = 0
        if goal[1] - 1 >= 0:    maze[goal[0], goal[1] - 1] = 0

        if is_solvable(maze, start, goal):
            return maze
    # Fallback: carve a simple corridor if retries fail
    maze = np.zeros((rows, cols), dtype=np.int32)
    start = (rows // 2, 0)
    goal = (rows // 2, cols - 1)
    maze[start] = 2
    maze[goal] = 3
    return maze

def is_solvable(maze, start, goal):
    rows, cols = maze.shape
    q = deque([start])
    visited = set([start])
    dirs = [(-1,0),(1,0),(0,-1),(0,1)]
    while q:
        r, c = q.popleft()
        if (r, c) == goal: return True
        for dr, dc in dirs:
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols and (nr, nc) not in visited:
                if maze[nr, nc] != 1:  # not a wall
                    visited.add((nr, nc))
                    q.append((nr, nc))
    return False

# =========================
# Environment (Gym-like)
# =========================
class MazeEnv:
    ACTIONS = [(-1,0),(1,0),(0,-1),(0,1)]  # up, down, left, right

    def __init__(self, rows=7, cols=7, wall_frac=0.25, max_steps=None):
        self.rows, self.cols = rows, cols
        self.wall_frac = wall_frac
        self.max_steps = max_steps or (rows * cols * 2)
        self.reset()

    def reset(self):
        self.maze = generate_random_maze(self.rows, self.cols, self.wall_frac)
        self.start = tuple(np.argwhere(self.maze == 2)[0])
        self.goal  = tuple(np.argwhere(self.maze == 3)[0])
        self.agent = self.start
        self.steps = 0
        return self._obs()

    def _obs(self):
        # Channels: [walls, agent, goal] as 0/1 floats
        walls = (self.maze == 1).astype(np.float32)
        agent = np.zeros_like(walls, dtype=np.float32)
        goal  = (self.maze == 3).astype(np.float32)
        agent[self.agent] = 1.0
        obs = np.stack([walls, agent, goal], axis=0)  # (3, H, W)
        return obs

    def step(self, action_idx):
        self.steps += 1
        dr, dc = MazeEnv.ACTIONS[action_idx]
        nr, nc = self.agent[0] + dr, self.agent[1] + dc
        reward = -0.01  # small step cost
        done = False

        # Invalid/out-of-bounds or wall -> penalty, stay in place
        if not (0 <= nr < self.rows and 0 <= nc < self.cols) or self.maze[nr, nc] == 1:
            reward += -0.5
            nr, nc = self.agent
        else:
            # Move
            self.agent = (nr, nc)

        if self.agent == self.goal:
            reward += 10.0
            done = True

        if self.steps >= self.max_steps:
            done = True

        return self._obs(), reward, done, {}

# =========================
# DQN Model (size-agnostic via GAP)
# =========================
class DQNCNN(nn.Module):
    def __init__(self, n_actions):
        super().__init__()
        # input: (B, 3, H, W)
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(),
        )
        # Global Average Pooling -> (B, 64)
        self.head = nn.Sequential(
            nn.Linear(64, 128), nn.ReLU(),
            nn.Linear(128, n_actions)
        )

    def forward(self, x):
        # x: (B,3,H,W)
        z = self.conv(x)               # (B,64,H,W)
        z = z.mean(dim=(2,3))          # global average pool (B,64)
        q = self.head(z)               # (B,n_actions)
        return q

# =========================
# Replay Buffer
# =========================
Transition = namedtuple("Transition", "s a r s2 d")

class ReplayBuffer:
    def __init__(self, capacity=50_000):
        self.buf = deque(maxlen=capacity)

    def push(self, *args):
        self.buf.append(Transition(*args))

    def sample(self, batch_size):
        batch = random.sample(self.buf, batch_size)
        s   = torch.tensor(np.stack([t.s for t in batch]), dtype=torch.float32)
        a   = torch.tensor([t.a for t in batch], dtype=torch.long)
        r   = torch.tensor([t.r for t in batch], dtype=torch.float32)
        s2  = torch.tensor(np.stack([t.s2 for t in batch]), dtype=torch.float32)
        d   = torch.tensor([t.d for t in batch], dtype=torch.float32)
        return s, a, r, s2, d

    def __len__(self):
        return len(self.buf)

# =========================
# Training
# =========================
def train_dqn(
    episodes=3000,
    maze_size=(7,7),
    wall_frac=0.25,
    gamma=0.99,
    lr=1e-3,
    batch_size=64,
    start_learning=1000,
    target_update=1000,
    eps_start=1.0,
    eps_end=0.05,
    eps_decay=20000,
    print_every=100,
    device="cpu"
):
    env = MazeEnv(rows=maze_size[0], cols=maze_size[1], wall_frac=wall_frac)
    n_actions = 4

    policy = DQNCNN(n_actions).to(device)
    target = DQNCNN(n_actions).to(device)
    target.load_state_dict(policy.state_dict())
    target.eval()

    opt = torch.optim.Adam(policy.parameters(), lr=lr)
    buffer = ReplayBuffer(100_000)

    steps_done = 0
    def epsilon_by_step(t):
        # exponential decay over steps
        return eps_end + (eps_start - eps_end) * np.exp(-t / eps_decay)

    running_success = deque(maxlen=100)
    for ep in range(1, episodes+1):
        obs = env.reset()
        done = False
        ep_reward = 0.0
        while not done:
            steps_done += 1
            eps = epsilon_by_step(steps_done)
            print(f"episode: {ep} and epsilon: {eps}")

            # ε-greedy
            if random.random() < eps:
                a = random.randrange(n_actions)
            else:
                with torch.no_grad():
                    q = policy(torch.tensor(obs[None], device=device))
                    a = int(torch.argmax(q, dim=1).item())

            obs2, r, done, _ = env.step(a)
            ep_reward += r

            buffer.push(obs, a, r, obs2, float(done))
            obs = obs2

            # Learn
            if len(buffer) >= start_learning:
                s, a_b, r_b, s2, d_b = buffer.sample(batch_size)
                s, a_b, r_b, s2, d_b = s.to(device), a_b.to(device), r_b.to(device), s2.to(device), d_b.to(device)

                with torch.no_grad():
                    q_next = target(s2).max(dim=1)[0]
                    target_q = r_b + gamma * (1.0 - d_b) * q_next

                q_pred = policy(s).gather(1, a_b.view(-1,1)).squeeze(1)
                loss = F.smooth_l1_loss(q_pred, target_q)

                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(policy.parameters(), 5.0)
                opt.step()

                if steps_done % target_update == 0:
                    target.load_state_dict(policy.state_dict())

        success = (obs2[1].sum() == 0.0)  # agent channel should be zero at terminal; but better: success if reward ended with +1 bonus
        # more robust: success if agent is at goal:
        success = (env.agent == env.goal)
        running_success.append(1 if success else 0)

        if ep % print_every == 0:
            print(f"Episode {ep:4d} | avg success(100)={np.mean(running_success):.2f} | eps={eps:.2f} | last ep reward={ep_reward:.2f}")
        
        

    return policy, env

# =========================
# Policy rollout on unseen mazes
# =========================
def greedy_rollout(policy, rows=7, cols=7, wall_frac=0.25, max_steps=None, device="cpu", verbose=True):
    env = MazeEnv(rows, cols, wall_frac, max_steps=max_steps)
    obs = env.reset()
    path = [env.agent]
    for _ in range(env.max_steps):
        with torch.no_grad():
            q = policy(torch.tensor(obs[None], dtype=torch.float32, device=device))
            a = int(torch.argmax(q, dim=1).item())
        obs, r, done, _ = env.step(a)
        path.append(env.agent)
        if done:
            break
    success = (env.agent == env.goal)
    if verbose:
        print_maze_with_path(env.maze, path)
        print("Success:", success, "| Steps:", len(path)-1)
    return success, path

def print_maze_with_path(maze, path):
    rows, cols = maze.shape
    char = {0: " ", 1: "█", 2: "S", 3: "G"}
    grid = [[char[int(maze[r,c])] for c in range(cols)] for r in range(rows)]
    # overlay path
    for r,c in path[1:-1]:
        if maze[r,c] == 0:
            grid[r][c] = "."
    print("\n".join("".join(row) for row in grid))

# =========================
# Main
# =========================
if __name__ == "__main__":
   # set_seeds(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Train on random mazes (7x7). You can later vary sizes at eval time.
    policy, _ = train_dqn(
        episodes=2000,        # increase (e.g., 5000+) for stronger policy
        maze_size=(20,20),
        wall_frac=0.25,
        gamma=0.99,
        lr=1e-3,
        batch_size=64,
        start_learning=1500,
        target_update=1000,
        eps_start=1.0,
        eps_end=0.05,
        eps_decay=300000,
        print_every=100,
        device=device
    )

    torch.save(policy.state_dict(), "dqn_maze_model.pth")

    # Evaluate on unseen random mazes
    trials = 20
    wins = 0
    for i in range(trials):
        success, _ = greedy_rollout(policy, rows=7, cols=7, wall_frac=0.25, device=device, verbose=False)
        wins += int(success)
    print(f"Unseen 7x7 mazes — success: {wins}/{trials}")

    # Show a couple of printed paths
    print("\nExample rollout 1:")
    greedy_rollout(policy, rows=7, cols=7, wall_frac=0.25, device=device, verbose=True)

    print("\nExample rollout 2 (different maze):")
    greedy_rollout(policy, rows=7, cols=7, wall_frac=0.25, device=device, verbose=True)

#!/usr/bin/env python3
"""
DQN Maze Visualizer (3-channel obs: [walls, agent, goal])

- Loads a trained 3-channel GAP CNN (same as your training script).
- Generates a brand-new solvable random maze.
- Animates greedy actions step by step in Tkinter.
- Highlights visited cells (soft yellow) and the final path (cyan).
"""

import argparse
import numpy as np
import random
from collections import deque
import tkinter as tk
import torch
import torch.nn as nn
import torch.nn.functional as F

# =========================
# Model (same as training)
# =========================
class DQNCNN(nn.Module):
    def __init__(self, n_actions: int = 4):
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
        z = self.conv(x)          # (B,64,H,W)
        z = z.mean(dim=(2,3))     # GAP -> (B,64)
        q = self.head(z)          # (B,n_actions)
        return q


# =========================
# Maze generation (solvable)
# =========================
ACTIONS = [(-1,0),(1,0),(0,-1),(0,1)]  # up, down, left, right

def is_solvable(maze, start, goal):
    rows, cols = maze.shape
    q = deque([start])
    visited = {start}
    for_dr = [(-1,0),(1,0),(0,-1),(0,1)]
    while q:
        r, c = q.popleft()
        if (r, c) == goal:
            return True
        for dr, dc in for_dr:
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols and (nr, nc) not in visited:
                if maze[nr, nc] != 1:  # not a wall
                    visited.add((nr, nc))
                    q.append((nr, nc))
    return False

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

        # Carve borders to reduce edge traps
        maze[:, 0] = 0
        maze[:, -1] = 0

        # Start/goal
        start = (np.random.randint(rows), 0)
        goal  = (np.random.randint(rows), cols - 1)
        maze[start] = 2
        maze[goal]  = 3

        # Make immediate neighbors free
        if start[1] + 1 < cols: maze[start[0], start[1] + 1] = 0
        if goal[1] - 1 >= 0:    maze[goal[0],  goal[1]  - 1] = 0

        if is_solvable(maze, start, goal):
            return maze

    # Fallback: simple corridor if retries fail
    maze = np.zeros((rows, cols), dtype=np.int32)
    start = (rows // 2, 0)
    goal  = (rows // 2, cols - 1)
    maze[start] = 2
    maze[goal]  = 3
    return maze


# =========================
# Environment (3 channels)
# =========================
class MazeEnv:
    ACTIONS = ACTIONS

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
        walls = (self.maze == 1).astype(np.float32)
        agent = np.zeros_like(walls, dtype=np.float32); agent[self.agent] = 1.0
        goal  = (self.maze == 3).astype(np.float32)
        return np.stack([walls, agent, goal], axis=0)  # (3, H, W)

    def step(self, action_idx):
        self.steps += 1
        dr, dc = MazeEnv.ACTIONS[action_idx]
        nr, nc = self.agent[0] + dr, self.agent[1] + dc

        # Block on walls/OOB
        if not (0 <= nr < self.rows and 0 <= nc < self.cols) or self.maze[nr, nc] == 1:
            nr, nc = self.agent
        else:
            self.agent = (nr, nc)

        done = (self.agent == self.goal) or (self.steps >= self.max_steps)
        return self._obs(), done


# =========================
# Tkinter Visualizer
# =========================
class MazeViewer:
    def __init__(self, env: MazeEnv, cell: int = 40):
        self.env = env
        self.cell = cell
        self.root = tk.Tk()
        self.root.title("DQN Maze — Step-by-Step (Visited + Path)")
        self.canvas = tk.Canvas(self.root, width=env.cols * cell, height=env.rows * cell)
        self.canvas.pack()

        self.agent_rect = None
        self.visited = set()
        self.path = []

        self.draw_static()

    def _cell_rect(self, r, c):
        x0, y0 = c * self.cell, r * self.cell
        x1, y1 = x0 + self.cell, y0 + self.cell
        return x0, y0, x1, y1

    def draw_static(self):
        self.canvas.delete("all")
        # Base grid
        for r in range(self.env.rows):
            for c in range(self.env.cols):
                x0, y0, x1, y1 = self._cell_rect(r, c)
                v = self.env.maze[r, c]
                fill = "white"
                if v == 1: fill = "black"   # wall
                elif v == 2: fill = "blue"  # start
                elif v == 3: fill = "green" # goal
                self.canvas.create_rectangle(x0, y0, x1, y1, fill=fill, outline="gray")

        # Visited (soft yellow)
        for (vr, vc) in self.visited:
            if self.env.maze[vr, vc] == 1:
                continue
            x0, y0, x1, y1 = self._cell_rect(vr, vc)
            self.canvas.create_rectangle(x0, y0, x1, y1, fill="#fff2b2", outline="gray")

        # Path-so-far (light cyan)
        for (pr, pc) in self.path:
            if self.env.maze[pr, pc] == 1:
                continue
            x0, y0, x1, y1 = self._cell_rect(pr, pc)
            self.canvas.create_rectangle(x0, y0, x1, y1, fill="#b7ecff", outline="gray")

        # Agent on top
        self.draw_agent()
        self.root.update()

    def draw_agent(self):
        r, c = self.env.agent
        x0, y0, x1, y1 = self._cell_rect(r, c)
        if self.agent_rect is not None:
            self.canvas.delete(self.agent_rect)
        self.agent_rect = self.canvas.create_rectangle(x0, y0, x1, y1, fill="red", outline="gray")

    def animate(self, policy: nn.Module, delay_ms: int = 120, device: str = "cpu", max_steps: int | None = None):
        # reset layers
        self.visited.clear()
        self.path.clear()

        # include start
        self.visited.add(self.env.agent)
        self.path.append(self.env.agent)
        self.draw_static()

        obs = self.env._obs()
        steps = 0
        while True:
            with torch.no_grad():
                q = policy(torch.tensor(obs[None], dtype=torch.float32, device=device))
                a = int(torch.argmax(q, dim=1).item())

            obs, done = self.env.step(a)

            self.visited.add(self.env.agent)
            self.path.append(self.env.agent)

            self.draw_static()
            self.root.after(delay_ms)
            self.root.update()

            steps += 1
            if done or (max_steps and steps >= max_steps):
                break

        # Final overlay with brighter cyan for the whole path
        for (pr, pc) in self.path:
            if self.env.maze[pr, pc] == 1:
                continue
            x0, y0, x1, y1 = self._cell_rect(pr, pc)
            self.canvas.create_rectangle(x0, y0, x1, y1, fill="#6fd6ff", outline="gray")

        # Agent on top again
        self.draw_agent()
        self.root.update()

        print(f"Finished. Success: {self.env.agent == self.env.goal}, Steps: {steps}")


# =========================
# Main
# =========================
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--weights", type=str, default="dqn_maze_model_7_7_4000_episodes_reward_50.pth",
                   help="Path to model .pth (trained with 3-channel input)")
    p.add_argument("--rows", type=int, default=7)
    p.add_argument("--cols", type=int, default=7)
    p.add_argument("--wall_frac", type=float, default=0.4)
    p.add_argument("--cell_px", type=int, default=50, help="Cell size in pixels")
    p.add_argument("--delay_ms", type=int, default=120, help="Animation delay between steps")
    p.add_argument("--cpu", action="store_true", help="Force CPU even if CUDA is available")
    args = p.parse_args()

    device = "cpu" if args.cpu or (not torch.cuda.is_available()) else "cuda"

    # Build model and load weights
    policy = DQNCNN(n_actions=4).to(device)
    # weights_only is a newer arg; omit if your PyTorch doesn't support it
    state = torch.load(args.weights, map_location=device)
    policy.load_state_dict(state)
    policy.eval()
    print(f"Loaded model from {args.weights}")

    # Fresh random maze (you can try larger sizes too, e.g., --rows 20 --cols 20)
    env = MazeEnv(rows=args.rows, cols=args.cols, wall_frac=args.wall_frac)

    # Visualize
    viewer = MazeViewer(env, cell=args.cell_px)
    viewer.animate(policy, delay_ms=args.delay_ms, device=device)

    # Keep window open
    viewer.root.mainloop()


if __name__ == "__main__":
    main()

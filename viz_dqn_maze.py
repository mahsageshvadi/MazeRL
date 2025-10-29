#!/usr/bin/env python3
import time
import numpy as np
import random
from collections import deque
import tkinter as tk
import torch
import torch.nn as nn

# ==========
# Model (same as training)
# ==========
class DQNCNN(nn.Module):
    def __init__(self, n_actions):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1), nn.ReLU(),
        )
        self.head = nn.Sequential(
            nn.Linear(128, 256), nn.ReLU(),
            nn.Linear(256, n_actions),
        )

    def forward(self, x):
        z = self.conv(x)          # (B,128,H,W)
        z = z.mean(dim=(2,3))     # GAP -> (B,128)
        return self.head(z)       # (B,4)



# ==========
# Env (same as training, solvable random mazes)
# ==========
ACTIONS = [(-1,0),(1,0),(0,-1),(0,1)]  # up, down, left, right

def is_solvable(maze, start, goal):
    R, C = maze.shape
    q = deque([start])
    seen = {start}
    while q:
        r, c = q.popleft()
        if (r, c) == goal:
            return True
        for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < R and 0 <= nc < C and (nr, nc) not in seen and maze[nr, nc] != 1:
                seen.add((nr, nc))
                q.append((nr, nc))
    return False

def generate_random_maze(rows, cols, wall_frac=0.25, max_tries=50):
    assert rows >= 5 and cols >= 5
    for _ in range(max_tries):
        maze = np.zeros((rows, cols), dtype=np.int32)
        n_cells = rows*cols
        n_walls = int(n_cells*wall_frac)
        wr, wc = np.unravel_index(np.random.choice(n_cells, n_walls, replace=False), (rows, cols))
        maze[wr, wc] = 1
        maze[:, 0] = 0; maze[:, -1] = 0
        start = (np.random.randint(rows), 0)
        goal  = (np.random.randint(rows), cols-1)
        maze[start] = 2; maze[goal] = 3
        if start[1]+1 < cols: maze[start[0], start[1]+1] = 0
        if goal[1]-1 >= 0:    maze[goal[0], goal[1]-1] = 0
        if is_solvable(maze, start, goal): return maze
    # fallback corridor
    maze = np.zeros((rows, cols), dtype=np.int32)
    maze[rows//2, 0] = 2; maze[rows//2, cols-1] = 3
    return maze

class MazeEnv:
    def __init__(self, rows=7, cols=7, wall_frac=0.25, max_steps=None):
        self.rows, self.cols = rows, cols
        self.wall_frac = wall_frac
        self.max_steps = max_steps or (rows*cols*2)
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
        dr, dc = ACTIONS[action_idx]
        nr, nc = self.agent[0] + dr, self.agent[1] + dc
        if not (0 <= nr < self.rows and 0 <= nc < self.cols) or self.maze[nr, nc] == 1:
            # hit wall / OOB: stay
            nr, nc = self.agent
        else:
            self.agent = (nr, nc)
        done = (self.agent == self.goal) or (self.steps >= self.max_steps)
        return self._obs(), done

# ==========
# Tkinter visualizer
# ==========
class MazeViewer:
    def __init__(self, env, cell=40):
        self.env = env
        self.cell = cell
        self.root = tk.Tk()
        self.root.title("DQN Maze — Step-by-Step (Visited + Path)")
        self.canvas = tk.Canvas(self.root, width=env.cols*cell, height=env.rows*cell)
        self.canvas.pack()

        # layers
        self.agent_rect = None
        self.visited = set()         # all cells the agent has ever stepped on
        self.path = []               # ordered positions for the rollout

        self.draw_static()

    def _cell_rect(self, r, c):
        x0, y0 = c*self.cell, r*self.cell
        x1, y1 = x0 + self.cell, y0 + self.cell
        return x0, y0, x1, y1

    def draw_static(self):
        self.canvas.delete("all")
        for r in range(self.env.rows):
            for c in range(self.env.cols):
                x0, y0, x1, y1 = self._cell_rect(r, c)
                v = self.env.maze[r, c]
                fill = "white"
                if v == 1: fill = "black"   # wall
                elif v == 2: fill = "blue"  # start
                elif v == 3: fill = "green" # goal
                self.canvas.create_rectangle(x0, y0, x1, y1, fill=fill, outline="gray")

        # draw visited layer (soft yellow)
        for (vr, vc) in self.visited:
            # don't paint walls
            if self.env.maze[vr, vc] == 1: 
                continue
            x0, y0, x1, y1 = self._cell_rect(vr, vc)
            self.canvas.create_rectangle(x0, y0, x1, y1, fill="#fff2b2", outline="gray")

        # draw path-so-far (light cyan)
        for (pr, pc) in self.path:
            if self.env.maze[pr, pc] == 1:
                continue
            x0, y0, x1, y1 = self._cell_rect(pr, pc)
            self.canvas.create_rectangle(x0, y0, x1, y1, fill="#b7ecff", outline="gray")

        # agent on top
        self.draw_agent()
        self.root.update()

    def draw_agent(self):
        r, c = self.env.agent
        x0, y0, x1, y1 = self._cell_rect(r, c)
        if self.agent_rect is not None:
            self.canvas.delete(self.agent_rect)
        self.agent_rect = self.canvas.create_rectangle(x0, y0, x1, y1, fill="red", outline="gray")

    def animate(self, policy, delay_ms=150, device="cpu", max_steps=None):
        # reset layers
        self.visited = set()
        self.path = []

        # include start as visited
        self.visited.add(self.env.agent)
        self.path.append(self.env.agent)

        self.draw_static()
        obs = self.env._obs()
        steps = 0

        while True:
            with torch.no_grad():
                q = policy(torch.tensor(obs[None], dtype=torch.float32, device=device))
                a = int(torch.argmax(q, dim=1).item())   # greedy action

            obs, done = self.env.step(a)

            # record layers
            self.visited.add(self.env.agent)
            self.path.append(self.env.agent)

            # redraw layers
            self.draw_static()
            self.root.after(delay_ms)
            self.root.update()

            steps += 1
            if done or (max_steps and steps >= max_steps):
                break

        # final pretty overlay of the full path (brighter cyan)
        for (pr, pc) in self.path:
            if self.env.maze[pr, pc] == 1:
                continue
            x0, y0, x1, y1 = self._cell_rect(pr, pc)
            self.canvas.create_rectangle(x0, y0, x1, y1, fill="#6fd6ff", outline="gray")
        # agent on top again
        self.draw_agent()
        self.root.update()

        print(f"Finished. Success: {self.env.agent == self.env.goal}, Steps: {steps}")

# ==========
# Run: load weights, test on a fresh random maze, visualize decisions
# ==========
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 1) Build model and load weights
    policy = DQNCNN(n_actions=4).to(device)
    policy.load_state_dict(torch.load("dqn_maze_model_20_20.pth", map_location=device))
    policy.eval()
    print("Loaded model from dqn_maze_model.pth")

    # 2) Make a brand-new random maze (change size/wall_frac if you want)
    env = MazeEnv(rows=20, cols=20, wall_frac=0.25)

    # 3) Visualize step-by-step decisions
    viewer = MazeViewer(env, cell=40)
    viewer.animate(policy, delay_ms=150, device=device)

    # keep the window open
    viewer.root.mainloop()

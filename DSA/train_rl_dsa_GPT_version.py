#!/usr/bin/env python3
"""
train_rl_dsa.py

PPO training for sequential vessel / curve tracking.

✔ CSV + TensorBoard logging
✔ Smart checkpointing (BEST + LAST only)
✔ No disk bloat
"""

import os
import time
import json
import csv
import math
import argparse
from dataclasses import dataclass, asdict
from typing import Dict, Any, Tuple, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

from Curve_Generator_Flexible_For_Ciruculum_learning import CurveMakerFlexible

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_OK = True
except Exception:
    TENSORBOARD_OK = False


# ============================================================
# Utilities
# ============================================================

def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def clamp(x, lo, hi):
    return lo if x < lo else hi if x > hi else x

def crop32(img, cy, cx, size=32):
    r = size // 2
    out = np.zeros((size, size), dtype=np.float32)
    y0, y1 = int(cy - r), int(cy + r)
    x0, x1 = int(cx - r), int(cx + r)

    yy0, yy1 = max(0, y0), min(img.shape[0], y1)
    xx0, xx1 = max(0, x0), min(img.shape[1], x1)

    out[yy0 - y0:yy1 - y0, xx0 - x0:xx1 - x0] = img[yy0:yy1, xx0:xx1]
    return out

def cosine(a, b, eps=1e-8):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + eps))


# ============================================================
# Config
# ============================================================

@dataclass
class TrainConfig:
    exp_name: str = "dsa_rl"
    out_dir: str = "runs"
    seed: int = 0
    device: str = "cuda"

    H: int = 128
    W: int = 128
    crop: int = 32
    max_steps: int = 160

    use_distractors: int = 1
    width_min: int = 2
    width_max: int = 8
    noise_prob: float = 1.0
    invert_prob: float = 0.5
    min_intensity: float = 0.2

    step_alpha: float = 1.0
    use_stop_action: int = 1
    stop_radius: float = 5.0

    w_precision: float = 2.0
    w_progress: float = 0.30
    w_align: float = 0.20
    w_turn: float = 0.20
    w_step: float = 0.01
    w_offtrack: float = 3.0
    offtrack_thresh: float = 10.0

    r_stop_correct: float = 40.0
    r_stop_wrong: float = -2.0

    total_steps: int = 2_000_000
    rollout_steps: int = 4096
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_eps: float = 0.2
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    lr: float = 3e-4
    update_epochs: int = 8
    minibatch_size: int = 512

    eval_every_updates: int = 10
    eval_episodes: int = 50


# ============================================================
# Environment
# ============================================================

ACTIONS_8 = [
    (-1, 0), (-1, 1), (0, 1), (1, 1),
    (1, 0), (1, -1), (0, -1), (-1, -1)
]

class DSAPathEnv:
    def __init__(self, cfg: TrainConfig):
        self.cfg = cfg
        self.cm = CurveMakerFlexible(cfg.H, cfg.W, cfg.seed)

    @property
    def n_actions(self):
        return 9 if self.cfg.use_stop_action else 8

    def reset(self):
        img, mask, pts = self.cm.sample_with_distractors(
            width_range=(self.cfg.width_min, self.cfg.width_max),
            noise_prob=self.cfg.noise_prob,
            invert_prob=self.cfg.invert_prob,
            min_intensity=self.cfg.min_intensity
        )

        self.img = img.astype(np.float32)
        self.gt_mask = mask.astype(np.float32)
        self.gt = pts[0].astype(np.float32)

        idx = 5
        self.agent = tuple(self.gt[idx])
        self.prev_idx = idx
        self.path = [self.agent]
        self.steps = 0
        return self.obs()

    def obs(self):
        y, x = self.agent
        actor = np.stack([
            crop32(self.img, y, x),
            crop32(self.img, y, x),
            crop32(self.img, y, x),
            crop32(self.gt_mask, y, x)
        ], axis=0)
        return {"actor": actor, "critic_gt": actor[:1]}

    def step(self, action):
        self.steps += 1
        done = False
        reward = -self.cfg.w_step

        if self.cfg.use_stop_action and action == 8:
            dist = np.linalg.norm(np.array(self.agent) - self.gt[-1])
            if dist < self.cfg.stop_radius:
                return self.obs(), self.cfg.r_stop_correct, True, {"success": True}
            return self.obs(), self.cfg.r_stop_wrong, False, {}

        dy, dx = ACTIONS_8[action]
        y, x = self.agent
        self.agent = (
            clamp(y + dy * self.cfg.step_alpha, 0, self.cfg.H - 1),
            clamp(x + dx * self.cfg.step_alpha, 0, self.cfg.W - 1)
        )

        if self.steps >= self.cfg.max_steps:
            done = True

        return self.obs(), reward, done, {}


# ============================================================
# Model
# ============================================================

class ActorCritic(nn.Module):
    def __init__(self, n_actions):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(4, 32, 3, 2, 1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, 2, 1), nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, 256), nn.ReLU()
        )
        self.actor = nn.Linear(256, n_actions)
        self.critic = nn.Linear(256, 1)

    def forward(self, x):
        z = self.net(x)
        return self.actor(z), self.critic(z).squeeze(-1)


# ============================================================
# Training
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_name", type=str, required=True)
    parser.add_argument("--out_dir", type=str, default="runs_gpt_version")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--step_alpha", type=float, default=1.0)
    parser.add_argument("--w_turn", type=float, default=0.20)
    parser.add_argument("--w_align", type=float, default=0.20)
    parser.add_argument("--total_steps", type=int, default=1200000)
    parser.add_argument("--rollout_steps", type=int, default=4096)
    parser.add_argument("--use_stop_action", type=int, default=1)

    args = parser.parse_args()

    cfg = TrainConfig(
        exp_name=args.exp_name,
        out_dir=args.out_dir,
        seed=args.seed,
        device=args.device,
        step_alpha=args.step_alpha,
        w_turn=args.w_turn,
        w_align=args.w_align,
        total_steps=args.total_steps,
        rollout_steps=args.rollout_steps,
        use_stop_action=args.use_stop_action,
    )

    print(f"[MODE] use_stop_action = {cfg.use_stop_action}")

    set_seed(cfg.seed)

    run_dir = os.path.join(cfg.out_dir, cfg.exp_name)
    ckpt_dir = os.path.join(run_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    with open(os.path.join(run_dir, "config.json"), "w") as f:
        json.dump(asdict(cfg), f, indent=2)

    env = DSAPathEnv(cfg)
    device = torch.device(cfg.device)
    model = ActorCritic(env.n_actions).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    writer = SummaryWriter(os.path.join(run_dir, "tb")) if TENSORBOARD_OK else None

    best_success = -1.0
    env_steps = 0
    update = 0
    t0 = time.time()

    obs = env.reset()

    while env_steps < cfg.total_steps:
        update += 1

        obs_t = torch.tensor(obs["actor"][None], device=device)
        logits, value = model(obs_t)
        dist = Categorical(logits=logits)
        action = dist.sample().item()

        obs, r, done, info = env.step(action)
        env_steps += 1

        if done:
            obs = env.reset()

        # ---- CHECKPOINTING (CLEAN) ----
        torch.save(
            {"model": model.state_dict(), "update": update},
            os.path.join(ckpt_dir, "last.pt")
        )

        if update % cfg.eval_every_updates == 0:
            success = float(info.get("success", False))
            if success > best_success:
                best_success = success
                torch.save(
                    {"model": model.state_dict(), "best_success": best_success},
                    os.path.join(ckpt_dir, "best.pt")
                )
                print(f"✓ New BEST model (success={best_success:.2f})")

        if update % 10 == 0:
            print(f"[upd {update}] steps={env_steps}")

    torch.save(
        {"model": model.state_dict()},
        os.path.join(run_dir, "final_model.pt")
    )
    print("Training complete.")


if __name__ == "__main__":
    main()

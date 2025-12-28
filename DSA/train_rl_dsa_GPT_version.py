#!/usr/bin/env python3
"""
train_rl_dsa.py

PPO training for sequential vessel/curve tracking with:
- domain randomization & distractors (via CurveMakerFlexible)
- smoothness/zig-zag penalty (turn angle)
- progress + alignment rewards
- stopping (either STOP action or termination head)
- metrics + logging (CSV + TensorBoard)
- checkpointing

Requires: numpy, torch, scipy, tensorboard
Your generator file must be importable in the same folder:
  Curve_Generator_Flexible_For_Ciruculum_learning.py
"""

import os
import math
import time
import json
import csv
import argparse
from dataclasses import dataclass, asdict
from typing import Dict, Any, Tuple, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from scipy.ndimage import gaussian_filter

try:
    from Curve_Generator_Flexible_For_Ciruculum_learning import CurveMakerFlexible
except ImportError as e:
    raise ImportError(
        "Could not import CurveMakerFlexible. Put Curve_Generator_Flexible_For_Ciruculum_learning.py "
        "in the same directory, or add it to PYTHONPATH."
    ) from e

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_OK = True
except Exception:
    TENSORBOARD_OK = False


# -------------------------
# Utilities
# -------------------------
def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def clamp(x, lo, hi):
    return lo if x < lo else (hi if x > hi else x)

def crop32(img: np.ndarray, cy: int, cx: int, size: int = 32) -> np.ndarray:
    """Centered crop with padding."""
    H, W = img.shape
    r = size // 2
    y0, y1 = cy - r, cy + r
    x0, x1 = cx - r, cx + r

    out = np.zeros((size, size), dtype=np.float32)
    yy0 = max(0, y0); yy1 = min(H, y1)
    xx0 = max(0, x0); xx1 = min(W, x1)

    oy0 = yy0 - y0
    ox0 = xx0 - x0
    out[oy0:oy0 + (yy1 - yy0), ox0:ox0 + (xx1 - xx0)] = img[yy0:yy1, xx0:xx1]
    return out

def polyline_length(pts: np.ndarray) -> float:
    if len(pts) < 2:
        return 0.0
    dif = pts[1:] - pts[:-1]
    return float(np.sum(np.sqrt(np.sum(dif * dif, axis=1))))

def nearest_gt_index(pos_yx: Tuple[float, float], gt_poly: np.ndarray) -> int:
    p = np.array([pos_yx[0], pos_yx[1]], dtype=np.float32)
    d2 = np.sum((gt_poly - p[None, :]) ** 2, axis=1)
    return int(np.argmin(d2))

def distance_to_polyline_pointwise(pos_yx: Tuple[float, float], gt_poly: np.ndarray) -> float:
    """Distance to nearest gt point (cheap & stable)."""
    idx = nearest_gt_index(pos_yx, gt_poly)
    dy = float(gt_poly[idx, 0] - pos_yx[0])
    dx = float(gt_poly[idx, 1] - pos_yx[1])
    return float(math.sqrt(dy * dy + dx * dx))

def unit(v: np.ndarray, eps=1e-8) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / (n + eps)

def cosine(a: np.ndarray, b: np.ndarray, eps=1e-8) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + eps))

def turning_penalty(p2, p1, p0) -> float:
    """Penalty based on angle change between last two moves.
    p2 -> p1 is previous direction, p1 -> p0 is current direction.
    Returns (1 - cos(theta)) in [0,2].
    """
    v_prev = np.array([p1[0] - p2[0], p1[1] - p2[1]], dtype=np.float32)
    v_curr = np.array([p0[0] - p1[0], p0[1] - p1[1]], dtype=np.float32)
    if np.linalg.norm(v_prev) < 1e-6 or np.linalg.norm(v_curr) < 1e-6:
        return 0.0
    c = cosine(v_prev, v_curr)
    return 1.0 - c  # 0 when straight, ~1 when 90deg, 2 when opposite


# 8-connected moves (dy, dx)
ACTIONS_8 = [
    (-1, 0),  # N
    (-1, 1),  # NE
    (0, 1),   # E
    (1, 1),   # SE
    (1, 0),   # S
    (1, -1),  # SW
    (0, -1),  # W
    (-1, -1), # NW
]


# -------------------------
# Config
# -------------------------
@dataclass
class TrainConfig:
    # Experiment
    exp_name: str = "dsa_rl"
    out_dir: str = "runs"
    seed: int = 0
    device: str = "cuda"  # "cpu" also ok

    # Environment
    H: int = 128
    W: int = 128
    crop: int = 32
    max_steps: int = 160

    # Domain randomization / curriculum knobs
    use_distractors: int = 1
    width_min: int = 2
    width_max: int = 8
    noise_prob: float = 1.0
    invert_prob: float = 0.5
    min_intensity: float = 0.2

    # Motion
    step_alpha: float = 1.0  # IMPORTANT for tight curves: 1.0 is usually smoother than 2.0
    action_repeat: int = 1

    # Stopping formulation
    use_stop_action: int = 1           # if 1: action space includes STOP
    use_termination_head: int = 0      # if 1: separate stop head (recommended long-term)
    stop_radius: float = 5.0           # distance to end to be considered "correct stop"
    allow_bad_stop_continue: int = 1   # if 1: wrong stop does NOT terminate episode

    # Reward weights
    w_precision: float = 2.0           # centerline closeness shaping (gaussian of distance)
    w_progress: float = 0.30           # forward progress reward
    w_align: float = 0.20              # direction alignment with GT tangent
    w_turn: float = 0.20               # zig-zag penalty weight (turning)
    w_step: float = 0.01               # per-step penalty
    w_offtrack: float = 3.0            # penalty when too far
    offtrack_thresh: float = 10.0      # beyond this => "off track"

    # Stop rewards
    r_stop_correct: float = 40.0
    r_stop_wrong: float = -2.0
    r_end_bonus_per_step: float = 0.2  # small reward if near end but not stopped

    # PPO
    total_steps: int = 2_000_000
    rollout_steps: int = 4096
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_eps: float = 0.2
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    lr: float = 3e-4
    update_epochs: int = 8
    minibatch_size: int = 512

    # Checkpointing / eval
    ckpt_every_updates: int = 20
    eval_every_updates: int = 10
    eval_episodes: int = 50


# -------------------------
# Environment
# -------------------------
class DSAPathEnv:
    """
    Observation (actor):
      4 channels: current crop, prev crop, prev2 crop, path_history crop
    Critic gets privileged extra channel: GT mask crop (optional).
    """

    def __init__(self, cfg: TrainConfig):
        self.cfg = cfg
        self.cm = CurveMakerFlexible(h=cfg.H, w=cfg.W, seed=cfg.seed)

        self.img = None
        self.gt_mask = None
        self.gt_poly = None

        self.path_mask = None
        self.history_pos: List[Tuple[float, float]] = []

        self.agent = (0.0, 0.0)
        self.prev_idx = 0
        self.steps = 0
        self.L_prev = 999.0

        self.done = False

    @property
    def n_actions(self) -> int:
        base = 8
        if self.cfg.use_stop_action:
            return base + 1
        return base

    def reset(self) -> Dict[str, np.ndarray]:
        cfg = self.cfg
        if cfg.use_distractors:
            img, mask, pts_all = self.cm.sample_with_distractors(
                width_range=(cfg.width_min, cfg.width_max),
                noise_prob=cfg.noise_prob,
                invert_prob=cfg.invert_prob,
                min_intensity=cfg.min_intensity
            )
        else:
            img, mask, pts_all = self.cm.sample_curve(
                width_range=(cfg.width_min, cfg.width_max),
                noise_prob=cfg.noise_prob,
                invert_prob=cfg.invert_prob,
                min_intensity=cfg.min_intensity,
                branches=False
            )

        self.img = img.astype(np.float32)
        self.gt_mask = mask.astype(np.float32)
        self.gt_poly = pts_all[0].astype(np.float32)
        if len(self.gt_poly) < 10:
            # rare degenerate case; regenerate
            return self.reset()

        # agent start: slightly into the curve to give momentum (optional)
        start_idx = 5 if len(self.gt_poly) > 12 else 0
        curr = self.gt_poly[start_idx]
        p1 = self.gt_poly[max(0, start_idx - 1)]
        p2 = self.gt_poly[max(0, start_idx - 2)]

        self.agent = (float(curr[0]), float(curr[1]))
        self.history_pos = [tuple(p2), tuple(p1), tuple(curr)]
        self.prev_idx = start_idx

        self.path_mask = np.zeros_like(self.img, dtype=np.float32)
        self._stamp_path(self.agent)

        self.steps = 0
        self.done = False
        self.L_prev = distance_to_polyline_pointwise(self.agent, self.gt_poly)
        return self.obs()

    def _stamp_path(self, pos_yx: Tuple[float, float], radius: int = 1):
        y, x = int(pos_yx[0]), int(pos_yx[1])
        H, W = self.cfg.H, self.cfg.W
        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                yy = clamp(y + dy, 0, H - 1)
                xx = clamp(x + dx, 0, W - 1)
                self.path_mask[yy, xx] = 1.0

    def obs(self) -> Dict[str, np.ndarray]:
        cfg = self.cfg
        (y0, x0) = self.history_pos[-1]
        (y1, x1) = self.history_pos[-2]
        (y2, x2) = self.history_pos[-3]

        ch0 = crop32(self.img, int(y0), int(x0), size=cfg.crop)
        ch1 = crop32(self.img, int(y1), int(x1), size=cfg.crop)
        ch2 = crop32(self.img, int(y2), int(x2), size=cfg.crop)
        ch3 = crop32(self.path_mask, int(y0), int(x0), size=cfg.crop)
        actor = np.stack([ch0, ch1, ch2, ch3], axis=0).astype(np.float32)

        gt_crop = crop32(self.gt_mask, int(y0), int(x0), size=cfg.crop)[None, ...].astype(np.float32)
        return {"actor": actor, "critic_gt": gt_crop}

    def step(self, action: int) -> Tuple[Dict[str, np.ndarray], float, bool, Dict[str, Any]]:
        cfg = self.cfg
        info: Dict[str, Any] = {}
        self.steps += 1

        # STOP action
        if cfg.use_stop_action and action == 8:
            dist_to_end = distance_to_polyline_pointwise(self.agent, self.gt_poly[-1:])  # end point only
            info["dist_to_end"] = dist_to_end
            if dist_to_end < cfg.stop_radius:
                # correct stop
                r = cfg.r_stop_correct
                done = True
                info["stopped_correctly"] = True
                info["reached_end"] = True
                self.done = done
                return self.obs(), r, done, info
            else:
                # wrong stop
                r = cfg.r_stop_wrong
                info["stopped_correctly"] = False
                info["reached_end"] = False
                if cfg.allow_bad_stop_continue:
                    return self.obs(), r, False, info
                else:
                    self.done = True
                    return self.obs(), r, True, info

        # MOVE action
        a = int(action)
        dy, dx = ACTIONS_8[a]
        ny = clamp(self.agent[0] + dy * cfg.step_alpha, 0, cfg.H - 1)
        nx = clamp(self.agent[1] + dx * cfg.step_alpha, 0, cfg.W - 1)
        self.agent = (float(ny), float(nx))
        self._stamp_path(self.agent)

        # update history
        self.history_pos.append(self.agent)
        self.history_pos = self.history_pos[-3:]

        # ----- metrics for reward -----
        L_t = distance_to_polyline_pointwise(self.agent, self.gt_poly)
        best_idx = nearest_gt_index(self.agent, self.gt_poly)
        progress_delta = best_idx - self.prev_idx

        # clamp monotonic progress (helps)
        if best_idx > self.prev_idx:
            self.prev_idx = best_idx

        # precision shaping (gaussian of distance)
        sigma = 1.0
        precision = math.exp(-(L_t * L_t) / (2.0 * sigma * sigma))

        # progress reward
        prog = float(max(0, progress_delta))

        # alignment reward (direction ~ tangent)
        # tangent via local difference
        k = 3
        i0 = clamp(best_idx - k, 0, len(self.gt_poly) - 1)
        i1 = clamp(best_idx + k, 0, len(self.gt_poly) - 1)
        gt_vec = self.gt_poly[i1] - self.gt_poly[i0]
        move_vec = np.array([self.history_pos[-1][0] - self.history_pos[-2][0],
                             self.history_pos[-1][1] - self.history_pos[-2][1]], dtype=np.float32)
        align = max(0.0, cosine(gt_vec, move_vec)) if np.linalg.norm(move_vec) > 1e-6 else 0.0

        # turn penalty
        turn = turning_penalty(self.history_pos[-3], self.history_pos[-2], self.history_pos[-1])

        # offtrack penalty
        offtrack = 1.0 if (L_t > cfg.offtrack_thresh) else 0.0

        # end vicinity reward
        dist_to_end = float(np.linalg.norm(np.array(self.agent) - self.gt_poly[-1]))
        end_bonus = cfg.r_end_bonus_per_step if dist_to_end < cfg.stop_radius else 0.0

        # assemble reward
        r = (
            cfg.w_precision * precision
            + cfg.w_progress * prog
            + cfg.w_align * align
            - cfg.w_turn * turn
            - cfg.w_step
            - cfg.w_offtrack * offtrack
            + end_bonus
        )

        # termination conditions
        done = False
        if self.steps >= cfg.max_steps:
            done = True
        # hard terminate if very off-track for too long (simple)
        if L_t > (cfg.offtrack_thresh * 2.0):
            done = True
            info["hard_offtrack"] = True

        info.update({
            "L_t": L_t,
            "precision": precision,
            "progress_delta": progress_delta,
            "align": align,
            "turn": turn,
            "offtrack": offtrack,
            "dist_to_end": dist_to_end,
            "best_idx": best_idx,
            "steps": self.steps,
        })

        self.L_prev = L_t
        self.done = done
        return self.obs(), float(r), bool(done), info


# -------------------------
# Model
# -------------------------
class ActorCritic(nn.Module):
    def __init__(self, n_actions: int, use_termination_head: bool):
        super().__init__()
        self.use_termination_head = use_termination_head

        # Actor CNN for 4x32x32
        self.actor_cnn = nn.Sequential(
            nn.Conv2d(4, 32, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=2, padding=1),
            nn.ReLU(),
        )

        # Critic CNN sees actor obs + privileged gt crop (1 channel)
        self.critic_cnn = nn.Sequential(
            nn.Conv2d(5, 32, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=2, padding=1),
            nn.ReLU(),
        )

        # Flatten sizes
        with torch.no_grad():
            dummy_a = torch.zeros(1, 4, 32, 32)
            dummy_c = torch.zeros(1, 5, 32, 32)
            fa = self.actor_cnn(dummy_a).flatten(1).shape[1]
            fc = self.critic_cnn(dummy_c).flatten(1).shape[1]

        self.actor_head = nn.Sequential(
            nn.Linear(fa, 256),
            nn.ReLU(),
            nn.Linear(256, n_actions),
        )
        self.critic_head = nn.Sequential(
            nn.Linear(fc, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )

        if self.use_termination_head:
            self.term_head = nn.Sequential(
                nn.Linear(fa, 128),
                nn.ReLU(),
                nn.Linear(128, 1)  # logits for stop prob
            )

        self.apply(self._init)

    def _init(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
            if m.bias is not None:
                nn.init.constant_(m.bias, 0.0)

    def forward(self, actor_obs: torch.Tensor, critic_gt: torch.Tensor):
        # actor
        fa = self.actor_cnn(actor_obs).flatten(1)
        logits = self.actor_head(fa)

        # critic
        critic_in = torch.cat([actor_obs, critic_gt], dim=1)
        fc = self.critic_cnn(critic_in).flatten(1)
        value = self.critic_head(fc).squeeze(-1)

        term_logits = None
        if self.use_termination_head:
            term_logits = self.term_head(fa).squeeze(-1)

        return logits, value, term_logits


# -------------------------
# Rollout Buffer
# -------------------------
class RolloutBuffer:
    def __init__(self, T: int, obs_shape=(4, 32, 32)):
        self.T = T
        self.actor_obs = np.zeros((T, *obs_shape), dtype=np.float32)
        self.critic_gt = np.zeros((T, 1, obs_shape[1], obs_shape[2]), dtype=np.float32)
        self.actions = np.zeros((T,), dtype=np.int64)
        self.logp = np.zeros((T,), dtype=np.float32)
        self.values = np.zeros((T,), dtype=np.float32)
        self.rewards = np.zeros((T,), dtype=np.float32)
        self.dones = np.zeros((T,), dtype=np.float32)

        # optional: termination labels/logp
        self.term_logp = None
        self.term_taken = None

    def enable_termination(self):
        self.term_logp = np.zeros((self.T,), dtype=np.float32)
        self.term_taken = np.zeros((self.T,), dtype=np.float32)

    def add(self, t: int, obs: Dict[str, np.ndarray], action: int, logp: float, value: float,
            reward: float, done: bool, term_logp=None, term_taken=None):
        self.actor_obs[t] = obs["actor"]
        self.critic_gt[t] = obs["critic_gt"]
        self.actions[t] = action
        self.logp[t] = logp
        self.values[t] = value
        self.rewards[t] = reward
        self.dones[t] = float(done)
        if self.term_logp is not None and term_logp is not None:
            self.term_logp[t] = float(term_logp)
            self.term_taken[t] = float(term_taken)

    def compute_gae(self, last_value: float, gamma: float, lam: float):
        T = self.T
        adv = np.zeros((T,), dtype=np.float32)
        last_gae = 0.0
        for t in reversed(range(T)):
            next_nonterminal = 1.0 - self.dones[t]
            next_value = last_value if t == T - 1 else self.values[t + 1]
            delta = self.rewards[t] + gamma * next_value * next_nonterminal - self.values[t]
            last_gae = delta + gamma * lam * next_nonterminal * last_gae
            adv[t] = last_gae
        ret = adv + self.values
        return adv, ret


# -------------------------
# Logging
# -------------------------
class CSVLogger:
    def __init__(self, path: str, fieldnames: List[str]):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.path = path
        self.fieldnames = fieldnames
        self._init_file()

    def _init_file(self):
        if not os.path.exists(self.path):
            with open(self.path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=self.fieldnames)
                writer.writeheader()

    def log(self, row: Dict[str, Any]):
        with open(self.path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.fieldnames)
            writer.writerow({k: row.get(k, "") for k in self.fieldnames})


# -------------------------
# Evaluation
# -------------------------
def evaluate(cfg: TrainConfig, model: ActorCritic, device: torch.device, episodes: int) -> Dict[str, float]:
    env = DSAPathEnv(cfg)
    model.eval()

    successes = 0
    offtrack_eps = 0
    mean_L = []
    mean_turn = []
    mean_align = []
    stop_tp = stop_fp = stop_fn = 0

    for _ in range(episodes):
        obs = env.reset()
        ep_L = []
        ep_turn = []
        ep_align = []
        stopped_correctly = False
        stopped_attempted = False
        reached_end = False

        for _t in range(cfg.max_steps):
            aobs = torch.tensor(obs["actor"][None, ...], device=device)
            cgt = torch.tensor(obs["critic_gt"][None, ...], device=device)
            with torch.no_grad():
                logits, value, term_logits = model(aobs, cgt)
                dist = Categorical(logits=logits)
                action = int(torch.argmax(dist.probs, dim=-1).item())  # greedy eval

                # optional termination head
                if cfg.use_termination_head:
                    p_stop = torch.sigmoid(term_logits)[0].item()
                    if p_stop > 0.5:
                        stopped_attempted = True
                        # emulate stop as an env "STOP"
                        dist_to_end = float(np.linalg.norm(np.array(env.agent) - env.gt_poly[-1]))
                        if dist_to_end < cfg.stop_radius:
                            stopped_correctly = True
                            reached_end = True
                            break

            obs, r, done, info = env.step(action)

            ep_L.append(float(info.get("L_t", 0.0)))
            ep_turn.append(float(info.get("turn", 0.0)))
            ep_align.append(float(info.get("align", 0.0)))

            if done:
                reached_end = bool(info.get("reached_end", False)) or (float(info.get("dist_to_end", 999.0)) < cfg.stop_radius)
                if info.get("hard_offtrack", False) or info.get("offtrack", 0.0) > 0:
                    offtrack_eps += 1
                if info.get("stopped_correctly", False):
                    stopped_correctly = True
                    stopped_attempted = True
                break

        # stopping confusion (if stop action exists)
        if cfg.use_stop_action:
            # if episode ended with correct stop => TP
            if stopped_correctly:
                stop_tp += 1
            else:
                # if it attempted stop but wrong => FP; if never stopped and reached end-ish => FN
                # we approximate: reached_end if final position is within stop_radius
                dist_to_end = float(np.linalg.norm(np.array(env.agent) - env.gt_poly[-1]))
                if dist_to_end < cfg.stop_radius:
                    reached_end = True
                if stopped_attempted:
                    stop_fp += 1
                else:
                    if reached_end:
                        stop_fn += 1

        if len(ep_L) > 0:
            mean_L.append(float(np.mean(ep_L)))
            mean_turn.append(float(np.mean(ep_turn)))
            mean_align.append(float(np.mean(ep_align)))

        if stopped_correctly or (cfg.use_termination_head and reached_end):
            successes += 1

    model.train()

    out = {
        "eval/success_rate": successes / max(1, episodes),
        "eval/offtrack_rate": offtrack_eps / max(1, episodes),
        "eval/mean_L": float(np.mean(mean_L)) if mean_L else 0.0,
        "eval/mean_turn": float(np.mean(mean_turn)) if mean_turn else 0.0,
        "eval/mean_align": float(np.mean(mean_align)) if mean_align else 0.0,
    }
    if cfg.use_stop_action:
        prec = stop_tp / max(1, (stop_tp + stop_fp))
        rec = stop_tp / max(1, (stop_tp + stop_fn))
        out["eval/stop_precision"] = prec
        out["eval/stop_recall"] = rec
    return out


# -------------------------
# PPO Trainer
# -------------------------
def ppo_update(cfg: TrainConfig, model: ActorCritic, opt: torch.optim.Optimizer,
               buf: RolloutBuffer, adv: np.ndarray, ret: np.ndarray, device: torch.device):
    T = buf.T
    # normalize advantages
    adv_t = (adv - adv.mean()) / (adv.std() + 1e-8)

    actor_obs = torch.tensor(buf.actor_obs, device=device)
    critic_gt = torch.tensor(buf.critic_gt, device=device)
    actions = torch.tensor(buf.actions, device=device)
    old_logp = torch.tensor(buf.logp, device=device)
    old_values = torch.tensor(buf.values, device=device)
    returns = torch.tensor(ret, device=device)
    advantages = torch.tensor(adv_t, device=device)

    idxs = np.arange(T)
    clip_eps = cfg.clip_eps

    for _epoch in range(cfg.update_epochs):
        np.random.shuffle(idxs)
        for start in range(0, T, cfg.minibatch_size):
            mb = idxs[start:start + cfg.minibatch_size]
            mb_a = actor_obs[mb]
            mb_c = critic_gt[mb]
            mb_act = actions[mb]
            mb_oldlogp = old_logp[mb]
            mb_adv = advantages[mb]
            mb_ret = returns[mb]

            logits, value, term_logits = model(mb_a, mb_c)
            dist = Categorical(logits=logits)
            logp = dist.log_prob(mb_act)

            ratio = torch.exp(logp - mb_oldlogp)
            surr1 = ratio * mb_adv
            surr2 = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * mb_adv
            policy_loss = -torch.mean(torch.min(surr1, surr2))

            # value loss
            v_loss = F.mse_loss(value, mb_ret)

            # entropy
            ent = torch.mean(dist.entropy())

            loss = policy_loss + cfg.vf_coef * v_loss - cfg.ent_coef * ent

            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
            opt.step()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--exp_name", type=str, default="dsa_rl")
    p.add_argument("--out_dir", type=str, default="runs")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default="cuda")

    # quick variant knobs
    p.add_argument("--use_distractors", type=int, default=1)
    p.add_argument("--use_stop_action", type=int, default=1)
    p.add_argument("--use_termination_head", type=int, default=0)
    p.add_argument("--step_alpha", type=float, default=1.0)

    p.add_argument("--w_turn", type=float, default=0.20)
    p.add_argument("--w_align", type=float, default=0.20)
    p.add_argument("--w_precision", type=float, default=2.0)
    p.add_argument("--w_progress", type=float, default=0.30)

    p.add_argument("--total_steps", type=int, default=2_000_000)
    p.add_argument("--rollout_steps", type=int, default=4096)

    args = p.parse_args()

    cfg = TrainConfig(
        exp_name=args.exp_name,
        out_dir=args.out_dir,
        seed=args.seed,
        device=args.device,
        use_distractors=args.use_distractors,
        use_stop_action=args.use_stop_action,
        use_termination_head=args.use_termination_head,
        step_alpha=args.step_alpha,
        w_turn=args.w_turn,
        w_align=args.w_align,
        w_precision=args.w_precision,
        w_progress=args.w_progress,
        total_steps=args.total_steps,
        rollout_steps=args.rollout_steps,
    )

    # output dirs
    run_dir = os.path.join(cfg.out_dir, cfg.exp_name)
    os.makedirs(run_dir, exist_ok=True)
    ckpt_dir = os.path.join(run_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    # save config
    with open(os.path.join(run_dir, "config.json"), "w") as f:
        json.dump(asdict(cfg), f, indent=2)

    set_seed(cfg.seed)
    device = torch.device(cfg.device if torch.cuda.is_available() or cfg.device == "cpu" else "cpu")

    env = DSAPathEnv(cfg)
    model = ActorCritic(n_actions=env.n_actions, use_termination_head=bool(cfg.use_termination_head)).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    writer = None
    if TENSORBOARD_OK:
        writer = SummaryWriter(log_dir=os.path.join(run_dir, "tb"))

    csv_fields = [
        "update", "env_steps",
        "train/ep_return_mean", "train/ep_len_mean",
        "train/mean_L", "train/mean_turn", "train/offtrack_rate",
        "eval/success_rate", "eval/offtrack_rate", "eval/mean_L", "eval/mean_turn", "eval/mean_align",
        "eval/stop_precision", "eval/stop_recall",
        "time_sec"
    ]
    csv_logger = CSVLogger(os.path.join(run_dir, "metrics.csv"), csv_fields)

    # training loop
    obs = env.reset()

    env_steps = 0
    update = 0
    t0 = time.time()

    # episode accumulators for training metrics
    ep_returns = []
    ep_lens = []
    ep_Ls = []
    ep_turns = []
    ep_offtracks = []

    curr_ep_ret = 0.0
    curr_ep_len = 0
    curr_ep_L = []
    curr_ep_turn = []
    curr_ep_off = 0

    while env_steps < cfg.total_steps:
        remaining_steps = cfg.total_steps - env_steps
        rollout_steps = min(cfg.rollout_steps, remaining_steps)
        buf = RolloutBuffer(rollout_steps, obs_shape=(4, cfg.crop, cfg.crop))

        # rollout
        for t in range(rollout_steps):
            aobs = torch.tensor(obs["actor"][None, ...], device=device)
            cgt = torch.tensor(obs["critic_gt"][None, ...], device=device)
            with torch.no_grad():
                logits, value, term_logits = model(aobs, cgt)
                dist = Categorical(logits=logits)
                action = dist.sample()
                logp = dist.log_prob(action)

                # termination head (optional): sample stop decision but do NOT end episode here;
                # best practice is: add an explicit STOP action OR map stop decision to STOP.
                # For now, we keep it simple: if termination head says stop, we execute STOP action.
                if cfg.use_termination_head and cfg.use_stop_action:
                    p_stop = torch.sigmoid(term_logits)[0].item()
                    if p_stop > 0.5:
                        action = torch.tensor([8], device=device)
                        logp = dist.log_prob(action.clamp(max=env.n_actions - 1))

            act = int(action.item())
            next_obs, r, done, info = env.step(act)

            buf.add(
                t=t, obs=obs, action=act, logp=float(logp.item()),
                value=float(value.item()), reward=float(r), done=bool(done)
            )

            # training metric accum
            curr_ep_ret += float(r)
            curr_ep_len += 1
            if "L_t" in info:
                curr_ep_L.append(float(info["L_t"]))
            if "turn" in info:
                curr_ep_turn.append(float(info["turn"]))
            if float(info.get("offtrack", 0.0)) > 0:
                curr_ep_off = 1

            obs = next_obs
            env_steps += 1

            if done:
                ep_returns.append(curr_ep_ret)
                ep_lens.append(curr_ep_len)
                ep_Ls.append(float(np.mean(curr_ep_L)) if curr_ep_L else 0.0)
                ep_turns.append(float(np.mean(curr_ep_turn)) if curr_ep_turn else 0.0)
                ep_offtracks.append(curr_ep_off)

                obs = env.reset()
                curr_ep_ret = 0.0
                curr_ep_len = 0
                curr_ep_L = []
                curr_ep_turn = []
                curr_ep_off = 0

        # bootstrap last value
        with torch.no_grad():
            aobs = torch.tensor(obs["actor"][None, ...], device=device)
            cgt = torch.tensor(obs["critic_gt"][None, ...], device=device)
            _, last_v, _ = model(aobs, cgt)
            last_v = float(last_v.item())

        adv, ret = buf.compute_gae(last_value=last_v, gamma=cfg.gamma, lam=cfg.gae_lambda)
        ppo_update(cfg, model, opt, buf, adv, ret, device)

        update += 1

        # aggregate train metrics
        train_ep_return_mean = float(np.mean(ep_returns[-50:])) if ep_returns else 0.0
        train_ep_len_mean = float(np.mean(ep_lens[-50:])) if ep_lens else 0.0
        train_mean_L = float(np.mean(ep_Ls[-50:])) if ep_Ls else 0.0
        train_mean_turn = float(np.mean(ep_turns[-50:])) if ep_turns else 0.0
        train_offtrack_rate = float(np.mean(ep_offtracks[-50:])) if ep_offtracks else 0.0

        # eval occasionally
        eval_metrics = {}
        if update % cfg.eval_every_updates == 0:
            eval_metrics = evaluate(cfg, model, device, cfg.eval_episodes)

        # checkpoint occasionally
        if update % cfg.ckpt_every_updates == 0:
            ckpt_path = os.path.join(ckpt_dir, f"model_update_{update:06d}.pt")
            torch.save({
                "cfg": asdict(cfg),
                "model": model.state_dict(),
                "opt": opt.state_dict(),
                "update": update,
                "env_steps": env_steps,
            }, ckpt_path)

        # tensorboard
        if writer is not None:
            writer.add_scalar("train/ep_return_mean", train_ep_return_mean, env_steps)
            writer.add_scalar("train/ep_len_mean", train_ep_len_mean, env_steps)
            writer.add_scalar("train/mean_L", train_mean_L, env_steps)
            writer.add_scalar("train/mean_turn", train_mean_turn, env_steps)
            writer.add_scalar("train/offtrack_rate", train_offtrack_rate, env_steps)
            for k, v in eval_metrics.items():
                writer.add_scalar(k, v, env_steps)

        # csv row
        row = {
            "update": update,
            "env_steps": env_steps,
            "train/ep_return_mean": train_ep_return_mean,
            "train/ep_len_mean": train_ep_len_mean,
            "train/mean_L": train_mean_L,
            "train/mean_turn": train_mean_turn,
            "train/offtrack_rate": train_offtrack_rate,
            "time_sec": float(time.time() - t0),
        }
        row.update(eval_metrics)
        # stop metrics keys might not exist if not evaluated yet
        csv_logger.log(row)

        if update % 5 == 0:
            print(
                f"[upd {update:5d}] steps={env_steps:8d} "
                f"R={train_ep_return_mean:7.2f} L={train_mean_L:5.2f} "
                f"turn={train_mean_turn:5.2f} off={train_offtrack_rate:4.2f} "
                + (f" | eval succ={eval_metrics.get('eval/success_rate', 0.0):.2f} "
                   f"eval L={eval_metrics.get('eval/mean_L', 0.0):.2f}" if eval_metrics else "")
            )

    if writer is not None:
        writer.close()

    # final save
    torch.save({"cfg": asdict(cfg), "model": model.state_dict()}, os.path.join(run_dir, "final_model.pt"))
    print("Done. Saved to:", run_dir)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# Synth_simple_v1.2.py
# Stable PPO tracker on synthetic curves (2D DSA-style)
# - State: crops at p_t, p_{t-1}, p_{t-2} + path crop  -> (4,33,33)
# - Action: 8-neighborhood, step size alpha=2
# - Reward: paper-style (B_t +/- log(eps + |ΔL_t|/D0)) with LOCAL surface distance
# - Policy: CNN (GroupNorm) + LSTM over fixed-length action history (K one-hots)
# - Robustness: NaN/Inf guards, logits clamp, GroupNorm, reward clipping
#
# Usage:
#   python Synth_simple_v1.2.py --train --episodes 20000 --save ckpt_curveppo.pth
#   python Synth_simple_v1.2.py --view  --weights ckpt_curveppo.pth

import argparse, math, random
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

# ---------- your curve generator ----------
from Curve_Generator import CurveMaker

# ---------- globals / utils ----------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
ACTIONS_8 = [(-1, 0), (1, 0), (0,-1), (0, 1), (-1,-1), (-1,1), (1,-1), (1,1)]
STEP_ALPHA = 2
CROP = 33

def set_seeds(seed=123):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if DEVICE == "cuda": torch.cuda.manual_seed_all(seed)

def clamp(v, lo, hi): return max(lo, min(v, hi))

def crop32(img: np.ndarray, cy: int, cx: int, size=CROP):
    """Zero-padded square crop centered at (cy,cx). Always returns (size,size)."""
    h, w = img.shape
    r = size // 2
    y0, y1 = cy - r, cy + r + 1  # [y0,y1)
    x0, x1 = cx - r, cx + r + 1  # [x0,x1)
    sy0, sy1 = clamp(y0, 0, h), clamp(y1, 0, h)
    sx0, sx1 = clamp(x0, 0, w), clamp(x1, 0, w)
    out = np.zeros((size, size), dtype=img.dtype)
    oy0 = sy0 - y0; ox0 = sx0 - x0
    sh  = sy1 - sy0; sw  = sx1 - sx0
    oy1 = oy0 + sh; ox1 = ox0 + sw
    if sh > 0 and sw > 0:
        out[oy0:oy1, ox0:ox1] = img[sy0:sy1, sx0:sx1]
    return out

def fixed_window_history(ahist_list, K, n_actions):
    """(K,n_actions) left-padded with zeros."""
    out = np.zeros((K, n_actions), dtype=np.float32)
    if len(ahist_list) == 0: return out
    tail = ahist_list[-K:]
    out[-len(tail):] = np.stack(tail, axis=0)
    return out

def nearest_gt_index(pt, poly):
    """Return (index, euclidean_distance) of the closest GT poly point to pt=(y,x)."""
    dif = poly - np.array(pt, dtype=np.float32)
    d2 = np.sum(dif * dif, axis=1)
    i = int(np.argmin(d2))
    return i, float(np.sqrt(d2[i]))

# ---------- environment ----------
@dataclass
class CurveEpisode:
    img: np.ndarray
    mask: np.ndarray
    gt_poly: np.ndarray
    start: Tuple[int,int]
    init_dir: Tuple[int,int]

class CurveEnv:
    """Directed curve tracking in 2D."""
    def __init__(self, h=128, w=128, branches=False, max_steps=400,
                 d0=2.0, overlap_dist=1.0, revisit_penalty=0.2, stall_patience=50):
        self.h, self.w = h, w
        self.max_steps = max_steps
        self.cm = CurveMaker(h=h, w=w, thickness=1.5, seed=None)
        self.branches = branches
        self.D0 = d0
        self.overlap_dist = overlap_dist
        self.revisit_penalty = revisit_penalty
        self.stall_patience = stall_patience
        self.reset()

    def reset(self):
        img, mask, pts_all = self.cm.sample_curve(branches=self.branches)
        gt_poly = pts_all[0].astype(np.float32)
        p0 = gt_poly[0].astype(int)
        p1 = gt_poly[min(5, len(gt_poly)-1)].astype(int)
        init_vec = np.sign(np.array([p1[0]-p0[0], p1[1]-p0[1]], dtype=np.int32))
        init_vec[init_vec==0] = 1

        self.ep = CurveEpisode(img=img, mask=mask, gt_poly=gt_poly,
                               start=(int(p0[0]), int(p0[1])),
                               init_dir=(int(init_vec[0]), int(init_vec[1])))

        self.agent = (int(p0[0]), int(p0[1]))
        self.prev  = [self.agent, self.agent]
        self.steps = 0

        self.path_mask = np.zeros_like(mask, dtype=np.float32)
        self.path_points: List[Tuple[int,int]] = [self.agent]
        self.path_mask[self.agent] = 1.0

        # Progress & local distance memory
        self.best_idx = 0
        self.no_progress_steps = 0
        _, d0_local = nearest_gt_index(self.agent, self.ep.gt_poly)
        self.L_prev_local = d0_local

        return self.obs()

    def obs(self):
        p_t, p_1, p_2 = self.agent, self.prev[0], self.prev[1]
        ch0 = crop32(self.ep.img,  p_t[0], p_t[1])
        ch1 = crop32(self.ep.img,  p_1[0], p_1[1])
        ch2 = crop32(self.ep.img,  p_2[0], p_2[1])
        ch3 = crop32(self.path_mask, p_t[0], p_t[1])
        obs = np.stack([ch0, ch1, ch2, ch3], axis=0).astype(np.float32)
        return obs

    def step(self, a_idx: int):
        self.steps += 1
        dy, dx = ACTIONS_8[a_idx]
        ny = clamp(self.agent[0] + dy*STEP_ALPHA, 0, self.h-1)
        nx = clamp(self.agent[1] + dx*STEP_ALPHA, 0, self.w-1)
        new_pos = (ny, nx)

        # Move & mark
        self.prev = [self.agent, self.prev[0]]
        
        # Check revisit BEFORE moving (so we check the destination)
        revisited = self.path_mask[new_pos] > 0.5
        
        self.agent = new_pos
        self.path_points.append(self.agent)
        self.path_mask[self.agent] = 1.0

        # Local surface distance at current position
        idx, d_gt = nearest_gt_index(self.agent, self.ep.gt_poly)
        delta = d_gt - self.L_prev_local
        self.L_prev_local = d_gt

        # ============ REWARD COMPUTATION ============
        r = 0.0
        
        # 1. Base reward: on-curve vs off-curve (small per-step)
        on_curve = d_gt < self.overlap_dist
        if on_curve:
            r += 0.5  # Small positive for being on curve
        else:
            r -= 0.1  # Small penalty for being off
        
        # 2. Progress reward (most important!)
        if delta < -0.2:  # Getting closer to curve
            r += 0.3
        elif delta > 0.2:  # Getting farther
            r -= 0.2
        
        # 3. Revisit penalty (only once!)
        if revisited:
            r -= self.revisit_penalty
        
        # 4. Initial direction guidance (first few steps)
        if self.steps <= 3:
            v0 = np.array(self.ep.init_dir, dtype=np.float32)
            vt = np.array([dy, dx], dtype=np.float32)
            if np.linalg.norm(v0) > 0 and np.linalg.norm(vt) > 0:
                cos_sim = v0.dot(vt) / (np.linalg.norm(v0) * np.linalg.norm(vt))
                if cos_sim < 0:  # Wrong direction
                    r -= 0.3
        
        # 5. Clip to reasonable range
        r = float(np.clip(r, -2.0, 2.0))
        
        # ============ PROGRESS TRACKING ============
        if idx > self.best_idx:
            self.best_idx = idx
            self.no_progress_steps = 0
            # Bonus for making progress along the curve
            r += 0.5
        else:
            self.no_progress_steps += 1
        
        # ============ EPISODE TERMINATION ============
        end_margin = 5
        reached_end = (self.best_idx >= len(self.ep.gt_poly) - 1 - end_margin)
        timeout = (self.steps >= self.max_steps)
        stalled = (self.no_progress_steps >= self.stall_patience)
        
        done = reached_end or stalled or timeout
        
        # Terminal rewards
        if reached_end:
            r += 10.0  # Big success bonus
        elif stalled:
            r -= 2.0   # Penalty for getting stuck
        
        return self.obs(), float(r), done, {
            "overlap": 1.0 if on_curve else 0.0,
            "L_local": d_gt,
            "idx": idx,
            "reached_end": reached_end,
            "stalled": stalled,
            "timeout": timeout
        }

# ---------- model / PPO ----------
def gn(c, g=8):  # simple GroupNorm helper
    g = max(1, min(g, c))
    return nn.GroupNorm(g, c, eps=1e-5, affine=True)

class ActorCritic(nn.Module):
    def __init__(self, n_actions=8, K=8):
        super().__init__()
        self.n_actions = n_actions
        self.K = K
        self.cnn = nn.Sequential(
            nn.Conv2d(4, 32, 3, padding=1, dilation=1), gn(32), nn.PReLU(),
            nn.Conv2d(32,32, 3, padding=2, dilation=2), gn(32), nn.PReLU(),
            nn.Conv2d(32,32, 3, padding=3, dilation=3), gn(32), nn.PReLU(),
            nn.Conv2d(32,64, 1),                         gn(64), nn.PReLU(),
        )
        self.gap = nn.AdaptiveAvgPool2d((1,1))
        self.lstm = nn.LSTM(input_size=n_actions, hidden_size=64, num_layers=1, batch_first=True)
        self.actor = nn.Sequential(nn.Linear(64+64,128), nn.PReLU(), nn.Linear(128, n_actions))
        self.critic = nn.Sequential(nn.Linear(64+64,128), nn.PReLU(), nn.Linear(128, 1))

        # safe init
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, a=0.25, nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

    def forward(self, x, ahist_onehot, hc=None):
        # NaN/Inf guard on inputs
        x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        ahist_onehot = torch.nan_to_num(ahist_onehot, nan=0.0, posinf=0.0, neginf=0.0)

        z = self.cnn(x)                            # (B,64,33,33)
        z = self.gap(z).squeeze(-1).squeeze(-1)    # (B,64)
        out, hc = self.lstm(ahist_onehot, hc)      # (B,K,64)
        h_last = out[:, -1, :]                     # (B,64)
        h = torch.cat([z, h_last], dim=1)          # (B,128)

        logits = self.actor(h)                     # (B,8)
        value  = self.critic(h).squeeze(-1)        # (B,)

        # Clamp logits to avoid inf/nan in softmax
        logits = torch.nan_to_num(logits, nan=0.0, posinf=20.0, neginf=-20.0)
        logits = logits.clamp(-20, 20)
        value  = torch.nan_to_num(value, nan=0.0, posinf=0.0, neginf=0.0)
        return logits, value, hc

class PPO:
    def __init__(self, model: ActorCritic, n_actions=8, clip=0.2, gamma=0.99, lam=0.95,
                 lr=1e-4, epochs=4, minibatch=64, entropy_coef=0.03, value_coef=0.5, max_grad_norm=1.0):
        self.model = model
        self.clip = clip
        self.gamma = gamma
        self.lam = lam
        self.epochs = epochs
        self.minibatch = minibatch
        self.opt = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.max_grad_norm = max_grad_norm

    @staticmethod
    def compute_gae(rewards, values, dones, gamma, lam):
        T = len(rewards)
        adv = np.zeros(T, dtype=np.float32)
        gae = 0.0
        for t in reversed(range(T)):
            next_v = 0.0 if dones[t] else (values[t+1] if t+1 < len(values) else 0.0)
            delta = rewards[t] + gamma * next_v - values[t]
            gae = delta + gamma * lam * (0.0 if dones[t] else gae)
            adv[t] = gae
        ret = adv + values[:T]
        return adv, ret

    def update(self, buf):
        obs      = torch.tensor(np.stack(buf["obs"]), dtype=torch.float32, device=DEVICE)   # (N,4,33,33)
        ahist    = torch.tensor(np.stack(buf["ahist"]), dtype=torch.float32, device=DEVICE) # (N,K,8)
        act      = torch.tensor(np.array(buf["act"]),  dtype=torch.long,    device=DEVICE)  # (N,)
        old_logp = torch.tensor(np.array(buf["logp"]),dtype=torch.float32, device=DEVICE)   # (N,)
        adv      = torch.tensor(np.array(buf["adv"]), dtype=torch.float32, device=DEVICE)   # (N,)
        ret      = torch.tensor(np.array(buf["ret"]), dtype=torch.float32, device=DEVICE)   # (N,)

        # Normalize advantages; guard against degenerate std
        adv = adv - adv.mean()
        adv_std = adv.std()
        adv = adv / (adv_std + 1e-8)

        N = obs.size(0)
        idx = np.arange(N)
        for _ in range(self.epochs):
            np.random.shuffle(idx)
            for s in range(0, N, self.minibatch):
                mb = idx[s:s+self.minibatch]
                x = torch.nan_to_num(obs[mb])
                A = torch.nan_to_num(ahist[mb])

                logits, value, _ = self.model(x, A, None)

                # Safety: if any NaNs slipped in, skip this minibatch
                if not torch.isfinite(logits).all() or not torch.isfinite(value).all():
                    continue

                dist = Categorical(logits=logits)
                logp = dist.log_prob(act[mb])

                ratio = torch.exp(logp - old_logp[mb])
                surr1 = ratio * adv[mb]
                surr2 = torch.clamp(ratio, 1.0 - self.clip, 1.0 + self.clip) * adv[mb]
                policy_loss = -torch.min(surr1, surr2).mean()

                value_loss = F.mse_loss(value, ret[mb])
                entropy = dist.entropy().mean()

                loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy
                self.opt.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                self.opt.step()

# ---------- training / viewing ----------
def train(args):
    set_seeds(123)
    env = CurveEnv(h=128, w=128, branches=args.branches, max_steps=400,
                   d0=2.0, overlap_dist=1.0, revisit_penalty=0.2, stall_patience=50)
    K = 8
    nA = len(ACTIONS_8)
    model = ActorCritic(n_actions=nA, K=K).to(DEVICE)
    ppo = PPO(model, lr=args.lr, gamma=args.gamma, lam=args.lam, clip=args.clip,
              epochs=args.epochs, minibatch=args.minibatch, entropy_coef=args.entropy_coef)

    ep_returns = []
    for ep in range(1, args.episodes+1):
        obs = env.reset()
        done = False
        ahist = []  # rolling one-hots
        traj = {"obs":[], "ahist":[], "act":[], "logp":[], "val":[], "rew":[], "done":[]}
        ep_ret = 0.0

        while not done:
            x = torch.tensor(obs[None], dtype=torch.float32, device=DEVICE)
            A = fixed_window_history(ahist, K, nA)[None, ...]         # (1,K,8)
            A_t = torch.tensor(A, dtype=torch.float32, device=DEVICE)

            logits, value, _ = model(x, A_t, None)
            # If something went off, sanitize logits
            logits = torch.nan_to_num(logits, nan=0.0, posinf=20.0, neginf=-20.0).clamp(-20, 20)
            dist = Categorical(logits=logits)

            action = int(dist.sample().item())
            logp = float(dist.log_prob(torch.tensor(action, device=DEVICE)).item())
            val = float(value.item())

            obs2, r, done, info = env.step(action)

            traj["obs"].append(obs)
            traj["ahist"].append(A[0])  # (K,8)
            traj["act"].append(action)
            traj["logp"].append(logp)
            traj["val"].append(val)
            traj["rew"].append(r)
            traj["done"].append(done)

            a1h = np.zeros(nA, dtype=np.float32); a1h[action] = 1.0
            ahist.append(a1h)

            obs = obs2
            ep_ret += r

        # GAE on CPU-friendly arrays
        values = np.array(traj["val"] + [0.0], dtype=np.float32)  # simple 0 bootstrap
        adv, ret = PPO.compute_gae(np.array(traj["rew"], dtype=np.float32),
                                   values, traj["done"], args.gamma, args.lam)

        buf = {
            "obs":   np.array(traj["obs"], dtype=np.float32),
            "ahist": np.array(traj["ahist"], dtype=np.float32),
            "act":   traj["act"],
            "logp":  traj["logp"],
            "adv":   adv,
            "ret":   ret,
        }
        ppo.update(buf)
        ep_returns.append(ep_ret)

        if ep % 100 == 0:
            avg = float(np.mean(ep_returns[-100:]))
            print(f"Episode {ep:6d} | return(avg100)={avg:7.3f}")
        if args.save and ep % args.save_every == 0:
            torch.save(model.state_dict(), args.save)
            print(f"Saved to {args.save}")

    if args.save:
        torch.save(model.state_dict(), args.save)
        print(f"Saved final weights to {args.save}")

def view(args):
    set_seeds(123)
    env = CurveEnv(h=128, w=128, branches=args.branches, max_steps=400)
    K = 8
    nA = len(ACTIONS_8)
    model = ActorCritic(n_actions=nA, K=K).to(DEVICE)
    if args.weights:
        state = torch.load(args.weights, map_location=DEVICE)
        model.load_state_dict(state)
        print(f"Loaded weights: {args.weights}")
    model.eval()

    obs = env.reset()
    done = False
    ahist = []
    steps = 0
    with torch.no_grad():
        while not done:
            x = torch.tensor(obs[None], dtype=torch.float32, device=DEVICE)
            A = fixed_window_history(ahist, K, nA)[None, ...]
            A_t = torch.tensor(A, dtype=torch.float32, device=DEVICE)
            logits, value, _ = model(x, A_t, None)
            action = int(torch.argmax(logits, dim=1).item())
            obs, r, done, info = env.step(action)
            a1h = np.zeros(nA, dtype=np.float32); a1h[action] = 1.0
            ahist.append(a1h)
            steps += 1
    print(f"[VIEW] steps={steps}  L_end(local)={info['L_local']:.3f}  idx_end={info['idx']}")

# ---------- CLI ----------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--train", action="store_true")
    p.add_argument("--view",  action="store_true")
    p.add_argument("--episodes", type=int, default=5000)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--lam", type=float, default=0.95)
    p.add_argument("--clip", type=float, default=0.2)
    p.add_argument("--epochs", type=int, default=4)
    p.add_argument("--minibatch", type=int, default=64)
    p.add_argument("--entropy_coef", type=float, default=0.1)
    p.add_argument("--save", type=str, default="ckpt_curveppo.pth")
    p.add_argument("--save_every", type=int, default=2000)
    p.add_argument("--weights", type=str, default="")
    p.add_argument("--branches", action="store_true")
    args = p.parse_args()

    if args.train: train(args)
    if args.view:  view(args)

if __name__ == "__main__":
    main()

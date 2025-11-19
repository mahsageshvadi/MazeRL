#!/usr/bin/env python3
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
# Actions: Up, Down, Left, Right, diagonals...
ACTIONS_8 = [(-1, 0), (1, 0), (0,-1), (0, 1), (-1,-1), (-1,1), (1,-1), (1,1)]
STEP_ALPHA = 2.0  # Changed to 2.0 (Paper uses alpha=2 to move faster)
CROP = 33
EPSILON = 1e-6    # For log stability

def set_seeds(seed=123):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if DEVICE == "cuda": torch.cuda.manual_seed_all(seed)

def clamp(v, lo, hi): return max(lo, min(v, hi))

def crop32(img: np.ndarray, cy: int, cx: int, size=CROP):
    """Zero-padded square crop centered at (cy,cx)."""
    h, w = img.shape
    r = size // 2
    y0, y1 = cy - r, cy + r + 1
    x0, x1 = cx - r, cx + r + 1
    sy0, sy1 = clamp(y0, 0, h), clamp(y1, 0, h)
    sx0, sx1 = clamp(x0, 0, w), clamp(x1, 0, w)
    out = np.zeros((size, size), dtype=img.dtype)
    oy0 = sy0 - y0; ox0 = sx0 - x0
    sh  = sy1 - sy0; sw  = sx1 - sx0
    if sh > 0 and sw > 0:
        out[oy0:oy0+sh, ox0:ox0+sw] = img[sy0:sy1, sx0:sx1]
    return out

def fixed_window_history(ahist_list, K, n_actions):
    out = np.zeros((K, n_actions), dtype=np.float32)
    if len(ahist_list) == 0: return out
    tail = ahist_list[-K:]
    out[-len(tail):] = np.stack(tail, axis=0)
    return out

def get_distance_to_poly(pt, poly):
    """Return euclidean distance of pt=(y,x) to closest point on poly."""
    dif = poly - np.array(pt, dtype=np.float32)
    d2 = np.sum(dif * dif, axis=1)
    return np.sqrt(np.min(d2))

@dataclass
class CurveEpisode:
    img: np.ndarray
    mask: np.ndarray
    gt_poly: np.ndarray # Centerline points
    start: Tuple[int,int]
    
class CurveEnv:
    def __init__(self, h=128, w=128, branches=False, max_steps=200):
        self.h, self.w = h, w
        self.max_steps = max_steps
        self.cm = CurveMaker(h=h, w=w, thickness=1.5, seed=None)
        self.branches = branches
        self.reset()

    def reset(self):
        img, mask, pts_all = self.cm.sample_curve(branches=self.branches)
        gt_poly = pts_all[0].astype(np.float32)
        
        # Create a "Ternary" Ground Truth Map for the Critic
        # 1.0 = on centerline, -1.0 = other centerlines (if branches), 0.0 = background
        self.gt_map = np.zeros_like(img)
        # Mark main path as 1.0
        for pt in gt_poly:
            r, c = int(pt[0]), int(pt[1])
            if 0<=r<self.h and 0<=c<self.w:
                self.gt_map[r,c] = 1.0
        # If branches exist, mark them as -1.0 (distractors)
        if len(pts_all) > 1:
            for i in range(1, len(pts_all)):
                for pt in pts_all[i]:
                    r, c = int(pt[0]), int(pt[1])
                    if 0<=r<self.h and 0<=c<self.w:
                        # Don't overwrite the main path
                        if self.gt_map[r,c] != 1.0:
                            self.gt_map[r,c] = -1.0

        p0 = gt_poly[0].astype(int)
        self.ep = CurveEpisode(img=img, mask=mask, gt_poly=gt_poly, start=(p0[0], p0[1]))

        self.agent = (float(p0[0]), float(p0[1]))
        self.prev_pos = (float(p0[0]), float(p0[1]))
        self.history_pos = [self.agent] * 3 # for crop history
        self.steps = 0

        self.path_mask = np.zeros_like(mask, dtype=np.float32)
        self.path_points = [self.agent]
        
        # Initial distance to curve
        self.L_prev = get_distance_to_poly(self.agent, self.ep.gt_poly)
        
        return self.obs()

    def obs(self):
        # State for ACTOR: 3 time steps of Image + 1 Path Mask
        curr = self.history_pos[-1]
        p1 = self.history_pos[-2]
        p2 = self.history_pos[-3]
        
        ch0 = crop32(self.ep.img, int(curr[0]), int(curr[1]))
        ch1 = crop32(self.ep.img, int(p1[0]), int(p1[1]))
        ch2 = crop32(self.ep.img, int(p2[0]), int(p2[1]))
        ch3 = crop32(self.path_mask, int(curr[0]), int(curr[1]))
        
        actor_obs = np.stack([ch0, ch1, ch2, ch3], axis=0).astype(np.float32) # (4, 33, 33)
        
        # State for CRITIC: Needs the Ground Truth Map
        # The paper says Critic input is Actor Input + Ternary GT.
        # We will return GT separately to feed into Critic head.
        gt_crop = crop32(self.gt_map, int(curr[0]), int(curr[1]))
        gt_obs = gt_crop[None, ...] # (1, 33, 33)

        return {"actor": actor_obs, "critic_gt": gt_obs}

    def step(self, a_idx: int):
        self.steps += 1
        dy, dx = ACTIONS_8[a_idx]
        
        # Move agent
        ny = clamp(self.agent[0] + dy * STEP_ALPHA, 0, self.h-1)
        nx = clamp(self.agent[1] + dx * STEP_ALPHA, 0, self.w-1)
        self.prev_pos = self.agent
        self.agent = (ny, nx)
        
        self.history_pos.append(self.agent)
        self.path_points.append(self.agent)
        
        # Update visualization mask (draw a line/dot)
        # Simply marking the pixel for now
        ir, ic = int(ny), int(nx)
        self.path_mask[ir, ic] = 1.0

        # --- REWARD CALCULATION (Paper Eq 3) ---
        # Lt = Distance from current agent pos to closest point on GT centerline
        L_t = get_distance_to_poly(self.agent, self.ep.gt_poly)
        
        # Binary Overlap B_t: Are we "on" the vessel?
        # Assume vessel radius ~1.5, so within 2.0 pixels is a "hit"
        B_t = 1.0 if L_t < 2.0 else 0.0
        
        # Difference in distance
        dist_diff = abs(L_t - self.L_prev)
        
        # Paper Eq 3:
        # If getting closer (L_t < L_prev): Reward = B_t + log(epsilon + diff)
        # If getting further (L_t > L_prev): Reward = B_t - log(epsilon + diff)
        # Note: The paper might imply minimizing surface distance, but for tracking,
        # we simply want to encourage moving TOWARDS the line and staying ON it.
        
        if L_t < self.L_prev:
            # Getting closer
            r = B_t + np.log(EPSILON + dist_diff)
            # Clip high values to prevent explosion when diff is huge
            r = float(np.clip(r, -5.0, 5.0))
        else:
            # Getting further or staying same distance
            r = B_t - np.log(EPSILON + dist_diff)
            r = float(np.clip(r, -5.0, 5.0))

        self.L_prev = L_t
        
        # --- TERMINATION ---
        # 1. End of vessel (agent is close to the last point of GT)
        dist_to_end = np.sqrt((self.agent[0]-self.ep.gt_poly[-1][0])**2 + (self.agent[1]-self.ep.gt_poly[-1][1])**2)
        reached_end = dist_to_end < 5.0
        
        # 2. Off track (Paper says 1.8mm, let's say 6 pixels)
        off_track = L_t > 6.0
        
        # 3. Max Length
        too_long = len(self.path_points) > len(self.ep.gt_poly) * 1.5
        
        done = reached_end or off_track or too_long or (self.steps >= self.max_steps)
        
        if reached_end:
            r += 10.0 # Bonus
        if off_track:
            r -= 2.0 # Penalty

        info = {
            "L_t": L_t,
            "reached_end": reached_end,
            "off_track": off_track
        }
        
        return self.obs(), r, done, info

# ---------- Asymmetric Actor-Critic ----------
def gn(c): return nn.GroupNorm(4, c)

class AsymmetricActorCritic(nn.Module):
    def __init__(self, n_actions=8, K=8):
        super().__init__()
        
        # --- Shared Feature Extractor (Optional, but usually better to keep separate) ---
        # Paper suggests somewhat separate architectures.
        
        # --- ACTOR ENCODER (Sees Image + History) ---
        self.actor_cnn = nn.Sequential(
            nn.Conv2d(4, 32, 3, padding=1), gn(32), nn.PReLU(),
            nn.Conv2d(32, 32, 3, padding=2, dilation=2), gn(32), nn.PReLU(),
            nn.Conv2d(32, 64, 3, padding=3, dilation=3), gn(64), nn.PReLU(),
            nn.AdaptiveAvgPool2d((1,1))
        )
        self.actor_lstm = nn.LSTM(input_size=n_actions, hidden_size=64, batch_first=True)
        self.actor_head = nn.Sequential(
            nn.Linear(64+64, 128), nn.PReLU(),
            nn.Linear(128, n_actions)
        )

        # --- CRITIC ENCODER (Sees Image + History + GROUND TRUTH) ---
        # Input channels: 4 (actor obs) + 1 (GT map) = 5
        self.critic_cnn = nn.Sequential(
            nn.Conv2d(5, 32, 3, padding=1), gn(32), nn.PReLU(),
            nn.Conv2d(32, 32, 3, padding=2, dilation=2), gn(32), nn.PReLU(),
            nn.Conv2d(32, 64, 3, padding=3, dilation=3), gn(64), nn.PReLU(),
            nn.AdaptiveAvgPool2d((1,1))
        )
        # Critic uses the same action history
        self.critic_lstm = nn.LSTM(input_size=n_actions, hidden_size=64, batch_first=True)
        self.critic_head = nn.Sequential(
            nn.Linear(64+64, 128), nn.PReLU(),
            nn.Linear(128, 1)
        )

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
            if m.bias is not None: nn.init.constant_(m.bias, 0)

    def forward(self, actor_obs, critic_gt, ahist_onehot, hc_actor=None, hc_critic=None):
        # --- ACTOR FORWARD ---
        feat_a = self.actor_cnn(actor_obs).flatten(1)      # (B, 64)
        lstm_a, hc_actor = self.actor_lstm(ahist_onehot, hc_actor)
        h_last_a = lstm_a[:, -1, :]                        # (B, 64)
        joint_a = torch.cat([feat_a, h_last_a], dim=1)     # (B, 128)
        logits = self.actor_head(joint_a)

        # --- CRITIC FORWARD ---
        # Concatenate Image Obs and GT Obs for Critic
        # actor_obs: (B,4,33,33), critic_gt: (B,1,33,33) -> (B,5,33,33)
        critic_input = torch.cat([actor_obs, critic_gt], dim=1)
        feat_c = self.critic_cnn(critic_input).flatten(1)
        lstm_c, hc_critic = self.critic_lstm(ahist_onehot, hc_critic)
        h_last_c = lstm_c[:, -1, :]
        joint_c = torch.cat([feat_c, h_last_c], dim=1)
        value = self.critic_head(joint_c).squeeze(-1)

        return logits, value, hc_actor, hc_critic

# ---------- PPO Update (Modified for Dict Obs) ----------
def update_ppo(ppo_opt, model, buf, clip=0.2, epochs=4, minibatch=16, ent_coef=0.05):
    obs_a = torch.tensor(np.stack([x['actor'] for x in buf['obs']]), dtype=torch.float32, device=DEVICE)
    obs_c = torch.tensor(np.stack([x['critic_gt'] for x in buf['obs']]), dtype=torch.float32, device=DEVICE)
    ahist = torch.tensor(np.stack(buf['ahist']), dtype=torch.float32, device=DEVICE)
    act   = torch.tensor(np.array(buf['act']), dtype=torch.long, device=DEVICE)
    old_logp = torch.tensor(np.array(buf['logp']), dtype=torch.float32, device=DEVICE)
    adv   = torch.tensor(np.array(buf['adv']), dtype=torch.float32, device=DEVICE)
    ret   = torch.tensor(np.array(buf['ret']), dtype=torch.float32, device=DEVICE)

    # Normalize Advantage
    adv = (adv - adv.mean()) / (adv.std() + 1e-8)

    N = len(buf['obs'])
    idxs = np.arange(N)
    
    for _ in range(epochs):
        np.random.shuffle(idxs)
        for s in range(0, N, minibatch):
            mb = idxs[s:s+minibatch]
            
            logits, val, _, _ = model(obs_a[mb], obs_c[mb], ahist[mb])
            dist = Categorical(logits=logits)
            logp = dist.log_prob(act[mb])
            entropy = dist.entropy().mean()

            ratio = torch.exp(logp - old_logp[mb])
            surr1 = ratio * adv[mb]
            surr2 = torch.clamp(ratio, 1.0-clip, 1.0+clip) * adv[mb]
            
            p_loss = -torch.min(surr1, surr2).mean()
            v_loss = F.mse_loss(val, ret[mb])
            
            loss = p_loss + 0.5 * v_loss - ent_coef * entropy
            
            ppo_opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            ppo_opt.step()

# ---------- Main Loop ----------
def train(args):
    set_seeds(42)
    env = CurveEnv(h=128, w=128, branches=args.branches)
    K = 8
    nA = len(ACTIONS_8)
    
    model = AsymmetricActorCritic(n_actions=nA, K=K).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=1e-4) # Slightly higher LR for initial convergence

    ep_returns = []
    success_rate = []

    for ep in range(1, args.episodes+1):
        obs_dict = env.reset()
        done = False
        ahist = []
        traj = {"obs":[], "ahist":[], "act":[], "logp":[], "val":[], "rew":[], "done":[]}
        
        ep_ret = 0
        
        while not done:
            # Prepare tensors
            obs_a = torch.tensor(obs_dict['actor'][None], dtype=torch.float32, device=DEVICE)
            obs_c = torch.tensor(obs_dict['critic_gt'][None], dtype=torch.float32, device=DEVICE)
            
            A = fixed_window_history(ahist, K, nA)[None, ...]
            A_t = torch.tensor(A, dtype=torch.float32, device=DEVICE)

            with torch.no_grad():
                logits, value, _, _ = model(obs_a, obs_c, A_t)
                dist = Categorical(logits=logits)
                action = dist.sample().item()
                logp = dist.log_prob(torch.tensor(action, device=DEVICE)).item()
                val = value.item()

            next_obs, r, done, info = env.step(action)

            traj["obs"].append(obs_dict)
            traj["ahist"].append(A[0])
            traj["act"].append(action)
            traj["logp"].append(logp)
            traj["val"].append(val)
            traj["rew"].append(r)
            traj["done"].append(done)
            
            # Update history
            a_onehot = np.zeros(nA); a_onehot[action] = 1.0
            ahist.append(a_onehot)
            obs_dict = next_obs
            ep_ret += r

        ep_returns.append(ep_ret)
        success_rate.append(1 if info['reached_end'] else 0)

        # GAE Calculation
        rews = np.array(traj["rew"])
        vals = np.array(traj["val"] + [0.0]) # Bootstrap 0 for terminal
        delta = rews + 0.9 * vals[1:] - vals[:-1]
        adv = np.zeros_like(rews)
        acc = 0
        for t in reversed(range(len(rews))):
            acc = delta[t] + 0.9 * 0.95 * acc
            adv[t] = acc
        ret = adv + vals[:-1]
        
        traj["adv"] = adv
        traj["ret"] = ret
        
        # Update (per episode or batch up episodes here)
        update_ppo(opt, model, traj)

        if ep % 50 == 0:
            avg_r = np.mean(ep_returns[-50:])
            avg_s = np.mean(success_rate[-50:])
            print(f"Ep {ep} | Avg Rew: {avg_r:.2f} | Success Rate: {avg_s:.2f}")

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--episodes", type=int, default=3000)
    p.add_argument("--branches", action="store_true")
    args = p.parse_args()
    train(args)

if __name__ == "__main__":
    main()
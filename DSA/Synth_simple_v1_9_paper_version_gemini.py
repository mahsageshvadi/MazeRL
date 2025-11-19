#!/usr/bin/env python3
import argparse, math, random
from dataclasses import dataclass
from typing import List, Tuple
import os

import numpy as np
import matplotlib.pyplot as plt # Needed for view()

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

from Curve_Generator import CurveMaker

# ---------- globals / utils ----------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
ACTIONS_8 = [(-1, 0), (1, 0), (0,-1), (0, 1), (-1,-1), (-1,1), (1,-1), (1,1)]
STEP_ALPHA = 2.0
CROP = 33
EPSILON = 1e-6

def set_seeds(seed=123):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if DEVICE == "cuda": torch.cuda.manual_seed_all(seed)

def nearest_gt_index(pt, poly):
    """Return the index of the closest point on the polyline."""
    dif = poly - np.array(pt, dtype=np.float32)
    d2 = np.sum(dif * dif, axis=1)
    return int(np.argmin(d2))
    
def clamp(v, lo, hi): return max(lo, min(v, hi))

def crop32(img: np.ndarray, cy: int, cx: int, size=CROP):
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
    dif = poly - np.array(pt, dtype=np.float32)
    d2 = np.sum(dif * dif, axis=1)
    return np.sqrt(np.min(d2))

@dataclass
class CurveEpisode:
    img: np.ndarray
    mask: np.ndarray
    gt_poly: np.ndarray
    start: Tuple[int,int] # Added to fix TypeError
    
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
        
        # Create Ternary GT Map for Critic
        self.gt_map = np.zeros_like(img)
        for pt in gt_poly:
            r, c = int(pt[0]), int(pt[1])
            if 0<=r<self.h and 0<=c<self.w:
                self.gt_map[r,c] = 1.0
        
        # Add branches as distractors (-1.0)
        if len(pts_all) > 1:
            for i in range(1, len(pts_all)):
                for pt in pts_all[i]:
                    r, c = int(pt[0]), int(pt[1])
                    if 0<=r<self.h and 0<=c<self.w:
                        if self.gt_map[r,c] != 1.0:
                            self.gt_map[r,c] = -1.0

        p0 = gt_poly[0].astype(int)
        # Fixed instantiation
        self.ep = CurveEpisode(img=img, mask=mask, gt_poly=gt_poly, start=(p0[0], p0[1]))

        self.agent = (float(p0[0]), float(p0[1]))
        self.history_pos = [self.agent] * 3 
        self.steps = 0

        # --- PROGRESS TRACKING ---
        self.current_idx = 0
        self.prev_idx = 0
        # -------------------------

        self.path_mask = np.zeros_like(mask, dtype=np.float32)
        self.path_points = [self.agent]
        self.L_prev = get_distance_to_poly(self.agent, self.ep.gt_poly)
        
        return self.obs()

    def obs(self):
        curr = self.history_pos[-1]
        p1 = self.history_pos[-2]
        p2 = self.history_pos[-3]
        
        ch0 = crop32(self.ep.img, int(curr[0]), int(curr[1]))
        ch1 = crop32(self.ep.img, int(p1[0]), int(p1[1]))
        ch2 = crop32(self.ep.img, int(p2[0]), int(p2[1]))
        ch3 = crop32(self.path_mask, int(curr[0]), int(curr[1]))
        
        actor_obs = np.stack([ch0, ch1, ch2, ch3], axis=0).astype(np.float32)
        
        gt_crop = crop32(self.gt_map, int(curr[0]), int(curr[1]))
        gt_obs = gt_crop[None, ...]

        return {"actor": actor_obs, "critic_gt": gt_obs}

    # Fixed Indentation: step is now a method of CurveEnv
    def step(self, a_idx: int):
        self.steps += 1
        dy, dx = ACTIONS_8[a_idx]
        ny = clamp(self.agent[0] + dy * STEP_ALPHA, 0, self.h-1)
        nx = clamp(self.agent[1] + dx * STEP_ALPHA, 0, self.w-1)
        self.agent = (ny, nx)
        
        self.history_pos.append(self.agent)
        self.path_points.append(self.agent)
        
        ir, ic = int(ny), int(nx)
        self.path_mask[ir, ic] = 1.0

        # --- REWARD LOGIC ---
        L_t = get_distance_to_poly(self.agent, self.ep.gt_poly)
        dist_diff = abs(L_t - self.L_prev)
        
        best_idx = nearest_gt_index(self.agent, self.ep.gt_poly)
        progress_delta = best_idx - self.prev_idx
        
        # Base: Log Distance Change
        if L_t < self.L_prev:
            r = np.log(EPSILON + dist_diff)
        else:
            r = -np.log(EPSILON + dist_diff)
        r = float(np.clip(r, -2.0, 2.0))

        on_curve = (L_t < 2.0)
        
        # Bonus for moving forward along the curve
        if on_curve and progress_delta > 0:
            r += 1.0
        # Penalty for loitering (on curve but not moving forward)
        elif on_curve and progress_delta <= 0:
            r -= 0.1
            
        # Step Cost (Time Penalty)
        r -= 0.05 

        self.L_prev = L_t
        self.prev_idx = max(self.prev_idx, best_idx)
        
        # Termination
        dist_to_end = np.sqrt((self.agent[0]-self.ep.gt_poly[-1][0])**2 + (self.agent[1]-self.ep.gt_poly[-1][1])**2)
        reached_end = dist_to_end < 5.0
        off_track = L_t > 6.0
        too_long = len(self.path_points) > len(self.ep.gt_poly) * 2.0
        
        done = reached_end or off_track or too_long or (self.steps >= self.max_steps)
        
        if reached_end:
            r += 50.0 # BIG WIN BONUS
        if off_track:
            r -= 5.0

        info = {"reached_end": reached_end, "steps": self.steps}
        return self.obs(), r, done, info

def gn(c): return nn.GroupNorm(4, c)

class AsymmetricActorCritic(nn.Module):
    def __init__(self, n_actions=8, K=8):
        super().__init__()
        self.actor_cnn = nn.Sequential(
            nn.Conv2d(4, 32, 3, padding=1), gn(32), nn.PReLU(),
            nn.Conv2d(32, 32, 3, padding=2, dilation=2), gn(32), nn.PReLU(),
            nn.Conv2d(32, 64, 3, padding=3, dilation=3), gn(64), nn.PReLU(),
            nn.AdaptiveAvgPool2d((1,1))
        )
        self.actor_lstm = nn.LSTM(input_size=n_actions, hidden_size=64, batch_first=True)
        self.actor_head = nn.Sequential(nn.Linear(128, 128), nn.PReLU(), nn.Linear(128, n_actions))

        self.critic_cnn = nn.Sequential(
            nn.Conv2d(5, 32, 3, padding=1), gn(32), nn.PReLU(),
            nn.Conv2d(32, 32, 3, padding=2, dilation=2), gn(32), nn.PReLU(),
            nn.Conv2d(32, 64, 3, padding=3, dilation=3), gn(64), nn.PReLU(),
            nn.AdaptiveAvgPool2d((1,1))
        )
        self.critic_lstm = nn.LSTM(input_size=n_actions, hidden_size=64, batch_first=True)
        self.critic_head = nn.Sequential(nn.Linear(128, 128), nn.PReLU(), nn.Linear(128, 1))

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
            if m.bias is not None: nn.init.constant_(m.bias, 0)

    def forward(self, actor_obs, critic_gt, ahist_onehot, hc_actor=None, hc_critic=None):
        # ACTOR
        feat_a = self.actor_cnn(actor_obs).flatten(1)
        lstm_a, hc_actor = self.actor_lstm(ahist_onehot, hc_actor)
        joint_a = torch.cat([feat_a, lstm_a[:, -1, :]], dim=1)
        logits = self.actor_head(joint_a)

        # CRITIC
        critic_input = torch.cat([actor_obs, critic_gt], dim=1)
        feat_c = self.critic_cnn(critic_input).flatten(1)
        lstm_c, hc_critic = self.critic_lstm(ahist_onehot, hc_critic)
        joint_c = torch.cat([feat_c, lstm_c[:, -1, :]], dim=1)
        value = self.critic_head(joint_c).squeeze(-1)

        return logits, value, hc_actor, hc_critic

def update_ppo(ppo_opt, model, buf_list, clip=0.2, epochs=4, minibatch=32):
    obs_a = torch.tensor(np.concatenate([x['obs']['actor'] for x in buf_list]), dtype=torch.float32, device=DEVICE)
    obs_c = torch.tensor(np.concatenate([x['obs']['critic_gt'] for x in buf_list]), dtype=torch.float32, device=DEVICE)
    ahist = torch.tensor(np.concatenate([x['ahist'] for x in buf_list]), dtype=torch.float32, device=DEVICE)
    act   = torch.tensor(np.concatenate([x['act'] for x in buf_list]), dtype=torch.long, device=DEVICE)
    logp  = torch.tensor(np.concatenate([x['logp'] for x in buf_list]), dtype=torch.float32, device=DEVICE)
    adv   = torch.tensor(np.concatenate([x['adv'] for x in buf_list]), dtype=torch.float32, device=DEVICE)
    ret   = torch.tensor(np.concatenate([x['ret'] for x in buf_list]), dtype=torch.float32, device=DEVICE)

    if adv.numel() > 1:
        adv_std = adv.std()
        if torch.isnan(adv_std) or adv_std < 1e-6:
            adv_std = torch.tensor(1.0, device=DEVICE)
        adv = (adv - adv.mean()) / (adv_std + 1e-8)
    else:
        adv = adv - adv.mean()

    N = obs_a.shape[0]
    idxs = np.arange(N)
    
    for _ in range(epochs):
        np.random.shuffle(idxs)
        for s in range(0, N, minibatch):
            mb = idxs[s:s+minibatch]
            if len(mb) == 0: continue

            logits, val, _, _ = model(obs_a[mb], obs_c[mb], ahist[mb])
            
            logits = torch.clamp(logits, -20, 20)
            
            dist = Categorical(logits=logits)
            new_logp = dist.log_prob(act[mb])
            entropy = dist.entropy().mean()

            ratio = torch.exp(new_logp - logp[mb])
            surr1 = ratio * adv[mb]
            surr2 = torch.clamp(ratio, 1.0-clip, 1.0+clip) * adv[mb]
            
            p_loss = -torch.min(surr1, surr2).mean()
            v_loss = F.mse_loss(val, ret[mb])
            
            loss = p_loss + 0.5 * v_loss - 0.01 * entropy
            
            ppo_opt.zero_grad()
            loss.backward()
            
            nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            ppo_opt.step()

def train(args):
    set_seeds(42)
    env = CurveEnv(h=128, w=128, branches=args.branches)
    K = 8
    nA = len(ACTIONS_8)
    
    model = AsymmetricActorCritic(n_actions=nA, K=K).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=1e-4)

    ep_returns = []
    success_rate = []
    
    batch_buffer = [] 
    BATCH_SIZE_EPISODES = 8 

    print("Starting training...")
    for ep in range(1, args.episodes+1):
        obs_dict = env.reset()
        done = False
        ahist = []
        
        ep_traj = {"obs":{'actor':[], 'critic_gt':[]}, "ahist":[], "act":[], "logp":[], "val":[], "rew":[]}
        
        ep_ret = 0
        
        while not done:
            obs_a = torch.tensor(obs_dict['actor'][None], dtype=torch.float32, device=DEVICE)
            obs_c = torch.tensor(obs_dict['critic_gt'][None], dtype=torch.float32, device=DEVICE)
            A = fixed_window_history(ahist, K, nA)[None, ...]
            A_t = torch.tensor(A, dtype=torch.float32, device=DEVICE)

            with torch.no_grad():
                logits, value, _, _ = model(obs_a, obs_c, A_t)
                logits = torch.clamp(logits, -20, 20)
                dist = Categorical(logits=logits)
                action = dist.sample().item()
                logp = dist.log_prob(torch.tensor(action, device=DEVICE)).item()
                val = value.item()

            next_obs, r, done, info = env.step(action)

            ep_traj["obs"]['actor'].append(obs_dict['actor'])
            ep_traj["obs"]['critic_gt'].append(obs_dict['critic_gt'])
            ep_traj["ahist"].append(A[0])
            ep_traj["act"].append(action)
            ep_traj["logp"].append(logp)
            ep_traj["val"].append(val)
            ep_traj["rew"].append(r)
            
            a_onehot = np.zeros(nA); a_onehot[action] = 1.0
            ahist.append(a_onehot)
            obs_dict = next_obs
            ep_ret += r

        ep_returns.append(ep_ret)
        success_rate.append(1 if info['reached_end'] else 0)
        
        if len(ep_traj["rew"]) > 2:
            rews = np.array(ep_traj["rew"])
            vals = np.array(ep_traj["val"] + [0.0])
            delta = rews + 0.9 * vals[1:] - vals[:-1]
            adv = np.zeros_like(rews)
            acc = 0
            for t in reversed(range(len(rews))):
                acc = delta[t] + 0.9 * 0.95 * acc
                adv[t] = acc
            ret = adv + vals[:-1]
            
            final_ep_data = {
                "obs": {
                    "actor": np.array(ep_traj["obs"]['actor']),
                    "critic_gt": np.array(ep_traj["obs"]['critic_gt'])
                },
                "ahist": np.array(ep_traj["ahist"]),
                "act": np.array(ep_traj["act"]),
                "logp": np.array(ep_traj["logp"]),
                "adv": adv,
                "ret": ret
            }
            batch_buffer.append(final_ep_data)
        
        if len(batch_buffer) >= BATCH_SIZE_EPISODES:
            update_ppo(opt, model, batch_buffer)
            batch_buffer = [] 

        if ep % 50 == 0:
            avg_r = np.mean(ep_returns[-50:])
            avg_s = np.mean(success_rate[-50:])
            print(f"Ep {ep} | Avg Rew: {avg_r:.2f} | Success Rate: {avg_s:.2f}")

    torch.save(model.state_dict(), "ppo_curve_agent.pth")
    print("Model saved to ppo_curve_agent.pth")

def view(args):
    print("--- Starting Evaluation Rollout ---")
    set_seeds(random.randint(0, 1000)) 
    
    env = CurveEnv(h=128, w=128, branches=args.branches)
    obs_dict = env.reset()
    
    K = 8
    nA = len(ACTIONS_8)
    model = AsymmetricActorCritic(n_actions=nA, K=K).to(DEVICE)
    
    if args.load and os.path.exists(args.load):
        model.load_state_dict(torch.load(args.load, map_location=DEVICE))
        print(f"Loaded weights from {args.load}")
    else:
        print("Warning: No weights loaded, running with random initialization.")
        
    model.eval()
    ahist = []
    done = False
    
    print("Agent is tracking...")
    
    while not done:
        obs_a = torch.tensor(obs_dict['actor'][None], dtype=torch.float32, device=DEVICE)
        dummy_gt = torch.zeros((1, 1, 33, 33), dtype=torch.float32, device=DEVICE)
        
        A = fixed_window_history(ahist, K, nA)[None, ...]
        A_t = torch.tensor(A, dtype=torch.float32, device=DEVICE)

        with torch.no_grad():
            logits, _, _, _ = model(obs_a, dummy_gt, A_t)
            action = torch.argmax(logits, dim=1).item()

        obs_dict, r, done, info = env.step(action)
        
        a_onehot = np.zeros(nA); a_onehot[action] = 1.0
        ahist.append(a_onehot)

    print(f"Done! Steps: {env.steps}")
    print(f"Success: {info['reached_end']}")

    path_y = [p[0] for p in env.path_points]
    path_x = [p[1] for p in env.path_points]
    
    gt_y = [p[0] for p in env.ep.gt_poly]
    gt_x = [p[1] for p in env.ep.gt_poly]

    plt.figure(figsize=(10, 10))
    plt.imshow(env.ep.img, cmap='gray', origin='upper')
    plt.plot(gt_x, gt_y, 'r--', linewidth=2, label='Ground Truth')
    plt.plot(path_x, path_y, 'b.-', markersize=8, linewidth=1, label='Agent Path')
    plt.plot(path_x[0], path_y[0], 'go', markersize=10, label='Start')
    plt.plot(path_x[-1], path_y[-1], 'rx', markersize=10, label='End')

    plt.title(f"Rollout Result (Success: {info['reached_end']})")
    plt.legend()
    plt.tight_layout()
    
    save_path = "rollout_result.png"
    plt.savefig(save_path)
    print(f"Trajectory saved to {save_path}")

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--episodes", type=int, default=3000)
    p.add_argument("--branches", action="store_true")
    p.add_argument("--view", action="store_true", help="Run inference instead of training")
    p.add_argument("--load", type=str, default="ppo_curve_agent.pth", help="Path to model weights")
    args = p.parse_args()

    if args.view:
        view(args)
    else:
        train(args)

if __name__ == "__main__":
    main()
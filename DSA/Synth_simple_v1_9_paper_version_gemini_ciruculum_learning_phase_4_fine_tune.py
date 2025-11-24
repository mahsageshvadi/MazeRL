#!/usr/bin/env python3
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from dataclasses import dataclass

# --- IMPORT GENERATOR ---
try:
    from Curve_Generator_Flexible_For_Ciruculum_learning import CurveMakerFlexible
except ImportError:
    print("ERROR: Could not import generator. Make sure 'Curve_Generator_Flexible_For_Ciruculum_learning.py' is in the folder.")
    exit()

# ---------- GLOBALS ----------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
ACTIONS_8 = [(-1, 0), (1, 0), (0,-1), (0, 1), (-1,-1), (-1,1), (1,-1), (1,1)]
STEP_ALPHA = 2.0
CROP = 33
EPSILON = 1e-6

# --- FIX 1: ROBUST SMART PADDING ---
def clamp(v, lo, hi): return max(lo, min(v, hi))

def crop32(img: np.ndarray, cy: int, cx: int, size=CROP):
    h, w = img.shape
    
    # Check 4 corners to determine background color
    corners = [img[0,0], img[0, w-1], img[h-1, 0], img[h-1, w-1]]
    bg_estimate = np.median(corners)
    
    pad_val = 1.0 if bg_estimate > 0.5 else 0.0
    
    r = size // 2
    y0, y1 = cy - r, cy + r + 1
    x0, x1 = cx - r, cx + r + 1
    
    out = np.full((size, size), pad_val, dtype=img.dtype)
    
    sy0, sy1 = clamp(y0, 0, h), clamp(y1, 0, h)
    sx0, sx1 = clamp(x0, 0, w), clamp(x1, 0, w)
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

def nearest_gt_index(pt, poly):
    dif = poly - np.array(pt, dtype=np.float32)
    d2 = np.sum(dif * dif, axis=1)
    return int(np.argmin(d2))

@dataclass
class CurveEpisode:
    img: np.ndarray
    mask: np.ndarray
    gt_poly: np.ndarray

# ---------- UPDATED ENV WITH GAUSSIAN REWARD & RUNNING START ----------
class CurveEnv:
    def __init__(self, h=128, w=128, max_steps=200):
        self.h, self.w = h, w
        self.max_steps = max_steps
        self.cm = CurveMakerFlexible(h=h, w=w, seed=None)
        self.config = {} 
        self.reset()

    def set_config(self, config):
        self.config = config

    def reset(self):
        # Config defaults to Phase 4 (Hard) if not set
        width = self.config.get('width', (1, 12))
        noise = self.config.get('noise', 1.0)
        invert = self.config.get('invert', 0.5)
        min_int = self.config.get('intensity', 0.15)

        img, mask, pts_all = self.cm.sample_curve(
            width_range=width, 
            noise_prob=noise,
            invert_prob=invert,
            min_intensity=min_int
        )
        
        gt_poly = pts_all[0].astype(np.float32)
        
        self.gt_map = np.zeros_like(img)
        for pt in gt_poly:
            r, c = int(pt[0]), int(pt[1])
            if 0<=r<self.h and 0<=c<self.w:
                self.gt_map[r,c] = 1.0

        self.ep = CurveEpisode(img=img, mask=mask, gt_poly=gt_poly)

        # --- FIX 2: RUNNING START ---
        # Start at index 5 to give history
        start_idx = 5
        if len(gt_poly) < 10: start_idx = 0
        
        curr_pt = gt_poly[start_idx]
        prev_pt = gt_poly[max(0, start_idx-1)]
        prev_pt2 = gt_poly[max(0, start_idx-2)]

        self.agent = (float(curr_pt[0]), float(curr_pt[1]))
        
        # Initialize history along the line (Momentum)
        self.history_pos = [
            (float(prev_pt2[0]), float(prev_pt2[1])),
            (float(prev_pt[0]), float(prev_pt[1])),
            self.agent
        ]
        
        self.steps = 0
        self.prev_idx = start_idx
        self.prev_action = -1 
        self.path_mask = np.zeros_like(mask, dtype=np.float32)
        self.path_points = [self.agent]
        self.path_mask[int(self.agent[0]), int(self.agent[1])] = 1.0
        
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

        L_t = get_distance_to_poly(self.agent, self.ep.gt_poly)
        dist_diff = abs(L_t - self.L_prev)
        best_idx = nearest_gt_index(self.agent, self.ep.gt_poly)
        progress_delta = best_idx - self.prev_idx
        
        # 1. Log Distance
        if L_t < self.L_prev:
            r = np.log(EPSILON + dist_diff)
        else:
            r = -np.log(EPSILON + dist_diff)
        r = float(np.clip(r, -2.0, 2.0))

        # --- FIX 3: GAUSSIAN REWARD ---
        # Bell curve: Max reward at 0 distance, decays quickly
        sigma = 2.0
        precision_score = np.exp(-(L_t**2) / (2 * sigma**2))
        
        if progress_delta > 0:
            # Reward based on how centered we are
            r += precision_score * 1.5 
        elif progress_delta <= 0:
            r -= 0.1
        
        # Smoothness
        if self.prev_action != -1 and self.prev_action != a_idx:
            r -= 0.05 
        self.prev_action = a_idx
        
        r -= 0.05 

        self.L_prev = L_t
        self.prev_idx = max(self.prev_idx, best_idx)
        
        # Termination
        dist_to_end = np.sqrt((self.agent[0]-self.ep.gt_poly[-1][0])**2 + (self.agent[1]-self.ep.gt_poly[-1][1])**2)
        reached_end = dist_to_end < 5.0
        off_track = L_t > 12.0 
        too_long = len(self.path_points) > len(self.ep.gt_poly) * 2.5
        
        done = reached_end or off_track or too_long or (self.steps >= self.max_steps)
        
        if reached_end: r += 50.0
        if off_track:   r -= 5.0

        info = {"reached_end": reached_end}
        return self.obs(), r, done, info

# ---------- MODEL ----------
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
        feat_a = self.actor_cnn(actor_obs).flatten(1)
        lstm_a, hc_actor = self.actor_lstm(ahist_onehot, hc_actor)
        joint_a = torch.cat([feat_a, lstm_a[:, -1, :]], dim=1)
        logits = self.actor_head(joint_a)

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

    if adv.numel() > 1: adv = (adv - adv.mean()) / (adv.std() + 1e-8)
    else: adv = adv - adv.mean()

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

# ---------- PHASE 4 TRAINING LOOP ----------
def train_phase4(args):
    print("--- STARTING PHASE 4: FINE-TUNING (PRECISION) ---")
    
    # 1. Setup
    env = CurveEnv(h=128, w=128)
    
    # Explicit Phase 4 / Phase 3 Config
    config = {'width': (1, 12), 'noise': 1.0, 'invert': 0.5, 'intensity': 0.15}
    env.set_config(config)
    print(f"Config: {config}")
    
    K = 8
    nA = len(ACTIONS_8)
    model = AsymmetricActorCritic(n_actions=nA, K=K).to(DEVICE)
    
    # 2. Load Previous Model
    try:
        print(f"Loading weights from: {args.load_path}")
        model.load_state_dict(torch.load(args.load_path, map_location=DEVICE))
        print("Weights loaded successfully.")
    except FileNotFoundError:
        print(f"ERROR: Model {args.load_path} not found.")
        return

    # 3. Low Learning Rate for Fine-Tuning
    # Using 1e-5 as requested for stability
    opt = torch.optim.Adam(model.parameters(), lr=1e-5)
    
    batch_buffer = []
    BATCH_SIZE_EPISODES = 32
    ep_returns = []
    
    for ep in range(1, args.episodes + 1):
        obs_dict = env.reset()
        done = False
        ahist = []
        ep_traj = {"obs":{'actor':[], 'critic_gt':[]}, "ahist":[], "act":[], "logp":[], "val":[], "rew":[]}
        
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
                "obs": {"actor": np.array(ep_traj["obs"]['actor']), "critic_gt": np.array(ep_traj["obs"]['critic_gt'])},
                "ahist": np.array(ep_traj["ahist"]),
                "act": np.array(ep_traj["act"]),
                "logp": np.array(ep_traj["logp"]),
                "adv": adv, "ret": ret
            }
            batch_buffer.append(final_ep_data)
            ep_returns.append(sum(rews))

        if len(batch_buffer) >= BATCH_SIZE_EPISODES:
            update_ppo(opt, model, batch_buffer)
            batch_buffer = []

        if ep % 100 == 0:
            avg_r = np.mean(ep_returns[-100:])
            # Threshold is slightly different because reward scale changed with Gaussian
            success_cnt = sum(1 for r in ep_returns[-100:] if r > 40)
            print(f"[Phase 4] Ep {ep} | Avg Rew: {avg_r:.2f} | Success: {success_cnt/100:.2f}")

        if ep % 500 == 0:
            save_name = f"ckpt_Phase4_ep{ep}.pth"
            torch.save(model.state_dict(), save_name)
            print(f"Saved checkpoint: {save_name}")

    final_name = "ppo_model_Phase4_final.pth"
    torch.save(model.state_dict(), final_name)
    print(f"--- Phase 4 Complete. Saved to {final_name} ---")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--episodes", type=int, default=5000)
    # Default loads your previous Phase 3 checkpoint
    p.add_argument("--load_path", type=str, default="ckpt_Phase3_ep9000.pth")
    args = p.parse_args()
    train_phase4(args)
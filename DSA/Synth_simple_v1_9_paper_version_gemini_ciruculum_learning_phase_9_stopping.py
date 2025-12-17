#!/usr/bin/env python3
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical
from dataclasses import dataclass

# Import Generator
try:
    from Curve_Generator_Flexible_For_Ciruculum_learning import CurveMakerFlexible
except ImportError:
    exit()

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# 9 ACTIONS NOW: 8 moves + 1 Stop
ACTIONS_9 = [(-1, 0), (1, 0), (0,-1), (0, 1), (-1,-1), (-1,1), (1,-1), (1,1), (0,0)]
STEP_ALPHA = 2.0
CROP = 33

def crop32(img, cy, cx, size=CROP):
    h, w = img.shape
    corners = [img[0,0], img[0, w-1], img[h-1, 0], img[h-1, w-1]]
    bg_estimate = np.median(corners)
    pad_val = 1.0 if bg_estimate > 0.5 else 0.0
    
    r = size // 2
    y0, y1 = cy - r, cy + r + 1
    x0, x1 = cx - r, cx + r + 1
    
    out = np.full((size, size), pad_val, dtype=img.dtype)
    # Clamp logic omitted for brevity, assume standard clamp
    sy0, sy1 = max(y0,0), min(y1,h); sx0, sx1 = max(x0,0), min(x1,w)
    oy0, ox0 = sy0-y0, sx0-x0
    if sy1>sy0 and sx1>sx0: out[oy0:oy0+(sy1-sy0), ox0:ox0+(sx1-sx0)] = img[sy0:sy1, sx0:sx1]
    return out

@dataclass
class CurveEpisode:
    img: np.ndarray; gt_poly: np.ndarray

class CurveEnvStop:
    def __init__(self, h=128, w=128):
        self.h, self.w = h, w
        self.cm = CurveMakerFlexible(h=h, w=w, seed=None)
        self.max_steps = 200
        # Phase 8 config for realism
        self.config = {'width': (2, 6), 'noise': 0.8, 'invert': 0.5} 
        self.reset()

    def reset(self):
        img, _, pts_all = self.cm.sample_curve(width_range=(2,6), noise_prob=0.8)
        gt_poly = pts_all[0].astype(np.float32)
        self.ep = CurveEpisode(img=img, gt_poly=gt_poly)
        
        # Standard Running Start
        start_idx = 5 if len(gt_poly) > 10 else 0
        curr = gt_poly[start_idx]
        p1 = gt_poly[max(0, start_idx-1)]; p2 = gt_poly[max(0, start_idx-2)]
        
        self.agent = (float(curr[0]), float(curr[1]))
        self.history_pos = [tuple(p2), tuple(p1), tuple(curr)]
        self.steps = 0
        self.path_mask = np.zeros_like(img, dtype=np.float32)
        return self.obs()

    def obs(self):
        curr = self.history_pos[-1]
        p1 = self.history_pos[-2]
        p2 = self.history_pos[-3]
        ch0 = crop32(self.ep.img, int(curr[0]), int(curr[1]))
        ch1 = crop32(self.ep.img, int(p1[0]), int(p1[1]))
        ch2 = crop32(self.ep.img, int(p2[0]), int(p2[1]))
        ch3 = crop32(self.path_mask, int(curr[0]), int(curr[1]))
        return torch.tensor(np.stack([ch0, ch1, ch2, ch3], axis=0)[None], dtype=torch.float32, device=DEVICE)

    def step(self, a_idx):
        self.steps += 1
        dist_to_end = np.sqrt((self.agent[0]-self.ep.gt_poly[-1][0])**2 + (self.agent[1]-self.ep.gt_poly[-1][1])**2)
        
        # --- NEW: STOP LOGIC ---
        if a_idx == 8: # STOP ACTION
            if dist_to_end < 6.0: 
                # Good Stop
                return self.obs(), 50.0, True, {"success": True}
            else:
                # Bad Stop (Too early)
                return self.obs(), -5.0, True, {"success": False}

        # Movement Logic
        dy, dx = ACTIONS_9[a_idx]
        ny = self.agent[0] + dy * STEP_ALPHA
        nx = self.agent[1] + dx * STEP_ALPHA
        
        # Check boundaries
        if not (0 <= ny < self.h and 0 <= nx < self.w):
            return self.obs(), -5.0, True, {"success": False} # Wall hit

        self.agent = (ny, nx)
        self.history_pos.append(self.agent)
        self.path_mask[int(ny), int(nx)] = 1.0

        # Reward for movement
        # (Simplified for brevity: You should copy the Gaussian reward from Phase 7 here)
        r = 0.0 
        
        # Penalty for walking past the end without stopping
        if dist_to_end < 3.0 and a_idx != 8:
            r -= 0.5 # Nag the agent to press stop

        done = (self.steps >= self.max_steps)
        return self.obs(), r, done, {"success": False}

# --- UPDATED MODEL (9 Actions) ---
def gn(c): return nn.GroupNorm(4, c)
class Actor9(nn.Module):
    def __init__(self):
        super().__init__()
        self.actor_cnn = nn.Sequential(
            nn.Conv2d(4, 32, 3, padding=1), gn(32), nn.PReLU(),
            nn.Conv2d(32, 64, 3, padding=2, dilation=2), gn(64), nn.PReLU(),
            nn.AdaptiveAvgPool2d((1,1))
        )
        # Input 9 actions (one-hot)
        self.actor_lstm = nn.LSTM(input_size=9, hidden_size=64, batch_first=True)
        # Output 9 logits
        self.actor_head = nn.Sequential(nn.Linear(128, 64), nn.PReLU(), nn.Linear(64, 9))

    def forward(self, obs, ahist):
        feat = self.actor_cnn(obs).flatten(1)
        lstm_out, _ = self.actor_lstm(ahist)
        joint = torch.cat([feat, lstm_out[:, -1, :]], dim=1)
        return self.actor_head(joint)

def train_phase9(args):
    print("--- PHASE 9: LEARNING TO STOP ---")
    env = CurveEnvStop()
    # 9 Actions now
    model = Actor9().to(DEVICE)
    
    # PROBLEM: Can't load 8-action weights into 9-action model directly.
    # SOLUTION: Surgery. Load weights, copy CNN/LSTM, re-init Head.
    try:
        saved_dict = torch.load(args.load_path, map_location=DEVICE)
        model_dict = model.state_dict()
        # Filter out mismatching keys (the head and lstm input size)
        pretrained_dict = {k: v for k, v in saved_dict.items() if k in model_dict and v.shape == model_dict[k].shape}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
        print("Loaded compatible weights. Head re-initialized.")
    except:
        print("Starting fresh (or error loading).")

    opt = torch.optim.Adam(model.parameters(), lr=1e-4) # Higher LR to learn the new action
    
    # Simple Training Loop (Simplified for brevity)
    for ep in range(10000):
        obs = env.reset()
        ahist = torch.zeros((1, 8, 9), device=DEVICE) # 8 steps history, 9 actions dim
        done = False
        while not done:
            with torch.no_grad():
                logits = model(obs, ahist)
                dist = Categorical(logits=logits)
                action = dist.sample().item()
            
            next_obs, r, done, info = env.step(action)
            
            # (Insert PPO Buffer logic here - same as Phase 8)
            # ...
            
            # Update History
            a_onehot = torch.zeros((1, 9), device=DEVICE)
            a_onehot[0, action] = 1.0
            ahist = torch.cat([ahist[:, 1:, :], a_onehot.unsqueeze(1)], dim=1)
            obs = next_obs

        if ep % 100 == 0: print(f"Ep {ep} complete")
        
    torch.save(model.state_dict(), "ppo_model_Phase9_Stop.pth")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--load_path", type=str, default="ppo_model_Phase8_Realism.pth")
    args = parser.parse_args()
    train_phase9(args)
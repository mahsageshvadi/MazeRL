import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.distributions import Categorical
from scipy.ndimage import gaussian_filter
import cv2
import time
import csv
import argparse

# ---------- ARGPARSE ----------
parser = argparse.ArgumentParser()
parser.add_argument("--run_name", type=str, required=True)
parser.add_argument("--out_dir", type=str, required=True)
parser.add_argument("--eps_per_stage", type=int, default=5000)
parser.add_argument("--lr", type=float, default=1e-4)
parser.add_argument("--stop_thresh", type=float, default=0.8)
parser.add_argument("--width_max", type=float, default=20.0)
parser.add_argument("--noise_mult", type=float, default=1.0)
parser.add_argument("--intensity_min", type=float, default=0.2)
args = parser.parse_args()

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
ACTIONS_MOVE = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]
N_MOVE = 8
STEP_ALPHA = 2.0

# Create output dir
os.makedirs(args.out_dir, exist_ok=True)
csv_path = os.path.join(args.out_dir, "metrics.csv")
if not os.path.exists(csv_path):
    with open(csv_path, 'w', newline='') as f:
        csv.writer(f).writerow(['Stage', 'Episode', 'AvgRew', 'SuccRate', 'TimeMin'])

# ---------- GENERATOR ----------
class UltimateMaker:
    def __init__(self, h=128, w=128): self.h, self.w = h, w
    def generate(self, w_range, n_prob, i_range, short=False):
        img = np.zeros((128, 128)); mask = np.zeros((128, 128))
        # Ensure points are not too close to each other
        p = [np.random.randint(20, 108, size=2) for _ in range(4)]
        n_pts = np.random.randint(30, 60) if short else np.random.randint(120, 250)
        t = np.linspace(0, 1, n_pts)
        pts = np.array([(1-ti)**3*p[0]+3*(1-ti)**2*ti*p[1]+3*(1-ti)*ti**2*p[2]+ti**3*p[3] for ti in t])
        pts = np.clip(pts, 0, 127)
        thick = np.random.uniform(*w_range)
        intens = np.random.uniform(*i_range)
        for pt in pts:
            cv2.circle(img, (int(pt[1]), int(pt[0])), int(thick), float(intens), -1)
            cv2.circle(mask, (int(pt[1]), int(pt[0])), int(thick), 1.0, -1)
        if np.random.rand() < n_prob:
            img = np.clip(img + (gaussian_filter(np.random.randn(128,128), 2)*0.3*args.noise_mult), 0, 1)
        return img.astype(np.float32), mask.astype(np.float32), pts

# ---------- ENVIRONMENT ----------
class UltimateEnv:
    def __init__(self): 
        self.m = UltimateMaker()
        self.last_m = (0,0)
    
    def get_patch(self, img, cy, cx, size, norm=True):
        padded = np.pad(img, size, mode='edge')
        patch = padded[int(cy):int(cy+size), int(cx):int(cx+size)]
        if norm and (patch.max() - patch.min() > 1e-5):
            patch = (patch - patch.min()) / (patch.max() - patch.min())
        return patch

    def reset(self, cfg):
        self.cfg = cfg
        self.img, self.mask, self.gt = self.m.generate(cfg['w'], cfg['n'], cfg['i'], cfg.get('s', False))
        # Start at index 5 to ensure a "Running Start"
        self.agent = (float(self.gt[5][0]), float(self.gt[5][1]))
        self.p_mask = np.zeros((128,128))
        self.steps = 0
        return self.obs()

    def obs(self):
        cy, cx = self.agent
        c1 = self.get_patch(self.img, cy, cx, 33)
        c2 = cv2.resize(self.get_patch(self.img, cy, cx, 65), (33,33))
        c3 = self.get_patch(self.img, cy + self.last_m[0]*8, cx + self.last_m[1]*8, 33)
        c4 = self.get_patch(self.p_mask, cy, cx, 33, False)
        return np.stack([c1, c2, c3, c4], axis=0).astype(np.float32)

    def step(self, m_idx, s_val):
        self.steps += 1
        curr_p = np.array(self.agent)
        d_end = np.linalg.norm(curr_p - self.gt[-1])

        # Stop Signal logic
        if s_val > args.stop_thresh:
            succ = d_end < 6.0
            return self.obs(), (200.0 if succ else -50.0), True, succ

        dy, dx = ACTIONS_MOVE[m_idx]
        self.last_m = (dy, dx)
        self.agent = (np.clip(self.agent[0]+dy*STEP_ALPHA,0,127), np.clip(self.agent[1]+dx*STEP_ALPHA,0,127))
        self.p_mask[int(self.agent[0]), int(self.agent[1])] = 1.0
        
        # Distance to centerline
        d_curve = np.sqrt(np.min(np.sum((self.gt - self.agent)**2, axis=1)))
        reward = np.exp(-(d_curve**2) / (2 * (self.cfg['w'][1]/2)**2)) - 0.1
        
        done = False
        # GRACE PERIOD: Agent cannot fail in the first 10 steps
        if self.steps > 10:
            if d_curve > (self.cfg['w'][1]*2 + 15): # Off-track
                reward -= 100.0
                done = True
        
        if self.steps > 350: done = True
        return self.obs(), reward, done, False

# ---------- MODEL ----------
class ACNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(4, 32, 3, padding=1), nn.GroupNorm(4, 32), nn.PReLU(),
            nn.Conv2d(32, 64, 3, padding=1, stride=2), nn.GroupNorm(8, 64), nn.PReLU(),
            nn.Conv2d(64, 128, 3, padding=1), nn.GroupNorm(16, 128), nn.PReLU(),
            nn.AdaptiveAvgPool2d((1,1))
        )
        self.lstm = nn.LSTM(128, 128, batch_first=True)
        self.actor = nn.Linear(128, 8)
        self.stop = nn.Linear(128, 1)
        self.critic = nn.Linear(128, 1)

    def forward(self, x, h=None):
        b, c, hi, wi = x.shape
        f = self.cnn(x).view(b, 1, 128)
        o, h = self.lstm(f, h); o = o.squeeze(1)
        return self.actor(o), torch.sigmoid(self.stop(o)), self.critic(o), h

# ---------- TRAIN ----------
def run():
    env = UltimateEnv(); model = ACNet().to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    start_t = time.time()
    
    stages = [
        {'id': 1, 'w': (2, 4), 'n': 0.0, 'i': (0.8, 1.0)},
        {'id': 2, 'w': (1, 8), 'n': 0.2, 'i': (0.5, 1.0)},
        {'id': 3, 'w': (0.5, args.width_max), 'n': 0.6, 'i': (args.intensity_min, 0.9)},
        {'id': 4, 'w': (2, 5), 'n': 0.1, 'i': (0.5, 1.0), 's': True}
    ]

    for st in stages:
        rewards, succs = [], []
        print(f"\n--- {args.run_name} | Starting Stage {st['id']} ---")
        for ep in range(1, args.eps_per_stage + 1):
            obs = env.reset(st); h = None; done = False; ep_r = 0
            while not done:
                obs_t = torch.tensor(obs[None,...], device=DEVICE)
                logits, s_prob, val, h = model(obs_t, h)
                dist = Categorical(logits=logits); m_idx = dist.sample()
                
                obs, rew, done, is_succ = env.step(m_idx.item(), s_prob.item())
                
                with torch.no_grad():
                    _, _, next_v, _ = model(torch.tensor(obs[None,...], device=DEVICE), h)
                
                # A2C Loss
                advantage = (rew + (0.99 * next_v if not done else 0)) - val
                target_s = 1.0 if np.linalg.norm(np.array(env.agent) - env.gt[-1]) < 6.0 else 0.0
                
                loss = -dist.log_prob(m_idx)*advantage.detach() + \
                       F.binary_cross_entropy(s_prob, torch.tensor([[target_s]], device=DEVICE)) + \
                       0.5 * F.mse_loss(val, rew + (0.99 * next_v.detach() if not done else 0))
                
                opt.zero_grad(); loss.backward(); opt.step()
                h = (h[0].detach(), h[1].detach()); ep_r += rew
            
            rewards.append(ep_r); succs.append(1 if is_succ else 0)
            if ep % 100 == 0:
                ar, asu = np.mean(rewards[-100:]), np.mean(succs[-100:])
                elap = (time.time()-start_t)/60
                print(f"St:{st['id']} Ep:{ep} AvgRew:{ar:.1f} Succ:{asu:.2f} Time:{elap:.1f}m")
                with open(csv_path, 'a', newline='') as f:
                    csv.writer(f).writerow([st['id'], ep, ar, asu, elap])
        
        # SAVE MODEL AFTER EACH STAGE
        save_path = os.path.join(args.out_dir, f"model_stage_{st['id']}.pth")
        torch.save(model.state_dict(), save_path)
        print(f"Saved: {save_path}")

if __name__ == "__main__":
    run()
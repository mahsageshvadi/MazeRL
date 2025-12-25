import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.distributions import Categorical
from scipy.ndimage import gaussian_filter
import cv2
import time
import logging
import csv
import argparse

# ---------- COMMAND LINE ARGUMENTS ----------
parser = argparse.ArgumentParser()
parser.add_argument("--run_name", type=str, default="Exp")
parser.add_argument("--out_dir", type=str, default="./results")
parser.add_argument("--eps_per_stage", type=int, default=5000)
parser.add_argument("--lr", type=float, default=1e-4)
parser.add_argument("--stop_thresh", type=float, default=0.8)
parser.add_argument("--noise_mult", type=float, default=1.0)
parser.add_argument("--width_max", type=float, default=20.0)
parser.add_argument("--intensity_min", type=float, default=0.2)
args = parser.parse_args()

# ---------- GLOBALS ----------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
ACTIONS_MOVE = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]
N_MOVE = 8
STEP_ALPHA = 2.0

# ---------- LOGGING SETUP ----------
os.makedirs(args.out_dir, exist_ok=True)
csv_file = os.path.join(args.out_dir, "metrics.csv")
with open(csv_file, mode='w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['Stage', 'Episode', 'AvgReward', 'SuccessRate', 'Elapsed_Min'])

log_path = os.path.join(args.out_dir, "train_log.txt")
logging.basicConfig(level=logging.INFO, format='%(message)s',
                    handlers=[logging.FileHandler(log_path), logging.StreamHandler()])
logger = logging.getLogger()

# ---------- UTILS & GENERATOR ----------
def get_robust_patch(img, cy, cx, size, normalize=True):
    h, w = img.shape
    r = size // 2
    padded = np.pad(img, size, mode='edge')
    patch = padded[int(cy):int(cy+size), int(cx):int(cx+size)]
    if normalize and (patch.max() - patch.min() > 1e-5):
        patch = (patch - patch.min()) / (patch.max() - patch.min())
    return patch

class MedicalCurveMaker:
    def __init__(self, h=128, w=128): self.h, self.w = h, w
    def generate(self, w_range, n_prob, i_range, short=False):
        img = np.zeros((128, 128)); mask = np.zeros((128, 128))
        p = [np.random.randint(15, 110, size=2) for _ in range(4)]
        n_pts = np.random.randint(20, 45) if short else np.random.randint(120, 250)
        t = np.linspace(0, 1, n_pts)
        pts = np.array([(1-ti)**3*p[0]+3*(1-ti)**2*ti*p[1]+3*(1-ti)*ti**2*p[2]+ti**3*p[3] for ti in t])
        pts = np.clip(pts, 0, 127)
        thick = np.random.uniform(*w_range)
        intens = np.random.uniform(*i_range)
        for pt in pts:
            cv2.circle(img, (int(pt[1]), int(pt[0])), int(thick), float(intens), -1)
            cv2.circle(mask, (int(pt[1]), int(pt[0])), int(thick), 1.0, -1)
        if np.random.rand() < n_prob:
            noise = gaussian_filter(np.random.randn(128,128), 3) * 0.15 * args.noise_mult
            img = np.clip(img + noise, 0, 1)
        return img, mask, pts

# ---------- ENVIRONMENT ----------
class SignalEnv:
    def __init__(self): 
        self.maker = MedicalCurveMaker()
        self.last_move = (0,0)
    def reset(self, cfg):
        self.cfg = cfg
        self.img, self.mask, self.gt = self.maker.generate(cfg['w'], cfg['n'], cfg['i'], cfg.get('s', False))
        self.agent = tuple(self.gt[5])
        self.p_mask = np.zeros((128,128))
        self.steps = 0
        return self.get_obs()
    def get_obs(self):
        cy, cx = self.agent
        c1 = get_robust_patch(self.img, cy, cx, 33)
        c2 = cv2.resize(get_robust_patch(self.img, cy, cx, 65), (33,33))
        c3 = get_robust_patch(self.img, cy + self.last_move[0]*8, cx + self.last_move[1]*8, 33)
        c4 = get_robust_patch(self.p_mask, cy, cx, 33, normalize=False)
        return np.stack([c1, c2, c3, c4], axis=0).astype(np.float32)
    def step(self, m_idx, s_val):
        self.steps += 1
        d_end = np.linalg.norm(np.array(self.agent) - self.gt[-1])
        if s_val > args.stop_thresh:
            success = d_end < 6.0
            return self.get_obs(), (200.0 if success else -50.0), True, success
        dy, dx = ACTIONS_MOVE[m_idx]; self.last_move = (dy, dx)
        self.agent = (np.clip(self.agent[0]+dy*STEP_ALPHA,0,127), np.clip(self.agent[1]+dx*STEP_ALPHA,0,127))
        self.p_mask[int(self.agent[0]), int(self.agent[1])] = 1.0
        d_curve = np.sqrt(np.min(np.sum((self.gt - self.agent)**2, axis=1)))
        reward = np.exp(-(d_curve**2) / (2 * (self.cfg['w'][1]/2)**2)) - 0.1
        done = d_curve > (self.cfg['w'][1]*2 + 12) or self.steps > 350
        return self.get_obs(), reward, done, False

# ---------- MODEL ----------
class SignalActorCritic(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(4, 32, 3, padding=1), nn.GroupNorm(4, 32), nn.PReLU(),
            nn.Conv2d(32, 64, 3, padding=1, stride=2), nn.GroupNorm(8, 64), nn.PReLU(),
            nn.Conv2d(64, 128, 3, padding=1), nn.GroupNorm(16, 128), nn.PReLU(),
            nn.AdaptiveAvgPool2d((1,1))
        )
        self.lstm = nn.LSTM(128, 128, batch_first=True)
        self.move_head = nn.Linear(128, 8)
        self.stop_head = nn.Linear(128, 1)
        self.critic = nn.Linear(128, 1)
    def forward(self, x, h=None):
        b, c, hi, wi = x.shape
        f = self.cnn(x).view(b, 1, 128)
        o, h = self.lstm(f, h); o = o.squeeze(1)
        return self.move_head(o), torch.sigmoid(self.stop_head(o)), self.critic(o), h

# ---------- TRAINING LOOP ----------
def train():
    env = SignalEnv(); model = SignalActorCritic().to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    start_t = time.time()
    
    stages = [
        {'id': 1, 'w': (2, 4), 'n': 0.0, 'i': (0.8, 1.0)},
        {'id': 2, 'w': (1, 8), 'n': 0.2, 'i': (0.5, 1.0)},
        {'id': 3, 'w': (0.5, args.width_max), 'n': 0.6, 'i': (args.intensity_min, 0.9)},
        {'id': 4, 'w': (2, 5), 'n': 0.1, 'i': (0.6, 1.0), 's': True}
    ]

    for st in stages:
        logger.info(f"\n>>> Stage {st['id']} - Name: {args.run_name}")
        rew_his, succ_his = [], []
        for ep in range(1, args.eps_per_stage + 1):
            obs = env.reset(st); h = None; done = False; ep_r = 0
            while not done:
                obs_t = torch.tensor(obs[None,...], device=DEVICE)
                move_logits, s_prob, val, h = model(obs_t, h)
                m_dist = Categorical(logits=move_logits); m_idx = m_dist.sample()
                next_obs, rew, done, is_succ = env.step(m_idx.item(), s_prob.item())
                
                with torch.no_grad():
                    _, _, next_v, _ = model(torch.tensor(next_obs[None,...], device=DEVICE), h)
                
                # Actor-Critic Updates
                td_target = rew + (0.99 * next_v if not done else 0)
                advantage = td_target - val
                
                loss_move = -m_dist.log_prob(m_idx) * advantage.detach()
                dist_to_end = np.linalg.norm(np.array(env.agent) - env.gt[-1])
                target_stop = 1.0 if dist_to_end < 6.0 else 0.0
                loss_stop = F.binary_cross_entropy(s_prob, torch.tensor([[target_stop]], device=DEVICE))
                loss_v = F.mse_loss(val, td_target.detach())
                
                (loss_move + loss_stop + 0.5 * loss_v).backward()
                opt.step(); opt.zero_grad()
                h = (h[0].detach(), h[1].detach()); obs = next_obs; ep_r += rew

            rew_his.append(ep_r); succ_his.append(1 if is_succ else 0)
            if ep % 100 == 0:
                ar, asu = np.mean(rew_his[-100:]), np.mean(succ_his[-100:])
                elap = (time.time() - start_t) / 60
                logger.info(f"St {st['id']} | Ep {ep} | R: {ar:.1f} | S: {asu:.2f} | {elap:.1f} min")
                with open(csv_file, mode='a', newline='') as f:
                    csv.writer(f).writerow([st['id'], ep, round(ar,2), round(asu,2), round(elap,2)])
        
        torch.save(model.state_dict(), os.path.join(args.out_dir, f"weights_st{st['id']}.pth"))

if __name__ == "__main__":
    train()
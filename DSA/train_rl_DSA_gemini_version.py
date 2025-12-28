#!/usr/bin/env python3
import argparse
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from dataclasses import dataclass
from scipy.ndimage import gaussian_filter
from torch.utils.tensorboard import SummaryWriter
import datetime

# Import your specific generator
try:
    from Curve_Generator_Flexible_For_Ciruculum_learning import CurveMakerFlexible
except ImportError:
    print("Error: Generator file not found.")
    exit()

# ---------- ARGS & CONFIG ----------
parser = argparse.ArgumentParser()
parser.add_argument("--lr", type=float, default=1e-4, help="Learning Rate")
parser.add_argument("--gamma", type=float, default=0.99, help="Discount Factor")
parser.add_argument("--align_weight", type=float, default=2.0, help="Weight for Cosine Alignment Reward")
parser.add_argument("--smooth_weight", type=float, default=0.2, help="Weight for Smoothness Penalty")
parser.add_argument("--hidden_size", type=int, default=128, help="LSTM Hidden Size")
parser.add_argument("--seed", type=int, default=42, help="Random Seed")
parser.add_argument("--exp_name", type=str, default="default", help="Name for TensorBoard")
parser.add_argument("--gpu_id", type=int, default=0, help="GPU ID to use")
args = parser.parse_args()

DEVICE = f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu"
ACTIONS_MOVEMENT = [(-1, 0), (1, 0), (0,-1), (0, 1), (-1,-1), (-1,1), (1,-1), (1,1)]
ACTION_STOP_IDX = 8
STEP_ALPHA = 2.0
CROP = 33

# Set Seeds
torch.manual_seed(args.seed)
np.random.seed(args.seed)

# ... [Helper Functions: clamp, crop32, get_closest_point_info remain the same] ...
def clamp(v, lo, hi): return max(lo, min(v, hi))
def crop32(img: np.ndarray, cy: int, cx: int, size=CROP):
    h, w = img.shape
    corners = [img[0,0], img[0, w-1], img[h-1, 0], img[h-1, w-1]]
    bg_estimate = np.median(corners)
    pad_val = 1.0 if bg_estimate > 0.5 else 0.0
    r = size // 2
    y0, y1 = cy - r, cy + r + 1
    x0, x1 = cx - r, cx + r + 1
    out = np.full((size, size), pad_val, dtype=img.dtype)
    sy0, sy1 = clamp(y0, 0, h), clamp(y1, 0, h)
    sx0, sx1 = clamp(x0, 0, w), clamp(x1, 0, w)
    oy0, ox0 = sy0 - y0, sx0 - x0
    sh, sw = sy1 - sy0, sx1 - sx0
    if sh > 0 and sw > 0:
        out[oy0:oy0+sh, ox0:ox0+sw] = img[sy0:sy1, sx0:sx1]
    return out

def get_closest_point_info(pt, poly):
    dif = poly - np.array(pt, dtype=np.float32)
    d2 = np.sum(dif * dif, axis=1)
    min_idx = int(np.argmin(d2))
    return np.sqrt(d2[min_idx]), min_idx

@dataclass
class CurveEpisode:
    img: np.ndarray
    gt_poly: np.ndarray

# ---------- REFINED ENVIRONMENT ----------
class CurveEnvRefined:
    def __init__(self, h=128, w=128, max_steps=200):
        self.h, self.w = h, w
        self.max_steps = max_steps
        self.cm = CurveMakerFlexible(h=h, w=w, seed=args.seed)
        self.config = {'width': (2, 5), 'noise': 0.0, 'invert': 0.5, 'tissue': False, 'stopping_training': False, 'distractors': False}
        self.reset()

    def set_config(self, cfg):
        self.config.update(cfg)

    def reset(self):
        if self.config['distractors']:
            img, _, pts_all = self.cm.sample_with_distractors(width_range=self.config['width'], noise_prob=self.config['noise'], invert_prob=self.config['invert'])
            gt_poly = pts_all[0].astype(np.float32)
        else:
            img, _, pts_all = self.cm.sample_curve(width_range=self.config['width'], noise_prob=self.config['noise'], invert_prob=self.config['invert'])
            gt_poly = pts_all[0].astype(np.float32)

        if self.config['tissue']:
            noise = np.random.randn(self.h, self.w)
            tissue = gaussian_filter(noise, sigma=3.0)
            tissue = (tissue - tissue.min()) / (tissue.max() - tissue.min()) * 0.3
            is_white_bg = np.mean([img[0,0], img[0,-1]]) > 0.5
            if is_white_bg: img = np.clip(img - tissue, 0.0, 1.0)
            else: img = np.clip(img + tissue, 0.0, 1.0)

        self.ep = CurveEpisode(img=img, gt_poly=gt_poly)
        poly_len = len(gt_poly)
        
        if self.config['stopping_training']:
            start_idx = np.random.randint(int(poly_len * 0.8), poly_len - 5)
        else:
            start_idx = 5 if poly_len > 10 else 0

        self.agent = tuple(gt_poly[start_idx])
        if start_idx >= 2:
            self.history_pos = [tuple(gt_poly[start_idx-2]), tuple(gt_poly[start_idx-1]), self.agent]
        else:
            self.history_pos = [self.agent] * 3

        self.steps = 0
        self.path_mask = np.zeros_like(img)
        self.path_mask[int(self.agent[0]), int(self.agent[1])] = 1.0
        self.dist_to_line, self.closest_idx = get_closest_point_info(self.agent, gt_poly)
        self.prev_idx = self.closest_idx
        self.prev_action = -1
        return self.obs()

    def obs(self):
        curr, p1, p2 = self.history_pos[-1], self.history_pos[-2], self.history_pos[-3]
        ch0 = crop32(self.ep.img, int(curr[0]), int(curr[1]))
        ch1 = crop32(self.ep.img, int(p1[0]), int(p1[1]))
        ch2 = crop32(self.ep.img, int(p2[0]), int(p2[1]))
        ch3 = crop32(self.path_mask, int(curr[0]), int(curr[1]))
        actor_obs = np.stack([ch0, ch1, ch2, ch3], axis=0).astype(np.float32)
        dy = curr[0] - p1[0]; dx = curr[1] - p1[1]
        mag = np.sqrt(dy**2 + dx**2) + 1e-6
        vec_obs = np.array([dy/mag, dx/mag], dtype=np.float32)
        return {"img": actor_obs, "vec": vec_obs}

    def step(self, action):
        self.steps += 1
        dist_to_end = np.linalg.norm(np.array(self.agent) - self.ep.gt_poly[-1])

        if action == ACTION_STOP_IDX:
            if dist_to_end < 6.0: return self.obs(), 50.0, True, {"success": True}
            else: return self.obs(), -5.0, True, {"success": False}

        dy, dx = ACTIONS_MOVEMENT[action]
        ny = self.agent[0] + dy * STEP_ALPHA
        nx = self.agent[1] + dx * STEP_ALPHA
        
        if not (0 <= ny < self.h and 0 <= nx < self.w):
            return self.obs(), -5.0, True, {"success": False}

        self.agent = (ny, nx)
        self.history_pos.append(self.agent)
        self.path_mask[int(ny), int(nx)] = 1.0

        new_dist, new_idx = get_closest_point_info(self.agent, self.ep.gt_poly)
        r_dist = np.exp(-(new_dist**2) / (2 * 1.5**2)) 
        
        progress = new_idx - self.prev_idx
        r_prog = 1.5 if progress > 0 else -0.5

        # --- ALIGNMENT REWARD (TUNABLE) ---
        gt_len = len(self.ep.gt_poly)
        lookahead = min(new_idx + 4, gt_len - 1)
        gt_vec = self.ep.gt_poly[lookahead] - self.ep.gt_poly[new_idx]
        gt_norm = np.linalg.norm(gt_vec) + 1e-8
        action_vec = np.array([dy, dx])
        action_norm = np.linalg.norm(action_vec)
        cosine = np.dot(gt_vec, action_vec) / (gt_norm * action_norm)
        r_align = cosine * args.align_weight 

        # --- SMOOTHNESS PENALTY (TUNABLE) ---
        r_smooth = 0.0
        if self.prev_action != -1 and self.prev_action != action:
            r_smooth = -args.smooth_weight

        total_reward = r_dist + r_prog + r_align + r_smooth - 0.1

        self.prev_idx = max(self.prev_idx, new_idx)
        self.prev_action = action
        
        done = False
        if new_dist > 8.0: total_reward -= 5.0; done = True
        if self.steps >= self.max_steps: done = True
        if dist_to_end < 5.0 and progress > 5: total_reward -= 2.0; done = True

        return self.obs(), total_reward, done, {"success": False}

# ---------- NETWORK ----------
class ActorCriticSmooth(nn.Module):
    def __init__(self, n_actions=9, hidden_size=128):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(4, 32, 3, padding=1), nn.GroupNorm(4, 32), nn.PReLU(),
            nn.Conv2d(32, 64, 3, padding=2, dilation=2), nn.GroupNorm(8, 64), nn.PReLU(),
            nn.Conv2d(64, 64, 3, padding=3, dilation=3), nn.GroupNorm(8, 64), nn.PReLU(),
            nn.AdaptiveAvgPool2d((1,1))
        )
        self.lstm = nn.LSTM(input_size=66, hidden_size=hidden_size, batch_first=True)
        self.actor = nn.Linear(hidden_size, n_actions)
        self.critic = nn.Linear(hidden_size, 1)

    def forward(self, img, vec, hidden=None):
        B = img.shape[0]
        feats = self.cnn(img).view(B, -1)
        combined = torch.cat([feats, vec], dim=1).unsqueeze(1)
        lstm_out, new_hidden = self.lstm(combined, hidden)
        x = lstm_out[:, -1, :]
        return self.actor(x), self.critic(x), new_hidden

# ---------- TRAINING ----------
def train():
    log_dir = os.path.join("runs", f"{datetime.datetime.now().strftime('%m%d')}_{args.exp_name}")
    writer = SummaryWriter(log_dir)
    print(f"Logging to {log_dir}")

    env = CurveEnvRefined()
    model = ActorCriticSmooth(hidden_size=args.hidden_size).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    
    phases = [
        {'name': "Foundation", 'episodes': 3000, 'cfg': {'width':(3,6), 'noise':0.0, 'stopping_training': False, 'distractors': False}},
        {'name': "Stopping_Clinic", 'episodes': 3000, 'cfg': {'width':(3,6), 'noise':0.2, 'stopping_training': True, 'distractors': False}},
        {'name': "Robustness", 'episodes': 5000, 'cfg': {'width':(2,6), 'noise':0.8, 'stopping_training': True, 'tissue': True, 'distractors': False}},
        {'name': "Distractors", 'episodes': 5000, 'cfg': {'width':(2,5), 'noise':0.8, 'stopping_training': True, 'tissue': True, 'distractors': True}},
    ]

    total_ep = 0
    for phase in phases:
        print(f"=== PHASE: {phase['name']} ===")
        env.set_config(phase['cfg'])
        
        for ep in range(phase['episodes']):
            obs = env.reset()
            done = False
            lstm_h = None
            log_probs, values, rewards = [], [], []
            info_success = False
            
            while not done:
                img_t = torch.tensor(obs['img'][None], device=DEVICE)
                vec_t = torch.tensor(obs['vec'][None], device=DEVICE)
                logits, val, lstm_h = model(img_t, vec_t, lstm_h)
                dist = Categorical(logits=logits)
                action = dist.sample()
                next_obs, r, done, info = env.step(action.item())
                if info['success']: info_success = True
                
                log_probs.append(dist.log_prob(action))
                values.append(val)
                rewards.append(r)
                obs = next_obs

            # PPO-Lite Update
            returns = []
            R = 0
            for r in reversed(rewards):
                R = r + args.gamma * R
                returns.insert(0, R)
            returns = torch.tensor(returns, device=DEVICE)
            values = torch.cat(values).squeeze()
            log_probs = torch.cat(log_probs)
            advantage = returns - values.detach()
            actor_loss = -(log_probs * advantage).mean()
            critic_loss = F.mse_loss(values, returns)
            loss = actor_loss + 0.5 * critic_loss
            
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            opt.step()
            
            total_ep += 1
            # Logging
            if total_ep % 10 == 0:
                writer.add_scalar("Reward/Episode", sum(rewards), total_ep)
                writer.add_scalar("Loss/Total", loss.item(), total_ep)
                writer.add_scalar("Metric/Success", 1.0 if info_success else 0.0, total_ep)

    torch.save(model.state_dict(), f"{log_dir}/model_final.pth")
    writer.close()
    print("Training Complete")

if __name__ == "__main__":
    train()
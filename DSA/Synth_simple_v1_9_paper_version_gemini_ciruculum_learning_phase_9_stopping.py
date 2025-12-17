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
    print("Error: Generator not found")
    exit()

# Import Utils
from Synth_simple_v1_9_paper_version_gemini import (
    fixed_window_history, ACTIONS_8, DEVICE, clamp, crop32
)

# 9 ACTIONS: 0-7 = Move, 8 = STOP
ACTIONS_9 = ACTIONS_8 + [(0, 0)] 
STEP_ALPHA = 2.0
CROP = 33

@dataclass
class CurveEpisode:
    img: np.ndarray; mask: np.ndarray; gt_poly: np.ndarray

# --- MODEL WITH 9 OUTPUTS ---
def gn(c): return nn.GroupNorm(4, c)

class Actor9(nn.Module):
    def __init__(self, K=8):
        super().__init__()
        # CNN (Same as before)
        self.actor_cnn = nn.Sequential(
            nn.Conv2d(4, 32, 3, padding=1), gn(32), nn.PReLU(),
            nn.Conv2d(32, 32, 3, padding=2, dilation=2), gn(32), nn.PReLU(),
            nn.Conv2d(32, 64, 3, padding=3, dilation=3), gn(64), nn.PReLU(),
            nn.AdaptiveAvgPool2d((1,1))
        )
        # LSTM input size is now 9 (one-hot vector for 9 actions)
        self.actor_lstm = nn.LSTM(input_size=9, hidden_size=64, batch_first=True)
        
        # HEAD outputs 9 logits
        self.actor_head = nn.Sequential(nn.Linear(128, 64), nn.PReLU(), nn.Linear(64, 9))
        
        # CRITIC (Standard)
        self.critic_cnn = nn.Sequential(
            nn.Conv2d(5, 32, 3, padding=1), gn(32), nn.PReLU(),
            nn.Conv2d(32, 64, 3, padding=3, dilation=3), gn(64), nn.PReLU(),
            nn.AdaptiveAvgPool2d((1,1))
        )
        self.critic_lstm = nn.LSTM(input_size=9, hidden_size=64, batch_first=True)
        self.critic_head = nn.Sequential(nn.Linear(128, 64), nn.PReLU(), nn.Linear(64, 1))

    def forward(self, obs, gt_map, ahist, hc_actor=None, hc_critic=None):
        # Actor
        z = self.actor_cnn(obs).flatten(1)
        _, hc_actor = self.actor_lstm(ahist, hc_actor)
        h = hc_actor[0][-1]
        logits = self.actor_head(torch.cat([z, h], dim=1))
        
        # Critic
        z_c = self.critic_cnn(torch.cat([obs, gt_map], dim=1)).flatten(1)
        _, hc_critic = self.critic_lstm(ahist, hc_critic)
        h_c = hc_critic[0][-1]
        val = self.critic_head(torch.cat([z_c, h_c], dim=1))
        
        return logits, val.squeeze(-1), hc_actor, hc_critic

# --- ENV FOR STOPPING ---
class CurveEnvStopping:
    def __init__(self, h=128, w=128):
        self.h, self.w = h, w
        self.cm = CurveMakerFlexible(h=h, w=w)
        self.max_steps = 200
        
    def reset(self):
        # Generate Hard Curve (Phase 8 settings)
        img, mask, pts = self.cm.sample_curve(width_range=(2, 6), noise_prob=0.8, invert_prob=0.5, min_intensity=0.2)
        gt_poly = pts[0].astype(np.float32)
        
        # Random Reverse
        if np.random.rand() < 0.5: gt_poly = gt_poly[::-1]
        
        self.ep = CurveEpisode(img, mask, gt_poly)
        
        # Generate GT Map
        self.gt_map = np.zeros_like(img)
        for pt in gt_poly:
            self.gt_map[int(pt[0]), int(pt[1])] = 1.0

        # --- CRITICAL: END-GAME INITIALIZATION ---
        # 50% chance: Start normally (early in vessel)
        # 50% chance: Start NEAR THE END (15-20 steps away)
        
        total_len = len(gt_poly)
        if np.random.rand() < 0.5 and total_len > 30:
            # Start near end
            start_idx = np.random.randint(total_len - 25, total_len - 10)
        else:
            # Start near beginning
            start_idx = 5
            
        curr = gt_poly[start_idx]
        prev1 = gt_poly[start_idx-1]
        prev2 = gt_poly[start_idx-2]
        
        self.agent = (float(curr[0]), float(curr[1]))
        self.history_pos = [tuple(prev2), tuple(prev1), self.agent]
        self.path_mask = np.zeros_like(mask, dtype=np.float32)
        self.path_mask[int(self.agent[0]), int(self.agent[1])] = 1.0
        
        self.steps = 0
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

    def step(self, a_idx):
        self.steps += 1
        
        # Calculate Distance to End Point
        end_pt = self.ep.gt_poly[-1]
        dist_to_end = np.sqrt((self.agent[0]-end_pt[0])**2 + (self.agent[1]-end_pt[1])**2)
        
        # --- ACTION 8: STOP ---
        if a_idx == 8:
            if dist_to_end < 5.0:
                # JACKPOT: Stopped within 5px of end
                return self.obs(), 50.0, True, {"success": True}
            elif dist_to_end < 10.0:
                # Decent stop
                return self.obs(), 10.0, True, {"success": True}
            else:
                # Stopped too early (False Negative)
                return self.obs(), -5.0, True, {"success": False}

        # --- MOVEMENT (Actions 0-7) ---
        dy, dx = ACTIONS_9[a_idx]
        ny = self.agent[0] + dy * STEP_ALPHA
        nx = self.agent[1] + dx * STEP_ALPHA
        
        # Wall Hit
        if not (0 <= ny < self.h and 0 <= nx < self.w):
            return self.obs(), -5.0, True, {"success": False}

        self.agent = (ny, nx)
        self.history_pos.append(self.agent)
        self.path_mask[int(ny), int(nx)] = 1.0
        
        # --- THE CLIFF PENALTY ---
        # Did we walk PAST the end?
        # If we are farther from the end than we were last step, AND we are close to the end...
        # It means we overshoot.
        new_dist = np.sqrt((ny-end_pt[0])**2 + (nx-end_pt[1])**2)
        
        # Reward logic
        r = 0.0
        
        # Check if we are "Off the vessel" (using GT mask)
        # We check the pixel in the GT Mask. If it is 0, we walked into noise.
        iy, ix = int(ny), int(nx)
        on_vessel = self.ep.mask[iy, ix] > 0
        
        if not on_vessel:
            # We walked off the vessel.
            # If we were close to the end, this is a "Missed Stop" -> HUGE PENALTY
            if dist_to_end < 10.0:
                r = -20.0
                return self.obs(), r, True, {"success": False} # Kill episode immediately
            else:
                # Just wandering off track in the middle
                r = -1.0
        else:
            # On vessel movement
            r = 0.5

        # Max steps
        done = (self.steps >= self.max_steps)
        if done and dist_to_end < 5.0:
            # Ran out of time but was at the end? (Should have pressed stop)
            r -= 10.0
            
        return self.obs(), r, done, {"success": False}

def update_ppo(opt, model, buf):
    # Same PPO update as before, just ensure ahist tensor shape handles 9 actions
    obs_a = torch.tensor(np.concatenate([x['obs']['actor'] for x in buf]), dtype=torch.float32, device=DEVICE)
    obs_c = torch.tensor(np.concatenate([x['obs']['critic_gt'] for x in buf]), dtype=torch.float32, device=DEVICE)
    ahist = torch.tensor(np.concatenate([x['ahist'] for x in buf]), dtype=torch.float32, device=DEVICE)
    act   = torch.tensor(np.concatenate([x['act'] for x in buf]), dtype=torch.long, device=DEVICE)
    logp  = torch.tensor(np.concatenate([x['logp'] for x in buf]), dtype=torch.float32, device=DEVICE)
    adv   = torch.tensor(np.concatenate([x['adv'] for x in buf]), dtype=torch.float32, device=DEVICE)
    ret   = torch.tensor(np.concatenate([x['ret'] for x in buf]), dtype=torch.float32, device=DEVICE)

    if adv.numel()>1: adv = (adv - adv.mean()) / (adv.std() + 1e-8)

    for _ in range(4):
        logits, val, _, _ = model(obs_a, obs_c, ahist)
        # Clamp 9 logits
        logits = torch.clamp(logits, -20, 20)
        dist = Categorical(logits=logits)
        
        new_logp = dist.log_prob(act)
        ratio = torch.exp(new_logp - logp)
        surr1 = ratio * adv
        surr2 = torch.clamp(ratio, 0.8, 1.2) * adv
        
        loss = -torch.min(surr1, surr2).mean() + 0.5*F.mse_loss(val, ret) - 0.01*dist.entropy().mean()
        
        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        opt.step()

def train_stopping(args):
    print("--- STARTING PHASE 9: STOPPING MASTERY ---")
    
    # 1. Initialize New Model with 9 Actions
    model = Actor9().to(DEVICE)
    
    # 2. Transfer Weights (Surgery)
    # We load the 8-action model into the 9-action model.
    # The CNN weights match. The LSTM/Head weights will partially match or need reset.
    try:
        saved = torch.load(args.load_path, map_location=DEVICE)
        current = model.state_dict()
        
        # Filter matching keys
        to_load = {}
        for k, v in saved.items():
            if k in current:
                if v.shape == current[k].shape:
                    to_load[k] = v
                else:
                    print(f"Skipping {k} (Shape mismatch: {v.shape} vs {current[k].shape})")
                    # This is expected for 'actor_head' and 'actor_lstm'
        
        model.load_state_dict(to_load, strict=False)
        print("Transferred CNN weights. LSTM/Head initialized fresh.")
    except Exception as e:
        print(f"Error loading: {e}")
        return

    # 3. Train
    opt = torch.optim.Adam(model.parameters(), lr=1e-4) # Higher LR to learn new action
    env = CurveEnvStopping(h=128, w=128)
    
    batch = []
    ep_rewards = []
    
    for ep in range(1, args.episodes+1):
        obs_dict = env.reset()
        done = False
        
        # History now has 9 dims per step
        ahist = []
        # Pre-fill history with "Straight" moves (dummy) or zeros
        # Just zeros is fine, the LSTM will learn
        
        ep_traj = {"obs":{"actor":[], "critic_gt":[]}, "ahist":[], "act":[], "logp":[], "adv":[], "ret":[]}
        vals = []
        rews = []
        
        while not done:
            obs_a = torch.tensor(obs_dict['actor'][None], dtype=torch.float32, device=DEVICE)
            obs_c = torch.tensor(obs_dict['critic_gt'][None], dtype=torch.float32, device=DEVICE)
            A = fixed_window_history(ahist, 8, 9)[None, ...]
            A_t = torch.tensor(A, dtype=torch.float32, device=DEVICE)
            
            with torch.no_grad():
                logits, val, _, _ = model(obs_a, obs_c, A_t)
                dist = Categorical(logits=logits)
                action = dist.sample().item()
                logp = dist.log_prob(torch.tensor(action, device=DEVICE)).item()
            
            next_obs, r, done, info = env.step(action)
            
            # Store
            ep_traj["obs"]["actor"].append(obs_dict["actor"])
            ep_traj["obs"]["critic_gt"].append(obs_dict["critic_gt"])
            ep_traj["ahist"].append(A[0])
            ep_traj["act"].append(action)
            ep_traj["logp"].append(logp)
            vals.append(val.item())
            rews.append(r)
            
            # Update History
            onehot = np.zeros(9); onehot[action] = 1.0
            ahist.append(onehot)
            obs_dict = next_obs

        # GAE
        vals.append(0.0)
        rews = np.array(rews)
        vals = np.array(vals)
        delta = rews + 0.9 * vals[1:] - vals[:-1]
        adv = np.zeros_like(rews)
        acc = 0
        for t in reversed(range(len(rews))):
            acc = delta[t] + 0.9 * 0.95 * acc
            adv[t] = acc
        ret = adv + vals[:-1]
        
        ep_traj["adv"] = adv
        ep_traj["ret"] = ret
        
        batch.append(ep_traj)
        ep_rewards.append(sum(rews))
        
        if len(batch) >= 32:
            update_ppo(opt, model, batch)
            batch = []
            
        if ep % 100 == 0:
            # Check how often it stops successfully
            # "Success" in info means Good Stop
            print(f"Ep {ep} | Avg Rew: {np.mean(ep_rewards[-100:]):.2f}")
            
        if ep % 1000 == 0:
            torch.save(model.state_dict(), f"ckpt_Phase9_Stop_ep{ep}.pth")

    torch.save(model.state_dict(), "ppo_model_Phase9_FINAL.pth")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=50000)
    parser.add_argument("--load_path", type=str, default="ppo_model_Phase8_Realism.pth")
    args = parser.parse_args()
    train_stopping(args)
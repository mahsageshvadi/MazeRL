#!/usr/bin/env python3
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from dataclasses import dataclass

# --- IMPORTS ---
from Curve_Generator_Flexible_For_Ciruculum_learning import CurveMakerFlexible
from model_and_utils import RobustActorCritic, crop48, ACTIONS_8, DEVICE, CROP_SIZE, clamp

# --- CONFIGURATION ---
STEP_ALPHA = 2.0
EPSILON = 1e-6
BATCH_SIZE_EPISODES = 32 # Larger batch for stability with noisy data

# --- UTILS ---
def fixed_window_history(ahist_list, K, n_actions):
    """Creates a tensor of the last K actions."""
    out = np.zeros((K, n_actions), dtype=np.float32)
    if len(ahist_list) == 0: return out
    tail = ahist_list[-K:]
    out[-len(tail):] = np.stack(tail, axis=0)
    return out

def get_distance_to_poly(pt, poly):
    """Euclidean distance to closest point on GT curve."""
    dif = poly - np.array(pt, dtype=np.float32)
    d2 = np.sum(dif * dif, axis=1)
    return np.sqrt(np.min(d2))

def nearest_gt_index(pt, poly):
    """Index of the closest point on GT curve."""
    dif = poly - np.array(pt, dtype=np.float32)
    d2 = np.sum(dif * dif, axis=1)
    return int(np.argmin(d2))

def update_ppo(ppo_opt, model, buf_list, clip=0.2, epochs=4, minibatch=32):
    """Standard PPO Update Logic."""
    # Flatten buffer
    obs_a = torch.tensor(np.concatenate([x['obs']['actor'] for x in buf_list]), dtype=torch.float32, device=DEVICE)
    obs_c = torch.tensor(np.concatenate([x['obs']['critic_gt'] for x in buf_list]), dtype=torch.float32, device=DEVICE)
    ahist = torch.tensor(np.concatenate([x['ahist'] for x in buf_list]), dtype=torch.float32, device=DEVICE)
    act   = torch.tensor(np.concatenate([x['act'] for x in buf_list]), dtype=torch.long, device=DEVICE)
    logp  = torch.tensor(np.concatenate([x['logp'] for x in buf_list]), dtype=torch.float32, device=DEVICE)
    adv   = torch.tensor(np.concatenate([x['adv'] for x in buf_list]), dtype=torch.float32, device=DEVICE)
    ret   = torch.tensor(np.concatenate([x['ret'] for x in buf_list]), dtype=torch.float32, device=DEVICE)

    # Normalize Advantage
    if adv.numel() > 1: 
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
    
    N = obs_a.shape[0]
    idxs = np.arange(N)
    
    for _ in range(epochs):
        np.random.shuffle(idxs)
        for s in range(0, N, minibatch):
            mb = idxs[s:s+minibatch]
            if len(mb) == 0: continue

            logits, val, _ = model(obs_a[mb], obs_c[mb], ahist[mb])
            
            # Clamp for numerical stability
            logits = torch.clamp(logits, -20, 20)
            dist = Categorical(logits=logits)
            new_logp = dist.log_prob(act[mb])
            entropy = dist.entropy().mean()

            ratio = torch.exp(new_logp - logp[mb])
            surr1 = ratio * adv[mb]
            surr2 = torch.clamp(ratio, 1.0-clip, 1.0+clip) * adv[mb]
            
            p_loss = -torch.min(surr1, surr2).mean()
            v_loss = F.mse_loss(val, ret[mb])
            
            # Entropy coef can be lower now (0.005) since BC initialized a good policy
            loss = p_loss + 0.5 * v_loss - 0.005 * entropy
            
            ppo_opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            ppo_opt.step()

@dataclass
class CurveEpisode:
    img: np.ndarray
    mask: np.ndarray
    gt_poly: np.ndarray

# --- RL ENVIRONMENT ---
class CurveEnvRL:
    def __init__(self, h=128, w=128):
        self.h, self.w = h, w
        self.cm = CurveMakerFlexible(h=h, w=w)
        self.max_steps = 200
        
    def reset(self):
        # 1. HARD MODE CONFIGURATION
        # We start directly with the hardest data because we have BC pre-training.
        # Width: 2-10px (Scale Invariance)
        # Noise: 100% (Robustness)
        # Invert: 50% (Contrast Learning)
        # Intensity: Down to 0.15 (Low Signal)
        img, mask, pts = self.cm.sample_with_distractors(
            width_range=(2, 10), 
            noise_prob=1.0, 
            invert_prob=0.5, 
            min_intensity=0.15
        )
        gt_poly = pts[0].astype(np.float32)
        
        # 2. RANDOM REVERSAL (Rotational Invariance)
        # 50% chance to track backwards. Prevents bias towards "Up/Right".
        if np.random.rand() < 0.5: 
            gt_poly = gt_poly[::-1]
        
        # 3. GT Map for Critic
        self.gt_map = np.zeros_like(img)
        for pt in gt_poly:
            r, c = int(pt[0]), int(pt[1])
            if 0<=r<self.h and 0<=c<self.w: 
                self.gt_map[r,c] = 1.0

        self.ep = CurveEpisode(img, mask, gt_poly)
        
        # 4. ROBUST INITIALIZATION (Running Start)
        # Start at index 5 to generate history momentum
        start_idx = 5
        if len(gt_poly) < 10: start_idx = 0
        
        curr = gt_poly[start_idx]
        prev1 = gt_poly[max(0, start_idx-1)]
        prev2 = gt_poly[max(0, start_idx-2)]
        
        self.agent = (float(curr[0]), float(curr[1]))
        
        # Inject momentum into history so LSTM knows direction
        self.history_pos = [tuple(prev2), tuple(prev1), self.agent]
        self.path_points = [self.agent]
        self.path_mask = np.zeros_like(mask, dtype=np.float32)
        self.path_mask[int(self.agent[0]), int(self.agent[1])] = 1.0
        
        self.steps = 0
        self.prev_idx = start_idx
        self.prev_action = -1
        self.L_prev = get_distance_to_poly(self.agent, gt_poly)
        
        return self.obs()

    def obs(self):
        curr = self.history_pos[-1]
        p1 = self.history_pos[-2]
        p2 = self.history_pos[-3]
        
        # Uses crop48 (48x48) with Smart Padding logic
        ch0 = crop48(self.ep.img, int(curr[0]), int(curr[1]))
        ch1 = crop48(self.ep.img, int(p1[0]), int(p1[1]))
        ch2 = crop48(self.ep.img, int(p2[0]), int(p2[1]))
        ch3 = crop48(self.path_mask, int(curr[0]), int(curr[1]))
        
        actor_obs = np.stack([ch0, ch1, ch2, ch3], axis=0).astype(np.float32)
        gt_crop = crop48(self.gt_map, int(curr[0]), int(curr[1]))
        gt_obs = gt_crop[None, ...]
        return {"actor": actor_obs, "critic_gt": gt_obs}

    def step(self, a_idx):
        self.steps += 1
        dy, dx = ACTIONS_8[a_idx]
        
        # Movement
        ny = clamp(self.agent[0] + dy * STEP_ALPHA, 0, self.h-1)
        nx = clamp(self.agent[1] + dx * STEP_ALPHA, 0, self.w-1)
        self.agent = (ny, nx)
        
        self.history_pos.append(self.agent)
        self.path_points.append(self.agent)
        
        ir, ic = int(ny), int(nx)
        self.path_mask[ir, ic] = 1.0

        # Calculations
        L_t = get_distance_to_poly(self.agent, self.ep.gt_poly)
        dist_diff = abs(L_t - self.L_prev)
        best_idx = nearest_gt_index(self.agent, self.ep.gt_poly)
        progress_delta = best_idx - self.prev_idx
        
        # --- REWARD FUNCTION (PRECISION) ---
        
        # 1. Base Gradient (Log distance)
        if L_t < self.L_prev:
            r = np.log(EPSILON + dist_diff)
        else:
            r = -np.log(EPSILON + dist_diff)
        r = float(np.clip(r, -2.0, 2.0))

        # 2. Gaussian Precision Bonus
        # Sigma=1.0 forces tight centerline tracking
        sigma = 1.0 
        precision_score = np.exp(-(L_t**2) / (2 * sigma**2))
        
        if progress_delta > 0:
            # Huge bonus for moving forward while centered
            r += precision_score * 2.0 
        elif progress_delta <= 0:
            # Penalty for loitering
            r -= 0.1
            
        # 3. Penalties
        if self.prev_action != -1 and self.prev_action != a_idx:
            r -= 0.05 # Twitch penalty
        self.prev_action = a_idx
        r -= 0.05 # Step cost

        # Update State
        self.L_prev = L_t
        self.prev_idx = max(self.prev_idx, best_idx)
        
        # --- TERMINATION ---
        dist_to_end = np.sqrt((self.agent[0]-self.ep.gt_poly[-1][0])**2 + (self.agent[1]-self.ep.gt_poly[-1][1])**2)
        reached_end = dist_to_end < 5.0
        
        # Stricter off-track for precision (6.0px)
        off_track = L_t > 6.0 
        too_long = len(self.path_points) > len(self.ep.gt_poly) * 2.5
        
        done = reached_end or off_track or too_long or (self.steps >= self.max_steps)
        
        if reached_end: r += 50.0
        if off_track:   r -= 5.0

        return self.obs(), r, done, {"reached_end": reached_end}

# --- MAIN TRAINING LOOP ---
def train_rl(args):
    print("--- STARTING STEP 2: PPO FINE-TUNING ---")
    
    # 1. Initialize
    env = CurveEnvRL(h=128, w=128)
    K = 8
    nA = len(ACTIONS_8)
    model = RobustActorCritic(n_actions=nA, K=K).to(DEVICE)
    
    # 2. LOAD BC WEIGHTS (Critical Step)
    try:
        model.load_state_dict(torch.load(args.load_path, map_location=DEVICE))
        print(f"Loaded Pre-trained BC Weights from: {args.load_path}")
    except FileNotFoundError:
        print(f"CRITICAL ERROR: Could not find {args.load_path}.")
        print("Please run 'train_step1_bc.py' first to generate the teacher model.")
        return

    # 3. Optimizer
    # We use a Low Learning Rate (1e-5) because the model already knows "how to see".
    # We are just teaching it "persistence" and "precision".
    opt = torch.optim.Adam(model.parameters(), lr=1e-5)
    
    # 4. Training Loop
    batch_buffer = []
    ep_returns = []
    
    print(f"Training for {args.episodes} episodes...")
    
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
                logits, value, _ = model(obs_a, obs_c, A_t)
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

        # Calculate GAE
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

        # Update Model
        if len(batch_buffer) >= BATCH_SIZE_EPISODES:
            update_ppo(opt, model, batch_buffer)
            batch_buffer = []

        # Logging
        if ep % 100 == 0:
            avg_r = np.mean(ep_returns[-100:])
            # Threshold > 40 usually indicates reaching the end + picking up alignment bonuses
            success_cnt = sum(1 for r in ep_returns[-100:] if r > 40)
            print(f"[PPO Fine-Tune] Ep {ep} | Avg Rew: {avg_r:.2f} | Approx Success: {success_cnt/100:.2f}")

        # Checkpoints
        if ep % 1000 == 0:
            torch.save(model.state_dict(), f"ckpt_step2_ppo_ep{ep}.pth")
            print(f"Saved ckpt_step2_ppo_ep{ep}.pth")

    torch.save(model.state_dict(), "ppo_model_step2_final.pth")
    print("Training Complete. Saved to ppo_model_step2_final.pth")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=100000, help="Number of PPO episodes")
    parser.add_argument("--load_path", type=str, default="bc_pretrained_model.pth", help="Path to BC weights")
    args = parser.parse_args()
    
    train_rl(args)
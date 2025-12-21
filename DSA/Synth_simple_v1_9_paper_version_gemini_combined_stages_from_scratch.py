#!/usr/bin/env python3
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from dataclasses import dataclass
from scipy.ndimage import gaussian_filter

# --- IMPORT GENERATOR ---
try:
    from Curve_Generator_Flexible_For_Ciruculum_learning import CurveMakerFlexible
except ImportError:
    print("ERROR: Generator not found. Please ensure 'Curve_Generator_Flexible_For_Ciruculum_learning.py' is in the directory.")
    exit()

# ---------- GLOBALS ----------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# 8 Movement Actions + 1 Stop Action = 9 Total
ACTIONS_MOVEMENT = [(-1, 0), (1, 0), (0,-1), (0, 1), (-1,-1), (-1,1), (1,-1), (1,1)]
ACTION_STOP_IDX = 8
N_ACTIONS = 9

STEP_ALPHA = 2.0
CROP = 33
EPSILON = 1e-6

# ---------- HELPERS ----------
def clamp(v, lo, hi): return max(lo, min(v, hi))

def crop32(img: np.ndarray, cy: int, cx: int, size=CROP):
    h, w = img.shape
    # Smart Padding: Check corners to guess background (0.0 or 1.0)
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

# ---------- UNIFIED ENVIRONMENT ----------
class CurveEnvUnified:
    def __init__(self, h=128, w=128, max_steps=200):
        self.h, self.w = h, w
        self.max_steps = max_steps
        self.cm = CurveMakerFlexible(h=h, w=w, seed=None)
        
        # Default Stage 1 Configuration
        self.stage_config = {
            'stage_id': 1,
            'width': (2, 4),
            'noise': 0.0,
            'invert': 0.5,
            'tissue': False,
            'strict_stop': False, # If False, auto-completes when reaching end
            'mixed_start': False  # If False, always uses running start
        }
        self.reset()

    def set_stage(self, config):
        """Update the curriculum stage settings."""
        self.stage_config.update(config)
        print(f"\n[ENV] Config Updated for Stage {self.stage_config.get('stage_id')}:")
        print(f"      Width: {self.stage_config['width']}, Noise: {self.stage_config['noise']}")
        print(f"      Tissue: {self.stage_config['tissue']}, Strict Stop: {self.stage_config['strict_stop']}")

    def generate_tissue_noise(self):
        """Simulate X-ray tissue artifacts."""
        noise = np.random.randn(self.h, self.w)
        tissue = gaussian_filter(noise, sigma=np.random.uniform(2.0, 5.0))
        tissue = (tissue - tissue.min()) / (tissue.max() - tissue.min())
        return tissue * np.random.uniform(0.2, 0.4)

    def reset(self):
        # 1. Sample Curve based on Config
        img, mask, pts_all = self.cm.sample_curve(
            width_range=self.stage_config['width'], 
            noise_prob=self.stage_config['noise'],
            invert_prob=self.stage_config['invert'],
            min_intensity=0.2 if self.stage_config['stage_id'] > 1 else 0.6
        )
        gt_poly = pts_all[0].astype(np.float32)

        # 2. Apply Realism (Tissue) if enabled
        if self.stage_config['tissue']:
            tissue = self.generate_tissue_noise()
            # Determine background color to blend correctly
            is_white_bg = np.mean([img[0,0], img[0,-1]]) > 0.5
            if is_white_bg:
                img = np.clip(img - tissue, 0.0, 1.0)
            else:
                img = np.clip(img + tissue, 0.0, 1.0)

        self.gt_map = np.zeros_like(img)
        for pt in gt_poly:
            r, c = int(pt[0]), int(pt[1])
            if 0<=r<self.h and 0<=c<self.w:
                self.gt_map[r,c] = 1.0
        
        self.ep = CurveEpisode(img=img, mask=mask, gt_poly=gt_poly)

        # 3. Initialization Logic (Running vs Cold Start)
        use_cold_start = False
        if self.stage_config['mixed_start']:
            use_cold_start = (np.random.rand() < 0.5)

        if use_cold_start:
            # Cold Start: 0 velocity at index 0
            curr = gt_poly[0]
            self.history_pos = [tuple(curr)] * 3
            self.prev_idx = 0
            self.agent = (float(curr[0]), float(curr[1]))
        else:
            # Running Start: Start at index 5 with momentum
            start_idx = 5 if len(gt_poly) > 10 else 0
            curr = gt_poly[start_idx]
            p1 = gt_poly[max(0, start_idx-1)]
            p2 = gt_poly[max(0, start_idx-2)]
            self.history_pos = [tuple(p2), tuple(p1), tuple(curr)]
            self.prev_idx = start_idx
            self.agent = (float(curr[0]), float(curr[1]))

        self.steps = 0
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
        
        # Critic GT (Only used for training value function)
        gt_crop = crop32(self.gt_map, int(curr[0]), int(curr[1]))
        gt_obs = gt_crop[None, ...]
        
        return {"actor": actor_obs, "critic_gt": gt_obs}

    def step(self, a_idx: int):
        self.steps += 1
        dist_to_end = np.sqrt(
            (self.agent[0] - self.ep.gt_poly[-1][0])**2 + 
            (self.agent[1] - self.ep.gt_poly[-1][1])**2
        )

        # ----- ACTION: STOP -----
        if a_idx == ACTION_STOP_IDX:
            # Check if we are actually at the end (< 5px)
            if dist_to_end < 5.0:
                # HUGE Reward for correct stop
                return self.obs(), 50.0, True, {"reached_end": True, "stopped_correctly": True}
            else:
                # Penalty for laziness (stopping early)
                # We don't terminate; we tell it "No, keep going"
                return self.obs(), -2.0, False, {"reached_end": False, "stopped_correctly": False}

        # ----- ACTION: MOVE -----
        dy, dx = ACTIONS_MOVEMENT[a_idx]
        ny = clamp(self.agent[0] + dy * STEP_ALPHA, 0, self.h-1)
        nx = clamp(self.agent[1] + dx * STEP_ALPHA, 0, self.w-1)
        self.agent = (ny, nx)
        
        self.history_pos.append(self.agent)
        self.path_points.append(self.agent)
        self.path_mask[int(ny), int(nx)] = 1.0

        # Metrics
        L_t = get_distance_to_poly(self.agent, self.ep.gt_poly)
        dist_diff = abs(L_t - self.L_prev)
        best_idx = nearest_gt_index(self.agent, self.ep.gt_poly)
        progress_delta = best_idx - self.prev_idx
        
        # --- REWARD ENGINEERING ---
        # 1. Gaussian Precision Reward
        sigma = 1.5 if self.stage_config['stage_id'] == 1 else 1.0
        precision_score = np.exp(-(L_t**2) / (2 * sigma**2))
        
        if L_t < self.L_prev:
            r = np.log(EPSILON + dist_diff)
        else:
            r = -np.log(EPSILON + dist_diff)
        r = float(np.clip(r, -2.0, 2.0))

        if progress_delta > 0:
            r += precision_score * 2.0
        elif progress_delta <= 0:
            r -= 0.1 # Stagnation penalty

        # 2. Alignment Bonus (Phase 2+ Feature)
        if self.stage_config['stage_id'] >= 2:
            lookahead_idx = min(best_idx + 4, len(self.ep.gt_poly) - 1)
            gt_vec = self.ep.gt_poly[lookahead_idx] - self.ep.gt_poly[best_idx]
            act_vec = np.array([dy, dx])
            norm_gt = np.linalg.norm(gt_vec)
            norm_act = np.linalg.norm(act_vec)
            if norm_gt > 1e-6 and norm_act > 1e-6:
                cos_sim = np.dot(gt_vec, act_vec) / (norm_gt * norm_act)
                if cos_sim > 0: r += cos_sim * 0.5

        # 3. Smoothness
        if self.prev_action != -1 and self.prev_action != a_idx:
            r -= 0.05
        self.prev_action = a_idx
        r -= 0.05 # Step penalty

        self.L_prev = L_t
        self.prev_idx = max(self.prev_idx, best_idx)

        # --- TERMINATION CONDITIONS ---
        done = False
        reached_end = (dist_to_end < 5.0)
        
        # OVERSHOOT CHECK (Important for teaching STOP)
        # If we passed the end and are moving away -> Fail
        # Simple check: if we are far from poly AND far from end
        off_track_limit = 10.0 if self.stage_config['stage_id'] == 1 else 8.0
        off_track = L_t > off_track_limit
        
        if off_track:
            r -= 5.0
            done = True
        
        if self.steps >= self.max_steps:
            done = True

        # STAGE 1 LOGIC: Auto-complete (Training wheels)
        if self.stage_config['strict_stop'] is False:
            if reached_end:
                r += 20.0 # Good job, but pressing STOP (action 8) would have given +50
                done = True
        else:
            # STAGE 2/3 LOGIC: Strict Stop
            # Walking into the goal doesn't end episode. Must press button.
            # But we can give a small hint reward for hovering near goal
            if reached_end:
                r += 0.5 

        return self.obs(), r, done, {"reached_end": reached_end, "stopped_correctly": False}

# ---------- NETWORK ARCHITECTURE ----------
def gn(c): return nn.GroupNorm(4, c)

class AsymmetricActorCritic(nn.Module):
    def __init__(self, n_actions=9, K=8):
        super().__init__()
        # Actor CNN
        self.actor_cnn = nn.Sequential(
            nn.Conv2d(4, 32, 3, padding=1), gn(32), nn.PReLU(),
            nn.Conv2d(32, 32, 3, padding=2, dilation=2), gn(32), nn.PReLU(),
            nn.Conv2d(32, 64, 3, padding=3, dilation=3), gn(64), nn.PReLU(),
            nn.AdaptiveAvgPool2d((1,1))
        )
        # Actor LSTM (Stateful memory)
        self.actor_lstm = nn.LSTM(input_size=n_actions, hidden_size=64, batch_first=True)
        self.actor_head = nn.Sequential(nn.Linear(128, 128), nn.PReLU(), nn.Linear(128, n_actions))

        # Critic CNN (Privileged info)
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
        # Actor
        feat_a = self.actor_cnn(actor_obs).flatten(1)
        lstm_a, hc_actor = self.actor_lstm(ahist_onehot, hc_actor)
        # Concat Visual Features + LSTM Memory
        joint_a = torch.cat([feat_a, lstm_a[:, -1, :]], dim=1)
        logits = self.actor_head(joint_a)

        # Critic
        critic_input = torch.cat([actor_obs, critic_gt], dim=1)
        feat_c = self.critic_cnn(critic_input).flatten(1)
        lstm_c, hc_critic = self.critic_lstm(ahist_onehot, hc_critic)
        joint_c = torch.cat([feat_c, lstm_c[:, -1, :]], dim=1)
        value = self.critic_head(joint_c).squeeze(-1)
        
        return logits, value, hc_actor, hc_critic

# ---------- PPO UPDATE ----------
def update_ppo(ppo_opt, model, buf_list, clip=0.2, epochs=4, minibatch=32):
    obs_a = torch.tensor(np.concatenate([x['obs']['actor'] for x in buf_list]), dtype=torch.float32, device=DEVICE)
    obs_c = torch.tensor(np.concatenate([x['obs']['critic_gt'] for x in buf_list]), dtype=torch.float32, device=DEVICE)
    ahist = torch.tensor(np.concatenate([x['ahist'] for x in buf_list]), dtype=torch.float32, device=DEVICE)
    act   = torch.tensor(np.concatenate([x['act'] for x in buf_list]), dtype=torch.long, device=DEVICE)
    logp  = torch.tensor(np.concatenate([x['logp'] for x in buf_list]), dtype=torch.float32, device=DEVICE)
    adv   = torch.tensor(np.concatenate([x['adv'] for x in buf_list]), dtype=torch.float32, device=DEVICE)
    ret   = torch.tensor(np.concatenate([x['ret'] for x in buf_list]), dtype=torch.float32, device=DEVICE)

    if adv.numel() > 1: adv = (adv - adv.mean()) / (adv.std() + 1e-8)
    
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

# ---------- MAIN CURRICULUM MANAGER ----------
def run_unified_training():
    print("=== STARTING UNIFIED RL TRAINING (3 STAGES) ===")
    print(f"Device: {DEVICE} | Actions: {N_ACTIONS} (Inc. STOP)")

    # Define the 3 Master Stages
    stages = [
        # Stage 1: Bootstrapping (Learn to walk)
        # Auto-complete enabled. Clean images. Running start.
        {
            'name': 'Stage1_Bootstrap',
            'episodes': 16000,
            'lr': 1e-4,
            'config': {
                'stage_id': 1, 'width': (2, 4), 'noise': 0.0, 
                'tissue': False, 'strict_stop': False, 'mixed_start': False
            }
        },
        # Stage 2: Robustness (Learn to survive & stop)
        # Noise enabled. Mixed starts (Cold/Running). STRICT STOP enabled (must press button).
        {
            'name': 'Stage2_Robustness',
            'episodes': 20000,
            'lr': 5e-5,
            'config': {
                'stage_id': 2, 'width': (2, 8), 'noise': 0.5, 
                'tissue': False, 'strict_stop': True, 'mixed_start': True
            }
        },
        # Stage 3: Realism (Mastery)
        # Tissue enabled. Full difficulty.
    """    {
            'name': 'Stage3_Realism',
            'episodes': 30000,
            'lr': 1e-5,
            'config': {
                'stage_id': 3, 'width': (1, 10), 'noise': 0.8, 
                'tissue': True, 'strict_stop': True, 'mixed_start': True
            }
        } """ 
    ]  

    # Initialize
    env = CurveEnvUnified(h=128, w=128)
    model = AsymmetricActorCritic(n_actions=N_ACTIONS).to(DEVICE)
    K = 8

    # Loop through stages
    for stage in stages:
        print(f"\n=============================================")
        print(f"STARTING {stage['name']}")
        print(f"Episodes: {stage['episodes']} | LR: {stage['lr']}")
        print(f"=============================================")
        
        env.set_stage(stage['config'])
        opt = torch.optim.Adam(model.parameters(), lr=stage['lr'])
        
        batch_buffer = []
        ep_returns = []
        ep_successes = [] # Tracks if they actually stopped/finished
        
        for ep in range(1, stage['episodes'] + 1):
            obs_dict = env.reset()
            done = False
            
            # History Init
            ahist = []
            ep_traj = {
                "obs":{'actor':[], 'critic_gt':[]}, "ahist":[], 
                "act":[], "logp":[], "val":[], "rew":[]
            }
            
            # Episode Loop
            while not done:
                # Prepare Inputs
                obs_a = torch.tensor(obs_dict['actor'][None], dtype=torch.float32, device=DEVICE)
                obs_c = torch.tensor(obs_dict['critic_gt'][None], dtype=torch.float32, device=DEVICE)
                
                # Use fixed window for network input (matches architecture)
                A = fixed_window_history(ahist, K, N_ACTIONS)[None, ...]
                A_t = torch.tensor(A, dtype=torch.float32, device=DEVICE)

                # Inference
                with torch.no_grad():
                    logits, value, _, _ = model(obs_a, obs_c, A_t)
                    logits = torch.clamp(logits, -20, 20)
                    dist = Categorical(logits=logits)
                    action = dist.sample().item()
                    logp = dist.log_prob(torch.tensor(action, device=DEVICE)).item()
                    val = value.item()

                # Step
                next_obs, r, done, info = env.step(action)

                # Store
                ep_traj["obs"]['actor'].append(obs_dict['actor'])
                ep_traj["obs"]['critic_gt'].append(obs_dict['critic_gt'])
                ep_traj["ahist"].append(A[0])
                ep_traj["act"].append(action)
                ep_traj["logp"].append(logp)
                ep_traj["val"].append(val)
                ep_traj["rew"].append(r)
                
                # Update History
                a_onehot = np.zeros(N_ACTIONS); a_onehot[action] = 1.0
                ahist.append(a_onehot)
                obs_dict = next_obs

            # GAE Calculation
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
                
                # Check success depending on stage requirements
                if stage['config']['strict_stop']:
                    ep_successes.append(1 if info.get('stopped_correctly') else 0)
                else:
                    ep_successes.append(1 if info.get('reached_end') else 0)

            # Update Model
            if len(batch_buffer) >= 32:
                update_ppo(opt, model, batch_buffer)
                batch_buffer = []

            # Logging
            if ep % 100 == 0:
                avg_r = np.mean(ep_returns[-100:])
                succ_rate = np.mean(ep_successes[-100:]) if ep_successes else 0.0
                print(f"[{stage['name']}] Ep {ep} | Avg Rew: {avg_r:.2f} | Success: {succ_rate:.2f}")

            # Checkpoint
            if ep % 2000 == 0:
                torch.save(model.state_dict(), f"ckpt_{stage['name']}_ep{ep}.pth")

        # Save End of Stage Model
        torch.save(model.state_dict(), f"model_{stage['name']}_FINAL.pth")
        print(f"Finished {stage['name']}. Saved model.")

    print("\n=== TRAINING COMPLETE ===")

if __name__ == "__main__":
    run_unified_training()
#!/usr/bin/env python3
import argparse
import numpy as np
import torch
from torch.distributions import Categorical
from dataclasses import dataclass

# Import your existing modules
from Synth_simple_v1_9_paper_version_gemini import (
    AsymmetricActorCritic, fixed_window_history, ACTIONS_8, DEVICE, 
    clamp, crop32, get_distance_to_poly, nearest_gt_index, update_ppo
)

try:
    from Curve_Generator_Flexible_For_Ciruculum_learning import CurveMakerFlexible
except ImportError:
    print("Error: Generator not found.")
    exit()

# --- CONFIGURATION ---
BATCH_SIZE_EPISODES = 32
STEP_ALPHA = 2.0
EPSILON = 1e-6

@dataclass
class CurveEpisode:
    img: np.ndarray
    mask: np.ndarray
    gt_poly: np.ndarray

class CurveEnvDistractor:
    def __init__(self, h=128, w=128):
        self.h, self.w = h, w
        self.cm = CurveMakerFlexible(h=h, w=w, seed=None)
        self.max_steps = 200
        self.reset()

    def reset(self):
        # USE THE NEW DISTRACTOR GENERATOR
        img, mask, pts_all = self.cm.sample_with_distractors(
            width_range=(2, 8),
            noise_prob=1.0,   # Full Noise
            invert_prob=0.5   # Mixed Backgrounds
        )
        
        gt_poly = pts_all[0].astype(np.float32)
        
        # IMPORTANT: GT Map only has the TARGET vessel
        self.gt_map = np.zeros_like(img)
        for pt in gt_poly:
            r, c = int(pt[0]), int(pt[1])
            if 0<=r<self.h and 0<=c<self.w:
                self.gt_map[r,c] = 1.0

        self.ep = CurveEpisode(img=img, mask=mask, gt_poly=gt_poly)

        # Initialize Agent (Running Start)
        start_idx = 5 if len(gt_poly) > 10 else 0
        curr = gt_poly[start_idx]
        prev1 = gt_poly[max(0, start_idx-1)]
        prev2 = gt_poly[max(0, start_idx-2)]

        self.agent = (float(curr[0]), float(curr[1]))
        self.history_pos = [tuple(prev2), tuple(prev1), self.agent]
        
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
        # Critic sees the CLEAN GT (Target only), so it knows the distractor is fake
        gt_crop = crop32(self.gt_map, int(curr[0]), int(curr[1]))
        gt_obs = gt_crop[None, ...]
        return {"actor": actor_obs, "critic_gt": gt_obs}

    def step(self, a_idx):
        self.steps += 1
        dy, dx = ACTIONS_8[a_idx]
        ny = clamp(self.agent[0] + dy * STEP_ALPHA, 0, self.h-1)
        nx = clamp(self.agent[1] + dx * STEP_ALPHA, 0, self.w-1)
        self.agent = (ny, nx)
        self.history_pos.append(self.agent)
        self.path_points.append(self.agent)
        self.path_mask[int(ny), int(nx)] = 1.0

        # Distance to TARGET (Distractors are invisible to this metric)
        L_t = get_distance_to_poly(self.agent, self.ep.gt_poly)
        dist_diff = abs(L_t - self.L_prev)
        best_idx = nearest_gt_index(self.agent, self.ep.gt_poly)
        progress_delta = best_idx - self.prev_idx
        
        # --- REWARD LOGIC ---
        # If L_t is high (agent jumped to distractor), this reward plummets
        if L_t < self.L_prev:
            r = np.log(EPSILON + dist_diff)
        else:
            r = -np.log(EPSILON + dist_diff)
        r = float(np.clip(r, -2.0, 2.0))

        sigma = 2.0
        precision_score = np.exp(-(L_t**2) / (2 * sigma**2))
        
        if progress_delta > 0:
            r += precision_score * 1.5
        elif progress_delta <= 0:
            r -= 0.1
            
        if self.prev_action != -1 and self.prev_action != a_idx:
            r -= 0.05 
        self.prev_action = a_idx
        r -= 0.05

        self.L_prev = L_t
        self.prev_idx = max(self.prev_idx, best_idx)
        
        dist_to_end = np.sqrt((self.agent[0]-self.ep.gt_poly[-1][0])**2 + (self.agent[1]-self.ep.gt_poly[-1][1])**2)
        reached_end = dist_to_end < 5.0
        
        # STRICT off-track limit for Phase 5
        # If it jumps to a distractor (usually > 5px away), kill the episode
        off_track = L_t > 8.0 
        too_long = len(self.path_points) > len(self.ep.gt_poly) * 2.0
        
        done = reached_end or off_track or too_long or (self.steps >= self.max_steps)
        
        if reached_end: r += 50.0
        if off_track:   r -= 10.0 # Heavy penalty for jumping ship

        info = {"reached_end": reached_end}
        return self.obs(), r, done, info

def train_phase5(args):
    print("--- STARTING PHASE 5: DISTRACTOR TRAINING ---")
    print("Teaching agent to ignore overlapping/parallel vessels.")
    
    env = CurveEnvDistractor(h=128, w=128)
    K = 8; nA = len(ACTIONS_8)
    model = AsymmetricActorCritic(n_actions=nA, K=K).to(DEVICE)
    
    # Load Phase 4 Model
    try:
        model.load_state_dict(torch.load(args.load_path, map_location=DEVICE))
        print(f"Loaded: {args.load_path}")
    except:
        print("Error loading weights.")
        return

    opt = torch.optim.Adam(model.parameters(), lr=1e-5)
    
    batch_buffer = []
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

        # Batching Logic
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
            success_cnt = sum(1 for r in ep_returns[-100:] if r > 30)
            print(f"[Phase 5] Ep {ep} | Avg Rew: {avg_r:.2f} | Success: {success_cnt/100:.2f}")

        if ep % 500 == 0:
            torch.save(model.state_dict(), f"ckpt_Phase5_ep{ep}.pth")

    torch.save(model.state_dict(), "ppo_model_Phase5_final.pth")
    print("Phase 5 Complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=50000)
    parser.add_argument("--load_path", type=str, default="ppo_model_Phase4_final.pth")
    args = parser.parse_args()
    train_phase5(args)
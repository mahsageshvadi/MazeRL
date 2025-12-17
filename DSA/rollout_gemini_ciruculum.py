import torch
import numpy as np
import matplotlib.pyplot as plt
import random
from scipy.interpolate import splprep, splev

try:
    from Curve_Generator_Flexible_For_Ciruculum_learning import CurveMakerFlexible
except ImportError:
    print("Error: Generator file not found")
    exit()

# Import your Phase 7 setup specifically to ensure compatibility
from Synth_simple_v1_9_paper_version_gemini_ciruculum_learning_phase_7_fine_tune_precision import (
    AsymmetricActorCritic, fixed_window_history, ACTIONS_8, DEVICE, 
    crop32, CurveEpisode, clamp, STEP_ALPHA
)

class CurveEnvVisualizer:
    def __init__(self, h=128, w=128):
        self.h, self.w = h, w
        self.cm = CurveMakerFlexible(h=h, w=w, seed=None)
        self.max_steps = 400

    def reset(self, width_range, noise_prob, invert_prob, min_intensity):
        # 1. Generate Curve
        img, mask, pts_all = self.cm.sample_curve(
            width_range=width_range, 
            noise_prob=noise_prob, 
            invert_prob=invert_prob,
            min_intensity=min_intensity
        )
        gt_poly = pts_all[0].astype(np.float32)

        # 2. Setup Episode
        self.gt_map = np.zeros_like(img) # Not strictly needed for actor, but good practice
        self.ep = CurveEpisode(img=img, mask=mask, gt_poly=gt_poly)

        # 3. ROBUST START STRATEGY
        # Instead of placing agent exactly at [0] with no history,
        # we calculate the vector from [0] to [1] to simulate "looking forward"
        p0 = gt_poly[0]
        p1 = gt_poly[min(4, len(gt_poly)-1)] # Look a few pixels ahead
        
        self.agent = (float(p0[0]), float(p0[1]))
        
        # Calculate a "Ghost" history point behind the start
        # This helps the LSTM understand the initial orientation
        vec_y = p0[0] - p1[0]
        vec_x = p0[1] - p1[1]
        p_minus_1 = (p0[0] + vec_y, p0[1] + vec_x) # Simulate a point "behind" us

        # Initialize history with this orientation
        self.history_pos = [p_minus_1, self.agent, self.agent] 
        
        self.steps = 0
        self.path_mask = np.zeros_like(mask, dtype=np.float32)
        self.path_points = [self.agent]
        self.path_mask[int(self.agent[0]), int(self.agent[1])] = 1.0
        
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
        # Dummy critic input
        gt_dummy = np.zeros((1, 33, 33), dtype=np.float32)
        return {"actor": actor_obs, "critic_gt": gt_dummy}

    def step(self, a_idx):
        self.steps += 1
        dy, dx = ACTIONS_8[a_idx]
        ny = clamp(self.agent[0] + dy * STEP_ALPHA, 0, self.h-1)
        nx = clamp(self.agent[1] + dx * STEP_ALPHA, 0, self.w-1)
        self.agent = (ny, nx)
        self.history_pos.append(self.agent)
        self.path_points.append(self.agent)
        self.path_mask[int(ny), int(nx)] = 1.0
        
        dist_to_end = np.sqrt((self.agent[0]-self.ep.gt_poly[-1][0])**2 + (self.agent[1]-self.ep.gt_poly[-1][1])**2)
        reached_end = dist_to_end < 5.0
        done = reached_end or (self.steps >= self.max_steps)
        
        return self.obs(), 0, done, {"reached_end": reached_end}

def smooth_path(path_points):
    if len(path_points) < 4: return path_points
    y = [p[0] for p in path_points]; x = [p[1] for p in path_points]
    try:
        tck, u = splprep([y, x], s=10.0) 
        new_points = splev(np.linspace(0, 1, len(path_points) * 3), tck)
        return list(zip(new_points[0], new_points[1]))
    except: return path_points 

def view_rollout(model_path):
    print(f"--- Loading {model_path} ---")
    K = 8; nA = len(ACTIONS_8)
    model = AsymmetricActorCritic(n_actions=nA, K=K).to(DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()
    
    # NOTE: Matched parameters to Phase 7 Training
    # We increased width range to (2, 6) and reduced noise to 0.8 to match training
    scenarios = [
        {"name": "Standard (Clean)",       "inv": 0.0, "noise": 0.0},
        {"name": "Hard (Noise+Contrast)",  "inv": 0.0, "noise": 0.8},
        {"name": "Inverted (Standard)",    "inv": 1.0, "noise": 0.0},
        {"name": "Inverted (Hard)",        "inv": 1.0, "noise": 0.8},
    ]
    
    plt.figure(figsize=(15, 10))
    
    for i, scen in enumerate(scenarios):
        env = CurveEnvVisualizer(h=128, w=128)
        
        # --- FIX: Match Training Parameters ---
        # width=(2,6), noise=0.8, intensity=0.2
        obs_dict = env.reset(
            width_range=(2, 6), 
            noise_prob=scen['noise'], 
            invert_prob=scen['inv'], 
            min_intensity=0.2
        )
        
        ahist = []; done = False
        while not done:
            obs_a = torch.tensor(obs_dict['actor'][None], dtype=torch.float32, device=DEVICE)
            gt_dummy = torch.zeros((1, 1, 33, 33), device=DEVICE)
            A = fixed_window_history(ahist, K, nA)[None, ...]
            A_t = torch.tensor(A, dtype=torch.float32, device=DEVICE)

            with torch.no_grad():
                logits, _, _, _ = model(obs_a, gt_dummy, A_t)
                # Greedy action for testing
                action = torch.argmax(logits, dim=1).item()

            obs_dict, _, done, info = env.step(action)
            a_onehot = np.zeros(nA); a_onehot[action] = 1.0
            ahist.append(a_onehot)

        # Plot
        plt.subplot(2, 2, i+1)
        plt.imshow(env.ep.img, cmap='gray', vmin=0, vmax=1)
        
        smoothed = smooth_path(env.path_points)
        sy = [p[0] for p in smoothed]; sx = [p[1] for p in smoothed]
        
        plt.plot(sx, sy, 'cyan', linewidth=2, alpha=0.8, label='AI')
        plt.plot(env.ep.gt_poly[:,1], env.ep.gt_poly[:,0], 'r--', alpha=0.4, label='GT')
        
        status = "SUCCESS" if info['reached_end'] else "FAIL"
        plt.title(f"{scen['name']}\n{status}")
        plt.legend()

    plt.tight_layout()
    plt.savefig("final_test_result.png")
    print("Saved final_test_result.png")
    plt.show()

if __name__ == "__main__":
    view_rollout("DSA/ppo_model_Phase8_Realism.pth")
import torch
import numpy as np
import matplotlib.pyplot as plt
import random
from scipy.interpolate import splprep, splev

# Import your classes
try:
    from Curve_Generator_Flexible_For_Ciruculum_learning import CurveMakerFlexible
except ImportError:
    print("Error: Generator file not found")
    exit()

from Synth_simple_v1_9_paper_version_gemini import AsymmetricActorCritic, fixed_window_history, ACTIONS_8, DEVICE, crop32, CurveEpisode, clamp, STEP_ALPHA

# --- CUSTOM ENV FOR VISUALIZATION ---
class CurveEnvVisualizer:
    def __init__(self, h=128, w=128):
        self.h, self.w = h, w
        self.cm = CurveMakerFlexible(h=h, w=w, seed=None)
        self.max_steps = 400

    def reset(self, width_range, noise_prob, invert_prob, min_intensity):
        # Force params
        img, mask, pts_all = self.cm.sample_curve(
            width_range=width_range, 
            noise_prob=noise_prob, 
            invert_prob=invert_prob,
            min_intensity=min_intensity
        )
        gt_poly = pts_all[0].astype(np.float32)

        self.gt_map = np.zeros_like(img)
        # ... (GT map setup) ...

        p0 = gt_poly[0].astype(int)
        self.ep = CurveEpisode(img=img, mask=mask, gt_poly=gt_poly, start=(p0[0], p0[1])) # Fixed missing init_dir

        self.agent = (float(p0[0]), float(p0[1]))
        self.history_pos = [self.agent] * 3 
        self.steps = 0
        self.path_mask = np.zeros_like(mask, dtype=np.float32)
        self.path_points = [self.agent]
        self.path_mask[int(self.agent[0]), int(self.agent[1])] = 1.0
        
        return self.obs()

    def obs(self):
        curr = self.history_pos[-1]
        p1 = self.history_pos[-2]
        p2 = self.history_pos[-3]
        # Use the SMART PADDING crop
        ch0 = crop32(self.ep.img, int(curr[0]), int(curr[1]))
        ch1 = crop32(self.ep.img, int(p1[0]), int(p1[1]))
        ch2 = crop32(self.ep.img, int(p2[0]), int(p2[1]))
        ch3 = crop32(self.path_mask, int(curr[0]), int(curr[1]))
        
        actor_obs = np.stack([ch0, ch1, ch2, ch3], axis=0).astype(np.float32)
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
    
    # Test specifically the hard cases
    scenarios = [
        {"name": "Noisy & Low Contrast",  "inv": 0.0, "noise": 1.0},
        {"name": "Inverted & Noisy",      "inv": 1.0, "noise": 1.0},
        {"name": "Noisy & Low Contrast",  "inv": 0.0, "noise": 1.0},
        {"name": "Inverted & Noisy",      "inv": 1.0, "noise": 1.0},
    ]
    
    plt.figure(figsize=(15, 10))
    
    for i, scen in enumerate(scenarios):
        env = CurveEnvVisualizer(h=128, w=128)
        # Use Phase 3 Difficulty: Width 2-10, Low Intensity 0.15
        obs_dict = env.reset(width_range=(1,3), noise_prob=scen['noise'], invert_prob=scen['inv'], min_intensity=0.15)
        
        ahist = []; done = False
        while not done:
            obs_a = torch.tensor(obs_dict['actor'][None], dtype=torch.float32, device=DEVICE)
            gt_dummy = torch.zeros((1, 1, 33, 33), device=DEVICE)
            A = fixed_window_history(ahist, K, nA)[None, ...]
            A_t = torch.tensor(A, dtype=torch.float32, device=DEVICE)

            with torch.no_grad():
                logits, _, _, _ = model(obs_a, gt_dummy, A_t)
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
    plt.savefig("final_phase3_test.png")
    print("Saved final_phase3_test.png")
    plt.show()

if __name__ == "__main__":
    # Point to your LATEST checkpoint
    view_rollout("ppo_model_Phase7_final.pth")
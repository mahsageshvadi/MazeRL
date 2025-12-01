import torch
import numpy as np
import matplotlib.pyplot as plt
import random
from scipy.interpolate import splprep, splev

# --- IMPORTS ---
try:
    from Curve_Generator_Flexible_For_Ciruculum_learning import CurveMakerFlexible
except ImportError:
    print("Error: Generator file not found.")
    exit()

from model_and_utils import RobustActorCritic, crop48, ACTIONS_8, DEVICE, clamp, CROP_SIZE

# --- UTILS ---
def fixed_window_history(ahist_list, K, n_actions):
    out = np.zeros((K, n_actions), dtype=np.float32)
    if len(ahist_list) == 0: return out
    tail = ahist_list[-K:]
    out[-len(tail):] = np.stack(tail, axis=0)
    return out

def smooth_path(path_points):
    if len(path_points) < 4: return path_points
    y = [p[0] for p in path_points]; x = [p[1] for p in path_points]
    try:
        tck, u = splprep([y, x], s=5.0) 
        new_points = splev(np.linspace(0, 1, len(path_points) * 3), tck)
        return list(zip(new_points[0], new_points[1]))
    except: return path_points 

# --- VISUALIZATION ENV ---
class CurveEnvVisualizer:
    def __init__(self, h=128, w=128):
        self.h, self.w = h, w
        self.cm = CurveMakerFlexible(h=h, w=w, seed=None)
        self.max_steps = 300

    def reset(self):
        # TEST CONFIGURATION (Hard Mode)
        # We want to see if BC learned to handle noise and inversion
        width = (2, 8)
        noise = 1.0 
        invert = 0.5
        min_int = 0.15

        img, mask, pts_all = self.cm.sample_with_distractors(
            width_range=width, 
            noise_prob=noise, 
            invert_prob=invert,
            min_intensity=min_int
        )
        gt_poly = pts_all[0].astype(np.float32)
        
        # 50% Random Reverse to test Rotational Invariance
        if random.random() < 0.5:
            gt_poly = gt_poly[::-1]

        self.img = img
        self.gt_poly = gt_poly

        # ROBUST START (Index 5)
        start_idx = 5 if len(gt_poly) > 10 else 0
        curr = gt_poly[start_idx]
        prev1 = gt_poly[max(0, start_idx-1)]
        prev2 = gt_poly[max(0, start_idx-2)]

        self.agent = (float(curr[0]), float(curr[1]))
        
        # Pre-fill history so BC agent has momentum
        self.history_pos = [tuple(prev2), tuple(prev1), self.agent]
        self.path_points = [self.agent]
        
        # Path mask
        self.path_mask = np.zeros_like(img, dtype=np.float32)
        self.path_mask[int(self.agent[0]), int(self.agent[1])] = 1.0
        
        self.steps = 0
        return self.obs()

    def obs(self):
        curr = self.history_pos[-1]
        p1 = self.history_pos[-2]
        p2 = self.history_pos[-3]
        
        # Use crop48
        ch0 = crop48(self.img, int(curr[0]), int(curr[1]))
        ch1 = crop48(self.img, int(p1[0]), int(p1[1]))
        ch2 = crop48(self.img, int(p2[0]), int(p2[1]))
        ch3 = crop48(self.path_mask, int(curr[0]), int(curr[1]))
        
        actor_obs = np.stack([ch0, ch1, ch2, ch3], axis=0).astype(np.float32)
        gt_dummy = np.zeros((1, CROP_SIZE, CROP_SIZE), dtype=np.float32)
        return {"actor": actor_obs, "critic_gt": gt_dummy}

    def step(self, a_idx):
        self.steps += 1
        dy, dx = ACTIONS_8[a_idx]
        STEP_ALPHA = 2.0
        
        ny = clamp(self.agent[0] + dy * STEP_ALPHA, 0, self.h-1)
        nx = clamp(self.agent[1] + dx * STEP_ALPHA, 0, self.w-1)
        self.agent = (ny, nx)
        self.history_pos.append(self.agent)
        self.path_points.append(self.agent)
        self.path_mask[int(ny), int(nx)] = 1.0
        
        # Check success
        dist_to_end = np.sqrt((self.agent[0]-self.gt_poly[-1][0])**2 + (self.agent[1]-self.gt_poly[-1][1])**2)
        reached_end = dist_to_end < 5.0
        done = reached_end or (self.steps >= self.max_steps)
        
        return self.obs(), 0, done, {"reached_end": reached_end}

def view_bc(model_path="bc_pretrained_model.pth", num_episodes=5):
    print(f"--- Loading {model_path} ---")
    K = 8
    nA = len(ACTIONS_8)
    model = RobustActorCritic(n_actions=nA, K=K).to(DEVICE)
    
    try:
        model.load_state_dict(torch.load(model_path, map_location=DEVICE))
        model.eval()
        print("Weights loaded successfully.")
    except FileNotFoundError:
        print("Model file not found!")
        return

    for i in range(num_episodes):
        env = CurveEnvVisualizer(h=128, w=128)
        obs_dict = env.reset()
        
        ahist = []
        done = False
        
        while not done:
            obs_a = torch.tensor(obs_dict['actor'][None], dtype=torch.float32, device=DEVICE)
            gt_dummy = torch.tensor(obs_dict['critic_gt'][None], dtype=torch.float32, device=DEVICE)
            
            A = fixed_window_history(ahist, K, nA)[None, ...]
            A_t = torch.tensor(A, dtype=torch.float32, device=DEVICE)

            with torch.no_grad():
                logits, _, _ = model(obs_a, gt_dummy, A_t)
                # ARGMAX is crucial for BC evaluation (we want the most confident action)
                action = torch.argmax(logits, dim=1).item()

            obs_dict, _, done, info = env.step(action)
            
            a_onehot = np.zeros(nA); a_onehot[action] = 1.0
            ahist.append(a_onehot)

        # Plot
        raw = env.path_points
        smooth = smooth_path(raw)
        sy = [p[0] for p in smooth]; sx = [p[1] for p in smooth]
        
        plt.figure(figsize=(8, 8))
        plt.imshow(env.img, cmap='gray', vmin=0, vmax=1)
        
        # GT
        gty = env.gt_poly[:, 0]; gtx = env.gt_poly[:, 1]
        plt.plot(gtx, gty, 'r--', alpha=0.5, label='Ground Truth')
        
        # Agent
        plt.plot(sx, sy, 'cyan', linewidth=2, label='BC Agent')
        plt.plot(raw[0][1], raw[0][0], 'go', label='Start')
        
        status = "SUCCESS" if info['reached_end'] else "FAIL"
        plt.title(f"BC Test Ep {i+1}: {status}")
        plt.legend()
        
        save_name = f"bc_vis_ep{i+1}.png"
        plt.savefig(save_name)
        print(f"Saved {save_name}")
        plt.close()

if __name__ == "__main__":
    view_bc()
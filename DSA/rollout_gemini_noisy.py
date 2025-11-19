import torch
import numpy as np
import matplotlib.pyplot as plt
import random
from scipy.interpolate import splprep, splev

# 1. Import the Base Classes from your main script
# (Ensure this filename matches your actual file)
from Synth_simple_v1_9_paper_version_gemini import CurveEnv, AsymmetricActorCritic, fixed_window_history, ACTIONS_8, DEVICE

# 2. Import the Noisy Generator
try:
    from Curve_Generator_noisy import CurveMakerDSA
except ImportError:
    print("ERROR: Could not find 'Curve_Generator_DSA.py'")
    exit()

# 3. Define the Wrapper to use Noisy Images
# This effectively "patches" the environment to use the new generator
class CurveEnvDSA(CurveEnv):
    def __init__(self, h=128, w=128, branches=False):
        super().__init__(h, w, branches)
        # SWAP THE GENERATOR HERE
        self.cm = CurveMakerDSA(h=h, w=w, thickness=1.5, seed=None)

# --- Smoothing Function ---
def smooth_path(path_points):
    if len(path_points) < 4: return path_points
    y = [p[0] for p in path_points]
    x = [p[1] for p in path_points]
    try:
        tck, u = splprep([y, x], s=5.0) 
        new_points = splev(np.linspace(0, 1, len(path_points) * 2), tck) # Double the points for smoothness
        return list(zip(new_points[0], new_points[1]))
    except:
        return path_points 

def view_rollout(model_path, num_episodes=5):
    print(f"--- Loading Model from {model_path} ---")
    
    K = 8
    nA = len(ACTIONS_8)
    model = AsymmetricActorCritic(n_actions=nA, K=K).to(DEVICE)
    
    try:
        # Map location ensures it loads on CPU if CUDA is not available
        model.load_state_dict(torch.load(model_path, map_location=DEVICE))
        print("Weights loaded successfully.")
    except FileNotFoundError:
        print(f"ERROR: Could not find {model_path}")
        return

    model.eval()
    
    for i in range(num_episodes):
        print(f"Visualizing Episode {i+1}...")
        
        # USE THE DSA ENV HERE
        env = CurveEnvDSA(h=128, w=128, branches=False)
        obs_dict = env.reset()
        
        ahist = []
        done = False
        
        while not done:
            obs_a = torch.tensor(obs_dict['actor'][None], dtype=torch.float32, device=DEVICE)
            
            # Dummy GT for Critic
            gt_shape = obs_dict['critic_gt'][None].shape
            dummy_gt = torch.zeros(gt_shape, dtype=torch.float32, device=DEVICE)
            
            A = fixed_window_history(ahist, K, nA)[None, ...]
            A_t = torch.tensor(A, dtype=torch.float32, device=DEVICE)

            with torch.no_grad():
                logits, _, _, _ = model(obs_a, dummy_gt, A_t)
                action = torch.argmax(logits, dim=1).item()

            obs_dict, r, done, info = env.step(action)
            
            a_onehot = np.zeros(nA); a_onehot[action] = 1.0
            ahist.append(a_onehot)

        # --- PLOTTING ---
        path_y = [p[0] for p in env.path_points]
        path_x = [p[1] for p in env.path_points]
        gt_y = [p[0] for p in env.ep.gt_poly]
        gt_x = [p[1] for p in env.ep.gt_poly]
        
        # Generate Smoothed Path
        smoothed = smooth_path(env.path_points)
        s_y = [p[0] for p in smoothed]
        s_x = [p[1] for p in smoothed]

        plt.figure(figsize=(10, 10))
        
        # 1. Show the Noisy DSA Image
        plt.imshow(env.ep.img, cmap='gray', vmin=0, vmax=1)
        
        # 2. Plot GT (Red Dashed)
        plt.plot(gt_x, gt_y, 'r--', linewidth=1, alpha=0.6, label='Ground Truth')
        
        # 3. Plot Raw Agent Path (Blue Dots)
        plt.plot(path_x, path_y, 'b.', markersize=2, alpha=0.4, label='Raw Steps')
        
        # 4. Plot Smoothed Path (Cyan Line) - This is what you'd show a doctor
        plt.plot(s_x, s_y, 'cyan', linewidth=2.5, alpha=0.8, label='AI Prediction')
        
        # Start/End
        plt.plot(path_x[0], path_y[0], 'go', markersize=8, label='Start')
        plt.plot(path_x[-1], path_y[-1], 'rx', markersize=8, label='End')
        
        status = "SUCCESS" if info['reached_end'] else "FAILED"
        plt.title(f"DSA Test Ep {i+1}: {status} (Steps: {env.steps})")
        plt.legend()
        
        filename = f"dsa_vis_ep_{i+1}.png"
        plt.savefig(filename)
        print(f"Saved {filename}")
        plt.close()

if __name__ == "__main__":
    # CHANGE THIS FILENAME to match your latest saved checkpoint!
    # Example: "ppo_dsa_adapt_ep5000.pth"
    MODEL_FILENAME = "ppo_dsa_adapt_ep2000.pth" 
    
    view_rollout(MODEL_FILENAME, num_episodes=10)
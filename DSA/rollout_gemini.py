import torch
import numpy as np
import matplotlib.pyplot as plt
import random

from scipy.interpolate import splprep, splev

def smooth_path(path_points):
    """Smooths a jagged list of (y,x) tuples using B-Spline"""
    if len(path_points) < 4: return path_points
    
    # Separate Y and X
    y = [p[0] for p in path_points]
    x = [p[1] for p in path_points]
    
    # Fit Spline
    try:
        tck, u = splprep([y, x], s=5.0) # s is smoothness factor
        new_points = splev(np.linspace(0, 1, len(path_points)), tck)
        return list(zip(new_points[0], new_points[1]))
    except:
        return path_points # Fallback if spline fails


# --- IMPORT YOUR CLASSES ---
# Assuming your training script is named 'Synth_simple_v3_stable.py'
# If it is named something else, change this line!
from Synth_simple_v1_9_paper_version_gemini import CurveEnv, AsymmetricActorCritic, fixed_window_history, ACTIONS_8, DEVICE, set_seeds

def view_rollout(model_path="ppo_curve_agent.pth", num_episodes=5):
    print(f"--- Loading Model from {model_path} ---")
    
    # 1. Re-initialize the model architecture
    # (Must match the training architecture EXACTLY)
    K = 8
    nA = len(ACTIONS_8)
    model = AsymmetricActorCritic(n_actions=nA, K=K).to(DEVICE)
    
    try:
        model.load_state_dict(torch.load(model_path, map_location=DEVICE))
        print("Weights loaded successfully.")
    except FileNotFoundError:
        print("ERROR: Model file not found. Did you save it using torch.save()?")
        return

    model.eval()
    
    # 2. Run a few loops
    for i in range(num_episodes):
        print(f"Visualizing Episode {i+1}...")
        
        # Random seed for each episode to get different curves
        seed = random.randint(0, 10000)
        env = CurveEnv(h=128, w=128, branches=False) # Set seed inside if needed, or just rely on random
        obs_dict = env.reset()
        
        ahist = []
        done = False
        
        while not done:
                    # Prepare Actor Tensor
                    obs_a = torch.tensor(obs_dict['actor'][None], dtype=torch.float32, device=DEVICE)
                    
                    # --- FIX STARTS HERE ---
                    # Create a zero tensor with the same shape as the critic input
                    # We get the shape from the numpy array, but create the tensor directly in PyTorch
                    gt_shape = obs_dict['critic_gt'][None].shape
                    dummy_gt = torch.zeros(gt_shape, dtype=torch.float32, device=DEVICE)
                    # --- FIX ENDS HERE ---
                    
                    A = fixed_window_history(ahist, K, nA)[None, ...]
                    A_t = torch.tensor(A, dtype=torch.float32, device=DEVICE)

                    with torch.no_grad():
                        # Get Logits
                        logits, _, _, _ = model(obs_a, dummy_gt, A_t)
                        
                        # DETERMINISTIC ACTION (Argmax)
                        action = torch.argmax(logits, dim=1).item()

                    obs_dict, r, done, info = env.step(action)
                    
                    a_onehot = np.zeros(nA); a_onehot[action] = 1.0
                    ahist.append(a_onehot)

        # 3. Plotting
        path_y = [p[0] for p in env.path_points]
        path_x = [p[1] for p in env.path_points]
        gt_y = [p[0] for p in env.ep.gt_poly]
        gt_x = [p[1] for p in env.ep.gt_poly]
        
       # smoothed_path = smooth_path(env.path_points)

        plt.figure(figsize=(8, 8))
        plt.imshow(env.ep.img, cmap='gray')
        plt.plot(gt_x, gt_y, 'r--', linewidth=1.5, alpha=0.7, label='Ground Truth')
        plt.plot(path_x, path_y, 'cyan', linewidth=1.5, marker='.', markersize=3, label='Agent')
        plt.plot(path_x[0], path_y[0], 'go', label='Start')
        plt.plot(path_x[-1], path_y[-1], 'rx', label='End')

        #plt.plot([p[1] for p in smoothed_path], [p[0] for p in smoothed_path], 'cyan', linewidth=2)

        
        status = "SUCCESS" if info['reached_end'] else "FAILED"
        plt.title(f"Ep {i+1}: {status} (Steps: {env.steps})")
        plt.legend()
        
        filename = f"vis_ep_{i+1}.png"
        plt.savefig(filename)
        print(f"Saved plot to {filename}")
        plt.close()

if __name__ == "__main__":
    # Ensure you point this to your .pth file
    view_rollout("ppo_inversion_fix_ep5000.pth", num_episodes=20)
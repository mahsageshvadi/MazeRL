import numpy as np
import torch
from tqdm import tqdm
from Curve_Generator_Flexible_For_Ciruculum_learning import CurveMakerFlexible
from model_and_utils import crop48, get_action_from_vector, ACTIONS_8

# --- CONFIG ---
TOTAL_SAMPLES = 3000  # Size of dataset
SAMPLES_PER_IMAGE = 50  # Efficiency hack
SAVE_FILE = "bc_dataset.npz"

def generate():
    print(f"Generating {TOTAL_SAMPLES} samples...")
    
    # Storage arrays
    # Obs: (N, 4, 48, 48) - 4 channels: Curr, Prev1, Prev2, Mask
    all_obs = np.zeros((TOTAL_SAMPLES, 4, 48, 48), dtype=np.uint8)
    # History: (N, 8, 8)
    all_hist = np.zeros((TOTAL_SAMPLES, 8, 8), dtype=np.float32)
    # Label: (N,)
    all_labels = np.zeros((TOTAL_SAMPLES,), dtype=np.int64)
    
    cm = CurveMakerFlexible(h=128, w=128)
    count = 0
    
    pbar = tqdm(total=TOTAL_SAMPLES)
    
    while count < TOTAL_SAMPLES:
        # 1. Generate Image
        width = (2, 8)
        noise = 1.0 
        invert = 0.5 
        min_int = 0.15
        
        if np.random.rand() < 0.5:
            img, mask, pts_list = cm.sample_with_distractors(width, noise, invert, min_int)
        else:
            img, mask, pts_list = cm.sample_curve(width, noise, invert, min_int)
            
        gt_poly = pts_list[0]
        if np.random.rand() < 0.5: gt_poly = gt_poly[::-1]
        
        if len(gt_poly) < 50: continue
        
        # Convert image to uint8 (0-255) to save massive RAM/Disk space
        img_u8 = (img * 255).astype(np.uint8)
        mask_u8 = (mask * 255).astype(np.uint8)
        
        # 2. Extract Samples
        for _ in range(SAMPLES_PER_IMAGE):
            if count >= TOTAL_SAMPLES: break
            
            idx = np.random.randint(5, len(gt_poly) - 6)
            curr_pt = gt_poly[idx]
            
            # Lookahead
            future_pt = gt_poly[idx + 5]
            dy = future_pt[0] - curr_pt[0]
            dx = future_pt[1] - curr_pt[1]
            
            if (dy**2 + dx**2) < 2.0: continue
            
            target_action = get_action_from_vector(dy, dx)
            
            # History
            prev1 = gt_poly[idx-2]
            prev2 = gt_poly[idx-4]
            
            # Crops (Using custom crop48 that handles uint8)
            ch0 = crop48(img_u8, int(curr_pt[0]), int(curr_pt[1]))
            ch1 = crop48(img_u8, int(prev1[0]), int(prev1[1]))
            ch2 = crop48(img_u8, int(prev2[0]), int(prev2[1]))
            ch3 = crop48(mask_u8, int(curr_pt[0]), int(curr_pt[1]))
            
            # Stack
            obs = np.stack([ch0, ch1, ch2, ch3], axis=0)
            
            # Fake History
            prev_vec = np.zeros((8, 8), dtype=np.float32)
            p_act = get_action_from_vector(curr_pt[0]-prev2[0], curr_pt[1]-prev2[1])
            prev_vec[:, p_act] = 1.0
            
            # Store
            all_obs[count] = obs
            all_hist[count] = prev_vec
            all_labels[count] = target_action
            
            count += 1
            pbar.update(1)
            
    pbar.close()
    
    print("Saving to disk (this might take a moment)...")
    np.savez_compressed(SAVE_FILE, obs=all_obs, hist=all_hist, label=all_labels)
    print(f"Done! Saved {SAVE_FILE}")

if __name__ == "__main__":
    generate()
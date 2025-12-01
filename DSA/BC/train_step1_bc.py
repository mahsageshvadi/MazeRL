# --- CRITICAL: PREVENT OPENCV DEADLOCKS ---
import cv2
cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)
import os
os.environ["OMP_NUM_THREADS"] = "1"

import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
from torch.utils.data import IterableDataset, DataLoader
import time

# Imports
from Curve_Generator_Flexible_For_Ciruculum_learning import CurveMakerFlexible
from model_and_utils import RobustActorCritic, crop48, get_action_from_vector, ACTIONS_8, DEVICE, CROP_SIZE

# --- CONFIG ---
BATCH_SIZE = 64
LR = 3e-4
TOTAL_STEPS = 10000 
NUM_WORKERS = 4  # Set to 0 if debugging, 4 for speed

class FastInfiniteDataset(IterableDataset):
    def __init__(self, h=128, w=128):
        self.h = h
        self.w = w

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        seed = None
        if worker_info is not None:
            seed = worker_info.id * int(time.time())
        
        cm = CurveMakerFlexible(h=self.h, w=self.w, seed=seed)
        
        while True:
            # 1. Generate Image (Expensive Step)
            width = (2, 8)
            noise = 1.0 
            invert = 0.5 
            min_int = 0.15
            
            if np.random.rand() < 0.5:
                img, mask, pts_list = cm.sample_with_distractors(width, noise, invert, min_int)
            else:
                img, mask, pts_list = cm.sample_curve(width, noise, invert, min_int)
                
            gt_poly = pts_list[0]

            # Rotational Invariance (50% Flip)
            if np.random.rand() < 0.5:
                gt_poly = gt_poly[::-1]

            if len(gt_poly) < 50: continue
            
            # 2. Extract MANY samples (Cheap Step)
            # Re-using the image 50 times makes the pipeline 10x-50x faster
            for _ in range(50):
                idx = np.random.randint(5, len(gt_poly) - 6)
                curr_pt = gt_poly[idx]
                
                # LOOKAHEAD (Fixes direction noise)
                future_pt = gt_poly[idx + 5]
                
                dy = future_pt[0] - curr_pt[0]
                dx = future_pt[1] - curr_pt[1]
                
                if (dy**2 + dx**2) < 2.0: continue
                
                target_action = get_action_from_vector(dy, dx)
                
                # History (Momentum)
                prev1 = gt_poly[idx-2]
                prev2 = gt_poly[idx-4]
                
                # Crops
                ch0 = crop48(img, int(curr_pt[0]), int(curr_pt[1]))
                ch1 = crop48(img, int(prev1[0]), int(prev1[1]))
                ch2 = crop48(img, int(prev2[0]), int(prev2[1]))
                ch3 = crop48(mask.astype(np.float32), int(curr_pt[0]), int(curr_pt[1]))
                
                obs = np.stack([ch0, ch1, ch2, ch3], axis=0).astype(np.float32)
                
                # Fake History Vector
                prev_vec = np.zeros((8, 8), dtype=np.float32)
                p_act = get_action_from_vector(curr_pt[0]-prev2[0], curr_pt[1]-prev2[1])
                prev_vec[:, p_act] = 1.0 
                
                yield (torch.tensor(obs), torch.tensor(prev_vec), torch.tensor(target_action, dtype=torch.long))

def train_bc_fast():
    print(f"--- STARTING FAST BC TRAINING ({DEVICE}) ---")
    print(f"Using {NUM_WORKERS} CPU workers.")
    
    model = RobustActorCritic(n_actions=8, K=8).to(DEVICE)
    opt = optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.CrossEntropyLoss()
    
    ds = FastInfiniteDataset()
    dl = DataLoader(ds, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)
    
    iterator = iter(dl)
    losses = []
    
    model.train()
    
    try:
        for i in range(1, TOTAL_STEPS + 1):
            obs, ahist, target = next(iterator)
            
            obs = obs.to(DEVICE)
            ahist = ahist.to(DEVICE)
            target = target.to(DEVICE)
            
            dummy_gt = torch.zeros((obs.shape[0], 1, CROP_SIZE, CROP_SIZE)).to(DEVICE)
            
            logits, _, _ = model(obs, dummy_gt, ahist)
            loss = loss_fn(logits, target)
            
            opt.zero_grad()
            loss.backward()
            opt.step()
            
            losses.append(loss.item())
            
            if i % 100 == 0:
                avg_loss = np.mean(losses[-100:])
                preds = torch.argmax(logits, dim=1)
                acc = (preds == target).float().mean().item()
                print(f"Batch {i}/{TOTAL_STEPS} | Loss: {avg_loss:.4f} | Acc: {acc:.2f}")
                
            if i % 2000 == 0:
                torch.save(model.state_dict(), f"bc_checkpoint_{i}.pth")
                
    except KeyboardInterrupt:
        print("\nTraining stopped.")

    torch.save(model.state_dict(), "bc_pretrained_model.pth")
    print("BC Training Complete.")

if __name__ == "__main__":
    train_bc_fast()
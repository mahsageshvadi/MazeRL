import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
from torch.utils.data import IterableDataset, DataLoader
import math
import time

# Imports
from Curve_Generator_Flexible_For_Ciruculum_learning import CurveMakerFlexible
from model_and_utils import RobustActorCritic, crop48, get_action_from_vector, ACTIONS_8, DEVICE, CROP_SIZE

# --- CONFIG ---
BATCH_SIZE = 64
LR = 3e-4
TOTAL_STEPS = 10000 # How many batches to train on (Total samples = 10000 * 64 = 640,000)
NUM_WORKERS = 0  # Number of CPU cores to use for generation (Adjust based on your PC)

class FastInfiniteDataset(IterableDataset):
    def __init__(self, h=128, w=128):
        self.h = h
        self.w = w
        # We don't init the generator here because of multiprocessing pickling issues.
        # We init it inside __iter__

    def __iter__(self):
        # Each worker gets its own random seed to prevent duplicate data
        worker_info = torch.utils.data.get_worker_info()
        seed = None
        if worker_info is not None:
            seed = worker_info.id * int(time.time())
        
        cm = CurveMakerFlexible(h=self.h, w=self.w, seed=seed)
        
        while True:
            # 1. Generate Random Config (Hard Mode)
            width = (2, 8)
            noise = 1.0 
            invert = 0.5 
            min_int = 0.15
            
            # 50% chance of Distractors (Traps) vs Single Curve
            if np.random.rand() < 0.5:
                img, mask, pts_list = cm.sample_with_distractors(width, noise, invert, min_int)
            else:
                img, mask, pts_list = cm.sample_curve(width, noise, invert, min_int)
                
            gt_poly = pts_list[0]

            # 2. Rotational Invariance (50% Flip)
            # This fixes the "Always goes Right" bug
            if np.random.rand() < 0.5:
                gt_poly = gt_poly[::-1]

            # 3. Extract Samples from this image
            # We take multiple samples per image to be efficient
            if len(gt_poly) < 20: continue
            
            # Take 5 samples from this one image
            for _ in range(5):
                # Pick random point
                idx = np.random.randint(5, len(gt_poly) - 6)
                curr_pt = gt_poly[idx]
                
                # LOOKAHEAD (Fixes direction noise)
                future_pt = gt_poly[idx + 5]
                
                dy = future_pt[0] - curr_pt[0]
                dx = future_pt[1] - curr_pt[1]
                
                # Skip stationary points
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
    print(f"Using {NUM_WORKERS} CPU workers to generate data in parallel.")
    
    model = RobustActorCritic(n_actions=8, K=8).to(DEVICE)
    opt = optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.CrossEntropyLoss()
    
    # Dataset & Loader
    ds = FastInfiniteDataset()
    dl = DataLoader(ds, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)
    
    iterator = iter(dl)
    losses = []
    
    model.train()
    
    # Training Loop
    try:
        for i in range(1, TOTAL_STEPS + 1):
            obs, ahist, target = next(iterator)
            
            # Move to GPU
            obs = obs.to(DEVICE)
            ahist = ahist.to(DEVICE)
            target = target.to(DEVICE)
            
            dummy_gt = torch.zeros((obs.shape[0], 1, CROP_SIZE, CROP_SIZE)).to(DEVICE)
            
            # Forward
            logits, _, _ = model(obs, dummy_gt, ahist)
            loss = loss_fn(logits, target)
            
            # Backward
            opt.zero_grad()
            loss.backward()
            opt.step()
            
            losses.append(loss.item())
            
            if i % 100 == 0:
                avg_loss = np.mean(losses[-100:])
                # Check accuracy roughly
                preds = torch.argmax(logits, dim=1)
                acc = (preds == target).float().mean().item()
                print(f"Batch {i}/{TOTAL_STEPS} | Loss: {avg_loss:.4f} | Batch Acc: {acc:.2f}")
                
            if i % 2000 == 0:
                torch.save(model.state_dict(), f"bc_fast_checkpoint_{i}.pth")
                
    except KeyboardInterrupt:
        print("\nTraining stopped by user.")

    torch.save(model.state_dict(), "bc_pretrained_model.pth")
    print("BC Training Complete. Saved to bc_pretrained_model.pth")

if __name__ == "__main__":
    train_bc_fast()
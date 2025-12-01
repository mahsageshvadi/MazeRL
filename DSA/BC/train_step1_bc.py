import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
from torch.utils.data import IterableDataset, DataLoader
from tqdm import tqdm

from Curve_Generator_Flexible_For_Ciruculum_learning import CurveMakerFlexible
from model_and_utils import RobustActorCritic, crop48, get_action_from_vector, ACTIONS_8, DEVICE, CROP_SIZE

class InfiniteCurveDataset(IterableDataset):
    def __init__(self, h=128, w=128):
        self.cm = CurveMakerFlexible(h=h, w=w)
        self.K = 8 # History length
        self.nA = len(ACTIONS_8)

    def __iter__(self):
        while True:
            # 1. Generate a RANDOM challenging curve
            # We mix phases here to make it robust from day 1
            width = (2, 8)
            noise = 1.0 if np.random.rand() > 0.3 else 0.0
            invert = 0.5
            intensity = 0.2 if np.random.rand() > 0.5 else 0.6
            
            # Use Distractor generation 30% of time
            if np.random.rand() < 0.3:
                img, mask, pts_list = self.cm.sample_with_distractors(width, noise, invert, intensity)
            else:
                img, mask, pts_list = self.cm.sample_curve(width, noise, invert, intensity)
            
            gt_poly = pts_list[0]
            
            # 2. Randomly Traverse the Curve (Forward or Backward)
            if np.random.rand() < 0.5:
                gt_poly = gt_poly[::-1]
            
            # 3. Generate Samples along the curve
            # We skip the very ends to ensure history exists
            if len(gt_poly) < 20: continue
            
            # Take random slices along the path
            for _ in range(10): # Extract 10 samples per image
                idx = np.random.randint(5, len(gt_poly) - 5)
                
                curr_pt = gt_poly[idx]
                next_pt = gt_poly[idx + 1] # Ideally where we want to go
                
                # Calculate Ideal Action
                dy = next_pt[0] - curr_pt[0]
                dx = next_pt[1] - curr_pt[1]
                target_action = get_action_from_vector(dy, dx)
                
                # Construct History (Visual Momentum)
                # We look back at previous GT points
                prev_1 = gt_poly[idx-1]
                prev_2 = gt_poly[idx-2]
                
                # Create Observation Stack [Curr, P1, P2, Mask]
                # Note: Mask is just a dummy here or local path mask
                # For BC, we can just pass the raw image 3 times + empty mask
                # effectively relying on the LSTM for history
                
                ch0 = crop48(img, int(curr_pt[0]), int(curr_pt[1]))
                ch1 = crop48(img, int(prev_1[0]), int(prev_1[1]))
                ch2 = crop48(img, int(prev_2[0]), int(prev_2[1]))
                ch3 = crop48(mask.astype(np.float32), int(curr_pt[0]), int(curr_pt[1]))
                
                obs = np.stack([ch0, ch1, ch2, ch3], axis=0).astype(np.float32)
                
                # Construct Action History (Vector)
                # Simulate that we took the correct actions previously
                # (Ideally we'd calculate exact vectors, but one-hot of correct prev actions is good approx)
                prev_act_vec = np.zeros((self.K, self.nA), dtype=np.float32)
                # Fill last few steps with "Correct" previous moves
                # This teaches Momentum
                p_dy = curr_pt[0] - prev_1[0]
                p_dx = curr_pt[1] - prev_1[1]
                p_act = get_action_from_vector(p_dy, p_dx)
                prev_act_vec[-1, p_act] = 1.0
                prev_act_vec[-2, p_act] = 1.0
                
                yield torch.tensor(obs), torch.tensor(prev_act_vec), torch.tensor(target_action)

def train_bc():
    print("--- STARTING STEP 1: BEHAVIORAL CLONING ---")
    
    model = RobustActorCritic(n_actions=8, K=8).to(DEVICE)
    opt = optim.Adam(model.parameters(), lr=3e-4)
    loss_fn = nn.CrossEntropyLoss()
    
    ds = InfiniteCurveDataset()
    dl = DataLoader(ds, batch_size=64)
    
    # Train for 5000 Batches (approx 320,000 samples)
    iterator = iter(dl)
    losses = []
    
    for i in tqdm(range(5000)):
        obs, ahist, target = next(iterator)
        obs, ahist, target = obs.to(DEVICE), ahist.to(DEVICE), target.to(DEVICE)
        
        # Dummy GT for Critic (not used in BC loss, but needed for forward)
        dummy_gt = torch.zeros((64, 1, CROP_SIZE, CROP_SIZE)).to(DEVICE)
        
        logits, _, _ = model(obs, dummy_gt, ahist)
        
        loss = loss_fn(logits, target)
        
        opt.zero_grad()
        loss.backward()
        opt.step()
        
        losses.append(loss.item())
        
        if i % 100 == 0:
            print(f"Batch {i} | Loss: {np.mean(losses[-100:]):.4f}")
            
    torch.save(model.state_dict(), "bc_pretrained_model.pth")
    print("BC Training Complete. Saved to bc_pretrained_model.pth")

if __name__ == "__main__":
    train_bc()
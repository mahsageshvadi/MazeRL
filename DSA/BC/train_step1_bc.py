import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from tqdm import tqdm

from Curve_Generator_Flexible_For_Ciruculum_learning import CurveMakerFlexible
from model_and_utils import RobustActorCritic, crop48, get_action_from_vector, ACTIONS_8, DEVICE, CROP_SIZE

# --- CONFIG ---
SAMPLES_PER_EPOCH = 50000 
BATCH_SIZE = 64
EPOCHS = 10

class BalancedCurveDataset(Dataset):
    def __init__(self, count=10000):
        self.cm = CurveMakerFlexible(h=128, w=128)
        self.data = []
        self.generate_data(count)
        
    def generate_data(self, count):
        print("Generating Balanced Dataset...")
        
        # We generate samples until we hit the count
        # We try to keep directional balance
        
        pbar = tqdm(total=count)
        while len(self.data) < count:
            # 1. Random Parameters
            width = (2, 8)
            noise = 1.0 if np.random.rand() > 0.3 else 0.0
            invert = 0.5
            min_int = 0.15
            
            img, mask, pts_list = self.cm.sample_with_distractors(
                width_range=width, noise_prob=noise, invert_prob=invert, min_intensity=min_int
            )
            gt_poly = pts_list[0]
            
            # 50% Reverse (Crucial for eliminating Right-Bias)
            if np.random.rand() < 0.5:
                gt_poly = gt_poly[::-1]
            
            # Extract samples along the line
            # Step size 3 to avoid highly correlated samples
            for i in range(5, len(gt_poly) - 10, 3):
                curr_pt = gt_poly[i]
                
                # --- FIX: LOOKAHEAD ---
                # Look 5 steps ahead to determine direction clearly
                future_pt = gt_poly[i + 5]
                
                dy = future_pt[0] - curr_pt[0]
                dx = future_pt[1] - curr_pt[1]
                
                # Filter out stationary points (blobs)
                if np.sqrt(dy**2 + dx**2) < 2.0: continue
                
                target_action = get_action_from_vector(dy, dx)
                
                # History (Momentum)
                prev1 = gt_poly[i-2]
                prev2 = gt_poly[i-4]
                
                # Crops
                ch0 = crop48(img, int(curr_pt[0]), int(curr_pt[1]))
                ch1 = crop48(img, int(prev1[0]), int(prev1[1]))
                ch2 = crop48(img, int(prev2[0]), int(prev2[1]))
                ch3 = crop48(mask.astype(np.float32), int(curr_pt[0]), int(curr_pt[1]))
                
                obs = np.stack([ch0, ch1, ch2, ch3], axis=0).astype(np.float32)
                
                # Fake History Vector
                # We assume the agent was moving correctly previously
                prev_vec = np.zeros((8, 8), dtype=np.float32)
                
                # Calculate previous actual move
                p_dy = curr_pt[0] - prev2[0]
                p_dx = curr_pt[1] - prev2[1]
                p_act = get_action_from_vector(p_dy, p_dx)
                
                # Fill history with that momentum
                prev_vec[:, p_act] = 1.0 
                
                self.data.append({
                    'obs': obs,
                    'ahist': prev_vec,
                    'label': target_action
                })
                
                if len(self.data) >= count: break
            pbar.update(len(self.data) - pbar.n)
        pbar.close()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        return (torch.tensor(item['obs']), 
                torch.tensor(item['ahist']), 
                torch.tensor(item['label'], dtype=torch.long))

def train_bc_robust():
    print("--- STARTING ROBUST BC TRAINING ---")
    
    # 1. Generate Static Dataset (To allow weighting)
    ds = BalancedCurveDataset(count=SAMPLES_PER_EPOCH)
    
    # 2. Calculate Class Weights
    # We count how many samples exist for "Up", "Down", "Left", "Right"
    targets = [d['label'] for d in ds.data]
    class_counts = np.bincount(targets, minlength=8)
    print(f"Class Distribution: {class_counts}")
    
    # Calculate weight for each sample
    # Weight = 1 / (count of that class)
    total_samples = len(ds)
    class_weights = 1. / (class_counts + 1e-6)
    sample_weights = [class_weights[t] for t in targets]
    
    # 3. Create Weighted Sampler
    # This FORCES the batch to have equal representation of all directions
    sampler = WeightedRandomSampler(sample_weights, total_samples)
    dl = DataLoader(ds, batch_size=BATCH_SIZE, sampler=sampler)
    
    # 4. Model & Opt
    model = RobustActorCritic(n_actions=8, K=8).to(DEVICE)
    opt = optim.Adam(model.parameters(), lr=1e-3) # Higher LR for BC
    loss_fn = nn.CrossEntropyLoss()
    
    # 5. Loop
    model.train()
    for epoch in range(EPOCHS):
        total_loss = 0
        correct = 0
        total = 0
        
        pbar = tqdm(dl, desc=f"Epoch {epoch+1}")
        for obs, ahist, target in pbar:
            obs, ahist, target = obs.to(DEVICE), ahist.to(DEVICE), target.to(DEVICE)
            dummy_gt = torch.zeros((obs.shape[0], 1, CROP_SIZE, CROP_SIZE)).to(DEVICE)
            
            logits, _, _ = model(obs, dummy_gt, ahist)
            
            loss = loss_fn(logits, target)
            
            opt.zero_grad()
            loss.backward()
            opt.step()
            
            total_loss += loss.item()
            
            # Accuracy
            preds = torch.argmax(logits, dim=1)
            correct += (preds == target).sum().item()
            total += target.size(0)
            
            pbar.set_postfix({'loss': loss.item(), 'acc': correct/total})
            
        print(f"Epoch {epoch+1} Avg Loss: {total_loss/len(dl):.4f} | Acc: {correct/total:.4f}")
    
    torch.save(model.state_dict(), "bc_pretrained_model.pth")
    print("Saved 'bc_pretrained_model.pth'")

if __name__ == "__main__":
    train_bc_robust()
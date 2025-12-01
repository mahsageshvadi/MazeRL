import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from model_and_utils import RobustActorCritic, DEVICE, CROP_SIZE

def train_offline():
    print("--- Loading Dataset into RAM ---")
    data = np.load("bc_dataset.npz")
    
    # Convert to Tensor (Normalize images back to 0-1)
    # Using uint8 saves disk space, we convert to float32 here
    obs = torch.tensor(data['obs'], dtype=torch.float32) / 255.0
    hist = torch.tensor(data['hist'], dtype=torch.float32)
    label = torch.tensor(data['label'], dtype=torch.long)
    
    print(f"Loaded {len(label)} samples.")
    
    # Create standard DataLoader
    ds = TensorDataset(obs, hist, label)
    dl = DataLoader(ds, batch_size=128, shuffle=True) # Batch 128 is faster
    
    # Setup Model
    model = RobustActorCritic(n_actions=8, K=8).to(DEVICE)
    opt = optim.Adam(model.parameters(), lr=3e-4)
    loss_fn = nn.CrossEntropyLoss()
    
    EPOCHS = 15
    print(f"--- Starting Training on {DEVICE} ---")
    
    model.train()
    for epoch in range(EPOCHS):
        total_loss = 0
        correct = 0
        total = 0
        
        for batch_obs, batch_hist, batch_y in dl:
            batch_obs, batch_hist, batch_y = batch_obs.to(DEVICE), batch_hist.to(DEVICE), batch_y.to(DEVICE)
            
            # Dummy GT for Critic
            dummy_gt = torch.zeros((batch_obs.size(0), 1, CROP_SIZE, CROP_SIZE), device=DEVICE)
            
            logits, _, _ = model(batch_obs, dummy_gt, batch_hist)
            loss = loss_fn(logits, batch_y)
            
            opt.zero_grad()
            loss.backward()
            opt.step()
            
            total_loss += loss.item()
            
            # Calc Acc
            preds = torch.argmax(logits, dim=1)
            correct += (preds == batch_y).sum().item()
            total += batch_y.size(0)
            
        print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {total_loss/len(dl):.4f} | Acc: {correct/total:.4f}")
        
    torch.save(model.state_dict(), "bc_pretrained_model.pth")
    print("Saved 'bc_pretrained_model.pth'")

if __name__ == "__main__":
    train_offline()
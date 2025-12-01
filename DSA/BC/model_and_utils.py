import torch
import torch.nn as nn
import numpy as np

# --- CONFIGURATION ---
CROP_SIZE = 48  # Increased from 33
ACTIONS_8 = [(-1, 0), (1, 0), (0,-1), (0, 1), (-1,-1), (-1,1), (1,-1), (1,1)]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def clamp(v, lo, hi): return max(lo, min(v, hi))

def crop48(img: np.ndarray, cy: int, cx: int):
    """Robust 48x48 crop with Smart Padding (Corner Detection)."""
    h, w = img.shape
    size = CROP_SIZE
    
    # Smart Padding: Check 4 corners to guess background
    corners = [img[0,0], img[0, w-1], img[h-1, 0], img[h-1, w-1]]
    bg_estimate = np.median(corners)
    pad_val = 1.0 if bg_estimate > 0.5 else 0.0
    
    r = size // 2
    y0, y1 = cy - r, cy + r
    x0, x1 = cx - r, cx + r
    
    out = np.full((size, size), pad_val, dtype=img.dtype)
    
    sy0, sy1 = clamp(y0, 0, h), clamp(y1, 0, h)
    sx0, sx1 = clamp(x0, 0, w), clamp(x1, 0, w)
    
    oy0 = sy0 - y0; ox0 = sx0 - x0
    sh  = sy1 - sy0; sw  = sx1 - sx0
    
    if sh > 0 and sw > 0:
        out[oy0:oy0+sh, ox0:ox0+sw] = img[sy0:sy1, sx0:sx1]
    return out

def get_action_from_vector(dy, dx):
    """Converts a continuous vector to the closest discrete action index."""
    best_idx = -1
    max_dot = -float('inf')
    mag = np.sqrt(dy**2 + dx**2) + 1e-6
    uy, ux = dy/mag, dx/mag
    
    for i, (ay, ax) in enumerate(ACTIONS_8):
        amag = np.sqrt(ay**2 + ax**2)
        ny, nx = ay/amag, ax/amag
        dot = uy*ny + ux*nx
        if dot > max_dot:
            max_dot = dot
            best_idx = i
    return best_idx

def gn(c): return nn.GroupNorm(8, c)

class RobustActorCritic(nn.Module):
    def __init__(self, n_actions=8, K=8):
        super().__init__()
        
        # Deeper CNN for 48x48 input
        self.cnn = nn.Sequential(
            # 48x48 -> 24x24
            nn.Conv2d(4, 32, 3, padding=1), gn(32), nn.LeakyReLU(0.2),
            nn.MaxPool2d(2), 
            
            # 24x24 -> 12x12
            nn.Conv2d(32, 64, 3, padding=1), gn(64), nn.LeakyReLU(0.2),
            nn.MaxPool2d(2),
            
            # 12x12 -> 6x6
            nn.Conv2d(64, 128, 3, padding=1), gn(128), nn.LeakyReLU(0.2),
            nn.MaxPool2d(2),
            
            # Global Average Pooling -> 128 feature vector
            nn.AdaptiveAvgPool2d((1,1)) 
        )
        
        # LSTM for Momentum (Critical for handling gaps/intersections)
        self.lstm = nn.LSTM(input_size=n_actions, hidden_size=128, num_layers=1, batch_first=True)
        
        # Combined Head
        self.actor = nn.Sequential(
            nn.Linear(128+128, 256), nn.LeakyReLU(0.2),
            nn.Linear(256, n_actions)
        )
        
        self.critic = nn.Sequential(
            nn.Linear(128+128, 256), nn.LeakyReLU(0.2),
            nn.Linear(256, 1)
        )
        
        # Critic specific CNN (Accepts GT map as 5th channel)
        self.critic_cnn = nn.Sequential(
            nn.Conv2d(5, 32, 3, padding=1), gn(32), nn.LeakyReLU(0.2),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), gn(64), nn.LeakyReLU(0.2),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), gn(128), nn.LeakyReLU(0.2),
            nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d((1,1))
        )

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
            if m.bias is not None: nn.init.constant_(m.bias, 0)

    def forward(self, x, gt_map, ahist, hc=None):
        # 1. Visual Features
        z = self.cnn(x).flatten(1) # (B, 128)
        
        # 2. Temporal Features
        _, hc = self.lstm(ahist, hc)
        h_last = hc[0][-1] # (B, 128)
        
        # 3. Combine
        joint = torch.cat([z, h_last], dim=1) # (B, 256)
        
        # 4. Actor Output
        logits = self.actor(joint)
        
        # 5. Critic Output (Uses separate CNN with GT)
        # Concatenate GT map (B,1,48,48) to input (B,4,48,48) -> (B,5,48,48)
        critic_input = torch.cat([x, gt_map], dim=1)
        z_c = self.critic_cnn(critic_input).flatten(1)
        joint_c = torch.cat([z_c, h_last], dim=1)
        value = self.critic(joint_c).squeeze(-1)
        
        return logits, value, hc
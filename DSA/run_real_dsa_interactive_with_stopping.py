#!/usr/bin/env python3
import argparse
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import cv2
from scipy.interpolate import splprep, splev

# ---------- CONFIGURATION ----------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
ACTIONS_MOVEMENT = [(-1, 0), (1, 0), (0,-1), (0, 1), (-1,-1), (-1,1), (1,-1), (1,1)]
ACTION_STOP_IDX = 8
N_ACTIONS = 9
CROP_SIZE = 33

# ---------- MODEL ARCHITECTURE ----------
# We define this here to make the script standalone
def gn(c): return nn.GroupNorm(4, c)

class AsymmetricActorCritic(nn.Module):
    def __init__(self, n_actions=9, K=8):
        super().__init__()
        self.actor_cnn = nn.Sequential(
            nn.Conv2d(4, 32, 3, padding=1), gn(32), nn.PReLU(),
            nn.Conv2d(32, 32, 3, padding=2, dilation=2), gn(32), nn.PReLU(),
            nn.Conv2d(32, 64, 3, padding=3, dilation=3), gn(64), nn.PReLU(),
            nn.AdaptiveAvgPool2d((1,1))
        )
        self.actor_lstm = nn.LSTM(input_size=n_actions, hidden_size=64, batch_first=True)
        self.actor_head = nn.Sequential(nn.Linear(128, 128), nn.PReLU(), nn.Linear(128, n_actions))

        self.critic_cnn = nn.Sequential(
            nn.Conv2d(5, 32, 3, padding=1), gn(32), nn.PReLU(),
            nn.Conv2d(32, 32, 3, padding=2, dilation=2), gn(32), nn.PReLU(),
            nn.Conv2d(32, 64, 3, padding=3, dilation=3), gn(64), nn.PReLU(),
            nn.AdaptiveAvgPool2d((1,1))
        )
        self.critic_lstm = nn.LSTM(input_size=n_actions, hidden_size=64, batch_first=True)
        self.critic_head = nn.Sequential(nn.Linear(128, 128), nn.PReLU(), nn.Linear(128, 1))
    
    def forward(self, actor_obs, critic_gt, ahist_onehot, hc_actor=None, hc_critic=None):
        feat_a = self.actor_cnn(actor_obs).flatten(1)
        lstm_a, hc_actor = self.actor_lstm(ahist_onehot, hc_actor)
        joint_a = torch.cat([feat_a, lstm_a[:, -1, :]], dim=1)
        logits = self.actor_head(joint_a)
        return logits, None, hc_actor, None

# ---------- UTILITIES ----------
def clamp(v, lo, hi): return max(lo, min(v, hi))

def crop32_inference(img: np.ndarray, cy: int, cx: int, size=CROP_SIZE):
    h, w = img.shape
    # Smart Padding inference
    corners = [img[0,0], img[0, w-1], img[h-1, 0], img[h-1, w-1]]
    bg_estimate = np.median(corners)
    pad_val = 1.0 if bg_estimate > 0.5 else 0.0
    
    r = size // 2
    y0, y1 = cy - r, cy + r + 1
    x0, x1 = cx - r, cx + r + 1
    
    out = np.full((size, size), pad_val, dtype=img.dtype)
    sy0, sy1 = clamp(y0, 0, h), clamp(y1, 0, h)
    sx0, sx1 = clamp(x0, 0, w), clamp(x1, 0, w)
    
    oy0 = sy0 - y0; ox0 = sx0 - x0
    sh  = sy1 - sy0; sw  = sx1 - sx0
    
    if sh > 0 and sw > 0:
        out[oy0:oy0+sh, ox0:ox0+sw] = img[sy0:sy1, sx0:sx1]
    return out

def fixed_window_history(ahist_list, K, n_actions):
    out = np.zeros((K, n_actions), dtype=np.float32)
    if len(ahist_list) == 0: return out
    tail = ahist_list[-K:]
    out[-len(tail):] = np.stack(tail, axis=0)
    return out

def get_closest_action_vector(dy, dx):
    """Finds the closest discrete movement action to the user's click vector."""
    best_idx = 0
    max_dot = -float('inf')
    mag = np.sqrt(dy**2 + dx**2) + 1e-6
    uy, ux = dy/mag, dx/mag
    
    for i, (ay, ax) in enumerate(ACTIONS_MOVEMENT):
        amag = np.sqrt(ay**2 + ax**2)
        ny, nx = ay/amag, ax/amag
        dot = uy*ny + ux*nx
        if dot > max_dot:
            max_dot = dot
            best_idx = i
    return best_idx

def preprocess_full_image(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None: return None
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(16,16))
    img = clahe.apply(img)
    img = img.astype(np.float32) / 255.0
    # Auto invert if background seems white
    if np.median(img) > 0.5:
        print("Detected Light Background. Inverting...")
        img = 1.0 - img
    return img

# ---------- INFERENCE ENVIRONMENT ----------
class InferenceEnv:
    def __init__(self, full_img, start_pt, start_vector, max_steps=1000):
        self.img = full_img
        self.h, self.w = full_img.shape
        self.max_steps = max_steps
        
        # 1. Position Setup
        # start_pt is (y, x)
        self.agent = tuple(start_pt)
        
        # 2. History Initialization (Momentum)
        # We back-project the start vector to create a fake history
        dy, dx = start_vector
        mag = np.sqrt(dy**2 + dx**2) + 1e-6
        dy, dx = (dy/mag) * 2.0, (dx/mag) * 2.0
        
        p_prev1 = (self.agent[0] - dy, self.agent[1] - dx)
        p_prev2 = (self.agent[0] - dy*2, self.agent[1] - dx*2)
        
        self.history_pos = [p_prev2, p_prev1, self.agent]
        self.path_points = [self.agent]
        self.steps = 0
        
        # Visited mask
        self.path_mask = np.zeros_like(full_img, dtype=np.float32)
        self.path_mask[int(self.agent[0]), int(self.agent[1])] = 1.0
        self.visited = set()
        self.visited.add((int(self.agent[0]), int(self.agent[1])))

    def obs(self):
        curr = self.history_pos[-1]
        p1 = self.history_pos[-2]
        p2 = self.history_pos[-3]
        
        ch0 = crop32_inference(self.img, int(curr[0]), int(curr[1]))
        ch1 = crop32_inference(self.img, int(p1[0]), int(p1[1]))
        ch2 = crop32_inference(self.img, int(p2[0]), int(p2[1]))
        ch3 = crop32_inference(self.path_mask, int(curr[0]), int(curr[1]))
        
        actor_obs = np.stack([ch0, ch1, ch2, ch3], axis=0).astype(np.float32)
        return actor_obs

    def step(self, a_idx):
        self.steps += 1
        
        # --- STOP ACTION ---
        if a_idx == ACTION_STOP_IDX:
            return self.obs(), True, "Agent Stopped"

        # --- MOVEMENT ---
        dy, dx = ACTIONS_MOVEMENT[a_idx]
        STEP_ALPHA = 2.0
        
        ny = self.agent[0] + dy * STEP_ALPHA
        nx = self.agent[1] + dx * STEP_ALPHA
        
        # Boundary Check
        if not (0 <= ny < self.h and 0 <= nx < self.w):
            return self.obs(), True, "Hit Border"

        iy, ix = int(ny), int(nx)
        
        # Loop Check
        if (iy, ix) in self.visited and self.steps > 20:
             # Just strict loop checking might kill valid crossings
             # We allow crossing if it's been a while, but here we keep it simple
             pass 
        self.visited.add((iy, ix))

        self.agent = (ny, nx)
        self.history_pos.append(self.agent)
        self.path_points.append(self.agent)
        self.path_mask[iy, ix] = 1.0

        if self.steps >= self.max_steps:
            return self.obs(), True, "Max Steps Reached"

        return self.obs(), False, "Running"

# ---------- UI & MAIN ----------
coords = []
def onclick(event):
    global coords
    if event.xdata and event.ydata:
        ix, iy = int(event.xdata), int(event.ydata)
        coords.append((ix, iy))
        plt.plot(ix, iy, 'ro', markersize=5)
        if len(coords) > 1:
            plt.plot([coords[-2][0], coords[-1][0]], [coords[-2][1], coords[-1][1]], 'r-', linewidth=2)
        plt.draw()
        if len(coords) == 2:
            print("Direction captured. Closing window...")
            plt.pause(0.5)
            plt.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_path", type=str, required=True)
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--max_steps", type=int, default=1000)
    args = parser.parse_args()

    # 1. Load Image
    raw_img = cv2.imread(args.image_path, cv2.IMREAD_GRAYSCALE)
    processed_img = preprocess_full_image(args.image_path)
    if processed_img is None:
        print("Error reading image.")
        return

    # 2. Load Model
    K = 8
    model = AsymmetricActorCritic(n_actions=N_ACTIONS, K=K).to(DEVICE)
    try:
        model.load_state_dict(torch.load(args.model_path, map_location=DEVICE))
    except Exception as e:
        print(f"Failed to load model: {e}")
        print("Ensure you are using the model trained with 9 actions.")
        return
    model.eval()

    # 3. Get User Input
    print("--- CLICK START POINT AND DIRECTION ---")
    plt.figure(figsize=(10, 10))
    plt.imshow(raw_img, cmap='gray')
    plt.title("Click 1: Start | Click 2: Direction")
    plt.connect('button_press_event', onclick)
    plt.show()

    if len(coords) < 2:
        print("No inputs provided.")
        return

    # Matplotlib is (x,y), Agent is (row, col) i.e. (y,x)
    p1_x, p1_y = coords[0]
    p2_x, p2_y = coords[1]
    
    vec_y = p2_y - p1_y
    vec_x = p2_x - p1_x

    # 4. Initialize Inference
    env = InferenceEnv(processed_img, start_pt=(p1_y, p1_x), start_vector=(vec_y, vec_x), max_steps=args.max_steps)
    
    # Priming the LSTM history
    # We assume the user click represents the "previous" action
    start_action_idx = get_closest_action_vector(vec_y, vec_x)
    a_onehot = np.zeros(N_ACTIONS); a_onehot[start_action_idx] = 1.0
    ahist = [a_onehot] * K # Fill history with this initial momentum

    print(f"Rolling out... Start: {env.agent}")
    
    done = False
    reason = "Unknown"
    
    while not done:
        actor_obs = env.obs()
        obs_t = torch.tensor(actor_obs[None], dtype=torch.float32, device=DEVICE)
        
        # Prepare History
        A = fixed_window_history(ahist, K, N_ACTIONS)[None, ...]
        A_t = torch.tensor(A, dtype=torch.float32, device=DEVICE)
        
        # Dummy Critic Input (Not used in Actor forward pass usually, but required by signature)
        # Note: In the revised class above, I modified forward to ignore critic input for actor logits
        dummy_gt = torch.zeros((1, 1, 33, 33), device=DEVICE)

        with torch.no_grad():
            logits, _, _, _ = model(obs_t, dummy_gt, A_t)
            probs = torch.softmax(logits, dim=1)
            action = torch.argmax(probs, dim=1).item()

        _, done, reason = env.step(action)
        
        new_onehot = np.zeros(N_ACTIONS); new_onehot[action] = 1.0
        ahist.append(new_onehot)

    print(f"Done. Reason: {reason} | Steps: {env.steps}")

    # 5. Smoothing & Plotting
    path = env.path_points
    if len(path) > 3:
        try:
            y = [p[0] for p in path]
            x = [p[1] for p in path]
            tck, u = splprep([y, x], s=10.0) # Smooth spline
            new = splev(np.linspace(0, 1, len(path)*5), tck)
            sy, sx = new[0], new[1]
        except:
            sy = [p[0] for p in path]
            sx = [p[1] for p in path]
    else:
        sy = [p[0] for p in path]
        sx = [p[1] for p in path]

    plt.figure(figsize=(12, 12))
    plt.imshow(raw_img, cmap='gray')
    plt.plot(sx, sy, 'cyan', linewidth=2.5, alpha=0.8, label='Agent Path')
    plt.plot(p1_x, p1_y, 'go', markersize=10, label='Start')
    # Mark the end point
    plt.plot(sx[-1], sy[-1], 'rx', markersize=10, markeredgewidth=3, label='Stop')
    plt.legend()
    plt.title(f"Result: {reason}")
    plt.show()

if __name__ == "__main__":
    main()
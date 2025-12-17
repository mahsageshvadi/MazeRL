import torch
import numpy as np
import matplotlib.pyplot as plt
import argparse
import cv2
import torch.nn as nn
from scipy.interpolate import splprep, splev

# ---------- GLOBALS ----------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# 8 Moves + 1 Stop (Action 8)
ACTIONS_9 = [(-1, 0), (1, 0), (0,-1), (0, 1), (-1,-1), (-1,1), (1,-1), (1,1), (0,0)]

# --- UTILS ---
def clamp(v, lo, hi): return max(lo, min(v, hi))

def crop32_inference(img: np.ndarray, cy: int, cx: int, size=33):
    h, w = img.shape
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

def preprocess_full_image(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None: return None
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(16,16))
    img = clahe.apply(img)
    img = img.astype(np.float32) / 255.0
    if np.median(img) > 0.5:
        print("Detected Light Background. Inverting Full Image...")
        img = 1.0 - img
    return img

def get_closest_action(dy, dx):
    """Match click vector to discrete action (Indices 0-7 only)"""
    best_idx = -1
    max_dot = -float('inf')
    mag = np.sqrt(dy**2 + dx**2) + 1e-6
    uy, ux = dy/mag, dx/mag
    for i in range(8): 
        ay, ax = ACTIONS_9[i]
        amag = np.sqrt(ay**2 + ax**2)
        ny, nx = ay/amag, ax/amag
        dot = uy*ny + ux*nx
        if dot > max_dot:
            max_dot = dot
            best_idx = i
    return best_idx

def fixed_window_history(ahist_list, K, n_actions):
    out = np.zeros((K, n_actions), dtype=np.float32)
    if len(ahist_list) == 0: return out
    tail = ahist_list[-K:]
    out[-len(tail):] = np.stack(tail, axis=0)
    return out

# --- ARCHITECTURE MATCHING YOUR CHECKPOINT ---
def gn(c): return nn.GroupNorm(4, c)

class Actor9(nn.Module):
    def __init__(self):
        super().__init__()
        # This matches the "Simplified" architecture used in the Phase 9 snippet
        self.actor_cnn = nn.Sequential(
            nn.Conv2d(4, 32, 3, padding=1), gn(32), nn.PReLU(),
            nn.Conv2d(32, 64, 3, padding=2, dilation=2), gn(64), nn.PReLU(),
            nn.AdaptiveAvgPool2d((1,1))
        )
        self.actor_lstm = nn.LSTM(input_size=9, hidden_size=64, batch_first=True)
        # Head maps 128 -> 64 -> 9
        self.actor_head = nn.Sequential(nn.Linear(128, 64), nn.PReLU(), nn.Linear(64, 9))

    def forward(self, obs, ahist):
        feat = self.actor_cnn(obs).flatten(1)
        lstm_out, _ = self.actor_lstm(ahist)
        joint = torch.cat([feat, lstm_out[:, -1, :]], dim=1)
        return self.actor_head(joint)

# --- ENVIRONMENT ---
class FullImageEnv:
    def __init__(self, full_img, start_pt, start_vector, max_steps=1000):
        self.img = full_img
        self.h, self.w = full_img.shape
        self.max_steps = max_steps
        self.agent = tuple(start_pt) 
        
        # Momentum Setup
        dy, dx = start_vector
        mag = np.sqrt(dy**2 + dx**2) + 1e-6
        dy, dx = (dy/mag) * 2.0, (dx/mag) * 2.0
        p_prev1 = (self.agent[0] - dy, self.agent[1] - dx)
        p_prev2 = (self.agent[0] - dy*2, self.agent[1] - dx*2)
        
        self.history_pos = [p_prev2, p_prev1, self.agent]
        self.path_points = [self.agent]
        self.steps = 0
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
        
        if a_idx == 8: return True, "Stopped (Action 8)"

        dy, dx = ACTIONS_9[a_idx]
        STEP_ALPHA = 2.0 
        ny = self.agent[0] + dy * STEP_ALPHA
        nx = self.agent[1] + dx * STEP_ALPHA
        
        if not (0 <= ny < self.h and 0 <= nx < self.w):
            return True, "Border"

        iy, ix = int(ny), int(nx)
        if (iy, ix) in self.visited and self.steps > 20:
             return True, "Loop"
        self.visited.add((iy, ix))

        self.agent = (ny, nx)
        self.history_pos.append(self.agent)
        self.path_points.append(self.agent)
        self.path_mask[iy, ix] = 1.0

        if self.steps >= self.max_steps: return True, "Timeout"
        return False, "Run"

# --- MAIN ---
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
            print("Direction set. Starting...")
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
    if raw_img is None: print("Error loading image."); return
    processed_img = preprocess_full_image(args.image_path)

    # 2. Load Model (Using Actor9 class)
    model = Actor9().to(DEVICE)
    try:
        model.load_state_dict(torch.load(args.model_path, map_location=DEVICE))
        print("Weights loaded successfully.")
    except Exception as e:
        print(f"Error loading weights: {e}")
        return
    model.eval()

    # 3. UI
    print("Click START then DIRECTION.")
    plt.figure(figsize=(10, 10))
    plt.imshow(raw_img, cmap='gray')
    plt.connect('button_press_event', onclick)
    plt.show()

    if len(coords) < 2: return
    p1_x, p1_y = coords[0]; p2_x, p2_y = coords[1]
    vec_y = p2_y - p1_y; vec_x = p2_x - p1_x

    # 4. Env Setup
    env = FullImageEnv(processed_img, start_pt=(p1_y, p1_x), start_vector=(vec_y, vec_x), max_steps=args.max_steps)
    
    K = 8; nA = 9
    start_action = get_closest_action(vec_y, vec_x)
    a_onehot = np.zeros(nA); a_onehot[start_action] = 1.0
    ahist = [a_onehot] * K
    
    done = False
    print("Tracking...")
    
    while not done:
        obs_np = env.obs()
        obs_a = torch.tensor(obs_np[None], dtype=torch.float32, device=DEVICE)
        A = fixed_window_history(ahist, K, nA)[None, ...]
        A_t = torch.tensor(A, dtype=torch.float32, device=DEVICE)

        with torch.no_grad():
            # Actor9 returns only logits (no value, no hidden states)
            logits = model(obs_a, A_t)
            action = torch.argmax(logits, dim=1).item()

        done, reason = env.step(action)
        
        new_onehot = np.zeros(nA); new_onehot[action] = 1.0
        ahist.append(new_onehot)

    print(f"Finished: {reason} ({env.steps} steps)")

    # 5. Plot
    path = env.path_points
    try:
        y = [p[0] for p in path]; x = [p[1] for p in path]
        tck, u = splprep([y, x], s=20.0)
        new = splev(np.linspace(0, 1, len(path)*3), tck)
        sy, sx = new[0], new[1]
    except:
        sy, sx = [p[0] for p in path], [p[1] for p in path]

    plt.figure(figsize=(10, 10))
    plt.imshow(raw_img, cmap='gray')
    plt.plot(sx, sy, 'cyan', linewidth=2, label='Path')
    plt.plot(p1_x, p1_y, 'go', label='Start')
    if "Stopped" in reason:
        plt.plot(path[-1][1], path[-1][0], 'rx', markersize=10, markeredgewidth=3, label='Stop')
    plt.legend()
    plt.title(f"Result: {reason}")
    plt.show()

if __name__ == "__main__":
    main()
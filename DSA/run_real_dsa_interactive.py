import torch
import numpy as np
import matplotlib.pyplot as plt
import argparse
import cv2
from scipy.interpolate import splprep, splev

# Import Model Architecture
from Synth_simple_v1_9_paper_version_gemini import AsymmetricActorCritic, fixed_window_history, ACTIONS_8, DEVICE, clamp

# --- UTILS ---
def crop32_inference(img: np.ndarray, cy: int, cx: int, size=33):
    """
    Extracts the Agent's vision (33x33) from the Full Image.
    Handles boundaries by padding if the agent gets close to the edge of the 1024x1024 image.
    """
    h, w = img.shape
    
    # Smart Padding: Check corners of the FULL image to guess background
    # (Usually in DSA, the borders are consistent)
    corners = [img[0,0], img[0, w-1], img[h-1, 0], img[h-1, w-1]]
    bg_estimate = np.median(corners)
    pad_val = 1.0 if bg_estimate > 0.5 else 0.0
    
    r = size // 2
    y0, y1 = cy - r, cy + r + 1
    x0, x1 = cx - r, cx + r + 1
    
    # Initialize output with the background color
    out = np.full((size, size), pad_val, dtype=img.dtype)
    
    # Calculate overlap between the requested crop and the image
    # This handles cases where the agent is at the very edge (e.g. pixel 0,0)
    sy0, sy1 = clamp(y0, 0, h), clamp(y1, 0, h)
    sx0, sx1 = clamp(x0, 0, w), clamp(x1, 0, w)
    
    oy0 = sy0 - y0; ox0 = sx0 - x0
    sh  = sy1 - sy0; sw  = sx1 - sx0
    
    if sh > 0 and sw > 0:
        out[oy0:oy0+sh, ox0:ox0+sw] = img[sy0:sy1, sx0:sx1]
        
    return out

def preprocess_full_image(path):
    """
    Loads and cleans the entire DSA image.
    """
    # 1. Load
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None: return None
    
    # 2. CLAHE (Enhance Contrast globally)
    # We increase grid size slightly for larger images to keep local contrast consistent
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(16,16))
    img = clahe.apply(img)
    
    # 3. Normalize
    img = img.astype(np.float32) / 255.0
    
    # 4. Auto-Invert
    if np.median(img) > 0.5:
        print("Detected Light Background. Inverting Full Image...")
        img = 1.0 - img
        
    return img

def get_closest_action(dy, dx):
    """Match click vector to discrete action"""
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

# --- FREE ROAMING ENVIRONMENT ---
class FullImageEnv:
    def __init__(self, full_img, start_pt, start_vector, max_steps=1000):
        self.img = full_img
        self.h, self.w = full_img.shape
        self.max_steps = max_steps
        
        # 1. GLOBAL Position
        self.agent = tuple(start_pt) # (Global Y, Global X)
        
        # 2. Momentum Setup
        dy, dx = start_vector
        mag = np.sqrt(dy**2 + dx**2) + 1e-6
        dy, dx = (dy/mag) * 2.0, (dx/mag) * 2.0
        
        # History behind the start point
        p_prev1 = (self.agent[0] - dy, self.agent[1] - dx)
        p_prev2 = (self.agent[0] - dy*2, self.agent[1] - dx*2)
        
        self.history_pos = [p_prev2, p_prev1, self.agent]
        self.path_points = [self.agent]
        self.steps = 0
        
        # We maintain a sparse mask or set for visited pixels on the full image
        # Creating a 1024x1024 mask is cheap in memory
        self.path_mask = np.zeros_like(full_img, dtype=np.float32)
        self.path_mask[int(self.agent[0]), int(self.agent[1])] = 1.0
        
        self.visited = set()
        self.visited.add((int(self.agent[0]), int(self.agent[1])))

    def obs(self):
        curr = self.history_pos[-1]
        p1 = self.history_pos[-2]
        p2 = self.history_pos[-3]
        
        # KEY DIFFERENCE: We crop from the FULL image at the current global pos
        ch0 = crop32_inference(self.img, int(curr[0]), int(curr[1]))
        ch1 = crop32_inference(self.img, int(p1[0]), int(p1[1]))
        ch2 = crop32_inference(self.img, int(p2[0]), int(p2[1]))
        
        # Crop the mask too
        ch3 = crop32_inference(self.path_mask, int(curr[0]), int(curr[1]))
        
        actor_obs = np.stack([ch0, ch1, ch2, ch3], axis=0).astype(np.float32)
        gt_dummy = np.zeros((1, 33, 33), dtype=np.float32)
        return {"actor": actor_obs, "critic_gt": gt_dummy}

    def step(self, a_idx):
        self.steps += 1
        dy, dx = ACTIONS_8[a_idx]
        STEP_ALPHA = 2.0 
        
        # Move in Global Coordinates
        ny = self.agent[0] + dy * STEP_ALPHA
        nx = self.agent[1] + dx * STEP_ALPHA
        
        # Check Global Boundaries
        if not (0 <= ny < self.h and 0 <= nx < self.w):
            return self.obs(), True, "Hit Image Border"

        iy, ix = int(ny), int(nx)
        
        # Loop Detection
        if (iy, ix) in self.visited and self.steps > 20:
             return self.obs(), True, "Loop Detected"
        self.visited.add((iy, ix))

        # Update State
        self.agent = (ny, nx)
        self.history_pos.append(self.agent)
        self.path_points.append(self.agent)
        self.path_mask[iy, ix] = 1.0

        if self.steps >= self.max_steps:
            return self.obs(), True, "Max Steps"

        return self.obs(), False, "Running"

# --- INTERFACE ---
coords = []
def onclick(event):
    global coords
    if event.xdata and event.ydata:
        ix, iy = int(event.xdata), int(event.ydata)
        coords.append((ix, iy))
        
        # Draw Markers
        plt.plot(ix, iy, 'ro', markersize=5)
        if len(coords) > 1:
            plt.plot([coords[-2][0], coords[-1][0]], [coords[-2][1], coords[-1][1]], 'r-', linewidth=2)
        plt.draw()
        
        if len(coords) == 2:
            print("Direction set. Starting tracking...")
            plt.pause(0.5)
            plt.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_path", type=str, required=True)
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--max_steps", type=int, default=1000, help="Max steps before stopping")
    args = parser.parse_args()

    # 1. Load Full Image (Raw for display, Processed for Agent)
    raw_img = cv2.imread(args.image_path, cv2.IMREAD_GRAYSCALE)
    if raw_img is None:
        print("Error loading image.")
        return
    
    processed_img = preprocess_full_image(args.image_path)

    # 2. Load Model
    K = 8; nA = len(ACTIONS_8)
    model = AsymmetricActorCritic(n_actions=nA, K=K).to(DEVICE)
    model.load_state_dict(torch.load(args.model_path, map_location=DEVICE))
    model.eval()

    # 3. Get Clicks
    print("\n--- INSTRUCTIONS ---")
    print("1. Click START.")
    print("2. Click DIRECTION.")
    print("--------------------")

    plt.figure(figsize=(12, 12))
    plt.imshow(raw_img, cmap='gray')
    plt.title("Select Start & Direction")
    plt.connect('button_press_event', onclick)
    plt.show()

    if len(coords) < 2: return

    # Note: Matplotlib gives (x, y), we need (y, x) for numpy/agent
    p1_x, p1_y = coords[0]
    p2_x, p2_y = coords[1]
    
    # Vector (y, x)
    vec_y = p2_y - p1_y
    vec_x = p2_x - p1_x

    # 4. Initialize Full Env
    # We pass the FULL processed image here
    env = FullImageEnv(processed_img, start_pt=(p1_y, p1_x), start_vector=(vec_y, vec_x), max_steps=args.max_steps)
    
    # 5. Action Priming (Critical!)
    start_action = get_closest_action(vec_y, vec_x)
    a_onehot = np.zeros(nA); a_onehot[start_action] = 1.0
    ahist = [a_onehot] * K
    
    done = False
    print(f"Tracking started... (Max steps: {args.max_steps})")
    
    while not done:
        obs_dict = env.obs()
        obs_a = torch.tensor(obs_dict['actor'][None], dtype=torch.float32, device=DEVICE)
        gt_dummy = torch.zeros((1, 1, 33, 33), device=DEVICE)
        A = fixed_window_history(ahist, K, nA)[None, ...]
        A_t = torch.tensor(A, dtype=torch.float32, device=DEVICE)

        with torch.no_grad():
            logits, _, _, _ = model(obs_a, gt_dummy, A_t)
            action = torch.argmax(logits, dim=1).item()

        _, done, reason = env.step(action)
        
        new_onehot = np.zeros(nA); new_onehot[action] = 1.0
        ahist.append(new_onehot)

    print(f"Finished: {reason} ({env.steps} steps)")

    # 6. Visualization
    # We plot the result on the ORIGINAL full-size image
    path = env.path_points
    try:
        y = [p[0] for p in path]; x = [p[1] for p in path]
        tck, u = splprep([y, x], s=20.0) # Higher smoothing for long paths
        new = splev(np.linspace(0, 1, len(path)*3), tck)
        sy, sx = new[0], new[1]
    except:
        sy, sx = [p[0] for p in path], [p[1] for p in path]

    plt.figure(figsize=(12, 12))
    plt.imshow(raw_img, cmap='gray')
    plt.plot(sx, sy, 'cyan', linewidth=2, label='Tracked Path')
    plt.plot(p1_x, p1_y, 'go', markersize=8, label='Start')
    plt.legend()
    plt.title(f"Full Image Tracking: {reason}")
    plt.show()

if __name__ == "__main__":
    main()
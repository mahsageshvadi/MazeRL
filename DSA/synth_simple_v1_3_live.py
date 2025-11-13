
import argparse, math, random
from dataclasses import dataclass
from typing import List, Tuple, Optional
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

try:
    from Curve_Generator import CurveMaker  
    _HAVE_EXTERNAL_CURVE = True
except Exception:
    _HAVE_EXTERNAL_CURVE = False

    class CurveMaker:
        """
        Fallback curve generator: creates one smooth polyline (optionally noisy).
        Produces (img, mask, [poly]) where:
          - img: 2D float image with a faint vessel-like line
          - mask: binary centerline mask
          - polys: list with a single Nx2 (y,x) float array
        """
        def __init__(self, h=128, w=128, thickness=1.5, seed=None):
            self.h, self.w = h, w
            self.thickness = thickness
            if seed is not None:
                random.seed(seed); np.random.seed(seed)

        def _polyline(self, n=80, amp=20):
            xs = np.linspace(10, self.w-10, n)
            ys = (self.h/2.0
                  + amp*np.sin(xs/12.0 + random.random()*2*np.pi)
                  + np.random.normal(0, 1.0, size=n))
            ys = np.clip(ys, 5, self.h-6)
            return np.stack([ys, xs], axis=1).astype(np.float32)

        def _draw_line(self, img, p0, p1, val):
            # Simple DDA
            y0, x0 = p0; y1, x1 = p1
            n = int(max(abs(y1-y0), abs(x1-x0))) + 1
            for t in range(n+1):
                a = t/float(max(n,1))
                y = int(round((1-a)*y0 + a*y1))
                x = int(round((1-a)*x0 + a*x1))
                if 0 <= y < img.shape[0] and 0 <= x < img.shape[1]:
                    img[y,x] = max(img[y,x], val)

        def sample_curve(self, branches=False):
            poly = self._polyline(n=random.randint(70, 100),
                                  amp=random.randint(12, 24))
            img  = np.zeros((self.h, self.w), dtype=np.float32)
            mask = np.zeros_like(img, dtype=np.uint8)
            # Paint the centerline
            for i in range(len(poly)-1):
                self._draw_line(mask, poly[i], poly[i+1], 1)
                self._draw_line(img,  poly[i], poly[i+1], 0.7)
            # Cheap blur-ish
            img = 0.8*img + 0.2*np.clip(np.roll(img, 1, 0) + np.roll(img, -1, 0) + np.roll(img, 1, 1) + np.roll(img, -1, 1), 0, 1)/4.0
            img = img.astype(np.float32)
            return img, mask, [poly]

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
ACTIONS_8 = [(-1, 0), (1, 0), (0,-1), (0, 1), (-1,-1), (-1,1), (1,-1), (1,1)]
ACTIONS_8_NAMES = ["N","S","W","E","NW","NE","SW","SE"]  # match ACTIONS_8 order
STEP_ALPHA = 2
CROP = 33
DILATE_RADIUS = 4
LAMBDA_B     = 2.0
LAMBDA_DELTA = 0.7
EPS_LOG      = 1e-2
R_LOG_CLAMP  = 3.0

def set_seeds(seed=123):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if DEVICE == "cuda": torch.cuda.manual_seed_all(seed)

def clamp(v, lo, hi): return max(lo, min(v, hi))

def crop32(img: np.ndarray, cy: int, cx: int, size=CROP):
    h, w = img.shape
    r = size // 2
    y0, y1 = cy - r, cy + r + 1
    x0, x1 = cx - r, cx + r + 1
    sy0, sy1 = clamp(y0, 0, h), clamp(y1, 0, h)
    sx0, sx1 = clamp(x0, 0, w), clamp(x1, 0, w)
    out = np.zeros((size, size), dtype=img.dtype)
    oy0 = sy0 - y0; ox0 = sx0 - x0
    sh  = sy1 - sy0; sw  = sx1 - sx0
    oy1 = oy0 + sh; ox1 = ox0 + sw
    if sh > 0 and sw > 0:
        out[oy0:oy1, ox0:ox1] = img[sy0:sy1, sx0:sx1]
    return out

def fixed_window_history(ahist_list, K, n_actions):
    out = np.zeros((K, n_actions), dtype=np.float32)
    if len(ahist_list) == 0: return out
    tail = ahist_list[-K:]
    out[-len(tail):] = np.stack(tail, axis=0)
    return out

def fixed_window_features(fhist_list, K, Fdim):
    out = np.zeros((K, Fdim), dtype=np.float32)
    if len(fhist_list) == 0: return out
    tail = fhist_list[-K:]
    out[-len(tail):] = np.stack(tail, axis=0)
    return out

def nearest_gt_index(pt, poly):
    dif = poly - np.array(pt, dtype=np.float32)
    d2 = np.sum(dif * dif, axis=1)
    i = int(np.argmin(d2))
    return i, float(np.sqrt(d2[i]))

def dtw_curve_distance(path_points: List[Tuple[int,int]], gt_poly: np.ndarray) -> float:
    if len(path_points) == 0 or len(gt_poly) == 0:
        return 0.0
    P = np.asarray(path_points, dtype=np.float32)
    G = np.asarray(gt_poly, dtype=np.float32)
    n, m = P.shape[0], G.shape[0]
    C = np.empty((n, m), dtype=np.float32)
    for i in range(n):
        for j in range(m):
            C[i, j] = np.linalg.norm(P[i] - G[j])
    dp = np.full((n, m), np.inf, dtype=np.float32)
    dp[0, 0] = C[0, 0]
    for i in range(1, n): dp[i, 0] = C[i, 0] + dp[i-1, 0]
    for j in range(1, m): dp[0, j] = C[0, j] + dp[0, j-1]
    for i in range(1, n):
        for j in range(1, m):
            dp[i, j] = C[i, j] + min(dp[i-1, j], dp[i, j-1], dp[i-1, j-1])
    norm = float(n + m)
    return float(dp[n-1, m-1] / max(norm, 1.0))

def compute_ccs(L_t: float, L0: float) -> float:
    L0 = max(L0, 1e-6)
    return float(1.0 - (L_t / L0))

def disk_offsets(radius: int):
    offs = []
    r2 = radius * radius
    for dy in range(-radius, radius+1):
        for dx in range(-radius, radius+1):
            if dy*dy + dx*dx <= r2:
                offs.append((dy, dx))
    return offs

def dilate_mask(mask: np.ndarray, radius: int) -> np.ndarray:
    H, W = mask.shape
    out = np.zeros_like(mask, dtype=np.uint8)
    ones = np.argwhere(mask > 0)
    offs = disk_offsets(radius)
    for (y, x) in ones:
        for (dy, dx) in offs:
            ny, nx = y + dy, x + dx
            if 0 <= ny < H and 0 <= nx < W:
                out[ny, nx] = 1
    return out

@dataclass
class CurveEpisode:
    img: np.ndarray
    mask: np.ndarray
    gt_poly: np.ndarray
    start: Tuple[int,int]
    init_dir: Tuple[int,int]
    ternary_map: np.ndarray
    dilated_gt: np.ndarray

class CurveEnv:
    def __init__(self, h=128, w=128, branches=False, max_steps=400,
                 d0=3.0, dilate_radius=DILATE_RADIUS, start_mode: str = "begin", overlap_dist=1.0):
        self.h, self.w = h, w
        self.max_steps = max_steps
        self.cm = CurveMaker(h=h, w=w, thickness=1.5, seed=None)
        self.branches = branches
        self.D0 = d0
        self.dilate_radius = dilate_radius
        self.off_track_thresh = 1.8
        self.start_mode = start_mode  # 'begin' | 'left_end' | 'random'
        self.overlap_dist = overlap_dist
        self.reset()

    def _apply_start_mode(self, gt_poly: np.ndarray) -> np.ndarray:
        if self.start_mode == "left_end":
            e0, e1 = gt_poly[0], gt_poly[-1]
            # Ensure we start from the smaller x endpoint (leftmost)
            if e0[1] <= e1[1]:
                return gt_poly
            else:
                return gt_poly[::-1].copy()
        elif self.start_mode == "random":
            idx = int(np.random.randint(0, len(gt_poly)))
            return np.concatenate([gt_poly[idx:], gt_poly[:idx]], axis=0)
        # default: 'begin'
        return gt_poly

    def reset(self):
        img, mask, pts_all = self.cm.sample_curve(branches=self.branches)
        gt_poly = pts_all[0].astype(np.float32)
        gt_poly = self._apply_start_mode(gt_poly)

        p0 = gt_poly[0].astype(int)
        p1 = gt_poly[min(5, len(gt_poly)-1)].astype(int)
        init_vec = np.sign(np.array([p1[0]-p0[0], p1[1]-p0[1]], dtype=np.int32))
        init_vec[init_vec==0] = 1

        tern = np.zeros_like(mask, dtype=np.float32)
        for (yy, xx) in gt_poly.astype(int):
            if 0 <= yy < tern.shape[0] and 0 <= xx < tern.shape[1]:
                tern[yy, xx] = 1.0
        dil = dilate_mask(tern > 0, self.dilate_radius).astype(np.float32)

        self.ep = CurveEpisode(img=img, mask=mask, gt_poly=gt_poly,
                               start=(int(p0[0]), int(p0[1])),
                               init_dir=(int(init_vec[0]), int(init_vec[1])),
                               ternary_map=tern,
                               dilated_gt=dil)
        self.agent = (int(p0[0]), int(p0[1]))
        self.prev  = [self.agent, self.agent]
        self.steps = 0
        self.path_mask = np.zeros_like(mask, dtype=np.float32)
        self.path_points: List[Tuple[int,int]] = [self.agent]
        self.path_mask[self.agent] = 1.0
        self.prev_index = -1

        self.best_idx = 0
        _, d0_local = nearest_gt_index(self.agent, self.ep.gt_poly)
        self.L_prev_local = d0_local

        stuck_path = [self.agent] * len(self.ep.gt_poly)
        self.L0 = dtw_curve_distance(stuck_path, self.ep.gt_poly)
        if self.L0 < 1e-6: self.L0 = 1.0
        self.last_reward = 0.0
        return self.obs()

    def obs(self):
        p_t, p_1, p_2 = self.agent, self.prev[0], self.prev[1]
        ch0 = crop32(self.ep.img,  p_t[0], p_t[1])
        ch1 = crop32(self.ep.img,  p_1[0], p_1[1])
        ch2 = crop32(self.ep.img,  p_2[0], p_2[1])
        ch3 = crop32(self.path_mask, p_t[0], p_t[1])
        ch4 = crop32(self.ep.ternary_map, p_t[0], p_t[1])
        obs = np.stack([ch0, ch1, ch2, ch3, ch4], axis=0).astype(np.float32)
        return obs

    def step(self, a_idx: int):
        self.steps += 1
        dy, dx = ACTIONS_8[a_idx]
        ny = clamp(self.agent[0] + dy*STEP_ALPHA, 0, self.h-1)
        nx = clamp(self.agent[1] + dx*STEP_ALPHA, 0, self.w-1)

        print(f"ny: {ny}, nx: {nx}")
        print(f"agent: {self.agent}")
        new_pos = (ny, nx)

        # Move & mark
        self.prev = [self.agent, self.prev[0]]
        self.agent = new_pos
        self.path_points.append(self.agent)
        self.path_mask[self.agent] = 1.0

        # Local surface distance at current position
        idx, d_gt = nearest_gt_index(self.agent, self.ep.gt_poly)
        delta = d_gt - self.L_prev_local
         
        # Binary overlap metric
        on_curve = d_gt < self.overlap_dist
        B_t = 1.0 if on_curve else 0.0
         
        # Curve-to-curve distance reward (Equation 3 from paper)
        eps = 1e-4
        delta_abs = abs(delta)
 
        print("#############################")
        print(f"d_gt: {d_gt}")
        print(f"delta: {delta}")
        print(f"delta_abs: {delta_abs / self.D0}")
        
       # if delta < 0:  # Getting closer to the curve
       #     r =  math.log(eps + delta_abs / self.D0) # + B_t
       # else:  # Getting farther or staying same distance
       #         r =  - math.log(eps + delta_abs / self.D0) #+ B_t

        # Clip reward to reasonable range
     #  r = float(np.clip(r, -10.0, 10.0))
        
        if delta < 0:
            r = math.log(eps + delta_abs / self.D0)
        else:
            r = - math.log(eps + delta_abs / self.D0)
        # Update local distance for next step
        self.L_prev_local = d_gt
        
        if idx <= self.prev_index:
            r -= 1.0
        else:
            r += 1.0
        self.prev_index = idx

        if idx > self.best_idx:
            self.best_idx = idx
         

        ref_length = len(self.ep.gt_poly)
        track_length = len(self.path_points)
        exceeded_length = track_length > 1.5 * ref_length
        
        # (2) Agent is off reference by 1.8mm (4 voxels with 0.45mm spacing)
        off_track = d_gt > 5
        
        # (3) Target reached (near end of curve)
        end_margin = 5
        reached_end = (self.best_idx >= len(self.ep.gt_poly) - 1 - end_margin)
        
        # (4) Timeout
        timeout = (self.steps >= self.max_steps)
        
        done = exceeded_length  or off_track or reached_end or timeout
        
        # Terminal n rewards
        if reached_end:
            r += 10.0  # Success bonus

        L_t = dtw_curve_distance(self.path_points, self.ep.gt_poly)
        ccs = compute_ccs(L_t, self.L0)
        self.last_reward = r

        print(f"r: {r}") 
        print(f"B_t: {B_t}")
        print(f"INDEX: {idx}")
        print("#############################")

        return self.obs(), float(r), done, {
            "overlap": 1.0 if on_curve else 0.0,
            "L_local": d_gt,
            "idx": idx,
            "reached_end": reached_end,
            "timeout": timeout,
            "exceeded_length": exceeded_length,
            "off_track": off_track,
            "ccs": ccs,
            "agent": self.agent,
        }

def inorm(c): 
    return nn.InstanceNorm2d(c, eps=1e-5, affine=True)

class ActorCritic(nn.Module):
    def __init__(self, n_actions=8, K=8, feat_dim=4, in_ch=5):
        super().__init__()
        self.n_actions = n_actions
        self.K = K
        self.feat_dim = feat_dim

        self.cnn = nn.Sequential(
            nn.Conv2d(in_ch, 32, 3, padding=1, dilation=1), inorm(32), nn.PReLU(),
            nn.Conv2d(32, 32, 3, padding=2, dilation=2),    inorm(32), nn.PReLU(),
            nn.Conv2d(32, 32, 3, padding=3, dilation=3),    inorm(32), nn.PReLU(),
            nn.Conv2d(32, 64, 1),                           inorm(64), nn.PReLU(),
        )
        self.gap = nn.AdaptiveAvgPool2d((1,1))
        self.lstm_actor  = nn.LSTM(input_size=n_actions, hidden_size=64, num_layers=1, batch_first=True)
        self.lstm_critic = nn.LSTM(input_size=feat_dim,  hidden_size=64, num_layers=1, batch_first=True)
        self.actor_head  = nn.Sequential(nn.Linear(64+64,128), nn.PReLU(), nn.Linear(128, n_actions))
        self.critic_head = nn.Sequential(nn.Linear(64+64,128), nn.PReLU(), nn.Linear(128, 1))

        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_uniform_(m.weight, a=0.25, nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

    def forward(self, x, ahist_onehot, fhist_feat, hc_actor=None, hc_critic=None):
        x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        ahist_onehot = torch.nan_to_num(ahist_onehot, nan=0.0, posinf=0.0, neginf=0.0)
        fhist_feat   = torch.nan_to_num(fhist_feat,   nan=0.0, posinf=0.0, neginf=0.0)

        z = self.cnn(x)
        z = self.gap(z).squeeze(-1).squeeze(-1)

        out_a, hc_actor  = self.lstm_actor(ahist_onehot, hc_actor)
        out_c, hc_critic = self.lstm_critic(fhist_feat,  hc_critic)
        h_a = out_a[:, -1, :]
        h_c = out_c[:, -1, :]

        logits = self.actor_head(torch.cat([z, h_a], dim=1))
        value  = self.critic_head(torch.cat([z, h_c], dim=1)).squeeze(-1)

        logits = torch.nan_to_num(logits, nan=0.0, posinf=20.0, neginf=-20.0).clamp(-20, 20)
        value  = torch.nan_to_num(value,  nan=0.0, posinf=0.0,  neginf=0.0)
        return logits, value, hc_actor, hc_critic

class PPO:
    def __init__(self, model: ActorCritic, n_actions=8, clip=0.2, gamma=0.9, lam=0.95,
                 lr=1e-5, epochs=10, minibatch=8, entropy_coef=0.03, value_coef=0.5, max_grad_norm=1.0):
        self.model = model
        self.clip = clip
        self.gamma = gamma
        self.lam = lam
        self.epochs = epochs
        self.minibatch = minibatch
        self.opt = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.max_grad_norm = max_grad_norm
        self.lr = lr
        self.lr_lower_bound = 1e-6
        self.patience = 5
        self.patience_counter = 0
        self.best_val_score = -float('inf')

    def update_learning_rate(self, val_score):
        if val_score > self.best_val_score:
            self.best_val_score = val_score
            self.patience_counter = 0
        else:
            self.patience_counter += 1
        if self.patience_counter >= self.patience:
            self.lr = max(self.lr / 2.0, self.lr_lower_bound)
            for pg in self.opt.param_groups: pg['lr'] = self.lr
            self.entropy_coef = max(self.entropy_coef * 0.7, 0.005)
            self.patience_counter = 0
            print(f"Learning rate reduced to {self.lr} | entropy_coef -> {self.entropy_coef}")

    @staticmethod
    def compute_gae(rewards, values, dones, gamma, lam):
        T = len(rewards)
        adv = np.zeros(T, dtype=np.float32)
        gae = 0.0
        for t in reversed(range(T)):
            next_v = 0.0 if dones[t] else (values[t+1] if t+1 < len(values) else 0.0)
            delta = rewards[t] + gamma * next_v - values[t]
            gae = delta + gamma * lam * (0.0 if dones[t] else gae)
            adv[t] = gae
        ret = adv + values[:T]
        return adv, ret

    def update(self, buf):
        obs      = torch.tensor(np.stack(buf["obs"]),   dtype=torch.float32, device=DEVICE)
        ahist    = torch.tensor(np.stack(buf["ahist"]), dtype=torch.float32, device=DEVICE)
        fhist    = torch.tensor(np.stack(buf["fhist"]), dtype=torch.float32, device=DEVICE)
        act      = torch.tensor(np.array(buf["act"]),   dtype=torch.long,    device=DEVICE)
        old_logp = torch.tensor(np.array(buf["logp"]),  dtype=torch.float32, device=DEVICE)
        adv      = torch.tensor(np.array(buf["adv"]),   dtype=torch.float32, device=DEVICE)
        ret      = torch.tensor(np.array(buf["ret"]),   dtype=torch.float32, device=DEVICE)

        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        N = obs.size(0); idx = np.arange(N)
        for _ in range(self.epochs):
            np.random.shuffle(idx)
            for s in range(0, N, self.minibatch):
                mb = idx[s:s+self.minibatch]
                x = torch.nan_to_num(obs[mb])
                A = torch.nan_to_num(ahist[mb])
                Fh= torch.nan_to_num(fhist[mb])

                logits, value, _, _ = self.model(x, A, Fh, None, None)
                if not torch.isfinite(logits).all() or not torch.isfinite(value).all(): 
                    continue

                dist = Categorical(logits=logits)
                logp = dist.log_prob(act[mb])

                ratio = torch.exp(logp - old_logp[mb])
                surr1 = ratio * adv[mb]
                surr2 = torch.clamp(ratio, 1.0 - self.clip, 1.0 + self.clip) * adv[mb]
                policy_loss = -torch.min(surr1, surr2).mean()

                value_loss = F.mse_loss(value, ret[mb])
                entropy = dist.entropy().mean()

                loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy
                self.opt.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                self.opt.step()


def extract_critic_features(obs_crop: np.ndarray, last_reward: float):
    img = obs_crop[0]
    center = img[CROP//2, CROP//2]
    meanv = float(img.mean())
    feat = np.array([
        float(last_reward),
        float(math.log1p(abs(last_reward))),
        float(center),
        float(meanv)
    ], dtype=np.float32)
    return feat

class LiveVisualizer:
    def __init__(self, env: CurveEnv, draw_every:int=1, manual:bool=False, title:str="Training"):
        import matplotlib.pyplot as plt
        self.plt = plt
        self.draw_every = max(1, int(draw_every))
        self.manual = manual
        self.fig = plt.figure(figsize=(11, 6))
        gs = self.fig.add_gridspec(2, 3, width_ratios=[1.2, 0.9, 1.1], height_ratios=[1,1])
        # Left: image + GT + path
        self.ax_img = self.fig.add_subplot(gs[:,0])
        self.ax_img.set_title("Env (img + GT + path)")
        self.ax_img.set_xticks([]); self.ax_img.set_yticks([])
        self.im = self.ax_img.imshow(env.ep.img, cmap="gray", vmin=0, vmax=1, animated=False)
        # overlay GT centerline (green)
        gt = env.ep.gt_poly
        self.gt_line, = self.ax_img.plot(gt[:,1], gt[:,0], color='lime', lw=1.5, alpha=0.9, label='GT')
        # overlay dilated GT (light green dots)
        yy, xx = np.where(env.ep.dilated_gt > 0.5)
        self.gt_dots = self.ax_img.scatter(xx, yy, s=2, c='lightgreen', alpha=0.3)
        # START marker
        self.start_dot = self.ax_img.scatter([env.ep.start[1]], [env.ep.start[0]], s=50, c='red', marker='x', label='START')
        # path line & agent dot
        self.path_line, = self.ax_img.plot([], [], color='cyan', lw=2, alpha=0.9, label='path')
        self.agent_dot = self.ax_img.scatter([], [], s=35, c='red', label='agent')
        self.ax_img.legend(loc='lower right', fontsize=8)

        self.ax_logits = self.fig.add_subplot(gs[0,1])
        self.ax_logits.set_title("Action logits")
        self.ax_logits.set_ylim(-5, 5)
        self.ax_logits.set_xticks(range(len(ACTIONS_8_NAMES)))
        self.ax_logits.set_xticklabels(ACTIONS_8_NAMES, rotation=0)
        self.bar_container = self.ax_logits.bar(range(len(ACTIONS_8_NAMES)), [0]*len(ACTIONS_8_NAMES))

        self.ax_text = self.fig.add_subplot(gs[1,1])
        self.ax_text.axis('off')
        self.txt = self.ax_text.text(0.02, 0.65, "", fontsize=11, family='monospace')

        self.ax_reward = self.fig.add_subplot(gs[0,2])
        self.ax_reward.set_title("Per-step reward (last 200)")
        self.ax_reward.set_ylim(-5, 110)  # allow success bonus to show
        self.ax_reward.set_xlim(0, 200)
        self.rew_x = deque(maxlen=200)
        self.rew_y = deque(maxlen=200)
        self.rew_plot, = self.ax_reward.plot([], [], lw=1.5)

        self.ax_ccs = self.fig.add_subplot(gs[1,2])
        self.ax_ccs.set_title("CCS (last 200)")
        self.ax_ccs.set_ylim(-1.0, 1.0)
        self.ax_ccs.set_xlim(0, 200)
        self.ccs_x = deque(maxlen=200)
        self.ccs_y = deque(maxlen=200)
        self.ccs_plot, = self.ax_ccs.plot([], [], lw=1.5)

        plt.tight_layout()
        plt.pause(0.001)

        self._step_counter = 0
        self.closed = False

        self._proceed = not manual  
        self._last_key = None

        self.cid_close = self.fig.canvas.mpl_connect('close_event', self._on_close)
        self.cid_key   = self.fig.canvas.mpl_connect('key_press_event', self._on_key)
        self.cid_btn   = self.fig.canvas.mpl_connect('button_press_event', self._on_click)

    def _on_close(self, evt):
        self.closed = True
        self._proceed = True  # let loop exit cleanly

    def _on_key(self, event):
        self._last_key = event.key
        if event.key in (' ', 'space', 'enter'):
            self._proceed = True
        elif event.key and event.key.lower() == 'q':
            self.closed = True
            self._proceed = True

    def _on_click(self, event):
        self._proceed = True

    def wait_if_manual(self):
        if not self.manual:
            return
        import time
        self._proceed = False
        while not self._proceed and not self.closed:
            self.plt.pause(0.01)  # keep UI responsive
            time.sleep(0.01)

    def update(self, env: CurveEnv, logits: Optional[np.ndarray], action_idx: int, reward: float, ep:int, step:int, ep_ret:float, ccs:float):
        path = np.array(env.path_points, dtype=np.int32)
        if len(path) > 1:
            self.path_line.set_data(path[:,1], path[:,0])
        else:
            self.path_line.set_data([], [])
        self.agent_dot.set_offsets(np.array([[env.agent[1], env.agent[0]]], dtype=np.float32))
        self.start_dot.set_offsets(np.array([[env.ep.start[1], env.ep.start[0]]], dtype=np.float32))

        if logits is not None:
            vals = np.array(logits, dtype=np.float32).tolist()
            for bar, v in zip(self.bar_container, vals):
                bar.set_height(float(v))
            self.ax_logits.set_ylim(min(-5, float(np.min(vals))-0.5), max(5, float(np.max(vals))+0.5))

        act_name = ACTIONS_8_NAMES[action_idx] if 0 <= action_idx < len(ACTIONS_8_NAMES) else "?"
        control_hint = " [SPACE/click=next, Q=quit]" if self.manual else ""
        self.txt.set_text(f"Action: {act_name:>2}   Reward: {reward:8.4f}\nEpisode: {ep}  Step: {step}  Return: {ep_ret:9.3f}\nCCS: {ccs:8.4f}{control_hint}")

        self.rew_x.append(len(self.rew_x))
        self.rew_y.append(reward)
        self.rew_plot.set_data(self.rew_x, self.rew_y)
        self.ax_reward.set_xlim(max(0, len(self.rew_x)-200), len(self.rew_x))

        self.ccs_x.append(len(self.ccs_x))
        self.ccs_y.append(ccs)
        self.ccs_plot.set_data(self.ccs_x, self.ccs_y)
        self.ax_ccs.set_xlim(max(0, len(self.ccs_x)-200), len(self.ccs_x))

        self.ax_img.set_title(f"Env (img + GT + path) — Ep {ep}, Step {step}, Ret {ep_ret:.2f}, CCS {ccs:.3f}")

        # Throttle drawing
        self._step_counter += 1
        if (self._step_counter % self.draw_every) == 0:
            self.plt.pause(0.001)

def train(args):
    set_seeds(123)
    env = CurveEnv(h=128, w=128, branches=args.branches, max_steps=400,
                   d0=2.0, dilate_radius=DILATE_RADIUS, start_mode=args.start_mode)
    K = 8; nA = len(ACTIONS_8); Fdim = 4
    model = ActorCritic(n_actions=nA, K=K, feat_dim=Fdim, in_ch=5).to(DEVICE)
    ppo = PPO(model, lr=1e-5, gamma=0.9, lam=0.95, clip=0.2, epochs=10, minibatch=8, entropy_coef=0.05)

    rollout_buf = {k: [] for k in ["obs","ahist","fhist","act","logp","adv","ret"]}
    ep_returns, ep_ccs_scores = [], []

    viz = LiveVisualizer(env, draw_every=args.draw_every, manual=args.manual) if args.live else None

    for ep in range(1, args.episodes+1):
        # curriculum
        if ep < 2000:
            env.dilate_radius = 5; env.off_track_thresh = 3.0
        elif ep < 4000:
            env.dilate_radius = 4; env.off_track_thresh = 2.4
        else:
            env.dilate_radius = 3; env.off_track_thresh = 1.8

        obs = env.reset()
        if viz:
            viz.im.set_data(env.ep.img)
            gt = env.ep.gt_poly
            viz.gt_line.set_data(gt[:,1], gt[:,0])
            yy, xx = np.where(env.ep.dilated_gt > 0.5)
            viz.gt_dots.set_offsets(np.c_[xx, yy])
            viz.start_dot.set_offsets(np.array([[env.ep.start[1], env.ep.start[0]]], dtype=np.float32))
            viz.plt.pause(0.001)

        done = False
        ahist, fhist = [], []
        traj = {"obs":[], "ahist":[], "fhist":[], "act":[], "logp":[], "val":[], "rew":[], "done":[]}
        ep_ret = 0.0

        # seed critic features
        feat = extract_critic_features(obs, last_reward=0.0)
        fhist.append(feat)

        last_value_for_bootstrap = 0.0
        last_done = False
        step = 0

        while not done:
            step += 1
            x = torch.tensor(obs[None], dtype=torch.float32, device=DEVICE)
            A = fixed_window_history(ahist, K, nA)[None, ...]
            Fh= fixed_window_features(fhist, K, Fdim)[None, ...]
            A_t = torch.tensor(A,  dtype=torch.float32, device=DEVICE)
            F_t = torch.tensor(Fh, dtype=torch.float32, device=DEVICE)

            logits, value, _, _ = model(x, A_t, F_t, None, None)
            logits = torch.nan_to_num(logits, nan=0.0, posinf=20.0, neginf=-20.0).clamp(-20, 20)
            dist = Categorical(logits=logits)

            action = int(dist.sample().item())
            logp = float(dist.log_prob(torch.tensor(action, device=DEVICE)).item())
            val = float(value.item())

            obs2, r, done, info = env.step(action)

            traj["obs"].append(obs)
            traj["ahist"].append(A[0])
            traj["fhist"].append(Fh[0])
            traj["act"].append(action)
            traj["logp"].append(logp)
            traj["val"].append(val)
            traj["rew"].append(r)
            traj["done"].append(done)

            a1h = np.zeros(nA, dtype=np.float32); a1h[action] = 1.0
            ahist.append(a1h)

            feat = extract_critic_features(obs2, last_reward=r)
            fhist.append(feat)

            obs = obs2
            ep_ret += r
            last_value_for_bootstrap = val
            last_done = done

            if viz:
                with torch.no_grad():
                    viz_logits = logits[0].detach().cpu().numpy()
                viz.update(env, viz_logits, action, r, ep, step, ep_ret, info.get("ccs", 0.0))
                # manual step waits *after* showing what just happened
                viz.wait_if_manual()
                if viz.closed:
                    done = True  # stop episode

        ep_ccs_scores.append(info.get('ccs', 0.0))

        values = np.array(traj["val"] + [last_value_for_bootstrap if last_done and info.get("timeout", False) else 0.0],
                          dtype=np.float32)
        adv, ret = PPO.compute_gae(np.array(traj["rew"], dtype=np.float32),
                                   values, traj["done"], 0.9, 0.95)

        ep_buf = {
            "obs":   np.array(traj["obs"], dtype=np.float32),
            "ahist": np.array(traj["ahist"], dtype=np.float32),
            "fhist": np.array(traj["fhist"], dtype=np.float32),
            "act":   np.array(traj["act"], dtype=np.int64),
            "logp":  np.array(traj["logp"], dtype=np.float32),
            "adv":   adv,
            "ret":   ret,
        }
        for k in rollout_buf:
            rollout_buf[k].append(ep_buf[k])

        if len(rollout_buf["obs"]) >= ppo.minibatch:
            cat_buf = {k: np.concatenate(rollout_buf[k], axis=0) for k in rollout_buf}
            ppo.update(cat_buf)
            rollout_buf = {k: [] for k in rollout_buf}
            ep_returns.append(ep_ret)

        if ep % 100 == 0:
            avg_ret = float(np.mean(ep_returns[-100:])) if len(ep_returns) else ep_ret
            avg_ccs = float(np.mean(ep_ccs_scores[-100:])) if len(ep_ccs_scores) else 0.0
            print(f"Episode {ep:6d} | return(avg100)={avg_ret:7.3f} | CCS(avg100)={avg_ccs:7.3f}")
            ppo.update_learning_rate(avg_ccs)

        if args.save and (args.save_every > 0) and (ep % args.save_every == 0):
            torch.save(model.state_dict(), args.save)
            print(f"Saved to {args.save}")

    if args.save:
        torch.save(model.state_dict(), args.save)
        print(f"Saved final weights to {args.save}")

def view(args):
    set_seeds(123)
    env = CurveEnv(h=128, w=128, branches=args.branches, max_steps=400,
                   start_mode=args.start_mode)
    K = 8; nA = len(ACTIONS_8); Fdim = 4
    model = ActorCritic(n_actions=nA, K=K, feat_dim=Fdim, in_ch=5).to(DEVICE)
    if args.weights:
        state = torch.load(args.weights, map_location=DEVICE)
        model.load_state_dict(state); print(f"Loaded weights: {args.weights}")
    model.eval()

    viz = LiveVisualizer(env, draw_every=args.draw_every, manual=args.manual, title="View") if args.live else None

    obs = env.reset()
    if viz:
        viz.im.set_data(env.ep.img)
        gt = env.ep.gt_poly
        viz.gt_line.set_data(gt[:,1], gt[:,0])
        yy, xx = np.where(env.ep.dilated_gt > 0.5)
        viz.gt_dots.set_offsets(np.c_[xx, yy])
        viz.start_dot.set_offsets(np.array([[env.ep.start[1], env.ep.start[0]]], dtype=np.float32))
        viz.plt.pause(0.001)

    done = False
    ahist, fhist = [], []
    steps = 0
    feat = extract_critic_features(obs, 0.0)
    fhist.append(feat)

    with torch.no_grad():
        while not done:
            steps += 1
            x  = torch.tensor(obs[None], dtype=torch.float32, device=DEVICE)
            A  = fixed_window_history(ahist, K, nA)[None, ...]
            Fh = fixed_window_features(fhist, K, Fdim)[None, ...]
            A_t = torch.tensor(A,  dtype=torch.float32, device=DEVICE)
            F_t = torch.tensor(Fh, dtype=torch.float32, device=DEVICE)
            logits, value, _, _ = model(x, A_t, F_t, None, None)
            action = int(torch.argmax(logits, dim=1).item())
            obs, r, done, info = env.step(action)
            a1h = np.zeros(nA, dtype=np.float32); a1h[action] = 1.0
            ahist.append(a1h)
            feat = extract_critic_features(obs, r)
            fhist.append(feat)

            if viz:
                viz.update(env, logits[0].detach().cpu().numpy(), action, r, ep=0, step=steps, ep_ret=0.0, ccs=info.get("ccs", 0.0))
                viz.wait_if_manual()
                if viz.closed: break

    print(f"[VIEW] steps={steps}  L_end(local)={info['L_local']:.3f}  idx_end={info['idx']}  CCS={info['ccs']:.3f}")

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--train", default=True, action="store_true")
    p.add_argument("--view",  action="store_true")
    p.add_argument("--episodes", type=int, default=200)
    p.add_argument("--entropy_coef", type=float, default=0.03)
    p.add_argument("--save", type=str, default="ckpt_curveppo.pth")
    p.add_argument("--save_every", type=int, default=0)
    p.add_argument("--weights", type=str, default="")
    p.add_argument("--branches", action="store_true")
    p.add_argument("--live", default=True, action="store_true", help="Enable live Matplotlib visualization")
    p.add_argument("--manual", default=True, action="store_true", help="Step manually (SPACE/click to advance, Q to quit)")
    p.add_argument("--draw_every", type=int, default=1, help="Render every N steps (throttle drawing)")
    p.add_argument("--start_mode", choices=["begin","left_end","random"], default="begin",
                   help="Episode start policy: begin=gt_poly[0], left_end=smaller x endpoint, random=rotate poly")
    args = p.parse_args()

    if not args.train and not args.view:
        args.train = True

    if args.train: train(args)
    if args.view:  view(args)

if __name__ == "__main__":
    main()

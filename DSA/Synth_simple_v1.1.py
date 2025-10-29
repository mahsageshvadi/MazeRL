# curvewalk_ppo.py
# 2D Directed Vessel Tracking on Synthetic Curves (PPO/A2C)
# - Uses Curve_Generator.py (CurveMaker) for data
# - State: [crop(p_t), crop(p_{t-1}), crop(p_{t-2}), crop(path_t)]
# - Reward: overlap bonus + delta of bidirectional Chamfer curve distance (log scaled)
#
# Inspired by Su et al. (MEDIA 2023) Section 2.1 (state/action/reward/architecture).
# Mirrors the style of your maze visualizer structure (simple animate loop, compact env).
#
# Files expected next to this script:
#   - Curve_Generator.py  (your curve synth)   [used directly]
#
# Minimal deps: numpy, torch, tkinter

import argparse, math, random, time
import numpy as np
import tkinter as tk
from dataclasses import dataclass
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical

# =========================
# Synthetic curves (your code)
# =========================
# We import your CurveMaker directly.
from Curve_Generator import CurveMaker  # :contentReference[oaicite:2]{index=2}


# =========================
# Utilities
# =========================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
ACTIONS_8 = [(-1, 0), (1, 0), (0,-1), (0, 1), (-1,-1), (-1,1), (1,-1), (1,1)]
STEP_ALPHA = 2
CROP = 33  # odd size, 16 px radius around center

def set_seeds(seed=123):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if DEVICE == "cuda": torch.cuda.manual_seed_all(seed)

def clamp(v, lo, hi):

    return max(lo, min(v, hi))

def crop32(img: np.ndarray, cy: int, cx: int, size=33):

    h, w = img.shape
    r = size // 2
    y0, y1 = cy - r, cy + r + 1   # [y0, y1)
    x0, x1 = cx - r, cx + r + 1   # [x0, x1)

    # ✅ clamp the VALUES (y0,y1,...) not 0
    sy0, sy1 = clamp(y0, 0, h), clamp(y1, 0, h)
    sx0, sx1 = clamp(x0, 0, w), clamp(x1, 0, w)

    out = np.zeros((size, size), dtype=img.dtype)

    # destination spans exactly the same width/height we pull from source
    oy0 = sy0 - y0
    ox0 = sx0 - x0
    sh  = sy1 - sy0
    sw  = sx1 - sx0
    oy1 = oy0 + sh
    ox1 = ox0 + sw

    if sh > 0 and sw > 0:
        out[oy0:oy1, ox0:ox1] = img[sy0:sy1, sx0:sx1]

    return out

def chamfer_distance_bidirectional(P: np.ndarray, Q: np.ndarray) -> float:
    """Bidirectional Chamfer distance between two polylines (list of (y,x))."""
    if len(P) == 0 or len(Q) == 0:
        return 1e3
    # Downsample for speed
    P_ = P[::max(1, len(P)//200 + 1)]
    Q_ = Q[::max(1, len(Q)//200 + 1)]

    # P->Q
    d1 = []
    for p in P_:
        dy = Q_[:,0] - p[0]; dx = Q_[:,1] - p[1]
        d1.append(np.min(dy*dy + dx*dx))
    # Q->P
    d2 = []
    for q in Q_:
        dy = P_[:,0] - q[0]; dx = P_[:,1] - q[1]
        d2.append(np.min(dy*dy + dx*dx))

    # sqrt of mean squared min-dists
    return float(math.sqrt((np.mean(d1) + np.mean(d2)) * 0.5))


# =========================
# Environment
# =========================
@dataclass
class CurveEpisode:
    img: np.ndarray      # (H,W) float32 in [0,1]
    mask: np.ndarray     # (H,W) uint8 0/1 centerline mask
    gt_poly: np.ndarray  # (N,2) list of (y,x) along main curve
    start: Tuple[int,int]
    init_dir: Tuple[int,int]

class CurveEnv:
    """
    Directed curve tracking in 2D. Step in 8-neighborhood with step size alpha.
    Observation = 4x (33x33): crop around p_t, p_{t-1}, p_{t-2}, and binary agent path crop.
    """
    def __init__(self, h=128, w=128, branches=False, max_steps=400):
        self.h, self.w = h, w
        self.max_steps = max_steps
        self.cm = CurveMaker(h=h, w=w, thickness=1.5, seed=None)  # your generator
        self.branches = branches

        self.reset()

    def reset(self):
        img, mask, pts_all = self.cm.sample_curve(branches=self.branches)
        # We’ll track the MAIN curve (first polyline)
        gt_poly = pts_all[0].astype(np.float32)

        # Start near its first point, set initial direction along the first segment
        p0 = gt_poly[0].astype(int)
        p1 = gt_poly[min(5, len(gt_poly)-1)].astype(int)
        init_vec = np.array([p1[0]-p0[0], p1[1]-p0[1]], dtype=np.int32)
        init_vec = np.sign(init_vec); init_vec[init_vec==0] = 1  # nudge

        self.ep = CurveEpisode(img=img, mask=mask, gt_poly=gt_poly, start=(int(p0[0]), int(p0[1])),
                               init_dir=(int(init_vec[0]), int(init_vec[1])))

        self.agent = (int(p0[0]), int(p0[1]))
        self.prev = [self.agent, self.agent]  # p_{t-1}, p_{t-2}
        self.steps = 0

        self.path_mask = np.zeros_like(mask, dtype=np.float32)
        self.path_points: List[Tuple[int,int]] = [self.agent]
        self.path_mask[self.agent] = 1.0

        # distances for reward
        self.L_prev = chamfer_distance_bidirectional(np.array(self.path_points, dtype=np.float32),
                                                     self.ep.gt_poly)

        return self.obs()

    def obs(self):
        p_t, p_1, p_2 = self.agent, self.prev[0], self.prev[1]
        ch0 = crop32(self.ep.img,  p_t[0], p_t[1])   # current image crop
        ch1 = crop32(self.ep.img,  p_1[0], p_1[1])   # prev image crop
        ch2 = crop32(self.ep.img,  p_2[0], p_2[1])   # prev2 image crop
        ch3 = crop32(self.path_mask, p_t[0], p_t[1]) # path crop
        obs = np.stack([ch0, ch1, ch2, ch3], axis=0) # (4,33,33)
        return obs.astype(np.float32)

    def step(self, a_idx: int):
        self.steps += 1
        dy, dx = ACTIONS_8[a_idx]
        ny = clamp(self.agent[0] + dy*STEP_ALPHA, 0, self.h-1)
        nx = clamp(self.agent[1] + dx*STEP_ALPHA, 0, self.w-1)
        new_pos = (ny, nx)

        # Update path/order
        self.prev = [self.agent, self.prev[0]]
        self.agent = new_pos
        self.path_points.append(self.agent)
        self.path_mask[self.agent] = 1.0

        # Reward pieces
        overlap = 1.0 if self.ep.mask[self.agent] > 0 else 0.0

        L_cur = chamfer_distance_bidirectional(np.array(self.path_points, dtype=np.float32),
                                               self.ep.gt_poly)
        delta = L_cur - self.L_prev
        self.L_prev = L_cur

        # Direction gating at very beginning: discourage moving strongly opposite to init
        if self.steps <= 3:
            v0 = np.array(self.ep.init_dir, dtype=np.float32)
            vt = np.array([dy, dx], dtype=np.float32)
            if v0.dot(vt) < 0:
                delta += 0.25  # mild penalty

        # Log-scaled shaping (Eq. 3 style)
        eps = 1e-6
        if delta < 0:
            r = overlap - math.log(eps + abs(delta))
        else:
            r = overlap + math.log(eps + abs(delta))

        done = (self.steps >= self.max_steps)
        # Optional success heuristic: near end of curve → small L_cur
        if L_cur < 0.75:
            done = True
            r += 2.0  # terminal bonus for good alignment

        return self.obs(), float(r), done, {"overlap": overlap, "L": L_cur}


# =========================
# PPO Agent (A2C with GAE)
# =========================
class ActorCritic(nn.Module):
    def __init__(self, n_actions=8):
        super().__init__()
        # Dilated CNN backbone on 4x33x33
        self.cnn = nn.Sequential(
            nn.Conv2d(4, 32, 3, padding=1, dilation=1), nn.InstanceNorm2d(32), nn.PReLU(),
            nn.Conv2d(32,32, 3, padding=2, dilation=2), nn.InstanceNorm2d(32), nn.PReLU(),
            nn.Conv2d(32,32, 3, padding=3, dilation=3), nn.InstanceNorm2d(32), nn.PReLU(),
            nn.Conv2d(32,64, 1),                         nn.InstanceNorm2d(64), nn.PReLU(),
        )
        # Global Average Pooling
        self.gap = nn.AdaptiveAvgPool2d((1,1))

        # LSTM over action history scalars (one-hot of last action)
        self.lstm = nn.LSTM(input_size=n_actions, hidden_size=64, num_layers=1, batch_first=True)

        # Heads
        self.actor = nn.Sequential(nn.Linear(64+64,128), nn.PReLU(), nn.Linear(128, n_actions))
        self.critic = nn.Sequential(nn.Linear(64+64,128), nn.PReLU(), nn.Linear(128, 1))

    def forward(self, x, ahist_onehot, hc=None):
        # x: (B,4,33,33), ahist_onehot: (B,T,8)
        z = self.cnn(x)               # (B,64,33,33)
        z = self.gap(z).squeeze(-1).squeeze(-1)  # (B,64)

        if ahist_onehot is None:
            # empty history → zeros + a dummy LSTM pass
            ahist_onehot = torch.zeros(x.size(0), 1, self.actor[-1].out_features, device=x.device)
        out, hc = self.lstm(ahist_onehot, hc)  # (B,T,64)
        h_last = out[:, -1, :]                 # (B,64)

        h = torch.cat([z, h_last], dim=1)      # (B,128)
        logits = self.actor(h)                 # (B,8)
        value  = self.critic(h)                # (B,1)
        return logits, value.squeeze(-1), hc


class PPO:
    def __init__(self, model: ActorCritic, n_actions=8, clip=0.2, gamma=0.95, lam=0.95, lr=3e-4, epochs=4, minibatch=64):
        self.model = model
        self.clip = clip
        self.gamma = gamma
        self.lam = lam
        self.epochs = epochs
        self.minibatch = minibatch
        self.opt = torch.optim.Adam(self.model.parameters(), lr=lr)

    def compute_gae(self, rewards, values, dones):
        # rewards/values/dones: (T,)
        T = len(rewards)
        adv = np.zeros(T, dtype=np.float32)
        gae = 0.0
        for t in reversed(range(T)):
            delta = rewards[t] + (0 if dones[t] else self.gamma * (values[t+1] if t+1 < T else 0)) - values[t]
            gae = delta + (0 if dones[t] else self.gamma * self.lam * gae)
            adv[t] = gae
        ret = adv + values[:-1]  # targets for V
        return adv, ret

    def update(self, buf):
        # buffer is a dict of lists of tensors/arrays concatenable along dim 0
        obs      = torch.tensor(np.stack(buf["obs"]), dtype=torch.float32, device=DEVICE)
        ahist    = torch.tensor(np.stack(buf["ahist"]), dtype=torch.float32, device=DEVICE)  # (N,T,8)
        act      = torch.tensor(np.array(buf["act"]),  dtype=torch.long,    device=DEVICE)
        old_logp = torch.tensor(np.array(buf["logp"]),dtype=torch.float32, device=DEVICE)
        adv      = torch.tensor(np.array(buf["adv"]), dtype=torch.float32, device=DEVICE)
        ret      = torch.tensor(np.array(buf["ret"]), dtype=torch.float32, device=DEVICE)

        # normalize adv
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        N = obs.size(0)
        idx = np.arange(N)

        for _ in range(self.epochs):
            np.random.shuffle(idx)
            for start in range(0, N, self.minibatch):
                mb = idx[start:start+self.minibatch]
                logits, value, _ = self.model(obs[mb], ahist[mb], None)
                dist = Categorical(logits=logits)
                logp = dist.log_prob(act[mb])

                ratio = torch.exp(logp - old_logp[mb])
                surr1 = ratio * adv[mb]
                surr2 = torch.clamp(ratio, 1.0 - self.clip, 1.0 + self.clip) * adv[mb]
                policy_loss = -torch.min(surr1, surr2).mean()

                value_loss = F.mse_loss(value, ret[mb])

                entropy = dist.entropy().mean()
                loss = policy_loss + 0.5 * value_loss - 0.01 * entropy

                self.opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.opt.step()


# =========================
# Training Loop
# =========================
def train(args):
    set_seeds(123)
    env = CurveEnv(h=128, w=128, branches=args.branches, max_steps=400)
    model = ActorCritic(n_actions=len(ACTIONS_8)).to(DEVICE)
    ppo = PPO(model, lr=args.lr, gamma=args.gamma, lam=args.lam, clip=args.clip, epochs=args.epochs, minibatch=args.minibatch)

    ep_returns = []
    ahist_len = 8

    for ep in range(1, args.episodes+1):
        obs = env.reset()
        done = False

        # rolling history: one-hot last K actions (start with "init" token as no-op)
        ahist = [np.zeros(len(ACTIONS_8), dtype=np.float32)]
        traj = {"obs":[], "ahist":[], "act":[], "logp":[], "val":[], "rew":[], "done":[]}

        ep_ret = 0.0
        while not done:
            x = torch.tensor(obs[None], dtype=torch.float32, device=DEVICE)
            A = np.stack(ahist, axis=0)[None, ...]  # (1,T,8)
            A = torch.tensor(A, dtype=torch.float32, device=DEVICE)

            logits, value, _ = model(x, A, None)
            dist = Categorical(logits=logits)
            action = int(dist.sample().item())
            logp = float(dist.log_prob(torch.tensor(action, device=DEVICE)).item())
            val = float(value.item())

            obs2, r, done, info = env.step(action)

            traj["obs"].append(obs)
            traj["ahist"].append(np.stack(ahist, axis=0))
            traj["act"].append(action)
            traj["logp"].append(logp)
            traj["val"].append(val)
            traj["rew"].append(r)
            traj["done"].append(done)

            # update history
            a1h = np.zeros(len(ACTIONS_8), dtype=np.float32); a1h[action] = 1.0
            ahist.append(a1h)
            if len(ahist) > ahist_len:
                ahist.pop(0)

            obs = obs2
            ep_ret += r

        # bootstrap value for last state as 0 (episode ended)
        values = np.array(traj["val"] + [0.0], dtype=np.float32)
        adv, ret = ppo.compute_gae(np.array(traj["rew"], dtype=np.float32), values, traj["done"])

        buf = {
            "obs":   np.array(traj["obs"], dtype=np.float32),
            "ahist": np.array(traj["ahist"], dtype=np.float32),
            "act":   traj["act"],
            "logp":  traj["logp"],
            "adv":   adv,
            "ret":   ret,
        }
        ppo.update(buf)
        ep_returns.append(ep_ret)

        if ep % 100 == 0:
            avg = np.mean(ep_returns[-100:])
            print(f"Episode {ep:6d} | return(avg100)={avg:7.3f}")
        if args.save and ep % args.save_every == 0:
            torch.save(model.state_dict(), args.save)
            print(f"Saved to {args.save}")

    if args.save:
        torch.save(model.state_dict(), args.save)
        print(f"Saved final weights to {args.save}")


# =========================
# Viewer (greedy animate)
# =========================
class CurveViewer:
    def __init__(self, env: CurveEnv, cell=4):
        self.env = env
        self.cell = cell
        self.root = tk.Tk()
        self.root.title("CurveWalk-PPO — Step-by-Step")
        H, W = env.h, env.w
        self.canvas = tk.Canvas(self.root, width=W*cell, height=H*cell)
        self.canvas.pack()

    def draw(self):
        self.canvas.delete("all")
        img = (self.env.ep.img * 255).astype(np.uint8)
        # background
        for y in range(self.env.h):
            for x in range(self.env.w):
                v = img[y,x]
                color = f"#{v:02x}{v:02x}{v:02x}"
                self.canvas.create_rectangle(x*self.cell, y*self.cell, (x+1)*self.cell, (y+1)*self.cell, outline="", fill=color)
        # GT mask (green overlay)
        ys, xs = np.where(self.env.ep.mask > 0)
        for y, x in zip(ys, xs):
            self.canvas.create_rectangle(x*self.cell, y*self.cell, (x+1)*self.cell, (y+1)*self.cell, outline="", fill="#88ff88")
        # Agent path (cyan)
        for (y, x) in self.env.path_points:
            self.canvas.create_rectangle(x*self.cell, y*self.cell, (x+1)*self.cell, (y+1)*self.cell, outline="", fill="#6fd6ff")
        # Agent (red)
        ay, ax = self.env.agent
        self.canvas.create_rectangle(ax*self.cell, ay*self.cell, (ax+1)*self.cell, (ay+1)*self.cell, outline="", fill="#ff4444")
        self.root.update()

    def animate(self, policy: ActorCritic, delay_ms=100):
        obs = self.env.reset()
        done = False
        self.draw()

        ahist = [np.zeros(len(ACTIONS_8), dtype=np.float32)]
        with torch.no_grad():
            while not done:
                x = torch.tensor(obs[None], dtype=torch.float32, device=DEVICE)
                A = np.stack(ahist, axis=0)[None, ...]
                A = torch.tensor(A, dtype=torch.float32, device=DEVICE)

                logits, value, _ = policy(x, A, None)
                action = int(torch.argmax(logits, dim=1).item())  # greedy
                obs, r, done, info = self.env.step(action)
                a1h = np.zeros(len(ACTIONS_8), dtype=np.float32); a1h[action] = 1.0
                ahist.append(a1h)
                if len(ahist) > 8: ahist.pop(0)

                self.draw()
                self.root.after(delay_ms)
                self.root.update()

        print(f"Finished. Steps: {len(self.env.path_points)}; L_end={info['L']:.3f}")

def view(args):
    env = CurveEnv(h=128, w=128, branches=args.branches, max_steps=400)
    model = ActorCritic(n_actions=len(ACTIONS_8)).to(DEVICE)
    if args.weights:
        state = torch.load(args.weights, map_location=DEVICE)
        model.load_state_dict(state)
        print(f"Loaded weights: {args.weights}")
    model.eval()
    viewer = CurveViewer(env, cell=args.cell_px)
    viewer.animate(model, delay_ms=args.delay_ms)
    viewer.root.mainloop()


# =========================
# CLI
# =========================
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--train", action="store_true")
    p.add_argument("--view",  action="store_true")
    p.add_argument("--episodes", type=int, default=5000)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--gamma", type=float, default=0.95)
    p.add_argument("--lam", type=float, default=0.95)
    p.add_argument("--clip", type=float, default=0.2)
    p.add_argument("--epochs", type=int, default=4)
    p.add_argument("--minibatch", type=int, default=64)
    p.add_argument("--save", type=str, default="ckpt_curveppo.pth")
    p.add_argument("--save_every", type=int, default=2000)
    p.add_argument("--weights", type=str, default="")
    p.add_argument("--branches", action="store_true", help="include branchy curves")
    p.add_argument("--delay_ms", type=int, default=120)
    p.add_argument("--cell_px", type=int, default=4)
    args = p.parse_args()

    if args.train:
        train(args)
    if args.view:
        view(args)

if __name__ == "__main__":
    main()

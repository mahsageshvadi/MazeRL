#!/usr/bin/env python3
# Synth_simple_v1.3.py - Paper-aligned reward + DTW distance + critic features
# Based on: "Deep reinforcement learning for cerebral anterior vessel tree extraction"

import argparse, math, random
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

# ---------- your curve generator ----------
from Curve_Generator import CurveMaker

# ---------- globals / utils ----------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
ACTIONS_8 = [(-1, 0), (1, 0), (0,-1), (0, 1), (-1,-1), (-1,1), (1,-1), (1,1)]
STEP_ALPHA = 2              # paper uses step scaling
CROP = 33                   # local window size
DILATE_RADIUS = 4           # ~ 1.8mm -> ~4 voxels (paper)
LAMBDA_B     = 2.0    # was 1.0
LAMBDA_DELTA = 0.7    # was 1.0
EPS_LOG      = 1e-2   # was 1e-3
R_LOG_CLAMP  = 3.0    # was 6.0       # weight for log term

def set_seeds(seed=123):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if DEVICE == "cuda": torch.cuda.manual_seed_all(seed)

def clamp(v, lo, hi): return max(lo, min(v, hi))

def crop32(img: np.ndarray, cy: int, cx: int, size=CROP):
    """Zero-padded square crop centered at (cy,cx). Always returns (size,size)."""
    h, w = img.shape
    r = size // 2
    y0, y1 = cy - r, cy + r + 1  # [y0,y1)
    x0, x1 = cx - r, cx + r + 1  # [x0,x1)
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
    """(K,n_actions) left-padded with zeros."""
    out = np.zeros((K, n_actions), dtype=np.float32)
    if len(ahist_list) == 0: return out
    tail = ahist_list[-K:]
    out[-len(tail):] = np.stack(tail, axis=0)
    return out

def fixed_window_features(fhist_list, K, Fdim):
    """(K,Fdim) left-padded with zeros."""
    out = np.zeros((K, Fdim), dtype=np.float32)
    if len(fhist_list) == 0: return out
    tail = fhist_list[-K:]
    out[-len(tail):] = np.stack(tail, axis=0)
    return out

def nearest_gt_index(pt, poly):
    """Return (index, euclidean_distance) of the closest GT poly point to pt=(y,x)."""
    dif = poly - np.array(pt, dtype=np.float32)
    d2 = np.sum(dif * dif, axis=1)
    i = int(np.argmin(d2))
    return i, float(np.sqrt(d2[i]))

def euclid(a, b):
    return float(np.linalg.norm(np.asarray(a, dtype=np.float32) - np.asarray(b, dtype=np.float32)))

# ---------- global curve-to-curve distance (DTW-style optimal alignment) ----------
# --- replace your dtw_curve_distance() with this normalized version ---
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
    for i in range(1, n):
        dp[i, 0] = C[i, 0] + dp[i-1, 0]
    for j in range(1, m):
        dp[0, j] = C[0, j] + dp[0, j-1]
    for i in range(1, n):
        for j in range(1, m):
            dp[i, j] = C[i, j] + min(dp[i-1, j], dp[i, j-1], dp[i-1, j-1])
    # normalize by an approximate alignment length to keep the scale stable
    norm = float(n + m)
    return float(dp[n-1, m-1] / max(norm, 1.0))


def compute_ccs(L_t: float, L0: float) -> float:
    L0 = max(L0, 1e-6)
    return float(1.0 - (L_t / L0))

# ---------- simple binary dilation (disk) ----------
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

# ---------- environment ----------
@dataclass
class CurveEpisode:
    img: np.ndarray
    mask: np.ndarray
    gt_poly: np.ndarray
    start: Tuple[int,int]
    init_dir: Tuple[int,int]
    ternary_map: np.ndarray   # +1 for target centerline, 0 elsewhere
    dilated_gt: np.ndarray    # dilated GT for robust overlap

class CurveEnv:
    """Directed curve tracking in 2D."""
    def __init__(self, h=128, w=128, branches=False, max_steps=400,
                 d0=2.0, dilate_radius=DILATE_RADIUS):
        self.h, self.w = h, w
        self.max_steps = max_steps
        self.cm = CurveMaker(h=h, w=w, thickness=1.5, seed=None)
        self.branches = branches
        self.D0 = d0
        self.dilate_radius = dilate_radius
        self.off_track_thresh = 1.8
        self.reset()

    def reset(self):
        img, mask, pts_all = self.cm.sample_curve(branches=self.branches)
        gt_poly = pts_all[0].astype(np.float32)
        p0 = gt_poly[0].astype(int)
        p1 = gt_poly[min(5, len(gt_poly)-1)].astype(int)
        init_vec = np.sign(np.array([p1[0]-p0[0], p1[1]-p0[1]], dtype=np.int32))
        init_vec[init_vec==0] = 1

        # ternary map: +1 on this target centerline; (no negatives available -> 0 elsewhere)
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

        # Progress & local distance memory
        self.best_idx = 0
        _, d0_local = nearest_gt_index(self.agent, self.ep.gt_poly)
        self.L_prev_local = d0_local

        # Learning/scheduling baseline L0:
        # use DTW distance between the GT and a "stuck-at-start" path of same length (non-zero, stable)
        stuck_path = [self.agent] * len(self.ep.gt_poly)
        self.L0 = dtw_curve_distance(stuck_path, self.ep.gt_poly)
        if self.L0 < 1e-6:
            self.L0 = 1.0


        self.last_reward = 0.0  # for critic feature vector
        return self.obs()

    def obs(self):
        p_t, p_1, p_2 = self.agent, self.prev[0], self.prev[1]
        ch0 = crop32(self.ep.img,  p_t[0], p_t[1])
        ch1 = crop32(self.ep.img,  p_1[0], p_1[1])
        ch2 = crop32(self.ep.img,  p_2[0], p_2[1])
        ch3 = crop32(self.path_mask, p_t[0], p_t[1])
        ch4 = crop32(self.ep.ternary_map, p_t[0], p_t[1])   # ternary GT channel (+1 on centerline)
        obs = np.stack([ch0, ch1, ch2, ch3, ch4], axis=0).astype(np.float32)
        return obs

    def step(self, a_idx: int):
        self.steps += 1
        dy, dx = ACTIONS_8[a_idx]
        ny = clamp(self.agent[0] + dy*STEP_ALPHA, 0, self.h-1)
        nx = clamp(self.agent[1] + dx*STEP_ALPHA, 0, self.w-1)
        new_pos = (ny, nx)

        # Move & mark
        self.prev = [self.agent, self.prev[0]]
        self.agent = new_pos
        self.path_points.append(self.agent)
        self.path_mask[self.agent] = 1.0

        # Local distance & progress
        idx, d_gt = nearest_gt_index(self.agent, self.ep.gt_poly)
        delta = d_gt - self.L_prev_local

        # ============ REWARD COMPUTATION (Paper Eq. 3, numerically safe) ============
        on_curve = (self.ep.dilated_gt[self.agent] > 0.5)
        B_t = 1.0 if on_curve else 0.0

        delta_abs = abs(delta)
        log_term = math.log(EPS_LOG + (delta_abs / self.D0))
        log_term = float(np.clip(log_term, -R_LOG_CLAMP, R_LOG_CLAMP))

        if delta < 0:  # getting closer
            r = LAMBDA_B * B_t - LAMBDA_DELTA * log_term
        else:          # farther/same
            r = LAMBDA_B * B_t + LAMBDA_DELTA * log_term

        # Update local memory for next step
        self.L_prev_local = d_gt

        # Termination rules (paper)
        ref_length = len(self.ep.gt_poly)
        track_length = len(self.path_points)
        exceeded_length = track_length > 1.5 * ref_length
        off_track = d_gt > self.off_track_thresh
        end_margin = 5
        reached_end = (idx >= len(self.ep.gt_poly) - 1 - end_margin)
        timeout = (self.steps >= self.max_steps)
        done = exceeded_length or off_track or reached_end or timeout

        if reached_end:
            r += 100.0  # success bonus

        # Global curve distance + CCS
        L_t = dtw_curve_distance(self.path_points, self.ep.gt_poly)
        ccs = compute_ccs(L_t, self.L0)

        self.last_reward = r  # for critic features

        return self.obs(), float(r), done, {
            "overlap": 1.0 if on_curve else 0.0,
            "L_local": d_gt,
            "idx": idx,
            "reached_end": reached_end,
            "timeout": timeout,
            "exceeded_length": exceeded_length,
            "off_track": off_track,
            "ccs": ccs
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

        # safe init
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_uniform_(m.weight, a=0.25, nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

    def forward(self, x, ahist_onehot, fhist_feat, hc_actor=None, hc_critic=None):
        # NaN/Inf guard
        x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        ahist_onehot = torch.nan_to_num(ahist_onehot, nan=0.0, posinf=0.0, neginf=0.0)
        fhist_feat = torch.nan_to_num(fhist_feat, nan=0.0, posinf=0.0, neginf=0.0)

        z = self.cnn(x)                         # (B,64,33,33)
        z = self.gap(z).squeeze(-1).squeeze(-1) # (B,64)

        out_a, hc_actor  = self.lstm_actor(ahist_onehot, hc_actor)   # (B,K,64)
        out_c, hc_critic = self.lstm_critic(fhist_feat,  hc_critic)  # (B,K,64)
        h_a = out_a[:, -1, :]                                        # (B,64)
        h_c = out_c[:, -1, :]                                        # (B,64)

        h_cat = torch.cat([z, h_a], dim=1)
        logits = self.actor_head(h_cat)

        h_val = torch.cat([z, h_c], dim=1)
        value  = self.critic_head(h_val).squeeze(-1)

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

        # LR schedule
        self.lr = lr
        self.lr_lower_bound = 1e-6
        self.patience = 5
        self.patience_counter = 0
        self.best_val_score = -float('inf')

    # --- inside PPO.update_learning_rate ---
    def update_learning_rate(self, val_score):
        if val_score > self.best_val_score:
            self.best_val_score = val_score
            self.patience_counter = 0
        else:
            self.patience_counter += 1

        if self.patience_counter >= self.patience:
            self.lr = max(self.lr / 2.0, self.lr_lower_bound)
            for pg in self.opt.param_groups:
                pg['lr'] = self.lr
            # also reduce entropy pressure a bit
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

        # Normalize advantages
        adv = adv - adv.mean()
        adv = adv / (adv.std() + 1e-8)

        N = obs.size(0)
        idx = np.arange(N)
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

# ---------- training / viewing ----------
def extract_critic_features(obs_crop: np.ndarray, last_reward: float):
    """
    Build critic features at current step:
    [ r_{t-1}, log1p|r_{t-1}|, center_intensity, mean_intensity ]
    obs_crop shape: (5, CROP, CROP) with ch0 = image @ current agent
    """
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

def train(args):
    set_seeds(123)
    env = CurveEnv(h=128, w=128, branches=args.branches, max_steps=400, d0=2.0, dilate_radius=DILATE_RADIUS)
    K = 8
    nA = len(ACTIONS_8)
    Fdim = 4
    model = ActorCritic(n_actions=nA, K=K, feat_dim=Fdim, in_ch=5).to(DEVICE)

    ppo = PPO(model, lr=1e-5, gamma=0.9, lam=0.95, clip=0.2,
              epochs=10, minibatch=8, entropy_coef=0.05)
    rollout_buf = {k: [] for k in ["obs","ahist","fhist","act","logp","adv","ret"]}

    ep_returns = []
    ep_ccs_scores = []

    for ep in range(1, args.episodes+1):

        # per-episode curriculum
        if ep < 2000:
            env.dilate_radius = 5
            env.off_track_thresh = 3.0
        elif ep < 4000:
            env.dilate_radius = 4
            env.off_track_thresh = 2.4
        else:
            env.dilate_radius = 3  # or your original
            env.off_track_thresh = 1.8

        obs = env.reset()
        done = False
        ahist = []
        fhist = []

        traj = {"obs":[], "ahist":[], "fhist":[], "act":[], "logp":[], "val":[], "rew":[], "done":[]}
        ep_ret = 0.0

        # seed first feature row (uses last_reward=0.0 from env.reset)
        feat = extract_critic_features(obs, last_reward=0.0)
        fhist.append(feat)

        last_value_for_bootstrap = 0.0
        last_done = False

        while not done:
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

            # update histories
            a1h = np.zeros(nA, dtype=np.float32); a1h[action] = 1.0
            ahist.append(a1h)

            feat = extract_critic_features(obs2, last_reward=r)
            fhist.append(feat)

            obs = obs2
            ep_ret += r

            last_value_for_bootstrap = val
            last_done = done

        # Store CCS metric
        ep_ccs_scores.append(info.get('ccs', 0.0))

        # GAE with timeout bootstrap
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

    # only update once we have minibatch episodes
    if len(rollout_buf["obs"]) >= ppo.minibatch:
        # concatenate episodes along batch dimension
        cat_buf = {k: np.concatenate(rollout_buf[k], axis=0) for k in rollout_buf}
        ppo.update(cat_buf)
        # clear for next cycle
        rollout_buf = {k: [] for k in rollout_buf}
        ep_returns.append(ep_ret)

        if ep % 100 == 0:
            avg_ret = float(np.mean(ep_returns[-100:]))
            avg_ccs = float(np.mean(ep_ccs_scores[-100:]))
            print(f"Episode {ep:6d} | return(avg100)={avg_ret:7.3f} | CCS(avg100)={avg_ccs:7.3f}")
            ppo.update_learning_rate(avg_ccs)

        if args.save and ep % args.save_every == 0:
            torch.save(model.state_dict(), args.save)
            print(f"Saved to {args.save}")

    if args.save:
        torch.save(model.state_dict(), args.save)
        print(f"Saved final weights to {args.save}")

def view(args):
    set_seeds(123)
    env = CurveEnv(h=128, w=128, branches=args.branches, max_steps=400)
    K = 8
    nA = len(ACTIONS_8)
    Fdim = 4
    model = ActorCritic(n_actions=nA, K=K, feat_dim=Fdim, in_ch=5).to(DEVICE)
    if args.weights:
        state = torch.load(args.weights, map_location=DEVICE)
        model.load_state_dict(state)
        print(f"Loaded weights: {args.weights}")
    model.eval()

    obs = env.reset()
    done = False
    ahist, fhist = [], []
    steps = 0
    feat = extract_critic_features(obs, 0.0)
    fhist.append(feat)

    with torch.no_grad():
        while not done:
            x = torch.tensor(obs[None], dtype=torch.float32, device=DEVICE)
            A = fixed_window_history(ahist, K, nA)[None, ...]
            Fh= fixed_window_features(fhist, K, Fdim)[None, ...]
            A_t = torch.tensor(A,  dtype=torch.float32, device=DEVICE)
            F_t = torch.tensor(Fh, dtype=torch.float32, device=DEVICE)

            logits, value, _, _ = model(x, A_t, F_t, None, None)
            action = int(torch.argmax(logits, dim=1).item())
            obs, r, done, info = env.step(action)

            a1h = np.zeros(nA, dtype=np.float32); a1h[action] = 1.0
            ahist.append(a1h)

            feat = extract_critic_features(obs, r)
            fhist.append(feat)
            steps += 1
    print(f"[VIEW] steps={steps}  L_end(local)={info['L_local']:.3f}  idx_end={info['idx']}  CCS={info['ccs']:.3f}")

# ---------- CLI ----------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--train", default=True, action="store_true")
    p.add_argument("--view",  action="store_true")
    p.add_argument("--episodes", type=int, default=5000)
    p.add_argument("--entropy_coef", type=float, default=0.03)
    p.add_argument("--save", type=str, default="ckpt_curveppo.pth")
    p.add_argument("--save_every", type=int, default=2000)
    p.add_argument("--weights", type=str, default="")
    p.add_argument("--branches", action="store_true")
    args = p.parse_args()

    if args.train: train(args)
    if args.view:  view(args)

if __name__ == "__main__":
    main()

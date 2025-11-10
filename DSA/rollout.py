#!/usr/bin/env python3
# rollout.py — testset creation, rollout with saved weights, and per-action images.
# Robust to training envs that omit some info fields (e.g., "exceeded_length").

import os, json, csv, argparse, math, random
from dataclasses import dataclass
from typing import Tuple, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# =========================================================
# Try to import your training module; else use local copies
# =========================================================
try:
    from Synth_simple_v1_2 import (
        CurveEnv, ActorCritic, fixed_window_history,
        ACTIONS_8, DEVICE
    )
    _HAVE_TRAINING_MODULE = True
except Exception as e:
    print("[rollout] Could not import Synth_simple_v1_2:", e)
    print("[rollout] Using built-in minimal definitions (must match your training code).")
    _HAVE_TRAINING_MODULE = False

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    ACTIONS_8 = [(-1, 0), (1, 0), (0,-1), (0, 1), (-1,-1), (-1,1), (1,-1), (1,1)]
    STEP_ALPHA = 2
    CROP = 33

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

    def _nearest_gt_index(pt, poly):
        dif = poly - np.array(pt, dtype=np.float32)
        d2 = np.sum(dif * dif, axis=1)
        i = int(np.argmin(d2))
        return i, float(np.sqrt(d2[i]))

    def _curve_to_curve_distance(path_points, gt_poly):
        if len(path_points) == 0:
            return 0.0
        path_arr = np.array(path_points, dtype=np.float32)
        total_dist = 0.0
        for pt in path_arr:
            dif = gt_poly - pt
            d2 = np.sum(dif * dif, axis=1)
            total_dist += np.sqrt(np.min(d2))
        return total_dist

    @dataclass
    class CurveEpisode:
        img: np.ndarray
        mask: np.ndarray
        gt_poly: np.ndarray
        start: Tuple[int,int]
        init_dir: Tuple[int,int]

    class CurveEnv:
        def __init__(self, h=128, w=128, max_steps=400, d0=2.0, overlap_dist=1.0):
            self.h, self.w = h, w
            self.max_steps = max_steps
            self.D0 = d0
            self.overlap_dist = overlap_dist
            self.ep = None
            self.agent = None
            self.prev = None
            self.steps = 0
            self.path_mask = None
            self.path_points = []
            self.prev_index = -1
            self.best_idx = 0
            self.L_prev_local = 0.0
            self.L0 = 1.0

        def obs(self):
            p_t, p_1, p_2 = self.agent, self.prev[0], self.prev[1]
            ch0 = crop32(self.ep.img,  p_t[0], p_t[1])
            ch1 = crop32(self.ep.img,  p_1[0], p_1[1])
            ch2 = crop32(self.ep.img,  p_2[0], p_2[1])
            ch3 = crop32(self.path_mask, p_t[0], p_t[1])
            obs = np.stack([ch0, ch1, ch2, ch3], axis=0).astype(np.float32)
            return obs

        def step(self, a_idx: int):
            self.steps += 1
            dy, dx = ACTIONS_8[a_idx]
            ny = clamp(self.agent[0] + dy*STEP_ALPHA, 0, self.h-1)
            nx = clamp(self.agent[1] + dx*STEP_ALPHA, 0, self.w-1)
            new_pos = (ny, nx)
            self.prev = [self.agent, self.prev[0]]
            self.agent = new_pos
            self.path_points.append(self.agent)
            self.path_mask[self.agent] = 1.0

            idx, d_gt = _nearest_gt_index(self.agent, self.ep.gt_poly)
            delta = d_gt - self.L_prev_local
            on_curve = d_gt < self.overlap_dist
            B_t = 1.0 if on_curve else 0.0
            eps = 1
            delta_abs = abs(delta)
            if delta < 0:
                r = math.log(eps + delta_abs /  self.D0) + B_t
            else:
                r = -math.log(eps + delta_abs / self.D0) + B_t
            self.L_prev_local = d_gt
            if idx <= self.prev_index: r -= 1.0
            else: r += 1.0
            self.prev_index = idx
            if idx > self.best_idx: self.best_idx = idx

            ref_length = len(self.ep.gt_poly)
            track_length = len(self.path_points)
            exceeded_length = track_length > 1.5 * ref_length
            off_track = d_gt > 10
            end_margin = 5
            reached_end = (self.best_idx >= len(self.ep.gt_poly) - 1 - end_margin)
            timeout = (self.steps >= self.max_steps)
            done = exceeded_length or off_track or reached_end or timeout
            if reached_end:
                r += 10.0
            L_t = _curve_to_curve_distance(self.path_points, self.ep.gt_poly)
            ccs = 1.0 - (L_t / self.L0)
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

    def gn(c, g=8):
        g = max(1, min(g, c))
        return nn.GroupNorm(g, c, eps=1e-5, affine=True)

    class ActorCritic(nn.Module):
        def __init__(self, n_actions=8, K=8):
            super().__init__()
            self.n_actions = n_actions
            self.K = K
            self.cnn = nn.Sequential(
                nn.Conv2d(4, 32, 3, padding=1, dilation=1), gn(32), nn.PReLU(),
                nn.Conv2d(32,32, 3, padding=2, dilation=2), gn(32), nn.PReLU(),
                nn.Conv2d(32,32, 3, padding=3, dilation=3), gn(32), nn.PReLU(),
                nn.Conv2d(32,64, 1),                         gn(64), nn.PReLU(),
            )
            self.gap = nn.AdaptiveAvgPool2d((1,1))
            self.lstm = nn.LSTM(input_size=n_actions, hidden_size=64, num_layers=1, batch_first=True)
            self.actor = nn.Sequential(nn.Linear(64+64,128), nn.PReLU(), nn.Linear(128, n_actions))
            self.critic = nn.Sequential(nn.Linear(64+64,128), nn.PReLU(), nn.Linear(128, 1))
            for m in self.modules():
                if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                    nn.init.kaiming_uniform_(m.weight, a=0.25, nonlinearity='leaky_relu')
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0.0)

        def forward(self, x, ahist_onehot, hc=None):
            x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
            ahist_onehot = torch.nan_to_num(ahist_onehot, nan=0.0, posinf=0.0, neginf=0.0)
            z = self.cnn(x)
            z = self.gap(z).squeeze(-1).squeeze(-1)
            out, hc = self.lstm(ahist_onehot, hc)
            h_last = out[:, -1, :]
            h = torch.cat([z, h_last], dim=1)
            logits = self.actor(h)
            value  = self.critic(h).squeeze(-1)
            logits = torch.nan_to_num(logits, nan=0.0, posinf=20.0, neginf=-20.0).clamp(-20, 20)
            value  = torch.nan_to_num(value,  nan=0.0, posinf=0.0, neginf=0.0)
            return logits, value, hc

# =========================================
# Episode I/O helpers
# =========================================

def ensure_dir(p):
    os.makedirs(p, exist_ok=True)
    return p

def save_episode_npz(path, img, mask, gt_poly, start, init_dir, meta=None):
    meta = meta or {}
    np.savez_compressed(
        path,
        img=img.astype(np.float32),
        mask=mask.astype(np.float32),
        gt_poly=gt_poly.astype(np.float32),
        start=np.array(start, dtype=np.int32),
        init_dir=np.array(init_dir, dtype=np.int32),
        meta=json.dumps(meta)
    )

def load_episode_npz(path):
    d = np.load(path, allow_pickle=False)
    img = d["img"].astype(np.float32)
    mask = d["mask"].astype(np.float32)
    gt_poly = d["gt_poly"].astype(np.float32)
    start = tuple(d["start"].tolist())
    init_dir = tuple(d["init_dir"].tolist())
    meta = json.loads(str(d["meta"]))
    return img, mask, gt_poly, start, init_dir, meta

# =========================================
# Compatibility injector for CurveEnv
# =========================================

def _nearest_gt_index(pt, poly):
    dif = poly - np.array(pt, dtype=np.float32)
    d2 = np.sum(dif * dif, axis=1)
    i = int(np.argmin(d2))
    return i, float(np.sqrt(d2[i]))

def _curve_to_curve_distance(path_points, gt_poly):
    if len(path_points) == 0:
        return 0.0
    path_arr = np.array(path_points, dtype=np.float32)
    total_dist = 0.0
    for pt in path_arr:
        dif = gt_poly - pt
        d2 = np.sum(dif * dif, axis=1)
        total_dist += np.sqrt(np.min(d2))
    return total_dist

def set_episode_compat(env, img, mask, gt_poly, start, init_dir):
    if hasattr(env, "set_episode"):
        return env.set_episode(img, mask, gt_poly, start, init_dir)

    # Construct an ep object similar to training
    try:
        from Synth_simple_v1_2 import CurveEpisode
        ep = CurveEpisode(
            img=img.astype(np.float32),
            mask=mask.astype(np.float32),
            gt_poly=gt_poly.astype(np.float32),
            start=(int(start[0]), int(start[1])),
            init_dir=(int(init_dir[0]), int(init_dir[1])),
        )
    except Exception:
        class _Ep: pass
        ep = _Ep()
        ep.img = img.astype(np.float32)
        ep.mask = mask.astype(np.float32)
        ep.gt_poly = gt_poly.astype(np.float32)
        ep.start = (int(start[0]), int(start[1]))
        ep.init_dir = (int(init_dir[0]), int(init_dir[1]))

    env.ep = ep
    env.agent = (int(start[0]), int(start[1]))
    env.prev  = [env.agent, env.agent]
    env.steps = 0
    env.path_mask = np.zeros_like(env.ep.mask, dtype=np.float32)
    env.path_points = [env.agent]
    env.path_mask[env.agent] = 1.0
    env.prev_index = -1
    env.best_idx = 0

    _, d0_local = _nearest_gt_index(env.agent, env.ep.gt_poly)
    env.L_prev_local = d0_local
    L0 = _curve_to_curve_distance([env.agent], env.ep.gt_poly)
    env.L0 = 1.0 if L0 < 1e-6 else L0

    return env.obs()

# =========================================
# Test set creation
# =========================================

def make_testset(args):
    from Curve_Generator import CurveMaker
    rng = np.random.default_rng(args.seed)
    ensure_dir(args.testset_dir)

    count = 0
    for i in range(args.num_cases):
        try:
            cm = CurveMaker(h=args.h, w=args.w, thickness=1.5,
                            seed=int(rng.integers(0, 2**31-1)))
        except TypeError:
            cm = CurveMaker(h=args.h, w=args.w, thickness=1.5)

        img, mask, pts_all = cm.sample_curve(branches=args.branches)
        gt_poly = pts_all[0].astype(np.float32)

        p0 = gt_poly[0].astype(int)
        p1 = gt_poly[min(5, len(gt_poly)-1)].astype(int)
        init_vec = np.sign(np.array([p1[0]-p0[0], p1[1]-p0[1]], dtype=np.int32))
        init_vec[init_vec==0] = 1
        start = (int(p0[0]), int(p0[1]))
        init_dir = (int(init_vec[0]), int(init_vec[1]))

        fname = os.path.join(args.testset_dir, f"case_{i:05d}.npz")
        meta = {"h": args.h, "w": args.w, "branches": bool(args.branches)}
        save_episode_npz(fname, img, mask, gt_poly, start, init_dir, meta)
        count += 1

    print(f"[make_testset] Wrote {count} cases to {args.testset_dir}")

# =========================================
# Plotting & rollout
# =========================================

def plot_episode(img, gt_poly, path_points, out_png):
    H, W = img.shape
    fig, ax = plt.subplots(figsize=(W/100, H/100), dpi=100)
    ax.imshow(img, cmap='gray', vmin=0, vmax=1, origin='upper')
    if gt_poly is not None and len(gt_poly) > 1:
        ax.plot(gt_poly[:,1], gt_poly[:,0], lw=2, alpha=0.85, label='GT')
    if len(path_points) > 1:
        py = [p[0] for p in path_points]
        px = [p[1] for p in path_points]
        ax.plot(px, py, lw=2, alpha=0.9, label='Agent')
        ax.scatter([px[0]],[py[0]], s=10)
        ax.scatter([px[-1]],[py[-1]], s=10)
    ax.set_axis_off()
    ax.legend(loc='lower right')
    fig.tight_layout(pad=0)
    fig.savefig(out_png, bbox_inches='tight', pad_inches=0)
    plt.close(fig)

def _safe_bool(info, key, default=False):
    return bool(info[key]) if key in info else bool(default)

def _safe_float(info, key, default=0.0):
    return float(info[key]) if key in info else float(default)

def _safe_int(info, key, default=0):
    return int(info[key]) if key in info else int(default)

def _fallback_ccs(env):
    # If env exposes path and L0, compute CCS; else 0.0
    try:
        L_t = _curve_to_curve_distance(env.path_points, env.ep.gt_poly)
        denom = env.L0 if hasattr(env, "L0") and env.L0 > 1e-6 else 1.0
        return float(1.0 - (L_t / denom))
    except Exception:
        return 0.0

def rollout_one(env, model, episode_npz_path, outdir, deterministic=False):
    img, mask, gt_poly, start, init_dir, meta = load_episode_npz(episode_npz_path)
    obs = set_episode_compat(env, img, mask, gt_poly, start, init_dir)

    K = 8
    nA = len(ACTIONS_8)
    model.eval()
    ahist = []
    done = False
    steps = 0

    traj_jsonl_path = os.path.join(outdir, os.path.basename(episode_npz_path).replace(".npz", "_traj.jsonl"))
    traj_npy_path   = os.path.join(outdir, os.path.basename(episode_npz_path).replace(".npz", "_path.npy"))
    png_path        = os.path.join(outdir, os.path.basename(episode_npz_path).replace(".npz", "_plot.png"))

    last_info = {}
    with open(traj_jsonl_path, "w") as fjsonl, torch.no_grad():
        while not done:
            x = torch.tensor(obs[None], dtype=torch.float32, device=DEVICE)
            A = fixed_window_history(ahist, K, nA)[None, ...]
            A_t = torch.tensor(A, dtype=torch.float32, device=DEVICE)

            logits, value, _ = model(x, A_t, None)
            if deterministic:
                action = int(torch.argmax(logits, dim=1).item())
            else:
                dist = Categorical(logits=logits)
                action = int(dist.sample().item())

            obs2, r, done, info = env.step(action)
            last_info = info  # keep the latest

            rec = {
                "t": steps,
                "action": int(action),
                "reward": float(r),
                "agent_yx": [int(env.agent[0]), int(env.agent[1])],
                "idx": _safe_int(info, "idx", 0),
                "L_local": _safe_float(info, "L_local", 0.0),
                "overlap": _safe_float(info, "overlap", 0.0),
                "reached_end": _safe_bool(info, "reached_end", False),
                "timeout": _safe_bool(info, "timeout", False),
                "exceeded_length": _safe_bool(info, "exceeded_length", False),
                "off_track": _safe_bool(info, "off_track", False),
                "ccs": _safe_float(info, "ccs", _fallback_ccs(env)),
            }
            fjsonl.write(json.dumps(rec) + "\n")

            a1h = np.zeros(nA, dtype=np.float32); a1h[action] = 1.0
            ahist.append(a1h)
            obs = obs2
            steps += 1

    path_arr = np.array(env.path_points, dtype=np.int32)
    np.save(traj_npy_path, path_arr)
    plot_episode(img, gt_poly, env.path_points, png_path)

    # Final metrics, using safe fallbacks
    return {
        "case": os.path.basename(episode_npz_path),
        "steps": steps,
        "final_y": int(env.agent[0]),
        "final_x": int(env.agent[1]),
        "final_idx": _safe_int(last_info, "idx", 0),
        "reached_end": _safe_bool(last_info, "reached_end", False),
        "timeout": _safe_bool(last_info, "timeout", False),
        "exceeded_length": _safe_bool(last_info, "exceeded_length", False),
        "off_track": _safe_bool(last_info, "off_track", False),
        "ccs": _safe_float(last_info, "ccs", _fallback_ccs(env)),
    }

def rollout_testset(args):
    ensure_dir(args.outdir)

    model = ActorCritic(n_actions=len(ACTIONS_8), K=8).to(DEVICE)
    try:
        state = torch.load(args.weights, map_location=DEVICE, weights_only=True)
    except TypeError:
        state = torch.load(args.weights, map_location=DEVICE)
    model.load_state_dict(state)
    print(f"[rollout] Loaded weights: {args.weights}")

    env = CurveEnv(h=args.h, w=args.w, max_steps=args.max_steps, d0=2.0, overlap_dist=1.0)

    all_cases = sorted([os.path.join(args.testset_dir, f)
                        for f in os.listdir(args.testset_dir) if f.endswith(".npz")])
    if args.limit is not None:
        all_cases = all_cases[:args.limit]
    print(f"[rollout] Found {len(all_cases)} cases in {args.testset_dir}")

    csv_path = os.path.join(args.outdir, "summary.csv")
    with open(csv_path, "w", newline="") as fcsv:
        writer = csv.DictWriter(fcsv, fieldnames=[
            "case","steps","final_y","final_x","final_idx",
            "reached_end","timeout","exceeded_length","off_track","ccs"
        ])
        writer.writeheader()

        for i, case_path in enumerate(all_cases, 1):
            metrics = rollout_one(env, model, case_path, args.outdir, deterministic=not args.stochastic)
            writer.writerow(metrics)
            if i % 10 == 0 or i == len(all_cases):
                print(f"[rollout] {i}/{len(all_cases)} done | case={metrics['case']} | CCS={metrics['ccs']:.3f}")

    print(f"[rollout] Wrote per-episode artifacts to: {args.outdir}")
    print(f"[rollout] Summary CSV: {csv_path}")

# =========================================
# PER-ACTION VISUALIZATION
# =========================================

def _plot_points_only(img, gt_poly, points, out_png, title=None):
    H, W = img.shape
    fig, ax = plt.subplots(figsize=(W/100, H/100), dpi=100)
    ax.imshow(img, cmap='gray', vmin=0, vmax=1, origin='upper')
    if gt_poly is not None and len(gt_poly) > 1:
        ax.plot(gt_poly[:,1], gt_poly[:,0], lw=2, alpha=0.6, label='GT')
    if len(points) > 0:
        py = [p[0] for p in points]
        px = [p[1] for p in points]
        ax.plot(px, py, lw=2, alpha=0.95, label='Selected')
        ax.scatter([px[0]],[py[0]], s=10)
        ax.scatter([px[-1]],[py[-1]], s=10)
    if title:
        ax.set_title(title, fontsize=8)
    ax.set_axis_off()
    ax.legend(loc='lower right')
    fig.tight_layout(pad=0)
    fig.savefig(out_png, bbox_inches='tight', pad_inches=0)
    plt.close(fig)

def per_action_policy_images(env, model, episode_npz_path, outdir):
    img, mask, gt_poly, start, init_dir, meta = load_episode_npz(episode_npz_path)
    obs = set_episode_compat(env, img, mask, gt_poly, start, init_dir)
    K = 8; nA = len(ACTIONS_8)
    model.eval(); ahist = []; done = False
    per_action_pts = {a: [] for a in range(nA)}
    with torch.no_grad():
        while not done:
            x = torch.tensor(obs[None], dtype=torch.float32, device=DEVICE)
            A = fixed_window_history(ahist, K, nA)[None, ...]
            A_t = torch.tensor(A, dtype=torch.float32, device=DEVICE)
            logits, value, _ = model(x, A_t, None)
            action = int(torch.argmax(logits, dim=1).item())
            obs, r, done, info = env.step(action)
            per_action_pts[action].append((int(env.agent[0]), int(env.agent[1])))
            a1h = np.zeros(nA, dtype=np.float32); a1h[action] = 1.0
            ahist.append(a1h)
    base = os.path.basename(episode_npz_path).replace(".npz", "")
    for a in range(nA):
        out_png = os.path.join(outdir, f"{base}_policy_action{a}.png")
        _plot_points_only(img, gt_poly, per_action_pts[a], out_png,
                          title=f"Policy-chosen points for action {a}")

def per_action_forced_images(env, episode_npz_path, outdir, which_actions=None, max_steps=None):
    img, mask, gt_poly, start, init_dir, meta = load_episode_npz(episode_npz_path)
    if which_actions is None:
        which_actions = list(range(len(ACTIONS_8)))
    base = os.path.basename(episode_npz_path).replace(".npz", "")
    for a in which_actions:
        set_episode_compat(env, img, mask, gt_poly, start, init_dir)
        steps = 0; done = False; path_pts = []
        while not done:
            obs, r, done, info = env.step(a)
            path_pts.append((int(env.agent[0]), int(env.agent[1])))
            steps += 1
            if max_steps is not None and steps >= max_steps:
                break
        out_png = os.path.join(outdir, f"{base}_forced_action{a}.png")
        _plot_points_only(img, gt_poly, path_pts, out_png,
                          title=f"Forced trajectory for action {a}")

# =========================================
# CLI
# =========================================

def main():
    p = argparse.ArgumentParser(description="Create test set, roll out a trained model, and visualize per-action trajectories.")
    sub = p.add_subparsers(dest="cmd", required=True)

    p_make = sub.add_parser("make_testset", help="Generate a fixed test set of NPZ episodes.")
    p_make.add_argument("--testset_dir", type=str, default="testset")
    p_make.add_argument("--num_cases", type=int, default=100)
    p_make.add_argument("--h", type=int, default=128)
    p_make.add_argument("--w", type=int, default=128)
    p_make.add_argument("--max_steps", type=int, default=400)
    p_make.add_argument("--branches", default=False, action="store_true")
    p_make.add_argument("--seed", type=int, default=2025)

    p_roll = sub.add_parser("rollout", help="Run a saved model on the fixed test set.")
    p_roll.add_argument("--weights", type=str, required=True)
    p_roll.add_argument("--testset_dir", type=str, default="testset")
    p_roll.add_argument("--outdir", type=str, default="rollout_out")
    p_roll.add_argument("--h", type=int, default=128)
    p_roll.add_argument("--w", type=int, default=128)
    p_roll.add_argument("--max_steps", type=int, default=400)
    p_roll.add_argument("--stochastic", action="store_true")
    p_roll.add_argument("--limit", type=int, default=None)

    p_pa = sub.add_parser("per_action", help="Save per-action images (policy or forced).")
    p_pa.add_argument("--weights", type=str, default="")
    p_pa.add_argument("--testset_dir", type=str, default="testset")
    p_pa.add_argument("--outdir", type=str, default="per_action_out")
    p_pa.add_argument("--h", type=int, default=128)
    p_pa.add_argument("--w", type=int, default=128)
    p_pa.add_argument("--max_steps", type=int, default=400)
    p_pa.add_argument("--limit", type=int, default=1)
    p_pa.add_argument("--mode", choices=["policy", "forced"], default="policy")
    p_pa.add_argument("--action", type=int, default=None)

    args = p.parse_args()
    random.seed(123); np.random.seed(123); torch.manual_seed(123)

    if args.cmd == "make_testset":
        make_testset(args)

    elif args.cmd == "rollout":
        rollout_testset(args)

    elif args.cmd == "per_action":
        ensure_dir(args.outdir)
        cases = sorted([os.path.join(args.testset_dir, f)
                        for f in os.listdir(args.testset_dir) if f.endswith(".npz")])
        if args.limit is not None:
            cases = cases[:args.limit]
        print(f"[per_action] Using {len(cases)} case(s) from {args.testset_dir}")
        env = CurveEnv(h=args.h, w=args.w, max_steps=args.max_steps, d0=2.0, overlap_dist=1.0)

        if args.mode == "policy":
            if not args.weights:
                raise SystemExit("per_action (policy) requires --weights")
            model = ActorCritic(n_actions=len(ACTIONS_8), K=8).to(DEVICE)
            try:
                state = torch.load(args.weights, map_location=DEVICE, weights_only=True)
            except TypeError:
                state = torch.load(args.weights, map_location=DEVICE)
            model.load_state_dict(state)
            print(f"[per_action] Loaded weights: {args.weights}")
            for i, case in enumerate(cases, 1):
                per_action_policy_images(env, model, case, args.outdir)
                print(f"[per_action] policy {i}/{len(cases)}: {os.path.basename(case)}")
        else:
            which = [args.action] if args.action is not None else list(range(len(ACTIONS_8)))
            for i, case in enumerate(cases, 1):
                per_action_forced_images(env, case, args.outdir, which_actions=which, max_steps=args.max_steps)
                print(f"[per_action] forced {i}/{len(cases)}: {os.path.basename(case)}")
    else:
        raise ValueError("Unknown command")

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
train_rl_dsa_GPT_version_2.py  (FIXED)

Fixes:
1) v_loss shape bug (no more [4096,4096] broadcasting)
2) "backward through graph a second time" (adv/ret/logp/val are detached properly)

Supports two modes:
- Version A: STOP as an action (9 actions, action==8 means STOP)
- Version B: Termination head (8 actions + stop head, supervised with soft target)

Run examples:
python train_rl_dsa_GPT_version_2.py --exp_name test --out_dir runs --seed 0 --device cuda --use_stop_action 1 --use_termination_head 0
python train_rl_dsa_GPT_version_2.py --exp_name test --out_dir runs --seed 0 --device cuda --use_stop_action 0 --use_termination_head 1
"""

import os, time, math, csv, json, argparse
from dataclasses import dataclass, asdict
from typing import Dict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from torch.utils.tensorboard import SummaryWriter

from Curve_Generator_Flexible_For_Ciruculum_learning import CurveMakerFlexible


# -----------------------------
# Helpers
# -----------------------------
ACTIONS_8 = [(-1, 0), (-1, 1), (0, 1), (1, 1),
            (1, 0), (1, -1), (0, -1), (-1, -1)]

def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def clamp(v, lo, hi):
    return lo if v < lo else hi if v > hi else v

def crop(img: np.ndarray, cy: float, cx: float, size: int = 32) -> np.ndarray:
    H, W = img.shape
    r = size // 2
    out = np.zeros((size, size), dtype=np.float32)
    y0, y1 = int(round(cy - r)), int(round(cy + r))
    x0, x1 = int(round(cx - r)), int(round(cx + r))
    yy0, yy1 = max(0, y0), min(H, y1)
    xx0, xx1 = max(0, x0), min(W, x1)
    out[yy0 - y0:yy1 - y0, xx0 - x0:xx1 - x0] = img[yy0:yy1, xx0:xx1]
    return out

def nearest_idx(p: np.ndarray, poly: np.ndarray) -> int:
    d2 = np.sum((poly - p[None]) ** 2, axis=1)
    return int(np.argmin(d2))

def dist_to_poly(p: np.ndarray, poly: np.ndarray) -> float:
    i = nearest_idx(p, poly)
    return float(np.linalg.norm(poly[i] - p))

def cosine(a: np.ndarray, b: np.ndarray, eps=1e-8) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + eps))

def turn_penalty(p2, p1, p0) -> float:
    v1 = np.array(p1) - np.array(p2)
    v2 = np.array(p0) - np.array(p1)
    if np.linalg.norm(v1) < 1e-6 or np.linalg.norm(v2) < 1e-6:
        return 0.0
    return 1.0 - cosine(v1, v2)


# -----------------------------
# Config
# -----------------------------
@dataclass
class Cfg:
    exp_name: str = "B_termhead_v2"
    out_dir: str = "runs_gpt_version_2"
    seed: int = 0
    device: str = "cuda"

    # generator/env
    H: int = 128
    W: int = 128
    crop: int = 32
    max_steps: int = 180

    use_distractors: int = 1
    step_alpha: float = 1.0

    # mode flags
    use_stop_action: int = 0
    use_termination_head: int = 1

    # stop settings
    stop_radius: float = 5.0
    stop_soft_scale: float = 3.0
    stop_threshold: float = 0.7
    no_prog_k: int = 12
    no_prog_eps: float = 1.0

    # reward weights (tracking)
    w_precision: float = 2.0
    w_progress: float = 0.25
    w_align: float = 0.25
    w_turn: float = 0.25
    w_step: float = 0.01
    offtrack_L: float = 6.0

    # stop-action reward (Version A)
    r_stop_correct: float = 40.0
    r_stop_wrong: float = -2.0

    # PPO
    total_steps: int = 1_200_000
    rollout_steps: int = 4096
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_eps: float = 0.2
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    stop_coef: float = 1.5
    lr: float = 3e-4
    update_epochs: int = 6
    minibatch: int = 512
    max_grad_norm: float = 1.0

    # eval/log
    eval_every_updates: int = 10
    eval_episodes: int = 50
    save_every_updates: int = 50


# -----------------------------
# Environment
# -----------------------------
class TrackEnv:
    """
    Synthetic curve tracking env.
    Observation: 4 channels:
      0: image crop
      1: prev1 crop
      2: prev2 crop
      3: visited crop
    """
    def __init__(self, cfg: Cfg):
        self.cfg = cfg
        self.gen = CurveMakerFlexible(cfg.H, cfg.W, seed=cfg.seed)
        self.reset()

    @property
    def n_actions(self) -> int:
        if self.cfg.use_stop_action:
            return 9  # 8 moves + STOP
        return 8

    def reset(self):
        # Use distractors if your generator supports it
        out = self.gen.sample_with_distractors() if self.cfg.use_distractors else self.gen.sample()
        if len(out) == 3:
            img, mask, pts = out
        else:
            raise RuntimeError("Generator must return (img, mask, pts)")

        self.img = img.astype(np.float32)
        self.mask = mask.astype(np.float32)
        self.poly = pts[0].astype(np.float32)  # (N,2) in (y,x)
        self.end = self.poly[-1].copy()

        self.gt_idx = 3
        self.agent = self.poly[self.gt_idx].copy()

        self.prev = [self.agent.copy(), self.agent.copy(), self.agent.copy()]
        self.visited = np.zeros_like(self.img, dtype=np.float32)

        self.steps = 0
        return self._obs()

    def _obs(self):
        y, x = self.agent
        obs = np.stack([
            crop(self.img, y, x, self.cfg.crop),
            crop(self.img, *self.prev[-1], self.cfg.crop),
            crop(self.img, *self.prev[-2], self.cfg.crop),
            crop(self.visited, y, x, self.cfg.crop),
        ]).astype(np.float32)
        return obs

    def step(self, action: int):
        self.steps += 1

        # Version A: STOP as action
        if self.cfg.use_stop_action and action == 8:
            d_end = float(np.linalg.norm(self.agent - self.end))
            if d_end < self.cfg.stop_radius:
                return self._obs(), float(self.cfg.r_stop_correct), True, {"success": True, "d_end": d_end, "L": dist_to_poly(self.agent, self.poly), "idx": self.gt_idx}
            return self._obs(), float(self.cfg.r_stop_wrong), False, {"success": False, "d_end": d_end, "L": dist_to_poly(self.agent, self.poly), "idx": self.gt_idx}

        # move
        dy, dx = ACTIONS_8[action]
        self.agent[0] = clamp(self.agent[0] + dy * self.cfg.step_alpha, 0, self.cfg.H - 1)
        self.agent[1] = clamp(self.agent[1] + dx * self.cfg.step_alpha, 0, self.cfg.W - 1)

        self.prev.append(self.agent.copy())
        self.prev = self.prev[-3:]
        self.visited[int(round(self.agent[0])), int(round(self.agent[1]))] = 1.0

        # shaping metrics
        L = dist_to_poly(self.agent, self.poly)
        idx = nearest_idx(self.agent, self.poly)
        prog = max(0, idx - self.gt_idx)
        self.gt_idx = max(self.gt_idx, idx)

        k = 3
        gt_vec = self.poly[min(len(self.poly) - 1, idx + k)] - self.poly[max(0, idx - k)]
        mv = self.prev[-1] - self.prev[-2]
        align = max(0.0, cosine(gt_vec, mv))
        turn = turn_penalty(self.prev[-3], self.prev[-2], self.prev[-1])

        precision = math.exp(-(L * L) / 2.0)

        reward = (
            self.cfg.w_precision * precision +
            self.cfg.w_progress * float(prog) +
            self.cfg.w_align * float(align) -
            self.cfg.w_turn * float(turn) -
            self.cfg.w_step
        )

        d_end = float(np.linalg.norm(self.agent - self.end))
        stop_target = float(np.exp(-d_end / self.cfg.stop_soft_scale))  # soft label
        stop_target = min(stop_target, 1.0)
        near_end = float(d_end < self.cfg.stop_radius)

        done = (self.steps >= self.cfg.max_steps)

        info = {
            "L": float(L),
            "turn": float(turn),
            "align": float(align),
            "off": float(L > self.cfg.offtrack_L),
            "idx": int(idx),
            "d_end": d_end,
            "stop_target": stop_target,
            "near_end": near_end,
            "success": False,
        }
        return self._obs(), float(reward), done, info


# -----------------------------
# Model
# -----------------------------
class PolicyNet(nn.Module):
    def __init__(self, obs_ch=4, n_actions=8, use_term_head=True):
        super().__init__()
        self.use_term_head = use_term_head
        self.cnn = nn.Sequential(
            nn.Conv2d(obs_ch, 32, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=2, padding=1), nn.ReLU(),
        )
        self.fc = nn.Linear(64 * 4 * 4, 256)
        self.pi = nn.Linear(256, n_actions)
        self.v  = nn.Linear(256, 1)
        self.stop = nn.Linear(256, 1)

    def forward(self, x):
        h = self.cnn(x).flatten(1)
        h = F.relu(self.fc(h))
        logits = self.pi(h)
        value  = self.v(h).squeeze(-1)           # (B,)
        stop_logit = self.stop(h).squeeze(-1)    # (B,)
        return logits, value, stop_logit


# -----------------------------
# PPO Buffer
# -----------------------------
class Buffer:
    def __init__(self, T, obs_shape, device):
        self.T = T
        self.device = device
        self.obs = torch.zeros((T, *obs_shape), dtype=torch.float32, device=device)
        self.act = torch.zeros((T,), dtype=torch.long, device=device)
        self.logp = torch.zeros((T,), dtype=torch.float32, device=device)
        self.val  = torch.zeros((T,), dtype=torch.float32, device=device)  # IMPORTANT: 1D
        self.rew  = torch.zeros((T,), dtype=torch.float32, device=device)
        self.done = torch.zeros((T,), dtype=torch.float32, device=device)
        self.stop_target = torch.zeros((T,), dtype=torch.float32, device=device)
        self.near_end = torch.zeros((T,), dtype=torch.float32, device=device)

    def add(self, t, obs, act, logp, val, rew, done, stop_target, near_end):
        self.obs[t] = obs
        self.act[t] = act
        self.logp[t] = logp
        self.val[t] = val
        self.rew[t] = rew
        self.done[t] = float(done)
        self.stop_target[t] = stop_target
        self.near_end[t] = near_end


def compute_gae(rew, val, done, gamma, lam):
    """
    All inputs are shape (T,) and outputs are shape (T,).
    Returns are detached to avoid reusing graphs across PPO epochs.
    """
    T = rew.shape[0]
    adv = torch.zeros((T,), device=rew.device)
    last = 0.0
    for t in reversed(range(T)):
        nonterm = 1.0 - done[t]
        next_v = val[t + 1] if t < T - 1 else 0.0
        delta = rew[t] + gamma * next_v * nonterm - val[t]
        last = delta + gamma * lam * nonterm * last
        adv[t] = last
    ret = adv + val
    return adv.detach(), ret.detach()


@torch.no_grad()
def evaluate(cfg: Cfg, model: PolicyNet, device: torch.device) -> Dict[str, float]:
    model.eval()
    env = TrackEnv(cfg)

    succ = 0
    mean_L = []
    stop_attempts = 0
    stop_correct = 0

    for _ in range(cfg.eval_episodes):
        obs = env.reset()
        idx_hist = []
        stopped = False
        last_info = {}

        for _t in range(cfg.max_steps):
            o = torch.tensor(obs[None], device=device)
            logits, _, stop_logit = model(o)

            act = torch.argmax(logits, dim=-1).item()
            p_stop = torch.sigmoid(stop_logit)[0].item()

            obs2, _, done, info = env.step(act)
            last_info = info
            idx_hist.append(info.get("idx", 0))

            # Termination decision (only if using termination head)
            if cfg.use_termination_head:
                no_prog = False
                if len(idx_hist) >= cfg.no_prog_k:
                    delta = idx_hist[-1] - idx_hist[-cfg.no_prog_k]
                    no_prog = (delta <= cfg.no_prog_eps)
                if (p_stop > cfg.stop_threshold) and no_prog:
                    stopped = True
                    stop_attempts += 1
                    if info.get("d_end", 1e9) < cfg.stop_radius:
                        stop_correct += 1
                    break

            obs = obs2
            if done:
                break

        if cfg.use_stop_action:
            # for stop-action, count success from env info
            if last_info.get("success", False):
                succ += 1
        else:
            # for termination head, success = stopped near end
            if stopped and (last_info.get("d_end", 1e9) < cfg.stop_radius):
                succ += 1

        mean_L.append(float(last_info.get("L", 1e9)))

    out = {
        "eval/success_rate": succ / float(cfg.eval_episodes),
        "eval/mean_L": float(np.mean(mean_L)),
        "eval/stop_precision": (stop_correct / stop_attempts) if stop_attempts > 0 else 0.0,
        "eval/stop_attempt_rate": stop_attempts / float(cfg.eval_episodes),
    }
    model.train()
    return out


def train(cfg: Cfg):
    set_seed(cfg.seed)
    device = torch.device(cfg.device if (cfg.device == "cuda" and torch.cuda.is_available()) else "cpu")

    run_dir = os.path.join(cfg.out_dir, cfg.exp_name)
    os.makedirs(run_dir, exist_ok=True)

    with open(os.path.join(run_dir, "config.json"), "w") as f:
        json.dump(asdict(cfg), f, indent=2)

    writer = SummaryWriter(os.path.join(run_dir, "tb"))
    csv_path = os.path.join(run_dir, "metrics.csv")
    csv_f = open(csv_path, "w", newline="")
    csv_w = csv.DictWriter(csv_f, fieldnames=[
        "update", "steps", "train/Rsum", "train/L", "train/turn", "train/off",
        "train/stop_loss", "train/p_stop_mean", "train/p_stop_nearend",
        "eval/success_rate", "eval/mean_L", "eval/stop_precision", "eval/stop_attempt_rate"
    ])
    csv_w.writeheader()
    csv_f.flush()

    env = TrackEnv(cfg)
    model = PolicyNet(n_actions=env.n_actions, use_term_head=bool(cfg.use_termination_head)).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    obs = env.reset()
    obs_shape = obs.shape

    total_steps = 0
    update = 0
    t0 = time.time()
    best_succ = -1.0

    while total_steps < cfg.total_steps:
        buf = Buffer(cfg.rollout_steps, obs_shape, device)

        # -------- Rollout --------
        R_acc, L_acc, T_acc, O_acc = [], [], [], []
        pstop_all, pstop_ne = [], []
        stop_loss_last = 0.0

        for t in range(cfg.rollout_steps):
            o = torch.tensor(obs, dtype=torch.float32, device=device)
            logits, val, stop_logit = model(o[None])

            dist = Categorical(logits=logits)
            act = dist.sample()
            logp = dist.log_prob(act)

            obs2, r, done, info = env.step(int(act.item()))

            # IMPORTANT: detach logp/val before storing
            buf.add(
                t=t,
                obs=o,
                act=act.squeeze(0),
                logp=logp.squeeze(0).detach(),
                val=val.squeeze(0).detach(),
                rew=torch.tensor(r, device=device),
                done=done,
                stop_target=torch.tensor(info.get("stop_target", 0.0), device=device),
                near_end=torch.tensor(info.get("near_end", 0.0), device=device)
            )

            total_steps += 1
            obs = obs2

            R_acc.append(float(r))
            L_acc.append(float(info.get("L", 0.0)))
            T_acc.append(float(info.get("turn", 0.0)))
            O_acc.append(float(info.get("off", 0.0)))

            p = torch.sigmoid(stop_logit).item()
            pstop_all.append(p)
            if info.get("near_end", 0.0) > 0.5:
                pstop_ne.append(p)

            if done:
                obs = env.reset()

        # -------- Compute advantages --------
        adv, ret = compute_gae(buf.rew, buf.val, buf.done, cfg.gamma, cfg.gae_lambda)
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        # -------- PPO updates --------
        idxs = np.arange(cfg.rollout_steps)

        for _ep in range(cfg.update_epochs):
            np.random.shuffle(idxs)
            for start in range(0, cfg.rollout_steps, cfg.minibatch):
                mb_idx = idxs[start:start + cfg.minibatch]
                mb = torch.tensor(mb_idx, device=device)

                logits, v, stop_logit = model(buf.obs[mb])
                dist = Categorical(logits=logits)
                new_logp = dist.log_prob(buf.act[mb])
                entropy = dist.entropy().mean()

                ratio = torch.exp(new_logp - buf.logp[mb])
                surr1 = ratio * adv[mb]
                surr2 = torch.clamp(ratio, 1 - cfg.clip_eps, 1 + cfg.clip_eps) * adv[mb]
                pi_loss = -torch.min(surr1, surr2).mean()

                # FIXED: both are (B,)
                v_loss = F.mse_loss(v.squeeze(-1), ret[mb])

                loss = pi_loss + cfg.vf_coef * v_loss - cfg.ent_coef * entropy

                # termination head supervised loss (only if enabled)
                if cfg.use_termination_head:
                    targets = buf.stop_target[mb]
                    weights = (buf.near_end[mb] > 0.5).float()
                    if weights.sum() < 1:
                        weights = torch.ones_like(weights)

                    stop_bce = F.binary_cross_entropy_with_logits(stop_logit, targets, reduction="none")
                    stop_loss = (stop_bce * weights).sum() / (weights.sum() + 1e-8)
                    stop_loss_last = float(stop_loss.item())

                    loss = loss + cfg.stop_coef * stop_loss

                opt.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
                opt.step()

        update += 1

        # -------- Logging --------
        train_Rsum = float(np.sum(R_acc))
        train_L = float(np.mean(L_acc)) if L_acc else 0.0
        train_turn = float(np.mean(T_acc)) if T_acc else 0.0
        train_off = float(np.mean(O_acc)) if O_acc else 0.0
        p_stop_mean = float(np.mean(pstop_all)) if pstop_all else 0.0
        p_stop_near = float(np.mean(pstop_ne)) if pstop_ne else 0.0

        row = {
            "update": update,
            "steps": total_steps,
            "train/Rsum": train_Rsum,
            "train/L": train_L,
            "train/turn": train_turn,
            "train/off": train_off,
            "train/stop_loss": stop_loss_last,
            "train/p_stop_mean": p_stop_mean,
            "train/p_stop_nearend": p_stop_near,
            "eval/success_rate": 0.0,
            "eval/mean_L": 0.0,
            "eval/stop_precision": 0.0,
            "eval/stop_attempt_rate": 0.0,
        }

        if (update % cfg.eval_every_updates) == 0:
            ev = evaluate(cfg, model, device)
            row.update(ev)

            # keep BEST + LAST only
            if ev["eval/success_rate"] > best_succ:
                best_succ = ev["eval/success_rate"]
                torch.save({"cfg": asdict(cfg), "model": model.state_dict(), "best": best_succ},
                           os.path.join(run_dir, "best.pt"))
                print(f"✓ New BEST saved: success={best_succ:.3f}, mean_L={ev['eval/mean_L']:.2f}")

            writer.add_scalar("eval/success_rate", ev["eval/success_rate"], update)
            writer.add_scalar("eval/mean_L", ev["eval/mean_L"], update)
            writer.add_scalar("eval/stop_precision", ev["eval/stop_precision"], update)
            writer.add_scalar("eval/stop_attempt_rate", ev["eval/stop_attempt_rate"], update)

        torch.save({"cfg": asdict(cfg), "model": model.state_dict(), "update": update, "steps": total_steps},
                   os.path.join(run_dir, "last.pt"))

        writer.add_scalar("train/Rsum", train_Rsum, update)
        writer.add_scalar("train/L", train_L, update)
        writer.add_scalar("train/turn", train_turn, update)
        writer.add_scalar("train/off", train_off, update)
        writer.add_scalar("train/stop_loss", stop_loss_last, update)
        writer.add_scalar("train/p_stop_mean", p_stop_mean, update)
        writer.add_scalar("train/p_stop_nearend", p_stop_near, update)

        csv_w.writerow(row)
        csv_f.flush()

        if update % 5 == 0:
            msg = (f"[upd {update:4d}] steps={total_steps:8d} "
                   f"Rsum={train_Rsum:8.2f} L={train_L:5.2f} turn={train_turn:5.2f} off={train_off:4.2f} "
                   f"| stop_loss={stop_loss_last:6.3f} pstop(mean)={p_stop_mean:5.2f} pstop(ne)={p_stop_near:5.2f}")
            if (update % cfg.eval_every_updates) == 0:
                msg += f" | eval succ={row['eval/success_rate']:.2f} eval L={row['eval/mean_L']:.2f}"
            print(msg)

    csv_f.close()
    writer.close()
    print("Done. Saved to:", run_dir)


def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument("--exp_name", type=str, default="B_termhead_v2")
    p.add_argument("--out_dir", type=str, default="runs_gpt_version_2")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default="cuda")

    p.add_argument("--use_distractors", type=int, default=1)
    p.add_argument("--use_stop_action", type=int, default=0)
    p.add_argument("--use_termination_head", type=int, default=1)

    p.add_argument("--step_alpha", type=float, default=1.0)

    p.add_argument("--w_turn", type=float, default=0.25)
    p.add_argument("--w_align", type=float, default=0.25)
    p.add_argument("--w_precision", type=float, default=2.0)
    p.add_argument("--w_progress", type=float, default=0.25)

    p.add_argument("--total_steps", type=int, default=1_200_000)
    p.add_argument("--rollout_steps", type=int, default=4096)

    p.add_argument("--stop_threshold", type=float, default=0.7)
    p.add_argument("--stop_coef", type=float, default=1.5)
    p.add_argument("--stop_radius", type=float, default=5.0)

    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    cfg = Cfg(
        exp_name=args.exp_name,
        out_dir=args.out_dir,
        seed=args.seed,
        device=args.device,
        use_distractors=args.use_distractors,
        use_stop_action=args.use_stop_action,
        use_termination_head=args.use_termination_head,
        step_alpha=args.step_alpha,
        w_turn=args.w_turn,
        w_align=args.w_align,
        w_precision=args.w_precision,
        w_progress=args.w_progress,
        total_steps=args.total_steps,
        rollout_steps=args.rollout_steps,
        stop_threshold=args.stop_threshold,
        stop_coef=args.stop_coef,
        stop_radius=args.stop_radius,
    )

    # safety: don't allow both at once
    if cfg.use_stop_action and cfg.use_termination_head:
        raise SystemExit("Set either --use_stop_action 1 OR --use_termination_head 1, not both.")

    train(cfg)

#!/usr/bin/env python3
"""
simple_rollout.py

Very small script:
- Creates a single synthetic curve with CurveEnv
- Loads a trained ActorCritic model
- Lets the model track the curve (deterministic: argmax)
- Saves:
    * final_trajectory.png      (always)
    * step_0000.png, ...        (optional: --save_steps)
"""

import os
import argparse
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Import your training components
from Synth_simple_v1_8_paper_version import (
    CurveEnv, ActorCritic,
    fixed_window_history, ACTIONS_8,
    DEVICE, set_seeds
)


# ---------- plotting helpers ----------

def plot_trajectory(img, gt_poly, path_points, out_path, title=None):
    """
    Draw:
      - gray background image
      - cyan ground-truth curve
      - red agent trajectory
    and save to out_path.
    """
    fig, ax = plt.subplots()

    ax.imshow(img, cmap="gray", vmin=0, vmax=1, origin="upper")

    # Ground-truth curve (cyan)
    if gt_poly is not None and len(gt_poly) > 1:
        ax.plot(gt_poly[:, 1], gt_poly[:, 0],
                lw=2, alpha=0.9, color="cyan", label="GT")

    # Agent trajectory (red)
    if path_points:
        py = [p[0] for p in path_points]
        px = [p[1] for p in path_points]
        ax.plot(px, py, lw=2, alpha=0.9, color="red", label="Agent")
        # Start (green) and end (yellow) markers
        ax.scatter([px[0]],  [py[0]],  s=25, color="lime")
        ax.scatter([px[-1]], [py[-1]], s=25, color="yellow")

    if title:
        ax.set_title(title, fontsize=8)

    ax.set_axis_off()
    ax.legend(loc="lower right")
    fig.tight_layout(pad=0)
    fig.savefig(out_path, bbox_inches="tight", pad_inches=0)
    plt.close(fig)


# ---------- rollout ----------

def run_one_rollout(weights, outdir, max_steps=400,
                    branches=False, save_steps=False, deterministic=True):
    os.makedirs(outdir, exist_ok=True)

    # 1) Recreate env and model exactly as in training
    set_seeds(123)
    env = CurveEnv(h=128, w=128, branches=branches, max_steps=max_steps)
    K = 8
    nA = len(ACTIONS_8)

    model = ActorCritic(n_actions=nA, K=K).to(DEVICE)
    state = torch.load(weights, map_location=DEVICE)
    model.load_state_dict(state)
    model.eval()
    print(f"[simple_rollout] Loaded weights: {weights}")

    # 2) Reset env → generates a single curve
    obs = env.reset()
    done = False
    ahist = []
    step = 0

    # For convenience
    img = env.ep.img
    gt_poly = env.ep.gt_poly

    # Optional: save the initial state with just the start point
    if save_steps:
        plot_trajectory(
            img, gt_poly, env.path_points,
            os.path.join(outdir, f"step_{step:04d}.png"),
            title=f"step {step} (start)"
        )

    # 3) Run the policy until termination
    with torch.no_grad():
        while not done and step < max_steps:
            x = torch.tensor(obs[None], dtype=torch.float32, device=DEVICE)
            A = fixed_window_history(ahist, K, nA)[None, ...]
            A_t = torch.tensor(A, dtype=torch.float32, device=DEVICE)

            logits, value, _ = model(x, A_t, None)

            if deterministic:
                # Greedy action
                action = int(torch.argmax(logits, dim=1).item())
            else:
                # Stochastic sampling
                from torch.distributions import Categorical
                dist = Categorical(logits=logits)
                action = int(dist.sample().item())

            obs, r, done, info = env.step(action)

            # Update action history (one-hot)
            a1h = np.zeros(nA, dtype=np.float32)
            a1h[action] = 1.0
            ahist.append(a1h)

            step += 1

            # (Optional) save step-by-step frame
            if save_steps:
                idx = info.get("idx", -1)
                ccs = info.get("ccs", 0.0)
                title = f"step {step} | a={action} | r={r:.3f} | idx={idx} | CCS={ccs:.3f}"
                out_step = os.path.join(outdir, f"step_{step:04d}.png")
                plot_trajectory(img, gt_poly, env.path_points, out_step, title=title)

    idx_end = info.get("idx", -1)
    L_end = info.get("L_local", 0.0)
    ccs_end = info.get("ccs", 0.0)

    print(
        f"[simple_rollout] Done in {step} steps. "
        f"idx_end={idx_end}, L_end={L_end:.3f}, CCS={ccs_end:.3f}"
    )

    # 4) Save a final summary image of the whole trajectory
    final_path = os.path.join(outdir, "final_trajectory.png")
    title = (
        f"Final | steps={step}, idx={idx_end}, "
        f"L_end={L_end:.3f}, CCS={ccs_end:.3f}"
    )
    plot_trajectory(img, gt_poly, env.path_points, final_path, title=title)
    print(f"[simple_rollout] Saved final image to {final_path}")



# ---------- CLI ----------

def main():
    ap = argparse.ArgumentParser(
        description="Minimal script to run a trained curve-tracking model "
                    "on a single synthetic curve and save trajectory images."
    )
    ap.add_argument("--weights", type=str, required=True,
                    help="Path to trained weights (ckpt_curveppo*.pth)")
    ap.add_argument("--outdir", type=str, default="simple_rollout_out",
                    help="Directory to save images")
    ap.add_argument("--max_steps", type=int, default=400,
                    help="Max steps in rollout (should match training)")
    ap.add_argument("--branches", action="store_true",
                    help="Use branched curves (same as training flag)")
    ap.add_argument("--save_steps", action="store_true",
                    help="If set, saves one PNG per step.")
    ap.add_argument("--stochastic", action="store_true",
                    help="If set, sample from policy instead of argmax.")
    args = ap.parse_args()

    run_one_rollout(
        weights=args.weights,
        outdir=args.outdir,
        max_steps=args.max_steps,
        branches=args.branches,
        save_steps=args.save_steps,
        deterministic=not args.stochastic,
    )


if __name__ == "__main__":
    main()

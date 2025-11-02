# evaluate_curveppo.py
# Evaluate a trained PPO curve follower on unseen curves and save visualizations.

import os
import argparse
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")  # headless-friendly
import matplotlib.pyplot as plt

from Synth_simple_v1_1 import ActorCritic, CurveEnv, fixed_window_history, ACTIONS_8, DEVICE

def render_frame(env, path, dpi=100):
    """Render a single frame and return an RGB np.uint8 image (H,W,3)."""
    img = env.ep.img
    gt  = env.ep.gt_poly
    H, W = img.shape
    # Tight figure that matches pixel dims nicely
    fig, ax = plt.subplots(figsize=(W/100, H/100), dpi=dpi)
    ax.imshow(img, cmap='gray', vmin=0, vmax=1, origin='upper')
    # GT in green
    ax.plot(gt[:,1], gt[:,0], lw=2, alpha=0.85, color='lime', label='GT')
    # Agent path in red + endpoints
    if len(path) > 1:
        py = [p[0] for p in path]
        px = [p[1] for p in path]
        ax.plot(px, py, lw=2, alpha=0.9, color='red', label='Path')
        ax.scatter([px[0]],[py[0]], s=20, color='cyan', label='start', zorder=5)
        ax.scatter([px[-1]],[py[-1]], s=20, color='yellow', label='end', zorder=5)
    ax.set_axis_off()
    ax.legend(loc='lower right', fontsize=6, framealpha=0.4)

    # Draw then extract RGBA buffer (portable across Matplotlib versions)
    fig.canvas.draw()
    width, height = fig.canvas.get_width_height()
    buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
    rgba = buf.reshape((height, width, 4))
    rgb = rgba[:, :, :3].copy()
    plt.close(fig)
    return rgb

def save_png(rgb, path_png):
    import imageio.v2 as iio
    iio.imwrite(path_png, rgb)

def save_gif(frames, path_gif, fps=20):
    if not frames:
        return
    import imageio.v2 as iio
    iio.mimsave(path_gif, frames, fps=fps)

def rollout_once(model, env, K=8, greedy=True, max_frames_gif=400):
    """Run one rollout and return (path, last_info, frames_list)."""
    model.eval()
    obs = env.reset()
    done = False
    ahist = []
    path = [env.agent]
    frames = []
    info = {}

    with torch.no_grad():
        steps = 0
        while not done:
            x = torch.tensor(obs[None], dtype=torch.float32, device=DEVICE)
            A = fixed_window_history(ahist, K, len(ACTIONS_8))[None, ...]
            A_t = torch.tensor(A, dtype=torch.float32, device=DEVICE)

            logits, value, _ = model(x, A_t, None)
            if greedy:
                action = int(torch.argmax(logits, dim=1).item())
            else:
                dist = torch.distributions.Categorical(logits=logits)
                action = int(dist.sample().item())

            obs, r, done, info = env.step(action)
            a1h = np.zeros(len(ACTIONS_8), dtype=np.float32); a1h[action] = 1.0
            ahist.append(a1h)

            path.append(env.agent)
            steps += 1

            if steps <= max_frames_gif:
                frames.append(render_frame(env, path))

    return path, info, frames

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", type=str, default="DSA/ckpt_curveppo.pth",
                    help="Path to trained weights .pth (use absolute or relative path).")
    ap.add_argument("--episodes", type=int, default=10, help="Number of unseen curves to evaluate.")
    ap.add_argument("--outdir", type=str, default="eval_out", help="Output directory for PNG/GIF.")
    ap.add_argument("--gif", action="store_true", help="Also save GIFs.")
    ap.add_argument("--greedy", action="store_true", help="Use argmax instead of sampling.")
    ap.add_argument("--branches", action="store_true", help="Generate curves with branches.")
    ap.add_argument("--h", type=int, default=128)
    ap.add_argument("--w", type=int, default=128)
    ap.add_argument("--max_steps", type=int, default=400)
    args = ap.parse_args()

    # 1) Verify weights path
    if not os.path.isfile(args.weights):
        raise FileNotFoundError(f"Weights not found: {args.weights}\n"
                                f"Tip: pass an absolute path or cd into the folder with the .pth")

    # 2) Build env/model the same way as during training
    env = CurveEnv(h=args.h, w=args.w, branches=args.branches, max_steps=args.max_steps)
    K = 8
    nA = len(ACTIONS_8)
    model = ActorCritic(n_actions=nA, K=K).to(DEVICE)

    # 3) Load weights (safe mode)
    state = torch.load(args.weights, map_location=DEVICE, weights_only=True) if "weights_only" in torch.load.__code__.co_varnames else torch.load(args.weights, map_location=DEVICE)
    model.load_state_dict(state)
    print(f"Loaded weights: {args.weights}")

    os.makedirs(args.outdir, exist_ok=True)

    # 4) Evaluate
    for i in range(1, args.episodes+1):
        path, info_last, frames = rollout_once(model, env, K=K, greedy=args.greedy)
        # Save final overlay PNG
        final_rgb = render_frame(env, path)
        png_path = os.path.join(args.outdir, f"rollout_{i:03d}.png")
        save_png(final_rgb, png_path)
        # Optional GIF
        if args.gif:
            gif_path = os.path.join(args.outdir, f"rollout_{i:03d}.gif")
            save_gif(frames, gif_path)

        reached = info_last.get("reached_end", False)
        stalled = info_last.get("stalled", False)
        timeout = info_last.get("timeout", False)
        L_end  = info_last.get("L_local", np.nan)
        idx    = info_last.get("idx", -1)

        print(f"[Eval {i:03d}] reached_end={reached} stalled={stalled} timeout={timeout} "
              f"L_local_end={L_end:.3f} idx_end={idx} -> saved {png_path}")

if __name__ == "__main__":
    main()

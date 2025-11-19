import torch
import numpy as np
import argparse
# Import your Model and PPO Logic
from Synth_simple_v1_9_paper_version_gemini import CurveEnv, AsymmetricActorCritic, update_ppo, DEVICE, fixed_window_history, ACTIONS_8

# Import the NEW Noisy Generator
from Curve_Generator_noisy import CurveMaker

def train_dsa_adapt(args):
    print("--- STARTING DSA DOMAIN ADAPTATION ---")
    
    # 1. Override the Environment to use the Noisy Generator
    # We subclass CurveEnv to swap the generator easily
    class CurveEnvDSA(CurveEnv):
        def __init__(self, h=128, w=128, branches=False):
            super().__init__(h, w, branches)
            # SWAP THE GENERATOR HERE
            self.cm = CurveMaker(h=h, w=w, thickness=1.5, seed=None)
            
    env = CurveEnvDSA(h=128, w=128, branches=args.branches)
    
    # 2. Initialize Model
    K = 8
    nA = len(ACTIONS_8)
    model = AsymmetricActorCritic(n_actions=nA, K=K).to(DEVICE)
    
    # 3. LOAD EXPERT WEIGHTS (The Clean Model)
    load_path = args.load_path
    try:
        model.load_state_dict(torch.load(load_path, map_location=DEVICE))
        print(f"Loaded Expert Weights from: {load_path}")
    except FileNotFoundError:
        print(f"ERROR: Could not find {load_path}. Please train the clean model first!")
        return

    # 4. LOW LEARNING RATE (Fine-Tuning)
    # We use 1e-5 to gently adjust the CNN filters to handle noise
    opt = torch.optim.Adam(model.parameters(), lr=1e-4)
    
    # 5. Training Loop (Same as before, but focused on Adaptation)
    BATCH_SIZE_EPISODES = 16 # Larger batch for noisy data stability
    batch_buffer = []
    ep_returns = []
    
    for ep in range(1, args.episodes + 1):
        obs_dict = env.reset()
        done = False
        ahist = []
        ep_traj = {"obs":{'actor':[], 'critic_gt':[]}, "ahist":[], "act":[], "logp":[], "val":[], "rew":[]}
        ep_ret = 0
        
        while not done:
            obs_a = torch.tensor(obs_dict['actor'][None], dtype=torch.float32, device=DEVICE)
            obs_c = torch.tensor(obs_dict['critic_gt'][None], dtype=torch.float32, device=DEVICE)
            A = fixed_window_history(ahist, K, nA)[None, ...]
            A_t = torch.tensor(A, dtype=torch.float32, device=DEVICE)

            with torch.no_grad():
                logits, value, _, _ = model(obs_a, obs_c, A_t)
                logits = torch.clamp(logits, -20, 20)
                dist = torch.distributions.Categorical(logits=logits)
                action = dist.sample().item()
                logp = dist.log_prob(torch.tensor(action, device=DEVICE)).item()
                val = value.item()

            next_obs, r, done, info = env.step(action)

            ep_traj["obs"]['actor'].append(obs_dict['actor'])
            ep_traj["obs"]['critic_gt'].append(obs_dict['critic_gt'])
            ep_traj["ahist"].append(A[0])
            ep_traj["act"].append(action)
            ep_traj["logp"].append(logp)
            ep_traj["val"].append(val)
            ep_traj["rew"].append(r)
            
            a_onehot = np.zeros(nA); a_onehot[action] = 1.0
            ahist.append(a_onehot)
            obs_dict = next_obs
            ep_ret += r

        ep_returns.append(ep_ret)
        
        # GAE Calculation & Buffering
        if len(ep_traj["rew"]) > 2:
            rews = np.array(ep_traj["rew"])
            vals = np.array(ep_traj["val"] + [0.0])
            delta = rews + 0.9 * vals[1:] - vals[:-1]
            adv = np.zeros_like(rews)
            acc = 0
            for t in reversed(range(len(rews))):
                acc = delta[t] + 0.9 * 0.95 * acc
                adv[t] = acc
            ret = adv + vals[:-1]
            
            final_ep_data = {
                "obs": {
                    "actor": np.array(ep_traj["obs"]['actor']),
                    "critic_gt": np.array(ep_traj["obs"]['critic_gt'])
                },
                "ahist": np.array(ep_traj["ahist"]),
                "act": np.array(ep_traj["act"]),
                "logp": np.array(ep_traj["logp"]),
                "adv": adv,
                "ret": ret
            }
            batch_buffer.append(final_ep_data)

        if len(batch_buffer) >= BATCH_SIZE_EPISODES:
            update_ppo(opt, model, batch_buffer)
            batch_buffer = []

        # Logging
        if ep % 100 == 0:
            avg_r = np.mean(ep_returns[-100:])
            # Check success (approximate via reward magnitude)
            # If reward > 50, likely reached end
            success_count = sum(1 for r in ep_returns[-100:] if r > 50)
            print(f"Adaptation Ep {ep} | Avg Rew: {avg_r:.2f} | Approx Success: {success_count/100:.2f}")
            
        # Checkpoint
        if ep % 1000 == 0:
            save_name = f"ppo_dsa_adapt_ep{ep}.pth"
            torch.save(model.state_dict(), save_name)
            print(f"Saved checkpoint: {save_name}")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--episodes", type=int, default=20000)
    p.add_argument("--load_path", type=str, default="ppo_curve_agent.pth", help="Path to CLEAN model")
    p.add_argument("--branches", action="store_true")
    args = p.parse_args()
    
    train_dsa_adapt(args)
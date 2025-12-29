#!/usr/bin/env python3
"""
train_rl_dsa_B_termhead_v2.py

Version B (termination head) — FIXED
✔ Binary stop decision
✔ Terminal reward
✔ Smoothness penalty
✔ Curriculum stop learning
✔ Clean logging + checkpoints
"""

import os, time, json, csv, argparse, math
from dataclasses import dataclass, asdict
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from torch.utils.tensorboard import SummaryWriter

from Curve_Generator_Flexible_For_Ciruculum_learning import CurveMakerFlexible

# ============================================================
# Utils
# ============================================================

ACTIONS_8 = [(-1,0),(-1,1),(0,1),(1,1),(1,0),(1,-1),(0,-1),(-1,-1)]

def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)

def clamp(x, lo, hi):
    return lo if x < lo else hi if x > hi else x

def cosine(a,b,eps=1e-8):
    return float(np.dot(a,b)/(np.linalg.norm(a)*np.linalg.norm(b)+eps))

# ============================================================
# Config
# ============================================================

@dataclass
class Cfg:
    exp_name: str
    out_dir: str = "runs_gpt_version_3"
    seed: int = 0
    device: str = "cuda"

    H: int = 128
    W: int = 128
    crop: int = 32
    max_steps: int = 180

    step_alpha: float = 1.0
    stop_radius: float = 5.0

    # rewards
    w_precision: float = 2.0
    w_progress: float = 0.3
    w_align: float = 0.3
    w_turn: float = 0.3
    w_smooth: float = 0.5
    w_step: float = 0.01

    r_stop_correct: float = 25.0
    r_stop_wrong: float = -5.0

    # PPO
    total_steps: int = 1_200_000
    rollout_steps: int = 4096
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_eps: float = 0.2
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    stop_coef: float = 2.0
    lr: float = 3e-4
    update_epochs: int = 6
    minibatch: int = 512

    # curriculum
    stop_warmup_updates: int = 40

    eval_every_updates: int = 10
    eval_episodes: int = 40


# ============================================================
# Environment
# ============================================================

class Env:
    def __init__(self,cfg):
        self.cfg = cfg
        self.gen = CurveMakerFlexible(cfg.H,cfg.W,cfg.seed)

    def reset(self):
        img,mask,pts = self.gen.sample_with_distractors()
        self.img = img.astype(np.float32)
        self.poly = pts[0].astype(np.float32)
        self.end = self.poly[-1]

        self.idx = 3
        self.agent = self.poly[self.idx].copy()
        self.prev = [self.agent.copy()]*3
        self.steps = 0
        return self.obs()

    def obs(self):
        y,x = self.agent
        crop = self.img[int(y-16):int(y+16),int(x-16):int(x+16)]
        if crop.shape!=(32,32):
            crop = np.zeros((32,32),np.float32)
        return crop[None]

    def step(self,action):
        dy,dx = ACTIONS_8[action]
        self.agent[0]=clamp(self.agent[0]+dy,0,self.cfg.H-1)
        self.agent[1]=clamp(self.agent[1]+dx,0,self.cfg.W-1)

        self.prev.append(self.agent.copy())
        self.prev=self.prev[-3:]
        self.steps+=1

        # metrics
        dists = np.linalg.norm(self.poly-self.agent,axis=1)
        new_idx = int(np.argmin(dists))
        progress = max(0,new_idx-self.idx)
        self.idx=max(self.idx,new_idx)

        precision = math.exp(-(dists[new_idx]**2)/2)
        gt_vec = self.poly[min(len(self.poly)-1,new_idx+2)]-self.poly[max(0,new_idx-2)]
        mv = self.prev[-1]-self.prev[-2]
        align = max(0,cosine(gt_vec,mv))

        turn = 1-cosine(self.prev[-1]-self.prev[-2],self.prev[-2]-self.prev[-3])
        smooth = turn

        reward = (
            self.cfg.w_precision*precision +
            self.cfg.w_progress*progress +
            self.cfg.w_align*align -
            self.cfg.w_turn*turn -
            self.cfg.w_smooth*smooth -
            self.cfg.w_step
        )

        d_end = np.linalg.norm(self.agent-self.end)

        done=False
        info={"d_end":d_end}

        if self.steps>=self.cfg.max_steps:
            done=True

        return self.obs(),reward,done,info


# ============================================================
# Model
# ============================================================

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn=nn.Sequential(
            nn.Conv2d(1,32,3,2,1),nn.ReLU(),
            nn.Conv2d(32,64,3,2,1),nn.ReLU(),
            nn.Flatten()
        )
        self.fc=nn.Linear(64*8*8,256)
        self.pi=nn.Linear(256,8)
        self.v=nn.Linear(256,1)
        self.stop=nn.Linear(256,1)

    def forward(self,x):
        h=F.relu(self.fc(self.cnn(x)))
        return self.pi(h),self.v(h).squeeze(-1),self.stop(h).squeeze(-1)


# ============================================================
# Training
# ============================================================

def train(cfg):
    set_seed(cfg.seed)
    device=torch.device(cfg.device if torch.cuda.is_available() else "cpu")

    run_dir=os.path.join(cfg.out_dir,cfg.exp_name)
    ckpt_dir=os.path.join(run_dir,"checkpoints")
    os.makedirs(ckpt_dir,exist_ok=True)

    with open(os.path.join(run_dir,"config.json"),"w") as f:
        json.dump(asdict(cfg),f,indent=2)

    writer=SummaryWriter(os.path.join(run_dir,"tb"))
    env=Env(cfg)
    net=Net().to(device)
    opt=torch.optim.Adam(net.parameters(),lr=cfg.lr)

    obs=env.reset()
    total_steps=0
    update=0
    best_succ=0.0

    while total_steps<cfg.total_steps:
        update+=1
        obs_buf,act_buf,logp_buf,val_buf,rew_buf,done_buf,stop_tgt=[],[],[],[],[],[],[]

        for _ in range(cfg.rollout_steps):
            o=torch.tensor(obs[None],device=device)
            logits,val,stop_logit=net(o)
            dist=Categorical(logits=logits)
            act=dist.sample()
            logp=dist.log_prob(act)

            obs2,r,done,info=env.step(act.item())

            # binary stop label
            stop_target=float(info["d_end"]<cfg.stop_radius)

            obs_buf.append(o.squeeze(0))
            act_buf.append(act)
            logp_buf.append(logp)
            val_buf.append(val)
            rew_buf.append(r)
            done_buf.append(done)
            stop_tgt.append(stop_target)

            obs=obs2
            total_steps+=1
            if done:
                obs=env.reset()

        # convert
        obs_buf=torch.stack(obs_buf)
        act_buf=torch.stack(act_buf)
        logp_buf=torch.stack(logp_buf)
        val_buf=torch.stack(val_buf)
        rew_buf=torch.tensor(rew_buf,device=device)
        stop_tgt=torch.tensor(stop_tgt,device=device)

        # advantages
        adv=rew_buf-val_buf.detach()
        adv=(adv-adv.mean())/(adv.std()+1e-8)
        ret=adv+val_buf

        # PPO update
        for _ in range(cfg.update_epochs):
            logits,v,stop_logit=net(obs_buf)
            dist=Categorical(logits=logits)
            new_logp=dist.log_prob(act_buf)
            ratio=torch.exp(new_logp-logp_buf)

            pi_loss=-torch.min(
                ratio*adv,
                torch.clamp(ratio,1-cfg.clip_eps,1+cfg.clip_eps)*adv
            ).mean()

            v_loss=F.mse_loss(v,ret)

            # curriculum: stop head off early
            if update>cfg.stop_warmup_updates:
                stop_loss=F.binary_cross_entropy_with_logits(stop_logit,stop_tgt)
            else:
                stop_loss=torch.tensor(0.0,device=device)

            loss=pi_loss+cfg.vf_coef*v_loss+cfg.stop_coef*stop_loss-cfg.ent_coef*dist.entropy().mean()

            opt.zero_grad()
            loss.backward()
            opt.step()

        # save last
        torch.save({"model":net.state_dict()},os.path.join(ckpt_dir,"last.pt"))

        if update%cfg.eval_every_updates==0:
            succ=evaluate(cfg,net,device)
            if succ>best_succ:
                best_succ=succ
                torch.save({"model":net.state_dict()},os.path.join(ckpt_dir,"best.pt"))
                print(f"✓ New BEST success={best_succ:.2f}")

        if update%5==0:
            print(f"[upd {update:4d}] steps={total_steps}")

    torch.save({"model":net.state_dict()},os.path.join(run_dir,"final.pt"))
    print("Training done:",run_dir)


# ============================================================
# Evaluation
# ============================================================

@torch.no_grad()
def evaluate(cfg,net,device):
    env=Env(cfg)
    net.eval()
    success=0
    for _ in range(cfg.eval_episodes):
        obs=env.reset()
        idx_hist=[]
        for _ in range(cfg.max_steps):
            o=torch.tensor(obs[None],device=device)
            logits,v,stop_logit=net(o)
            act=torch.argmax(logits,dim=-1).item()
            p_stop=torch.sigmoid(stop_logit).item()
            obs,r,done,info=env.step(act)
            idx_hist.append(info["d_end"])
            if p_stop>0.8 and len(idx_hist)>10 and idx_hist[-1]>idx_hist[-10]:
                if info["d_end"]<cfg.stop_radius:
                    success+=1
                break
            if done:
                break
    net.train()
    return success/cfg.eval_episodes


# ============================================================
# Main
# ============================================================

if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--exp_name",required=True)
    ap.add_argument("--seed",type=int,default=0)
    args=ap.parse_args()

    cfg=Cfg(exp_name=args.exp_name,seed=args.seed)
    train(cfg)

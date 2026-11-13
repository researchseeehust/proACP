"""Stage 2, offline: the contact planner (:mod:`proacp.models.planner`) learned from a frozen Stage 1 tracker.

The tracker (a proacp.train.stage1 checkpoint) runs in the simulator with its training randomization, action
noise and reference contact targets, from uniform start phases (the paper's Stage 2 episode starts). The replay buffer
of simulator transitions records the episodes; finished ones are labeled in hindsight
(:func:`proacp.models.planner.label_episodes`) into the planner's buffer, and the planner trains on it with the paper's
Table S5 settings. The planner sees the exact height scan of the episode's terrain variant (plus noise). Every 10th clip
of each motion folder is held out: its samples only go to a validation buffer, on which ``planner_val.csv`` reports
per-body occupancy precision / recall / F1, the median top-1 error and the share of top-1 points within 0.1 m.

    python -m proacp.train.planner_offline --tracker logs/acp_stage1/ckpt_060000.pt --log_dir logs/acp_planner
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import time

parser = argparse.ArgumentParser()
parser.add_argument("--tracker", type=str, required=True, help="proacp.train.stage1 checkpoint")
parser.add_argument("--num_envs", type=int, default=4096)
parser.add_argument("--steps", type=int, default=20_000, help="simulator steps")
parser.add_argument("--updates_per_step", type=int, default=1, help="planner updates per simulator step")
parser.add_argument("--buffer_steps", type=int, default=1350, help="replay rows per environment: at least the longest clip")
parser.add_argument("--capacity", type=int, default=1_000_000, help="planner replay samples (validation: a fifth)")
parser.add_argument("--action_noise", type=float, default=0.1)
parser.add_argument("--val_samples", type=int, default=16384)
parser.add_argument("--goal_radius", type=float, default=0.3, help="planner goal d_xy / max(|d_xy|, r)")
parser.add_argument("--scan_noise", type=float, default=0.03, help="Gaussian noise (m) on the planner's height scan")
parser.add_argument("--start_max", type=float, default=0.1, help="success filter: episodes started within this fraction of their clip")
parser.add_argument("--reach_min", type=float, default=0.9, help="success filter: and reached at least this fraction")
parser.add_argument("--log_every", type=int, default=100)
parser.add_argument("--val_every", type=int, default=500)
parser.add_argument("--ckpt_every", type=int, default=2000)
parser.add_argument("--log_dir", type=str, required=True)
parser.add_argument("--seed", type=int, default=0)
args = parser.parse_args()

import torch

import proacp.models.networks as networks
from proacp.train.buffer import ReplayBuffer
from proacp.sim.env import TrackingSim
from proacp.robot import CONTACT_BODIES, MOTION_DIR, default_joint_pos_list
from proacp.sim.kinematics import Kinematics
from proacp.tasks.tracking import MDP, OBS_DIM, REF_ORI_DIMS
from proacp.clips.library import MotionLib
from proacp.models.networks import Policy
from proacp.models.planner import ContactPlanner, PlannerBuffer, PlannerCfg, PlannerScan, label_episodes, planner_metrics, planner_update
from proacp.tasks.contact import ContactCfg, ContactTask

torch.backends.cuda.matmul.allow_tf32 = True
torch.manual_seed(args.seed)
dev = "cuda:0"
os.makedirs(args.log_dir, exist_ok=True)
json.dump(vars(args), open(os.path.join(args.log_dir, "args.json"), "w"), indent=2)

ck = torch.load(args.tracker, map_location=dev)
ta = ck["args"]
motion_dir = ta.get("motion_dir") or MOTION_DIR
kin = Kinematics(dev)
motion = MotionLib.from_dir(motion_dir, kin, dev)
mdp = MDP(kin, motion, torch.tensor(default_joint_pos_list(), device=dev))
ctc = ContactTask(ContactCfg(**ck["ctc_cfg"]), mdp, motion_dir, dev)
Hh = ta["hist"]
networks.AMP_DTYPE = torch.bfloat16 if ta.get("amp", "none") == "bf16" else None
pi = Policy(OBS_DIM, ctc.cmd_dim, hist=Hh).to(dev)
pi.load_state_dict(ck["pi"])
pi.eval()

pcfg = PlannerCfg(capacity=args.capacity, goal_radius=args.goal_radius, scan_noise=args.scan_noise, start_max=args.start_max,
                 reach_min=args.reach_min)
scan = PlannerScan(pcfg, dev)
planner = ContactPlanner(OBS_DIM - REF_ORI_DIMS, pcfg).to(dev)
opt = torch.optim.Adam(planner.parameters(), lr=pcfg.lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0)
pbuf = PlannerBuffer(pcfg.capacity, pcfg.hist, planner.prop_dim, scan.num_z, dev)
vbuf = PlannerBuffer(args.capacity // 5, pcfg.hist, planner.prop_dim, scan.num_z, dev)
folders = sorted({n.split("/")[0] for n in motion.names})
val_clip = torch.zeros(len(motion.names), dtype=torch.bool, device=dev)
for g in folders:  # every 10th clip of each folder
    idx = [i for i, n in enumerate(motion.names) if n.split("/")[0] == g]
    val_clip[idx[::10]] = True
assert args.buffer_steps > int(motion.clip_len.max()), "the replay buffer must hold a whole episode"

N = args.num_envs
env = TrackingSim(N, motion, mdp, dev, start_phase=(0.0, 0.98))
buffer = ReplayBuffer(args.buffer_steps, N, dev)
obs_hist = torch.zeros(N, Hh, OBS_DIM, device=dev)
goal_steps = ctc.sample_goal_steps(N)

BODIES = list(CONTACT_BODIES)
log_f = open(os.path.join(args.log_dir, "train_log.csv"), "w")
log_f.write("step,wall_h,steps_per_s,ended,admitted,admit_rate,labeled,buffer,val_buffer,aff,occ\n")
val_f = open(os.path.join(args.log_dir, "planner_val.csv"), "w")
val_f.write("step,aff,occ," + ",".join(f"{m}_{b}" for m in ("precision", "recall", "f1", "median_err", "within_0.1") for b in BODIES) + "\n")
t0 = t_log = time.time()
acc = {"ended": 0, "admitted": 0, "labeled": 0, "loss": []}


def save(step: int):
    torch.save({"planner": planner.state_dict(), "opt": opt.state_dict(), "planner_cfg": dataclasses.asdict(pcfg), "step": step,
                "tracker": args.tracker, "prop_dim": planner.prop_dim, "args": vars(args)},
               os.path.join(args.log_dir, f"planner_{step:06d}.pt"))  # fmt: skip


for step in range(1, args.steps + 1):
    with torch.no_grad():
        o = env.observe()
        fresh = env.ep_step == 0
        obs_hist = torch.where(fresh[:, None, None], o.unsqueeze(1).expand_as(obs_hist), torch.cat((obs_hist[:, 1:], o.unsqueeze(1)), 1))
        a = pi(obs_hist, ctc.command(env.state(), env.f, env.v, goal_steps, noise=True))
        out = env.step(torch.clamp(a + args.action_noise * torch.randn_like(a), -20, 20))
        out["goal_steps"] = goal_steps.clone()
        buffer.add(out)
        done = out["done"]
        envs = done.nonzero(as_tuple=False).squeeze(-1)
        if len(envs) > 0:
            row = torch.full_like(envs, (buffer.ptr - 1) % buffer.T)
            d, n_adm = label_episodes(buffer, envs, row, out["ep_step"][envs] + 1, ctc, scan, pcfg)
            acc["ended"] += len(envs)
            acc["admitted"] += n_adm
            if d is not None:
                val = val_clip[d["clip"]]
                acc["labeled"] += int((~val).sum())
                if (~val).any():  # (the normalizer turns NaN on an empty batch)
                    planner.prop_norm.update(d["prop"][~val].float())
                pbuf.add({k: v[~val] for k, v in d.items()})
                vbuf.add({k: v[val] for k, v in d.items()})
            goal_steps[done] = ctc.sample_goal_steps(len(envs))
    if pbuf.size >= pcfg.min_entries:
        for _ in range(args.updates_per_step):
            acc["loss"].append(planner_update(planner, opt, pbuf, scan, pcfg))
    if step % args.log_every == 0:
        sps = args.log_every / (time.time() - t_log)
        t_log = time.time()
        loss = torch.stack(acc["loss"]).mean(0).tolist() if acc["loss"] else [float("nan")] * 2
        row = [step, (time.time() - t0) / 3600, sps, acc["ended"], acc["admitted"], acc["admitted"] / max(acc["ended"], 1), acc["labeled"],
               pbuf.size, vbuf.size, *loss]  # fmt: skip
        log_f.write(",".join(f"{x:.6g}" if isinstance(x, float) else str(x) for x in row) + "\n")
        log_f.flush()
        print(f"step {step} | {row[1]:.3f} h | {sps:.1f} steps/s | ended {acc['ended']} admitted {acc['admitted']} ({row[5]:.3f}) "
              f"labeled {acc['labeled']} | buffer {pbuf.size} val {vbuf.size} | aff {loss[0]:.3f} occ {loss[1]:.3f}", flush=True)  # fmt: skip
        acc = {"ended": 0, "admitted": 0, "labeled": 0, "loss": []}
    if step % args.val_every == 0 and vbuf.size > 0:
        planner.eval()
        mt = planner_metrics(planner, vbuf.sample(min(args.val_samples, vbuf.size)), scan, pcfg)
        planner.train()
        keys = ("precision", "recall", "f1", "median_err", "within_0.1")
        val_f.write(",".join(str(x) for x in [step, mt["aff"], mt["occ"]] + [v for k in keys for v in mt[k]]) + "\n")
        val_f.flush()
        print("  val " + " | ".join(f"{b} F1 {mt['f1'][i]:.2f} err {mt['median_err'][i]:.3f} <0.1 {mt['within_0.1'][i]:.2f}"
                                    for i, b in enumerate(BODIES[:6])), flush=True)  # fmt: skip
    if step % args.ckpt_every == 0 or step == args.steps:
        save(step)

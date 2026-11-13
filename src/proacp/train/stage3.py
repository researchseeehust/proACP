"""Stage 3 of the contact paper on SuperMimic: reference following and goal reaching on procedural terrains, in one run.

The Stage 2 tracker and contact planner (``--init``: a proacp.train.stage2 last.pt) keep co-training on two
kinds of environment at once (the paper's Fig. 3):

- reference-following environments (the first ``1 - --goal_frac``): the clips, every episode from the clip's first
  frame, the tracking reward with the planned-contact term in place of the reference-contact terms (eta_rew = 1) and the
  reference-anchored termination;
- goal-reaching environments (the last ``--goal_frac``): the procedurally generated terrains of
  :mod:`proacp.tasks.goal` (stairs, walled stairs, platforms, climbs, gaps, stepping stones, beams, pallets; 10 levels
  under the paper's curriculum), the goal-reaching reward, falls the only early termination, at most 15 s.

Both groups follow the planner's contact targets only (eta_obs = 1), and their goal command is the paper's bounded
direction and relative heading (radius 0.3 m toward the reference ahead, 0.5 m toward the terrain goal). The planner
learns from hindsight labels of both groups' successful episodes (reference: started in the first 10% of the clip and
reached 90%; goal: came within 0.5 m horizontally and 0.1 m vertically of the goal, labeled up to the first step within
0.2 m), in a reference buffer and a goal buffer partitioned by terrain type with balanced sampling; a planner batch draws
from the two in proportion to their sizes.

Differences from the paper's PPO setup:
- Two critics as in the paper, V_m (reference: the Stage 2 critic, with (eta_obs, eta_rew) = (1, 1) among its inputs)
  and V_fs (goal: new, fixed learning rate 1e-3), each bootstrapping its own group's imagined return. For the first
  ``--critic_warmup`` iterations goal samples fit V_fs but stay out of the actor loss (the paper: 500 PPO iterations).
- The paper normalizes advantages per group. There are no advantages here: each group's imagined return is divided by
  a running estimate of its own spread (EMA of its batch standard deviation, ``--scale_ema``) before the actor loss.
- The goal reward's indicators are sigmoids; stumbling (contact forces) is left out; the quality-based eviction of the
  goal buffer is FIFO per terrain type.

Usage: python -m proacp.train.stage3 --init logs/acp_stage2/last.pt --log_dir logs/acp_stage3
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import os
import time

parser = argparse.ArgumentParser()
parser.add_argument("--init", type=str, default=None, help="proacp.train.stage2 last.pt")
parser.add_argument("--num_envs", type=int, default=4096)
parser.add_argument("--goal_frac", type=float, default=0.5, help="share of goal-reaching environments and imagination samples")
parser.add_argument("--iterations", type=int, default=30_000)
parser.add_argument("--buffer_steps", type=int, default=1350, help="replay rows per environment: at least the longest episode")
parser.add_argument("--utd", type=float, default=0.4, help="updates per vectorized simulator step")
parser.add_argument("--horizon", type=int, default=32)
parser.add_argument("--wm_window", type=int, default=8)
parser.add_argument("--wm_batches", type=int, default=10)
parser.add_argument("--wm_batch", type=int, default=2048)
parser.add_argument("--pi_batch", type=int, default=4096)
parser.add_argument("--lr", type=float, default=2e-4)
parser.add_argument("--lr_fs", type=float, default=1e-3, help="the goal-reaching critic's fixed learning rate (paper)")
parser.add_argument("--gamma", type=float, default=0.99)
parser.add_argument("--lam", type=float, default=0.95)
parser.add_argument("--action_noise", type=float, default=0.1)
parser.add_argument("--critic_epochs", type=int, default=5)
parser.add_argument("--critic_minibatches", type=int, default=4)
parser.add_argument("--critic_warmup", type=int, default=500, help="iterations before the goal group enters the actor loss")
parser.add_argument("--scale_ema", type=float, default=0.99, help="EMA rate of each group's return spread (the actor-loss scale)")
parser.add_argument("--warmup_steps", type=int, default=64, help="sim steps before any update")
parser.add_argument("--lr_decay_iters", type=int, default=250_000)
parser.add_argument("--planner_updates", type=int, default=2, help="planner updates per iteration")
parser.add_argument("--planner_capacity", type=int, default=300_000, help="per group (reference, goal)")
parser.add_argument("--term_window", type=float, default=0.0, help="s: reference termination window either side of the frame")
parser.add_argument("--terrain_instances", type=int, default=16, help="random courses per terrain type and level")
parser.add_argument("--terrain_seed", type=int, default=0)
parser.add_argument("--start_level", type=int, default=0, help="goal environments start at a uniform level up to this")
parser.add_argument("--ckpt_every", type=int, default=2000)
parser.add_argument("--log_every", type=int, default=50)
parser.add_argument("--val_every", type=int, default=500)
parser.add_argument("--log_dir", type=str, required=True)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--resume", type=str, default=None)
parser.add_argument("--amp", type=str, default="bf16", choices=["none", "bf16"], help="autocast dtype for MLP matmuls")
args = parser.parse_args()
if not (args.init or args.resume):
    parser.error("--init (or --resume) is required")
if not 0 < args.goal_frac < 1:
    parser.error("--goal_frac in (0, 1)")

import torch
import torch.nn as nn
import warp as wp

import proacp.models.networks as networks
from proacp.train.buffer import ReplayBuffer
from proacp.sim.env import TrackingSim
from proacp.robot import CONTACT_BODIES, MOTION_DIR, NB, NUM_J, default_joint_pos_list
from proacp.tasks.goal import LEVELS, TERRAIN_TYPES, GoalCfg, GoalCurriculum, GoalTask, make_terrains
from proacp.sim.kinematics import Kinematics
from proacp.tasks.tracking import INV_DIM, MAX_STEP_REWARD, MDP, OBS_DIM, REF_ORI_DIMS, XI_DIM, RewardCfg, normalize_xi, unflatten_state
from proacp.clips.library import CONTROL_DT, MotionLib
from proacp.models.networks import Critic, Policy, WorldModel
from proacp.models.planner import (ContactPlanner, GroupedPlannerBuffer, MixedPlannerBuffer, PlannerBuffer, PlannerCfg, PlannerScan,
                                    label_episodes, plan, planner_metrics, planner_update)  # fmt: skip
from proacp.tasks.contact import ContactCfg, ContactTask, PaperReward, PaperRewardCfg, to_base

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.manual_seed(args.seed)
networks.AMP_DTYPE = torch.bfloat16 if args.amp == "bf16" else None
dev = "cuda:0"
os.makedirs(args.log_dir, exist_ok=True)
json.dump(vars(args), open(os.path.join(args.log_dir, "args.json"), "w"), indent=2)

ck = torch.load(args.resume or args.init, map_location=dev)
ia = ck["init_args"]  # the Stage 1 run's arguments (Stage 2 and 3 checkpoints carry them)
motion_dir = ia.get("motion_dir") or MOTION_DIR
kin = Kinematics(dev)
motion = MotionLib.from_dir(motion_dir, kin, dev)
mdp = MDP(kin, motion, torch.tensor(default_joint_pos_list(), device=dev), RewardCfg(term_window=int(round(args.term_window / CONTROL_DT))))
ctc_cfg = ContactCfg(**ck["ctc_cfg"])
ctc = ContactTask(ctc_cfg, mdp, motion_dir, dev)
reward_cfg = PaperRewardCfg(**ck["reward_cfg"])
paper_reward = PaperReward(reward_cfg, ctc)
pcfg = PlannerCfg(**{**ck["planner_cfg"], "capacity": args.planner_capacity})
scan = PlannerScan(pcfg, dev)
terrain_args = ck["terrain_args"] if args.resume else {"instances": args.terrain_instances, "seed": args.terrain_seed}
terrains = make_terrains(**terrain_args)
motion.attach_terrains(terrains["boxes"])  # terrain index v >= V: the goal-reaching bank
gcfg = GoalCfg()
goal = GoalTask(gcfg, ctc, paper_reward, terrains)
Hh, HZ = ia["hist"], args.horizon
MAX_R = {"ref": reward_cfg.max_step_reward, "goal": gcfg.max_step_reward}
CRITIC_EXTRA = NB * 3 + 2  # V_m: the planner's targets (base frame) and (eta_obs, eta_rew), as in Stage 2
ETAS = torch.ones(2, device=dev)

N = args.num_envs
N_GOAL = int(round(N * args.goal_frac))
N_REF = N - N_GOAL
B = args.pi_batch
B_GOAL = int(round(B * args.goal_frac))
B_REF = B - B_GOAL
REF_ENVS, GOAL_ENVS = torch.arange(N_REF, device=dev), torch.arange(N_REF, N, device=dev)

wm = WorldModel(mdp.scan_dim).to(dev)
pi = Policy(OBS_DIM, ctc.cmd_dim, hist=Hh).to(dev)
critic_m = Critic(INV_DIM + mdp.critic_extra + ctc.cmd_dim + CRITIC_EXTRA).to(dev)
critic_fs = Critic(INV_DIM + XI_DIM + mdp.scan_dim + goal.critic_dim + NB * 3).to(dev)
planner = ContactPlanner(OBS_DIM - REF_ORI_DIMS, pcfg).to(dev)
opt_wm = torch.optim.Adam(wm.parameters(), lr=args.lr)
opt_pi = torch.optim.Adam(pi.parameters(), lr=args.lr, betas=(0.9, 0.95))
opt_vm = torch.optim.Adam(critic_m.parameters(), lr=args.lr)
opt_vfs = torch.optim.Adam(critic_fs.parameters(), lr=args.lr_fs)
opt_pl = torch.optim.Adam(planner.parameters(), lr=pcfg.lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0)
wm.load_state_dict(ck["wm"]), pi.load_state_dict(ck["pi"]), planner.load_state_dict(ck["planner"])
opt_wm.load_state_dict(ck["opt_wm"]), opt_pi.load_state_dict(ck["opt_pi"]), opt_pl.load_state_dict(ck["opt_planner"])
curriculum = GoalCurriculum(terrains, N_GOAL, dev, max_start_level=args.start_level)
scale = {"ref": None, "goal": None}  # running spread of each group's imagined return
start_it, wall_offset, sim_steps = 0, 0.0, 0
if args.resume:
    critic_m.load_state_dict(ck["critic_m"]), opt_vm.load_state_dict(ck["opt_vm"])
    critic_fs.load_state_dict(ck["critic_fs"]), opt_vfs.load_state_dict(ck["opt_vfs"])
    start_it, wall_offset, sim_steps, lr_it0 = ck["iteration"], ck["wall_time"], ck["sim_steps"], ck["lr_it0"]
    scale = {k: None if v is None else torch.tensor(v, device=dev) for k, v in ck["scale"].items()}
    curriculum.level.copy_(ck["curriculum_level"]), curriculum.success.copy_(ck["curriculum_success"])
else:  # V_m: the Stage 2 critic; V_fs: new; the learning-rate schedule continues from Stage 2
    critic_m.load_state_dict(ck["critic"]), opt_vm.load_state_dict(ck["opt_v"])
    lr_it0 = ck["iteration"] + ck.get("lr_it0", 0)
stage2_args = ck["stage2_args"] if args.resume else ck["args"]


# per-component floors on the error normalization (pos, rot, lin vel, ang vel, q, qd increments)
ERR_FLOOR = torch.cat((torch.full((3,), 3e-3), torch.full((3,), 5e-3), torch.full((3,), 2e-2), torch.full((3,), 5e-2),
                       torch.full((NUM_J,), 3e-3), torch.full((NUM_J,), 5e-2))).to(dev)  # fmt: skip


def _wm_rollout(s, a, xi, f, v):
    S = unflatten_state(s[:, 0])
    preds = []
    for j in range(args.wm_window):
        S = wm.step(S, a[:, j], normalize_xi(xi[:, j]), mdp.scan(S, motion.boxes(f[:, j], v[:, j])))
        preds.append(S)
    return tuple(torch.stack([P[i] for P in preds], 1) for i in range(6))  # (B, k, ...)


def _wm_err(S_hat, s1, mask):
    std = torch.maximum(wm.out_norm.std(), ERR_FLOOR)
    e = (wm.state_error(S_hat, unflatten_state(s1)) / std) ** 2  # (B, k, D)
    e = e.mean(-1)
    return (e * mask).sum() / s1.shape[0], e[:, -1]


def goal_commands(S, f, gs, gw, is_goal, radius_ref: float):
    """(B, 3) each sample's goal command: toward the reference ``gs`` steps ahead (radius ``radius_ref``) or, for goal
    samples, toward the terrain goal ``gw`` (radius 0.5)."""
    return torch.where(is_goal.unsqueeze(-1), goal.command(S, gw), ctc.reference_goal(S, f, gs, radius_ref))


def _imag_step(S, o_hist, f, v, gs, gw, is_goal, xi, xi_n, boxes, tw, tv):
    a = pi(o_hist, ctc.command(S, f, v, gs, noise=True, targets_w=tw, valid=tv, goal=goal_commands(S, f, gs, gw, is_goal, ctc_cfg.goal_radius)))
    S1 = wm.step(S, a, xi_n, mdp.scan(S, boxes))
    o1 = mdp.add_noise(mdp.obs(S1, a, xi))
    return S1, a, torch.cat((o_hist[:, 1:], o1.unsqueeze(1)), 1), torch.where(is_goal, f, f + 1)


def _plan(o_hist, S, f, v, gs, gw, is_goal, noise: bool = True):
    return plan(planner, scan, pcfg, ctc, o_hist, S, f, v, gs, noise=noise, goal=goal_commands(S, f, gs, gw, is_goal, pcfg.goal_radius))


def _reward_ref(S, S_prev, f, v, a, a_prev, a_prev2, xi, plan_w, plan_v):
    return paper_reward(S, S_prev, f, v, a, a_prev, a_prev2, xi, plan_w, plan_v, ETAS[1])


def _reward_goal(S, S_prev, boxes, gw, vmax, a, a_prev, a_prev2, xi, plan_w, plan_v):
    return goal.reward(S, S_prev, boxes, gw, vmax, a, a_prev, a_prev2, xi, plan_w, plan_v), goal.soft_termination(S, boxes)


def _critic_input_m(S, f, v, gs, xi_n, plan_w, plan_v):
    tb = torch.where(plan_v.unsqueeze(-1), to_base(plan_w, S[0], S[1]), 0.0).flatten(-2)
    return torch.cat((ctc.critic_input(S, f, v, gs, xi_n), tb, ETAS.expand(S[0].shape[0], 2)), -1)


wm_rollout = torch.compile(_wm_rollout)
wm_err = torch.compile(_wm_err)
imag_step = torch.compile(_imag_step)
plan_c = torch.compile(_plan)
# max_fusion_size 32: Triton 3.5.0 fails to compile the fully fused backward kernels of these rewards
reward_ref = torch.compile(_reward_ref, options={"max_fusion_size": 32})
reward_goal = torch.compile(_reward_goal, options={"max_fusion_size": 32})
critic_input_m = torch.compile(_critic_input_m)
critic_input_fs = torch.compile(goal.critic_features)


def lr_at(it: int) -> float:
    it += lr_it0
    if it >= args.lr_decay_iters:
        return args.lr * 0.03
    return args.lr * (0.03 + 0.97 * 0.5 * (1 + math.cos(math.pi * it / args.lr_decay_iters)))


def set_requires_grad(m: nn.Module, flag: bool):
    for p in m.parameters():
        p.requires_grad_(flag)


# ------------------------------------------------------------------------------------------------
def wm_update():
    w = buffer.sample_windows(args.wm_batch, args.wm_window)
    S_hat = wm_rollout(w["s"], w["a"], w["xi"], w["f"], w["v"])
    loss, e_last = wm_err(S_hat, w["s1"], w["mask"])
    opt_wm.zero_grad(set_to_none=True)
    loss.backward()
    nn.utils.clip_grad_norm_(wm.parameters(), 1.0)
    opt_wm.step()
    m = w["mask"][:, -1]
    return loss.detach() / args.wm_window, (e_last.detach() * m).sum() / m.sum().clamp(min=1)


def sample_starts(n: int, envs: torch.Tensor):
    """``n`` start points of environments ``envs`` with H frames left in their clip (goal worlds keep their clip's first
    frame), the action before their previous one (``a_prev2``) and the observations the policy saw before them."""
    starts, hists, k = [], [], 0
    while k < n:
        s, h = buffer.sample_starts(2 * n, Hh, envs)
        ok = s["f"] + HZ <= motion.clip_end[motion.clip_id[s["f"]]]
        idx = ok.nonzero(as_tuple=False).squeeze(-1)[: n - k]
        starts.append({key: v[idx] for key, v in s.items()}), hists.append({key: v[idx] for key, v in h.items()})
        k += len(idx)
    cat = lambda ds: {key: torch.cat([d[key] for d in ds]) for key in ds[0]}  # noqa: E731
    s, h = cat(starts), cat(hists)
    s["a_prev2"] = h["a_prev"][:, -2]  # the history stays inside the episode: zeros right after a reset
    return s, h["o"]


def fit_critic(critic: Critic, opt: torch.optim.Optimizer, X: torch.Tensor, rews: torch.Tensor, dones: torch.Tensor) -> torch.Tensor:
    """GAE(lambda) targets of the imagined transitions (critic inputs ``X`` (H+1, B, D)) and the critic's epochs."""
    critic.norm.update(X)
    with torch.no_grad():
        V = critic(X)
        R = V[-1]
        targets = []
        for t in range(HZ - 1, -1, -1):
            R = rews[t] + args.gamma * (1 - dones[t]) * ((1 - args.lam) * V[t + 1] + args.lam * R)
            targets.append(R)
        targets = torch.stack(targets[::-1])  # (H, B)
    Xf, Yf = X[:-1].reshape(-1, X.shape[-1]), targets.reshape(-1)
    n = Xf.shape[0]
    mb = n // args.critic_minibatches
    for _ in range(args.critic_epochs):
        perm = torch.randperm(n, device=dev)
        for i in range(args.critic_minibatches):
            idx = perm[i * mb : (i + 1) * mb]
            v_loss = ((critic(Xf[idx]) - Yf[idx]) ** 2).mean()
            opt.zero_grad(set_to_none=True)
            v_loss.backward()
            nn.utils.clip_grad_norm_(critic.parameters(), 1.0)
            opt.step()
    return v_loss.detach()


def policy_update(train_goal: bool) -> dict:
    """Imagine H steps from B_REF reference and B_GOAL goal starts; step the actor on both groups' scaled returns (the
    reference group only while ``train_goal`` is off) and each critic on its group's GAE targets."""
    (sr, orf), (sg, og) = sample_starts(B_REF, REF_ENVS), sample_starts(B_GOAL, GOAL_ENVS)
    s = {k: torch.cat((sr[k], sg[k])) for k in sr}
    o_hist = torch.cat((orf, og))
    S = unflatten_state(s["s"])
    xi, f, v, a_prev, a_prev2, gs, gw = s["xi"], s["f"], s["v"], s["a_prev"], s["a_prev2"], s["goal_steps"], s["goal"]
    is_goal = torch.arange(B, device=dev) >= B_REF
    xi_n = normalize_xi(xi)
    boxes = motion.boxes(f, v)  # the rollout stays on its terrain
    t0 = s["ep_step"].float()
    for m in (wm, critic_m, critic_fs):
        set_requires_grad(m, False)
    S0, f0 = S, f
    Ss, Sps, As, Aps, Ap2s, F1s, Pws, Pvs = [], [], [], [], [], [], [], []
    for t in range(HZ):
        pw, pv = plan_c(o_hist, S, f, v, gs, gw, is_goal)  # the planner on the imagined state (no gradient)
        Sps.append(S), Aps.append(a_prev), Ap2s.append(a_prev2), Pws.append(pw), Pvs.append(pv)
        a_prev2 = a_prev
        S, a_prev, o_hist, f = imag_step(S, o_hist, f, v, gs, gw, is_goal, xi, xi_n, boxes, pw, pv)
        Ss.append(S), As.append(a_prev), F1s.append(f)
    pw_H, pv_H = plan_c(o_hist, S, f, v, gs, gw, is_goal)
    disc = args.gamma ** torch.arange(HZ, device=dev, dtype=torch.float32).unsqueeze(-1)
    stats, Js, fit = {}, {}, {}
    for key, g, nb in (("ref", slice(0, B_REF), B_REF), ("goal", slice(B_REF, B), B_GOAL)):
        stk = lambda xs: torch.stack(xs)[:, g].flatten(0, 1)  # noqa: E731  (H*nb, ...)
        rep = lambda x: x[g].repeat((HZ,) + (1,) * (x.dim() - 1))  # noqa: E731  (H*nb, ...) of a per-sample tensor
        S_all, S_prev = tuple(stk([St[i] for St in Ss]) for i in range(6)), tuple(stk([St[i] for St in Sps]) for i in range(6))
        f_all = stk(F1s)
        S_H = tuple(x[g] for x in S)
        if key == "ref":
            r = reward_ref(S_all, S_prev, f_all, rep(v), stk(As), stk(Aps), stk(Ap2s), rep(xi), stk(Pws), stk(Pvs))
            d = mdp.soft_termination(S_all, f_all)  # eager: inside the compiled reward it changed the fused backward kernels
            boot = critic_m(critic_input_m(S_H, f[g], v[g], gs[g], xi_n[g], pw_H[g], pv_H[g]))
        else:
            r, d = reward_goal(S_all, S_prev, rep(boxes), rep(gw), goal.v_max(rep(v)), stk(As), stk(Aps), stk(Ap2s), rep(xi), stk(Pws), stk(Pvs))
            boot = critic_fs(critic_input_fs(S_H, xi_n[g], boxes[g], gw[g], (t0[g] + HZ) / gcfg.max_steps, pw_H[g], pv_H[g]))
        r, d = r.view(HZ, nb), d.view(HZ, nb)
        alive = torch.cumprod(1 - d, 0)  # c_{t+1}
        c_t = torch.cat((torch.ones(1, nb, device=dev), alive[:-1]), 0)
        Js[key] = (disc * c_t * r).sum(0) + (args.gamma**HZ) * alive[-1] * boot
        sd = Js[key].detach().std()
        scale[key] = sd if scale[key] is None else args.scale_ema * scale[key] + (1 - args.scale_ema) * sd
        fit[key] = (r.detach(), d.detach())
        stats[f"imag_r_{key}"] = ((c_t * r).detach().sum(0) / HZ / MAX_R[key]).mean()
        stats[f"surv_{key}"] = alive[-1].detach().mean()
    # the paper's per-group advantage normalization, as a per-group scale of the imagined return (no advantages here)
    w_goal = 1.0 if train_goal else 0.0
    loss = -((Js["ref"] / scale["ref"].clamp(min=1e-3)).sum() + w_goal * (Js["goal"] / scale["goal"].clamp(min=1e-3)).sum()) / (B_REF + w_goal * B_GOAL)
    opt_pi.zero_grad(set_to_none=True)
    loss.backward()
    gn = nn.utils.clip_grad_norm_(pi.parameters(), 1.0)
    nn.utils.clip_grad_value_(pi.parameters(), 0.1)
    if torch.isfinite(gn):
        opt_pi.step()
    for m in (wm, critic_m, critic_fs):
        set_requires_grad(m, True)
    # ---------------- critics: each on its group's GAE(lambda) targets ----------------
    with torch.no_grad():
        S_c = tuple(torch.cat((S0[i].unsqueeze(0), torch.stack([St[i].detach() for St in Ss])), 0) for i in range(6))  # (H+1, B, ...)
        f_c = torch.cat((f0.unsqueeze(0), torch.stack(F1s)), 0)
        P_c, V_c = torch.stack(Pws + [pw_H]), torch.stack(Pvs + [pv_H])
        grp = lambda x, g: x[:, g].flatten(0, 1)  # noqa: E731
        rep1 = lambda x, g: x[g].repeat((HZ + 1,) + (1,) * (x.dim() - 1))  # noqa: E731
        R_, G_ = slice(0, B_REF), slice(B_REF, B)
        X_m = critic_input_m(tuple(grp(x, R_) for x in S_c), grp(f_c, R_), rep1(v, R_), rep1(gs, R_), rep1(xi_n, R_), grp(P_c, R_),
                             grp(V_c, R_)).view(HZ + 1, B_REF, -1)  # fmt: skip
        t_c = ((t0[G_].unsqueeze(0) + torch.arange(HZ + 1, device=dev).unsqueeze(-1)) / gcfg.max_steps).flatten()
        X_fs = critic_input_fs(tuple(grp(x, G_) for x in S_c), rep1(xi_n, G_), rep1(boxes, G_), rep1(gw, G_), t_c, grp(P_c, G_),
                               grp(V_c, G_)).view(HZ + 1, B_GOAL, -1)  # fmt: skip
    stats["v_m"] = fit_critic(critic_m, opt_vm, X_m, *fit["ref"])
    stats["v_fs"] = fit_critic(critic_fs, opt_vfs, X_fs, *fit["goal"])
    stats.update(pi=loss.detach(), gn=gn.detach(), scale_ref=scale["ref"], scale_goal=scale["goal"])
    return stats


# ------------------------------------------------------------------------------------------------
assert args.buffer_steps > max(int(motion.clip_len.max()), gcfg.max_steps), "the replay buffer must hold a whole episode"
env = TrackingSim(N, motion, mdp, dev, gamma=args.gamma, start_phase=(0.0, 0.0), goal=goal, num_goal_envs=N_GOAL, goal_curriculum=curriculum)
# Warp's pool otherwise returns its sim-step scratch to the driver at every sync and re-allocates it in the next graph
# launch, which can fail once PyTorch's cache has grown into that space; keep it reserved instead
wp.set_mempool_release_threshold(dev, torch.cuda.get_device_properties(dev).total_memory)
buffer = ReplayBuffer(args.buffer_steps, N, dev)
pbuf_ref = PlannerBuffer(pcfg.capacity, pcfg.hist, planner.prop_dim, scan.num_z, dev)
pbuf_goal = GroupedPlannerBuffer(len(TERRAIN_TYPES), pcfg.capacity, pcfg.hist, planner.prop_dim, scan.num_z, dev)
pbuf = MixedPlannerBuffer(pbuf_ref, pbuf_goal)
vbuf = PlannerBuffer(pcfg.capacity // 5, pcfg.hist, planner.prop_dim, scan.num_z, dev)
folders = sorted({n.split("/")[0] for n in motion.names})
val_clip = torch.zeros(len(motion.names), dtype=torch.bool, device=dev)
for gname in folders:  # every 10th clip of each folder: planner validation only (the tracker trains on every clip)
    idx = [i for i, n in enumerate(motion.names) if n.split("/")[0] == gname]
    val_clip[idx[::10]] = True
obs_hist = torch.zeros(N, Hh, OBS_DIM, device=dev)
goal_steps = ctc.sample_goal_steps(N)  # reference group: its goal's look-ahead, one draw per episode
is_goal_env = env.is_goal


@torch.no_grad()
def collect(n_steps: int, stats: dict):
    global obs_hist, sim_steps
    for _ in range(n_steps):
        o = env.observe()
        fresh = env.ep_step == 0
        obs_hist = torch.where(fresh[:, None, None], o.unsqueeze(1).expand_as(obs_hist), torch.cat((obs_hist[:, 1:], o.unsqueeze(1)), 1))
        S = env.state()
        pw, pv = plan_c(obs_hist, S, env.f, env.v, goal_steps, env.goal_w, is_goal_env)
        cmd = ctc.command(S, env.f, env.v, goal_steps, noise=True, targets_w=pw, valid=pv,
                          goal=goal_commands(S, env.f, goal_steps, env.goal_w, is_goal_env, ctc_cfg.goal_radius))  # fmt: skip
        pi.obs_norm.update(o)
        pi.goal_norm.update(cmd)
        a = pi(obs_hist, cmd)
        a = torch.clamp(a + args.action_noise * torch.randn_like(a), -20, 20)
        out = env.step(a)
        out["goal_steps"] = goal_steps.clone()
        buffer.add(out)
        S0, S1 = unflatten_state(out["s"]), unflatten_state(out["s1"])
        wm.in_norm.update(wm.inputs(S0, out["a"], normalize_xi(out["xi"]), mdp.scan(S0, motion.boxes(out["f"], out["v"]))))
        wm.out_norm.update(wm.targets(S0, S1))
        sim_steps += N
        stats["r_ref"].append(out["r"][:N_REF].mean() / MAX_STEP_REWARD)
        stats["r_goal"].append(out["r"][N_REF:].mean() / ((gcfg.w_move + gcfg.w_stand) * CONTROL_DT))
        done = out["done"]
        envs = done.nonzero(as_tuple=False).squeeze(-1)
        if len(envs) > 0:
            eg = out["ep_goal"]
            lens = out["ep_step"][envs] + 1
            row = torch.full_like(envs, (buffer.ptr - 1) % buffer.T)
            for key, sel in (("ref", ~eg), ("goal", eg)):
                stats[f"n_ep_{key}"].append(sel.sum()), stats[f"len_{key}"].append(torch.where(sel, out["ep_len"], 0.0).sum())
                stats[f"term_{key}"].append((sel & out["term"][envs]).sum())
            stats["reached"].append((eg & out["ep_reached"]).sum()), stats["success"].append((eg & out["ep_reached"] & ~out["term"][envs]).sum())
            # hindsight labels: reference episodes by their progress, goal episodes that reached their goal
            r_ids, g_ids = (~eg).nonzero(as_tuple=False).squeeze(-1), eg.nonzero(as_tuple=False).squeeze(-1)
            if len(r_ids) > 0:
                d, n_adm = label_episodes(buffer, envs[r_ids], row[r_ids], lens[r_ids], ctc, scan, pcfg)
                stats["admitted_ref"].append(n_adm)
                if d is not None:
                    val = val_clip[d["clip"]]
                    vbuf.add({k: x[val] for k, x in d.items()})
                    d = {k: x[~val] for k, x in d.items()}
                    if len(d["occ"]) > 0:
                        planner.prop_norm.update(d["prop"].float())
                        pbuf_ref.add(d)
            if len(g_ids) > 0:
                fc = out["ep_first_close"][g_ids]
                keep = torch.where(fc >= 0, fc + 1, lens[g_ids])
                d, n_adm = label_episodes(buffer, envs[g_ids], row[g_ids], lens[g_ids], ctc, scan, pcfg, success=out["ep_reached"][g_ids],
                                          keep=keep, goal_radius=gcfg.radius)  # fmt: skip
                stats["admitted_goal"].append(n_adm)
                if d is not None and len(d["occ"]) > 0:
                    planner.prop_norm.update(d["prop"].float())
                    pbuf_goal.add(d, terrains["type"].to(dev)[d["v"] - motion.num_variants])
        goal_steps[done] = ctc.sample_goal_steps(int(done.sum()))


stats_keys = ("r_ref", "r_goal", "n_ep_ref", "n_ep_goal", "len_ref", "len_goal", "term_ref", "term_goal", "reached", "success",
              "admitted_ref", "admitted_goal", "pl")  # fmt: skip
stats = {k: [] for k in stats_keys}
collect(args.warmup_steps, stats)  # the replay buffer is not checkpointed: a resumed run refills it
BODIES = list(CONTACT_BODIES)
csv_path = os.path.join(args.log_dir, "train_log.csv")
new_log = not os.path.exists(csv_path) or os.path.getsize(csv_path) == 0
log_f = open(csv_path, "a")
val_f = open(os.path.join(args.log_dir, "planner_val.csv"), "a")
COLS = ("iteration,wall_h,sim_steps,wm_loss,wm_err8,pi_loss,v_m,v_fs,pi_gradnorm,imag_r_ref,surv_ref,imag_r_goal,surv_goal,scale_ref,"
        "scale_goal,sim_r_ref,sim_r_goal,ep_len_ref,ep_len_goal,term_ref,fell_goal,reached_goal,success_goal,admitted_ref,admitted_goal,"
        "planner_ref_buffer,planner_goal_buffer,planner_aff,planner_occ,it_per_s,lr" + "".join(f",level_{t}" for t in TERRAIN_TYPES))  # fmt: skip
if new_log:
    log_f.write(COLS + "\n")
    val_f.write("iteration,aff,occ," + ",".join(f"{m}_{b}" for m in ("precision", "recall", "f1", "median_err", "within_0.1") for b in BODIES) + "\n")
    log_f.flush(), val_f.flush()
t_start = time.time() - wall_offset
t_log = time.time()
step_acc = 0.0
agg = {}
for it in range(start_it, args.iterations):
    lr = lr_at(it)
    for o in (opt_wm, opt_pi, opt_vm):
        for gparam in o.param_groups:
            gparam["lr"] = lr
    step_acc += 1.0 / args.utd
    n = int(step_acc)
    step_acc -= n
    collect(n, stats)
    for _ in range(args.wm_batches):
        l, e8 = wm_update()
    agg.setdefault("wm", []).append(l), agg.setdefault("wm8", []).append(e8)
    for k, x in policy_update(train_goal=it - start_it >= args.critic_warmup or args.resume is not None).items():
        agg.setdefault(k, []).append(x)
    if pbuf.size >= pcfg.min_entries:
        for _ in range(args.planner_updates):
            stats["pl"].append(planner_update(planner, opt_pl, pbuf, scan, pcfg))
    if (it + 1) % args.log_every == 0:
        mean = lambda k: float(sum(float(v) for v in agg.get(k, [])) / max(len(agg.get(k, [])), 1))  # noqa: E731
        total = lambda k: float(sum(float(v) for v in stats[k]))  # noqa: E731
        ratio = lambda a, b: total(a) / max(total(b), 1)  # noqa: E731
        avg = lambda k: float(sum(float(v) for v in stats[k]) / max(len(stats[k]), 1))  # noqa: E731
        pl = torch.stack(stats["pl"]).mean(0).tolist() if stats["pl"] else [float("nan")] * 2
        ips = args.log_every / (time.time() - t_log)
        t_log = time.time()
        wall_h = (time.time() - t_start) / 3600
        levels = [float(curriculum.level[curriculum.type == i].float().mean()) for i in range(len(TERRAIN_TYPES))]
        row = [it + 1, wall_h, sim_steps, mean("wm"), mean("wm8"), mean("pi"), mean("v_m"), mean("v_fs"), mean("gn"), mean("imag_r_ref"),
               mean("surv_ref"), mean("imag_r_goal"), mean("surv_goal"), mean("scale_ref"), mean("scale_goal"), avg("r_ref"), avg("r_goal"),
               ratio("len_ref", "n_ep_ref"), ratio("len_goal", "n_ep_goal"), ratio("term_ref", "n_ep_ref"), ratio("term_goal", "n_ep_goal"),
               ratio("reached", "n_ep_goal"), ratio("success", "n_ep_goal"), int(total("admitted_ref")), int(total("admitted_goal")),
               pbuf_ref.size, pbuf_goal.size, *pl, ips, lr] + levels  # fmt: skip
        log_f.write(",".join(f"{x:.6g}" if isinstance(x, float) else str(x) for x in row) + "\n")
        log_f.flush()
        warm = "" if it - start_it >= args.critic_warmup or args.resume else " (V_fs warm-up)"
        print(f"it {it + 1} | {wall_h:.3f} h | wm {row[3]:.4f} e8 {row[4]:.4f} | pi {row[5]:.3f} v_m {row[6]:.4f} v_fs {row[7]:.4f}{warm} | "
              f"REF imag_r {row[9]:.3f} surv {row[10]:.3f} sim_r {row[15]:.3f} ep_len {row[17]:.0f} term {row[19]:.2f} | "
              f"GOAL imag_r {row[11]:.3f} surv {row[12]:.3f} sim_r {row[16]:.3f} ep_len {row[18]:.0f} fell {row[20]:.2f} "
              f"reached {row[21]:.2f} success {row[22]:.2f} | planner adm {row[23]}/{row[24]} buf {pbuf_ref.size}/{pbuf_goal.size} "
              f"aff {pl[0]:.3f} occ {pl[1]:.3f} | level " + " ".join(f"{t[:6]} {lv:.1f}" for t, lv in zip(TERRAIN_TYPES, levels))
              + f" | {ips:.2f} it/s", flush=True)  # fmt: skip
        stats = {k: [] for k in stats_keys}
        agg = {}
    if (it + 1) % args.val_every == 0 and vbuf.size > 0:
        planner.eval()
        mt = planner_metrics(planner, vbuf.sample(min(16384, vbuf.size)), scan, pcfg)
        planner.train()
        keys = ("precision", "recall", "f1", "median_err", "within_0.1")
        val_f.write(",".join(str(x) for x in [it + 1, mt["aff"], mt["occ"]] + [v for k in keys for v in mt[k]]) + "\n")
        val_f.flush()
    if (it + 1) % args.ckpt_every == 0 or it + 1 == args.iterations:
        ck = {"wm": wm.state_dict(), "pi": pi.state_dict(), "critic_m": critic_m.state_dict(), "critic_fs": critic_fs.state_dict(),
              "planner": planner.state_dict(), "opt_wm": opt_wm.state_dict(), "opt_pi": opt_pi.state_dict(), "opt_vm": opt_vm.state_dict(),
              "opt_vfs": opt_vfs.state_dict(), "opt_planner": opt_pl.state_dict(), "iteration": it + 1, "wall_time": time.time() - t_start,
              "sim_steps": sim_steps, "args": vars(args), "init_args": ia, "stage2_args": stage2_args, "ctc_cfg": dataclasses.asdict(ctc_cfg),
              "reward_cfg": dataclasses.asdict(reward_cfg), "planner_cfg": dataclasses.asdict(pcfg), "goal_cfg": dataclasses.asdict(gcfg),
              "terrain_args": terrain_args, "lr_it0": lr_it0, "scale": {k: None if v is None else float(v) for k, v in scale.items()},
              "curriculum_level": curriculum.level.clone(), "curriculum_success": curriculum.success.clone(),
              "saved_at": time.strftime("%Y-%m-%d %H:%M:%S")}  # fmt: skip
        last = os.path.join(args.log_dir, "last.pt")
        torch.save(ck, last + ".tmp")
        os.replace(last + ".tmp", last)
        small = {k: ck[k] for k in ("pi", "wm", "planner", "iteration", "wall_time", "sim_steps", "args", "init_args", "ctc_cfg",
                                    "planner_cfg", "goal_cfg", "terrain_args", "saved_at")}  # fmt: skip
        path = os.path.join(args.log_dir, f"ckpt_{it + 1:06d}.pt")
        torch.save(small, path + ".tmp")
        os.replace(path + ".tmp", path)

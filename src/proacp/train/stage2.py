"""Stage 2: contact planner and tracker co-training through one world model (:mod:`proacp.models.planner`).

The tracker of proacp.train.stage1 (``--init``: its last.pt, terrain-point targets) keeps training with the
paper's reward and the reference-following termination, while the contact planner (``--planner_init``: a
proacp.train.planner_offline checkpoint) learns from hindsight labels of the tracker's own successful episodes.
The planner reads the exact height scan of the episode's terrain (plus noise) in the simulator and on every imagined
state. Episodes start at uniform phases of their clip. Two linear schedules (the paper's Eq. 3) hand the tracker over
to the planner: with probability eta_obs, per step and environment, the actor's contact targets are the planner's
instead of the reference's; eta_rew moves the reward from the three reference-contact terms to the planned-contact
term (:class:`proacp.tasks.contact.PaperReward`).

Differences from the paper's PPO setup:
- The actor learns in world-model imagination, so the planner runs on every imagined state, and the target mixing
  and the reward schedule apply there as well as in the simulator.
- The planner's top-1 targets are lifted to the world (no gradient) and enter the command through the same
  differentiable base-frame transform as the reference targets.
- The critic, whose gradient reaches the actor, also sees the planner's targets and both schedules. It starts as
  the Stage 1 critic, widened with zero weights.
- Planner updates (``--planner_updates``) and the schedules count SuperMimic iterations.
- The replay buffer of simulator transitions doubles as the trajectory recorder (``--buffer_steps`` covers the
  longest clip).

``--hold_admit`` (0 = off) slows the ramps when fewer episodes pass the success filter: the schedules' clock stops while
the admission rate over the last ``--hold_window`` iterations is below that share of its rate just before ``--u_obs``.
``--term_window`` (s, 0 = off) relaxes the reference-anchored termination in the simulator and in imagination: a state
is ended only if its root height and tilt match no reference frame within that time either side of its own.

Usage: python -m proacp.train.stage2 --init logs/acp_stage1/last.pt \\
           --planner_init logs/acp_planner/planner_020000.pt --log_dir logs/acp_stage2
"""

from __future__ import annotations

import argparse
import collections
import dataclasses
import json
import math
import os
import time

parser = argparse.ArgumentParser()
parser.add_argument("--init", type=str, default=None, help="proacp.train.stage1 last.pt (terrain-point targets)")
parser.add_argument("--planner_init", type=str, default=None, help="proacp.train.planner_offline checkpoint")
parser.add_argument("--num_envs", type=int, default=4096)
parser.add_argument("--iterations", type=int, default=40_000)
parser.add_argument("--buffer_steps", type=int, default=1350, help="replay rows per environment: at least the longest clip")
parser.add_argument("--utd", type=float, default=0.4, help="updates per vectorized simulator step")
parser.add_argument("--horizon", type=int, default=32)
parser.add_argument("--wm_window", type=int, default=8)
parser.add_argument("--wm_batches", type=int, default=10)
parser.add_argument("--wm_batch", type=int, default=2048)
parser.add_argument("--pi_batch", type=int, default=4096)
parser.add_argument("--lr", type=float, default=2e-4)
parser.add_argument("--gamma", type=float, default=0.99)
parser.add_argument("--lam", type=float, default=0.95)
parser.add_argument("--action_noise", type=float, default=0.1)
parser.add_argument("--critic_epochs", type=int, default=5)
parser.add_argument("--critic_minibatches", type=int, default=4)
parser.add_argument("--warmup_steps", type=int, default=64, help="sim steps before any update")
parser.add_argument("--lr_decay_iters", type=int, default=250_000)
parser.add_argument("--start_phase", type=float, nargs=2, default=(0.0, 0.98), help="uniform episode start phases (paper Stage 2)")
parser.add_argument("--planner_updates", type=int, default=2, help="planner updates per iteration")
parser.add_argument("--planner_capacity", type=int, default=300_000)
parser.add_argument("--u_obs", type=int, default=2000, help="iteration where eta_obs starts to rise")
parser.add_argument("--u_rew", type=int, default=12_000, help="iteration where eta_rew starts to rise")
parser.add_argument("--n_ann", type=int, default=20_000, help="iterations of each ramp")
parser.add_argument("--hold_admit", type=float, default=0.6,
                    help="hold the schedules while the admission rate (last --hold_window iterations) is below this share of its eta=0 rate; 0: off")
parser.add_argument("--hold_window", type=int, default=500)
parser.add_argument("--eta_rew_max", type=float, default=1.0,
                    help="cap on eta_rew: past it the reward stops moving from the reference contacts to the planned ones")
parser.add_argument("--term_window", type=float, default=0.0,
                    help="s: end a state only if it matches no reference root (height, tilt) this far either side of its frame; 0: its frame only")
parser.add_argument("--planned_std", type=float, default=None, help="m: width of the planned-contact term (default: the checkpoint's)")
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

import torch
import torch.nn as nn
import warp as wp

import proacp.models.networks as networks
from proacp.train.buffer import ReplayBuffer
from proacp.sim.env import TrackingSim
from proacp.robot import CONTACT_BODIES, MOTION_DIR, NB, NUM_J, default_joint_pos_list
from proacp.sim.kinematics import Kinematics
from proacp.tasks.tracking import INV_DIM, MAX_STEP_REWARD, MDP, OBS_DIM, REF_ORI_DIMS, RewardCfg, normalize_xi, unflatten_state
from proacp.clips.library import CONTROL_DT, MotionLib
from proacp.models.networks import Critic, Policy, WorldModel
from proacp.models.planner import (ContactPlanner, PlannerBuffer, PlannerCfg, PlannerScan, label_episodes, plan, planner_batch_loss,
                                    planner_metrics, planner_update)  # fmt: skip
from proacp.tasks.contact import PLACE_WINDOW, ContactCfg, ContactTask, PaperReward, PaperRewardCfg, placement_error, to_base

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.manual_seed(args.seed)
networks.AMP_DTYPE = torch.bfloat16 if args.amp == "bf16" else None
dev = "cuda:0"
os.makedirs(args.log_dir, exist_ok=True)
json.dump(vars(args), open(os.path.join(args.log_dir, "args.json"), "w"), indent=2)

ck = torch.load(args.resume or args.init, map_location=dev)
ia = ck["init_args"] if args.resume else ck["args"]  # the Stage 1 run's arguments
assert ck.get("reward_cfg") is not None, "Stage 2 continues the paper reward (train.stage1 --reward paper)"
motion_dir = ia.get("motion_dir") or MOTION_DIR
kin = Kinematics(dev)
motion = MotionLib.from_dir(motion_dir, kin, dev)
mdp = MDP(kin, motion, torch.tensor(default_joint_pos_list(), device=dev), RewardCfg(term_window=int(round(args.term_window / CONTROL_DT))))
ctc_cfg = ContactCfg(**ck["ctc_cfg"])
if ctc_cfg.target_point != "terrain":
    print("warning: the tracker's targets are body links, the planner's are terrain points", flush=True)
ctc = ContactTask(ctc_cfg, mdp, motion_dir, dev)
reward_cfg = PaperRewardCfg(**{**ck["reward_cfg"], **({"planned_std": args.planned_std} if args.planned_std else {})})
paper_reward = PaperReward(reward_cfg, ctc)
MAX_R = reward_cfg.max_step_reward
Hh, HZ = ia["hist"], args.horizon
pk = torch.load(args.planner_init, map_location=dev) if args.planner_init and not args.resume else None
saved_pcfg = ck["planner_cfg"] if args.resume else pk["planner_cfg"] if pk else {}  # the planner's inputs as trained
pcfg = PlannerCfg(**{**saved_pcfg, "capacity": args.planner_capacity})
scan = PlannerScan(pcfg, dev)
CRITIC_EXTRA = NB * 3 + 2  # the planner's targets (base frame) and (eta_obs, eta_rew)

wm = WorldModel(mdp.scan_dim).to(dev)
pi = Policy(OBS_DIM, ctc.cmd_dim, hist=Hh).to(dev)
critic = Critic(INV_DIM + mdp.critic_extra + ctc.cmd_dim + CRITIC_EXTRA).to(dev)
planner = ContactPlanner(OBS_DIM - REF_ORI_DIMS, pcfg).to(dev)
opt_wm = torch.optim.Adam(wm.parameters(), lr=args.lr)
opt_pi = torch.optim.Adam(pi.parameters(), lr=args.lr, betas=(0.9, 0.95))
opt_v = torch.optim.Adam(critic.parameters(), lr=args.lr)
opt_pl = torch.optim.Adam(planner.parameters(), lr=pcfg.lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0)
wm.load_state_dict(ck["wm"]), pi.load_state_dict(ck["pi"])
opt_wm.load_state_dict(ck["opt_wm"]), opt_pi.load_state_dict(ck["opt_pi"])
start_it, wall_offset, sim_steps = 0, 0.0, 0
if args.resume:
    critic.load_state_dict(ck["critic"]), opt_v.load_state_dict(ck["opt_v"])
    planner.load_state_dict(ck["planner"]), opt_pl.load_state_dict(ck["opt_planner"])
    start_it, wall_offset, sim_steps, lr_it0 = ck["iteration"], ck["wall_time"], ck["sim_steps"], ck["lr_it0"]
    sched_it, admit_base = ck["sched_it"], ck["admit_base"]
else:
    lr_it0 = ck["iteration"] + ck.get("lr_it0", 0)  # the learning-rate schedule continues from Stage 1
    sched_it, admit_base = 0, None  # the schedules' clock (stops while held) and the eta=0 admission rate
    sd, new = ck["critic"], critic.state_dict()  # the Stage 1 critic, its input widened with zero weights
    n_in = sd["net.0.weight"].shape[1]
    for k, v in sd.items():
        if k == "net.0.weight":
            new[k].zero_()[:, :n_in] = v
        elif k in ("norm.mean", "norm.var"):
            new[k][:n_in] = v
        else:
            new[k] = v
    critic.load_state_dict(new)
    if pk:
        planner.load_state_dict(pk["planner"]), opt_pl.load_state_dict(pk["opt"])
init_args = ia


def eta(it: int, u: int) -> float:
    """The paper's Eq. 3: 0 until u, then linear to 1 over n_ann iterations."""
    return min(max((it - u) / args.n_ann, 0.0), 1.0)


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


def _imag_step(S, o_hist, f, v, gs, xi, xi_n, boxes, tw, tv):
    a = pi(o_hist, ctc.command(S, f, v, gs, noise=True, targets_w=tw, valid=tv))
    S1 = wm.step(S, a, xi_n, mdp.scan(S, boxes))
    o1 = mdp.add_noise(mdp.obs(S1, a, xi))
    return S1, a, torch.cat((o_hist[:, 1:], o1.unsqueeze(1)), 1), f + 1


def _reward_term(S, S_prev, f, v, a, a_prev, a_prev2, xi, plan_w, plan_v, eta_rew):
    return paper_reward(S, S_prev, f, v, a, a_prev, a_prev2, xi, plan_w, plan_v, eta_rew)


def _critic_input(S, f, v, gs, xi_n, plan_w, plan_v, etas):
    tb = torch.where(plan_v.unsqueeze(-1), to_base(plan_w, S[0], S[1]), 0.0).flatten(-2)
    return torch.cat((ctc.critic_input(S, f, v, gs, xi_n), tb, etas.expand(S[0].shape[0], 2)), -1)


wm_rollout = torch.compile(_wm_rollout)
wm_err = torch.compile(_wm_err)
imag_step = torch.compile(_imag_step)
# eager: with the planned term, Triton 3.5.0 fails to compile a fused backward reduction of this reward, whatever the
# fusion limit (Stage 1 compiles it with max_fusion_size 32)
reward_term = _reward_term
critic_input = torch.compile(_critic_input)
plan = torch.compile(plan)  # 33 calls per policy update


def mix(f: torch.Tensor, v: torch.Tensor, plan_w: torch.Tensor, plan_v: torch.Tensor, eta_obs: float):
    """The actor's targets: the planner's with probability eta_obs per sample, else the reference's; and which."""
    use = torch.rand(f.shape[0], device=dev) < eta_obs
    return (torch.where(use[:, None, None], plan_w, ctc.next_pos[f, v]), torch.where(use[:, None], plan_v, ctc.next_valid[f, v]), use)


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


def sample_starts(B: int):
    """Start points with H reference frames left in their clip, the action before their previous one (``a_prev2``)
    and the observations the policy saw before them."""
    starts, hists, n = [], [], 0
    while n < B:
        s, h = buffer.sample_starts(2 * B, Hh)
        ok = s["f"] + HZ <= motion.clip_end[motion.clip_id[s["f"]]]
        idx = ok.nonzero(as_tuple=False).squeeze(-1)[: B - n]
        starts.append({k: v[idx] for k, v in s.items()}), hists.append({k: v[idx] for k, v in h.items()})
        n += len(idx)
    cat = lambda ds: {k: torch.cat([d[k] for d in ds]) for k in ds[0]}  # noqa: E731
    s, h = cat(starts), cat(hists)
    s["a_prev2"] = h["a_prev"][:, -2]  # the history stays inside the episode: zeros right after a reset
    return s, h["o"]


def policy_update(eta_obs: float, eta_rew: float):
    s, o_hist = sample_starts(args.pi_batch)
    S = unflatten_state(s["s"])
    xi, f, v, a_prev, a_prev2, gs = s["xi"], s["f"], s["v"], s["a_prev"], s["a_prev2"], s["goal_steps"]
    xi_n = normalize_xi(xi)
    boxes = motion.boxes(f, v)  # the rollout stays inside its clip
    etas = torch.tensor([eta_obs, eta_rew], device=dev)
    set_requires_grad(wm, False)
    set_requires_grad(critic, False)
    B = args.pi_batch
    S0, f0 = S, f
    Ss, Sps, As, Aps, Ap2s, F1s, Pws, Pvs = [], [], [], [], [], [], [], []
    for t in range(HZ):
        pw, pv = plan(planner, scan, pcfg, ctc, o_hist, S, f, v, gs)  # the planner on the imagined state (no gradient)
        tw, tv, _ = mix(f, v, pw, pv, eta_obs)
        Sps.append(S), Aps.append(a_prev), Ap2s.append(a_prev2), Pws.append(pw), Pvs.append(pv)
        a_prev2 = a_prev
        S, a_prev, o_hist, f = imag_step(S, o_hist, f, v, gs, xi, xi_n, boxes, tw, tv)
        Ss.append(S), As.append(a_prev), F1s.append(f)
    pw_H, pv_H = plan(planner, scan, pcfg, ctc, o_hist, S, f, v, gs)
    flat = lambda xs: torch.stack(xs).flatten(0, 1)  # noqa: E731  (H*B, ...)
    S_all = tuple(flat([St[i] for St in Ss]) for i in range(6))
    S_prev = tuple(flat([St[i] for St in Sps]) for i in range(6))
    xi_all = xi.unsqueeze(0).expand(HZ, -1, -1).flatten(0, 1)
    f_all, v_all = flat(F1s), v.repeat(HZ)
    r = reward_term(S_all, S_prev, f_all, v_all, flat(As), flat(Aps), flat(Ap2s), xi_all, flat(Pws), flat(Pvs), etas[1])
    d = mdp.soft_termination(S_all, f_all)  # eager: inside the compiled reward it changed the fused backward kernels
    r, d = r.view(HZ, B), d.view(HZ, B)
    alive = torch.cumprod(1 - d, 0)  # c_{t+1}
    c_t = torch.cat((torch.ones(1, B, device=dev), alive[:-1]), 0)
    disc = args.gamma ** torch.arange(HZ, device=dev, dtype=torch.float32).unsqueeze(-1)
    Jr = (disc * c_t * r).sum(0)
    c = alive[-1]
    r_sum = (c_t * r).detach().sum(0)
    rews, dones = r.detach(), d.detach()
    Jr = Jr + (args.gamma**HZ) * c * critic(critic_input(S, f, v, gs, xi_n, pw_H, pv_H, etas))
    loss = -Jr.mean()
    opt_pi.zero_grad(set_to_none=True)
    loss.backward()
    gn = nn.utils.clip_grad_norm_(pi.parameters(), 1.0)
    nn.utils.clip_grad_value_(pi.parameters(), 0.1)
    if torch.isfinite(gn):
        opt_pi.step()
    set_requires_grad(wm, True)
    set_requires_grad(critic, True)
    # ---------------- critic: GAE(lambda) targets over the imagined transitions ----------------
    with torch.no_grad():
        S_c = tuple(torch.cat((S0[i].unsqueeze(0), torch.stack([St[i].detach() for St in Ss])), 0).flatten(0, 1) for i in range(6))
        f_c = torch.cat((f0.unsqueeze(0), torch.stack(F1s)), 0).flatten()
        v_c, gs_c = v.repeat(HZ + 1), gs.repeat(HZ + 1)
        xi_c = xi_n.unsqueeze(0).expand(HZ + 1, -1, -1).flatten(0, 1)
        X = critic_input(S_c, f_c, v_c, gs_c, xi_c, torch.stack(Pws + [pw_H]).flatten(0, 1), torch.stack(Pvs + [pv_H]).flatten(0, 1),
                         etas).view(HZ + 1, B, -1)  # fmt: skip
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
            opt_v.zero_grad(set_to_none=True)
            v_loss.backward()
            nn.utils.clip_grad_norm_(critic.parameters(), 1.0)
            opt_v.step()
    imag_r = (r_sum / HZ / MAX_R).mean()
    return loss.detach(), v_loss.detach(), imag_r, gn.detach(), c.mean().detach()


# ------------------------------------------------------------------------------------------------
N = args.num_envs
assert args.buffer_steps > int(motion.clip_len.max()), "the replay buffer must hold a whole episode"
env = TrackingSim(N, motion, mdp, dev, gamma=args.gamma, start_phase=tuple(args.start_phase))
# Warp's pool otherwise returns its sim-step scratch to the driver at every sync and re-allocates it in the next graph
# launch, which can fail once PyTorch's cache has grown into that space; keep it reserved instead
wp.set_mempool_release_threshold(dev, torch.cuda.get_device_properties(dev).total_memory)
buffer = ReplayBuffer(args.buffer_steps, N, dev)
pbuf = PlannerBuffer(pcfg.capacity, pcfg.hist, planner.prop_dim, scan.num_z, dev)
vbuf = PlannerBuffer(pcfg.capacity // 5, pcfg.hist, planner.prop_dim, scan.num_z, dev)
folders = sorted({n.split("/")[0] for n in motion.names})
val_clip = torch.zeros(len(motion.names), dtype=torch.bool, device=dev)
for g in folders:  # every 10th clip of each folder: planner validation only (the tracker trains on every clip)
    idx = [i for i, n in enumerate(motion.names) if n.split("/")[0] == g]
    val_clip[idx[::10]] = True
obs_hist = torch.zeros(N, Hh, OBS_DIM, device=dev)
goal_steps = ctc.sample_goal_steps(N)
prev_touch = torch.ones(N, NB, dtype=torch.bool, device=dev)
tw_win = torch.zeros(PLACE_WINDOW, N, NB, 3, device=dev)  # recent commands' targets, for the placement error
tv_win = torch.zeros(PLACE_WINDOW, N, NB, dtype=torch.bool, device=dev)
n_ended, n_admitted = 0, 0
adm_hist = collections.deque(maxlen=args.hold_window + 1)  # (ended, admitted) totals after each iteration


def admit_rate() -> float | None:
    """Share of finished episodes admitted to the planner's buffer over the last ``hold_window`` iterations."""
    if len(adm_hist) < adm_hist.maxlen or adm_hist[-1][0] == adm_hist[0][0]:
        return None
    return (adm_hist[-1][1] - adm_hist[0][1]) / (adm_hist[-1][0] - adm_hist[0][0])


@torch.no_grad()
def collect(n_steps: int, stats: dict, eta_obs: float):
    global obs_hist, sim_steps, prev_touch, tw_win, tv_win, n_ended, n_admitted
    for _ in range(n_steps):
        o = env.observe()
        fresh = env.ep_step == 0
        obs_hist = torch.where(fresh[:, None, None], o.unsqueeze(1).expand_as(obs_hist), torch.cat((obs_hist[:, 1:], o.unsqueeze(1)), 1))
        S = env.state()
        pw, pv = plan(planner, scan, pcfg, ctc, obs_hist, S, env.f, env.v, goal_steps)
        tw, tv, use = mix(env.f, env.v, pw, pv, eta_obs)
        tw_win, tv_win = torch.cat((tw_win[1:], tw[None])), torch.cat((tv_win[1:], tv[None]))
        cmd = ctc.command(S, env.f, env.v, goal_steps, noise=True, targets_w=tw, valid=tv)
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
        # contacts in the simulator: touch per body vs the reference label; where new contacts land vs their target
        f1 = out["f"] + 1
        boxes1 = motion.boxes(f1, out["v"])
        fk1 = kin.forward(*S1)
        touch = ctc.body_distance(S1, boxes1, fk1) <= ctc.cfg.touch_dist
        label = ctc.labels[f1, out["v"]]
        stats["tp"].append((touch & label).sum()), stats["n_touch"].append(touch.sum()), stats["n_label"].append(label.sum())
        err = placement_error(ctc.contact_point(S1, boxes1, fk1), tw_win, tv_win)
        onset = touch & ~prev_touch & err.isfinite()
        for key, sel in (("plan", onset & use[:, None]), ("ref", onset & ~use[:, None])):
            stats[f"err_{key}"].append(torch.where(sel, err, 0.0).sum()), stats[f"n_{key}"].append(sel.sum())
        stats["use"].append(use.float().mean())
        stats["r"].append(out["r"].mean() / MAX_STEP_REWARD)
        done = out["done"]
        prev_touch = torch.where(done[:, None], True, touch)  # no onset on an episode's first step
        tv_win[:, done] = False
        envs = done.nonzero(as_tuple=False).squeeze(-1)
        if len(envs) > 0:
            stats["len"].append(out["ep_len"].sum()), stats["n_ep"].append(out["ep_len"].new_tensor(len(envs)))
            stats["ret"].append(out["ep_ret_norm"].sum()), stats["term"].append(out["term"][done].float().sum())
            row = torch.full_like(envs, (buffer.ptr - 1) % buffer.T)
            d, n_adm = label_episodes(buffer, envs, row, out["ep_step"][envs] + 1, ctc, scan, pcfg)
            stats["ended"].append(len(envs)), stats["admitted"].append(n_adm)
            n_ended, n_admitted = n_ended + len(envs), n_admitted + n_adm
            if d is not None:
                val = val_clip[d["clip"]]
                vbuf.add({k: x[val] for k, x in d.items()})
                d = {k: x[~val] for k, x in d.items()}
                if len(d["occ"]) > 0:
                    if pbuf.size >= pcfg.min_entries:  # the planner on data it has not trained on yet
                        (l_aff, l_occ), _ = planner_batch_loss(planner, {k: x[:4096] for k, x in d.items()}, scan, pcfg)
                        stats["fresh"].append(torch.stack((l_aff, l_occ)))
                    planner.prop_norm.update(d["prop"].float())
                    pbuf.add(d)
                    stats["labeled"].append(len(d["occ"]))
        goal_steps[done] = ctc.sample_goal_steps(int(done.sum()))


stats_keys = ("r", "len", "ret", "n_ep", "term", "tp", "n_touch", "n_label", "err_plan", "n_plan", "err_ref", "n_ref", "use",
              "ended", "admitted", "labeled", "fresh", "pl")  # fmt: skip
stats = {k: [] for k in stats_keys}
collect(args.warmup_steps, stats, eta(sched_it, args.u_obs))  # the replay buffer is not checkpointed: a resumed run refills it
BODIES = list(CONTACT_BODIES)
csv_path = os.path.join(args.log_dir, "train_log.csv")
new_log = not os.path.exists(csv_path) or os.path.getsize(csv_path) == 0
log_f = open(csv_path, "a")
val_f = open(os.path.join(args.log_dir, "planner_val.csv"), "a")
if new_log:
    log_f.write("iteration,wall_h,sim_steps,wm_loss,wm_err8,pi_loss,v_loss,imag_rew,pi_gradnorm,imag_survival,"
                "sim_rew,ep_len,ep_ret_norm,term_frac,touch_precision,touch_recall,it_per_s,lr,eta_obs,eta_rew,plan_frac,"
                "onset_err_plan,onset_err_ref,ended,admitted,labeled,planner_buffer,planner_aff,planner_occ,fresh_aff,fresh_occ,"
                "sched_it,admit_rate_window,held\n")  # fmt: skip
    val_f.write("iteration,aff,occ," + ",".join(f"{m}_{b}" for m in ("precision", "recall", "f1", "median_err", "within_0.1") for b in BODIES) + "\n")
    log_f.flush(), val_f.flush()
t_start = time.time() - wall_offset
t_log = time.time()
step_acc = 0.0
agg = {"wm": [], "wm8": [], "pi": [], "v": [], "ir": [], "gn": [], "surv": []}
for it in range(start_it, args.iterations):
    lr = lr_at(it)
    for o in (opt_wm, opt_pi, opt_v):
        for gparam in o.param_groups:
            gparam["lr"] = lr
    e_obs, e_rew = eta(sched_it, args.u_obs), min(eta(sched_it, args.u_rew), args.eta_rew_max)
    step_acc += 1.0 / args.utd
    n = int(step_acc)
    step_acc -= n
    collect(n, stats, e_obs)
    for _ in range(args.wm_batches):
        l, e8 = wm_update()
    agg["wm"].append(l), agg["wm8"].append(e8)
    for k, x in zip(("pi", "v", "ir", "gn", "surv"), policy_update(e_obs, e_rew)):
        agg[k].append(x)
    if pbuf.size >= pcfg.min_entries:
        for _ in range(args.planner_updates):
            stats["pl"].append(planner_update(planner, opt_pl, pbuf, scan, pcfg))
    # the schedules' clock: with --hold_admit it stops while fewer episodes than that share of the eta=0 rate are admitted
    adm_hist.append((n_ended, n_admitted))
    rate = admit_rate()
    if admit_base is None and sched_it + 1 >= args.u_obs and rate:  # (a rate of 0: nothing finished well yet)
        admit_base = rate
    held = args.hold_admit > 0 and sched_it >= args.u_obs and admit_base is not None and (rate is None or rate < args.hold_admit * admit_base)
    if not held:
        sched_it += 1
    if (it + 1) % args.log_every == 0:
        mean = lambda x: float(sum(float(v) for v in x) / max(len(x), 1))  # noqa: E731
        total = lambda k: float(sum(float(v) for v in stats[k]))  # noqa: E731
        ratio = lambda a, b: total(a) / max(total(b), 1)  # noqa: E731
        ratio_nan = lambda a, b: total(a) / total(b) if total(b) > 0 else float("nan")  # noqa: E731
        pair = lambda k: torch.stack(stats[k]).mean(0).tolist() if stats[k] else [float("nan")] * 2  # noqa: E731
        ips = args.log_every / (time.time() - t_log)
        t_log = time.time()
        wall_h = (time.time() - t_start) / 3600
        row = [it + 1, wall_h, sim_steps, mean(agg["wm"]), mean(agg["wm8"]), mean(agg["pi"]), mean(agg["v"]), mean(agg["ir"]),
               mean(agg["gn"]), mean(agg["surv"]), mean(stats["r"]), ratio("len", "n_ep"), ratio("ret", "n_ep"),
               ratio("term", "n_ep"), ratio("tp", "n_touch"), ratio("tp", "n_label"), ips, lr, e_obs, e_rew, mean(stats["use"]),
               ratio_nan("err_plan", "n_plan"), ratio_nan("err_ref", "n_ref"), int(total("ended")), int(total("admitted")),
               int(total("labeled")), pbuf.size, *pair("pl"), *pair("fresh"), sched_it, float("nan") if rate is None else rate,
               int(held)]  # fmt: skip
        log_f.write(",".join(f"{x:.6g}" if isinstance(x, float) else str(x) for x in row) + "\n")
        log_f.flush()
        print(f"it {it + 1} | {wall_h:.3f} h | wm {row[3]:.4f} e8 {row[4]:.4f} | pi {row[5]:.3f} v {row[6]:.4f} imag_r {row[7]:.3f} "
              f"surv {row[9]:.3f} | sim_r {row[10]:.3f} ep_len {row[11]:.1f} ep_ret {row[12]:.3f} term {row[13]:.2f} "
              f"touch P/R {row[14]:.3f}/{row[15]:.3f} | eta {e_obs:.2f}/{e_rew:.2f} onset err plan {row[21]:.3f} ref {row[22]:.3f} "
              f"| admitted {row[24]}/{row[23]} buf {pbuf.size} aff {row[27]:.3f} occ {row[28]:.3f} fresh {row[29]:.3f}/{row[30]:.3f} "
              f"| sched {sched_it}{' HELD' if held else ''} admit {row[32]:.3f} | {ips:.2f} it/s "
              f"| torch mem {torch.cuda.memory_reserved(dev) / 2**30:.1f}/{torch.cuda.max_memory_reserved(dev) / 2**30:.1f} GiB", flush=True)  # fmt: skip
        stats = {k: [] for k in stats_keys}
        agg = {k: [] for k in agg}
    if (it + 1) % args.val_every == 0 and vbuf.size > 0:
        planner.eval()
        mt = planner_metrics(planner, vbuf.sample(min(16384, vbuf.size)), scan, pcfg)
        planner.train()
        keys = ("precision", "recall", "f1", "median_err", "within_0.1")
        val_f.write(",".join(str(x) for x in [it + 1, mt["aff"], mt["occ"]] + [v for k in keys for v in mt[k]]) + "\n")
        val_f.flush()
    if (it + 1) % args.ckpt_every == 0 or it + 1 == args.iterations:
        ck = {"wm": wm.state_dict(), "pi": pi.state_dict(), "critic": critic.state_dict(), "planner": planner.state_dict(),
              "opt_wm": opt_wm.state_dict(), "opt_pi": opt_pi.state_dict(), "opt_v": opt_v.state_dict(),
              "opt_planner": opt_pl.state_dict(), "iteration": it + 1, "wall_time": time.time() - t_start, "sim_steps": sim_steps,
              "args": vars(args), "init_args": init_args, "ctc_cfg": dataclasses.asdict(ctc_cfg),
              "reward_cfg": dataclasses.asdict(reward_cfg), "planner_cfg": dataclasses.asdict(pcfg), "lr_it0": lr_it0,
              "eta": (e_obs, e_rew), "sched_it": sched_it, "admit_base": admit_base, "saved_at": time.strftime("%Y-%m-%d %H:%M:%S")}  # fmt: skip
        last = os.path.join(args.log_dir, "last.pt")
        torch.save(ck, last + ".tmp")
        os.replace(last + ".tmp", last)
        small = {k: ck[k] for k in ("pi", "wm", "planner", "iteration", "wall_time", "sim_steps", "args", "init_args", "ctc_cfg",
                                    "planner_cfg", "eta", "saved_at")}  # fmt: skip
        path = os.path.join(args.log_dir, f"ckpt_{it + 1:06d}.pt")
        torch.save(small, path + ".tmp")
        os.replace(path + ".tmp", path)

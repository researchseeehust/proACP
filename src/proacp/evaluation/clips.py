"""Evaluate a contact tracker (proacp.train.stage1 / stage2 checkpoint) on every clip of its motion set.

Protocol of supermimic_vibed's ``supermimic/evaluate.py``: each clip plays once from its first frame on its first valid terrain variant, no
randomization and no noise, and fails if the rollout triggers the training termination; normalized return is the
motion-tracking reward sum over its maximum. The contact command is noise-free with the goal 1 s ahead. Contact
metrics count, over the frames the rollout is alive, the bodies touching the terrain (sphere approximation within
``touch_dist``) against the reference labels: precision and recall per body. ``--planner`` gives the tracker the
contact planner's targets (a train.planner_offline or train.stage2 checkpoint; noise-free height scan) instead of the
reference's, with probability ``--eta_obs`` per step and clip. Placement error: per body, the distance from where each
new contact lands to the nearest target given over the last :data:`proacp.tasks.contact.PLACE_WINDOW` steps.

    python -m proacp.evaluation.clips --ckpt logs/acp_stage2/ckpt_040000.pt --planner logs/acp_stage2/ckpt_040000.pt
"""

from __future__ import annotations

import argparse
import json

parser = argparse.ArgumentParser()
parser.add_argument("--ckpt", type=str, required=True)
parser.add_argument("--goal_s", type=float, default=1.0)
parser.add_argument("--out", type=str, default=None, help="results json (default: next to the checkpoint)")
parser.add_argument("--record", type=str, default=None, help="save trajectories (npz)")
parser.add_argument("--planner", type=str, default=None, help="contact targets from this checkpoint's planner")
parser.add_argument("--eta_obs", type=float, default=1.0, help="with --planner: per step and clip, the planner's targets with this probability")
parser.add_argument("--term_window", type=float, default=None, help="s: termination window either side of the frame; default: the checkpoint's, else 0")
parser.add_argument("--train_noise", nargs="*", default=None, choices=["obs", "command", "action"],
                    help="the training noise (all three without a value): observations; contact targets and height scans;"
                         " actions (0.1)")
parser.add_argument("--seed", type=int, default=0)
args = parser.parse_args()
_noise = set() if args.train_noise is None else set(args.train_noise or ("obs", "command", "action"))
NOISE_OBS, NOISE_CMD, NOISE_ACT = "obs" in _noise, "command" in _noise, "action" in _noise

import numpy as np
import torch

import proacp.models.networks as networks
from proacp.sim.env import TrackingSim
from proacp.robot import CONTACT_BODIES, MOTION_DIR, NB, default_joint_pos_list
from proacp.sim.kinematics import Kinematics
from proacp.tasks.tracking import MAX_STEP_REWARD, MDP, OBS_DIM, REF_ORI_DIMS, RewardCfg
from proacp.clips.library import CONTROL_DT, MotionLib
from proacp.models.networks import Policy
from proacp.models.planner import ContactPlanner, PlannerCfg, PlannerScan, plan
from proacp.tasks.contact import PLACE_WINDOW, ContactCfg, ContactTask, placement_error
from proacp.sim.rotations import mat_to_quat

torch.backends.cuda.matmul.allow_tf32 = True
torch.manual_seed(args.seed)
dev = "cuda:0"
ck = torch.load(args.ckpt, map_location=dev)
a_ = ck["args"]
ia = ck.get("init_args", a_)
motion_dir = ia.get("motion_dir") or MOTION_DIR
kin = Kinematics(dev)
motion = MotionLib.from_dir(motion_dir, kin, dev)
term_win = a_.get("term_window", 0.0) if args.term_window is None else args.term_window
mdp = MDP(kin, motion, torch.tensor(default_joint_pos_list(), device=dev), RewardCfg(term_window=int(round(term_win / CONTROL_DT))))
ctc = ContactTask(ContactCfg(**ck["ctc_cfg"]), mdp, motion_dir, dev)
Hh = ia["hist"]
networks.AMP_DTYPE = torch.bfloat16 if a_.get("amp", "none") == "bf16" else None
pi = Policy(OBS_DIM, ctc.cmd_dim, hist=Hh).to(dev)
pi.load_state_dict(ck["pi"])
pi.eval()

if args.planner:
    pk = torch.load(args.planner, map_location=dev)
    pcfg = PlannerCfg(**pk["planner_cfg"])
    scan = PlannerScan(pcfg, dev)
    planner = ContactPlanner(OBS_DIM - REF_ORI_DIMS, pcfg).to(dev)
    planner.load_state_dict(pk["planner"])
    planner.eval()

C = len(motion.names)
FOLDERS = sorted({n.split("/")[0] for n in motion.names})
folder_of = torch.tensor([FOLDERS.index(n.split("/")[0]) for n in motion.names], device=dev)
env = TrackingSim(C, motion, mdp, dev, eval_mode=True)


@torch.no_grad()
def evaluate() -> dict:
    ids = torch.arange(C, device=dev)
    env.reset(ids, motion.clip_start.clone())
    clip_end = motion.clip_end
    gs = torch.full((C,), int(round(args.goal_s / CONTROL_DT)), device=dev)
    ret = torch.zeros(C, device=dev)
    tp, n_touch, n_label = (torch.zeros(NB, device=dev) for _ in range(3))
    prev_touch = torch.ones(C, NB, dtype=torch.bool, device=dev)
    tw_win = torch.zeros(PLACE_WINDOW, C, NB, 3, device=dev)
    tv_win = torch.zeros(PLACE_WINDOW, C, NB, dtype=torch.bool, device=dev)
    place = []  # (C, NB) per step: placement error at new contacts, NaN elsewhere
    hist = None
    traj = {"root_p": [], "root_quat": [], "q": [], "f": [], "alive": [], "tgt_w": [], "tgt_valid": []}
    while not bool(env.done_eval.all()):
        S = env.state()
        o = env.observe()
        if NOISE_OBS:
            o = mdp.add_noise(o)
        hist = o.unsqueeze(1).repeat(1, Hh, 1) if hist is None else torch.cat((hist[:, 1:], o.unsqueeze(1)), 1)
        tw, tv = ctc.next_pos[env.f, env.v], ctc.next_valid[env.f, env.v]
        if args.planner:
            pw, pv = plan(planner, scan, pcfg, ctc, hist, S, env.f, env.v, gs, noise=NOISE_CMD)
            use = torch.rand(C, device=dev) < args.eta_obs
            tw, tv = torch.where(use[:, None, None], pw, tw), torch.where(use[:, None], pv, tv)
        cmd = ctc.command(S, env.f, env.v, gs, noise=NOISE_CMD, targets_w=tw, valid=tv)
        alive = ~env.done_eval.clone()
        tw_win, tv_win = torch.cat((tw_win[1:], tw[None])), torch.cat((tv_win[1:], tv[None]))
        if args.record:
            traj["root_p"].append(S[0].cpu()), traj["root_quat"].append(mat_to_quat(S[1]).cpu())
            traj["q"].append(S[4].cpu()), traj["f"].append(env.f.cpu()), traj["alive"].append(alive.cpu())
            traj["tgt_w"].append(tw.cpu()), traj["tgt_valid"].append(tv.cpu())
        a = pi(hist, cmd)
        out = env.step(a + 0.1 * torch.randn_like(a) if NOISE_ACT else a)
        ret += torch.where(alive, out["r"], 0.0)
        f1 = out["f"] + 1
        S1 = env.state()
        boxes1 = motion.boxes(f1, out["v"])
        touch = (ctc.body_distance(S1, boxes1) <= ctc.cfg.touch_dist) & alive.unsqueeze(-1)
        pe = placement_error(ctc.contact_point(S1, boxes1), tw_win, tv_win)
        place.append(torch.where(touch & ~prev_touch & pe.isfinite(), pe, torch.nan))
        prev_touch = touch
        label = ctc.labels[f1, out["v"]] & alive.unsqueeze(-1)
        tp += (touch & label).sum(0)
        n_touch += touch.sum(0)
        n_label += label.sum(0)
        env.f = torch.minimum(env.f, clip_end - 1)  # finished clips keep stepping; keep f+1 in range
    if args.record:
        np.savez_compressed(args.record, names=np.array(motion.names), variant=env.v.cpu().numpy(),
                            boxes=motion.clip_boxes[torch.arange(C, device=dev), env.v].cpu().numpy(),
                            **{k: torch.stack(v).numpy() for k, v in traj.items()}, ref_root_p=motion.root_p.cpu().numpy(),
                            ref_root_R=motion.root_R.cpu().numpy(), ref_q=motion.q.cpu().numpy(),
                            clip_start=motion.clip_start.cpu().numpy(), clip_len=motion.clip_len.cpu().numpy())  # fmt: skip
    norm = ret / ((motion.clip_len - 1).float() * MAX_STEP_REWARD)
    place = torch.stack(place).flatten(0, 1)
    pl = [place[:, b][~place[:, b].isnan()] for b in range(NB)]
    rnd = lambda x: round(x.item(), 3)  # noqa: E731
    failed = env.failed_eval.float()
    return {
        "ckpt": args.ckpt,
        "interface": "planner" if args.planner else "reference",
        "eta_obs": args.eta_obs if args.planner else 0.0,
        "term_window": term_win,
        "norm_return": round(norm.mean().item(), 4),
        "n_failed": int(failed.sum().item()),
        "failed": [motion.names[i] for i in failed.nonzero().flatten().tolist()],
        "per_folder": {g: {"norm_return": round(norm[folder_of == i].mean().item(), 4), "fail_rate": round(failed[folder_of == i].mean().item(), 3)}
                       for i, g in enumerate(FOLDERS)},  # fmt: skip
        "contacts": {n: {"precision": round((tp[b] / n_touch[b]).item(), 3) if n_touch[b] > 0 else None,
                         "recall": round((tp[b] / n_label[b]).item(), 3) if n_label[b] > 0 else None,
                         "label_frames": int(n_label[b].item()), "touch_frames": int(n_touch[b].item()),
                         "placement_err": rnd(pl[b].mean()) if len(pl[b]) else None,
                         "placement_median": rnd(pl[b].median()) if len(pl[b]) else None,
                         "placement_within_0.1": rnd((pl[b] <= 0.1).float().mean()) if len(pl[b]) else None,
                         "onsets": len(pl[b])} for b, n in enumerate(CONTACT_BODIES)},  # fmt: skip
    }


res = evaluate()
print(json.dumps({k: v for k, v in res.items() if k != "failed"}, indent=1))
print(f"failed {res['n_failed']}/{C}: {res['failed'][:10]}{' ...' if res['n_failed'] > 10 else ''}")
tag = res["interface"] + ("" if args.eta_obs == 1 or not args.planner else f"_eta{args.eta_obs:g}") + (f"_tw{term_win:g}" if term_win else "")
out = args.out or args.ckpt.replace(".pt", f"_eval_{tag}.json")
json.dump(res, open(out, "w"), indent=1)
print("wrote", out)

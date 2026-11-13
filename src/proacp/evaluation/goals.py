"""Evaluate goal reaching (a proacp.train.stage3 checkpoint, or zero-shot a train.stage2 one) on the procedural
terrains of :mod:`proacp.tasks.goal`.

Every terrain type and level plays ``--episodes`` of its instances once (no randomization, no noise, the start yaw
unturned), the tracker following the checkpoint's planner. Success, as the paper's curriculum counts it: the robot comes
within 0.5 m horizontally and 0.1 m vertically of the goal and the episode ends without a fall (15 s at most). Reports
per type and level: success rate, fall rate, reach rate and the mean time to the goal of the successful episodes.

    python -m proacp.evaluation.goals --ckpt logs/acp_stage3/ckpt_030000.pt
"""

from __future__ import annotations

import argparse
import json

parser = argparse.ArgumentParser()
parser.add_argument("--ckpt", type=str, required=True)
parser.add_argument("--episodes", type=int, default=4, help="instances per terrain type and level (at most the bank's)")
parser.add_argument("--terrain_seed", type=int, default=None, help="default: the checkpoint's terrains (Stage 2: seed 0)")
parser.add_argument("--out", type=str, default=None, help="results json (default: next to the checkpoint)")
parser.add_argument("--seed", type=int, default=0)
args = parser.parse_args()

import torch

import proacp.models.networks as networks
from proacp.sim.env import TrackingSim
from proacp.robot import MOTION_DIR, default_joint_pos_list
from proacp.tasks.goal import LEVELS, TERRAIN_TYPES, GoalCfg, GoalTask, make_terrains
from proacp.sim.kinematics import Kinematics
from proacp.tasks.tracking import MDP, OBS_DIM, REF_ORI_DIMS
from proacp.clips.library import CONTROL_DT, MotionLib
from proacp.models.networks import Policy
from proacp.models.planner import ContactPlanner, PlannerCfg, PlannerScan, plan
from proacp.tasks.contact import ContactCfg, ContactTask, PaperReward, PaperRewardCfg

torch.backends.cuda.matmul.allow_tf32 = True
torch.manual_seed(args.seed)
dev = "cuda:0"
ck = torch.load(args.ckpt, map_location=dev)
ia = ck["init_args"]
motion_dir = ia.get("motion_dir") or MOTION_DIR
kin = Kinematics(dev)
motion = MotionLib.from_dir(motion_dir, kin, dev)
mdp = MDP(kin, motion, torch.tensor(default_joint_pos_list(), device=dev))
ctc = ContactTask(ContactCfg(**ck["ctc_cfg"]), mdp, motion_dir, dev)
terrain_args = dict(ck.get("terrain_args", {"instances": 16, "seed": 0}))
if args.terrain_seed is not None:
    terrain_args["seed"] = args.terrain_seed
terrains = make_terrains(**terrain_args)
motion.attach_terrains(terrains["boxes"])
gcfg = GoalCfg(**ck["goal_cfg"]) if "goal_cfg" in ck else GoalCfg()
goal = GoalTask(gcfg, ctc, PaperReward(PaperRewardCfg(), ctc), terrains)
Hh = ia["hist"]
networks.AMP_DTYPE = torch.bfloat16 if ck["args"].get("amp", "none") == "bf16" else None
pi = Policy(OBS_DIM, ctc.cmd_dim, hist=Hh).to(dev)
pi.load_state_dict(ck["pi"])
pi.eval()
pcfg = PlannerCfg(**ck["planner_cfg"])
scan = PlannerScan(pcfg, dev)
planner = ContactPlanner(OBS_DIM - REF_ORI_DIMS, pcfg).to(dev)
planner.load_state_dict(ck["planner"])
planner.eval()

E = min(args.episodes, terrain_args["instances"])
idx = torch.tensor([(t * LEVELS + lv) * terrain_args["instances"] + i for t in range(len(TERRAIN_TYPES)) for lv in range(LEVELS) for i in range(E)], device=dev)
N = len(idx)
env = TrackingSim(N, motion, mdp, dev, eval_mode=True, goal=goal, num_goal_envs=N)
env.next_terrains = idx


@torch.no_grad()
def evaluate() -> dict:
    env.reset(torch.arange(N, device=dev))
    is_goal = torch.ones(N, dtype=torch.bool, device=dev)
    reached = torch.zeros(N, dtype=torch.bool, device=dev)
    t_reach = torch.full((N,), float("nan"), device=dev)
    hist = None
    while not bool(env.done_eval.all()):
        S = env.state()
        o = env.observe()
        hist = o.unsqueeze(1).repeat(1, Hh, 1) if hist is None else torch.cat((hist[:, 1:], o.unsqueeze(1)), 1)
        g = goal.command(S, env.goal_w)
        pw, pv = plan(planner, scan, pcfg, ctc, hist, S, env.f, env.v, None, noise=False, goal=g)
        alive = ~env.done_eval.clone()
        env.step(pi(hist, ctc.command(S, env.f, env.v, None, noise=False, targets_w=pw, valid=pv, goal=g)))
        new = alive & ~reached & goal.reached(env.state(), env.goal_w)
        t_reach = torch.where(new, env.ep_step.float() * CONTROL_DT, t_reach)
        reached |= new
    fell = env.failed_eval
    success = reached & ~fell
    typ, lvl = terrains["type"].to(dev)[idx], terrains["level"].to(dev)[idx]
    res = {"ckpt": args.ckpt, "terrain_args": terrain_args, "episodes_per_level": E, "success": round(success.float().mean().item(), 3),
           "fell": round(fell.float().mean().item(), 3), "per_type": {}}  # fmt: skip
    for t, name in enumerate(TERRAIN_TYPES):
        sel = typ == t
        res["per_type"][name] = {
            "success": round(success[sel].float().mean().item(), 3),
            "fell": round(fell[sel].float().mean().item(), 3),
            "reached": round(reached[sel].float().mean().item(), 3),
            "success_per_level": [round(success[sel & (lvl == lv)].float().mean().item(), 2) for lv in range(LEVELS)],
            "time_to_goal_s": round(t_reach[sel & success].mean().item(), 2) if (sel & success).any() else None,
        }
    return res


res = evaluate()
print(f"success {res['success']:.3f}, fell {res['fell']:.3f} over {N} episodes")
for name, r in res["per_type"].items():
    print(f"  {name:16s} success {r['success']:.2f} fell {r['fell']:.2f} reached {r['reached']:.2f} | per level " + " ".join(f"{x:.2f}" for x in r["success_per_level"]))
out = args.out or args.ckpt.replace(".pt", "_eval_goal.json")
json.dump(res, open(out, "w"), indent=1)
print("wrote", out)

"""Motion-tracking simulator on box terrain, on mjlab (MuJoCo-Warp): the simulator of the contact pipeline.

As supermimic_vibed's ``supermimic/env.py`` (50 Hz control, 4 physics substeps of 5 ms, implicitfast, MuJoCo collisions; the paper's
sim context xi = friction, joint offsets, torso CoM offset), plus the terrain: every world holds a fixed set of box
slots, which each reset moves onto the boxes of the episode's clip, in a valid terrain variant of it drawn per
episode (``v``). The simulator's reward is the tracking reward of :meth:`proacp.tasks.tracking.MDP.reward` (start-bin
sampling and logs); the contact policy learns from its own reward in imagination.

Episode starts: SuperMimic's return-based bin sampling; ``backward_curriculum`` the contact paper's Stage 1
curriculum (a uniform clip from its start phase: 0.9 of the clip at first, 5% earlier after an episode reaching 99%
of its clip, 5% later otherwise, per clip); ``start_phase`` (lo, hi) a uniform clip at a uniform phase (Stage 2).
The last two run episodes to the clip's end.

Goal reaching (Stage 3, :mod:`proacp.tasks.goal`): the last ``num_goal_envs`` worlds play the procedurally generated
terrains instead (terrain index v = V + bank index, picked by the goal curriculum), each episode starting from a clip's
standing first frame placed at the course start, turned by U(-0.4, 0.4) rad. Their reference frame stays fixed (nothing
of the reference reaches them), they end on a fall or after 15 s, and their simulator reward is the goal reward.
"""

from __future__ import annotations

import mujoco
import torch
import warp as wp
from mjlab.entity import EntityCfg
from mjlab.scene import Scene, SceneCfg
from mjlab.sim import MujocoCfg, Simulation, SimulationCfg
from mjlab.terrains import TerrainEntityCfg

from proacp.robot import JOINT_NAMES, NUM_J, TORSO_BODY, action_scale, robot_cfg
from proacp.tasks.goal import GoalCurriculum, GoalTask
from proacp.tasks.tracking import MAX_STEP_REWARD, MDP, XI_DIM, flatten_state
from proacp.clips.library import CONTROL_DT, MotionLib
from proacp.train.reset_sampling import update_bin_returns
from proacp.sim.rotations import mat_to_quat, quat_to_mat, so3_exp

DECIMATION = 4  # 4 physics substeps of 5 ms per 50 Hz control step
PHYSICS_DT = CONTROL_DT / DECIMATION
BOX_PARK = (0.0, 0.0, -5.0)
"""Where unused box slots wait (below the ground plane, out of reach)."""
CURRICULUM_START, CURRICULUM_STEP, CURRICULUM_MAX, CURRICULUM_REACH = 0.9, 0.05, 0.99, 0.99
"""Backward curriculum: first start phase, phase step, latest start, success phase."""
FRICTION = (0.3, 1.2)
COM_RANGE = ((-0.025, 0.025), (-0.05, 0.05), (-0.05, 0.05))
JOINT_OFFSET = 0.01
# reset noise half-ranges (as supermimic_vibed's supermimic/env.py)
RESET_POS, RESET_ROT = (0.05, 0.05, 0.01), (0.1, 0.1, 0.2)
RESET_LIN_VEL, RESET_ANG_VEL, RESET_JOINT_POS = (0.5, 0.5, 0.2), (0.52, 0.52, 0.78), 0.1


def obstacles_cfg(n_boxes: int) -> EntityCfg:
    """A static entity with ``n_boxes`` box slots, parked below the ground until :class:`TrackingSim` moves them
    per world. Contact parameters are MuJoCo's defaults, the same as the ground plane's."""

    def spec_fn() -> mujoco.MjSpec:
        spec = mujoco.MjSpec()
        body = spec.worldbody.add_body(name="obstacles")
        for k in range(n_boxes):
            body.add_geom(name=f"box{k}", type=mujoco.mjtGeom.mjGEOM_BOX, size=(0.01, 0.01, 0.01), pos=BOX_PARK,
                          rgba=(0.62, 0.55, 0.45, 1.0))  # fmt: skip
        return spec

    return EntityCfg(spec_fn=spec_fn)


def make_scene_cfg(num_envs: int, n_boxes: int) -> SceneCfg:
    return SceneCfg(num_envs=num_envs, env_spacing=0.0, terrain=TerrainEntityCfg(terrain_type="plane"),
                    entities={"robot": robot_cfg(), "obstacles": obstacles_cfg(n_boxes)})  # fmt: skip


def make_sim(njmax: int = 500, nconmax: int = 128) -> SimulationCfg:
    return SimulationCfg(njmax=njmax, nconmax=nconmax,
                         mujoco=MujocoCfg(timestep=PHYSICS_DT, integrator="implicitfast", cone="pyramidal", solver="newton",
                                          iterations=100, ls_iterations=50, tolerance=1e-6, ccd_iterations=35))  # fmt: skip


class TrackingSim:
    """Batched G1 motion-tracking simulator on the clips' box terrain."""

    def __init__(self, num_envs: int, motion: MotionLib, mdp: MDP, device: str, gamma: float = 0.99, eval_mode: bool = False,
                 max_episode_steps: int = 500, reset_noise: bool = True, backward_curriculum: bool = False,
                 start_phase: tuple[float, float] | None = None, sim_cfg: SimulationCfg | None = None, goal: GoalTask | None = None,
                 num_goal_envs: int = 0, goal_curriculum: GoalCurriculum | None = None, start_yaw: float = 0.4):  # fmt: skip
        self.device = dev = device
        n_boxes = motion.clip_boxes.shape[2]
        self.scene = Scene(make_scene_cfg(num_envs, n_boxes), device=device)
        self.sim = Simulation(num_envs=num_envs, cfg=sim_cfg or make_sim(), model=self.scene.compile(), device=device)
        self.scene.initialize(mj_model=self.sim.mj_model, model=self.sim.model, data=self.sim.data)
        self.robot = self.scene["robot"]
        assert list(self.robot.joint_names) == JOINT_NAMES, self.robot.joint_names
        self.N = num_envs
        self.motion, self.mdp, self.gamma = motion, mdp, gamma
        self.eval_mode = eval_mode
        self.curriculum = backward_curriculum and not eval_mode
        self.start_phase = None if eval_mode else start_phase
        assert not (self.curriculum and self.start_phase), "one episode-start rule"
        self.max_episode_steps = max_episode_steps if not (eval_mode or self.curriculum or self.start_phase) else 10**9
        self.reset_noise = reset_noise and not eval_mode
        self.default_q = self.robot.data.default_joint_pos[0].clone()
        self.alpha = torch.tensor([action_scale(n) for n in JOINT_NAMES], device=dev)

        # ---- box slots, placed on the episode's clip and variant at every reset ----
        self.box_geoms = self.scene["obstacles"].indexing.geom_ids
        m = self.sim.mj_model
        plane = torch.tensor([i for i in range(m.ngeom) if m.geom_type[i] == mujoco.mjtGeom.mjGEOM_PLANE], device=dev)
        self._terrain_geoms = torch.cat((plane, self.box_geoms.to(dev)))
        torso = self.robot.indexing.body_ids[self.robot.find_bodies(TORSO_BODY)[0][0]]
        self.sim.expand_model_fields(("geom_pos", "geom_quat", "geom_size", "geom_rbound", "geom_aabb", "geom_friction", "body_ipos"))

        # ---- sim context xi = [friction, joint offsets (29), torso CoM offset (3)], friction and CoM once per env ----
        self.xi = torch.zeros(self.N, XI_DIM, device=dev)
        # terrain (ground and boxes) at the lowest friction of the range, since MuJoCo takes the max of a pair
        self.sim.model.geom_friction[:, self._terrain_geoms, 0] = 1.0 if eval_mode else FRICTION[0]
        self.xi[:, 0] = 1.0 if eval_mode else torch.empty(self.N, device=dev).uniform_(*FRICTION)
        self.sim.model.geom_friction[:, self.robot.indexing.geom_ids, 0] = self.xi[:, :1]
        if not eval_mode:
            com = torch.stack([torch.empty(self.N, device=dev).uniform_(lo, hi) for lo, hi in COM_RANGE], -1)
            self.xi[:, 1 + NUM_J :] = com
            self.sim.model.body_ipos[:, torso] = self.sim.get_default_field("body_ipos")[torso] + com

        # ---- episode bookkeeping ----
        self.f = torch.zeros(self.N, dtype=torch.long, device=dev)
        self.v = torch.zeros(self.N, dtype=torch.long, device=dev)  # terrain variant
        self.ep_step = torch.zeros(self.N, dtype=torch.long, device=dev)
        self.a_prev = torch.zeros(self.N, NUM_J, device=dev)
        self.start_bin = torch.zeros(self.N, dtype=torch.long, device=dev)
        self.ep_disc_ret = torch.zeros(self.N, device=dev)
        self.ep_ret = torch.zeros(self.N, device=dev)
        self.G_max = MAX_STEP_REWARD / (1 - gamma)
        self.bin_G = torch.full((motion.num_bins,), 0.0, device=dev)
        self.bin_w_avg = torch.full((motion.num_bins,), 1.0 / motion.num_bins, device=dev)
        self.phase = torch.full((len(motion.clip_len),), CURRICULUM_START, device=dev)  # backward curriculum, per clip
        self.done_eval = torch.zeros(self.N, dtype=torch.bool, device=dev)
        self.failed_eval = torch.zeros(self.N, dtype=torch.bool, device=dev)
        self._obs = None

        # ---- goal reaching (Stage 3): the last num_goal_envs worlds ----
        self.goal, self.goal_curriculum, self.start_yaw = goal, goal_curriculum, start_yaw
        self.is_goal = torch.zeros(self.N, dtype=torch.bool, device=dev)
        self.N_ref = self.N - num_goal_envs
        if goal is not None and num_goal_envs > 0:
            self.is_goal[self.N_ref :] = True
            m = motion
            still = m.root_v[m.clip_start].norm(dim=-1) < 0.05  # clips that start standing still
            self.start_frames = m.clip_start[still] if still.any() else m.clip_start
        self.goal_w = torch.zeros(self.N, 4, device=dev)  # (x, y, pelvis height, heading)
        self.reached = torch.zeros(self.N, dtype=torch.bool, device=dev)  # the goal reached at some step
        self.min_dist = torch.full((self.N,), float("inf"), device=dev)
        self.first_close = torch.full((self.N,), -1, dtype=torch.long, device=dev)  # step first within close_xy
        self.next_terrains = None  # evaluation: bank terrains of the goal worlds' next reset
        self.reset(torch.arange(self.N, device=dev))

    @property
    def num_envs(self) -> int:
        return self.N

    # ------------------------------------------------------------------ state
    def state(self):
        # root state from the free joint (qpos/qvel are current after a step; derived xpos lag a substep)
        idx = self.robot.indexing
        root_q = self.sim.data.qpos[:, idx.free_joint_q_adr]
        root_v = self.sim.data.qvel[:, idx.free_joint_v_adr]
        p = root_q[:, :3] - self.scene.env_origins
        R = quat_to_mat(root_q[:, 3:7])  # MuJoCo uses wxyz
        # free joint velocity: world-frame linear velocity of the root origin, angular velocity in the root frame
        w = (R @ root_v[:, 3:6].unsqueeze(-1)).squeeze(-1)
        d = self.robot.data
        return (p, R, root_v[:, :3].clone(), w, d.joint_pos.clone(), d.joint_vel.clone())

    def boxes(self) -> torch.Tensor:
        """(N, K, 8) every world's current boxes."""
        return self.motion.boxes(self.f, self.v)

    # ------------------------------------------------------------------ reset
    def sample_frames(self, n: int) -> tuple[torch.Tensor, torch.Tensor]:
        m = self.motion
        if self.curriculum:  # a uniform clip from its start phase
            clips = torch.randint(0, len(m.clip_len), (n,), device=self.device)
            f = m.clip_start[clips] + (self.phase[clips] * (m.clip_len[clips] - 1).float()).long()
            return f, m.frame_bin[f]
        if self.start_phase is not None:  # a uniform clip at a uniform phase
            clips = torch.randint(0, len(m.clip_len), (n,), device=self.device)
            u = torch.empty(n, device=self.device).uniform_(*self.start_phase)
            f = m.clip_start[clips] + (u * (m.clip_len[clips] - 1).float()).long()
            return f, m.frame_bin[f]
        soft = torch.softmax(-25.0 * self.bin_G / self.G_max, 0)
        self.bin_w_avg.mul_(0.999).add_(0.001 * soft)
        probs = 0.5 / m.num_bins + 0.4 * soft + 0.1 * self.bin_w_avg / self.bin_w_avg.sum()
        bins = torch.multinomial(probs, n, replacement=True)
        start = m.bin_start[bins]
        end = torch.minimum(start + 50, m.clip_end[m.bin_clip[bins]])  # keep >= 1 step before the clip end
        f = start + (torch.rand(n, device=self.device) * (end - start).float()).long()
        return f, bins

    def reset(self, env_ids: torch.Tensor, frames: torch.Tensor | None = None, variants: torch.Tensor | None = None):
        """Start episodes of ``env_ids`` on reference ``frames`` (default: sampled) in terrain ``variants`` (default:
        a valid one, uniform in training, each clip's first in evaluation)."""
        if len(env_ids) == 0:
            return
        n = len(env_ids)
        dev, m = self.device, self.motion
        if frames is None:
            frames, bins = self.sample_frames(n)
        else:
            bins = m.frame_bin[frames]
        goal_rows = self.is_goal[env_ids]
        if goal_rows.any():  # goal worlds: a standing first frame and a bank terrain
            gi = goal_rows.nonzero(as_tuple=False).squeeze(-1)
            frames = frames.clone()
            frames[gi] = self.start_frames[torch.randint(0, len(self.start_frames), (len(gi),), device=dev)]
            if self.next_terrains is not None:
                g = self.next_terrains[env_ids[gi] - self.N_ref]
            elif self.goal_curriculum is not None:
                g = self.goal_curriculum.sample(env_ids[gi] - self.N_ref)
            else:
                g = torch.randint(0, len(self.goal.terrains["type"]), (len(gi),), device=dev)
        clips = m.clip_id[frames]
        if variants is None:
            variants = m.first_variants(clips) if self.eval_mode else m.sample_variants(clips)
        p, R, v, w, q, qd = (x.clone() for x in m.ref_state(frames))
        if goal_rows.any():  # the frame turned and moved onto the course start
            T = self.goal.terrains
            variants = variants.clone()
            variants[gi] = m.num_variants + g
            st = T["start"][g]
            yaw0 = torch.atan2(R[gi, 1, 0], R[gi, 0, 0])
            turn = st[:, 3] - yaw0 + (0.0 if self.eval_mode else torch.empty(len(gi), device=dev).uniform_(-self.start_yaw, self.start_yaw))
            zero = torch.zeros_like(turn)
            Rz = so3_exp(torch.stack((zero, zero, turn), -1))
            R[gi], v[gi], w[gi] = Rz @ R[gi], (Rz @ v[gi].unsqueeze(-1)).squeeze(-1), (Rz @ w[gi].unsqueeze(-1)).squeeze(-1)
            p[gi] = torch.cat((st[:, :2], p[gi, 2:] + st[:, 2:3]), -1)
            self.goal_w[env_ids[gi]] = T["goal"][g]
        if self.reset_noise:
            u = lambda half: torch.stack([torch.empty(n, device=dev).uniform_(-h, h) for h in half], -1)  # noqa: E731
            p = p + u(RESET_POS)
            R = so3_exp(u(RESET_ROT)) @ R
            v = v + u(RESET_LIN_VEL)
            w = w + u(RESET_ANG_VEL)
            q = q + torch.empty_like(q).uniform_(-RESET_JOINT_POS, RESET_JOINT_POS)
        q = torch.clamp(q, self.mdp.kin.q_lo, self.mdp.kin.q_hi)
        # mjlab's reset clears solver warm-start data and actuator targets, so it runs before the writes
        self.sim.reset(env_ids)
        self.scene.reset(env_ids)
        self._write_boxes(env_ids, clips, variants)
        pose = torch.cat((p + self.scene.env_origins[env_ids], mat_to_quat(R)), -1)
        self.robot.write_root_link_pose_to_sim(pose, env_ids=env_ids)
        self.robot.write_root_link_velocity_to_sim(torch.cat((v, w), -1), env_ids=env_ids)
        self.robot.write_joint_position_to_sim(q, env_ids=env_ids)
        self.robot.write_joint_velocity_to_sim(qd, env_ids=env_ids)
        # reset noise near a box pushes a foot or hand into it: lift the robot by its deepest terrain contact
        pose[:, 2] += self._terrain_penetration(env_ids)
        self.robot.write_root_link_pose_to_sim(pose, env_ids=env_ids)
        self.robot.set_joint_position_target(q, env_ids=env_ids)
        self.f[env_ids] = frames
        self.v[env_ids] = variants
        self.ep_step[env_ids] = 0
        self.a_prev[env_ids] = 0.0
        self.start_bin[env_ids] = bins
        self.ep_disc_ret[env_ids] = 0.0
        self.ep_ret[env_ids] = 0.0
        self.reached[env_ids] = False
        self.min_dist[env_ids] = float("inf")
        self.first_close[env_ids] = -1
        if not self.eval_mode:
            self.xi[env_ids, 1 : 1 + NUM_J] = torch.empty(n, NUM_J, device=dev).uniform_(-JOINT_OFFSET, JOINT_OFFSET)
        self._obs = None

    def _write_boxes(self, env_ids: torch.Tensor, clips: torch.Tensor, variants: torch.Tensor):
        """Move the box slots of worlds ``env_ids`` onto the boxes of ``clips`` in ``variants`` (the rest stay parked),
        with the broadphase bounds of their new sizes (as mjlab's ``dr.geom_size``)."""
        b = self.motion.boxes_of(clips, variants)  # (n, K, 8): cx, cy, top z, cos yaw, sin yaw, half-sizes
        used = (b[..., 5] > 0).unsqueeze(-1)
        pos = torch.stack((b[..., 0], b[..., 1], b[..., 2] - b[..., 7]), -1) + self.scene.env_origins[env_ids].unsqueeze(1)
        half_yaw = 0.5 * torch.atan2(b[..., 4], b[..., 3])
        zero = torch.zeros_like(half_yaw)
        quat = torch.stack((torch.cos(half_yaw), zero, zero, torch.sin(half_yaw)), -1)  # wxyz
        size = torch.where(used, b[..., 5:8], 0.01)
        model, ids = self.sim.model, (env_ids[:, None], self.box_geoms[None, :])
        model.geom_pos[ids] = torch.where(used, pos, torch.tensor(BOX_PARK, device=self.device))
        model.geom_quat[ids] = torch.where(used, quat, torch.tensor((1.0, 0.0, 0.0, 0.0), device=self.device))
        model.geom_size[ids] = size
        model.geom_rbound[ids] = size.norm(dim=-1)
        model.geom_aabb[env_ids[:, None], self.box_geoms[None, :], 1] = size

    def _terrain_penetration(self, env_ids: torch.Tensor) -> torch.Tensor:
        """(n,) depth of the deepest robot-terrain (plane or box) contact of worlds ``env_ids`` in the current
        state, 0 without one."""
        self.sim.forward()
        d = self.sim.wp_data
        dist, geom, world = wp.to_torch(d.contact.dist), wp.to_torch(d.contact.geom), wp.to_torch(d.contact.worldid).long()
        valid = torch.arange(len(dist), device=self.device) < wp.to_torch(d.nacon)
        terrain = torch.isin(geom, self._terrain_geoms)
        pair = valid & (terrain[:, 0] ^ terrain[:, 1]) & torch.isin(geom, self.robot.indexing.geom_ids).any(-1)
        depth = torch.zeros(self.N, device=self.device)
        depth.scatter_reduce_(0, torch.where(pair, world, 0), torch.where(pair, -dist, 0.0), "amax")
        return depth[env_ids]

    def refresh(self):
        """Recompute derived quantities (body poses) so the robot's mjlab data match ``state()`` after a reset."""
        self.sim.forward()

    # ------------------------------------------------------------------ observation
    def observe(self) -> torch.Tensor:
        """Policy observation of the current state (training: noisy; eval: noise-free)."""
        o = self.mdp.obs(self.state(), self.a_prev, self.xi)
        self._obs = o if self.eval_mode else self.mdp.add_noise(o)
        return self._obs

    # ------------------------------------------------------------------ step
    def step(self, a: torch.Tensor) -> dict:
        o = self._obs if self._obs is not None else self.observe()
        self._obs = None
        S = self.state()
        f, v = self.f.clone(), self.v.clone()
        xi = self.xi.clone()
        a_prev = self.a_prev.clone()
        ep_step = self.ep_step.clone()
        target = self.default_q + xi[:, 1 : 1 + NUM_J] + self.alpha * a
        self.robot.set_joint_position_target(target)
        for _ in range(DECIMATION):
            self.scene.write_data_to_sim()
            self.sim.step()
            self.scene.update(PHYSICS_DT)
        S1 = self.state()
        g = self.is_goal
        f1 = torch.where(g, f, f + 1)  # goal worlds keep their (unused) reference frame
        r = self.mdp.reward(S1, f1, a, a_prev)
        term = self.mdp.termination(S1, f1)
        clip_end = self.motion.clip_end[self.motion.clip_id[f]]
        timeout = (f1 >= clip_end) | (ep_step + 1 >= self.max_episode_steps)
        goal_w = self.goal_w.clone()
        if self.goal is not None and g.any():
            boxes1 = self.motion.boxes(f1, v)
            r = torch.where(g, self.goal.sim_reward(S1, goal_w, self.goal.v_max(v)), r)
            term = torch.where(g, self.goal.termination(S1, boxes1), term)
            timeout = torch.where(g, ep_step + 1 >= self.goal.cfg.max_steps, timeout)
            dg, dz, _ = self.goal.offsets(S1, goal_w)
            self.reached |= g & (dg < self.goal.cfg.reach_xy) & (dz.abs() < self.goal.cfg.reach_z)
            self.min_dist = torch.where(g, torch.minimum(self.min_dist, dg), self.min_dist)
            close = g & (self.first_close < 0) & (dg < self.goal.cfg.close_xy)
            self.first_close = torch.where(close, ep_step + 1, self.first_close)
        done = term | timeout
        out = {"s": flatten_state(S), "a": a.clone(), "a_prev": a_prev, "f": f, "v": v, "s1": flatten_state(S1), "r": r, "o": o,
               "term": term, "done": done, "ep_step": ep_step, "xi": xi, "goal": goal_w}  # fmt: skip
        # bookkeeping
        self.ep_disc_ret += (self.gamma ** ep_step.float()) * r
        self.ep_ret += r
        self.f = f1
        self.ep_step = ep_step + 1
        self.a_prev = a.clone()
        if self.eval_mode:
            self.failed_eval |= term & ~self.done_eval
            self.done_eval |= done
            return out
        ids = done.nonzero(as_tuple=False).squeeze(-1)
        if len(ids) > 0:
            # bin return estimate; truncated episodes bootstrap with their mean reward
            T = self.ep_step[ids].float()
            boot = torch.where(term[ids], 0.0, (self.gamma**T) * (self.ep_ret[ids] / T) / (1 - self.gamma))
            G = self.ep_disc_ret[ids] + boot
            ref = ~g[ids]
            update_bin_returns(self.bin_G, self.start_bin[ids[ref]], G[ref])
            if self.curriculum:  # every finished episode moves its clip's start phase
                m, rid = self.motion, ids[ref]
                c = m.clip_id[f[rid]]
                reached = (f1[rid] - m.clip_start[c]).float() >= CURRICULUM_REACH * (m.clip_len[c] - 1).float()
                self.phase.index_add_(0, c, torch.where(reached, -CURRICULUM_STEP, CURRICULUM_STEP))
                self.phase.clamp_(0.0, CURRICULUM_MAX)
            out["ep_len"] = T
            max_r = MAX_STEP_REWARD if self.goal is None else torch.where(ref, MAX_STEP_REWARD, (self.goal.cfg.w_move + self.goal.cfg.w_stand) * CONTROL_DT)
            out["ep_ret_norm"] = self.ep_ret[ids] / (T * max_r)
            if self.goal is not None:
                out["ep_goal"], out["ep_reached"], out["ep_first_close"] = ~ref, self.reached[ids].clone(), self.first_close[ids].clone()
                out["ep_min_dist"] = self.min_dist[ids].clone()
                gid = ids[~ref]
                if len(gid) > 0 and self.goal_curriculum is not None:  # success: reached, and no early termination
                    self.goal_curriculum.update(gid - self.N_ref, self.reached[gid] & ~term[gid], self.min_dist[gid])
            self.reset(ids)
        return out

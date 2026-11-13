"""Fixed-size GPU replay buffer of simulator transitions, laid out as (time, env) (as supermimic_vibed's ``supermimic/buffer.py``).

It also records what the contact pipeline needs per transition: the observation the policy saw (``o``), the
episode's terrain (``v``), its goal horizon (``goal_steps``) and, in goal reaching, its goal (``goal``), plus any
``extra`` fields. It doubles as the trajectory recorder of the planner's hindsight labels, so it should hold the
longest episode.
"""

from __future__ import annotations

import torch

from proacp.robot import NUM_J
from proacp.tasks.tracking import OBS_DIM, STATE_DIM, XI_DIM


def fields(extra: dict | None = None) -> dict:
    return {
        "s": (STATE_DIM, torch.float32),
        "s1": (STATE_DIM, torch.float32),
        "a": (NUM_J, torch.float32),
        "a_prev": (NUM_J, torch.float32),
        "xi": (XI_DIM, torch.float32),
        "o": (OBS_DIM, torch.float32),
        "r": ((), torch.float32),
        "f": ((), torch.long),
        "v": ((), torch.long),
        "goal_steps": ((), torch.long),
        "goal": (4, torch.float32),
        "ep_step": ((), torch.int32),
        "done": ((), torch.bool),
    } | (extra or {})


class ReplayBuffer:
    def __init__(self, T: int, N: int, device: str, extra: dict | None = None):
        self.T, self.N, self.device = T, N, device
        self.fields = fields(extra)
        self.data = {}
        for k, (shape, dt) in self.fields.items():
            shp = (T, N) + ((shape,) if isinstance(shape, int) else shape)
            self.data[k] = torch.zeros(shp, dtype=dt, device=device)
        self.ptr, self.size = 0, 0

    def add(self, tr: dict):
        for k in self.fields:
            self.data[k][self.ptr] = tr[k]
        self.ptr = (self.ptr + 1) % self.T
        self.size = min(self.size + 1, self.T)

    def _row(self, age: torch.Tensor) -> torch.Tensor:
        """Map age-ordered time index (0 = oldest) to the storage row."""
        return (self.ptr - self.size + age) % self.T

    def sample_windows(self, B: int, k: int):
        """k-step windows; returns dict of (B, k, ...) tensors and a validity mask (B, k)."""
        if k < 1 or k > self.size:
            raise ValueError(f"Cannot sample {k}-step windows from {self.size} buffered steps")
        t0 = torch.randint(0, self.size - k + 1, (B,), device=self.device)
        env = torch.randint(0, self.N, (B,), device=self.device)
        ages = t0.unsqueeze(-1) + torch.arange(k, device=self.device)
        rows = self._row(ages)
        e = env.unsqueeze(-1).expand_as(rows)
        out = {key: self.data[key][rows, e] for key in ("s", "s1", "a", "xi", "f", "v", "done")}
        # transition j is valid if no episode ended at transitions 0..j-1
        d = out["done"].float()
        mask = torch.cumprod(torch.cat((torch.ones(B, 1, device=self.device), 1 - d[:, :-1]), -1), -1)
        out["mask"] = mask
        return out

    def sample_starts(self, B: int, hist: int, envs: torch.Tensor | None = None):
        """Start points for imagination plus their history (oldest first) of observations as seen and actions;
        ``envs`` restricts them to these environments."""
        age = torch.randint(0, self.size, (B,), device=self.device)
        env = torch.randint(0, self.N, (B,), device=self.device) if envs is None else envs[torch.randint(0, len(envs), (B,), device=self.device)]
        row = self._row(age)
        ep_step = self.data["ep_step"][row, env].long()
        back = torch.arange(hist - 1, -1, -1, device=self.device)  # hist-1 ... 0
        back = torch.minimum(back.unsqueeze(0), ep_step.unsqueeze(-1))  # stay inside the episode
        back = torch.minimum(back, age.unsqueeze(-1))  # stay inside the buffer
        hrows = self._row(age.unsqueeze(-1) - back)
        henv = env.unsqueeze(-1).expand_as(hrows)
        start = {k: self.data[k][row, env] for k in ("s", "a_prev", "xi", "f", "v", "goal_steps", "goal", "ep_step")}
        hist_d = {k: self.data[k][hrows, henv] for k in ("o", "a_prev")}
        return start, hist_d

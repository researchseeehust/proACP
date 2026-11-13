# Third-party notices

proACP is built next to [supermimic_vibed](https://github.com/zita-ch/supermimic_vibed), an unofficial reimplementation of SuperMimic
by Chong Zhang (ETH Zurich), and parts of it are derived from that code. The derived parts remain under its MIT License, reproduced
below as that license requires.

| module of `src/proacp/` | share | what it is |
|---|---|---|
| `sim/rotations.py`, `train/reset_sampling.py`; `tools/urdf_to_mjcf.py` | 99 %, 90 %; 100 % | taken as is |
| `models/networks.py`, `sim/kinematics.py` | 91 %, 93 % | adapted: the world model sees the height scan and the policy encodes the contact command; FK without Isaac Lab |
| `train/buffer.py`, `train/stage1.py`, `robots/g1/g1_constants.py`, `tasks/tracking.py` | 63 %, 57 %, 56 %, 58 % | adapted for the contact interface and for mjlab |
| `sim/env.py`, `clips/library.py`, `evaluation/clips.py`; `tools/render.py` | 25 %, 43 %, 18 %; 18 % | rewritten for mjlab and for terrains made of boxes; the structure follows upstream |

The share is the fraction of a proACP file's non-comment lines that also appear, in order, in the upstream file (`difflib`). The other
modules (`models/planner.py`, `tasks/goal.py`, `tasks/contact.py`, `sim/terrain.py`, `clips/labels.py`, `clips/refine.py`, `train/stage2.py`,
`train/stage3.py`, `train/planner_offline.py`, `evaluation/goals.py`, `robot.py`, `robots/template/`, and `tools/` apart from the two above) are not in upstream.

```
MIT License

Copyright (c) 2026 ETH Zurich, Chong Zhang

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

# depthvo

Minimal monocular visual odometry with a GTSAM backend, plus a 3D Gaussian Splatting demo
on its output.

- **Input**: a folder of frames and an intrinsics file.
- **Output**: the camera trajectory (TUM format, every frame) and a sparse map with keyframe
  poses (COLMAP format), which is what 3DGS trains from.
- **Pipeline**: OpenCV frontend (KLT tracking or ORB matching) → keyframes → sliding-window
  bundle adjustment in GTSAM → PnP for the frames in between. With depth maps, a depth
  constraint per observation makes the map metric.

## Setup

Python 3.13 and [uv](https://docs.astral.sh/uv/). Runs on macOS and Linux, CPU only:

    uv sync --group dev --group viz
    uv run pytest

## Run

    uv run run.py sequence=my_seq                      # data/my_seq/images/*.png + data/my_seq/intrinsics.yaml
    uv run run.py dataset=tum_rgbd sequence=freiburg1_xyz
    uv run run.py dataset=tum_rgbd sequence=freiburg1_xyz viewer=true   # watch it live (browser)
    uv run run.py dataset=tum_rgbd sequence=freiburg1_xyz 'stages=[viz]' # view a finished run
    uv run run.py dataset=tum_rgbd sequence=freiburg1_xyz frontend=orb backend=window_depth
    uv run scripts/webcam.py my_calib.yaml                  # live from a webcam (or --source clip.mp4)

Results go to `outputs/<dataset>/<sequence>/<frontend>_<backend>/`:

| file | content |
|---|---|
| `traj.txt` | pose per frame, TUM format `t tx ty tz qx qy qz qw` |
| `sparse/0/` | COLMAP model: keyframe poses, landmarks, observations |
| `reproj.csv`, `reproj.png` | per-frame reprojection error before / after the window solve |
| `ate.json`, `ate.png` | ATE after Sim(3) alignment, when the dataset has ground truth |

Options are Hydra overrides of the files in `configs/`. The ones worth knowing:
`frontend=klt|orb`, `backend=window|window_depth` (depth constraints from
`dataset.depth_dir`, metric map), `dataset.stride`, `dataset.max_frames`, `backend.window`
(keyframes per solve) and `backend.old_keyframes` (`fix`: keyframes that left the window
stay as fixed cameras, consistent map; `drop`: ignore them, slightly better trajectory).
`ate.json` reports the ATE after Sim(3) alignment (monocular) and after SE(3) alignment
(the meaningful one when depth made the map metric).

Reference numbers on TUM freiburg1_xyz (798 frames, ~30 s on a laptop CPU):

| frontend | backend | ATE RMSE Sim(3) | ATE RMSE SE(3) | scale |
|---|---|---|---|---|
| klt | window | 5.5 cm | n/a (monocular) | arbitrary |
| orb | window | 4.7 cm | n/a (monocular) | arbitrary |
| klt | window_depth | 4.3 cm | 4.3 cm | metric (0.99) |
| orb | window_depth | 4.4 cm | 4.4 cm | metric (0.98) |
| klt | window_depth, `depth.source=relative` | 4.9 cm | n/a (monocular) | arbitrary |

### Intrinsics file

    fx: 525.0
    fy: 525.0
    cx: 319.5
    cy: 239.5
    width: 640
    height: 480
    dist: [0, 0, 0, 0, 0]     # OpenCV k1 k2 p1 p2 k3; frames are undistorted on load

### TUM RGB-D

Unpack a sequence to `data/tum/<sequence>/` (the folder with `rgb/` and `groundtruth.txt`)
and set `dataset.calib` to its Freiburg number (`freiburg1` by default).

## As a library

`run.py` is a thin wrapper around one class; use it directly for anything the CLI
does not cover:

```python
from depthvo import VisualOdometry, Intrinsics, ImageSequence, DepthSource

intr = Intrinsics.load("data/my_seq/intrinsics.yaml")
vo = VisualOdometry(intr, frontend="klt", backend="window")      # names of files in configs/, or dicts
for frame in ImageSequence("data/my_seq/images", intr):
    r = vo.step(frame)          # FrameResult
    r.pose                      # T_w_c (4x4) at this step
    r.is_keyframe, r.n_tracks
    r.reproj_before             # this frame's observations vs the map, pose from PnP / init
    r.reproj_after              # same after the window solve (ReprojStats: n, rms, median, in px)
    r.solve                     # SolveStats of the solve a keyframe triggered (landmarks, iterations, cost)

vo.pose(312)                    # any processed frame, by index or file name, including later refinements
vo.reprojection_error(312)      # re-evaluated against the current map
vo.trajectory()                 # (timestamps, names, [T_w_c])
vo.backend.poses, vo.backend.landmarks, vo.backend.obs    # the map itself
vo.save("outputs/my_run")       # traj.txt, sparse/0, reproj.csv
```

`reproj.csv` has one row per frame: observations with a landmark, rms and median
reprojection error before and after that frame's solve, and against the final map.
With `backend="window_depth"` pass `depth=DepthSource(depth_dir, depth_scale, intr)`.

### Depth sources

`DepthSource` reads 16-bit PNG or float `.npy` maps paired to frames by file stem or
nearest timestamp. `backend.depth.source` says what the values mean:

- `metric` (default): sensor depth or a metric network. Each observation gets a range
  factor with relative noise `sigma_rel`; the map becomes metric.
- `relative`: a scale-ambiguous network. For each keyframe, the scale that maps its
  predicted depths onto the landmarks it already sees (median ratio over at least
  `min_anchors` of them) is fitted once, then its depths are used as above. The map
  scale stays the monocular one; what relative depth adds is consistency between
  frames, not metres. 

## 3D Gaussian Splatting demo

Training needs a machine with an NVIDIA GPU, the CUDA toolkit 12.x (`nvcc`), gcc and uv.
The environment is built once, compiling the three CUDA extensions of the official
graphdeco code (about 10 minutes). Tested with an RTX PRO 6000 Blackwell, CUDA 12.8,
gcc 13 and torch 2.7.1+cu128.

    git submodule update --init third_party/gaussian-splatting
    env/gaussian_splatting.sh                 # -> ~/envs/gaussian_splatting
    source env/gaussian_splatting.activate

Then either run the stage, which trains and renders held-out keyframes:

    uv run run.py dataset=tum_rgbd sequence=freiburg1_xyz 'stages=[gs]'

or open `notebooks/demo_3dgs.ipynb` with the `gaussian_splatting` kernel; it runs the whole
chain with figures. The model lands in `gs/point_cloud/iteration_7000/point_cloud.ply`
(open it in any 3DGS viewer, e.g. SuperSplat) and the renders in `gs/test/ours_7000/`.

## Conventions

One pose convention inside the code: `T_w_c`, a 4x4 world-from-camera matrix with the first
keyframe as the world origin and OpenCV camera axes (x right, y down, z forward). Monocular,
so the scale is arbitrary. OpenCV, COLMAP, GTSAM and TUM conventions are converted only in
`src/depthvo/geometry/transforms.py`, and `tests/test_conventions.py` checks them against each
other through a shared projection.



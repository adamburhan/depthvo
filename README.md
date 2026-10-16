# depthvo

Minimal monocular visual odometry with a GTSAM backend, plus a 3D Gaussian Splatting demo
on its output.

- **Input**: a folder of frames and an intrinsics file.
- **Output**: the camera trajectory (TUM format, every frame) and a sparse map with keyframe
  poses (COLMAP format), which is what 3DGS trains from.
- **Pipeline**: OpenCV KLT frontend → keyframes → sliding-window bundle adjustment in GTSAM
  → PnP for the frames in between. Other frontends plug in through `src/depthvo/frontend/`.

## Setup

Python 3.13 and [uv](https://docs.astral.sh/uv/). Runs on macOS and Linux, CPU only:

    uv sync --group dev --group viz
    uv run pytest

## Run

    uv run run.py sequence=my_seq                      # data/my_seq/images/*.png + data/my_seq/intrinsics.yaml
    uv run run.py dataset=tum_rgbd sequence=freiburg1_xyz
    uv run run.py dataset=tum_rgbd sequence=freiburg1_xyz viewer=true   # watch it live (browser)
    uv run run.py dataset=tum_rgbd sequence=freiburg1_xyz 'stages=[viz]' # view a finished run

Results go to `outputs/<dataset>/<sequence>/<frontend>_<backend>/`:

| file | content |
|---|---|
| `traj.txt` | pose per frame, TUM format `t tx ty tz qx qy qz qw` |
| `sparse/0/` | COLMAP model: keyframe poses, landmarks, observations |
| `ate.json`, `ate.png` | ATE after Sim(3) alignment, when the dataset has ground truth |

Options are Hydra overrides of the files in `configs/`. The ones worth knowing:
`dataset.stride`, `dataset.max_frames`, `backend.window` (keyframes per solve) and
`backend.old_keyframes` (`fix`: keyframes that left the window stay as fixed cameras,
consistent map; `drop`: ignore them, slightly better trajectory). On TUM freiburg1_xyz
the ATE is about 5 cm either way.

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



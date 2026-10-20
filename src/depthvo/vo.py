"""depthvo as a library: one object, one call per frame.

    vo = VisualOdometry(intrinsics, frontend="klt", backend="window")
    for frame in ImageSequence(image_dir, intrinsics):
        r = vo.step(frame)                 # FrameResult: pose, keyframe flag, reprojection stats
    vo.pose(i), vo.reprojection_error(i)   # any processed frame, against the current map
    vo.trajectory(); vo.save(out_dir)      # traj.txt, sparse/0 (COLMAP), reproj.csv

`frontend` / `backend` are names of files in configs/frontend and configs/backend,
or config objects (dict / OmegaConf) with the same fields. With backend depth
enabled, pass `depth=DepthSource(...)`. The frontend, backend and their state
(vo.backend.poses / landmarks / obs) stay accessible for anything beyond this.
"""

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from omegaconf import DictConfig, OmegaConf

from depthvo.backend.window import ReprojStats, SlidingWindowBackend, SolveStats
from depthvo.frontend import build_frontend

CONFIGS = Path(__file__).resolve().parents[2] / "configs"


@dataclass
class FrameResult:
    index: int                    # position in processing order (= argument for vo.pose / reprojection_error)
    name: str
    timestamp: float
    pose: np.ndarray              # T_w_c (4x4) as estimated at this step
    is_keyframe: bool
    n_tracks: int                 # 2D observations reported by the frontend
    reproj_before: ReprojStats    # this frame's observations vs the map, pose from PnP / init
    reproj_after: ReprojStats     # same after the window solve (keyframes; identical otherwise)
    solve: SolveStats | None      # the solve this keyframe triggered, None for other frames
    new_landmarks: int = 0

    def __str__(self):
        tag = "KF " if self.is_keyframe else "   "
        return f"{tag}{self.index:5d} {self.name}: reproj {self.reproj_before} -> {self.reproj_after}"


def load_config(group, spec):
    if isinstance(spec, DictConfig):
        return spec
    if isinstance(spec, dict):
        return OmegaConf.create(spec)
    path = CONFIGS / group / f"{spec}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"no {group} config {spec!r} ({path})")
    cfg = OmegaConf.load(path)
    if "defaults" in cfg:   # one level of hydra-style inheritance, e.g. window_depth -> window
        base = OmegaConf.load(CONFIGS / group / f"{cfg.defaults[0]}.yaml")
        cfg = OmegaConf.merge(base, {k: v for k, v in cfg.items() if k != "defaults"})
    return cfg


class VisualOdometry:
    def __init__(self, intrinsics, frontend="klt", backend="window", depth=None, seed=0):
        self.intrinsics = intrinsics
        self.frontend_cfg = load_config("frontend", frontend)
        self.backend_cfg = load_config("backend", backend)
        if self.backend_cfg.depth.enabled and depth is None:
            raise ValueError("backend has depth.enabled but no DepthSource was given")
        self.frontend = build_frontend(self.frontend_cfg, intrinsics)
        self.backend = SlidingWindowBackend(self.backend_cfg, intrinsics)
        self.depth = depth
        self.results = []     # FrameResult per processed frame
        cv2.setRNGSeed(seed)  # RANSAC in the essential matrix / PnP

    def step(self, frame) -> FrameResult:
        tracks = self.frontend.process(frame)
        i = len(self.backend.frames)
        if tracks.is_keyframe:
            d = self.depth.get(frame) if self.depth is not None else None
            n_new, before, after, solve = self.backend.add_keyframe(tracks, gray=frame.gray, depth=d)
        else:
            self.backend.track_frame(tracks)
            before = after = self.backend.reprojection_error(i)
            n_new, solve = 0, None
        r = FrameResult(i, frame.name, frame.timestamp, self.backend.pose(i), tracks.is_keyframe,
                        len(tracks.uv), before, after, solve, n_new)
        self.results.append(r)
        return r

    # ---- queries, any time after step() --------------------------------------

    def pose(self, frame):
        """T_w_c of a processed frame (index or name), including later refinements."""
        return self.backend.pose(self._index(frame))

    def reprojection_error(self, frame) -> ReprojStats:
        return self.backend.reprojection_error(self._index(frame))

    def trajectory(self):
        """(timestamps, names, [T_w_c]) for every processed frame."""
        return self.backend.trajectory()

    @property
    def keyframes(self):
        return [r for r in self.results if r.is_keyframe]

    def save(self, out_dir):
        """traj.txt (TUM, every frame), sparse/0 (COLMAP model of the keyframes),
        reproj.csv (per frame: n, rms and median before / after its solve, and now)."""
        from depthvo.io.colmap_writer import write_colmap_model
        from depthvo.io.trajectory import write_tum

        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        ts, names, poses = self.trajectory()
        write_tum(out / "traj.txt", ts, poses)
        b = self.backend
        write_colmap_model(out / "sparse" / "0", self.intrinsics, [n for n, _ in b.kf_meta], b.poses,
                           b.landmarks, b.obs, {t: (g, g, g) for t, g in b.colors.items()})
        with open(out / "reproj.csv", "w") as f:
            f.write("index,name,timestamp,is_keyframe,n_obs,rms_before,median_before,rms_after,median_after,rms_now,median_now\n")
            for r in self.results:
                now = self.reprojection_error(r.index)
                f.write(f"{r.index},{r.name},{r.timestamp:.6f},{int(r.is_keyframe)},{now.n},"
                        f"{r.reproj_before.rms:.4f},{r.reproj_before.median:.4f},"
                        f"{r.reproj_after.rms:.4f},{r.reproj_after.median:.4f},{now.rms:.4f},{now.median:.4f}\n")
        return out

    def _index(self, frame):
        if isinstance(frame, str):
            return next(i for i, (_, n, _, _) in enumerate(self.backend.frames) if n == frame)
        return int(frame)

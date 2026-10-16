"""Frontend contract. A frontend owns feature detection/association and keyframe
selection; it knows nothing about poses. Output per frame is the set of 2D
observations with PERSISTENT track ids: the same physical landmark keeps the same
id across frames, which is all the backend needs to build projection factors."""

from dataclasses import dataclass

import numpy as np


@dataclass
class FrameTracks:
    frame_idx: int
    timestamp: float
    name: str
    uv: np.ndarray          # (N, 2) pixel coords in the undistorted image
    track_ids: np.ndarray   # (N,) int64, persistent across frames
    is_keyframe: bool


class Frontend:
    def __init__(self, cfg, intrinsics):
        self.cfg = cfg
        self.intrinsics = intrinsics
        self._next_track_id = 0

    def new_track_ids(self, n):
        ids = np.arange(self._next_track_id, self._next_track_id + n, dtype=np.int64)
        self._next_track_id += n
        return ids

    def process(self, frame) -> FrameTracks:
        raise NotImplementedError

    def is_keyframe(self, n_tracked, n_tracked_at_last_kf, parallax_px):
        """Shared keyframe rule: too few survivors of the last keyframe's tracks,
        or enough median parallax to triangulate well."""
        return (
            n_tracked < self.cfg.keyframe.min_tracked_ratio * n_tracked_at_last_kf
            or parallax_px > self.cfg.keyframe.min_parallax_px
        )

"""Depth maps paired to frames: by identical filename, else (TUM) by nearest
timestamp parsed from the filename. 16-bit PNG / depth_scale = metres; 0 = no data."""

from pathlib import Path

import cv2
import numpy as np

from depthvo.io.sequence import IMAGE_EXTS


class DepthSource:
    def __init__(self, depth_dir, depth_scale, intrinsics, max_dt=0.03):
        self.dir, self.scale, self.intr, self.max_dt = Path(depth_dir), float(depth_scale), intrinsics, max_dt
        self.paths = sorted(p for p in self.dir.iterdir() if p.suffix.lower() in IMAGE_EXTS)
        if not self.paths:
            raise FileNotFoundError(f"no depth images in {self.dir}")
        self.by_name = {p.name: p for p in self.paths}
        try:
            self.ts = np.array([float(p.stem) for p in self.paths])
        except ValueError:
            self.ts = None

    def get(self, frame):
        """(H, W) float32 metres with NaN where there is no depth, or None if no map matches."""
        p = self.by_name.get(frame.name)
        if p is None and self.ts is not None:
            j = int(np.argmin(np.abs(self.ts - frame.timestamp)))
            p = self.paths[j] if abs(self.ts[j] - frame.timestamp) <= self.max_dt else None
        if p is None:
            return None
        d = cv2.imread(str(p), cv2.IMREAD_UNCHANGED).astype(np.float32) / self.scale
        d[d <= 0] = np.nan
        return self.intr.undistort_depth(d)

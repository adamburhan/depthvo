"""Camera intrinsics from a yaml file. Frames are undistorted to a pinhole model
on load, so everything downstream (frontend, GTSAM Cal3_S2, COLMAP PINHOLE)
sees a single distortion-free K."""

from dataclasses import dataclass, field

import cv2
import numpy as np
import yaml


@dataclass
class Intrinsics:
    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int
    dist: np.ndarray = field(default_factory=lambda: np.zeros(5))  # OpenCV k1 k2 p1 p2 k3

    @property
    def K(self):
        return np.array([[self.fx, 0, self.cx], [0, self.fy, self.cy], [0, 0, 1.0]])

    @property
    def has_distortion(self):
        return bool(np.any(self.dist != 0))

    def undistort(self, img):
        # Same K out as in: keeps the yaml's K valid for the undistorted stream.
        return cv2.undistort(img, self.K, self.dist, None, self.K) if self.has_distortion else img

    def undistort_depth(self, depth):
        """Same map as undistort(), nearest-neighbour so depth edges are not blended."""
        if not self.has_distortion:
            return depth
        m1, m2 = cv2.initUndistortRectifyMap(self.K, self.dist, None, self.K, (self.width, self.height), cv2.CV_32FC1)
        return cv2.remap(depth, m1, m2, cv2.INTER_NEAREST, borderValue=np.nan)

    @classmethod
    def load(cls, path):
        d = yaml.safe_load(open(path))
        dist = np.asarray(d.get("dist", [0, 0, 0, 0, 0]), dtype=float)
        return cls(d["fx"], d["fy"], d["cx"], d["cy"], d["width"], d["height"], dist)

"""Depth constraint on a landmark seen from a keyframe.

A depth d measured at pixel (u, v) fixes the landmark's distance from the camera
centre: range = d * ||K^-1 (u, v, 1)||. So the constraint is GTSAM's built-in
RangeFactor3D between the Pose3 and the Point3 (fast C++ factor, no Python
callback). The noise is relative, sigma = sigma_rel * range, which is the
linearisation of a log-depth error; Huber makes it robust to depth outliers.
Depth factors make the monocular map metric and pin its scale."""

import gtsam
import numpy as np


def depth_to_range(depth, uv, K_inv):
    ray = K_inv @ np.array([uv[0], uv[1], 1.0])
    return float(depth) * float(np.linalg.norm(ray))


def make_depth_factor(pose_key, point_key, depth, uv, K_inv, sigma_rel, huber_k):
    r = depth_to_range(depth, uv, K_inv)
    noise = gtsam.noiseModel.Robust.Create(
        gtsam.noiseModel.mEstimator.Huber.Create(huber_k),
        gtsam.noiseModel.Isotropic.Sigma(1, sigma_rel * r))
    return gtsam.RangeFactor3D(pose_key, point_key, r, noise)

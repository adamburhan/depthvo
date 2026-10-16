"""Pose conventions. Inside depthvo a pose is ALWAYS T_w_c: a 4x4 world-from-camera
matrix (the camera's pose in the world), world = camera 0. Every other convention
is converted at the boundary by the functions below, and nowhere else.

Boundaries:
  OpenCV  cv2.recoverPose(E, p1, p2) -> (R, t) with  x2 = R x1 + t,  i.e. T_c2_c1
  COLMAP  Image.cam_from_world  = T_c_w  (quaternion stored w,x,y,z in text files)
  GTSAM   Pose3 in PinholeCamera / GenericProjectionFactor = T_w_c
  TUM     tx ty tz qx qy qz qw of T_w_c
"""

import numpy as np
from scipy.spatial.transform import Rotation


def se3(R, t):
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = np.asarray(t, dtype=float).ravel()
    return T


def inv(T):
    R, t = T[:3, :3], T[:3, 3]
    return se3(R.T, -R.T @ t)


def opencv_relative_to_T_c1_c2(R, t):
    """cv2.recoverPose output (T_c2_c1, unit-norm t) -> T_c1_c2, so that
    T_w_c2 = T_w_c1 @ T_c1_c2."""
    return inv(se3(R, t))


def to_colmap_cam_from_world(T_w_c):
    """-> pycolmap.Rigid3d holding T_c_w."""
    import pycolmap

    T_c_w = inv(T_w_c)
    q_xyzw = Rotation.from_matrix(T_c_w[:3, :3]).as_quat()
    return pycolmap.Rigid3d(pycolmap.Rotation3d(q_xyzw), T_c_w[:3, 3])


def from_colmap_cam_from_world(cam_from_world):
    """pycolmap.Rigid3d (T_c_w) -> T_w_c."""
    return inv(cam_from_world.matrix() if hasattr(cam_from_world, "matrix")
               else se3(cam_from_world.rotation.matrix(), cam_from_world.translation))


def to_gtsam_pose3(T_w_c):
    import gtsam

    return gtsam.Pose3(gtsam.Rot3(T_w_c[:3, :3]), T_w_c[:3, 3])


def from_gtsam_pose3(pose):
    return pose.matrix()


def tum_line(timestamp, T_w_c):
    """'t tx ty tz qx qy qz qw' for a T_w_c pose."""
    q = Rotation.from_matrix(T_w_c[:3, :3]).as_quat()  # xyzw
    t = T_w_c[:3, 3]
    return " ".join(f"{v:.6f}" for v in (timestamp, *t, *q))


def from_tum_line(line):
    """'t tx ty tz qx qy qz qw' -> (timestamp, T_w_c)."""
    v = np.array(line.split(), dtype=float)
    return v[0], se3(Rotation.from_quat(v[4:8]).as_matrix(), v[1:4])

"""Pin the three external pose conventions against each other through a single
projection: the same world point must land on the same pixel via manual
K [R|t], via pycolmap's cam_from_world, and via GTSAM's PinholeCamera(T_w_c)."""

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from depthvo.geometry import transforms as tf

K = np.array([[500.0, 0, 320], [0, 500.0, 240], [0, 0, 1]])


def random_pose(seed=0):
    rng = np.random.default_rng(seed)
    return tf.se3(Rotation.random(random_state=seed).as_matrix(), rng.normal(size=3))


def project_manual(T_w_c, X_w):
    p_c = (tf.inv(T_w_c) @ np.append(X_w, 1.0))[:3]
    uv = K @ p_c
    return uv[:2] / uv[2]


def test_inverse():
    T = random_pose()
    assert np.allclose(tf.inv(T) @ T, np.eye(4))


def test_colmap_roundtrip_and_projection():
    pycolmap = pytest.importorskip("pycolmap")
    T = random_pose(1)
    X = np.array([0.3, -0.2, 4.0]) + T[:3, 3]
    cam_from_world = tf.to_colmap_cam_from_world(T)
    assert np.allclose(tf.from_colmap_cam_from_world(cam_from_world), T)
    p_c = cam_from_world * X
    uv = (K @ p_c)[:2] / p_c[2]
    assert np.allclose(uv, project_manual(T, X))


def test_gtsam_projection():
    gtsam = pytest.importorskip("gtsam")
    T = random_pose(2)
    X = T[:3, 3] + T[:3, :3] @ np.array([0.1, 0.1, 3.0])  # in front of the camera
    cam = gtsam.PinholeCameraCal3_S2(tf.to_gtsam_pose3(T), gtsam.Cal3_S2(500, 500, 0, 320, 240))
    assert np.allclose(cam.project(X), project_manual(T, X))
    assert np.allclose(tf.from_gtsam_pose3(tf.to_gtsam_pose3(T)), T)


def test_opencv_relative_chaining():
    """recoverPose's (R, t) is T_c2_c1: chaining must reproduce T_w_c2 from T_w_c1."""
    T1, T2 = random_pose(3), random_pose(4)
    T_c2_c1 = tf.inv(T2) @ T1
    R, t = T_c2_c1[:3, :3], T_c2_c1[:3, 3]
    assert np.allclose(T1 @ tf.opencv_relative_to_T_c1_c2(R, t), T2)


def test_tum_roundtrip():
    T = random_pose(5)
    ts, T2 = tf.from_tum_line(tf.tum_line(12.5, T))
    assert ts == 12.5 and np.allclose(T, T2, atol=1e-5)

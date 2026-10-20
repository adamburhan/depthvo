"""Sliding-window batch backend on GTSAM.

Map state is an observation table (poses, landmarks, obs below). Each keyframe:
initialise its pose (two-view essential matrix for the first pair, PnP against
existing landmarks after), triangulate tracks now seen from >= 2 keyframes,
rebuild a NonlinearFactorGraph over the last `window` keyframes and solve with
Levenberg-Marquardt. The two oldest in-window poses get tight priors: that
anchors the gauge AND the scale (monocular, no depth factors). Keyframes that
leave the window keep their last solved pose. Non-keyframes get a PnP pose,
stored relative to their keyframe so later refinements carry over.
"""

import logging
from dataclasses import dataclass

import cv2
import gtsam
import numpy as np
from gtsam.symbol_shorthand import L, X

from depthvo.backend.depth_factor import make_depth_factor
from depthvo.geometry.transforms import inv, opencv_relative_to_T_c1_c2, se3

log = logging.getLogger(__name__)


@dataclass
class ReprojStats:
    """Reprojection error of one frame's observations against the map (pixels)."""
    n: int            # observations with a triangulated landmark
    rms: float
    median: float

    def __str__(self):
        return f"{self.rms:.2f} px rms / {self.median:.2f} median over {self.n}" if self.n else "no landmarks"


@dataclass
class SolveStats:
    """One sliding-window solve."""
    window: int           # keyframes optimised
    fixed: int            # older keyframes kept as fixed cameras
    landmarks: int
    depth_factors: int
    iterations: int
    cost_before: float    # GTSAM graph error (sum of squared whitened residuals)
    cost_after: float
    culled: int           # landmarks dropped after the solve (reprojection > max_reproj_px)


class SlidingWindowBackend:
    def __init__(self, cfg, intrinsics):
        self.cfg = cfg
        self.K_np = intrinsics.K
        self.K = gtsam.Cal3_S2(intrinsics.fx, intrinsics.fy, 0.0, intrinsics.cx, intrinsics.cy)
        self.K_inv = np.linalg.inv(intrinsics.K)
        self.depth_obs = {}   # track_id -> {kf_idx: measured depth (m)}, when depth is enabled
        self.poses = {}       # kf_idx -> T_w_c (4x4)
        self.landmarks = {}   # track_id -> xyz (3,) world
        self.obs = {}         # track_id -> [(kf_idx, uv)]
        self.kf_meta = []     # (name, timestamp)
        self.frames = []      # (timestamp, name, kf_idx, T_ckf_c) for every frame
        self.frame_obs = []   # (track_ids, uv) for EVERY frame, same order as self.frames
        self.bad = set()      # track ids culled as outliers; never retried
        self.colors = {}      # track id -> gray intensity at first keyframe observation
        self.pix_noise = gtsam.noiseModel.Robust.Create(
            gtsam.noiseModel.mEstimator.Huber.Create(cfg.huber_k),
            gtsam.noiseModel.Isotropic.Sigma(2, cfg.pixel_sigma))
        self.prior_noise = gtsam.noiseModel.Diagonal.Sigmas(np.full(6, 1e-4))
        self.fixed_noise = gtsam.noiseModel.Diagonal.Sigmas(np.full(6, 1e-6))

    # ---- per-frame API -------------------------------------------------------

    def add_keyframe(self, tracks, gray=None, depth=None):
        """depth: optional (H, W) metres (NaN = none), sampled at the tracks when
        cfg.depth.enabled."""
        i = len(self.kf_meta)
        self.kf_meta.append((tracks.name, tracks.timestamp))
        for tid, uv in zip(tracks.track_ids.tolist(), tracks.uv):
            if tid not in self.bad:
                self.obs.setdefault(tid, []).append((i, np.asarray(uv, float)))
                if gray is not None and tid not in self.colors:
                    self.colors[tid] = int(gray[int(uv[1]), int(uv[0])])
                if depth is not None:
                    d = depth[min(int(round(uv[1])), depth.shape[0] - 1), min(int(round(uv[0])), depth.shape[1] - 1)]
                    if np.isfinite(d) and self.cfg.depth.min_m <= d <= self.cfg.depth.max_m:
                        self.depth_obs.setdefault(tid, {})[i] = float(d)
        if i == 0:
            self.poses[0] = np.eye(4)
        elif i == 1:
            self.poses[1] = self._two_view_init(0, 1)
        else:
            T = self._pnp(tracks)
            self.poses[i] = self._predict() if T is None else T
        n_new = self._triangulate_new(i)
        if i == 1 and self.cfg.depth.enabled:
            self._set_metric_scale()
        self.frames.append((tracks.timestamp, tracks.name, i, np.eye(4)))
        self.frame_obs.append((tracks.track_ids.copy(), tracks.uv.astype(float)))
        before = self.reprojection_error(len(self.frames) - 1)
        solve = self._solve_window() if i >= 1 else None
        after = self.reprojection_error(len(self.frames) - 1)
        log.info(f"kf {i} ({tracks.name}): +{n_new} landmarks, {len(self.landmarks)} total, "
                 f"reproj {before} -> {after}, {solve}")
        return n_new, before, after, solve

    def track_frame(self, tracks):
        """Non-keyframe: PnP pose, stored relative to the latest keyframe."""
        kf = len(self.kf_meta) - 1
        T_w_c = self._pnp(tracks)
        if T_w_c is None:
            T_w_c = self._predict()
        self.frames.append((tracks.timestamp, tracks.name, kf, inv(self.poses[kf]) @ T_w_c))
        self.frame_obs.append((tracks.track_ids.copy(), tracks.uv.astype(float)))

    def _predict(self):
        """Constant-velocity guess from the last two frame poses (PnP failed or no landmarks)."""
        if len(self.frames) < 2:
            return self.current_pose()
        T1, T0 = self.current_pose(), self.poses[self.frames[-2][2]] @ self.frames[-2][3]
        return T1 @ (inv(T0) @ T1)

    def current_pose(self):
        return self.pose(len(self.frames) - 1)

    def pose(self, frame):
        """T_w_c of a processed frame (by index in processing order), as of now:
        keyframe refinements since then are included."""
        t, n, k, rel = self.frames[frame]
        return self.poses[k] @ rel

    def reprojection_error(self, frame) -> ReprojStats:
        """Reprojection error of a frame's observations against the CURRENT pose
        and landmarks. Keyframes: all their tracks with a landmark. Other frames:
        the tracks PnP saw. Re-asking later reflects later refinements."""
        ids, uv = self.frame_obs[frame]
        T_c_w = inv(self.pose(frame))
        errs = []
        for tid, p in zip(ids.tolist(), uv):
            xyz = self.landmarks.get(tid)
            if xyz is None:
                continue
            q = self.K_np @ (T_c_w @ np.append(xyz, 1.0))[:3]
            if q[2] > 0:
                errs.append(np.linalg.norm(q[:2] / q[2] - p))
        if not errs:
            return ReprojStats(0, float("nan"), float("nan"))
        e = np.array(errs)
        return ReprojStats(len(e), float(np.sqrt(np.mean(e**2))), float(np.median(e)))

    def trajectory(self):
        """-> (timestamps, names, [T_w_c]) for every frame, keyframes refined."""
        ts, names, T = zip(*[(t, n, self.poses[k] @ rel) for t, n, k, rel in self.frames])
        return list(ts), list(names), list(T)

    # ---- geometry ------------------------------------------------------------

    def _corr(self, i, j):
        """Tracks observed in both keyframes -> (ids, uv_i, uv_j)."""
        ids, ui, uj = [], [], []
        for tid, ob in self.obs.items():
            d = dict(ob)
            if i in d and j in d:
                ids.append(tid); ui.append(d[i]); uj.append(d[j])
        return np.array(ids), np.array(ui).reshape(-1, 2), np.array(uj).reshape(-1, 2)

    def _set_metric_scale(self):
        """After the unit-baseline init: scale the map so triangulated depths match
        the measured ones (median ratio), so the first solve starts near metric."""
        ratios = [d / self._depth(self.poses[k], self.landmarks[tid])
                  for tid, dk in self.depth_obs.items() if tid in self.landmarks
                  for k, d in dk.items() if k in self.poses and self._depth(self.poses[k], self.landmarks[tid]) > 0]
        if len(ratios) < 10:
            log.warning(f"metric scale: only {len(ratios)} depth measurements, keeping unit baseline")
            return
        s = float(np.median(ratios))
        for k in self.poses:
            self.poses[k][:3, 3] *= s
        for tid in self.landmarks:
            self.landmarks[tid] = self.landmarks[tid] * s
        log.info(f"metric scale from {len(ratios)} depths: x{s:.3f}")

    def _two_view_init(self, i, j):
        ids, ui, uj = self._corr(i, j)
        E, inl = cv2.findEssentialMat(ui, uj, self.K_np, cv2.RANSAC, 0.999, self.cfg.ransac_px)
        _, R, t, inl = cv2.recoverPose(E, ui, uj, self.K_np, mask=inl)
        for tid in ids[inl.ravel() == 0]:
            self._cull(tid)
        log.info(f"two-view init kf{i}-kf{j}: {int((inl > 0).sum())}/{len(ids)} inliers, unit baseline")
        return self.poses[i] @ opencv_relative_to_T_c1_c2(R, t)

    def _pnp(self, tracks):
        has = [k for k, tid in enumerate(tracks.track_ids.tolist()) if tid in self.landmarks]
        if len(has) < 6:
            return None
        xyz = np.array([self.landmarks[tracks.track_ids[k]] for k in has])
        uv = tracks.uv[has].astype(float)
        # EPnP inside RANSAC (ITERATIVE's 4-point minimal solve is unstable),
        # then LM refinement on the inliers.
        ok, rvec, tvec, inl = cv2.solvePnPRansac(
            xyz, uv, self.K_np, None, reprojectionError=self.cfg.max_reproj_px,
            flags=cv2.SOLVEPNP_EPNP)
        if not ok or inl is None or len(inl) < 6:
            log.info(f"pnp failed: ok={ok} inliers={0 if inl is None else len(inl)}/{len(has)}")
            return None
        inl = inl.ravel()
        rvec, tvec = cv2.solvePnPRefineLM(xyz[inl], uv[inl], self.K_np, None, rvec, tvec)
        T_w_c = inv(se3(cv2.Rodrigues(rvec)[0], tvec))
        step = np.linalg.norm(T_w_c[:3, 3] - self.poses[max(self.poses)][:3, 3])
        log.info(f"pnp: {len(inl)}/{len(has)} inliers, step {step:.3f}")
        return T_w_c

    def _triangulate_new(self, i):
        """Triangulate tracks seen at kf i and >= 1 earlier solved kf, from their
        two most separated observations. Accept if in front of both cameras,
        reprojection error and parallax angle pass."""
        n = 0
        for tid, ob in self.obs.items():
            if tid in self.landmarks or ob[-1][0] != i or len(ob) < 2:
                continue
            (a, ua), (b, ub) = ob[0], ob[-1]
            Ta, Tb = self.poses[a], self.poses[b]
            Pa, Pb = self.K_np @ inv(Ta)[:3], self.K_np @ inv(Tb)[:3]
            Xh = cv2.triangulatePoints(Pa, Pb, ua.reshape(2, 1), ub.reshape(2, 1))
            xyz = (Xh[:3] / Xh[3]).ravel()
            ra, rb = xyz - Ta[:3, 3], xyz - Tb[:3, 3]
            cos = ra @ rb / (np.linalg.norm(ra) * np.linalg.norm(rb))
            if (self._depth(Ta, xyz) <= 0 or self._depth(Tb, xyz) <= 0
                    or np.degrees(np.arccos(np.clip(cos, -1, 1))) < self.cfg.min_parallax_deg
                    or max(self._reproj(Ta, xyz, ua), self._reproj(Tb, xyz, ub)) > self.cfg.max_reproj_px):
                continue
            self.landmarks[tid] = xyz
            n += 1
        return n

    def _depth(self, T_w_c, xyz):
        return (inv(T_w_c) @ np.append(xyz, 1.0))[2]

    def _reproj(self, T_w_c, xyz, uv):
        p = self.K_np @ (inv(T_w_c) @ np.append(xyz, 1.0))[:3]
        return np.linalg.norm(p[:2] / p[2] - uv)

    def _cull(self, tid):
        self.bad.add(tid)
        self.obs.pop(tid, None)
        self.landmarks.pop(tid, None)

    # ---- optimisation --------------------------------------------------------

    def _solve_window(self):
        return self._solve(sorted(self.poses)[-self.cfg.window:])

    def _solve(self, window):
        """Optimise the keyframes in `window` and the landmarks they see. What
        happens to keyframes that left the window is cfg.old_keyframes:
          drop  their observations are ignored
          fix   they join as fixed cameras with their observations kept
                (ORB-SLAM local BA), so landmarks stay consistent with the
                whole map, not just the window
        """
        fix = self.cfg.old_keyframes == "fix"
        graph, values = gtsam.NonlinearFactorGraph(), gtsam.Values()
        for i in window:
            values.insert(X(i), gtsam.Pose3(self.poses[i]))
        # gauge anchor; without depth the second prior also pins the (arbitrary) scale
        for i in window[:1 if self.cfg.depth.enabled else 2]:
            graph.add(gtsam.PriorFactorPose3(X(i), gtsam.Pose3(self.poses[i]), self.prior_noise))
        active, fixed, n_depth = {}, set(), 0
        for tid, xyz in self.landmarks.items():
            ob = [(i, uv) for i, uv in self.obs[tid] if i in self.poses]
            if not fix:
                ob = [(i, uv) for i, uv in ob if i >= window[0]]
            if len(ob) < 2 or not any(i >= window[0] for i, _ in ob):
                continue
            values.insert(L(tid), xyz)
            active[tid] = ob
            for i, uv in ob:
                if i < window[0] and i not in fixed:
                    values.insert(X(i), gtsam.Pose3(self.poses[i]))
                    graph.add(gtsam.PriorFactorPose3(X(i), gtsam.Pose3(self.poses[i]), self.fixed_noise))
                    fixed.add(i)
                graph.add(gtsam.GenericProjectionFactorCal3_S2(uv, self.pix_noise, X(i), L(tid), self.K))
                d = self.depth_obs.get(tid, {}).get(i)
                if d is not None and self.cfg.depth.enabled:
                    graph.add(make_depth_factor(X(i), L(tid), d, uv, self.K_inv, self.cfg.depth.sigma_rel, self.cfg.huber_k))
                    n_depth += 1
        if not active:
            return None
        params = gtsam.LevenbergMarquardtParams()
        params.setMaxIterations(self.cfg.lm_max_iters)
        opt = gtsam.LevenbergMarquardtOptimizer(graph, values, params)
        e0 = graph.error(values)
        result = opt.optimize()
        for i in window:
            self.poses[i] = result.atPose3(X(i)).matrix()
        errs, culled = [], 0
        for tid, ob in active.items():
            self.landmarks[tid] = result.atPoint3(L(tid))
            e = np.mean([self._reproj(self.poses[i], self.landmarks[tid], uv) for i, uv in ob])
            if e > self.cfg.max_reproj_px:
                self._cull(tid); culled += 1
            else:
                errs.append(e)
        return SolveStats(len(window), len(fixed), len(active), n_depth, opt.iterations(),
                          float(e0), float(graph.error(result)), culled)

"""ATE after Sim(3) alignment of an estimated TUM trajectory to ground truth.
Poses are associated by nearest timestamp within `max_dt`."""

import numpy as np

from depthvo.eval.alignment import umeyama_sim3
from depthvo.io.trajectory import read_tum


def associate(ts_est, ts_gt, max_dt=0.02):
    """-> list of (i_est, j_gt) pairs, each est stamp matched to its nearest gt stamp."""
    ts_gt = np.asarray(ts_gt)
    pairs = []
    for i, t in enumerate(ts_est):
        j = int(np.argmin(np.abs(ts_gt - t)))
        if abs(ts_gt[j] - t) <= max_dt:
            pairs.append((i, j))
    return pairs


def ate_rmse(est_path, gt_path, with_scale=True, max_dt=0.02, plot_path=None):
    ts_e, P_e = read_tum(est_path)
    ts_g, P_g = read_tum(gt_path)
    pairs = associate(ts_e, ts_g, max_dt)
    if len(pairs) < 3:
        raise ValueError(f"only {len(pairs)} associated poses")
    src = np.array([P_e[i][:3, 3] for i, _ in pairs])
    dst = np.array([P_g[j][:3, 3] for _, j in pairs])
    s, R, t = umeyama_sim3(src, dst, with_scale)
    aligned = (s * (R @ src.T)).T + t
    err = np.linalg.norm(dst - aligned, axis=1)
    s1, R1, t1 = umeyama_sim3(src, dst, with_scale=False)
    err_se3 = np.linalg.norm(dst - (R1 @ src.T).T - t1, axis=1)
    if plot_path is not None:
        from depthvo.eval.plot import plot_trajectory
        plot_trajectory(aligned, dst, [ts_e[i] for i, _ in pairs], plot_path)
    return {"ate_rmse": float(np.sqrt(np.mean(err**2))), "scale": float(s),
            "ate_rmse_se3": float(np.sqrt(np.mean(err_se3**2))), "n": len(pairs)}

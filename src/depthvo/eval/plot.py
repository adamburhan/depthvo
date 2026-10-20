"""Trajectory-vs-ground-truth figure: top-down and side views of the Sim(3)-aligned
estimate over GT, plus the per-frame position error."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def plot_trajectory(est_xyz, gt_xyz, timestamps, out_path, title=""):
    err = np.linalg.norm(est_xyz - gt_xyz, axis=1)
    t = np.asarray(timestamps) - timestamps[0]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2), constrained_layout=True)
    for ax, (a, b), lab in zip(axes[:2], [(0, 1), (0, 2)], ["top-down (x, y)", "side (x, z)"]):
        ax.plot(gt_xyz[:, a], gt_xyz[:, b], color="0.3", lw=1.5, label="ground truth")
        ax.plot(est_xyz[:, a], est_xyz[:, b], color="C0", lw=1.0, label="estimate (Sim3-aligned)")
        ax.scatter(*gt_xyz[0, [a, b]], color="0.3", marker="o", s=30, zorder=3)
        ax.set_xlabel(f"{'xyz'[a]} [m]"); ax.set_ylabel(f"{'xyz'[b]} [m]")
        ax.set_aspect("equal"); ax.set_title(lab)
    axes[0].legend(loc="best", fontsize=9)
    axes[2].plot(t, err, color="C3", lw=1.0)
    axes[2].axhline(np.sqrt(np.mean(err**2)), color="0.5", ls="--", lw=1, label=f"RMSE {np.sqrt(np.mean(err**2))*100:.1f} cm")
    axes[2].set_xlabel("time [s]"); axes[2].set_ylabel("position error [m]")
    axes[2].set_title("error after alignment"); axes[2].legend(fontsize=9)
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
    if title:
        fig.suptitle(title)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_reproj(csv_path, out_path):
    """Per-frame reprojection rms before / after the solve, keyframes marked."""
    import csv
    rows = list(csv.DictReader(open(csv_path)))
    idx = np.array([int(r["index"]) for r in rows])
    kf = np.array([r["is_keyframe"] == "1" for r in rows])
    before = np.array([float(r["rms_before"]) for r in rows])
    after = np.array([float(r["rms_after"]) for r in rows])
    now = np.array([float(r["rms_now"]) for r in rows])
    fig, ax = plt.subplots(figsize=(12, 3.8), constrained_layout=True)
    ax.plot(idx, now, color="0.6", lw=0.8, label="all frames, vs final map")
    ax.plot(idx[kf], before[kf], "o", color="C3", ms=3, label="keyframes, before solve (PnP pose)")
    ax.plot(idx[kf], after[kf], "o", color="C0", ms=3, label="keyframes, after solve")
    ax.set_xlabel("frame"); ax.set_ylabel("reprojection rms [px]")
    ax.set_ylim(0, np.nanpercentile(np.concatenate([before[kf], now]), 99) * 1.1)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(fontsize=9, loc="upper right", ncol=3)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)

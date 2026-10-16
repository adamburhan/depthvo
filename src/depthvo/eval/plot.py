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

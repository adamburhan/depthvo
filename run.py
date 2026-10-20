"""depthvo entrypoint (Hydra): `uv run run.py dataset=tum_rgbd sequence=freiburg1_xyz`

Stages, each reading the previous one's files from output_dir:
    vo    images + intrinsics -> traj.txt (TUM, every frame) + sparse/0 (COLMAP, keyframes)
    eval  traj.txt + ground truth -> ate.json, ate.png
    viz   sparse/0 -> viser viewer of a finished run (blocks)
    gs    sparse/0 + images -> 3DGS training + renders (GPU env, see env/)
viewer=true shows the vo stage live (image feed + map) in viser.
"""

import json
import logging
from pathlib import Path

import numpy as np

import hydra
from omegaconf import DictConfig

log = logging.getLogger(__name__)


def stage_vo(cfg: DictConfig) -> None:
    from depthvo import DepthSource, ImageSequence, Intrinsics, VisualOdometry

    intr = Intrinsics.load(cfg.dataset.intrinsics)
    seq = ImageSequence(cfg.dataset.image_dir, intr, cfg.dataset.stride, cfg.dataset.max_frames)
    depth = None
    if cfg.backend.depth.enabled:
        if cfg.dataset.depth_dir is None:
            raise ValueError("backend.depth.enabled needs dataset.depth_dir")
        depth = DepthSource(cfg.dataset.depth_dir, cfg.dataset.depth_scale, intr)
    vo = VisualOdometry(intr, cfg.frontend, cfg.backend, depth=depth, seed=cfg.seed)
    viewer = None
    if cfg.viewer:
        from depthvo.viz.live import LiveViewer
        viewer = LiveViewer(cfg.viewer_opts, intr)
    for frame in seq:
        r = vo.step(frame)
        if viewer:
            viewer.update(frame, r, vo)
    out = vo.save(cfg.output_dir)
    kf = vo.keyframes
    log.info(f"wrote {len(vo.results)} poses ({len(kf)} keyframes) to {out}: traj.txt, sparse/0, reproj.csv")
    log.info(f"keyframe reprojection rms, mean over keyframes: before solve "
             f"{np.nanmean([r.reproj_before.rms for r in kf[1:]]):.2f} px -> after {np.nanmean([r.reproj_after.rms for r in kf[1:]]):.2f} px")
    if viewer:
        viewer.block()


def stage_viz(cfg: DictConfig) -> None:
    from depthvo.viz.viewer import serve
    v = cfg.viewer_opts
    serve(Path(cfg.output_dir) / "sparse" / "0", v.port, v.frustum_scale, v.point_size)


def stage_eval(cfg: DictConfig) -> None:
    from depthvo.eval.ate import ate_rmse
    from depthvo.eval.plot import plot_reproj
    out = Path(cfg.output_dir)
    if (out / "reproj.csv").exists():
        plot_reproj(out / "reproj.csv", out / "reproj.png")
        log.info(f"reprojection plot {out / 'reproj.png'}")
    if cfg.dataset.gt_trajectory is None:
        log.warning("no gt_trajectory for this dataset, skipping ATE")
        return
    res = ate_rmse(out / "traj.txt", cfg.dataset.gt_trajectory, plot_path=out / "ate.png")
    (out / "ate.json").write_text(json.dumps(res, indent=2))
    log.info(f"ATE RMSE {res['ate_rmse']:.4f} m (Sim3, scale {res['scale']:.3f}) / "
             f"{res['ate_rmse_se3']:.4f} m (SE3) over {res['n']} poses, plot {out / 'ate.png'}")


GS_REPO = Path(__file__).resolve().parent / "third_party" / "gaussian-splatting"


def stage_gs(cfg: DictConfig) -> None:
    """Train + render 3DGS from the vo output. Layout train.py expects:
    <source>/images/<name> and <source>/sparse/0/{cameras,images,points3D}.bin,
    both symlinked from the dataset and the vo stage. Output: <out>/gs/ with
    point_cloud/iteration_N/point_cloud.ply and test/ours_N/{renders,gt}/."""
    import shlex
    import subprocess

    out = Path(cfg.output_dir).resolve()
    sparse = out / "sparse" / "0"
    if not (sparse / "images.bin").exists():
        raise FileNotFoundError(f"{sparse} missing: run stages=[vo] first")
    python = Path(cfg.gs.python).expanduser()
    if not python.exists():
        raise FileNotFoundError(f"3DGS interpreter {python} not found: run env/gaussian_splatting.sh")
    source, model = out / "gs_source", out / "gs"
    (source / "sparse").mkdir(parents=True, exist_ok=True)
    _symlink(Path(cfg.dataset.image_dir).resolve(), source / "images")
    _symlink(sparse, source / "sparse" / "0")
    n = str(cfg.gs.iterations)
    train = [python, "-u", GS_REPO / "train.py", "-s", source, "-m", model,
             "--iterations", n, "--test_iterations", n, "--save_iterations", n,
             "--resolution", str(cfg.gs.resolution), "--data_device", cfg.gs.data_device,
             "--disable_viewer"] + (["--eval"] if cfg.gs.eval else [])
    render = [python, GS_REPO / "render.py", "-m", model, "--iteration", n, "--skip_train"]
    with (out / "gs.log").open("a") as f:
        for cmd in (train, render):
            cmd = [str(c) for c in cmd]
            log.info(shlex.join(cmd))
            f.write(shlex.join(cmd) + "\n"); f.flush()
            subprocess.run(cmd, cwd=GS_REPO, stdout=f, stderr=subprocess.STDOUT, check=True)
    log.info(f"3DGS model in {model}, renders in {model / 'test' / f'ours_{n}'}, log {out / 'gs.log'}")


def _symlink(target: Path, link: Path) -> None:
    if link.is_symlink() or link.exists():
        link.unlink()
    link.symlink_to(target)


STAGES = {"vo": stage_vo, "viz": stage_viz, "eval": stage_eval, "gs": stage_gs}


@hydra.main(config_path="configs", config_name="config", version_base="1.3")
def main(cfg: DictConfig) -> None:
    Path(cfg.output_dir).mkdir(parents=True, exist_ok=True)
    for stage in cfg.stages:
        log.info(f"=== stage {stage} ===")
        STAGES[stage](cfg)


if __name__ == "__main__":
    main()

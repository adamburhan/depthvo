"""Frontend registry: `build_frontend(cfg, intrinsics)` picks by cfg.name.
To add one: subclass Frontend (base.py), add a branch here and a configs/frontend/<name>.yaml."""

from depthvo.frontend.base import Frontend, FrameTracks  # noqa: F401


def build_frontend(cfg, intrinsics):
    if cfg.name == "klt":
        from depthvo.frontend.klt import KLTFrontend
        return KLTFrontend(cfg, intrinsics)
    raise ValueError(f"unknown frontend {cfg.name!r}")

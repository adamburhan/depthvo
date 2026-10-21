"""Live VO from a webcam (or any video file), through the library API.

    uv run scripts/webcam.py calib.yaml                  # device 0
    uv run scripts/webcam.py calib.yaml --source 1       # another device
    uv run scripts/webcam.py calib.yaml --source clip.mp4 --out outputs/clip

calib.yaml is the intrinsics file of that camera (fx fy cx cy width height dist). Ctrl-C (or end of file) stops and
writes traj.txt / sparse/0 / reproj.csv to --out. Add --viewer for the viser window.
"""

import argparse
import time

import cv2

from depthvo import Frame, Intrinsics, VisualOdometry


def frames_from_capture(source, intr):
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"cannot open video source {source!r}")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, intr.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, intr.height)
    idx, t0 = 0, time.time()
    while True:
        ok, bgr = cap.read()
        if not ok:
            return
        if bgr.shape[1] != intr.width or bgr.shape[0] != intr.height:
            raise RuntimeError(f"camera gives {bgr.shape[1]}x{bgr.shape[0]}, calibration says {intr.width}x{intr.height}")
        gray = intr.undistort(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY))
        yield Frame(idx, time.time() - t0, f"{idx:06d}.png", gray)
        idx += 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("intrinsics")
    ap.add_argument("--source", default="0", help="device index or video path")
    ap.add_argument("--frontend", default="klt")
    ap.add_argument("--backend", default="window")
    ap.add_argument("--out", default="outputs/webcam")
    ap.add_argument("--viewer", action="store_true")
    args = ap.parse_args()

    intr = Intrinsics.load(args.intrinsics)
    source = int(args.source) if args.source.isdigit() else args.source
    vo = VisualOdometry(intr, args.frontend, args.backend)
    viewer = None
    if args.viewer:
        from omegaconf import OmegaConf
        from depthvo.viz.live import LiveViewer
        viewer = LiveViewer(OmegaConf.create({"port": 8080, "fps": 30, "point_size": 0.01, "frustum_scale": 0.15,
                                              "open_browser": True, "wait_for_client_s": 30}), intr)
    try:
        for frame in frames_from_capture(source, intr):
            r = vo.step(frame)
            if viewer:
                viewer.update(frame, r, vo)
            if r.is_keyframe:
                print(r)
    except KeyboardInterrupt:
        pass
    print("saved to", vo.save(args.out))
    if viewer:
        viewer.block()


if __name__ == "__main__":
    main()

"""Run a frontend over a sequence and write a video of the tracks: a line from
each track's position at the last keyframe to its current position, red frame
border on keyframes. Sanity check for a frontend before looking at the backend.

    uv run scripts/viz_tracks.py data/tum/freiburg1_xyz/rgb \
        configs/dataset/intrinsics/tum_freiburg1.yaml configs/frontend/klt.yaml \
        --out tracks.mp4 --max-frames 300
"""

import argparse

import cv2
import numpy as np
from omegaconf import OmegaConf

from depthvo.frontend import build_frontend
from depthvo.io.intrinsics import Intrinsics
from depthvo.io.sequence import ImageSequence


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("image_dir")
    ap.add_argument("intrinsics")
    ap.add_argument("frontend_cfg")
    ap.add_argument("--out", default="tracks.mp4")
    ap.add_argument("--max-frames", type=int, default=None)
    ap.add_argument("--stride", type=int, default=1)
    args = ap.parse_args()

    intr = Intrinsics.load(args.intrinsics)
    seq = ImageSequence(args.image_dir, intr, args.stride, args.max_frames)
    frontend = build_frontend(OmegaConf.load(args.frontend_cfg), intr)
    writer = cv2.VideoWriter(args.out, cv2.VideoWriter_fourcc(*"avc1"), 30, (intr.width, intr.height))

    kf_uv, n_kf, lengths = {}, 0, {}
    for frame in seq:
        t = frontend.process(frame)
        for i in t.track_ids:
            lengths[i] = lengths.get(i, 0) + 1
        img = cv2.cvtColor(frame.gray, cv2.COLOR_GRAY2BGR)
        for uv, i in zip(t.uv, t.track_ids):
            p = tuple(int(round(x)) for x in uv)
            if i in kf_uv:
                cv2.line(img, tuple(int(round(x)) for x in kf_uv[i]), p, (0, 255, 0), 1)
            cv2.circle(img, p, 2, (0, 255, 0) if i in kf_uv else (255, 128, 0), -1)
        if t.is_keyframe:
            n_kf += 1
            kf_uv = dict(zip(t.track_ids.tolist(), t.uv))
            cv2.rectangle(img, (0, 0), (intr.width - 1, intr.height - 1), (0, 0, 255), 6)
        cv2.putText(img, f"f{frame.idx} kf{n_kf} tracks={len(t.uv)}", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        writer.write(img)
    writer.release()
    L = np.array(list(lengths.values()))
    print(f"{len(seq)} frames, {n_kf} keyframes (every {len(seq)/max(n_kf,1):.1f} frames), "
          f"{len(L)} tracks, length median {np.median(L):.0f} / p90 {np.percentile(L, 90):.0f} frames")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()

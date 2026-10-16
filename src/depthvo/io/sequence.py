"""Image sequence = a directory of frames, sorted by filename. Timestamps come
from `timestamps.txt` (one float per line, same order) if present, else from the
filenames when they parse as floats (TUM: 1305031102.175304.png), else the index."""

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

IMAGE_EXTS = {".png", ".jpg", ".jpeg"}


@dataclass
class Frame:
    idx: int
    timestamp: float
    name: str
    gray: np.ndarray


class ImageSequence:
    def __init__(self, image_dir, intrinsics, stride=1, max_frames=None):
        self.image_dir = Path(image_dir)
        self.intrinsics = intrinsics
        self.paths = sorted(p for p in self.image_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS)
        self.paths = self.paths[::stride][:max_frames]
        if not self.paths:
            raise FileNotFoundError(f"no images in {self.image_dir}")
        ts_file = self.image_dir / "timestamps.txt"
        if ts_file.exists():
            ts = np.loadtxt(ts_file, dtype=float).ravel()[::stride][:max_frames]
            assert len(ts) == len(self.paths), "timestamps.txt length != image count"
            self.timestamps = ts
        else:
            try:
                self.timestamps = np.array([float(p.stem) for p in self.paths])
            except ValueError:
                self.timestamps = np.arange(len(self.paths), dtype=float)

    def __len__(self):
        return len(self.paths)

    def __iter__(self):
        for i, p in enumerate(self.paths):
            img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
            if img is None:
                raise IOError(f"could not read {p}")
            yield Frame(i, float(self.timestamps[i]), p.name, self.intrinsics.undistort(img))

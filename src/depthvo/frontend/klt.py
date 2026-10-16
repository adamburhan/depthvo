"""KLT frontend: Shi-Tomasi corners tracked frame-to-frame with pyramidal
Lucas-Kanade (forward-backward check), corners re-seeded when the count drops.
Dense, cheap tracks; a lost track is gone for good (no descriptors)."""

import cv2
import numpy as np

from depthvo.frontend.base import Frontend, FrameTracks

LK_PARAMS = dict(winSize=(21, 21), maxLevel=3,
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))


class KLTFrontend(Frontend):
    def __init__(self, cfg, intrinsics):
        super().__init__(cfg, intrinsics)
        self.prev_gray = None
        self.uv = np.zeros((0, 2), np.float32)
        self.ids = np.zeros(0, np.int64)
        self.kf_uv = {}   # track id -> uv at the last keyframe (for parallax + survival)

    def process(self, frame) -> FrameTracks:
        if self.prev_gray is not None and len(self.uv):
            self._track(frame.gray)
        survivors = np.array([i in self.kf_uv for i in self.ids], dtype=bool)
        if self.prev_gray is None:
            keyframe = True
        else:
            parallax = (np.median(np.linalg.norm(
                self.uv[survivors] - np.array([self.kf_uv[i] for i in self.ids[survivors]]), axis=1))
                if survivors.any() else np.inf)
            keyframe = self.is_keyframe(survivors.sum(), len(self.kf_uv), parallax)
        if keyframe or len(self.uv) < self.cfg.min_features:
            self._reseed(frame.gray)
        if keyframe:
            self.kf_uv = dict(zip(self.ids.tolist(), self.uv))
        self.prev_gray = frame.gray
        return FrameTracks(frame.idx, frame.timestamp, frame.name, self.uv.copy(), self.ids.copy(), keyframe)

    def _track(self, gray):
        p0 = self.uv.reshape(-1, 1, 2)
        p1, st, _ = cv2.calcOpticalFlowPyrLK(self.prev_gray, gray, p0, None, **LK_PARAMS)
        p0b, stb, _ = cv2.calcOpticalFlowPyrLK(gray, self.prev_gray, p1, None, **LK_PARAMS)
        fb = np.linalg.norm(p0 - p0b, axis=2).ravel()
        h, w = gray.shape
        inside = (p1[:, 0, 0] >= 0) & (p1[:, 0, 0] < w) & (p1[:, 0, 1] >= 0) & (p1[:, 0, 1] < h)
        keep = (st.ravel() == 1) & (stb.ravel() == 1) & (fb < self.cfg.fb_threshold_px) & inside
        self.uv, self.ids = p1[keep, 0], self.ids[keep]

    def _reseed(self, gray):
        n_new = self.cfg.max_features - len(self.uv)
        if n_new <= 0:
            return
        mask = np.full(gray.shape, 255, np.uint8)
        for u, v in self.uv:
            cv2.circle(mask, (int(u), int(v)), self.cfg.min_distance_px, 0, -1)
        pts = cv2.goodFeaturesToTrack(gray, n_new, self.cfg.quality_level,
                                      self.cfg.min_distance_px, mask=mask)
        if pts is None:
            return
        pts = pts[:, 0].astype(np.float32)
        self.uv = np.vstack([self.uv, pts])
        self.ids = np.concatenate([self.ids, self.new_track_ids(len(pts))])

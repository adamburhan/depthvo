"""ORB frontend: ORB keypoints matched (Hamming, Lowe ratio) against the ACTIVE
TRACKS, i.e. every track seen in the last `keep_keyframes` keyframes, each with
its latest descriptor and position. Matching against the recent map rather than
the last keyframe only lets a track survive a missed keyframe, which is what
gives tracks enough keyframes (and parallax) to triangulate. Outliers are
rejected with a fundamental-matrix RANSAC per source keyframe."""

import cv2
import numpy as np

from depthvo.frontend.base import Frontend, FrameTracks


class ORBFrontend(Frontend):
    def __init__(self, cfg, intrinsics):
        super().__init__(cfg, intrinsics)
        self.orb = cv2.ORB_create(nfeatures=cfg.n_features)
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
        self.tracks = {}      # track id -> [uv, descriptor, keyframe index of that observation]
        self.n_kf = 0
        self.ref_matches = None   # matches to the last keyframe at the first frame after it

    def process(self, frame) -> FrameTracks:
        kp, des = self.orb.detectAndCompute(frame.gray, None)
        uv = np.array([k.pt for k in kp], np.float32).reshape(-1, 2)
        ids = np.full(len(uv), -1, np.int64)
        if not self.tracks:
            keyframe = True
        else:
            q, tid = self._match(uv, des)
            ids[q] = tid
            last = self.n_kf - 1
            at_last = np.array([self.tracks[t][2] == last for t in tid], bool)
            n_last = int(at_last.sum())
            if self.ref_matches is None:
                self.ref_matches = max(n_last, 1)
            par = (np.median(np.linalg.norm(uv[q[at_last]] - np.array([self.tracks[t][0] for t in tid[at_last]]), axis=1))
                   if n_last else np.inf)
            keyframe = self.is_keyframe(n_last, self.ref_matches, par)
        if keyframe:
            new = ids < 0
            ids[new] = self.new_track_ids(int(new.sum()))
            for k in range(len(uv)):
                self.tracks[ids[k]] = [uv[k], des[k], self.n_kf]
            oldest = self.n_kf - self.cfg.keep_keyframes + 1
            self.tracks = {t: v for t, v in self.tracks.items() if v[2] >= oldest}
            self.n_kf += 1
            self.ref_matches = None
        matched = ids >= 0
        return FrameTracks(frame.idx, frame.timestamp, frame.name, uv[matched], ids[matched], keyframe)

    def _match(self, uv, des):
        """Ratio-test matches frame -> active tracks, one per track, then a
        fundamental-matrix RANSAC per source keyframe. -> (frame idx, track id)."""
        tids = np.fromiter(self.tracks.keys(), np.int64, len(self.tracks))
        if des is None or len(tids) == 0:
            return np.zeros(0, int), np.zeros(0, np.int64)
        t_uv = np.array([self.tracks[t][0] for t in tids], np.float32)
        t_des = np.array([self.tracks[t][1] for t in tids], np.uint8)
        t_kf = np.array([self.tracks[t][2] for t in tids], int)
        # track -> frame direction: the ratio test then compares two distinct
        # frame keypoints, not two duplicate tracks of one point; one track per keypoint
        best = {}
        for m in self.matcher.knnMatch(t_des, des, k=2):
            if len(m) == 2 and m[0].distance < self.cfg.ratio * m[1].distance:
                if m[0].trainIdx not in best or m[0].distance < best[m[0].trainIdx].distance:
                    best[m[0].trainIdx] = m[0]
        q = np.array([m.trainIdx for m in best.values()], int)   # frame keypoint
        t = np.array([m.queryIdx for m in best.values()], int)   # track
        keep = np.zeros(len(q), bool)
        for kf in np.unique(t_kf[t]):
            g = np.where(t_kf[t] == kf)[0]
            if len(g) < 8:
                continue
            _, inl = cv2.findFundamentalMat(uv[q[g]], t_uv[t[g]], cv2.FM_RANSAC, self.cfg.ransac_px, 0.999)
            if inl is not None:
                keep[g] = inl.ravel().astype(bool)
        return q[keep], tids[t[keep]]

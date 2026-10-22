"""Live viser viewer for the vo stage: image feed with tracks in the GUI panel,
landmarks, keyframe frusta, trajectory and the current camera in the 3D scene.
Opens http://localhost:<port>; the vo stage blocks at the end so you can inspect."""

import time
import webbrowser

import cv2
import numpy as np
import viser
from scipy.spatial.transform import Rotation


class LiveViewer:
    def __init__(self, cfg, intrinsics):
        self.cfg = cfg
        self.intr = intrinsics
        self.server = viser.ViserServer(port=cfg.port, verbose=False)
        self.server.scene.set_up_direction("-y")   # OpenCV camera frame: y down
        self.fov = 2 * np.arctan(intrinsics.height / (2 * intrinsics.fy))
        self.aspect = intrinsics.width / intrinsics.height
        self.image = self.server.gui.add_image(np.zeros((intrinsics.height, intrinsics.width, 3), np.uint8), label="tracks")
        self.status = self.server.gui.add_markdown("")
        self.kf_uv = {}
        self.n_kf = 0
        self.t_last = 0.0

        @self.server.on_client_connect
        def _(client):
            # start behind camera 0 looking down its optical axis (+Z), slightly above
            client.camera.position = (0.0, -0.3 * cfg.frustum_scale * 10, -1.5 * cfg.frustum_scale * 10)
            client.camera.look_at = (0.0, 0.0, 1.0)

        url = f"http://localhost:{cfg.port}"
        print(f"viewer at {url}")
        if cfg.open_browser:
            webbrowser.open(url)
        # wait for a client so the start of the sequence isn't missed
        t0 = time.time()
        while not self.server.get_clients() and time.time() - t0 < cfg.wait_for_client_s:
            time.sleep(0.1)

    def update(self, frame, result, vo):
        """result: the FrameResult of vo.step(frame)."""
        backend = vo.backend
        ids, uv = backend.frame_obs[result.index]
        img = self._draw(frame, uv, ids)
        if result.is_keyframe:
            cv2.rectangle(img, (0, 0), (img.shape[1] - 1, img.shape[0] - 1), (255, 0, 0), 6)
        self.image.image = img
        if result.is_keyframe:
            self.kf_uv = dict(zip(ids.tolist(), uv))
            self.n_kf += 1
            self._update_map(backend)
        T = result.pose
        if T is not None:
            self.server.scene.add_camera_frustum(
                "/current", self.fov, self.aspect, scale=self.cfg.frustum_scale, color=(255, 80, 0),
                wxyz=self._wxyz(T), position=T[:3, 3])
        rp = (f"reproj {result.reproj_before.rms:.2f} -> {result.reproj_after.rms:.2f} px"
              if result.is_keyframe else f"reproj {result.reproj_after.rms:.2f} px")
        self.status.content = (f"{result.state}  |  frame {frame.idx}  |  keyframes {self.n_kf}  |  "
                               f"landmarks {len(backend.landmarks)}  |  tracks {len(uv)}  |  {rp}")
        # pace to at most cfg.fps so the browser can keep up
        dt = 1.0 / self.cfg.fps - (time.time() - self.t_last)
        if dt > 0:
            time.sleep(dt)
        self.t_last = time.time()

    def block(self):
        print("vo done, viewer still serving (Ctrl-C to exit)")
        try:
            while True:
                time.sleep(1.0)
        except KeyboardInterrupt:
            pass

    # ---- helpers -------------------------------------------------------------

    def _draw(self, frame, uvs, ids):
        img = cv2.cvtColor(frame.gray, cv2.COLOR_GRAY2RGB)
        for uv, tid in zip(uvs, ids.tolist()):
            p = tuple(int(round(x)) for x in uv)
            if tid in self.kf_uv:
                cv2.line(img, tuple(int(round(x)) for x in self.kf_uv[tid]), p, (0, 200, 0), 1)
            cv2.circle(img, p, 2, (0, 200, 0) if tid in self.kf_uv else (0, 128, 255), -1)
        return img

    def _update_map(self, backend):
        if backend.landmarks:
            ids, xyz = zip(*backend.landmarks.items())
            g = np.array([backend.colors.get(i, 128) for i in ids], np.uint8)
            self.server.scene.add_point_cloud(
                "/map", np.array(xyz), np.stack([g, g, g], -1), point_size=self.cfg.point_size)
        kf = sorted(backend.poses)
        for i in kf[-2:]:   # new keyframe + its predecessor (just refined)
            T = backend.poses[i]
            self.server.scene.add_camera_frustum(
                f"/keyframes/{i}", self.fov, self.aspect, scale=self.cfg.frustum_scale * 0.6,
                color=(40, 120, 255), wxyz=self._wxyz(T), position=T[:3, 3])
        if len(kf) >= 2:
            P = np.array([backend.poses[i][:3, 3] for i in kf])
            self.server.scene.add_line_segments(
                "/trajectory", np.stack([P[:-1], P[1:]], 1), colors=(40, 120, 255),
                thickness=self.cfg.frustum_scale * 0.05)

    @staticmethod
    def _wxyz(T):
        q = Rotation.from_matrix(T[:3, :3]).as_quat()   # xyzw
        return np.array([q[3], q[0], q[1], q[2]])

"""viser viewer for a finished run: keyframe frusta, trajectory and the sparse
point cloud, read from the COLMAP model the vo stage wrote."""

import time

import numpy as np
import pycolmap
import viser
from scipy.spatial.transform import Rotation

from depthvo.geometry.transforms import from_colmap_cam_from_world


def serve(model_dir, port=8080, frustum_scale=0.15, point_size=0.01):
    rec = pycolmap.Reconstruction(model_dir)
    cam = next(iter(rec.cameras.values()))
    fov = 2 * np.arctan(cam.height / (2 * cam.focal_length_y))
    server = viser.ViserServer(port=port, verbose=False)
    server.scene.set_up_direction("-y")
    xyz = np.array([p.xyz for p in rec.points3D.values()])
    rgb = np.array([p.color for p in rec.points3D.values()], np.uint8)
    server.scene.add_point_cloud("/map", xyz, rgb, point_size=point_size)
    centers = []
    for image in sorted(rec.images.values(), key=lambda im: im.name):
        T = from_colmap_cam_from_world(image.cam_from_world())
        q = Rotation.from_matrix(T[:3, :3]).as_quat()
        server.scene.add_camera_frustum(f"/keyframes/{image.name}", fov, cam.width / cam.height,
                                        scale=frustum_scale, color=(40, 120, 255),
                                        wxyz=(q[3], q[0], q[1], q[2]), position=T[:3, 3])
        centers.append(T[:3, 3])
    P = np.array(centers)
    server.scene.add_line_segments("/trajectory", np.stack([P[:-1], P[1:]], 1), colors=(40, 120, 255),
                                   thickness=frustum_scale * 0.05)
    print(f"{len(rec.images)} keyframes, {len(xyz)} points at http://localhost:{port} (Ctrl-C to exit)")
    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        pass

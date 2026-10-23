"""viser viewer for a finished run: keyframe frusta, trajectory and the sparse
point cloud, read from the COLMAP model(s) the vo stage wrote (sparse/<segment>).
The last keyframe of every segment that ended in a tracking loss is drawn red."""

import time
from pathlib import Path

import numpy as np
import pycolmap
import viser
from scipy.spatial.transform import Rotation

from depthvo.geometry.transforms import from_colmap_cam_from_world


def serve(sparse_dir, port=8080, frustum_scale=0.15, point_size=0.01):
    sparse_dir = Path(sparse_dir)
    models = sorted((p for p in sparse_dir.iterdir() if p.is_dir() and p.name.isdigit()), key=lambda p: int(p.name))
    server = viser.ViserServer(port=port, verbose=False)
    server.scene.set_up_direction("-y")
    n_img = n_pts = 0
    for k, model in enumerate(models):
        rec = pycolmap.Reconstruction(model)
        cam = next(iter(rec.cameras.values()))
        fov = 2 * np.arctan(cam.height / (2 * cam.focal_length_y))
        xyz = np.array([p.xyz for p in rec.points3D.values()])
        rgb = np.array([p.color for p in rec.points3D.values()], np.uint8)
        server.scene.add_point_cloud(f"/seg{model.name}/map", xyz, rgb, point_size=point_size)
        images = sorted(rec.images.values(), key=lambda im: im.name)
        centers = []
        for j, image in enumerate(images):
            T = from_colmap_cam_from_world(image.cam_from_world())
            q = Rotation.from_matrix(T[:3, :3]).as_quat()
            lost_here = k < len(models) - 1 and j == len(images) - 1   # segment ended in a loss
            server.scene.add_camera_frustum(f"/seg{model.name}/keyframes/{image.name}", fov, cam.width / cam.height,
                                            scale=frustum_scale, color=(230, 30, 30) if lost_here else (40, 120, 255),
                                            wxyz=(q[3], q[0], q[1], q[2]), position=T[:3, 3])
            centers.append(T[:3, 3])
        P = np.array(centers)
        server.scene.add_line_segments(f"/seg{model.name}/trajectory", np.stack([P[:-1], P[1:]], 1),
                                       colors=(40, 120, 255), thickness=frustum_scale * 0.05)
        n_img += len(images); n_pts += len(xyz)
    print(f"{len(models)} segment(s), {n_img} keyframes, {n_pts} points at http://localhost:{port} (Ctrl-C to exit)")
    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        pass

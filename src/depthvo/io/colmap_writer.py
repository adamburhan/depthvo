"""Write the VO result as a COLMAP sparse model (cameras/images/points3D.bin).
This is the handoff format: gaussian-splatting, COLMAP GUI, nerfstudio and
viser all read it, so nothing downstream needs to know about depthvo."""

from pathlib import Path

import numpy as np
import pycolmap

from depthvo.geometry.transforms import to_colmap_cam_from_world


def write_colmap_model(out_dir, intrinsics, kf_names, poses, landmarks, obs, colors=None):
    """
    kf_names:  list of image names, index = keyframe index
    poses:     {kf_idx: T_w_c}
    landmarks: {track_id: xyz_w}
    obs:       {track_id: [(kf_idx, uv), ...]}
    colors:    {track_id: (r, g, b)} uint8, optional
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rec = pycolmap.Reconstruction()
    cam = pycolmap.Camera(model="PINHOLE", width=intrinsics.width, height=intrinsics.height,
                          params=[intrinsics.fx, intrinsics.fy, intrinsics.cx, intrinsics.cy], camera_id=1)
    rec.add_camera(cam)
    rig = pycolmap.Rig(rig_id=1)
    rig.add_ref_sensor(cam.sensor_id)
    rec.add_rig(rig)

    # per keyframe: its 2D observations of TRIANGULATED tracks, in a fixed order
    kf_obs = {i: [] for i in poses}             # kf_idx -> [(track_id, uv)]
    for tid, ob in obs.items():
        if tid in landmarks:
            for i, uv in ob:
                if i in kf_obs:
                    kf_obs[i].append((tid, uv))
    p2d_index = {}                               # (kf_idx, track_id) -> point2D idx
    for i in sorted(poses):
        image_id = i + 1                         # COLMAP ids start at 1
        frame = pycolmap.Frame(frame_id=image_id, rig_id=1)
        frame.rig_from_world = to_colmap_cam_from_world(poses[i])
        frame.add_data_id(pycolmap.data_t(cam.sensor_id, image_id))
        rec.add_frame(frame)
        pts = [pycolmap.Point2D(np.asarray(uv, float)) for _, uv in kf_obs[i]]
        for k, (tid, _) in enumerate(kf_obs[i]):
            p2d_index[(i, tid)] = k
        img = pycolmap.Image(name=kf_names[i], points2D=pycolmap.Point2DList(pts),
                             camera_id=1, image_id=image_id)
        img.frame_id = image_id
        rec.add_image(img)
        rec.register_frame(image_id)

    for tid, xyz in landmarks.items():
        track = pycolmap.Track()
        for i, _ in obs[tid]:
            if (i, tid) in p2d_index:
                track.add_element(i + 1, p2d_index[(i, tid)])
        if track.length() < 2:
            continue
        rgb = np.asarray(colors.get(tid, (128, 128, 128)) if colors else (128, 128, 128), np.uint8)
        pid = rec.add_point3D(np.asarray(xyz, float), track, rgb)
        errs = []
        for el in track.elements:
            img = rec.images[el.image_id]
            img.set_point3D_for_point2D(el.point2D_idx, pid)
            errs.append(np.linalg.norm(img.project_point(np.asarray(xyz, float)) - img.points2D[el.point2D_idx].xy))
        rec.points3D[pid].error = float(np.mean(errs))

    rec.write(out_dir)
    return rec


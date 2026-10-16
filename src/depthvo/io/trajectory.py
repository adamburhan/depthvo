"""TUM trajectory file I/O (one 'timestamp tx ty tz qx qy qz qw' line per pose, T_w_c)."""

from depthvo.geometry.transforms import from_tum_line, tum_line


def write_tum(path, timestamps, poses_w_c):
    with open(path, "w") as f:
        for ts, T in zip(timestamps, poses_w_c):
            f.write(tum_line(ts, T) + "\n")


def read_tum(path):
    stamps, poses = [], []
    for line in open(path):
        if line.strip() and not line.startswith("#"):
            ts, T = from_tum_line(line)
            stamps.append(ts)
            poses.append(T)
    return stamps, poses

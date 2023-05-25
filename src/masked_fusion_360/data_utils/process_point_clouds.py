import numpy as np


def spherical_projection(
    pc,
    fov_up=np.deg2rad(2),
    fov_down=np.deg2rad(-24.8),
    num_lasers=64,
    points_per_laser=1024,  # Exact: 360° / 0.35° = 1028
) -> np.ndarray:
    """Sperical projection of a point cloud

    pc-format: (x, y, z, intensity, range)
    """
    fov = np.abs(fov_up) + np.abs(fov_down)
    img = np.zeros((num_lasers, points_per_laser, pc.shape[-1]))

    yaw = -np.arctan2(pc[:, 1], pc[:, 0])
    pitch = np.arcsin(pc[:, 2] / pc[:, 4])

    u = 1.0 - (pitch + np.abs(fov_down)) / fov
    v = 0.5 * (yaw / np.pi + 1.0)
    u *= num_lasers
    v *= points_per_laser

    u = np.floor(u)
    v = np.floor(v)
    u = np.minimum(num_lasers - 1, u)
    v = np.minimum(points_per_laser - 1, v)
    u = np.maximum(0.0, u)
    v = np.maximum(0.0, v)

    u = np.nan_to_num(u, nan=0)
    v = np.nan_to_num(v, nan=0)

    for i in range(u.shape[0]):
        img[int(u[i]), int(v[i]), :] = pc[i, :]

    return u, v, img


def read_kitti_point_cloud(lidar_file):
    return np.fromfile(lidar_file, dtype=np.float32).reshape(-1, 4)


def add_range(pc: np.ndarray) -> np.ndarray:
    """Add range values to a point cloud"""
    range = np.sqrt(pc[:, 0] ** 2 + pc[:, 1] ** 2, pc[:, 2] ** 2)
    return np.concatenate((pc, range[:, np.newaxis]), axis=1)

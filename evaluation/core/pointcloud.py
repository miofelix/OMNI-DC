"""Small RGB-D point-cloud renderer for KITTI visualization."""

from pathlib import Path
from typing import Union

import numpy as np
from scipy.spatial import cKDTree

PathLike = Union[str, Path]


def load_intrinsics(path: PathLike) -> np.ndarray:
    matrix = np.loadtxt(str(path), dtype=np.float32)
    if matrix.shape == (9,):
        matrix = matrix.reshape(3, 3)
    if matrix.shape != (3, 3):
        raise ValueError(f"Expected 3x3 KITTI intrinsics, got {matrix.shape}: {path}")
    if not np.isfinite(matrix[0, 0]) or not np.isfinite(matrix[1, 1]):
        raise ValueError(f"KITTI intrinsics contain non-finite focal lengths: {path}")
    if matrix[0, 0] == 0 or matrix[1, 1] == 0:
        raise ValueError(f"KITTI intrinsics contain zero focal lengths: {path}")
    return matrix


def _filter_knn(points: np.ndarray, colors: np.ndarray, k: int, ratio: float):
    if k < 1 or len(points) <= k:
        return points, colors
    try:
        distances, _ = cKDTree(points).query(
            points, k=min(k + 1, len(points)), workers=-1
        )
    except Exception:
        return points, colors
    if distances.ndim == 1:
        return points, colors
    means = distances[:, 1:].mean(axis=1)
    finite = np.isfinite(means)
    if not finite.any():
        return points, colors
    values = means[finite]
    keep = finite & (means <= values.mean() + ratio * values.std())
    return (points[keep], colors[keep]) if keep.any() else (points, colors)


def render_pointcloud(
    depth: np.ndarray,
    rgb: np.ndarray,
    intrinsics: np.ndarray,
    rot_x_deg: float = 25.0,
    rot_y_deg: float = 15.0,
    knn_k: int = 16,
    knn_std_ratio: float = 2.0,
    disable_knn: bool = False,
) -> np.ndarray:
    depth = np.asarray(depth, dtype=np.float32).squeeze()
    rgb = np.asarray(rgb)
    if depth.ndim != 2 or rgb.shape[:2] != depth.shape:
        raise ValueError(f"RGB/depth shape mismatch: rgb={rgb.shape}, depth={depth.shape}")

    height, width = depth.shape
    yy, xx = np.indices((height, width), dtype=np.float32)
    valid = np.isfinite(depth) & (depth > 0)
    if not valid.any():
        return np.full((height, width, 3), 255, dtype=np.uint8)

    fx, fy = intrinsics[0, 0], intrinsics[1, 1]
    cx, cy = intrinsics[0, 2], intrinsics[1, 2]
    z = depth[valid]
    points = np.stack(((xx[valid] - cx) * z / fx, (yy[valid] - cy) * z / fy, z), axis=1)
    colors = np.clip(rgb, 0, 255).astype(np.uint8)[valid]
    if not disable_knn:
        points, colors = _filter_knn(points, colors, knn_k, knn_std_ratio)

    center = points.mean(axis=0)
    points = points - center
    rx, ry = np.radians(rot_x_deg), np.radians(rot_y_deg)
    cos_x, sin_x, cos_y, sin_y = np.cos(rx), np.sin(rx), np.cos(ry), np.sin(ry)
    y = points[:, 1] * cos_x - points[:, 2] * sin_x
    z = points[:, 1] * sin_x + points[:, 2] * cos_x
    x = points[:, 0]
    points = np.stack((x * cos_y + z * sin_y, y, -x * sin_y + z * cos_y), axis=1)
    points += center

    z = points[:, 2]
    keep = z > 1e-4
    if not keep.any():
        return np.full((height, width, 3), 255, dtype=np.uint8)
    u = np.round(points[keep, 0] * fx / z[keep] + cx).astype(np.int32)
    v = np.round(points[keep, 1] * fy / z[keep] + cy).astype(np.int32)
    colors, z = colors[keep], z[keep]
    keep = (u >= 0) & (u < width) & (v >= 0) & (v < height)
    u, v, colors, z = u[keep], v[keep], colors[keep], z[keep]

    image = np.full((height, width, 3), 255, dtype=np.uint8)
    order = np.argsort(-z)
    u, v, colors = u[order], v[order], colors[order]
    for du in range(2):
        for dv in range(2):
            uu = np.clip(u + du, 0, width - 1)
            vv = np.clip(v + dv, 0, height - 1)
            image[vv, uu] = colors
    return image

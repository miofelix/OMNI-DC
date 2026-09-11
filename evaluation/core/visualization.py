from __future__ import annotations

from pathlib import Path
from typing import Optional

import cv2
import matplotlib
import numpy as np

from evaluation.core.pointcloud import load_intrinsics, render_pointcloud


def colorize_depth(depth: np.ndarray, min_depth: float, max_depth: float) -> np.ndarray:
    valid = np.isfinite(depth) & (depth > 0.0)
    denominator = max(max_depth - min_depth, 1e-6)
    scaled = np.clip((np.nan_to_num(depth, nan=min_depth) - min_depth) / denominator, 0, 1)
    colored = cv2.applyColorMap((scaled * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
    colored[~valid] = 0
    return colored


def resize_depth(depth: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    if depth.shape == shape:
        return depth
    return cv2.resize(depth, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)


def add_label(image: np.ndarray, label: str) -> np.ndarray:
    result = image.copy()
    cv2.rectangle(result, (0, 0), (max(120, len(label) * 11), 30), (0, 0, 0), -1)
    cv2.putText(
        result,
        label,
        (8, 21),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    return result


def save_visualization(
    path: Path,
    rgb: np.ndarray,
    raw_depth: np.ndarray,
    prediction: np.ndarray,
    gt_depth: Optional[np.ndarray],
    min_depth: float,
    max_depth: float,
) -> None:
    target_shape = rgb.shape[:2]
    panels = [add_label(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), "RGB")]
    panels.append(
        add_label(
            colorize_depth(resize_depth(raw_depth, target_shape), min_depth, max_depth),
            "Raw depth",
        )
    )
    panels.append(
        add_label(
            colorize_depth(resize_depth(prediction, target_shape), min_depth, max_depth),
            "Prediction",
        )
    )
    if gt_depth is not None:
        panels.append(
            add_label(
                colorize_depth(resize_depth(gt_depth, target_shape), min_depth, max_depth),
                "Ground truth",
            )
        )

    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), np.concatenate(panels, axis=1)):
        raise OSError(f"Could not write visualization: {path}")


def colorize_kitti_depth(
    depth: np.ndarray, min_depth: float, max_depth: float
) -> np.ndarray:
    valid = np.isfinite(depth) & (depth > 0.0)
    denominator = max(max_depth - min_depth, 1e-6)
    scaled = np.clip((np.nan_to_num(depth, nan=min_depth) - min_depth) / denominator, 0, 1)
    colored = (matplotlib.colormaps["Spectral_r"](scaled)[..., :3] * 255).astype(np.uint8)
    colored[~valid] = 0
    return colored


def save_kitti_prediction_visualization(
    path: Path,
    prediction: np.ndarray,
    min_depth: float,
    max_depth: float,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    prediction_rgb = colorize_kitti_depth(prediction, min_depth, max_depth)
    if not cv2.imwrite(str(path), cv2.cvtColor(prediction_rgb, cv2.COLOR_RGB2BGR)):
        raise OSError(f"Could not write KITTI prediction visualization: {path}")


def save_kitti_pointcloud_visualization(
    pointcloud_path: Path,
    rgb: np.ndarray,
    prediction: np.ndarray,
    intrinsics_path: Path,
    rot_x_deg: float,
    rot_y_deg: float,
    knn_k: int,
    knn_std_ratio: float,
    disable_knn: bool,
) -> None:
    pointcloud_path.parent.mkdir(parents=True, exist_ok=True)
    pointcloud = render_pointcloud(
        prediction,
        rgb,
        load_intrinsics(intrinsics_path),
        rot_x_deg=rot_x_deg,
        rot_y_deg=rot_y_deg,
        knn_k=knn_k,
        knn_std_ratio=knn_std_ratio,
        disable_knn=disable_knn,
    )
    if not cv2.imwrite(str(pointcloud_path), cv2.cvtColor(pointcloud, cv2.COLOR_RGB2BGR)):
        raise OSError(f"Could not write KITTI point-cloud visualization: {pointcloud_path}")

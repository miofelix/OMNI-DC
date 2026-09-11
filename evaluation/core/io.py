from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Tuple

os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")

import cv2  # noqa: E402
import numpy as np  # noqa: E402


def squeeze_depth(depth: Any) -> np.ndarray:
    array = np.asarray(depth)
    if array.ndim == 3 and array.shape[0] == 1:
        array = array[0]
    if array.ndim == 3:
        array = array[..., 0]
    if array.ndim != 2:
        raise ValueError(f"Expected a 2D depth map, got shape {array.shape}")
    return array


def read_rgb(path: Path) -> np.ndarray:
    image_bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise ValueError(f"Could not read RGB image: {path}")
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)


def read_single_channel(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ValueError(f"Could not read depth image: {path}")
    return squeeze_depth(image)


def read_raw_depth(
    path: Path, depth_scale: float, min_depth: float, max_depth: float
) -> np.ndarray:
    if depth_scale <= 0:
        raise ValueError(f"Depth scale must be positive for {path}: {depth_scale}")
    depth = read_single_channel(path).astype(np.float32) / float(depth_scale)
    valid = np.isfinite(depth) & (depth >= float(min_depth)) & (depth <= float(max_depth))
    return np.where(valid, depth, 0.0).astype(np.float32, copy=False)


def read_gt_depth(path: Path, depth_scale: float, min_depth: float, max_depth: float) -> np.ndarray:
    if depth_scale <= 0:
        raise ValueError(f"Depth scale must be positive for {path}: {depth_scale}")
    depth = read_single_channel(path).astype(np.float32) / float(depth_scale)
    valid = np.isfinite(depth) & (depth >= float(min_depth)) & (depth <= float(max_depth))
    return np.where(valid, depth, np.nan).astype(np.float32, copy=False)


def normalize_prediction(prediction: Any, target_shape: Tuple[int, int]) -> np.ndarray:
    pred = squeeze_depth(prediction).astype(np.float32, copy=False)
    valid = np.isfinite(pred) & (pred > 0.0)

    if pred.shape != target_shape:
        height, width = target_shape
        finite_pred = np.where(valid, pred, 0.0).astype(np.float32, copy=False)
        pred = cv2.resize(finite_pred, (width, height), interpolation=cv2.INTER_LINEAR)
        valid = cv2.resize(
            valid.astype(np.uint8),
            (width, height),
            interpolation=cv2.INTER_NEAREST,
        ).astype(bool)

    pred = pred.astype(np.float32, copy=False)
    pred[~valid | ~np.isfinite(pred) | (pred <= 0.0)] = np.nan
    return pred


def normalize_asdepth_prediction(prediction: Any, target_shape: Tuple[int, int]) -> np.ndarray:
    """Normalize a prediction using AS-Depth's KITTI inference semantics.

    AS-Depth only restores the two-dimensional target shape; it does not turn
    finite zero/negative values into NaNs and does not sanitize non-finite
    values before saving them for evaluation.
    """

    pred = squeeze_depth(prediction).astype(np.float32, copy=False)
    if pred.shape != target_shape:
        height, width = target_shape
        pred = cv2.resize(pred, (width, height), interpolation=cv2.INTER_LINEAR)
    return pred.astype(np.float32, copy=False)


def align_prediction_for_evaluation(
    prediction: np.ndarray,
    target_shape: Tuple[int, int],
    allow_resize: bool,
    sample_id: str,
) -> np.ndarray:
    pred = squeeze_depth(prediction).astype(np.float32, copy=False)
    if pred.shape == target_shape:
        return pred
    if not allow_resize:
        raise ValueError(
            f"Prediction/GT shape mismatch for {sample_id}: "
            f"prediction={pred.shape}, gt={target_shape}"
        )

    height, width = target_shape
    valid = np.isfinite(pred) & (pred > 0.0)
    resized = cv2.resize(
        np.where(valid, pred, 0.0).astype(np.float32, copy=False),
        (width, height),
        interpolation=cv2.INTER_NEAREST,
    )
    resized_valid = cv2.resize(
        valid.astype(np.uint8),
        (width, height),
        interpolation=cv2.INTER_NEAREST,
    ).astype(bool)
    resized[~resized_valid] = np.nan
    return resized.astype(np.float32, copy=False)


def load_prediction(path: Path) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(f"Prediction not found: {path}")
    return squeeze_depth(np.load(path, allow_pickle=False)).astype(np.float32, copy=False)

from __future__ import annotations

from typing import Dict

import numpy as np
import torch

METRIC_NAMES = (
    "mae",
    "rmse",
    "abs_rel",
    "delta_1_05",
    "delta_1_10",
    "delta_1_25",
)


def compute_depth_metrics(prediction: np.ndarray, target: np.ndarray) -> Dict[str, float]:
    # AS-Depth's evaluator computes the delta ratios with float32 Torch
    # tensors. Compute only the ratios at that precision for strict threshold
    # behavior, while retaining float64 aggregates for stable error reporting.
    pred = np.asarray(prediction, dtype=np.float64)
    gt = np.asarray(target, dtype=np.float64)
    if pred.shape != gt.shape:
        raise ValueError(f"Metric shape mismatch: prediction={pred.shape}, target={gt.shape}")

    valid = np.isfinite(pred) & np.isfinite(gt) & (pred > 0.0) & (gt > 0.0)
    if not np.any(valid):
        return {name: float("nan") for name in METRIC_NAMES}

    pred_valid = pred[valid]
    gt_valid = gt[valid]
    difference = pred_valid - gt_valid
    ratio = np.maximum(
        np.asarray(pred_valid, dtype=np.float32) / np.asarray(gt_valid, dtype=np.float32),
        np.asarray(gt_valid, dtype=np.float32) / np.asarray(pred_valid, dtype=np.float32),
    )

    return {
        "mae": float(np.mean(np.abs(difference))),
        "rmse": float(np.sqrt(np.mean(np.square(difference)))),
        "abs_rel": float(np.mean(np.abs(difference) / gt_valid)),
        "delta_1_05": float(np.mean(ratio < 1.05)),
        "delta_1_10": float(np.mean(ratio < 1.10)),
        "delta_1_25": float(np.mean(ratio < 1.25)),
    }


def compute_asdepth_depth_metrics(
    prediction: np.ndarray, target: np.ndarray
) -> Dict[str, float]:
    """Compute the metrics used by AS-Depth's ``eval_mp.py``.

    AS-Depth converts both arrays to float32 Torch tensors and only removes
    non-finite predictions from the valid ground-truth mask.  In particular,
    finite zero and negative predictions remain in the metric calculation,
    including the original ratio-based delta accuracy behavior.
    """

    pred_array = np.asarray(prediction, dtype=np.float32)
    target_array = np.asarray(target, dtype=np.float32)
    if pred_array.shape != target_array.shape:
        raise ValueError(
            f"Metric shape mismatch: prediction={pred_array.shape}, target={target_array.shape}"
        )
    if pred_array.ndim != 2:
        raise ValueError(f"AS-Depth metrics expect 2D depth maps, got shape {pred_array.shape}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pred = torch.from_numpy(pred_array).unsqueeze(0).to(device)
    gt = torch.from_numpy(target_array).unsqueeze(0).to(device)
    valid = torch.isfinite(pred) & torch.isfinite(gt) & (gt > 0.0)
    n = valid.sum(dim=(-1, -2))
    if not bool(torch.all(n > 0)):
        return {name: float("nan") for name in METRIC_NAMES}

    denominator = n.to(dtype=torch.float32)
    difference = pred - gt

    abs_difference = torch.abs(difference)
    abs_difference[~valid] = 0.0
    mae = torch.sum(abs_difference, dim=(-1, -2)) / denominator

    squared_difference = torch.square(difference)
    squared_difference[~valid] = 0.0
    rmse = torch.sqrt(torch.sum(squared_difference, dim=(-1, -2)) / denominator)

    abs_relative = torch.abs(difference) / gt
    abs_relative[~valid] = 0.0
    abs_rel = torch.sum(abs_relative, dim=(-1, -2)) / denominator

    ratio = torch.maximum(pred / gt, gt / pred)

    def delta(threshold: float) -> torch.Tensor:
        threshold_values = (ratio < threshold).to(dtype=torch.float32)
        threshold_values[~valid] = 0.0
        return torch.sum(threshold_values, dim=(-1, -2)) / denominator

    values = {
        "mae": mae,
        "rmse": rmse,
        "abs_rel": abs_rel,
        "delta_1_05": delta(1.05),
        "delta_1_10": delta(1.10),
        "delta_1_25": delta(1.25),
    }
    return {name: float(value.mean().item()) for name, value in values.items()}

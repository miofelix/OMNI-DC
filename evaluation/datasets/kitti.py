from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from evaluation.core.types import EvaluationSample
from evaluation.datasets.base import (
    DatasetCollection,
    limit_per_subset,
    normalize_sample_id,
    read_jsonl,
    require_keys,
    resolve_path,
)

KITTI_DEPTH_SCALE = 256.0
KITTI_MIN_DEPTH = 1.0 / KITTI_DEPTH_SCALE
KITTI_MAX_EVAL_DEPTH = float("inf")
KITTI_DEFAULT_RAW_MAX_DEPTH = 80.0
KITTI_BENCHMARK_NAME = "KITTI Depth Completion val_selection_cropped"

_KITTI_SAMPLE_RE = re.compile(
    r"^(?P<date>\d{4}_\d{2}_\d{2})_drive_(?P<drive>\d{4})_sync_image_"
    r"(?P<frame>\d{10})_(?P<camera>image_0[23])(?:\.png)?$"
)


def _scene_frame(value: str, rgb_path: Path) -> Tuple[str, str]:
    """Return the stable scene/frame names used by KITTI visualizations."""

    candidates = [str(value), rgb_path.name, rgb_path.stem]
    for candidate in candidates:
        if "-" in candidate and not _KITTI_SAMPLE_RE.fullmatch(candidate):
            scene, frame = candidate.split("-", 1)
            if scene and frame:
                return scene, frame
        match = _KITTI_SAMPLE_RE.fullmatch(candidate)
        if match is not None:
            return (
                f"{match.group('date')}_drive_{match.group('drive')}",
                f"{match.group('frame')}_{match.group('camera')}",
            )
    raise ValueError(
        "Unable to resolve KITTI scene/frame from "
        f"name={value!r}, rgb={rgb_path}"
    )


def _raw_path(row: Dict[str, Any], root: Path, context: str) -> Path:
    for key in ("lidar", "velodyne_raw", "raw_depth"):
        if row.get(key):
            return resolve_path(root, row[key])
    raise ValueError(f"{context} must define one of lidar, velodyne_raw, raw_depth")


def load_kitti(
    manifest: Path,
    max_samples: Optional[int] = None,
    raw_max_depth: float = KITTI_DEFAULT_RAW_MAX_DEPTH,
) -> DatasetCollection:
    """Load KITTI ``val_selection_cropped`` rows with devkit semantics.

    KITTI depth PNG values are uint16 values divided by 256.  Zero is invalid,
    and the benchmark GT has no upper depth cutoff.  The raw Velodyne input is
    clipped separately (80 m by default), matching AS-Depth's KITTI pipeline.
    """

    manifest = manifest.expanduser().resolve()
    if raw_max_depth <= 0:
        raise ValueError("KITTI raw maximum depth must be greater than zero")

    root = manifest.parent
    samples = []
    for index, row in enumerate(read_jsonl(manifest), start=1):
        context = f"{manifest}:{index}"
        require_keys(row, ("rgb", "depth"), context)
        rgb_path = resolve_path(root, row["rgb"])
        name = str(row.get("name") or row.get("sample_name") or rgb_path.stem)
        sample_id = normalize_sample_id(name)
        scene, frame = _scene_frame(name, rgb_path)
        intrinsics = row.get("intrinsics")
        intrinsics_path = resolve_path(root, intrinsics) if intrinsics else None
        samples.append(
            EvaluationSample(
                sample_id=sample_id,
                subset="default",
                rgb_path=rgb_path,
                raw_depth_path=_raw_path(row, root, context),
                gt_depth_path=resolve_path(root, row["depth"]),
                depth_scale=KITTI_DEPTH_SCALE,
                min_depth=KITTI_MIN_DEPTH,
                max_depth=KITTI_MAX_EVAL_DEPTH,
                raw_max_depth=float(raw_max_depth),
                metadata={
                    "benchmark": KITTI_BENCHMARK_NAME,
                    "scene": scene,
                    "frame": frame,
                    "intrinsics_path": str(intrinsics_path) if intrinsics_path else None,
                    "source_name": name,
                },
            )
        )

    return DatasetCollection(
        name="kitti",
        samples=limit_per_subset(samples, max_samples),
        metadata={
            "manifest": str(manifest),
            "benchmark": KITTI_BENCHMARK_NAME,
            "depth_scale": KITTI_DEPTH_SCALE,
            "gt_min_depth": KITTI_MIN_DEPTH,
            "gt_max_depth": None,
            "gt_zero_is_invalid": True,
            "raw_max_depth": float(raw_max_depth),
        },
    )


__all__ = [
    "KITTI_BENCHMARK_NAME",
    "KITTI_DEFAULT_RAW_MAX_DEPTH",
    "KITTI_DEPTH_SCALE",
    "KITTI_MAX_EVAL_DEPTH",
    "KITTI_MIN_DEPTH",
    "load_kitti",
]

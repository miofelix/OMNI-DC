from __future__ import annotations

from pathlib import Path
from typing import Optional

from evaluation.core.types import EvaluationSample
from evaluation.datasets.base import (
    DatasetCollection,
    limit_per_subset,
    parse_depth_range,
    read_jsonl,
    require_keys,
    resolve_path,
    sample_id_from_path,
)

CAMERAS = ("d435", "l515", "tof")


def load_hammer(
    manifest: Path, camera: str, max_samples: Optional[int] = None
) -> DatasetCollection:
    manifest = manifest.expanduser().resolve()
    camera = camera.lower()
    if camera not in CAMERAS:
        raise ValueError(f"Unsupported HAMMER camera: {camera}")

    root = manifest.parent
    raw_key = f"{camera}_depth"
    samples = []
    for index, row in enumerate(read_jsonl(manifest), start=1):
        context = f"{manifest}:{index}"
        require_keys(row, ("rgb", "depth", raw_key, "depth-range"), context)
        min_depth, max_depth = parse_depth_range(row, context)
        rgb_path = resolve_path(root, row["rgb"])
        samples.append(
            EvaluationSample(
                sample_id=sample_id_from_path(rgb_path, root),
                subset=camera,
                rgb_path=rgb_path,
                raw_depth_path=resolve_path(root, row[raw_key]),
                gt_depth_path=resolve_path(root, row["depth"]),
                depth_scale=1000.0,
                min_depth=min_depth,
                max_depth=max_depth,
            )
        )

    return DatasetCollection(
        name="hammer",
        samples=limit_per_subset(samples, max_samples),
        metadata={"manifest": str(manifest), "camera": camera},
    )

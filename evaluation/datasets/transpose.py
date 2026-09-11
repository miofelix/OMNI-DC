from __future__ import annotations

from pathlib import Path
from typing import Optional

from evaluation.core.types import EvaluationSample
from evaluation.datasets.base import (
    DatasetCollection,
    limit_per_subset,
    normalize_sample_id,
    parse_depth_range,
    read_jsonl,
    require_keys,
    resolve_path,
)

TRANSPOSE_DEFAULT_DEPTH_RANGE = (0.1, 6.0)


def load_transpose(manifest: Path, max_samples: Optional[int] = None) -> DatasetCollection:
    """Load the current project's per-sample TRansPose L515 manifest."""

    manifest = manifest.expanduser().resolve()
    rows = read_jsonl(manifest)
    samples = []
    for index, row in enumerate(rows, start=1):
        context = f"{manifest}:{index}"
        require_keys(row, ("rgb", "l515_depth", "depth"), context)
        if "depth-range" in row:
            min_depth, max_depth = parse_depth_range(row, context)
        else:
            min_depth, max_depth = TRANSPOSE_DEFAULT_DEPTH_RANGE
        rgb_path = resolve_path(manifest.parent, row["rgb"])
        if row.get("seq_name"):
            sample_id = normalize_sample_id(str(row["seq_name"]))
        else:
            parts = rgb_path.parts
            fallback = f"{parts[-4]}_{rgb_path.stem}" if len(parts) >= 4 else rgb_path.stem
            sample_id = normalize_sample_id(fallback)
        samples.append(
            EvaluationSample(
                sample_id=sample_id,
                subset="default",
                rgb_path=rgb_path,
                raw_depth_path=resolve_path(manifest.parent, row["l515_depth"]),
                gt_depth_path=resolve_path(manifest.parent, row["depth"]),
                depth_scale=1000.0,
                min_depth=min_depth,
                max_depth=max_depth,
                metadata={"seq_name": row.get("seq_name")},
            )
        )

    return DatasetCollection(
        name="transpose",
        samples=limit_per_subset(samples, max_samples),
        metadata={
            "manifest": str(manifest),
            "raw_type": "l515",
            "depth_scale": 1000.0,
            "default_depth_range": list(TRANSPOSE_DEFAULT_DEPTH_RANGE),
        },
    )

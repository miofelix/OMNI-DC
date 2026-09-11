from __future__ import annotations

from pathlib import Path
from typing import Optional

from evaluation.core.types import EvaluationSample
from evaluation.datasets.base import (
    DatasetCollection,
    limit_per_subset,
    match_sequence_files,
    parse_depth_range,
    read_jsonl,
    require_keys,
    resolve_path,
    sample_id_from_path,
)


def load_clearpose(manifest: Path, max_samples: Optional[int] = None) -> DatasetCollection:
    manifest = manifest.expanduser().resolve()
    root = manifest.parent
    samples = []

    for index, row in enumerate(read_jsonl(manifest), start=1):
        context = f"{manifest}:{index}"
        require_keys(
            row,
            ("rgb", "rgb-suffix", "raw_depth-suffix", "depth-suffix", "depth-range"),
            context,
        )
        min_depth, max_depth = parse_depth_range(row, context)
        sequence_dir = resolve_path(root, row["rgb"])
        for rgb_path, raw_path, gt_path in match_sequence_files(
            sequence_dir,
            str(row["rgb-suffix"]),
            str(row["raw_depth-suffix"]),
            str(row["depth-suffix"]),
            context,
        ):
            samples.append(
                EvaluationSample(
                    sample_id=sample_id_from_path(rgb_path, root),
                    subset="default",
                    rgb_path=rgb_path,
                    raw_depth_path=raw_path,
                    gt_depth_path=gt_path,
                    depth_scale=1000.0,
                    min_depth=min_depth,
                    max_depth=max_depth,
                )
            )

    return DatasetCollection(
        name="clearpose",
        samples=limit_per_subset(samples, max_samples),
        metadata={"manifest": str(manifest)},
    )

from __future__ import annotations

from pathlib import Path
from typing import Mapping, Optional, Sequence

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

DREDS_VARIANTS = ("catknown", "catnovel")


def load_dreds(
    manifests: Mapping[str, Path],
    variants: Sequence[str],
    max_samples: Optional[int] = None,
) -> DatasetCollection:
    samples = []
    manifest_metadata = {}

    for variant in variants:
        if variant not in DREDS_VARIANTS:
            raise ValueError(f"Unsupported DREDS variant: {variant}")
        if variant not in manifests:
            raise ValueError(f"Missing manifest for DREDS variant: {variant}")

        manifest = manifests[variant].expanduser().resolve()
        root = manifest.parent
        manifest_metadata[variant] = str(manifest)
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
                        subset=variant,
                        rgb_path=rgb_path,
                        raw_depth_path=raw_path,
                        gt_depth_path=gt_path,
                        depth_scale=1.0,
                        min_depth=min_depth,
                        max_depth=max_depth,
                        allow_evaluation_resize=True,
                    )
                )

    return DatasetCollection(
        name="dreds",
        samples=limit_per_subset(samples, max_samples),
        metadata={"manifests": manifest_metadata, "variants": list(variants)},
    )

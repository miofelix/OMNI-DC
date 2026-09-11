from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

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

IBIMS_LEVELS = ("easy", "medium", "hard", "extreme")
IBIMS_DEPTH_MAX_M = 50.0
IBIMS_DEPTH_SCALE = 65535.0 / IBIMS_DEPTH_MAX_M
IBIMS_EXPECTED_SHAPE = (480, 640)
SYNTHETIC_RAW_DIR_NAME = "ibims1_synthetic_raw_depth"


def manifest_for_level(ibims_root: Path, level: str) -> Path:
    return ibims_root / SYNTHETIC_RAW_DIR_NAME / "manifests" / f"ibims_{level}.jsonl"


def load_ibims(
    ibims_root: Path,
    levels: Sequence[str],
    max_samples: Optional[int] = None,
    depth_scale_override: Optional[float] = None,
    max_depth_override: Optional[float] = None,
) -> DatasetCollection:
    ibims_root = ibims_root.expanduser().resolve()
    samples = []
    manifests = {}

    for level in levels:
        if level not in IBIMS_LEVELS:
            raise ValueError(f"Unsupported iBims level: {level}")
        manifest = manifest_for_level(ibims_root, level)
        manifests[level] = str(manifest)
        for index, row in enumerate(read_jsonl(manifest), start=1):
            context = f"{manifest}:{index}"
            require_keys(row, ("sample_id", "rgb", "raw_depth"), context)
            if row.get("dataset", "ibims") != "ibims":
                raise ValueError(f"Expected dataset=ibims in {context}")

            if "depth-range" in row:
                min_depth, max_depth = parse_depth_range(row, context)
            else:
                min_depth, max_depth = 0.01, IBIMS_DEPTH_MAX_M
            if max_depth_override is not None:
                max_depth = max_depth_override
            if max_depth <= min_depth:
                raise ValueError(f"Invalid effective depth range in {context}")

            depth_scale = (
                depth_scale_override
                if depth_scale_override is not None
                else float(row.get("depth_scale", IBIMS_DEPTH_SCALE))
            )
            if depth_scale <= 0:
                raise ValueError(f"Invalid depth scale in {context}: {depth_scale}")

            sample_id = normalize_sample_id(str(row["sample_id"]))
            samples.append(
                EvaluationSample(
                    sample_id=sample_id,
                    subset=level,
                    rgb_path=resolve_path(manifest.parent, row["rgb"]),
                    raw_depth_path=resolve_path(manifest.parent, row["raw_depth"]),
                    gt_depth_path=None,
                    depth_scale=depth_scale,
                    min_depth=min_depth,
                    max_depth=max_depth,
                    expected_shape=IBIMS_EXPECTED_SHAPE,
                    metadata={
                        "difficulty": level,
                        "intrinsics_source": "ibims_calibration",
                        "intrinsics_path": str(
                            ibims_root / "ibims1_core_raw" / "calib" / f"{sample_id}.txt"
                        ),
                    },
                )
            )

    return DatasetCollection(
        name="ibims",
        samples=limit_per_subset(samples, max_samples),
        metadata={"ibims_root": str(ibims_root), "manifests": manifests},
    )

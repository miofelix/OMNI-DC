"""Model-independent benchmark identity shared by all evaluation adapters."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

from evaluation.core.types import EvaluationSample
from evaluation.datasets.base import DatasetCollection

PROTOCOL_VERSION = 1


def canonical_value(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value.expanduser().resolve())
    if isinstance(value, float) and not math.isfinite(value):
        # KITTI has an unbounded GT range. Keep the signature valid JSON.
        if math.isnan(value):
            raise ValueError("NaN is not a valid benchmark configuration value")
        return "Infinity" if value > 0 else "-Infinity"
    if isinstance(value, Mapping):
        return {str(key): canonical_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [canonical_value(item) for item in value]
    return value


def signature(value: Any) -> str:
    payload = json.dumps(
        canonical_value(value), sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def selected_sample(sample: EvaluationSample) -> Dict[str, Any]:
    # Intrinsics/visualization metadata and model settings do not define scoring.
    # Paths resolve symlinks so repositories sharing a dataset get the same ID.
    return canonical_value({
        "sample_id": sample.sample_id,
        "subset": sample.subset,
        "rgb_path": sample.rgb_path,
        "raw_depth_path": sample.raw_depth_path,
        "gt_depth_path": sample.gt_depth_path,
        "depth_scale": float(sample.depth_scale),
        "min_depth": float(sample.min_depth),
        "max_depth": float(sample.max_depth),
        "raw_max_depth": (
            float(sample.raw_max_depth) if sample.raw_max_depth is not None else None
        ),
        "allow_evaluation_resize": sample.allow_evaluation_resize,
        "expected_shape": sample.expected_shape,
    })


def manifest_hashes(collection: DatasetCollection) -> Dict[str, str]:
    paths = list(collection.metadata.get("manifests", {}).values())
    if collection.metadata.get("manifest"):
        paths.append(collection.metadata["manifest"])
    resolved = {Path(path).expanduser().resolve() for path in paths}
    return {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in sorted(resolved)}


def official_script_hashes(ibims_root: Path) -> Dict[str, str]:
    return {
        path.relative_to(ibims_root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted((ibims_root / "evaluation_scripts").rglob("*.py"))
    }


def benchmark_metadata(
    collection: DatasetCollection,
    ibims_root: Optional[Path] = None,
    evaluation_seed: int = 0,
) -> Dict[str, Any]:
    is_kitti = collection.name == "kitti"
    samples = [selected_sample(sample) for sample in collection.samples]
    protocol = {
        "profile": "asdepth_kitti_legacy_v1" if is_kitti else "positive_depth_v1",
        "depth_unit": "meter",
        "sampling": "manifest order; sorted sequence frames; max-samples per subset",
        "depth_bounds": "inclusive per sample; raw max independent of GT max",
        "raw_invalid": "nonfinite or out of range -> zero",
        "gt_invalid": "nonfinite or out of range -> NaN; scoring requires positive GT",
        "prediction_format": "float32 metric depth; spatial grid restored by model adapter",
        "model_adaptation": "native input and output processing recorded with inference",
        "scoring_mask": (
            "finite prediction and finite positive GT"
            if is_kitti else "finite positive prediction and finite positive GT"
        ),
        "error_precision": "float32 Torch" if is_kitti else "float64 NumPy",
        "delta": "float32 ratios; strict < 1.05, 1.10, 1.25",
        "aggregation": "arithmetic mean of finite per-image scores; overall pools images",
        "evaluation_alignment": "no scale/shift; nearest spatial resize only if sample allows",
    }
    if collection.name == "ibims":
        protocol.update({
            "profile": "ibims_official_v1",
            "scoring_mask": "official MAT masks",
            "error_precision": "official evaluator",
            "delta": "official evaluator",
            "aggregation": "official evaluator per difficulty level",
        })
    value: Dict[str, Any] = {
        "version": PROTOCOL_VERSION,
        "dataset": collection.name,
        "selected_samples": samples,
        "selection_sha256": signature({"dataset": collection.name, "samples": samples}),
        "manifest_sha256": manifest_hashes(collection),
        "protocol": protocol,
    }
    if collection.name == "ibims":
        if ibims_root is None:
            root = collection.metadata.get("ibims_root")
            ibims_root = Path(root) if root else None
        if ibims_root is None:
            raise ValueError("iBims root is required to identify the official benchmark")
        ibims_root = ibims_root.expanduser().resolve()
        value["official_evaluator"] = {
            "seed": evaluation_seed,
            "gt_root": str(ibims_root / "ibims1_core_mat"),
            "scripts_sha256": official_script_hashes(ibims_root),
        }
    # Full manifests also contain model-only fields (for example intrinsics).
    # Retain their hashes as provenance, while identifying scoring by the parsed
    # selection and protocol so valid model adaptations remain comparable.
    value["sha256"] = signature({key: item for key, item in value.items()
                                 if key != "manifest_sha256"})
    return value


def validate_benchmark(recorded: Mapping[str, Any], current: Mapping[str, Any]) -> None:
    if not recorded:
        raise ValueError(
            "Run has no shared benchmark identity; rerun inference in a new directory"
        )
    if recorded.get("sha256") != current.get("sha256"):
        changed = [
            key for key in current
            if key not in ("sha256", "manifest_sha256") and recorded.get(key) != current[key]
        ]
        raise ValueError(
            "Selected samples or depth protocol differ from this run's inference "
            "(benchmark mismatch: " + ", ".join(changed) + ")"
        )

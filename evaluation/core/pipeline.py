from __future__ import annotations

import shutil
from collections import Counter
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from evaluation.core.inference import run_inference
from evaluation.core.output import (
    SCHEMA_VERSION,
    RunLayout,
    read_run_metadata,
    write_json,
)
from evaluation.core.types import RunConfig
from evaluation.datasets.base import DatasetCollection
from evaluation.evaluators.depth import run_depth_evaluation
from evaluation.evaluators.ibims_official import run_ibims_official_evaluation

VALID_STAGES = ("all", "infer", "evaluate")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def serialized_config(config: RunConfig) -> Dict[str, Any]:
    value = asdict(config)
    value["run_dir"] = str(config.run_dir)
    if config.intrinsics_path is not None:
        value["intrinsics_path"] = str(config.intrinsics_path)
    return value


def initial_metadata(collection: DatasetCollection, config: RunConfig) -> Dict[str, Any]:
    subset_counts = Counter(sample.subset for sample in collection.samples)
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "running",
        "dataset": collection.name,
        "subsets": collection.subsets,
        "num_samples": len(collection.samples),
        "subset_counts": dict(sorted(subset_counts.items())),
        "depth_unit": "meter",
        "stage": config.stage,
        "model_path": config.model_path,
        "started_at": utc_now(),
        "finished_at": None,
        "config": serialized_config(config),
        "dataset_metadata": collection.metadata,
        "results": {},
    }


def validate_collection(collection: DatasetCollection) -> None:
    if not collection.samples:
        raise ValueError(f"No samples were selected for {collection.name}")
    seen = set()
    for sample in collection.samples:
        key = (sample.subset, sample.sample_id)
        if key in seen:
            raise ValueError(f"Duplicate sample id in subset {sample.subset}: {sample.sample_id}")
        seen.add(key)


def run_pipeline(
    collection: DatasetCollection,
    config: RunConfig,
    ibims_root: Optional[Path] = None,
) -> RunLayout:
    if config.stage not in VALID_STAGES:
        raise ValueError(f"Unsupported stage: {config.stage}")
    validate_collection(collection)

    layout = RunLayout(config.run_dir.expanduser().resolve())
    if config.stage in ("all", "infer"):
        if layout.root.exists() and any(layout.root.iterdir()):
            raise FileExistsError(
                f"Run directory is not empty; choose a new --run-dir: {layout.root}"
            )
        layout.root.mkdir(parents=True, exist_ok=True)
        metadata = initial_metadata(collection, config)
    else:
        if not layout.root.is_dir():
            raise FileNotFoundError(f"Run directory not found: {layout.root}")
        metadata = read_run_metadata(layout)
        if metadata.get("dataset") != collection.name:
            raise ValueError(
                f"Run dataset is {metadata.get('dataset')!r}, but {collection.name!r} was requested"
            )
        metadata["status"] = "running"
        metadata["stage"] = config.stage
        metadata["config"] = serialized_config(config)
        metadata["finished_at"] = None

    write_json(layout.metadata_path, metadata)
    try:
        if config.stage in ("all", "infer"):
            metadata["results"]["inference"] = run_inference(collection, config, layout)
            write_json(layout.metadata_path, metadata)

        if config.stage in ("all", "evaluate"):
            if collection.name == "ibims":
                if ibims_root is None:
                    raise ValueError("iBims root is required for official evaluation")
                evaluation_result = run_ibims_official_evaluation(
                    collection,
                    layout,
                    ibims_root.expanduser().resolve(),
                )
            else:
                evaluation_result = run_depth_evaluation(collection, layout)
            metadata["results"]["evaluation"] = evaluation_result

            if config.cleanup_predictions and layout.predictions_dir.is_dir():
                shutil.rmtree(layout.predictions_dir)
                metadata["results"]["predictions_cleaned"] = True

        metadata["status"] = "completed"
        metadata["finished_at"] = utc_now()
        write_json(layout.metadata_path, metadata)
        return layout
    except Exception as exc:
        metadata["status"] = "failed"
        metadata["finished_at"] = utc_now()
        metadata["error"] = {"type": type(exc).__name__, "message": str(exc)}
        write_json(layout.metadata_path, metadata)
        raise

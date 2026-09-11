from __future__ import annotations

import csv
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Dict, Iterable, Mapping, Sequence

import numpy as np

from evaluation.core.types import EvaluationSample

SCHEMA_VERSION = 1


def sanitize_component(value: str) -> str:
    sanitized = re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("._-")
    return sanitized or "run"


def model_stem(model_path: str) -> str:
    return sanitize_component(Path(model_path.rstrip("/")).name or model_path)


def sample_relative_path(sample: EvaluationSample, suffix: str) -> Path:
    pure = PurePosixPath(sample.sample_id)
    if pure.is_absolute() or ".." in pure.parts or not pure.parts:
        raise ValueError(f"Unsafe sample id: {sample.sample_id!r}")
    relative = Path(*pure.parts)
    filename = f"{relative.name}{suffix}"
    return Path(sample.subset) / relative.parent / filename


@dataclass(frozen=True)
class RunLayout:
    root: Path

    @property
    def metadata_path(self) -> Path:
        return self.root / "run.json"

    @property
    def predictions_dir(self) -> Path:
        return self.root / "predictions"

    @property
    def visualizations_dir(self) -> Path:
        return self.root / "visualizations"

    @property
    def metrics_dir(self) -> Path:
        return self.root / "metrics"

    @property
    def official_dir(self) -> Path:
        return self.root / "official"

    def prediction_path(self, sample: EvaluationSample) -> Path:
        return self.predictions_dir / sample_relative_path(sample, ".npy")

    def visualization_path(self, sample: EvaluationSample) -> Path:
        return self.visualizations_dir / sample_relative_path(sample, "_vis.jpg")

    def kitti_prediction_visualization_path(self, sample: EvaluationSample) -> Path:
        scene, frame = kitti_scene_frame(sample)
        return self.root / "kitti_visualization" / "predictions" / scene / f"{frame}.jpg"

    def kitti_pointcloud_visualization_path(self, sample: EvaluationSample) -> Path:
        scene, frame = kitti_scene_frame(sample)
        return self.root / "kitti_visualization" / "pointclouds" / scene / f"{frame}.jpg"

    def official_prediction_dir(self, subset: str) -> Path:
        return self.official_dir / subset / "predictions"

    def official_workspace(self, subset: str) -> Path:
        return self.official_dir / subset / "workspace"

    def official_log_path(self, subset: str) -> Path:
        return self.official_dir / subset / "evaluator.log"


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as file:
        json.dump(
            value,
            file,
            indent=2,
            ensure_ascii=False,
            sort_keys=True,
            default=json_default,
        )
        file.write("\n")
    temporary.replace(path)


def json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fieldnames: Iterable[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    names = list(fieldnames)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=names)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name) for name in names})


def save_prediction(path: Path, prediction: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, np.asarray(prediction, dtype=np.float32), allow_pickle=False)


def kitti_scene_frame(sample: EvaluationSample) -> tuple[str, str]:
    scene = sanitize_component(str(sample.metadata.get("scene", "")))
    frame = sanitize_component(str(sample.metadata.get("frame", "")))
    if not scene or scene == "run" or not frame or frame == "run":
        raise ValueError(f"KITTI scene/frame metadata is missing for {sample.sample_id}")
    return scene, frame


def read_run_metadata(layout: RunLayout) -> Dict[str, Any]:
    if not layout.metadata_path.is_file():
        raise FileNotFoundError(f"Run metadata not found: {layout.metadata_path}")
    with layout.metadata_path.open("r", encoding="utf-8") as file:
        value = json.load(file)
    if value.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported run schema in {layout.metadata_path}: {value.get('schema_version')!r}"
        )
    return value

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from evaluation.core.types import EvaluationSample


@dataclass(frozen=True)
class DatasetCollection:
    name: str
    samples: Sequence[EvaluationSample]
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def subsets(self) -> List[str]:
        return sorted({sample.subset for sample in self.samples})


def resolve_path(base: Path, value: Any) -> Path:
    path = Path(str(value)).expanduser()
    if path.is_absolute():
        return path.resolve()
    return (base / path).resolve()


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"Manifest not found: {path}")

    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_number}: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"Expected a JSON object at {path}:{line_number}")
            rows.append(row)

    if not rows:
        raise ValueError(f"Manifest is empty: {path}")
    return rows


def require_keys(row: Mapping[str, Any], keys: Iterable[str], context: str) -> None:
    missing = [key for key in keys if key not in row]
    if missing:
        raise ValueError(f"{context} is missing required fields: {', '.join(missing)}")


def parse_depth_range(row: Mapping[str, Any], context: str) -> Tuple[float, float]:
    value = row.get("depth-range")
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError(f"{context} must define depth-range as [min, max]")
    min_depth, max_depth = float(value[0]), float(value[1])
    if min_depth < 0 or max_depth <= min_depth:
        raise ValueError(f"Invalid depth-range in {context}: {value}")
    return min_depth, max_depth


def normalize_sample_id(value: str) -> str:
    normalized = str(PurePosixPath(value.replace("\\", "/")))
    path = PurePosixPath(normalized)
    if path.is_absolute() or not normalized or normalized == "." or ".." in path.parts:
        raise ValueError(f"Unsafe sample id: {value!r}")
    return normalized


def sample_id_from_path(path: Path, root: Path) -> str:
    try:
        relative = path.resolve().relative_to(root.resolve())
    except ValueError:
        parts = path.parts[-3:]
        relative = Path(*parts)
    return normalize_sample_id(relative.with_suffix("").as_posix())


def match_sequence_files(
    sequence_dir: Path,
    rgb_suffix: str,
    raw_suffix: str,
    gt_suffix: str,
    context: str,
) -> List[Tuple[Path, Path, Path]]:
    if not sequence_dir.is_dir():
        raise FileNotFoundError(f"Sequence directory not found for {context}: {sequence_dir}")

    def indexed(suffix: str) -> Dict[str, Path]:
        paths = sorted(sequence_dir.glob(f"*{suffix}"))
        result: Dict[str, Path] = {}
        for path in paths:
            name = path.name
            key = name[: -len(suffix)] if suffix and name.endswith(suffix) else path.stem
            if key in result:
                raise ValueError(f"Duplicate frame key {key!r} in {sequence_dir}")
            result[key] = path.resolve()
        return result

    rgb_files = indexed(rgb_suffix)
    raw_files = indexed(raw_suffix)
    gt_files = indexed(gt_suffix)
    if not rgb_files:
        raise ValueError(f"No RGB frames matched for {context}: {sequence_dir}/*{rgb_suffix}")

    rgb_keys, raw_keys, gt_keys = set(rgb_files), set(raw_files), set(gt_files)
    if rgb_keys != raw_keys or rgb_keys != gt_keys:
        missing_raw = sorted(rgb_keys - raw_keys)
        missing_gt = sorted(rgb_keys - gt_keys)
        extra_raw = sorted(raw_keys - rgb_keys)
        extra_gt = sorted(gt_keys - rgb_keys)
        raise ValueError(
            f"Frame mismatch for {context}: missing_raw={missing_raw[:5]}, "
            f"missing_gt={missing_gt[:5]}, extra_raw={extra_raw[:5]}, "
            f"extra_gt={extra_gt[:5]}"
        )

    return [(rgb_files[key], raw_files[key], gt_files[key]) for key in sorted(rgb_keys)]


def limit_per_subset(
    samples: Sequence[EvaluationSample], max_samples: Optional[int]
) -> List[EvaluationSample]:
    if max_samples is None:
        return list(samples)
    if max_samples < 1:
        raise ValueError("--max-samples must be greater than zero")

    counts: Dict[str, int] = {}
    limited: List[EvaluationSample] = []
    for sample in samples:
        count = counts.get(sample.subset, 0)
        if count >= max_samples:
            continue
        limited.append(sample)
        counts[sample.subset] = count + 1
    return limited

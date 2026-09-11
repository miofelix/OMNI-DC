from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, List, Mapping, Sequence

import numpy as np

from evaluation.core.io import (
    align_prediction_for_evaluation,
    load_prediction,
    read_gt_depth,
)
from evaluation.core.metrics import (
    METRIC_NAMES,
    compute_asdepth_depth_metrics,
    compute_depth_metrics,
)
from evaluation.core.output import RunLayout, write_csv, write_json
from evaluation.datasets.base import DatasetCollection

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover

    def tqdm(iterable, **_kwargs):
        return iterable


def finite_mean(values: Sequence[float]) -> Any:
    finite = [value for value in values if np.isfinite(value)]
    if not finite:
        return None
    return float(np.mean(finite))


def summarize_records(records: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    by_subset: Dict[str, List[Mapping[str, Any]]] = defaultdict(list)
    for record in records:
        by_subset[str(record["subset"])].append(record)

    def summarize(group: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
        result: Dict[str, Any] = {"num_samples": len(group)}
        for metric in METRIC_NAMES:
            result[metric] = finite_mean([float(row[metric]) for row in group])
        return result

    return {
        "metric_names": list(METRIC_NAMES),
        "subsets": {name: summarize(rows) for name, rows in sorted(by_subset.items())},
        "overall": summarize(records),
    }


def run_depth_evaluation(collection: DatasetCollection, layout: RunLayout) -> Dict[str, Any]:
    records: List[Dict[str, Any]] = []
    for sample in tqdm(collection.samples, desc=f"{collection.name} evaluation"):
        if sample.gt_depth_path is None:
            raise ValueError(f"Ground truth is not defined for {sample.sample_id}")

        target = read_gt_depth(
            sample.gt_depth_path,
            sample.depth_scale,
            sample.min_depth,
            sample.max_depth,
        )
        prediction = load_prediction(layout.prediction_path(sample))
        prediction = align_prediction_for_evaluation(
            prediction,
            target.shape,
            sample.allow_evaluation_resize,
            sample.sample_id,
        )
        if collection.name == "kitti":
            metrics = compute_asdepth_depth_metrics(prediction, target)
            valid_mask = np.isfinite(prediction) & np.isfinite(target) & (target > 0.0)
        else:
            metrics = compute_depth_metrics(prediction, target)
            valid_mask = (
                np.isfinite(prediction)
                & np.isfinite(target)
                & (prediction > 0.0)
                & (target > 0.0)
            )
        valid_pixels = int(np.count_nonzero(valid_mask))
        records.append(
            {
                "subset": sample.subset,
                "sample_id": sample.sample_id,
                "valid_pixels": valid_pixels,
                **metrics,
            }
        )

    summary = summarize_records(records)
    per_sample_fields = ["subset", "sample_id", "valid_pixels", *METRIC_NAMES]
    write_csv(layout.metrics_dir / "per_sample.csv", records, per_sample_fields)

    summary_rows = []
    for subset, values in summary["subsets"].items():
        summary_rows.append({"subset": subset, **values})
    summary_rows.append({"subset": "overall", **summary["overall"]})
    write_csv(
        layout.metrics_dir / "summary.csv",
        summary_rows,
        ["subset", "num_samples", *METRIC_NAMES],
    )
    write_json(layout.metrics_dir / "summary.json", summary)
    return {
        "num_evaluated": len(records),
        "summary": summary,
    }

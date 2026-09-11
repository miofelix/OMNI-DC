"""Check common evaluation code and numerical behavior across local repositories.

Model inference, CLI model options, and extra datasets are intentionally outside
this check: each project's adapter owns its native input/output processing.
"""
from __future__ import annotations

import argparse
import ast
import json
import math
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Sequence

REPOSITORIES = ("lingbot-depth", "InfiniDepth", "Prior-Depth-Anything", "PromptDA", "OMNI-DC")
COMMON_FILES = (
    "core/io.py", "core/metrics.py", "core/output.py", "core/pointcloud.py",
    "core/protocol.py", "datasets/base.py", "datasets/hammer.py", "datasets/clearpose.py",
    "datasets/dreds.py", "datasets/ibims.py", "datasets/kitti.py", "evaluators/depth.py",
    "evaluators/ibims_official.py", "prepare_kitti_jsonl.py", "compare_runs.py",
)


def worker(project: Path) -> dict[str, Any]:
    # Each worker is a new interpreter to prevent evaluation-package import collisions.
    sys.path.insert(0, str(project))
    import numpy as np
    from PIL import Image

    from evaluation.core.io import align_prediction_for_evaluation, read_gt_depth, read_raw_depth
    from evaluation.core.metrics import compute_asdepth_depth_metrics, compute_depth_metrics
    from evaluation.core.protocol import benchmark_metadata
    from evaluation.core.types import EvaluationSample
    from evaluation.datasets.base import DatasetCollection
    from evaluation.evaluators.depth import summarize_records

    pred = np.array([[0, -1, 1.05, np.nan], [1.1, 1.25, 2, np.inf]], dtype=np.float32)
    gt = np.ones(pred.shape, dtype=np.float32)
    gt[1, 2] = np.nan
    with TemporaryDirectory() as temporary:
        depth_path = Path(temporary) / "depth.png"
        Image.fromarray(np.array([[0, 1, 256, 20480, 23040]], dtype=np.uint16)).save(depth_path)
        raw = read_raw_depth(depth_path, 256, 1 / 256, 80)
        target = read_gt_depth(depth_path, 256, 1 / 256, float("inf"))
    resized = align_prediction_for_evaluation(
        np.array([[1, np.nan], [2, 3]], dtype=np.float32), (4, 4), True, "fixture"
    )
    sample = EvaluationSample(
        "scene/frame", "default", Path("/benchmark/rgb.png"), Path("/benchmark/raw.png"),
        Path("/benchmark/gt.png"), 256, 1 / 256, float("inf"), raw_max_depth=80,
    )
    shared = compute_depth_metrics(pred, gt)
    records = [
        {"subset": "default", **shared},
        {"subset": "default", **dict.fromkeys(shared, 0.0)},
        {"subset": "other", **dict.fromkeys(shared, 1.0)},
    ]
    summary = summarize_records(records)
    kitti_metrics = compute_asdepth_depth_metrics(pred, gt)
    # Independent analytical checks: agreement with the reference alone does
    # not establish that either implementation follows its declared profile.
    expected_common = {
        "mae": 0.4 / 3, "rmse": math.sqrt(0.075 / 3), "abs_rel": 0.4 / 3,
        "delta_1_05": 0.0, "delta_1_10": 1 / 3, "delta_1_25": 2 / 3,
    }
    expected_kitti = {
        "mae": 3.4 / 5, "rmse": math.sqrt(5.075 / 5), "abs_rel": 3.4 / 5,
        "delta_1_05": 1 / 5, "delta_1_10": 2 / 5, "delta_1_25": 3 / 5,
    }
    contract_checks = {
        "metric_arithmetic": equivalent(shared, expected_common),
        "declared_kitti_legacy_arithmetic": equivalent(kitti_metrics, expected_kitti),
        "depth_decode_and_bounds": (
            equivalent(raw.tolist(), [[0, 1 / 256, 1, 80, 0]])
            and equivalent(target.tolist(), [[float("nan"), 1 / 256, 1, 80, 90]])
        ),
        "per_image_aggregation": all(
            equivalent(summary["overall"][key], (value + 1) / 3)
            and equivalent(summary["subsets"]["default"][key], value / 2)
            for key, value in expected_common.items()
        ),
    }
    return {
        "metrics": shared,
        "kitti_metrics": kitti_metrics,
        "contract_checks": contract_checks,
        "raw": raw.tolist(), "gt": target.tolist(), "resize": resized.tolist(),
        "summary": summary,
        "benchmark": benchmark_metadata(DatasetCollection("kitti", [sample])),
    }


def equivalent(left: Any, right: Any) -> bool:
    if isinstance(left, dict):
        return isinstance(right, dict) and left.keys() == right.keys() and all(
            equivalent(left[key], right[key]) for key in left
        )
    if isinstance(left, list):
        return isinstance(right, list) and len(left) == len(right) and all(
            equivalent(a, b) for a, b in zip(left, right)
        )
    if isinstance(left, float) and isinstance(right, (int, float)):
        return (math.isnan(left) and math.isnan(right)) or math.isclose(
            left, right, rel_tol=1e-6, abs_tol=1e-7
        )
    return left == right


def check_alignment(projects_root: Path, repositories: Sequence[str]) -> dict[str, Any]:
    reference = projects_root / "lingbot-depth" / "evaluation"
    expected = {
        name: {node.name for node in ast.parse((reference / name).read_text()).body
               if isinstance(node, (ast.FunctionDef, ast.ClassDef))
               and not node.name.startswith("_")}
        for name in COMMON_FILES
    }

    def check(name: str) -> dict[str, Any]:
        project = projects_root / name
        missing = {}
        for rel, symbols in expected.items():
            path = project / "evaluation" / rel
            if not path.is_file():
                missing[rel] = ["module"]
                continue
            present = {node.name for node in ast.parse(path.read_text()).body
                       if isinstance(node, (ast.FunctionDef, ast.ClassDef))}
            if symbols - present:
                missing[rel] = sorted(symbols - present)
        python = project / ".venv/bin/python"
        result = subprocess.run(
            [str(python) if python.exists() else sys.executable,
             str(Path(__file__).resolve()), "--worker", str(project)],
            cwd=project, text=True, capture_output=True, check=True,
        )
        return {"repository": name, "missing_interfaces": missing,
                "fixture": json.loads(result.stdout)}

    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(check, repositories))
    reference_result = next(row for row in results if row["repository"] == "lingbot-depth")
    reference_fixture = reference_result["fixture"]
    for row in results:
        row["contract_checks"] = row["fixture"]["contract_checks"]
        row["numerical_parity"] = equivalent(reference_fixture, row.pop("fixture"))
    return {
        "passed": all(not row["missing_interfaces"] and row["numerical_parity"]
                      and all(row["contract_checks"].values())
                      for row in results),
        "common_modules": len(COMMON_FILES),
        "checks": results,
        "scope": "shared data/scoring only; native model input/output policies remain independent",
    }


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--projects-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--repositories", nargs="+", default=list(REPOSITORIES))
    parser.add_argument("--worker", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.worker:
        print(json.dumps(worker(args.worker)))
        return
    if "lingbot-depth" not in args.repositories:
        parser.error("--repositories must include the lingbot-depth reference")
    report = check_alignment(args.projects_root.expanduser().resolve(), args.repositories)
    print(json.dumps(report, indent=2))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

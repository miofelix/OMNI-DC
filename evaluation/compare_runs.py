"""Compare completed runs only when their recorded benchmark identities match."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, Sequence

from evaluation.core.protocol import validate_benchmark


def compare_runs(run_dirs: Sequence[Path]) -> Dict[str, Any]:
    if len(run_dirs) < 2:
        raise ValueError("Select at least two evaluation run directories")
    runs = []
    for root in run_dirs:
        root = root.expanduser().resolve()
        with (root / "run.json").open(encoding="utf-8") as file:
            metadata = json.load(file)
        evaluation = metadata.get("results", {}).get("evaluation")
        if metadata.get("status") != "completed" or not evaluation:
            raise ValueError(f"Run has no completed evaluation: {root}")
        if not metadata.get("benchmark"):
            raise ValueError(f"Run has no shared benchmark identity: {root}")
        if evaluation.get("num_evaluated") != len(metadata["benchmark"]["selected_samples"]):
            raise ValueError(f"Run did not evaluate its complete recorded selection: {root}")
        if metadata["benchmark"]["dataset"] == "ibims":
            official = metadata["benchmark"]["official_evaluator"]
            if (evaluation.get("seed") != official["seed"]
                    or evaluation.get("official_script_sha256") != official["scripts_sha256"]):
                raise ValueError(
                    f"Actual iBims evaluator differs from the recorded benchmark: {root}"
                )
        runs.append((root, metadata, evaluation))

    benchmark = runs[0][1]["benchmark"]
    for root, metadata, _ in runs[1:]:
        try:
            validate_benchmark(benchmark, metadata["benchmark"])
        except ValueError as exc:
            raise ValueError(f"Cannot compare {root}: {exc}") from exc

    rows = []
    metric_names = []
    for root, metadata, evaluation in runs:
        summary = evaluation["summary"]
        if benchmark["dataset"] == "ibims":
            groups = summary
        else:
            groups = {**summary["subsets"], "overall": summary["overall"]}
        for subset, metrics in groups.items():
            row = {
                "run_dir": str(root),
                "model_path": metadata.get("model_path"),
                "subset": subset,
                **metrics,
            }
            rows.append(row)
            for name in metrics:
                if name != "num_samples" and name not in metric_names:
                    metric_names.append(name)
    return {
        "dataset": benchmark["dataset"],
        "benchmark_sha256": benchmark["sha256"],
        "metric_names": metric_names,
        "rows": rows,
        "model_adaptations": [
            {
                "run_dir": str(root),
                "config": metadata["config"],
                "manifest_sha256": metadata["benchmark"]["manifest_sha256"],
                "sample_metadata": metadata.get("sample_metadata", []),
                "inference": metadata["results"].get("inference", {}),
                "scoring_coverage": evaluation.get("coverage"),
            }
            for root, metadata, evaluation in runs
        ],
    }


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dirs", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, help="Write a combined metrics CSV")
    args = parser.parse_args(argv)
    try:
        report = compare_runs(args.run_dirs)
    except (ValueError, OSError, KeyError) as exc:
        parser.error(str(exc))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", encoding="utf-8", newline="") as file:
            fields = ["run_dir", "model_path", "subset", "num_samples", *report["metric_names"]]
            writer = csv.DictWriter(file, fieldnames=fields)
            writer.writeheader()
            writer.writerows(report["rows"])
        print(f"Comparable benchmark {report['benchmark_sha256']}: {args.output}")
    else:
        print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

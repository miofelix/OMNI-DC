from __future__ import annotations

import os
import re
import shutil
import subprocess
from collections import defaultdict
from pathlib import Path, PurePosixPath
from typing import Any, Dict, List, Mapping, Sequence

import numpy as np
from scipy.io import loadmat, savemat

from evaluation.core.io import load_prediction
from evaluation.core.output import RunLayout, write_csv, write_json
from evaluation.core.types import EvaluationSample
from evaluation.datasets.base import DatasetCollection
from evaluation.datasets.ibims import IBIMS_EXPECTED_SHAPE

RESULT_METRIC_KEYS = (
    "rel",
    "sq_rel",
    "rms",
    "log10",
    "thr1",
    "thr2",
    "thr3",
    "dde_0",
    "dde_p",
    "dde_m",
    "pe_fla",
    "pe_ori",
    "dbe_acc",
    "dbe_com",
)


def parse_eval_stdout(text: str) -> Dict[str, float]:
    results: Dict[str, float] = {}
    in_results = False
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not in_results:
            if line == "Results:":
                in_results = True
            continue
        if not line:
            continue
        match = re.match(r"(\S+)\s*=\s*([\d.eE+\-]+)$", line)
        if not match:
            break
        results[match.group(1)] = float(match.group(2))
    return results


def flat_ibims_id(sample: EvaluationSample) -> str:
    path = PurePosixPath(sample.sample_id)
    if len(path.parts) != 1:
        raise ValueError(
            f"iBims official evaluation requires a flat sample_id, got {sample.sample_id!r}"
        )
    return path.name


def export_official_predictions(
    samples: Sequence[EvaluationSample], layout: RunLayout, subset: str
) -> Path:
    prediction_dir = layout.official_prediction_dir(subset)
    prediction_dir.mkdir(parents=True, exist_ok=True)
    for sample in samples:
        prediction = load_prediction(layout.prediction_path(sample)).astype(np.float32, copy=False)
        if prediction.shape != IBIMS_EXPECTED_SHAPE:
            raise ValueError(
                f"Unexpected iBims prediction shape for {sample.sample_id}: "
                f"{prediction.shape}, expected {IBIMS_EXPECTED_SHAPE}"
            )
        prediction = prediction.copy()
        prediction[~np.isfinite(prediction) | (prediction <= 0.0)] = np.nan
        savemat(
            prediction_dir / f"{flat_ibims_id(sample)}_results.mat",
            {"pred_depths": prediction},
        )
    return prediction_dir


def link_or_copy(source: Path, destination: Path) -> None:
    source = source.resolve()
    if destination.exists() or destination.is_symlink():
        if destination.resolve() == source:
            return
        destination.unlink()
    try:
        destination.symlink_to(source)
    except OSError:
        shutil.copy2(source, destination)


def validate_official_prediction(path: Path) -> None:
    value = loadmat(path)
    if "pred_depths" not in value:
        raise ValueError(f"Missing pred_depths variable: {path}")
    if value["pred_depths"].shape != IBIMS_EXPECTED_SHAPE:
        raise ValueError(
            f"Unexpected MAT shape in {path}: {value['pred_depths'].shape}, "
            f"expected {IBIMS_EXPECTED_SHAPE}"
        )


def prepare_workspace(
    ibims_root: Path,
    samples: Sequence[EvaluationSample],
    prediction_dir: Path,
    workspace: Path,
) -> Path:
    mat_root = ibims_root / "ibims1_core_mat"
    eval_script = ibims_root / "evaluation_scripts" / "evaluate_ibims.py"
    for required in (mat_root, eval_script, prediction_dir):
        if not required.exists():
            raise FileNotFoundError(f"Missing required iBims path: {required}")

    workspace.mkdir(parents=True, exist_ok=True)
    names = [flat_ibims_id(sample) for sample in samples]
    with (workspace / "imagelist.txt").open("w", encoding="utf-8") as file:
        for name in names:
            file.write(f"{name}\n")

    for name in names:
        gt_mat = mat_root / f"{name}.mat"
        pred_mat = prediction_dir / f"{name}_results.mat"
        if not gt_mat.is_file():
            raise FileNotFoundError(f"Missing iBims ground truth: {gt_mat}")
        if not pred_mat.is_file():
            raise FileNotFoundError(f"Missing iBims prediction: {pred_mat}")
        validate_official_prediction(pred_mat)
        link_or_copy(gt_mat, workspace / gt_mat.name)
        link_or_copy(pred_mat, workspace / pred_mat.name)
    return eval_script


def run_evaluator(eval_script: Path, workspace: Path, log_path: Path) -> str:
    environment = os.environ.copy()
    script_dir = str(eval_script.parent)
    environment["PYTHONPATH"] = script_dir + os.pathsep + environment.get("PYTHONPATH", "")
    project_root = Path(__file__).resolve().parents[2]
    uv = shutil.which("uv") or "uv"
    result = subprocess.run(
        [
            uv,
            "run",
            "--project",
            str(project_root),
            "--extra",
            "evaluation",
            "python",
            str(eval_script),
        ],
        cwd=workspace,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as file:
        file.write(result.stdout)
        if result.stderr:
            file.write("\n[stderr]\n")
            file.write(result.stderr)
    if result.returncode != 0:
        raise RuntimeError(
            f"Official iBims evaluator failed with exit code {result.returncode}; see {log_path}"
        )
    return result.stdout


def run_ibims_official_evaluation(
    collection: DatasetCollection,
    layout: RunLayout,
    ibims_root: Path,
) -> Dict[str, Any]:
    grouped: Dict[str, List[EvaluationSample]] = defaultdict(list)
    for sample in collection.samples:
        grouped[sample.subset].append(sample)

    all_metrics: Dict[str, Dict[str, float]] = {}
    for subset, samples in sorted(grouped.items()):
        prediction_dir = export_official_predictions(samples, layout, subset)
        workspace = layout.official_workspace(subset)
        eval_script = prepare_workspace(ibims_root, samples, prediction_dir, workspace)
        stdout = run_evaluator(eval_script, workspace, layout.official_log_path(subset))
        metrics = parse_eval_stdout(stdout)
        if not metrics:
            raise ValueError(
                f"No metrics were parsed from the official iBims output for {subset}; "
                f"see {layout.official_log_path(subset)}"
            )
        all_metrics[subset] = metrics

    summary_rows: List[Mapping[str, Any]] = []
    discovered_keys = list(RESULT_METRIC_KEYS)
    for metrics in all_metrics.values():
        for key in metrics:
            if key not in discovered_keys:
                discovered_keys.append(key)
    for subset, metrics in sorted(all_metrics.items()):
        summary_rows.append({"subset": subset, **metrics})

    write_csv(
        layout.metrics_dir / "summary.csv",
        summary_rows,
        ["subset", *discovered_keys],
    )
    write_json(
        layout.metrics_dir / "summary.json",
        {
            "metric_names": discovered_keys,
            "subsets": all_metrics,
        },
    )
    return {
        "num_evaluated": sum(len(samples) for samples in grouped.values()),
        "summary": all_metrics,
    }

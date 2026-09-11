import json
from dataclasses import replace

import numpy as np
import pytest

from evaluation.compare_runs import compare_runs
from evaluation.core.output import save_prediction
from evaluation.core.pipeline import run_pipeline
from evaluation.core.protocol import benchmark_metadata, validate_benchmark
from evaluation.core.types import EvaluationSample, RunConfig
from evaluation.datasets.base import DatasetCollection


def collection(tmp_path, **overrides):
    sample = EvaluationSample(
        sample_id="scene/frame", subset="default", rgb_path=tmp_path / "rgb.png",
        raw_depth_path=tmp_path / "raw.png", gt_depth_path=tmp_path / "gt.png",
        depth_scale=256.0, min_depth=1 / 256, max_depth=float("inf"), raw_max_depth=80.0,
    )
    return DatasetCollection("kitti", [replace(sample, **overrides)])


def test_identity_is_model_independent_and_encodes_unbounded_kitti_gt(tmp_path):
    original = benchmark_metadata(collection(tmp_path))
    adapted = benchmark_metadata(collection(tmp_path, metadata={"intrinsics_path": "native.txt"}))
    assert original == adapted
    assert original["selected_samples"][0]["max_depth"] == "Infinity"
    json.dumps(original, allow_nan=False)
    assert original != benchmark_metadata(collection(tmp_path, raw_max_depth=50.0))
    assert original != benchmark_metadata(collection(tmp_path, max_depth=80.0))


def test_manifest_only_model_changes_preserve_benchmark_and_record_provenance(tmp_path):
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text('{"intrinsics": "camera-a.txt"}\n')
    selected = replace(collection(tmp_path), metadata={"manifest": str(manifest)})
    before = benchmark_metadata(selected)
    manifest.write_text('{"intrinsics": "camera-b.txt"}\n')
    after = benchmark_metadata(selected)
    assert before["selection_sha256"] == after["selection_sha256"]
    assert before["sha256"] == after["sha256"]
    assert before["manifest_sha256"] != after["manifest_sha256"]
    validate_benchmark(before, after)


def test_ibims_identity_includes_official_code_and_scoring_seed(tmp_path):
    selected = replace(collection(tmp_path), name="ibims")
    script = tmp_path / "evaluation_scripts/evaluate_ibims.py"
    script.parent.mkdir()
    script.write_text("print('version 1')")
    before = benchmark_metadata(selected, tmp_path, evaluation_seed=0)
    assert before != benchmark_metadata(selected, tmp_path, evaluation_seed=1)
    script.write_text("print('version 2')")
    assert before != benchmark_metadata(selected, tmp_path, evaluation_seed=0)


def fake_inference(selected, _config, layout):
    for sample in selected.samples:
        save_prediction(layout.prediction_path(sample), np.ones((2, 2), dtype=np.float32))
    return {"num_predictions": len(selected.samples)}


def fake_evaluation(selected, _layout):
    scores = {"num_samples": len(selected.samples), "rmse": 0.0}
    return {
        "num_evaluated": len(selected.samples),
        "summary": {"subsets": {"default": scores}, "overall": scores},
    }


def test_resume_and_comparison_reject_changed_benchmark(tmp_path, monkeypatch):
    # Exercise metadata validation and stage transitions without model dependencies.
    import sys
    from types import ModuleType
    inference = ModuleType("evaluation.core.inference")
    inference.run_inference = fake_inference
    monkeypatch.setitem(sys.modules, "evaluation.core.inference", inference)
    monkeypatch.setattr("evaluation.core.pipeline.run_depth_evaluation", fake_evaluation)
    selected = collection(tmp_path, metadata={"intrinsics_path": "model-a-camera.txt"})
    config = RunConfig("kitti", "all", tmp_path / "model-a", "checkpoint-a")
    first = run_pipeline(selected, config)
    second = run_pipeline(
        collection(tmp_path, metadata={"intrinsics_path": "model-b-camera.txt"}),
        replace(config, run_dir=tmp_path / "model-b",
                                            model_path="checkpoint-b"))
    report = compare_runs([first.root, second.root])
    assert len(report["rows"]) == 4
    assert report["model_adaptations"][0]["sample_metadata"] != (
        report["model_adaptations"][1]["sample_metadata"]
    )
    evaluate = replace(config, stage="evaluate", model_path=None)
    run_pipeline(selected, evaluate)
    saved = json.loads(first.metadata_path.read_text())
    assert saved["config"]["model_path"] == "checkpoint-a"
    assert saved["evaluation_config"]["model_path"] is None
    with pytest.raises(ValueError, match="benchmark mismatch"):
        run_pipeline(collection(tmp_path, raw_max_depth=50.0), evaluate)
    assert json.loads(first.metadata_path.read_text()) == saved
    third = run_pipeline(collection(tmp_path, raw_max_depth=50.0),
                         replace(config, run_dir=tmp_path / "changed"))
    with pytest.raises(ValueError, match="Cannot compare"):
        compare_runs([first.root, third.root])


def test_old_run_cannot_claim_shared_benchmark(tmp_path):
    run = tmp_path / "legacy"
    run.mkdir()
    (run / "run.json").write_text(json.dumps({
        "status": "completed", "results": {"evaluation": {"num_evaluated": 1}},
    }))
    with pytest.raises(ValueError, match="no shared benchmark identity"):
        compare_runs([run, run])


def test_all_stage_rechecks_official_code_after_inference(tmp_path, monkeypatch):
    import sys
    from types import ModuleType

    script = tmp_path / "evaluation_scripts/evaluate_ibims.py"
    script.parent.mkdir()
    script.write_text("print('version 1')")
    selected = replace(collection(tmp_path), name="ibims")
    config = RunConfig("ibims", "all", tmp_path / "run", "fake-model")

    def mutate_official_code(selected, config, layout):
        result = fake_inference(selected, config, layout)
        script.write_text("print('version 2')")
        return result

    inference = ModuleType("evaluation.core.inference")
    inference.run_inference = mutate_official_code
    monkeypatch.setitem(sys.modules, "evaluation.core.inference", inference)
    with pytest.raises(ValueError, match="official_evaluator"):
        run_pipeline(selected, config, ibims_root=tmp_path)
    assert json.loads((config.run_dir / "run.json").read_text())["status"] == "failed"


def test_compare_checks_actual_official_evaluator_provenance(tmp_path):
    selected = replace(collection(tmp_path), name="ibims")
    benchmark = benchmark_metadata(selected, tmp_path)
    run = tmp_path / "run"
    run.mkdir()
    (run / "run.json").write_text(json.dumps({
        "status": "completed", "benchmark": benchmark,
        "results": {"evaluation": {"num_evaluated": 1, "seed": 99,
                                    "official_script_sha256": {}}},
    }))
    with pytest.raises(ValueError, match="Actual iBims evaluator differs"):
        compare_runs([run, run])


def test_scoring_coverage_exposes_adapter_invalid_predictions(tmp_path):
    import cv2

    from evaluation.core.output import RunLayout
    from evaluation.evaluators.depth import run_depth_evaluation

    selected = collection(tmp_path)
    assert cv2.imwrite(str(selected.samples[0].gt_depth_path),
                       np.full((2, 2), 256, dtype=np.uint16))
    layout = RunLayout(tmp_path / "run")
    save_prediction(layout.prediction_path(selected.samples[0]),
                    np.array([[np.nan, 0], [1, 1]], dtype=np.float32))
    result = run_depth_evaluation(selected, layout)
    assert result["coverage"] == {"valid_pixels": 3, "gt_valid_pixels": 4,
                                   "prediction_coverage": 0.75}
    assert result["summary"]["overall"]["mae"] == pytest.approx(1 / 3)

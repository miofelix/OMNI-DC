"""CPU contract tests for the OMNI-DC adapter, without loading CUDA kernels."""

import json
import random
import subprocess
import sys
from dataclasses import fields, replace
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
import torch

from evaluation.cli import build_config, build_parser
from evaluation.core import inference
from evaluation.core.output import RunLayout
from evaluation.core.pipeline import run_pipeline
from evaluation.core.types import EvaluationSample, RunConfig
from evaluation.datasets.base import DatasetCollection
from evaluation.datasets.ibims import load_ibims
from evaluation.datasets.transpose import load_transpose
from evaluation.evaluators.ibims_official import run_evaluator


def run_config(tmp_path, dataset="kitti"):
    return RunConfig(
        dataset=dataset,
        stage="infer",
        run_dir=tmp_path / "run",
        model_path="mock-checkpoint.pt",
        save_visualizations=False,
    )


@pytest.mark.parametrize("dataset", ["kitti", "hammer"])
def test_inference_preserves_native_omni_prediction_policy(tmp_path, monkeypatch, dataset):
    """Shared scoring must not replace OMNI-DC's native output normalization."""
    rgb_path = tmp_path / "rgb.png"
    raw_path = tmp_path / "raw.png"
    cv2.imwrite(str(rgb_path), np.zeros((2, 4, 3), dtype=np.uint8))
    cv2.imwrite(
        str(raw_path), np.array([[0, 256, 20480, 20736], [512, 768, 1024, 1280]], dtype=np.uint16)
    )
    sample = EvaluationSample(
        sample_id="scene/frame",
        subset="default",
        rgb_path=rgb_path,
        raw_depth_path=raw_path,
        gt_depth_path=tmp_path / "gt-is-not-needed-for-inference.png",
        depth_scale=256.0,
        min_depth=1.0 / 256.0,
        max_depth=float("inf"),
        raw_max_depth=80.0,
    )
    collection = DatasetCollection(dataset, [sample])
    config = run_config(tmp_path, dataset)
    layout = RunLayout(config.run_dir)
    output = np.array([[0.0, -1.0, 2.0, np.nan], [np.inf, -np.inf, 3.0, 4.0]], dtype=np.float32)
    received = {}

    def fake_infer(model, rgb, raw_depth, intrinsics, device, use_fp16):
        received["raw_depth"] = raw_depth.copy()
        return output.copy()

    # Exercise the real orchestration, decoding and NPY writing without invoking
    # native CUDA-only OGNIDC or requiring a GPU in these contract tests.
    monkeypatch.setattr(inference, "select_device", lambda name: torch.device("cuda"))
    monkeypatch.setattr(inference, "load_model", lambda *args: object())
    monkeypatch.setattr(inference, "_infer_one", fake_infer)
    monkeypatch.setattr(inference.torch.cuda, "is_available", lambda: False)
    result = inference.run_inference(collection, config, layout)
    saved = np.load(layout.prediction_path(sample), allow_pickle=False)

    np.testing.assert_array_equal(received["raw_depth"][0], [0.0, 1.0, 80.0, 0.0])
    np.testing.assert_array_equal(saved, output)
    assert saved.dtype == np.float32
    assert result["model_class"] == "model.ognidc.OGNIDC"
    assert result["num_predictions"] == 1


def test_native_adapter_preserves_pixels_units_and_padding():
    class RecordingModel(torch.nn.Module):
        args = SimpleNamespace(num_resolution=3)

        def forward(self, sample):
            self.received = sample
            return {"pred": sample["dep"] + 1.0}

    model = RecordingModel()
    rgb = np.full((17, 19, 3), [255, 128, 0], dtype=np.uint8)
    raw = np.arange(17 * 19, dtype=np.float32).reshape(17, 19) / 10.0
    camera = np.array([[500, 0, 9], [0, 510, 8], [0, 0, 1]], dtype=np.float32)
    output = inference._infer_one(model, rgb, raw, camera, torch.device("cpu"), False)

    assert model.received["rgb"].shape == (1, 3, 32, 32)
    assert model.received["dep"].shape == (1, 1, 32, 32)
    np.testing.assert_allclose(
        model.received["rgb"][0, :, 0, 0].numpy(),
        (np.array([1.0, 128.0 / 255.0, 0.0]) - [0.485, 0.456, 0.406])
        / [0.229, 0.224, 0.225],
        rtol=1e-6,
    )
    np.testing.assert_array_equal(model.received["dep"][0, 0, :17, :19], raw)
    np.testing.assert_array_equal(model.received["K"][0], camera)
    assert not torch.count_nonzero(model.received["dep"][..., 17:, :])
    assert set(model.received) == {"rgb", "dep", "K", "pattern"}
    np.testing.assert_array_equal(output, raw + 1.0)


def test_ibims_calibration_metadata_and_per_sample_priority_are_preserved(tmp_path):
    manifest_dir = tmp_path / "ibims1_synthetic_raw_depth" / "manifests"
    manifest_dir.mkdir(parents=True)
    (manifest_dir / "ibims_easy.jsonl").write_text(
        json.dumps({"sample_id": "frame", "rgb": "rgb.png", "raw_depth": "raw.png"})
        + "\n"
    )
    collection = load_ibims(tmp_path, ["easy"])
    sample = collection.samples[0]
    calibration = tmp_path / "ibims1_core_raw" / "calib" / "frame.txt"
    assert sample.metadata == {
        "difficulty": "easy", "intrinsics_path": str(calibration),
        "intrinsics_source": "ibims_calibration",
    }
    config = replace(run_config(tmp_path, "ibims"), save_visualizations=True)
    np.testing.assert_array_equal(inference._intrinsics_for_sample(sample, config), np.eye(3))
    fallback = tmp_path / "camera.txt"
    with pytest.raises(FileNotFoundError, match="Intrinsics file not found"):
        inference._intrinsics_for_sample(sample, replace(config, intrinsics_path=fallback))
    np.savetxt(fallback, np.diag([510, 520, 1]))
    np.testing.assert_array_equal(
        inference._intrinsics_for_sample(sample, replace(config, intrinsics_path=fallback)),
        np.diag([510, 520, 1]),
    )

    calibration.parent.mkdir(parents=True)
    calibration.write_text("500 510 320 240\n")
    expected = np.array([[500, 0, 320], [0, 510, 240], [0, 0, 1]], dtype=np.float32)
    np.testing.assert_array_equal(inference._intrinsics_for_sample(sample, config), expected)
    fallback = tmp_path / "camera.txt"
    np.savetxt(fallback, np.eye(3))
    np.testing.assert_array_equal(
        inference._intrinsics_for_sample(sample, replace(config, intrinsics_path=fallback)),
        expected,
    )


def test_official_evaluator_preserves_uv_project_environment_and_seed(tmp_path, monkeypatch):
    script = tmp_path / "official.py"
    script.write_text(
        "import sys, random\nimport numpy as np\n"
        "print(sys.executable)\nprint(random.random())\nprint(np.random.random())\n"
    )
    log_path = tmp_path / "logs" / "official.log"
    commands = []
    original_run = subprocess.run

    def execute_launcher(command, **kwargs):
        commands.append((command, kwargs))
        # Check the production uv environment contract while executing the real
        # seeded launcher locally, without resolving/downloading any environment.
        return original_run([sys.executable, *command[7:]], **kwargs)

    monkeypatch.setattr("evaluation.evaluators.ibims_official.shutil.which", lambda name: "/bin/uv")
    monkeypatch.setattr("evaluation.evaluators.ibims_official.subprocess.run", execute_launcher)
    first = run_evaluator(script, tmp_path, log_path, seed=29)
    second = run_evaluator(script, tmp_path, log_path, seed=29)
    assert first == second == log_path.read_text()
    output = first.splitlines()
    assert output[0] == sys.executable
    assert float(output[1]) == random.Random(29).random()
    assert float(output[2]) == np.random.RandomState(29).random()
    command, kwargs = commands[0]
    assert command[:7] == [
        "/bin/uv", "run", "--project", str(Path(__file__).resolve().parents[1]),
        "--extra", "evaluation", "python",
    ]
    assert command[7] == "-c"
    assert command[-2:] == [str(script), "29"]
    assert kwargs["cwd"] == tmp_path
    assert kwargs["env"]["PYTHONHASHSEED"] == "29"


def test_transpose_adapter_retains_l515_depth_and_sequence_ids(tmp_path):
    manifest = tmp_path / "transpose.jsonl"
    manifest.write_text(
        json.dumps(
            {
                "seq_name": "scene/frame",
                "rgb": "rgb.png",
                "l515_depth": "raw.png",
                "depth": "gt.png",
            }
        ) + "\n"
    )
    collection = load_transpose(manifest)
    sample = collection.samples[0]
    assert collection.name == "transpose"
    assert sample.sample_id == "scene/frame"
    assert sample.raw_depth_path == tmp_path / "raw.png"
    assert sample.depth_scale == 1000.0
    assert (sample.min_depth, sample.max_depth) == (0.1, 6.0)


def test_cli_keeps_evaluation_and_model_seeds_independent(tmp_path):
    args = build_parser().parse_args(
        ["ibims", "--seed", "17", "--evaluation-seed", "29", "--no-load-dav2"]
    )
    config = build_config(args, tmp_path)
    assert config.seed == 17
    assert config.evaluation_seed == 29
    assert config.resolution_level == 3
    assert config.load_dav2 is False


def test_new_evaluation_seed_preserves_positional_model_config_fields():
    assert [field.name for field in fields(RunConfig)][-3:] == [
        "load_dav2", "seed", "evaluation_seed"
    ]


def test_zero_native_prediction_remains_a_kitti_error_end_to_end(tmp_path, monkeypatch):
    rgb_path, raw_path, gt_path = [tmp_path / name for name in ("rgb.png", "raw.png", "gt.png")]
    cv2.imwrite(str(rgb_path), np.zeros((2, 2, 3), dtype=np.uint8))
    for path in (raw_path, gt_path):
        cv2.imwrite(str(path), np.full((2, 2), 256, dtype=np.uint16))
    sample = EvaluationSample(
        "scene/frame", "default", rgb_path, raw_path, gt_path,
        depth_scale=256.0, min_depth=1.0 / 256.0, max_depth=float("inf"), raw_max_depth=80.0,
    )
    collection = DatasetCollection("kitti", [sample])
    config = replace(run_config(tmp_path), stage="all")
    monkeypatch.setattr(inference, "select_device", lambda name: torch.device("cuda"))
    monkeypatch.setattr(inference, "load_model", lambda *args: object())
    monkeypatch.setattr(inference.torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(
        inference, "_infer_one", lambda *args: np.array([[0, 1], [0, 1]], dtype=np.float32)
    )
    layout = run_pipeline(collection, config)
    metadata = json.loads(layout.metadata_path.read_text())
    summary = metadata["results"]["evaluation"]["summary"]["overall"]
    assert summary["mae"] == 0.5
    assert np.count_nonzero(np.load(layout.prediction_path(sample)) == 0) == 2

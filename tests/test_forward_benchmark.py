"""forward 协议、边界、共享参数和报告的 CPU 测试；不代表 CUDA 性能。"""

from __future__ import annotations

import json
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

from evaluation.benchmark import adapter
from evaluation.benchmark.core import (
    CudaEventClock,
    ForwardBenchmarkSettings,
    benchmark_forward,
    summarize_forward_timings,
)
from evaluation.benchmark.parameters import count_model_parameters
from evaluation.benchmark.support import (
    PreparedForward,
    SyntheticInputSettings,
    comparison_problems,
    probe_forward,
    resolve_config,
    synthetic_inputs,
    write_summary,
)


class _Clock:
    kind = "test"

    def __init__(self):
        self.active = False
        self.samples = 0
        self.synchronizations = 0

    def synchronize(self):
        assert not self.active
        self.synchronizations += 1

    def start(self):
        assert not self.active
        self.active = True

    def stop_ms(self):
        assert self.active
        self.active = False
        self.samples += 1
        return float(self.samples)


def test_timer_only_counts_forward_and_excludes_each_warmup():
    clock = _Clock()
    called = []

    def forward():
        assert torch.is_inference_mode_enabled()
        called.append(clock.active)
        return torch.ones(1)

    report = benchmark_forward(
        forward, clock=clock, settings=ForwardBenchmarkSettings(warmup=2, iterations=3, repeats=2)
    )
    assert called == [False, False, True, True, True] * 2
    assert clock.synchronizations == 2
    assert report["timings_ms"] == [1, 2, 3, 4, 5, 6]
    assert report["samples"] == 6
    assert report["total_ms"] == 21
    assert report["fps"] == pytest.approx(6000 / 21)
    assert report["fps"] != pytest.approx(np.mean([1000 / x for x in range(1, 7)]))
    assert report["clock"] == "test"


@pytest.mark.parametrize("values", [[], [0], [-1], [float("nan")], [float("inf")]])
def test_reject_invalid_timings(values):
    with pytest.raises(ValueError):
        summarize_forward_timings(values)


@pytest.mark.parametrize(
    "kwargs", [{"warmup": -1}, {"iterations": 0}, {"repeats": 0}, {"warmup": True}]
)
def test_reject_invalid_measurement_settings(kwargs):
    with pytest.raises(ValueError):
        ForwardBenchmarkSettings(**kwargs)


def test_parameters_include_frozen_and_shared_networks_but_not_buffers():
    model = nn.Module()
    model.coarse = nn.Linear(3, 2, bias=False)
    model.coarse.requires_grad_(False)
    model.fine = nn.Linear(2, 1)
    model.alias = model.coarse
    model.register_buffer("running", torch.ones(20))
    counts = count_model_parameters(model)
    assert counts["total_numel"] == 9
    assert counts["frozen_numel"] == 6
    assert counts["trainable_numel"] == 3
    assert counts["buffer_numel"] == 20
    assert len(counts["rows"]) == 3
    assert not any(row["name"].startswith("alias") for row in counts["rows"])
    assert not model.coarse.weight.requires_grad


def test_shared_synthetic_input_identity_and_full_density():
    rgb, depth, identity = synthetic_inputs(SyntheticInputSettings())
    assert rgb.shape == (480, 640, 3) and rgb.dtype == np.uint8
    assert depth.shape == (480, 640) and depth.dtype == np.float32
    assert np.isfinite(depth).all() and (depth > 0).all() and depth.max() > depth.min()
    assert identity["valid_pixels"] == 307200
    assert identity["sha256"] == "b88af328c3e8c296a24f566abfa8972f7b9b388823dc15eff699fcebeb867e93"
    assert synthetic_inputs(SyntheticInputSettings(seed=1))[2]["sha256"] != identity["sha256"]


@pytest.mark.parametrize(
    "kwargs",
    [
        {"valid_ratio": 0.05},
        {"batch_size": 2},
        {"seed": -1},
        {"width": 0},
        {"min_depth": 0},
        {"max_depth": float("nan")},
    ],
)
def test_input_protocol_rejects_unsupported_settings(kwargs):
    with pytest.raises(ValueError):
        SyntheticInputSettings(**kwargs)


def test_config_rejects_wrong_project_and_unknown_fields(tmp_path):
    with pytest.raises(ValueError):
        resolve_config({"project": "wrong"}, adapter, tmp_path)
    with pytest.raises(ValueError):
        resolve_config({"silent_resize": True}, adapter, tmp_path)
    with pytest.raises(ValueError):
        resolve_config({"model": {"made_up": True}}, adapter, tmp_path)
    assert resolve_config({}, adapter, tmp_path)["model"] == adapter.DEFAULT_MODEL


class _Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.patch_embed = nn.Identity()

    def forward(self, value):
        return self.patch_embed(value) * 2

    def infer(self, *args, **kwargs):
        raise AssertionError("inference wrapper must not be called")

    predict = infer


def test_probe_removes_hooks_before_measurement_even_on_failure():
    model = _Model()
    prepared = PreparedForward(model, args=(torch.ones(1, 2, 3),))
    result = probe_forward(prepared)
    assert "patch_embed" in result["native_modules"]
    assert not model.patch_embed._forward_pre_hooks
    with pytest.raises(ValueError, match="nonfinite"):
        probe_forward(PreparedForward(model, args=(torch.full((1,), float("nan")),)))
    assert not model.patch_embed._forward_pre_hooks


def test_cpu_clock_never_produces_gpu_fps():
    with pytest.raises(RuntimeError, match="requires CUDA"):
        CudaEventClock("cpu")


def test_failed_results_have_no_fps_and_comparison_detects_mismatch(tmp_path):
    report = {
        "project": adapter.PROJECT,
        "status": "failed",
        "forward": None,
        "parameters": None,
        "error": {"message": "checkpoint missing"},
    }
    write_summary([report], tmp_path)
    text = (tmp_path / "summary.csv").read_text()
    assert "checkpoint missing" in text
    assert (tmp_path / "summary.json").is_file()
    good = {
        "status": "ok",
        "forward": {"clock": "cuda_events", "boundary": "native_forward"},
        "input": {"sha256": "a"},
    }
    other = {**good, "input": {"sha256": "b"}}
    assert comparison_problems([good, other])
    assert not comparison_problems([good, report])


def test_cli_dry_run_has_no_measurements_or_cuda_construction():
    package = adapter.__package__
    process = subprocess.run(
        [sys.executable, "-m", package, "run", "--dry-run"],
        cwd=adapter.REPO_ROOT,
        capture_output=True,
        text=True,
    )
    assert process.returncode == 0, process.stderr
    payload = json.loads(process.stdout)
    assert payload["status"] == "planned"
    assert payload["forward"] is None and payload["parameters"] is None
    assert payload["input"]["valid_ratio"] == 1.0


def test_adapter_prepares_native_forward_without_timing_or_wrapper(monkeypatch):
    rgb, depth, _ = synthetic_inputs(SyntheticInputSettings())
    config = resolve_config({"device": "cpu"}, adapter, adapter.REPO_ROOT)
    if adapter.PROJECT == "asdepth":
        model = _Model()
        prepared = adapter.prepare(model, rgb, depth, config)
        assert prepared.args[0].shape[1] == 4
        assert bool((prepared.args[0][:, 3] > 0).all())
    elif adapter.PROJECT == "lingbot":
        model = SimpleNamespace(num_tokens_range=[1200, 3600])
        prepared = adapter.prepare(model, rgb, depth, config)
        assert prepared.kwargs["num_tokens"] == 3600
        assert prepared.args[0].shape == (1, 3, 480, 640)
        assert prepared.kwargs["depth"].shape == (1, 480, 640)
    elif adapter.PROJECT == "promptda":
        prepared = adapter.prepare(_Model(), rgb, depth, config)
        assert prepared.args[0].shape == (1, 3, 476, 630)
        assert prepared.args[1].shape == (1, 1, 480, 640)
        assert bool((prepared.args[1] > 0).all())
    elif adapter.PROJECT == "priorda":
        from prior_depth_anything.sparse_sampler import SparseSampler

        model = SimpleNamespace(
            args=SimpleNamespace(K=5, double_global=True), sampler=SparseSampler(device="cpu")
        )
        prepared = adapter.prepare(model, rgb, depth, config)
        assert prepared.kwargs["sparse_masks"].all()
        assert not model.args.double_global
        assert prepared.kwargs["geometric_depths"] is None
        assert prepared.metadata["knn_query_pixels"] == 0
    elif adapter.PROJECT == "omni_dc":
        model = SimpleNamespace(args=SimpleNamespace(num_resolution=3, GRU_iters=1, prop_time=6))
        prepared = adapter.prepare(model, rgb, depth, config)
        sample = prepared.args[0]
        assert sample["rgb"].shape == (1, 3, 480, 640)
        assert sample["dep"].shape == (1, 1, 480, 640)
        assert sample["dep"].min() > 0 and prepared.output_key == "pred"
    else:
        calls = []

        class Warp:
            def warp(self, values, **kwargs):
                calls.append(kwargs)
                return values + 7, kwargs["prompt_mask"], None

        model = SimpleNamespace(warp_func=Warp())
        prepared = adapter.prepare(model, rgb, depth, config)
        assert len(calls) == 1
        assert prepared.kwargs["x"].shape == (1, 3, 768, 1024)
        assert prepared.kwargs["coords"].shape == (1, 307200, 2)
        assert prepared.metadata["actual_prompt_samples"] == 1500
        assert not prepared.metadata["query_chunking"]
        assert not prepared.metadata["cached_features"]
        assert bool((prepared.kwargs["prompt_depth"] >= 7).all())


def test_missing_checkpoint_writes_failure_with_null_fps(tmp_path):
    process = subprocess.run(
        [
            sys.executable,
            "-m",
            adapter.__package__,
            "run",
            "--device",
            "cpu",
            "--checkpoint",
            str(tmp_path / "missing.ckpt"),
            "--output",
            str(tmp_path / "result"),
        ],
        cwd=adapter.REPO_ROOT,
        capture_output=True,
        text=True,
    )
    assert process.returncode == 1
    payload = json.loads((tmp_path / "result/benchmark.json").read_text())
    assert payload["status"] == "failed"
    assert payload["forward"] is None
    assert payload["error"]["stage"] == "resources"
    assert (tmp_path / "result/run.log").is_file()


def test_standalone_protocol_code_matches_local_contract():
    import hashlib
    import inspect
    from pathlib import Path

    package = Path(adapter.__file__).parent
    manifest = json.loads((package / "protocol.json").read_text())
    sources = {
        "core": Path(inspect.getfile(benchmark_forward)),
        "parameters": Path(inspect.getfile(count_model_parameters)),
        "support": package / "support.py",
        "cli": package / "cli.py",
    }
    assert manifest["protocol_version"] == "forward-benchmark-v1"
    for name, path in sources.items():
        assert hashlib.sha256(path.read_bytes()).hexdigest() == manifest["sha256"][name]

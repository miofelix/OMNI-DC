"""真实权重/CUDA server gate；Mac 不运行、不生成伪 FPS。

用法：FORWARD_BENCHMARK_CONFIG=/absolute/case.json python -m pytest 本文件。
该配置必须为本仓库的真实 checkpoint 配置；不下载、不缩小模型或查询数。
"""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from evaluation.benchmark import adapter


@pytest.mark.skipif(
    not os.environ.get("FORWARD_BENCHMARK_CONFIG"),
    reason="requires explicit real checkpoint config and CUDA",
)
def test_real_checkpoint_native_forward(tmp_path):
    config = Path(os.environ["FORWARD_BENCHMARK_CONFIG"]).resolve()
    process = subprocess.run(
        [
            sys.executable,
            "-m",
            adapter.__package__,
            "run",
            "--config",
            str(config),
            "--warmup",
            "1",
            "--iterations",
            "2",
            "--repeats",
            "1",
            "--output",
            str(tmp_path),
        ],
        cwd=adapter.REPO_ROOT,
        text=True,
        capture_output=True,
        timeout=1800,
    )
    assert process.returncode == 0, (
        process.stdout + process.stderr + (tmp_path / "run.log").read_text()
    )
    report = json.loads((tmp_path / "benchmark.json").read_text())
    assert report["status"] == "ok" and report["parameters"]["total_numel"] > 0
    assert report["forward"]["fps"] > 0 and report["forward"]["samples"] == 2
    assert report["forward"]["clock"] == "cuda_events"
    assert report["input"]["valid_pixels"] == 640 * 480

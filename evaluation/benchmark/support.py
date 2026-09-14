"""基准配置、合成输入、源码/权重指纹和报告支持。

用法：由本仓库 benchmark CLI 调用；依赖 NumPy、PyTorch，YAML 配置额外依赖 PyYAML。
不读取数据集，不把任何准备工作放入 forward 计时区间。
"""

from __future__ import annotations

import csv
import hashlib
import inspect
import json
import math
import os
import platform
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from .core import PROTOCOL_VERSION, ForwardBenchmarkSettings, benchmark_forward
from .parameters import count_model_parameters


@dataclass(frozen=True)
class SyntheticInputSettings:
    width: int = 640
    height: int = 480
    seed: int = 0
    min_depth: float = 0.1
    max_depth: float = 6.0
    batch_size: int = 1
    valid_ratio: float = 1.0

    def __post_init__(self) -> None:
        for name in ("width", "height"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.width * self.height < 2:
            raise ValueError("synthetic depth must have at least two pixels")
        if type(self.seed) is not int or not 0 <= self.seed < 2**32:
            raise ValueError("seed must be an integer in [0, 2**32)")
        if self.batch_size != 1 or type(self.batch_size) is not int:
            raise ValueError("forward-benchmark-v1 requires batch_size=1")
        if self.valid_ratio != 1.0:
            raise ValueError("forward-benchmark-v1 requires 100% valid synthetic depth")
        if not all(math.isfinite(value) for value in (self.min_depth, self.max_depth)):
            raise ValueError("depth bounds must be finite")
        if not 0 < self.min_depth < self.max_depth:
            raise ValueError("depth bounds must satisfy 0 < min_depth < max_depth")


def synthetic_inputs(settings: SyntheticInputSettings) -> tuple[Any, Any, dict[str, Any]]:
    import numpy as np

    rng = np.random.Generator(np.random.PCG64(settings.seed))
    rgb = rng.integers(0, 256, (settings.height, settings.width, 3), dtype=np.uint8)
    depth = rng.uniform(
        settings.min_depth, settings.max_depth, (settings.height, settings.width)
    ).astype(np.float32)
    if not np.isfinite(depth).all() or not (depth > 0).all() or depth.max() <= depth.min():
        raise ValueError("synthetic depth must be finite, positive and nonconstant in float32")
    digest = hashlib.sha256()
    for array in (rgb, depth):
        digest.update(json.dumps([str(array.dtype), list(array.shape)]).encode())
        digest.update(array.tobytes(order="C"))
    return (
        rgb,
        depth,
        {
            **asdict(settings),
            "generator": "numpy.PCG64/uniform-v1",
            "sha256": digest.hexdigest(),
            "valid_pixels": int(depth.size),
        },
    )


@dataclass
class PreparedForward:
    model: Any
    args: tuple[Any, ...] = ()
    kwargs: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    output_key: str | None = None

    def __call__(self) -> Any:
        return self.model(*self.args, **self.kwargs)


def tensor_descriptions(value: Any, prefix: str = "input") -> list[dict[str, Any]]:
    import torch

    if isinstance(value, torch.Tensor):
        return [
            {
                "name": prefix,
                "shape": list(value.shape),
                "dtype": str(value.dtype),
                "device": str(value.device),
            }
        ]
    if isinstance(value, dict):
        return [
            row
            for name, item in value.items()
            for row in tensor_descriptions(item, f"{prefix}.{name}")
        ]
    if isinstance(value, (list, tuple)):
        return [
            row
            for index, item in enumerate(value)
            for row in tensor_descriptions(item, f"{prefix}.{index}")
        ]
    return []


def check_cuda_inputs(prepared: PreparedForward, device: str) -> list[dict[str, Any]]:
    import torch

    requested = torch.device(device)
    index = requested.index if requested.index is not None else torch.cuda.current_device()
    target = f"cuda:{index}"
    inputs = tensor_descriptions((prepared.args, prepared.kwargs))
    if not inputs or any(item["device"] != target for item in inputs):
        raise ValueError(f"every forward input tensor must already reside on {target}")
    states = list(prepared.model.parameters()) + list(prepared.model.buffers())
    if any(str(item.device) != target for item in states):
        raise ValueError(f"every model parameter and buffer must already reside on {target}")
    return inputs


def probe_forward(prepared: PreparedForward) -> dict[str, Any]:
    """只在非计时探针装载 hook；探针结束后立即移除全部 hook。"""

    import torch

    observed: dict[str, Any] = {}
    handles = []

    def hook(name: str) -> Any:
        def capture(_module: Any, inputs: Any) -> None:
            enabled = torch.is_autocast_enabled()
            dtype = None
            if enabled:
                dtype = (
                    torch.get_autocast_dtype("cuda")
                    if hasattr(torch, "get_autocast_dtype")
                    else torch.get_autocast_gpu_dtype()
                )
            observed.setdefault(
                name,
                {
                    "inputs": tensor_descriptions(inputs),
                    "autocast_enabled": enabled,
                    "autocast_dtype": str(dtype),
                },
            )

        return capture

    for name, module in prepared.model.named_modules():
        if name.endswith(("patch_embed", "basic_encoder")):
            handles.append(module.register_forward_pre_hook(hook(name)))
    try:
        with torch.inference_mode():
            output = prepared()
        primary = output[prepared.output_key] if prepared.output_key is not None else output
        if not isinstance(primary, torch.Tensor) or primary.numel() == 0:
            raise ValueError("native forward must return a nonempty depth/disparity tensor")
        if not bool(torch.isfinite(primary).all()):
            raise ValueError("native forward produced nonfinite depth/disparity values")
        return {
            "primary_output": tensor_descriptions(primary, "output")[0],
            "native_modules": observed,
            "external_autocast": False,
        }
    finally:
        for handle in handles:
            handle.remove()


def read_config(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() in (".yaml", ".yml"):
        import yaml

        payload = yaml.safe_load(text)
    else:
        payload = json.loads(text)
    if not isinstance(payload, dict):
        raise ValueError("config must be a JSON/YAML object")
    return payload


def resolve_path(value: str, base: Path) -> Path:
    expanded = os.path.expandvars(value)
    if "$" in expanded:
        raise ValueError(f"unresolved environment variable in path: {value}")
    path = Path(expanded).expanduser()
    return (path if path.is_absolute() else base / path).resolve()


def resolve_config(payload: dict[str, Any], adapter: Any, base: Path) -> dict[str, Any]:
    allowed = {"protocol_version", "project", "model", "input", "timing", "device"}
    if extra := payload.keys() - allowed:
        raise ValueError(f"unknown config fields: {sorted(extra)}")
    if payload.get("protocol_version", PROTOCOL_VERSION) != PROTOCOL_VERSION:
        raise ValueError("unsupported benchmark protocol")
    if payload.get("project", adapter.PROJECT) != adapter.PROJECT:
        raise ValueError("config project does not match this checkout's benchmark")
    options = dict(adapter.DEFAULT_MODEL)
    supplied = payload.get("model", {})
    if not isinstance(supplied, dict) or supplied.keys() - options.keys():
        raise ValueError(f"model settings must use fields: {sorted(options)}")
    options.update(supplied)
    for key in adapter.CHECKPOINT_FIELDS:
        if options.get(key) is not None:
            options[key] = str(resolve_path(str(options[key]), base))
    adapter.validate_options(options)
    inputs = SyntheticInputSettings(**payload.get("input", {}))
    timing = ForwardBenchmarkSettings(**payload.get("timing", {}))
    device = payload.get("device", "cuda:0")
    if not isinstance(device, str) or not device:
        raise ValueError("device must be a nonempty string")
    return {
        "protocol_version": PROTOCOL_VERSION,
        "project": adapter.PROJECT,
        "model": options,
        "input": asdict(inputs),
        "timing": asdict(timing),
        "device": device,
    }


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def checkpoint_fingerprints(options: dict[str, Any], keys: tuple[str, ...]) -> dict[str, Any]:
    result = {}
    for key in keys:
        if not options.get(key):
            raise ValueError(
                f"model.{key} must name a local checkpoint; automatic download is disabled"
            )
        path = Path(options[key])
        if not path.exists():
            raise FileNotFoundError(f"checkpoint not found: {path}")
        files = [path] if path.is_file() else sorted(p for p in path.rglob("*") if p.is_file())
        if not files:
            raise ValueError(f"empty checkpoint directory: {path}")
        entries = [
            {
                "name": str(p.relative_to(path)) if path.is_dir() else path.name,
                "bytes": p.stat().st_size,
                "sha256": file_sha256(p),
            }
            for p in files
        ]
        result[key] = {"path": str(path), "files": entries}
    return result


def source_identity(root: Path) -> dict[str, Any]:
    def git(*args: str) -> subprocess.CompletedProcess[bytes]:
        return subprocess.run(["git", *args], cwd=root, capture_output=True, check=False)

    revision = git("rev-parse", "HEAD")
    diff = git("diff", "--binary", "HEAD")
    status = git("status", "--porcelain")
    code_paths = list(Path(__file__).parent.glob("*.py"))
    for function in (benchmark_forward, count_model_parameters):
        name = inspect.getsourcefile(function)
        if name:
            code_paths.append(Path(name))
    code = {str(p.relative_to(root)): file_sha256(p) for p in sorted(set(code_paths))}
    untracked = git("ls-files", "--others", "--exclude-standard", "-z").stdout.decode().split("\0")
    untracked_code = {
        name: file_sha256(root / name)
        for name in untracked
        if name.endswith(".py") and (root / name).is_file()
    }
    return {
        "repo": str(root),
        "commit": revision.stdout.decode().strip() or None,
        "dirty": bool(status.stdout),
        "status": status.stdout.decode(),
        "tracked_diff_sha256": hashlib.sha256(diff.stdout).hexdigest(),
        "benchmark_files": code,
        "untracked_python_files": untracked_code,
    }


def environment(device: str) -> dict[str, Any]:
    import torch

    result: dict[str, Any] = {
        "python": sys.version,
        "executable": sys.executable,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "torch": str(torch.__version__),
        "cuda": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "device": device,
        "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "torch_num_threads": torch.get_num_threads(),
        "cuda_launch_blocking": os.environ.get("CUDA_LAUNCH_BLOCKING", "0"),
        "container_image": os.environ.get("BENCHMARK_CONTAINER_IMAGE"),
    }
    if torch.cuda.is_available() and torch.device(device).type == "cuda":
        result.update(
            gpu=torch.cuda.get_device_name(device),
            capability=list(torch.cuda.get_device_capability(device)),
            arch_list=torch.cuda.get_arch_list(),
        )
        try:
            smi = subprocess.run(
                ["nvidia-smi", "--query-gpu=name,uuid,driver_version", "--format=csv,noheader"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            result["nvidia_smi"] = smi.stdout.strip()
        except (OSError, subprocess.TimeoutExpired):
            result["nvidia_smi"] = None
    for name in ("numpy", "torchvision", "xformers", "torch-cluster"):
        from importlib.metadata import PackageNotFoundError, version

        try:
            result[name] = version(name)
        except PackageNotFoundError:
            result[name] = None
    return result


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    try:
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def comparison_problems(reports: list[dict[str, Any]]) -> list[str]:
    measured = [r for r in reports if r.get("status") == "ok" and r.get("forward")]
    problems = []
    signatures = []
    for report in measured:
        forward = report["forward"]
        if forward.get("clock") != "cuda_events" or forward.get("boundary") != "native_forward":
            problems.append(f"{report.get('project')}: not a native CUDA forward measurement")
        env = report.get("environment", {})
        signature = {
            key: env.get(key)
            for key in (
                "torch",
                "cuda",
                "gpu",
                "capability",
                "matmul_allow_tf32",
                "cudnn_allow_tf32",
                "cudnn_benchmark",
                "torch_num_threads",
                "cuda_launch_blocking",
                "deterministic_algorithms",
                "numpy",
                "torchvision",
                "xformers",
            )
        }
        signature["input_sha256"] = report.get("input", {}).get("sha256")
        signature["timing"] = report.get("config", {}).get("timing")
        signatures.append(json.dumps(signature, sort_keys=True))
    if len(set(signatures)) > 1:
        problems.append("input, timing protocol or GPU/runtime differs between measured runs")
    return problems


def write_summary(reports: list[dict[str, Any]], output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    for report in reports:
        parameters = report.get("parameters") or {}
        forward = (report.get("forward") or {}) if report.get("status") == "ok" else {}
        rows.append(
            {
                "project": report.get("project"),
                "status": report.get("status"),
                "parameters": parameters.get("total_numel"),
                "parameters_m": parameters.get("total_m"),
                "forward_fps": forward.get("fps"),
                "mean_ms": forward.get("mean_ms"),
                "median_ms": forward.get("median_ms"),
                "p95_ms": forward.get("p95_ms"),
                "forward_inputs": json.dumps(report.get("forward_inputs"), ensure_ascii=False),
                "forward_output": json.dumps(
                    (report.get("probe") or {}).get("primary_output"), ensure_ascii=False
                ),
                "precision": json.dumps(report.get("precision"), ensure_ascii=False),
                "error": (report.get("error") or {}).get("message"),
            }
        )
    fields = list(rows[0]) if rows else ["project", "status"]
    with (output / "summary.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    problems = comparison_problems(reports)
    lines = [
        "# 原生 forward 基准",
        "",
        "GPU forward FPS；batch=1；不含外部准备和后处理。",
        "",
        "| 项目 | 状态 | 参数量 M | FPS | mean ms | median ms | P95 ms | "
        "forward 输入 → 输出 | 精度 |",
        "|---|---|---:|---:|---:|---:|---:|---|---|",
    ]
    if problems:
        lines[2] += " **比较条件不一致：请检查 summary.json。**"
    for row, report in zip(rows, reports, strict=True):

        def cell(key: str, values: dict[str, Any]) -> str:
            value = values[key]
            return (
                "—" if value is None else f"{value:.4f}" if isinstance(value, float) else str(value)
            )

        shape_text = "; ".join(
            "×".join(map(str, item["shape"])) for item in report.get("forward_inputs", [])
        )
        primary = (report.get("probe") or {}).get("primary_output")
        if primary:
            shape_text += " → " + "×".join(map(str, primary["shape"]))
        precision = report.get("precision") or {}
        precision_text = ",".join(precision.get("parameter_dtypes", []))
        amp = {
            item["autocast_dtype"]
            for item in precision.get("observed_modules", {}).values()
            if item.get("autocast_enabled")
        }
        if amp:
            precision_text += "; AMP: " + ",".join(sorted(amp))
        lines.append(
            "| "
            + " | ".join(
                cell(key, row)
                for key in (
                    "project",
                    "status",
                    "parameters_m",
                    "forward_fps",
                    "mean_ms",
                    "median_ms",
                    "p95_ms",
                )
            )
            + " | "
            + (shape_text or "—")
            + " | "
            + (precision_text or "—")
            + " |"
        )
    lines.extend(
        [
            "",
            "各模型的实际尺寸、精度、配置、权重和运行环境见 summary.json；"
            "不同配置不能视为同尺寸同精度比较。",
            "",
        ]
    )
    (output / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    write_json(
        output / "summary.json",
        {
            "protocol_version": PROTOCOL_VERSION,
            "comparison_compatible": not problems,
            "comparison_problems": problems,
            "reports": reports,
        },
    )

"""独立基准 CLI：doctor、params、run、summarize。

典型用法：python -m evaluation.benchmark run --config benchmark.json --output outputs/forward。
AS-Depth 使用 python -m tools.analysis.model_benchmark。依赖本仓库模型环境；FPS 仅支持 CUDA。
"""

from __future__ import annotations

import argparse
import contextlib
import json
import random
import sys
import traceback
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from . import adapter
from .core import CudaEventClock, ForwardBenchmarkSettings, benchmark_forward
from .parameters import count_model_parameters
from .support import (
    SyntheticInputSettings,
    check_cuda_inputs,
    checkpoint_fingerprints,
    comparison_problems,
    environment,
    probe_forward,
    read_config,
    resolve_config,
    source_identity,
    synthetic_inputs,
    write_json,
    write_summary,
)


def build_parser(extra_parser: Any = None) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=f"{adapter.PROJECT}: native forward 参数量和 CUDA FPS"
    )
    commands = parser.add_subparsers(dest="command", required=True)
    for command in ("doctor", "params", "run"):
        child = commands.add_parser(command)
        child.add_argument("--config", type=Path)
        child.add_argument("--checkpoint")
        if "mde_checkpoint" in adapter.CHECKPOINT_FIELDS:
            child.add_argument("--mde-checkpoint")
        child.add_argument("--device")
        child.add_argument("--output", type=Path)
        child.add_argument(
            "--dry-run", action="store_true", help="只验证配置和合成输入，不加载模型"
        )
        for name in ("width", "height", "seed", "warmup", "iterations", "repeats"):
            child.add_argument(f"--{name}", type=int)
        child.add_argument(
            "--probe", action="store_true", help="doctor 时加载真实模型并验证一次 forward"
        )
    child = commands.add_parser("summarize")
    child.add_argument("reports", nargs="+", type=Path)
    child.add_argument("--output", type=Path, required=True)
    if extra_parser is not None:
        extra_parser(commands)
    return parser


def command_config(args: argparse.Namespace) -> dict[str, Any]:
    payload = read_config(args.config) if args.config else {}
    base = args.config.resolve().parent if args.config else Path.cwd()
    payload.setdefault("model", {})
    for field in adapter.CHECKPOINT_FIELDS:
        if value := getattr(args, field, None):
            payload["model"][field] = value
    if args.device:
        payload["device"] = args.device
    for group, names in (
        ("input", ("width", "height", "seed")),
        ("timing", ("warmup", "iterations", "repeats")),
    ):
        payload.setdefault(group, {})
        for name in names:
            if (value := getattr(args, name)) is not None:
                payload[group][name] = value
    return resolve_config(payload, adapter, base)


def execute(args: argparse.Namespace) -> int:
    config = command_config(args)
    report: dict[str, Any] = {
        "protocol_version": config["protocol_version"],
        "project": adapter.PROJECT,
        "command": args.command,
        "status": "planned",
        "config": config,
        "parameters": None,
        "forward": None,
    }
    rgb, depth, input_info = synthetic_inputs(SyntheticInputSettings(**config["input"]))
    report["input"] = input_info
    if args.dry_run:
        print(json.dumps(report, indent=2, ensure_ascii=False))
        return 0
    output = (
        args.output
        or adapter.REPO_ROOT
        / "outputs"
        / "forward_benchmark"
        / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8])
    ).resolve()
    if (output / "benchmark.json").exists():
        raise ValueError(f"result already exists: {output}; choose a new output directory")
    output.mkdir(parents=True, exist_ok=True)
    report["output"] = str(output)
    stage = "environment"
    with (output / "run.log").open("w", encoding="utf-8") as log:
        try:
            with contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
                report["environment"] = environment(config["device"])
                report["source"] = source_identity(adapter.REPO_ROOT)
                stage = "resources"
                for field in adapter.CHECKPOINT_FIELDS:
                    if not config["model"].get(field):
                        raise ValueError(f"model.{field} must name a local checkpoint")
                    if not Path(config["model"][field]).exists():
                        raise FileNotFoundError(config["model"][field])
                if args.command == "doctor" and not args.probe:
                    CudaEventClock(config["device"])
                    report["status"] = "ready_for_probe"
                    report["note"] = (
                        "Environment/resources only; model kernels and FPS are not validated."
                    )
                else:
                    import numpy as np
                    import torch

                    random.seed(config["input"]["seed"])
                    np.random.seed(config["input"]["seed"])
                    torch.manual_seed(config["input"]["seed"])
                    # 原生构造器可能使用无参数 .cuda()；先选择目标设备。
                    if torch.device(config["device"]).type == "cuda":
                        clock = CudaEventClock(config["device"])
                        torch.cuda.set_device(
                            torch.device(config["device"]).index
                            if torch.device(config["device"]).index is not None
                            else torch.cuda.current_device()
                        )
                    elif args.command != "params":
                        raise RuntimeError(
                            "forward FPS/probe requires CUDA; CPU/MPS is not a fallback"
                        )
                    stage = "checkpoint_fingerprint"
                    report["checkpoints"] = checkpoint_fingerprints(
                        config["model"], adapter.CHECKPOINT_FIELDS
                    )
                    stage = "load_model"
                    model, model_metadata = adapter.load(config["model"], config["device"])
                    import inspect

                    model_source = Path(inspect.getfile(type(model))).resolve()
                    if not model_source.is_relative_to(adapter.REPO_ROOT):
                        raise ValueError(
                            f"model was imported from another checkout: {model_source}"
                        )
                    model_metadata["source_file"] = str(model_source)
                    model_metadata["attention_classes"] = sorted(
                        {
                            type(module).__module__ + "." + type(module).__name__
                            for module in model.modules()
                            if "attention" in type(module).__name__.lower()
                            or "attn" in type(module).__name__.lower()
                        }
                    )
                    model.eval()
                    report["model"] = model_metadata
                    report["parameters"] = count_model_parameters(model)
                    report["precision"] = {
                        "policy": "native_forward_no_external_amp",
                        "parameter_dtypes": sorted({str(p.dtype) for p in model.parameters()}),
                    }
                    if args.command != "params":
                        stage = "prepare_inputs"
                        with torch.inference_mode():
                            prepared = adapter.prepare(model, rgb, depth, config)
                        report["forward_inputs"] = check_cuda_inputs(prepared, config["device"])
                        report["native_settings"] = prepared.metadata
                        stage = "probe_forward"
                        report["probe"] = probe_forward(prepared)
                        report["precision"]["observed_modules"] = report["probe"]["native_modules"]
                        if args.command == "run":
                            stage = "measure_forward"
                            report["forward"] = benchmark_forward(
                                prepared,
                                clock=clock,
                                settings=ForwardBenchmarkSettings(**config["timing"]),
                            )
                    report["status"] = "ok"
        except Exception as exc:
            traceback.print_exc(file=log)
            report["status"] = "failed"
            report["forward"] = None
            report["error"] = {"stage": stage, "type": type(exc).__name__, "message": str(exc)}
    write_json(output / "benchmark.json", report)
    write_summary([report], output)
    print(
        json.dumps(
            {
                "project": adapter.PROJECT,
                "status": report["status"],
                "result": str(output / "benchmark.json"),
                "fps": (report.get("forward") or {}).get("fps"),
            },
            ensure_ascii=False,
        )
    )
    if report["status"] == "failed":
        print(
            f"{stage}: {report['error']['message']} (details: {output / 'run.log'})",
            file=sys.stderr,
        )
        return 1
    return 0


def main(argv: list[str] | None = None, extra_parser: Any = None) -> int:
    args = build_parser(extra_parser).parse_args(argv)
    try:
        if hasattr(args, "handler"):
            return int(args.handler(args))
        if args.command == "summarize":
            reports = [
                read_config(path / "benchmark.json" if path.is_dir() else path)
                for path in args.reports
            ]
            from .core import PROTOCOL_VERSION

            if any(report.get("protocol_version") != PROTOCOL_VERSION for report in reports):
                raise ValueError("cannot summarize incompatible benchmark protocols")
            write_summary(reports, args.output)
            return int(bool(comparison_problems(reports)))
        return execute(args)
    except (ValueError, TypeError, OSError, ImportError, RuntimeError) as exc:
        print(f"benchmark: {exc}", file=sys.stderr)
        return 2

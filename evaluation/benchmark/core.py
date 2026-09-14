"""原生 forward 的 CUDA Events 测速与确定性统计。

用法：benchmark_forward(prepared_call, clock=CudaEventClock("cuda:0"))。
依赖：PyTorch；统计函数只需标准库。真实测速要求 CUDA，输入准备由调用方完成。
协议：forward-benchmark-v1；其他仓库保留本模块的独立副本。
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from typing import Any, Protocol

PROTOCOL_VERSION = "forward-benchmark-v1"


@dataclass(frozen=True)
class ForwardBenchmarkSettings:
    warmup: int = 50
    iterations: int = 200
    repeats: int = 3

    def __post_init__(self) -> None:
        for name in ("warmup", "iterations", "repeats"):
            value = getattr(self, name)
            minimum = 0 if name == "warmup" else 1
            if type(value) is not int or value < minimum:
                raise ValueError(f"{name} must be an integer >= {minimum}")


class ForwardClock(Protocol):
    kind: str

    def synchronize(self) -> None: ...
    def start(self) -> None: ...
    def stop_ms(self) -> float: ...


class CudaEventClock:
    """同一设备当前 stream 上的计时；不会降级为 CPU/MPS 时钟。"""

    kind = "cuda_events"

    def __init__(self, device: str = "cuda:0") -> None:
        import torch

        self.device = torch.device(device)
        if self.device.type != "cuda" or not torch.cuda.is_available():
            raise RuntimeError("forward FPS requires CUDA; CPU/MPS cannot produce benchmark FPS")
        event_factory: Any = (
            torch.cuda.Event
        )  # PyTorch event bindings do not expose typed signatures.
        with torch.cuda.device(self.device):
            self._start: Any = event_factory(enable_timing=True)
            self._end: Any = event_factory(enable_timing=True)
        self._torch = torch

    def synchronize(self) -> None:
        self._torch.cuda.synchronize(self.device)

    def start(self) -> None:
        self._start.record(self._torch.cuda.current_stream(self.device))

    def stop_ms(self) -> float:
        self._end.record(self._torch.cuda.current_stream(self.device))
        self._end.synchronize()
        return float(self._start.elapsed_time(self._end))


def _percentile(ordered: list[float], fraction: float) -> float:
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def summarize_forward_timings(timings_ms: Sequence[float]) -> dict[str, float | int]:
    """以总耗时计算 batch=1 FPS；不平均逐帧 FPS 或各轮 FPS。"""

    values = [float(value) for value in timings_ms]
    if not values or any(not math.isfinite(value) or value <= 0 for value in values):
        raise ValueError("forward timings must be nonempty, finite and strictly positive")
    total = math.fsum(values)
    ordered = sorted(values)
    return {
        "samples": len(values),
        "total_ms": total,
        "mean_ms": total / len(values),
        "median_ms": _percentile(ordered, 0.5),
        "p95_ms": _percentile(ordered, 0.95),
        "fps": len(values) * 1000.0 / total,
    }


def benchmark_forward(
    forward: Callable[[], Any],
    *,
    clock: ForwardClock,
    settings: ForwardBenchmarkSettings | None = None,
) -> dict[str, Any]:
    """只包住 forward；输出释放、同步等待和统计在 event 区间外。"""

    import torch

    settings = settings or ForwardBenchmarkSettings()
    rounds: list[dict[str, Any]] = []
    timings: list[float] = []
    with torch.inference_mode():
        for repeat in range(settings.repeats):
            for _ in range(settings.warmup):
                output = forward()
                del output
            clock.synchronize()
            current: list[float] = []
            for _ in range(settings.iterations):
                clock.start()
                output = forward()
                elapsed = clock.stop_ms()
                del output
                current.append(elapsed)
            rounds.append({"repeat": repeat, **summarize_forward_timings(current)})
            timings.extend(current)
    return {
        "protocol_version": PROTOCOL_VERSION,
        "boundary": "native_forward",
        "clock": clock.kind,
        "settings": asdict(settings),
        **summarize_forward_timings(timings),
        "rounds": rounds,
        "timings_ms": timings,
    }

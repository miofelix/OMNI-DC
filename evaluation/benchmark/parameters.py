"""统计实际模型的去重参数，保留冻结状态并单列 buffer。

用法：count_model_parameters(model)。依赖 PyTorch 模型接口；不加载权重、不选择设备。
同时供原有参数统计工具和 forward benchmark 使用。
"""

from __future__ import annotations

import json
from typing import Any


def count_model_parameters(model: Any) -> dict[str, Any]:
    """统计 nn.Parameter 元素数；共享参数只计一次，buffer 不计入总参数。"""

    rows: list[dict[str, Any]] = []
    groups: dict[str, dict[str, Any]] = {}
    for name, parameter in model.named_parameters(remove_duplicate=True):
        size = int(parameter.numel())
        trainable = bool(parameter.requires_grad)
        group_name = name.split(".", 1)[0]
        group = groups.setdefault(
            group_name,
            {
                "name": group_name,
                "parameter_tensors": 0,
                "trainable_tensors": 0,
                "frozen_tensors": 0,
                "total_numel": 0,
                "total_bytes": 0,
                "trainable_numel": 0,
                "frozen_numel": 0,
            },
        )
        group["parameter_tensors"] += 1
        group["total_numel"] += size
        group["total_bytes"] += size * parameter.element_size()
        label = "trainable" if trainable else "frozen"
        group[f"{label}_numel"] += size
        group[f"{label}_tensors"] += 1
        rows.append(
            {
                "name": name,
                "top_module": group_name,
                "shape": json.dumps(list(parameter.shape)),
                "numel": size,
                "numel_m": f"{size / 1e6:.6f}",
                "requires_grad": trainable,
                "dtype": str(parameter.dtype).replace("torch.", ""),
                "bytes": size * parameter.element_size(),
            }
        )
    for group in groups.values():
        total = group["total_numel"]
        group["frozen_ratio"] = group["frozen_numel"] / total if total else 0.0
        group["trainable_ratio"] = group["trainable_numel"] / total if total else 0.0
    buffers = [
        {
            "name": name,
            "shape": list(value.shape),
            "numel": int(value.numel()),
            "dtype": str(value.dtype),
            "bytes": int(value.numel() * value.element_size()),
        }
        for name, value in model.named_buffers(remove_duplicate=True)
    ]
    return {
        "total_numel": sum(row["numel"] for row in rows),
        "total_m": sum(row["numel"] for row in rows) / 1e6,
        "trainable_numel": sum(row["numel"] for row in rows if row["requires_grad"]),
        "frozen_numel": sum(row["numel"] for row in rows if not row["requires_grad"]),
        "parameter_tensors": len(rows),
        "total_bytes": sum(row["bytes"] for row in rows),
        "rows": sorted(rows, key=lambda row: row["name"]),
        "groups": [groups[name] for name in sorted(groups)],
        "buffers": buffers,
        "buffer_numel": sum(row["numel"] for row in buffers),
    }

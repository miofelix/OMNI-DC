"""OMNI-DC 原生 forward 装配；包含 DA-V2、优化层与传播，不执行外部裁剪/保存。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .support import PreparedForward

PROJECT = "omni_dc"
REPO_ROOT = Path(__file__).resolve().parents[2]
CHECKPOINT_FIELDS = ("checkpoint", "mde_checkpoint")
DEFAULT_MODEL = {"checkpoint": None, "mde_checkpoint": None, "resolution_level": 3}


def validate_options(options: dict[str, Any]) -> None:
    if options["resolution_level"] != 3:
        raise ValueError("v1 OMNI-DC checkpoint requires resolution_level=3")


def load(options: dict[str, Any], device: str) -> tuple[Any, dict[str, Any]]:
    import torch

    from evaluation.core.inference import load_model
    from evaluation.core.types import RunConfig

    config = RunConfig(
        dataset="synthetic",
        stage="infer",
        run_dir=Path("."),
        model_path=options["checkpoint"],
        resolution_level=3,
        load_dav2=True,
    )
    model = load_model(
        options["checkpoint"],
        torch.device(device),
        config,
        dav2_checkpoint=options["mde_checkpoint"],
    )
    return model, {"class": "OGNIDC", "load_dav2": True, "args": vars(model.args)}


def prepare(model: Any, rgb: Any, depth: Any, config: dict[str, Any]) -> PreparedForward:
    import numpy as np
    import torch

    from evaluation.core.inference import _rgb_tensor

    device = torch.device(config["device"])
    image = _rgb_tensor(rgb, device)
    raw = torch.from_numpy(depth).unsqueeze(0).unsqueeze(0).to(device)
    camera = torch.from_numpy(np.eye(3, dtype=np.float32)).unsqueeze(0).to(device)
    divisor = 4 * 2 ** (model.args.num_resolution - 1)
    h, w = depth.shape
    pad = (0, (-w) % divisor, 0, (-h) % divisor)
    if any(pad):
        image = torch.nn.functional.pad(image, pad)
        raw = torch.nn.functional.pad(raw, pad)
    sample = {"rgb": image, "dep": raw, "K": camera, "pattern": 0}
    return PreparedForward(
        model,
        args=(sample,),
        output_key="pred",
        metadata={
            "entrypoint": "OGNIDC.forward",
            "GRU_iters": model.args.GRU_iters,
            "prop_time": model.args.prop_time,
            "num_resolution": model.args.num_resolution,
            "padding": list(pad),
        },
    )

from __future__ import annotations

import os
import random
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from evaluation.core.io import (
    read_gt_depth,
    read_raw_depth,
    read_rgb,
    squeeze_depth,
)
from evaluation.core.output import RunLayout, save_prediction
from evaluation.core.types import EvaluationSample, LoadedSample, RunConfig
from evaluation.core.visualization import (
    save_kitti_pointcloud_visualization,
    save_kitti_prediction_visualization,
    save_visualization,
)
from evaluation.datasets.base import DatasetCollection

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover

    def tqdm(iterable, **_kwargs):
        return iterable


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"


class InferenceInputDataset(Dataset):
    def __init__(self, samples: Sequence[EvaluationSample], load_gt: bool):
        self.samples = list(samples)
        self.load_gt = load_gt

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> LoadedSample:
        sample = self.samples[index]
        rgb = read_rgb(sample.rgb_path)
        raw_depth = read_raw_depth(
            sample.raw_depth_path,
            sample.depth_scale,
            sample.min_depth,
            sample.raw_max_depth if sample.raw_max_depth is not None else sample.max_depth,
        )
        gt_depth = None
        if self.load_gt and sample.gt_depth_path is not None:
            gt_depth = read_gt_depth(
                sample.gt_depth_path,
                sample.depth_scale,
                sample.min_depth,
                sample.max_depth,
            )
        return LoadedSample(sample=sample, rgb=rgb, raw_depth=raw_depth, gt_depth=gt_depth)


def collate_loaded_samples(items: List[LoadedSample]) -> List[LoadedSample]:
    return items


def select_device(name: str) -> torch.device:
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    if name == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS was requested but is not available")
    return torch.device(name)


def _default_model_args(config: RunConfig) -> SimpleNamespace:
    return SimpleNamespace(
        model="OGNIDC",
        load_dav2=bool(config.load_dav2),
        num_resolution=config.resolution_level,
        multi_resolution_learnable_gradients_weights="uniform",
        multi_resolution_learnable_input_weights=0,
        backbone_mode="rgbd",
        backbone="cformer",
        pred_confidence_input=1,
        pred_context_feature=True,
        pred_depth=False,
        depth_activation_format="exp",
        whiten_sparse_depths=1,
        GRU_iters=1,
        gru_internal_whiten_method="median",
        gru_hidden_dim=64,
        gru_context_dim=64,
        optim_layer_input_clamp=1.0,
        integration_alpha=5.0,
        max_depth=300.0,
        prop_time=6,
        prop_kernel=3,
        spn_type="dyspn",
        conf_prop=True,
        conf_min=1.0,
        preserve_input=False,
        affinity="TGASS",
        affinity_gamma=0.5,
        backbone_output_downsample_rate=4,
        depth_downsample_method="min",
        training_depth_random_shift_range=0.0,
        backbone_pattern_condition_format="none",
        num_pattern_types=3,
        loss="1.0*SeqL1+1.0*SeqL2",
    )


def _torch_load(path: Path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:  # torch < 2.6
        return torch.load(path, map_location="cpu")


def _state_dict(checkpoint):
    if isinstance(checkpoint, dict):
        for key in ("net", "model", "state_dict"):
            if key in checkpoint and isinstance(checkpoint[key], dict):
                checkpoint = checkpoint[key]
                break
    if not isinstance(checkpoint, dict):
        raise TypeError("OMNI-DC checkpoint must contain a state-dict mapping")
    return {
        key[7:] if isinstance(key, str) and key.startswith("module.") else key: value
        for key, value in checkpoint.items()
    }


def load_model(
    model_path: str, device: torch.device, config: RunConfig,
    *, dav2_checkpoint: str | None = None,
) -> torch.nn.Module:
    if str(SRC_DIR) not in sys.path:
        sys.path.insert(0, str(SRC_DIR))
    from model.ognidc import OGNIDC

    checkpoint_path = Path(model_path).expanduser().resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"OMNI-DC checkpoint not found: {checkpoint_path}")

    model_args = _default_model_args(config)
    if dav2_checkpoint is not None:
        model_args.dav2_checkpoint_path = str(Path(dav2_checkpoint).expanduser().resolve())
    previous_cwd = Path.cwd()
    os.chdir(SRC_DIR)
    try:
        model = OGNIDC(model_args)
    finally:
        os.chdir(previous_cwd)

    result = model.load_state_dict(_state_dict(_torch_load(checkpoint_path)), strict=False)
    if result.missing_keys:
        raise RuntimeError(
            "Checkpoint is missing required OMNI-DC keys:\n"
            + "\n".join(f"  - {key}" for key in result.missing_keys[:50])
        )
    if result.unexpected_keys:
        print("Warning: unexpected checkpoint keys ignored:")
        for key in result.unexpected_keys[:50]:
            print(f"  - {key}")
    return model.to(device).eval()


def _load_intrinsics(path: Path) -> np.ndarray:
    text = path.read_text(encoding="utf-8")
    cleaned = []
    key_values = {}
    for line in text.splitlines():
        line = line.split("#", 1)[0].strip()
        if not line:
            continue
        cleaned.append(line)
        if "=" in line or ":" in line:
            key, _, value = line.replace(":", "=").partition("=")
            try:
                key_values[key.strip().lower()] = float(value.strip())
            except ValueError:
                pass
    if {"fx", "fy", "cx", "cy"}.issubset(key_values):
        return np.array(
            [
                [key_values["fx"], 0.0, key_values["cx"]],
                [0.0, key_values["fy"], key_values["cy"]],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )
    values = [float(value) for line in cleaned for value in line.replace(",", " ").split()]
    if len(values) == 4:
        fx, fy, cx, cy = values
        return np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float32)
    if len(values) == 9:
        return np.asarray(values, dtype=np.float32).reshape(3, 3)
    if len(values) == 16:
        return np.asarray(values, dtype=np.float32).reshape(4, 4)[:3, :3]
    raise ValueError(f"Unsupported intrinsics format: {path}")


def _intrinsics_for_sample(sample: EvaluationSample, config: RunConfig) -> np.ndarray:
    """Prefer available sample calibration, then explicit K, then native identity.

    Native src/data/ibims.py uses an identity matrix, and BasicUpdateBlock
    currently accepts K without reading it. Automatically derived calibration
    paths are optional; an explicitly requested file must exist.
    """
    value = sample.metadata.get("intrinsics_path")
    if value is not None:
        path = Path(value).expanduser().resolve()
        if path.is_file():
            return _load_intrinsics(path)
    if config.intrinsics_path is not None:
        path = config.intrinsics_path.expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Intrinsics file not found: {path}")
        return _load_intrinsics(path)
    if value is not None and sample.metadata.get("intrinsics_source") != "ibims_calibration":
        raise FileNotFoundError(f"Intrinsics file not found: {Path(value).expanduser().resolve()}")
    return np.eye(3, dtype=np.float32)


def restore_native_prediction(prediction: np.ndarray, target_shape: tuple[int, int]) -> np.ndarray:
    """Preserve OGNIDC values after native padding has been removed.

    src/main.py forwards the cropped output directly to DCMetric. In particular,
    zero predictions remain errors on valid GT; they must not become NaNs here.
    Dataset-specific scoring masks are applied by the shared evaluator.
    """
    restored = squeeze_depth(prediction).astype(np.float32, copy=False)
    if restored.shape != target_shape:
        raise ValueError(
            f"Native OGNIDC prediction shape mismatch: got {restored.shape}, "
            f"expected {target_shape}"
        )
    return restored


def _rgb_tensor(rgb: np.ndarray, device: torch.device) -> torch.Tensor:
    value = rgb.astype(np.float32) / 255.0
    value = (value - np.array([0.485, 0.456, 0.406], dtype=np.float32)) / np.array(
        [0.229, 0.224, 0.225], dtype=np.float32
    )
    return torch.from_numpy(value).permute(2, 0, 1).unsqueeze(0).to(device)


def _infer_one(
    model: torch.nn.Module,
    rgb: np.ndarray,
    raw_depth: np.ndarray,
    intrinsics: np.ndarray,
    device: torch.device,
    use_fp16: bool,
) -> np.ndarray:
    image = _rgb_tensor(rgb, device)
    depth = torch.from_numpy(raw_depth).unsqueeze(0).unsqueeze(0).to(device)
    camera = torch.from_numpy(intrinsics).reshape(1, 3, 3).to(device=device, dtype=torch.float32)
    height, width = raw_depth.shape
    divisor = int(4 * 2 ** (model.args.num_resolution - 1))
    padded_height = ((height + divisor - 1) // divisor) * divisor
    padded_width = ((width + divisor - 1) // divisor) * divisor
    if padded_height != height or padded_width != width:
        image = torch.nn.functional.pad(image, (0, padded_width - width, 0, padded_height - height))
        depth = torch.nn.functional.pad(depth, (0, padded_width - width, 0, padded_height - height))
    sample = {"rgb": image, "dep": depth, "K": camera, "pattern": 0}
    autocast_enabled = use_fp16 and device.type == "cuda"
    with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=autocast_enabled):
        output = model(sample)
    if "pred" not in output:
        raise ValueError("OGNIDC did not return output['pred']")
    prediction = output["pred"][..., :height, :width]
    return prediction.squeeze().detach().float().cpu().numpy().astype(np.float32, copy=False)


@torch.inference_mode()
def run_inference(
    collection: DatasetCollection,
    config: RunConfig,
    layout: RunLayout,
) -> Dict[str, object]:
    if not config.model_path:
        raise ValueError("--model-path is required for inference")
    if config.batch_size < 1:
        raise ValueError("--batch-size must be greater than zero")
    if config.num_workers < 0:
        raise ValueError("--num-workers cannot be negative")
    if not 1 <= config.resolution_level <= 3:
        raise ValueError("--resolution-level must be between 1 and 3 for OMNI-DC")
    if config.apply_mask:
        raise ValueError("--apply-mask is not supported by the native OMNI-DC output")

    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)

    device = select_device(config.device)
    if device.type != "cuda":
        raise RuntimeError(
            "Native OMNI-DC inference requires CUDA; CPU/MPS are supported only for "
            "dataset, metric, and output validation."
        )
    model = load_model(config.model_path, device, config)
    if collection.name == "kitti" and config.save_visualizations:
        for sample in collection.samples:
            intrinsics_path = sample.metadata.get("intrinsics_path") or config.intrinsics_path
            if not intrinsics_path:
                raise ValueError(
                    "KITTI visualization requires an intrinsics path in every manifest row "
                    "or --intrinsics-path"
                )
    dataset = InferenceInputDataset(collection.samples, load_gt=config.save_visualizations)
    loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=device.type == "cuda",
        collate_fn=collate_loaded_samples,
    )

    written = 0
    use_fp16 = config.use_fp16 and device.type == "cuda"
    for batch in tqdm(loader, desc=f"{collection.name} inference"):
        for item in batch:
            sample = item.sample
            if item.raw_depth.shape != item.rgb.shape[:2]:
                raise ValueError(
                    f"RGB/raw-depth shape mismatch for {sample.sample_id}: "
                    f"rgb={item.rgb.shape[:2]}, raw={item.raw_depth.shape}"
                )
            if sample.expected_shape is not None and item.raw_depth.shape != sample.expected_shape:
                raise ValueError(
                    f"Unexpected input shape for {sample.sample_id}: "
                    f"got {item.raw_depth.shape}, expected {sample.expected_shape}"
                )
            intrinsics = _intrinsics_for_sample(sample, config)
            prediction = restore_native_prediction(
                _infer_one(model, item.rgb, item.raw_depth, intrinsics, device, use_fp16),
                item.raw_depth.shape,
            )
            if sample.expected_shape is not None and prediction.shape != sample.expected_shape:
                raise ValueError(
                    f"Unexpected prediction shape for {sample.sample_id}: "
                    f"got {prediction.shape}, expected {sample.expected_shape}"
                )
            save_prediction(layout.prediction_path(sample), prediction)
            if config.save_visualizations:
                save_visualization(
                    layout.visualization_path(sample),
                    item.rgb,
                    item.raw_depth,
                    prediction,
                    item.gt_depth,
                    config.visualization_min_depth,
                    config.visualization_max_depth,
                )
                if collection.name == "kitti":
                    save_kitti_prediction_visualization(
                        layout.kitti_prediction_visualization_path(sample),
                        prediction,
                        config.visualization_min_depth,
                        config.visualization_max_depth,
                    )
                    intrinsics_value = sample.metadata.get("intrinsics_path")
                    intrinsics_path = (
                        Path(intrinsics_value)
                        if intrinsics_value
                        else config.intrinsics_path
                    )
                    assert intrinsics_path is not None
                    save_kitti_pointcloud_visualization(
                        layout.kitti_pointcloud_visualization_path(sample),
                        item.rgb,
                        prediction,
                        intrinsics_path,
                        config.pointcloud_rot_x_deg,
                        config.pointcloud_rot_y_deg,
                        config.pointcloud_knn_k,
                        config.pointcloud_knn_std_ratio,
                        config.disable_pointcloud_knn_filter,
                    )
            written += 1

    return {
        "num_predictions": written,
        "device": str(device),
        "use_fp16": use_fp16,
        "model_class": "model.ognidc.OGNIDC",
        "input_preprocessing": (
            "RGB ImageNet normalization; sparse depth meters; native OGNIDC padding"
        ),
        "prediction_postprocessing": (
            "native padding removed; float32 metric depth; original values preserved"
        ),
        "intrinsics_policy": "available per-sample calibration; explicit global K; native identity",
        "depth_output": "metric_depth_meter",
        "seed": config.seed,
    }

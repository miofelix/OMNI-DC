from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Sequence

from evaluation import __version__
from evaluation.core.output import model_stem
from evaluation.core.pipeline import run_pipeline
from evaluation.core.types import RunConfig
from evaluation.datasets import (
    load_clearpose,
    load_dreds,
    load_hammer,
    load_ibims,
    load_kitti,
    load_transpose,
)
from evaluation.datasets.dreds import DREDS_VARIANTS
from evaluation.datasets.ibims import IBIMS_LEVELS
from evaluation.datasets.kitti import KITTI_DEFAULT_RAW_MAX_DEPTH

DEFAULT_OUTPUT_ROOT = Path("outputs/evaluation")


def add_common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model-path", default=None, help="Path to an OMNI-DC checkpoint")
    parser.add_argument(
        "--stage", choices=("all", "infer", "evaluate"), default="all", help="Pipeline stage"
    )
    parser.add_argument("--run-dir", type=Path, default=None, help="Evaluation run directory")
    parser.add_argument("--device", choices=("auto", "cuda", "mps", "cpu"), default="auto")
    parser.add_argument(
        "--resolution-level",
        type=int,
        choices=(1, 2, 3),
        default=3,
        help="Native OMNI-DC multiresolution level",
    )
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--use-fp16", action="store_true", help="Enable CUDA autocast")
    parser.add_argument(
        "--apply-mask",
        action="store_true",
        help="Rejected for OMNI-DC because its native output already applies propagation",
    )
    parser.add_argument(
        "--save-visualizations",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--cleanup-predictions", action="store_true")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--visualization-min-depth", type=float, default=0.1)
    parser.add_argument("--visualization-max-depth", type=float, default=5.0)
    parser.add_argument(
        "--intrinsics-path",
        type=Path,
        default=None,
        help="Global 3x3 or fx fy cx cy camera intrinsics file used by native OGNIDC",
    )
    parser.add_argument("--load-dav2", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--seed", type=int, default=0, help="OMNI-DC inference random seed")
    parser.add_argument(
        "--evaluation-seed",
        type=int,
        default=0,
        help="Random seed for the official iBims evaluator, independent of model inference",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="uv run python -m evaluation",
        description="OMNI-DC dataset evaluation pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--version", action="version", version=__version__)
    subparsers = parser.add_subparsers(dest="dataset", required=True)

    hammer = subparsers.add_parser(
        "hammer",
        help="Run HAMMER evaluation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    add_common_arguments(hammer)
    hammer.add_argument("--manifest", type=Path, required=True)
    hammer.add_argument("--camera", choices=("d435", "l515", "tof"), default="d435")

    clearpose = subparsers.add_parser(
        "clearpose",
        help="Run ClearPose evaluation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    add_common_arguments(clearpose)
    clearpose.add_argument("--manifest", type=Path, required=True)

    dreds = subparsers.add_parser(
        "dreds", help="Run DREDS evaluation", formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    add_common_arguments(dreds)
    dreds.add_argument("--known-manifest", type=Path, default=None)
    dreds.add_argument("--novel-manifest", type=Path, default=None)
    dreds.add_argument(
        "--variants", nargs="+", choices=DREDS_VARIANTS, default=list(DREDS_VARIANTS)
    )

    transpose = subparsers.add_parser(
        "transpose",
        help="Run TRansPose L515 evaluation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    add_common_arguments(transpose)
    transpose.add_argument("--manifest", type=Path, required=True)

    ibims = subparsers.add_parser(
        "ibims",
        help="Run official iBims evaluation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    add_common_arguments(ibims)
    ibims.add_argument("--ibims-root", type=Path, default=Path("data/ibims1"))
    ibims.add_argument("--levels", nargs="+", choices=IBIMS_LEVELS, default=list(IBIMS_LEVELS))
    ibims.add_argument("--depth-scale", type=float, default=None)
    ibims.add_argument("--max-depth", type=float, default=None)

    kitti = subparsers.add_parser(
        "kitti",
        help="Run KITTI Depth Completion val_selection_cropped evaluation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    add_common_arguments(kitti)
    kitti.add_argument("--manifest", type=Path, required=True)
    kitti.add_argument("--raw-max-depth", type=float, default=KITTI_DEFAULT_RAW_MAX_DEPTH)
    kitti.add_argument("--pointcloud-rot-x-deg", type=float, default=25.0)
    kitti.add_argument("--pointcloud-rot-y-deg", type=float, default=15.0)
    kitti.add_argument("--pointcloud-knn-k", type=int, default=16)
    kitti.add_argument("--pointcloud-knn-std-ratio", type=float, default=2.0)
    kitti.add_argument("--disable-pointcloud-knn-filter", action="store_true")
    kitti.set_defaults(visualization_max_depth=KITTI_DEFAULT_RAW_MAX_DEPTH)
    return parser


def resolve_run_dir(args: argparse.Namespace, parser: argparse.ArgumentParser) -> Path:
    if args.run_dir is not None:
        return args.run_dir
    if args.stage == "evaluate":
        parser.error("--run-dir is required when --stage=evaluate")
    if not args.model_path:
        parser.error("--model-path is required for inference")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return DEFAULT_OUTPUT_ROOT / args.dataset / f"{model_stem(args.model_path)}_{timestamp}"


def build_collection(args: argparse.Namespace, parser: argparse.ArgumentParser):
    if args.dataset == "hammer":
        return load_hammer(args.manifest, args.camera, args.max_samples), None
    if args.dataset == "clearpose":
        return load_clearpose(args.manifest, args.max_samples), None
    if args.dataset == "dreds":
        manifests: Dict[str, Path] = {}
        if args.known_manifest is not None:
            manifests["catknown"] = args.known_manifest
        if args.novel_manifest is not None:
            manifests["catnovel"] = args.novel_manifest
        missing = [variant for variant in args.variants if variant not in manifests]
        if missing:
            parser.error("missing DREDS manifest(s) for selected variants: " + ", ".join(missing))
        return load_dreds(manifests, args.variants, args.max_samples), None
    if args.dataset == "transpose":
        return load_transpose(args.manifest, args.max_samples), None
    if args.dataset == "ibims":
        return (
            load_ibims(
                args.ibims_root,
                args.levels,
                args.max_samples,
                depth_scale_override=args.depth_scale,
                max_depth_override=args.max_depth,
            ),
            args.ibims_root,
        )
    if args.dataset == "kitti":
        return load_kitti(args.manifest, args.max_samples, args.raw_max_depth), None
    parser.error(f"unsupported dataset: {args.dataset}")


def build_config(args: argparse.Namespace, run_dir: Path) -> RunConfig:
    if args.visualization_max_depth <= args.visualization_min_depth:
        raise ValueError("--visualization-max-depth must be greater than --visualization-min-depth")
    return RunConfig(
        dataset=args.dataset,
        stage=args.stage,
        run_dir=run_dir,
        model_path=args.model_path,
        device=args.device,
        resolution_level=args.resolution_level,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        use_fp16=args.use_fp16,
        apply_mask=args.apply_mask,
        save_visualizations=args.save_visualizations,
        cleanup_predictions=args.cleanup_predictions,
        max_samples=args.max_samples,
        visualization_min_depth=args.visualization_min_depth,
        visualization_max_depth=args.visualization_max_depth,
        intrinsics_path=args.intrinsics_path,
        pointcloud_rot_x_deg=getattr(args, "pointcloud_rot_x_deg", 25.0),
        pointcloud_rot_y_deg=getattr(args, "pointcloud_rot_y_deg", 15.0),
        pointcloud_knn_k=getattr(args, "pointcloud_knn_k", 16),
        pointcloud_knn_std_ratio=getattr(args, "pointcloud_knn_std_ratio", 2.0),
        disable_pointcloud_knn_filter=getattr(args, "disable_pointcloud_knn_filter", False),
        evaluation_seed=args.evaluation_seed,
        load_dav2=args.load_dav2,
        seed=args.seed,
    )


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    run_dir = resolve_run_dir(args, parser)
    collection, ibims_root = build_collection(args, parser)
    config = build_config(args, run_dir)
    layout = run_pipeline(collection, config, ibims_root=ibims_root)
    print(f"Evaluation run completed: {layout.root}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Create the JSONL manifest consumed by the KITTI evaluation pipeline.

The OpenDataLab archive contains the standard directory tree but does not
always ship the JSONL used by AS-Depth. This utility creates the same logical
rows without copying any image or depth files.
"""

import argparse
import json
import os
from pathlib import Path

EXPECTED_VAL_COUNT = 1000


def find_selection_root(dataset_root):
    dataset_root = Path(dataset_root).expanduser().resolve()
    direct_candidates = (
        dataset_root / "depth_selection" / "val_selection_cropped",
        dataset_root / "val_selection_cropped",
        dataset_root,
    )
    nested_candidates = sorted(dataset_root.glob("**/depth_selection/val_selection_cropped"))
    nested_candidates.extend(sorted(dataset_root.glob("**/val_selection_cropped")))
    candidates = tuple(dict.fromkeys((*direct_candidates, *nested_candidates)))
    for candidate in candidates:
        required_dirs = ("image", "velodyne_raw", "groundtruth_depth")
        if all((candidate / name).is_dir() for name in required_dirs):
            return candidate
    raise FileNotFoundError(
        "Could not find KITTI val_selection_cropped directories below "
        f"{dataset_root}; expected image/, velodyne_raw/, and groundtruth_depth/"
    )


def build_manifest(dataset_root, output_path=None, allow_partial=False):
    selection_root = find_selection_root(dataset_root)
    image_dir = selection_root / "image"
    raw_dir = selection_root / "velodyne_raw"
    gt_dir = selection_root / "groundtruth_depth"
    intrinsics_dir = selection_root / "intrinsics"

    image_paths = sorted(image_dir.glob("*.png"))
    if not image_paths:
        raise ValueError(f"No PNG images found in {image_dir}")
    if not allow_partial and len(image_paths) != EXPECTED_VAL_COUNT:
        raise ValueError(
            "KITTI val_selection_cropped must contain exactly "
            f"{EXPECTED_VAL_COUNT} images, found {len(image_paths)}. "
            "Use --allow-partial only for smoke tests."
        )

    output_path = None if output_path is None else Path(output_path).expanduser().resolve()
    path_base = output_path.parent if output_path is not None else selection_root
    rows = []
    missing = []
    for image_path in image_paths:
        image_name = image_path.name
        raw_name = image_name.replace("_sync_image_", "_sync_velodyne_raw_", 1)
        gt_name = image_name.replace("_sync_image_", "_sync_groundtruth_depth_", 1)
        raw_path = raw_dir / raw_name
        gt_path = gt_dir / gt_name
        intrinsics_path = intrinsics_dir / f"{image_path.stem}.txt"
        required = [raw_path, gt_path]
        if not all(path.is_file() for path in required):
            missing.append((image_path, raw_path, gt_path))
            continue

        def manifest_path(path):
            return os.path.relpath(path, path_base)

        row = {
            "rgb": manifest_path(image_path),
            "lidar": manifest_path(raw_path),
            "depth": manifest_path(gt_path),
            "name": image_path.stem,
        }
        if intrinsics_path.is_file():
            row["intrinsics"] = manifest_path(intrinsics_path)
        rows.append(row)

    if missing:
        first = missing[0]
        raise FileNotFoundError(
            "KITTI manifest has missing paired files; first missing row: "
            f"image={first[0]}, raw={first[1]}, gt={first[2]}"
        )
    if len(rows) != len(image_paths):
        raise RuntimeError(f"Only paired {len(rows)} of {len(image_paths)} KITTI images")

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as file:
            for row in rows:
                file.write(json.dumps(row, ensure_ascii=False) + "\n")
    return rows, selection_root


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Prepare a KITTI val_selection_cropped JSONL manifest",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--dataset-root",
        required=True,
        help="KITTI root or depth_selection/val_selection_cropped directory",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Manifest output path; omit with --check-only",
    )
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help="Allow a non-1000-frame manifest for smoke tests",
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Validate pairing and print the frame count without writing",
    )
    return parser.parse_args()


def main():
    args = parse_arguments()
    if args.check_only:
        output = None
    elif args.output:
        output = args.output
    else:
        raise SystemExit("--output is required unless --check-only is set")

    rows, selection_root = build_manifest(
        args.dataset_root,
        output_path=output,
        allow_partial=args.allow_partial,
    )
    print(f"KITTI selection root: {selection_root}")
    print(f"paired frames: {len(rows)}")
    if output:
        print(f"manifest: {Path(output).expanduser().resolve()}")


if __name__ == "__main__":
    main()

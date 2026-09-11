# OMNI-DC evaluation

This directory follows the benchmark layout used by the reference project:

```text
evaluation/
  __main__.py
  cli.py
  core/
    inference.py      # native OMNI-DC adapter and shared inference loop
    io.py             # RGB/depth decoding, units, masks, shape handling
    metrics.py        # shared depth metrics
    output.py         # run schema and output paths
    pipeline.py       # infer/evaluate orchestration
    pointcloud.py
    types.py
    visualization.py
  datasets/
    base.py
    hammer.py
    clearpose.py
    dreds.py
    transpose.py     # current-project dataset, same adapter contract
    ibims.py
    kitti.py
  evaluators/
    depth.py
    ibims_official.py
  scripts/
    run_*.sh
```

The old `infer.py`, `eval.py`, `dataset.py`, `evaluation_ibims/`, root-level
wrappers, fallback output names, and compatibility parameter aliases were
removed. Every route uses the single `uv run python -m evaluation <dataset>` entry
point and writes the same run schema:

```text
outputs/evaluation/<dataset>/<model>_<timestamp>/
  run.json
  predictions/<subset>/.../*.npy
  visualizations/<subset>/.../*_vis.jpg
  metrics/per_sample.csv
  metrics/summary.csv
  metrics/summary.json
  official/<subset>/...                 # iBims only
```

## Environment

Use `uv` for all environment, dependency, and command execution:

```bash
uv sync --extra evaluation --group dev
uv run --extra evaluation python -m evaluation --help
```

The checked-in `pyproject.toml` and `uv.lock` are the dependency contract.
The model adapter uses the repository's native OGNIDC implementation. Server
runs need a CUDA-capable PyTorch installation, the OMNI-DC checkpoint, the
`resnet34.pth`, `pvt.pth`, and (for v1.1) `depth_anything_v2_vitl.pth` weights
at the native paths under `src/`, plus the selected dataset files.

The Mac validation path uses `--stage evaluate` with fixed predictions or
`--max-samples` only. It does not run real OMNI-DC inference locally and does
not weaken the model's CUDA/server requirements.

## Commands

All commands below are run from the repository root and use `uv run`.

```bash
uv run --extra evaluation python -m evaluation hammer \
  --model-path checkpoints/modelv1.1_best_72epochs.pt \
  --manifest data/HAMMER/test_filled_d435.jsonl \
  --camera d435 \
  --intrinsics-path data/HAMMER/intrinsics.txt

uv run --extra evaluation python -m evaluation clearpose \
  --model-path checkpoints/modelv1.1_best_72epochs.pt \
  --manifest data/ClearPose/test.jsonl \
  --intrinsics-path data/HAMMER/intrinsics.txt

uv run --extra evaluation python -m evaluation dreds \
  --model-path checkpoints/modelv1.1_best_72epochs.pt \
  --known-manifest data/DREDS/test_std_catknown.jsonl \
  --novel-manifest data/DREDS/test_std_catnovel.jsonl \
  --intrinsics-path data/HAMMER/intrinsics.txt

uv run --extra evaluation python -m evaluation transpose \
  --model-path checkpoints/modelv1.1_best_72epochs.pt \
  --manifest data/TRansPose/sequences/dc_testset.jsonl \
  --intrinsics-path data/TRansPose/sequences/intrinsics.txt

uv run --extra evaluation python -m evaluation ibims \
  --model-path checkpoints/modelv1.1_best_72epochs.pt \
  --ibims-root data/ibims1

uv run --extra evaluation python evaluation/prepare_kitti_jsonl.py \
  --dataset-root data/KITTI_depth_completion \
  --output data/KITTI_depth_completion/val_selection_cropped.jsonl
uv run --extra evaluation python -m evaluation kitti \
  --model-path checkpoints/modelv1.1_best_72epochs.pt \
  --manifest data/KITTI_depth_completion/val_selection_cropped.jsonl \
  --raw-max-depth 80
```

Use `--stage infer` to write predictions only. Use `--stage evaluate --run-dir
<existing-run>` to score an existing run. `--max-samples N` limits each
subset independently for a smoke test. `--no-save-visualizations` disables
visualizations and permits identity intrinsics only when the model route has
no manifest-provided intrinsics.

## Protocol notes

HAMMER and ClearPose read uint16 depth in millimeters and divide by `1000`.
DREDS reads EXR depth already in meters and allows nearest-neighbor prediction
resize at evaluation time. TRansPose retains the old project's L515 millimeter
input and optional `seq_name` sample IDs. iBims keeps its official MAT evaluator,
per-level split, `480x640` shape, per-sample calibration lookup, and official
metric output. KITTI follows the reference val-selection protocol: PNG values
are divided by `256`, GT zero is invalid, GT has no upper cutoff, raw input is
clipped to `80 m`, and its AS-Depth float32 metric mask is preserved.

The OMNI-DC adapter records `model_class`, depth units, preprocessing, native
padding, `load_dav2`, resolution, device, and seed in `run.json`. It never
passes GT to OGNIDC. No scale/shift alignment is introduced.

## Validation status

Validated on the Mac environment: directory/module structure, Python syntax,
CLI help and dataset/metric imports, manifest parsing, output schema, and
CPU-only fixed prediction/GT scoring. A real model run, CUDA kernels,
checkpoint loading, Depth Anything v2 loading, official iBims script, and full
benchmark remain server-side checks.

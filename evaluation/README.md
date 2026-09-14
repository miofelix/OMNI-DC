# OMNI-DC evaluation

This directory follows the benchmark layout used by the reference project.
See [the alignment contract](ALIGNMENT.md) for shared scoring rules, native
adapter boundaries, validation commands, and interpretation limits.

```text
evaluation/
  __main__.py
  cli.py
  compare_runs.py     # compare metrics after shared benchmark validation
  core/
    inference.py      # native OMNI-DC adapter and shared inference loop
    io.py             # RGB/depth decoding, units, masks, shape handling
    metrics.py        # shared depth metrics
    output.py         # run schema and output paths
    pipeline.py       # infer/evaluate orchestration
    protocol.py       # shared benchmark identity and evaluate-only validation
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

CPU tests use fixed predictions and a mocked OGNIDC forward pass. Actual
inference, including `--max-samples` smoke runs, still requires CUDA.

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
  --manifest data/ClearPose/test.jsonl

uv run --extra evaluation python -m evaluation dreds \
  --model-path checkpoints/modelv1.1_best_72epochs.pt \
  --known-manifest data/DREDS/test_std_catknown.jsonl \
  --novel-manifest data/DREDS/test_std_catnovel.jsonl

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
visualizations. Calibration selection uses the first available per-sample
calibration, then an explicit `--intrinsics-path`, then the native identity-matrix
fallback. The iBims dataset retains its automatically derived
`<ibims-root>/ibims1_core_raw/calib/<sample_id>.txt` metadata, but a missing automatic
calibration file does not prevent inference. An explicitly requested missing
calibration file raises an error. KITTI point-cloud visualization still needs
real camera calibration.

## Protocol notes

HAMMER and ClearPose read uint16 depth in millimeters and divide by `1000`.
DREDS reads EXR depth already in meters and allows nearest-neighbor prediction
resize at evaluation time. TRansPose retains the old project's L515 millimeter
input and optional `seq_name` sample IDs. iBims keeps its official MAT evaluator,
per-level split, `480x640` shape, per-sample calibration lookup, and official
metric output. KITTI follows the reference val-selection protocol: PNG values
are divided by `256`, GT zero is invalid, GT has no upper cutoff, raw input is
clipped to `80 m`, and its AS-Depth float32 metric mask is preserved.
The adapter removes native padding, verifies the raw-depth grid, and stores
float32 predictions without filtering their values. This follows `src/main.py`,
which sends the cropped model output directly to `src/metric/dcmetric.py`.
The old evaluation adapter incorrectly converted zero predictions to `NaN`
before scoring: for predictions `[0, 1]` against GT `[1, 1]`, that changed KITTI
MAE from `0.5` to `0`. The regression test locks in the corrected result.

Scoring masks remain dataset protocol choices. Per-sample results report valid
GT pixels, scored pixels, and coverage so excluded predictions remain visible.
KITTI uses the shared AS-Depth scorer and keeps finite zero/negative predictions
at valid GT pixels. The common
indoor scorer retains its positive-prediction mask; that scoring rule is not a
requirement imposed by OGNIDC. Adapter postprocessing is recorded under
`results.inference.prediction_postprocessing`.

The OMNI-DC adapter records `model_class`, depth units, preprocessing, native
padding, `load_dav2`, resolution, device, and seed in `run.json`. It never
passes GT to OGNIDC. No scale/shift alignment is introduced.

The evaluation layout, data contracts, metrics, output schema, pipeline stages,
and comparison rules align with `lingbot-depth/evaluation`. Project adaptations
remain wherever needed: model preprocessing/postprocessing, CLI/config options,
dataset metadata, environment setup, and the additional TRansPose dataset.
Alignment does not require identical source files. `--resolution-level 1|2|3`,
`--load-dav2`, CUDA execution, ImageNet RGB normalization, native padding/cropping,
and `--seed` retain their OMNI-DC meanings; the resolution value is not a
cross-model resolution target.

Each run records `benchmark.sha256`, selected sample identities, manifest hashes,
depth bounds, scoring-mask rules, and metric aggregation in `run.json`.
Comparisons should select identical RGB/raw/GT paths (symlinks resolve), subsets,
frames, and raw-depth bounds, and match `benchmark.sha256`. Equivalent manifests
may retain project-specific metadata.
Manifest hashes record provenance; model-only metadata or manifest formatting
does not change the shared comparison identity. The identity records resolved
sample paths, but does not hash all RGB/depth files, so keep those files unchanged
during comparison.
For iBims, it also includes official script hashes and `--evaluation-seed`
(default `0`), which is independent of the OMNI-DC inference `--seed`.
The official evaluator retains OMNI-DC's `uv run --project <OMNI-DC root>
--extra evaluation python` environment selection, while the launcher seeds Python
and NumPy without modifying the official scoring script.

Evaluate-only runs validate the benchmark identity against the inference run.
Changing the selected inputs/GT, depth bounds, official script, or iBims
evaluation seed requires a new run. Original model configuration remains in
`config`; evaluate-only options are recorded separately in `evaluation_config`.
Older runs without benchmark identity must be regenerated for this protocol.

Compare completed runs from any of the aligned repositories with:

```bash
uv run --extra evaluation python -m evaluation.compare_runs \
  /path/to/lingbot-run /path/to/omni-run --output comparison.csv
```

The comparison command refuses mismatched benchmark identities or incomplete
evaluations. Each model keeps its own preprocessing and postprocessing; consult
its inference metadata when interpreting results.

## Validation status

Run the CPU contract tests with:

```bash
uv run --extra evaluation --group dev python -m pytest tests -q
```

The adapter tests cover unfiltered native predictions, an end-to-end KITTI
zero-prediction regression, raw clipping, ImageNet normalization, native
padding/cropping, no GT in model inputs, optional iBims calibration and its
priority, uv project environment selection and the seeded official launcher,
TRansPose, and independent inference/evaluation seeds. Real checkpoint loading,
CUDA kernels,
Depth Anything v2, the complete official iBims dataset/script, and full benchmark
inference still need validation on the server.

Native contract references: `src/main.py` defines right/bottom padding and output
cropping; `src/data/uniformat.py` and `src/data/ibims.py` define ImageNet RGB
normalization; `src/data/ibims.py` supplies identity K, and
`src/model/convgru.py::BasicUpdateBlock.forward` currently accepts but does not
use K. The adapter still forwards available calibration so that this interface
remains intact. No CUDA checkpoint run was used to infer these contracts.

## Native forward performance benchmark

Independent parameter counting and CUDA forward FPS measurement are available through
`python -m evaluation.benchmark`. See [the standalone benchmark guide](benchmark/README.md)
for the fixed synthetic-input protocol, Mac checks, and the DGX Spark environment.

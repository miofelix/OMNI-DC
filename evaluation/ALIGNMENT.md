# Evaluation alignment contract

Alignment follows model interfaces, dataset requirements and reproducible tests.
The existing LingBot evaluator supplies the initial structure, but its code and
the other repositories' old wrappers are both subject to review.

## Shared structure

```text
evaluation/
  cli.py, __main__.py         dataset commands and infer/evaluate/all stages
  datasets/                  manifests, sample selection, units and depth bounds
  core/types.py              EvaluationSample, RunConfig and LoadedSample
  core/inference.py          each project's model/runtime adapter
  core/io.py                 depth I/O and explicit restoration helpers
  core/protocol.py            scoring identity and input provenance
  core/output.py             float32 metric NPY, run.json, CSV/JSON results
  core/pipeline.py            stage lifecycle and benchmark validation
  evaluators/                common depth scoring and official iBims integration
  compare_runs.py             compare compatible runs and disclose adapter settings
  check_alignment.py          interface and numerical contract checks
```

Shared interfaces permit extra configuration, dataset metadata, functions and
visualizations. Source-file or AST equality is not a compatibility requirement.

## Boundaries

| Area | Common evaluation contract | Model/project adaptation |
| --- | --- | --- |
| Data | Same selected RGB/raw/GT samples, subsets, units and evaluation bounds | RGB decoder, calibrated metadata, additional datasets |
| Model input | Metric raw depth comes from the declared input; GT is not passed to inference | Resize/padding, tensor layout, normalization, sparse prompt selection, intrinsics, model depth/disparity transforms |
| Runtime | Consistent stage entry points and output records | CUDA requirements, library versions, AMP, seeds, loaders and uv launch settings |
| Predictions | float32 metric depth; evaluator receives the adapter's documented output | Native tensor conversion, spatial restoration, required invalid-value handling |
| Scoring | Explicit profile, GT range/mask, thresholds, aggregation and official evaluator identity | No silent model-specific replacement of the selected scoring formula |
| Reproducibility | Sample/scoring identity and immutable dataset paths | Separate model config, per-sample metadata, full manifest provenance and preprocessing details |

## Verified differences and corrections

- **LingBot:** `MDMModel.infer` keeps its resolution-level, mask and mixed-precision
  options. Its existing KITTI restoration retains finite nonpositive predictions.
- **InfiniDepth:** PIL/EXIF decoding agrees with native `load_image`; raw-depth
  prompting retains native resizing, strict prompt bounds, sampling seeds and
  raw-derived WarpMedian depth/disparity conversion. Native-source tests exercise
  the image path and a metric-depth round trip. TransCG/TRansPose remain supported.
- **Prior-Depth-Anything:** the native sampler accepts RGB and prior maps of
  different sizes. The wrapper now permits this and restores the RGB output grid.
  Native `infer_one_sample` returns a Tensor; explicit detach/float/CPU conversion
  fixes the old CUDA-to-NumPy failure. Model variants and sparse sampler options
  remain independent of the other projects.
- **PromptDA:** its native patch embedding requires dimensions divisible by 14;
  prompt and RGB sizes need not match. RGB preparation and prompt handling remain
  separate. Restoration onto the raw-depth grid is an evaluation-adapter choice,
  not a claim about the model's native output grid. TRansPose previews remain.
- **OMNI-DC:** ImageNet input normalization, padding/cropping and model loading
  follow the native implementation. Native scoring counts a zero prediction at
  valid GT; the old wrapper's conversion of zero to NaN could erase its error.
  The corrected adapter preserves native output values for the selected scorer.
  Camera metadata and project runtime settings remain supported.

Each project retains its own official-evaluator process launcher while executing
the same selected iBims scoring code with an explicit evaluation seed. Common
benchmark identity excludes model-only manifest fields; their hashes and original
metadata remain available for provenance.

## Scoring interpretation and verification limits

The indoor depth profile uses the existing positive prediction/GT intersection,
float64 errors and strict float32 delta ratios. KITTI uses the explicit
`asdepth_kitti_legacy_v1` compatibility profile: /256 PNG units, independent raw
80 m cutoff, unbounded positive GT, float32 scoring and finite-prediction masks.
Its ratio arithmetic, including behavior on negative predictions, is retained as
an identified legacy profile. These six metrics are not a claim of official KITTI
leaderboard equivalence. iBims delegates to the dataset-supplied evaluator.

Model postprocessing can affect which predictions reach a scorer. Per-sample and
overall coverage records expose exclusions; compare accuracy together with these
records and the adapter settings. This does not equate preprocessing choices or
make a benchmark fair solely because directory structures match.

The compatibility checker exercises interfaces, units, invalid pixels, resize,
threshold precision, aggregation and identity under each project's Python
environment. Model-adapter tests additionally check selected native interfaces.
They do not replace CUDA checkpoint inference and a complete benchmark run.
Full RGB/raw/GT file content is not hashed, so datasets must remain immutable.

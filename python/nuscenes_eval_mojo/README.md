# nuscenes-eval-mojo

Fast nuScenes 3D object **detection evaluation** — a drop-in-shaped
replacement for the `DetectionEval` accumulate path of
[`nuscenes-devkit`](https://pypi.org/project/nuscenes-devkit/) — powered by a
clean-room Mojo kernel, with a vendored pure-Python fallback for platforms
without a native build (including Windows).

The official evaluator re-runs a Python-level greedy matching loop for every
class × distance-threshold pair (10 classes × 4 thresholds — a ×40 fan-out),
with a NumPy scalar norm per candidate box pair. This package runs the whole
fan-out in one compiled kernel call. Loading, filtering, and all AP/TP
post-processing stay in shared Python/NumPy code, so both backends produce
identical results.

```python
import nuscenes_eval_mojo

metrics = nuscenes_eval_mojo.evaluate(gt_corpus, submission_json)
metrics["mean_ap"]    # mAP
metrics["nd_score"]   # NDS
metrics["tp_errors"]  # mATE / mASE / mAOE / mAVE / mAAE
```

- **Same values**: the returned dict has the
  `DetectionMetrics.serialize()` schema (`label_aps`, `mean_dist_aps`,
  `mean_ap`, `label_tp_errors`, `tp_errors`, `tp_scores`, `nd_score`,
  `eval_time`, `cfg`) and matches the oracle's full metrics dict within 1e-8
  on every corpus in the differential battery (measured agreement: 8.9e-16 on
  the benchmark corpora, ulp-level). Both the native and the pure-Python fallback backends are
  tested against the oracle.
- **Same semantics**: CVPR 2019 detection config by default (class ranges
  50/50/50/50/50/40/40/40/30/30, dist_ths 0.5/1/2/4, dist_th_tp 2.0,
  min_recall/min_precision 0.1, 500 boxes/sample, mAP weight 5); greedy
  center-distance matching in confidence order (ties by descending insertion
  index); ego-distance / zero-points / bike-rack filtering; the class-specific
  TP NaN rules (traffic_cone: attr/vel/orient, barrier: attr/vel); NaN-aware
  cumulative-mean error resampling.
- **Much faster**: 6.2x–7.0x warm, 13.8x–68.5x cold, depending on corpus size
  (Apple M4 Max; full method and numbers below). The remaining time on our
  side is dominated by Python input validation/grouping, not the kernel.
- **No toolchain needed**: per-platform wheels ship the compiled kernel,
  self-contained (the Mojo runtime is vendored in). Everywhere else the same
  API transparently runs on the vendored fallback.
- Force the fallback with `NUSCENES_EVAL_MOJO_DISABLE_NATIVE=1`; inspect the
  active backend with `nuscenes_eval_mojo.backend_info()`.

## Install

```
pip install nuscenes-eval-mojo
```

Per-platform wheels (macOS arm64, Linux x86_64) carry the native kernel. On
any other platform — including Windows — the same wheel API runs on the
vendored pure-Python fallback, silently and correctly. There is no sdist: a
source tarball cannot rebuild the native library. The only runtime dependency
is NumPy.

## API

### `evaluate(gt, predictions, config=None) -> dict`

- `predictions`: the **nuScenes submission JSON** shape (a dict, or a path to
  a JSON file): `{"meta": {...}, "results": {sample_token: [box, ...]}}` with
  `translation`/`size`/`rotation`/`velocity`/`detection_name`/
  `detection_score`/`attribute_name` per box (plus the box's `sample_token`;
  `ego_translation`/`num_pts` optional, exactly like the devkit's
  deserializer).
- `gt`: ground truth in **extracted-annotation form** (a dict, or a path):
  `{"samples": [{"token", "ego_translation", "annotations": [...]}]}` where an
  annotation carries `category_name`, `translation`, `size`, `rotation`,
  `velocity` (NaN allowed), `num_lidar_pts`, `num_radar_pts`, and an optional
  `attribute_name`. Category names are mapped to the 10 detection classes
  (`vehicle.car` → `car`, …); unmapped categories are dropped and
  `static_object.bicycle_rack` annotations are used for bike-rack filtering —
  the same fields `load_gt()` pulls out of a NuScenes DB.
- `config`: optional `DetectionConfig` (defaults to the CVPR 2019
  configuration used by the official eval).

Invalid input fails fast with a structured `EvaluationError` (`.code` /
`.details`): mismatched sample-token sets, more than `max_boxes_per_sample`
predictions, unknown `detection_name`/`attribute_name`, NaN scores or
translations, non-positive sizes, empty ground truth.

```python
import nuscenes_eval_mojo as ne

info = ne.backend_info()       # native_available, native_source, abi versions, ...
ne.native_available()          # bool
cfg = ne.DetectionConfig.default()
```

## Benchmarks

Measured on this machine (date: 2026-09-19; macOS arm64, Apple M4 Max; Python
3.12, NumPy 1.26.4, Mojo 1.1.0; oracle nuscenes-devkit 1.2.0) with
`benchmarks/bench_nuscenes_eval.py` on deterministic seeded synthetic corpora
— correctness against the oracle is asserted before timing. Both sides run the
full post-DB-load evaluation chain (box construction, filtering, ×40 matching,
AP/TP assembly); nuScenes DB loading is excluded on both sides (it is out of
scope for the kernel). "Cold" is the very first call in a fresh interpreter
(median of 5 process launches; imports already done — for nuscenes-eval-mojo
this includes the one-time `dlopen` + ABI handshake of the native kernel).
"Warm" is the steady-state per-evaluation latency (median of 5 runs).

Cold first evaluation (median of 5 fresh processes, imports done, first call
timed — includes the one-time `dlopen` + ABI handshake on our side):

| corpus | samples | DetectionEval path (s) | nuscenes_eval_mojo (s) | speedup |
|---|---:|---:|---:|---:|
| small | 60 | 1.896 | 0.028 | 68.5x |
| medium | 300 | 2.304 | 0.089 | 25.8x |
| large | 1,200 | 4.024 | 0.292 | 13.8x |

Warm steady state (s per full evaluation, median of 5 runs):

| corpus | samples | DetectionEval path (s) | nuscenes_eval_mojo (s) | speedup |
|---|---:|---:|---:|---:|
| small | 60 | 0.088 | 0.014 | 6.2x |
| medium | 300 | 0.443 | 0.064 | 7.0x |
| large | 1,200 | 1.630 | 0.242 | 6.7x |

Corpus shapes (small/medium/large): 60 / 300 / 1,200 samples with 8–20 gt
annotations and ~25–30 predictions per sample across all 10 classes.

## Correctness

The differential suite (`tests/test_nuscenes_eval_differential.py`) compares
the full metrics dict against the published `nuscenes-devkit` 1.2.0 package on
seeded synthetic corpora plus adversarial cases: exact confidence ties,
classes with zero gt (`npos == 0`), classes with gt but zero predictions,
classes with predictions but zero matches, heavy false-positive rates,
bike-rack filtering, NaN velocities, missing attributes, duplicate detections,
and a custom `DetectionConfig`. Gate: full-dict agreement within `atol=1e-8`;
measured agreement is
8.9e-16 on the benchmark corpora (ulp-level; the residual comes from the
quaternion→yaw closed form vs pyquaternion's BLAS matrix product). The whole
suite runs twice — native backend and forced fallback
(`NUSCENES_EVAL_MOJO_DISABLE_NATIVE=1`). The two backends agree with each
other to ~1 ulp: compiled code (the Mojo kernel, like the oracle's own
NumPy/BLAS paths) may contract `a*b+c` into a fused multiply-add, which
CPython bytecode cannot.

The oracle harness drives the devkit's own code
(`EvalBoxes`/`DetectionBox`, `add_center_dist`, `filter_eval_boxes`,
`accumulate`, `calc_ap`/`calc_tp`, `DetectionMetrics`) — the exact steps
`DetectionEval.evaluate()` runs after DB loading.

## Unsupported scope (honest list)

- **nuScenes DB loading** (`NuScenes(...)` table parsing, split resolution,
  `nusc.box_velocity` instance-tracking velocity estimation, attribute-token
  maps): out of scope. Ground truth is accepted in the extracted-annotation
  form above; the differential suite covers the equivalence of everything
  downstream of that extraction.
- **Other eval tasks**: tracking, prediction, panoptic/segmentation, lidarseg
  — not covered. Only 3D detection (`DetectionBox`, `boxes_3d`).
- **Only `dist_fcn='center_distance'`** (the official config's distance
  function) is supported.
- **Rendering** (`class_pr_curve`, `summary_plot`, sample visualizations,
  `metrics_details.json` output): not covered; the returned dict is the
  metrics summary only.
- `eval_time` is measured wall-clock and is not part of the parity comparison
  (it cannot match by definition).

## Development

```
# build the Mojo kernel (needs the repo pixi environment)
pixi run bash kernels/nuscenes-eval/build.sh

# oracle for the differential suite (pinned; installed into an unmanaged
# directory because `pixi run` prunes pip installs from the pixi env)
pixi run python -m ensurepip --upgrade
pixi run python -m pip install --target .oracle-nuscenes-eval "nuscenes-devkit==1.2.0"

# differential + loader tests, native then forced fallback
PYTHONPATH="python/nuscenes_eval_mojo:.oracle-nuscenes-eval:tests" PYTHONNOUSERSITE=1 \
  pixi run bash scripts/test_all_nuscenes_eval.sh

# benchmark (refuses to run on the fallback backend)
PYTHONPATH="python/nuscenes_eval_mojo:.oracle-nuscenes-eval:tests" PYTHONNOUSERSITE=1 \
  pixi run python benchmarks/bench_nuscenes_eval.py

# self-contained platform wheel (delocate/auditwheel repair)
PYTHONNOUSERSITE=1 pixi run bash python/nuscenes_eval_mojo/build_wheel.sh
```

Source, benchmarks, and development: <https://github.com/thyn-ai/mojo-kernels>

License: Apache-2.0, © 2026 Algenta

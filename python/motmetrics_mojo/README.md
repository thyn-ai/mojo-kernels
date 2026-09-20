# motmetrics-mojo

Fast MOTChallenge tracking metrics — **MOTA, MOTP, IDF1, IDP, IDR, MOSTLY_TRACKED /
PARTIALLY_TRACKED / MOSTLY_LOST**, plus precision, recall, fragmentation and the
supporting counts — matching [`motmetrics`](https://pypi.org/project/motmetrics/)
(py-motmetrics, `cheind/py-motmetrics`) value-for-value, powered by a clean-room
Mojo kernel, with a vendored pure-Python fallback for platforms without a native
build (including Windows).

```python
import motmetrics_mojo as mm

acc = mm.MOTAccumulator(auto_id=True)          # same call shape as motmetrics
acc.update([1, 2], [10, 20], [[0.1, 0.9], [0.8, 0.2]])
acc.update([1, 2], [10, 20], [[0.2, 0.9], [0.9, 0.1]])
acc.update([1], [20], [[0.3]])                 # an ID switch

s = mm.compute(acc, metrics=["mota", "motp", "idf1", "num_switches"])
s["mota"], s.motp, s.idf1, s.num_switches      # 0.8, 0.18, 1.0, 1
```

- **Same values**: metrics match the oracle (1.4.0, default scipy assignment
  solver) on the differential battery — every count *exactly* (event counts,
  MOSTLY_*, IDF1/IDP/IDR, MOTA, precision, recall are integer-derived and
  compared exactly), MOTP within 1e-10 (measured agreement over the battery:
  6.2e-15; its distance sum is the only floating-point accumulation). Both the
  native and the pure-Python fallback backends are tested against the oracle,
  and the two backends agree with each other bit-for-bit.
- **Same event semantics**: established tracks are carried forward per frame
  (object order), the remainder is assigned minimum-cost with the same
  scan/tie-breaking order as the reference solver (so even tie-heavy integer
  distance matrices produce identical assignments), and MATCH / SWITCH /
  TRANSFER / ASCEND / MIGRATE / MISS / FP events follow the reference rules,
  including `max_switch_time`.
- **Much faster**: 37-48x on 100-2,000-frame streams, cold and warm alike
  (Apple M4 Max; full method and numbers below). The kernel is a single-pass
  event-stream accumulator: it never materializes the per-frame pandas event
  table that the oracle spends its time on.
- **No toolchain needed**: per-platform wheels ship the compiled kernel,
  self-contained (the Mojo runtime is vendored in). Everywhere else the same
  API transparently runs on the vendored fallback.
- Force the fallback with `MOTMETRICS_MOJO_DISABLE_NATIVE=1`; inspect the
  active backend with `motmetrics_mojo.backend_info()`.

## Install

```
pip install motmetrics-mojo
```

Per-platform wheels (macOS arm64, Linux x86_64) carry the native kernel. On
any other platform — including Windows — the same wheel API runs on the
vendored pure-Python fallback, silently and correctly. There is no sdist: a
source tarball cannot rebuild the native library. The only runtime dependency
is NumPy; unlike the oracle, pandas and scipy are **not** required.

## API

`MOTAccumulator(auto_id=False, max_switch_time=float("inf"))` mirrors the
oracle's constructor. `update(oids, hids, dists, frameid=None)` takes one frame
of integer object ids, integer hypothesis ids, and their `len(oids) x
len(hids)` pairwise distance matrix (NaN and +/-inf entries signal do-not-pair
constellations), and returns the frame id used. `reset()` empties the
accumulator. `compute(acc, metrics=None)` returns a `MetricSummary` — an
ordered mapping with attribute access (`.mota`) and `.to_dict()` — instead of
the oracle's one-row pandas DataFrame (this package has no pandas dependency).
`metrics=None` computes the oracle's `motchallenge_metrics` set in the same
order; a single string is treated as a one-element list.

| function | signature | oracle equivalent |
|---|---|---|
| accumulator | `MOTAccumulator(auto_id=False, max_switch_time=inf)` | `mm.MOTAccumulator(auto_id=..., max_switch_time=...)` |
| update | `acc.update(oids, hids, dists, frameid=None)` | `acc.update(oids, hids, dists, frameid=...)` |
| compute | `compute(acc, metrics=None)` | `mm.metrics.create().compute(acc, metrics=...)` |

Supported metrics: the full MOTChallenge set (`idf1`, `idp`, `idr`, `recall`,
`precision`, `num_unique_objects`, `mostly_tracked`, `partially_tracked`,
`mostly_lost`, `num_false_positives`, `num_misses`, `num_switches`,
`num_fragmentations`, `mota`, `motp`, `num_transfer`, `num_ascend`,
`num_migrate`) plus `num_frames`, `num_objects`, `num_predictions`,
`num_matches`, `num_detections`, `idtp`, `idfp`, `idfn`.

## Benchmarks

Measured on this machine (date: 2026-09-19; macOS-26.6.2 arm64, Apple M4 Max;
Python 3.12.14, NumPy 2.5.3, Mojo 1.1.0; oracle motmetrics 1.4.0 on
numpy 2.5.3 / pandas 3.0.6 / scipy 1.18.1, default scipy solver path) with
`benchmarks/bench_motmetrics.py` — correctness against the oracle is asserted
before timing. A *pipeline* is what users actually time: accumulate a whole
synthetic MOTChallenge-style stream (seeded, deterministic: births/deaths, ID
switches, misses, false positives, NaN do-not-pair entries) frame by frame,
then compute the 18 MOTChallenge metrics. "Cold" is the full pipeline in a
fresh interpreter (median of 5 process launches; for motmetrics-mojo this
includes the one-time `dlopen` + ABI handshake of the native kernel). "Warm"
is the steady-state per-pipeline latency (median of 5 batches of 3 pipelines,
fresh accumulator each).

Cold first full pipeline (accumulate + compute; 1,000 frames x 40 objects x 44 hypotheses, median of 5 fresh processes):

| pipeline | py-motmetrics (s) | motmetrics-mojo (s) | speedup |
|---|---:|---:|---:|
| accumulate + compute | 2.055 | 0.0551 | 37.3x |

Warm steady state (seconds per full pipeline, median of 5 batches of 3):

| frames | objects | hypotheses | py-motmetrics (s) | motmetrics-mojo (s) | speedup |
|---:|---:|---:|---:|---:|---:|
| 100 | 10 | 11 | 0.033 | 0.00086 | 38.4x |
| 500 | 25 | 27 | 0.456 | 0.00997 | 45.7x |
| 1,000 | 40 | 44 | 2.000 | 0.04839 | 41.3x |
| 2,000 | 80 | 88 | 20.473 | 0.42267 | 48.4x |

## Correctness

The differential suite (`tests/test_motmetrics_differential.py`) compares
against the published `motmetrics` 1.4.0 package (default scipy
assignment-solver path) on seeded synthetic streams plus adversarial cases:
tie-heavy integer distance matrices, carry-forward vs. globally-better
reassignment, all-NaN matrices, empty frames, one-sided frames, negative and
+/-inf distances, fragmentation histories (including the trailing-miss edge
and the exactly-0.2 / exactly-0.8 MOSTLY boundaries), explicit frame ids, and
`max_switch_time` variants (0, 1, 3, inf). Gate: exact equality on every
count and integer-derived ratio (MOTA, precision, recall, IDP, IDR, IDF1) and
`atol=1e-10` on MOTP; measured agreement over the battery is 6.2e-15 (MOTP
only; the oracle's pairwise summation can differ from the kernel's sequential
one by ~1 ulp). The whole suite runs twice — native backend and forced
fallback (`MOTMETRICS_MOJO_DISABLE_NATIVE=1`) — and the two backends agree
with each other bit-for-bit, counters included.

IDF1 note: in the oracle's global-assignment construction, every optimal
assignment satisfies `idfp = num_predictions - idtp` and
`idfn = num_objects - idtp`, and `idtp` (the maximum total co-occurrence
count over object/hypothesis matchings) is unique even when the optimal
matching is not. The kernel therefore computes IDF1 as a max-weight matching
over the sparse co-occurrence counts, which is exactly the oracle's value
without building the (objects + hypotheses)^2 dense matrix.

Out of scope (oracle features not exposed):

- the per-frame event table (`acc.events`, `acc.mot_events`) and the metrics
  that need it: `track_ratios`, `obj_frequencies`, `pred_frequencies`,
  `id_global_assignment` (requesting one raises `NotImplementedError`);
- non-integer object/hypothesis/frame ids (the oracle accepts arbitrary
  hashables; MOTChallenge uses integers — this package accepts signed 64-bit
  integer ids);
- repeated frame ids across `update` calls (the oracle deduplicates them in
  `num_frames` and the IDF1 occurrence bases; frame ids must be unique per
  call here, as MOTChallenge loaders produce);
- alternative LAP solvers (`lapsolver`, `ortools`, `munkres`), the
  `motmetrics.distances` / `motmetrics.io` / `motmetrics.utils` helper
  modules, `preprocess_result`, and the `ana` caching argument of
  `MetricsHost.compute`.

## Development

```
# build the Mojo kernel (needs the repo pixi environment)
pixi run bash kernels/motmetrics/build.sh

# oracle for the differential suite (pinned; installed into an unmanaged
# directory because `pixi run` prunes pip installs from the pixi env)
pixi run python -m ensurepip --upgrade
pixi run python -m pip install --target .oracle-motmetrics \
  "motmetrics==1.4.0" "numpy==2.5.3" "pandas==3.0.6" "scipy==1.18.1"

# differential + loader tests, native then forced fallback
PYTHONPATH="python/motmetrics_mojo:.oracle-motmetrics" PYTHONNOUSERSITE=1 \
  pixi run bash scripts/test_all_motmetrics.sh

# benchmark (refuses to run on the fallback backend)
PYTHONPATH="python/motmetrics_mojo:.oracle-motmetrics" PYTHONNOUSERSITE=1 \
  pixi run python benchmarks/bench_motmetrics.py

# self-contained platform wheel (delocate/auditwheel repair)
PYTHONNOUSERSITE=1 pixi run bash python/motmetrics_mojo/build_wheel.sh
```

Source, benchmarks, and development: <https://github.com/thyn-ai/mojo-kernels>

License: Apache-2.0, © 2026 Algenta

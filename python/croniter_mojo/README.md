# croniter-mojo

A drop-in faster replacement for the [`croniter`](https://pypi.org/project/croniter/)
package's `get_next` / `get_prev` on 5-field cron expressions, powered by a
clean-room Mojo kernel — with a pure-Python engine for platforms without a
native build (including Windows — tested there in CI: [`windows-fallback`](https://github.com/thyn-ai/mojo-kernels/actions/workflows/windows-fallback.yml)). Zero runtime dependencies.

```python
from datetime import datetime
from zoneinfo import ZoneInfo
from croniter_mojo import get_next, get_prev  # same results as croniter

get_next("0 9 * * mon-fri", datetime(2026, 9, 19, 14, 30))
# datetime(2026, 9, 21, 9, 0)

get_prev("30 2 * * *", datetime(2026, 3, 10, 12, 0, tzinfo=ZoneInfo("America/New_York")))
# datetime(2026, 3, 10, 2, 30, tzinfo=ZoneInfo("America/New_York"))  (DST-aware)
```

- **Same results**: returned datetimes are identical to the published
  `croniter` package (6.2.4) for the supported scope — the differential
  suite compares thousands of cases element-for-element on both the native
  and fallback backends, including `expand_from_start_time` (a no-op in
  croniter 6.2.4) and zoneinfo DST boundaries (spring-forward gaps,
  fall-back folds, sub-hour Lord Howe shifts).
- **Supported scope**: `*`, `*/n`, `a-b` (wrapping allowed), lists, steps
  `a/n` and `a-b/n`, names (`jan`/`mon`, case-insensitive), `?` in the
  day-of-month/day-of-week fields, `l` (last day of month) in dom lists,
  nth-weekday hash specs (`5#3`), dom/dow OR-vs-AND (`day_or=True/False`),
  `start_date` truncation semantics, and naive or zoneinfo-aware datetimes.
  Not supported: 6/7-field expressions (seconds/year fields) and
  `ret_type=float` — this package always returns datetimes.
- **Much faster iteration**: native bitmask seek instead of Python-level
  datetime arithmetic for every candidate minute — see the benchmark below.
- **No toolchain needed**: per-platform wheels ship the compiled kernel.
  Everywhere else the package transparently uses its pure-Python engine,
  which returns identical results.
- Force the fallback with `CRONITER_MOJO_DISABLE_NATIVE=1`; inspect the
  active backend with `croniter_mojo.backend_info()`.

## Benchmark

Measured on this machine (Apple M4 Max, macOS arm64, Python 3.12, Mojo 1.1.0,
croniter 6.2.4), median of 5 runs. "Cold" = first call for an expression
(parse + seek, no cache); "warm" = steady-state per-call cost with the parsed
schedule cached. Chains = 1000 successive `get_next` calls (scheduler
roll-out workload). Correctness is asserted before any timing.

| workload | oracle croniter (us/call) | croniter_mojo native (us/call) | speedup |
|---|---:|---:|---:|
| cold get_next: every-5-min | 66.5 | 3.0 | 21.87x |
| cold get_next: daily-weekday | 61.5 | 3.3 | 18.69x |
| cold get_next: mixed-wrap | 116.8 | 2.5 | 45.96x |
| cold get_next: nth-weekday | 63.8 | 3.7 | 17.39x |
| cold get_next: sparse-feb29 | 59.5 | 3.9 | 15.34x |
| warm get_next: every-5-min | 64.08 | 2.67 | 24.03x |
| warm get_next: daily-weekday | 56.50 | 2.63 | 21.52x |
| warm get_next: mixed-wrap | 118.83 | 2.58 | 46.01x |
| warm get_next: nth-weekday | 60.46 | 3.50 | 17.27x |
| warm get_next: sparse-feb29 | 55.46 | 3.62 | 15.30x |
| chain next:every-5-min | 79.46 | 2.33 | 34.06x |
| chain next:daily-weekday | 61.91 | 2.35 | 26.37x |
| chain next:mixed-wrap | 157.68 | 3.82 | 41.23x |
| chain next:nth-weekday | 104.42 | 3.44 | 30.34x |
| chain next:sparse-feb29 | 110.49 | 4.30 | 25.72x |
| chain prev:*/15 | 52.76 | 2.63 | 20.03x |
| chain next:*/15 tz-aware | 75.17 | 11.78 | 6.38x |
| chain next:daily-0130 tz-DST | 70.59 | 15.72 | 4.49x |

The pure-Python fallback returns identical results; the native kernel is the
speedup above. Full method: `benchmarks/bench_croniter.py` in the repository.

Source, benchmarks, and development: <https://github.com/thyn-ai/mojo-kernels>

License: Apache-2.0, © 2026 Algenta

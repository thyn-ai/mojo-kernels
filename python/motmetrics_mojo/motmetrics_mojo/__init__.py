"""motmetrics-mojo: fast MOTChallenge metrics (MOTA / MOTP / IDF1 / MOSTLY_*),
drop-in-shaped for `motmetrics` (py-motmetrics).

Same accumulator/update call shape, same metric values — powered by a Mojo
kernel where the platform supports it (macOS arm64, Linux x86_64), with a
vendored pure-Python fallback everywhere else (including Windows).

    import motmetrics_mojo as mm

    acc = mm.MOTAccumulator(auto_id=True)
    acc.update([1, 2], [10, 20], [[0.1, 0.9], [0.8, 0.2]])
    acc.update([1, 2], [10, 20], [[0.2, 0.9], [0.9, 0.1]])
    summary = mm.compute(acc, metrics=["mota", "motp", "idf1"])
    summary["mota"], summary.motp       # 1.0, 0.15

Set MOTMETRICS_MOJO_DISABLE_NATIVE=1 to force the pure-Python fallback.
"""

from motmetrics_mojo._native import backend_info, native_available
from motmetrics_mojo.core import (
    MOTCHALLENGE_METRICS,
    EXTRA_METRICS,
    UNSUPPORTED_METRICS,
    MetricSummary,
    MOTAccumulator,
    compute,
)

__version__ = "0.1.2"  # x-release-please-version
__all__ = [
    "MOTAccumulator",
    "MetricSummary",
    "MOTCHALLENGE_METRICS",
    "EXTRA_METRICS",
    "UNSUPPORTED_METRICS",
    "compute",
    "backend_info",
    "native_available",
    "__version__",
]

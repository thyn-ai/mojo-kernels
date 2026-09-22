"""statistics-mojo: a drop-in faster replacement for the stdlib `statistics` hot paths.

Same functions, same call shapes, same values and types — powered by a Mojo
kernel where the platform supports it (macOS arm64, Linux x86_64), with a
pure-Python fallback (delegation to the stdlib module itself) everywhere
else, including Windows.

    import statistics_mojo as stats

    stats.mean([1, 2, 3, 4])                    # 2.5 (float, like the stdlib)
    stats.mean([1, 2, 3])                       # 2 (int, like the stdlib)
    stats.variance([2.75, 1.75, 1.25, 0.25])    # exact-rational result
    stats.stdev(big_float_list)                 # correctly rounded, 50x+
    stats.quantiles(data, n=100)                # percentiles
    stats.median(data)                          # radix-sorted median

    # Column batch API (one kernel pass over many columns):
    stats.variance_batch([col_a, col_b, col_c])
    stats.mean_batch([col_a, col_b, col_c])

Results are bit-identical to CPython 3.12 `statistics` for every supported
input, including exact-rational semantics for int inputs, float64
bit-exactness for float inputs, Decimal/Fraction exactness (via the stdlib
path), and identical StatisticsError messages. Set
STATISTICS_MOJO_DISABLE_NATIVE=1 to force the pure-Python fallback.
"""

from statistics_mojo._native import backend_info, native_available
from statistics_mojo.core import (
    StatisticsError,
    fmean,
    fmean_batch,
    mean,
    mean_batch,
    median,
    median_batch,
    median_high,
    median_low,
    mode,
    multimode,
    pstdev,
    pstdev_batch,
    pvariance,
    pvariance_batch,
    quantiles,
    quantiles_batch,
    stdev,
    stdev_batch,
    variance,
    variance_batch,
)

__version__ = "0.1.4"  # x-release-please-version
__all__ = [
    "StatisticsError",
    "mean",
    "fmean",
    "median",
    "median_low",
    "median_high",
    "mode",
    "multimode",
    "variance",
    "stdev",
    "pvariance",
    "pstdev",
    "quantiles",
    "mean_batch",
    "fmean_batch",
    "median_batch",
    "variance_batch",
    "stdev_batch",
    "pvariance_batch",
    "pstdev_batch",
    "quantiles_batch",
    "backend_info",
    "native_available",
    "__version__",
]

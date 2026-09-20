"""vol-mojo: batch registry-hive walking and pool-header validation.

A drop-in accelerator for two volatility3 hot paths — the
``RegistryHive.visit_nodes`` hive-tree walk and the
``poolscanner.PoolHeaderScanner`` per-hit constraint validation — powered by
a Mojo kernel where the platform supports it (macOS arm64, Linux x86_64),
with a vendored pure-Python fallback everywhere else (including Windows).
Also usable standalone on flat .dat hive files and raw buffers:

    from vol_mojo import walk_hive, scan_pool_headers, PoolConstraint

    tuples = walk_hive(open("SOFTWARE.dat", "rb").read())
    hits = scan_pool_headers(
        buf,
        [PoolConstraint(b"Proc", size=(600, None), page_type=PAGE_TYPE_NONPAGED | PAGE_TYPE_FREE)],
        alignment=0x10, layout="x64", vista_semantics=True,
    )

For volatility3-API-shaped helpers see ``vol_mojo.volatility3_integration``.
Set VOL_MOJO_DISABLE_NATIVE=1 to force the pure-Python fallback.
"""

from vol_mojo import volatility3_integration
from vol_mojo._native import backend_info, native_available
from vol_mojo.core import (
    PAGE_TYPE_FREE,
    PAGE_TYPE_NONPAGED,
    PAGE_TYPE_PAGED,
    HiveError,
    PoolConstraint,
    PoolScanError,
    scan_pool_headers,
    walk_hive,
)

__version__ = "0.1.0"  # x-release-please-version
__all__ = [
    "HiveError",
    "PoolConstraint",
    "PoolScanError",
    "PAGE_TYPE_FREE",
    "PAGE_TYPE_NONPAGED",
    "PAGE_TYPE_PAGED",
    "backend_info",
    "native_available",
    "scan_pool_headers",
    "volatility3_integration",
    "walk_hive",
    "__version__",
]

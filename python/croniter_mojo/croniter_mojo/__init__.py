"""croniter_mojo: drop-in faster 5-field cron iteration, powered by a Mojo kernel.

API-compatible with the `croniter` package's get_next/get_prev for the
supported scope, returning datetimes identical to the oracle. Runs on a
native Mojo kernel where available and falls back to a pure-Python engine
everywhere else (including Windows). Force the fallback with
``CRONITER_MOJO_DISABLE_NATIVE=1``; inspect the active backend with
:func:`backend_info`.
"""

from __future__ import annotations

from ._errors import (
    CroniterBadCronError,
    CroniterBadDateError,
    CroniterMojoError,
    CroniterNotAlphaError,
    CroniterUnsupportedSyntaxError,
)
from ._native import backend_info, native_available
from .core import get_next, get_prev

__version__ = "0.1.2"  # x-release-please-version
__all__ = [
    "get_next",
    "get_prev",
    "native_available",
    "backend_info",
    "CroniterMojoError",
    "CroniterBadCronError",
    "CroniterNotAlphaError",
    "CroniterBadDateError",
    "CroniterUnsupportedSyntaxError",
    "__version__",
]

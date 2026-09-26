"""dateutil-mojo: a drop-in faster replacement for ``dateutil.parser.parse``.

Same call shape, same results — powered by a Mojo kernel where the platform
supports it (macOS arm64, Linux x86_64), with a pure-Python fallback
everywhere else (including Windows).

    from dateutil_mojo import parse

    parse("2025-07-08T14:30:00+02:00")
    parse("Tue, 08 Jul 2025 14:30:00 +0200")
    parse("March 1st, 2025", fuzzy=True)

    from dateutil_mojo import parse_column  # ETL batch API
    parse_column(["2025-07-08", "2025-07-09", "2025-07-10"])

Set DATEUTIL_MOJO_DISABLE_NATIVE=1 to force the pure-Python fallback.
"""

from dateutil_mojo._native import backend_info, native_available
from dateutil_mojo._parser import ParserError, UnknownTimezoneWarning
from dateutil_mojo._tz import tzlocal, tzoffset, tzutc
from dateutil_mojo.core import parse, parse_column

__version__ = "0.1.5"  # x-release-please-version
__all__ = [
    "ParserError",
    "UnknownTimezoneWarning",
    "parse",
    "parse_column",
    "tzlocal",
    "tzoffset",
    "tzutc",
    "backend_info",
    "native_available",
    "__version__",
]

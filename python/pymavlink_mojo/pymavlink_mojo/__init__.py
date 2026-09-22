"""pymavlink-mojo: batch MAVLink v1/v2 log decoding, powered by a Mojo kernel.

Decode whole telemetry logs in one call — headers, payloads, CRC-16/MCRF4XX
validation, and robust resync-on-garbage semantics identical to pymavlink's
robust parser (the ``mavlogdump`` path) — powered by a Mojo kernel where
the platform supports it (macOS arm64, Linux x86_64), with a vendored
pure-Python fallback everywhere else (including Windows).

    import pymavlink_mojo

    result = pymavlink_mojo.parse_buffer(open("flight.tlog", "rb").read(),
                                         timestamps=True)
    for msg in result.messages:
        if msg.get_type() == "ATTITUDE":
            print(msg._timestamp, msg.roll, msg.pitch, msg.yaw)

Set PYMAVLINK_MOJO_DISABLE_NATIVE=1 to force the pure-Python fallback.
"""

from pymavlink_mojo._native import backend_info, native_available
from pymavlink_mojo.core import (
    BadData,
    Header,
    MAVLinkMessage,
    ParseResult,
    UnknownMessage,
    backend,
    iter_parse,
    parse_buffer,
)

__version__ = "0.1.4"  # x-release-please-version
__all__ = [
    "parse_buffer",
    "iter_parse",
    "ParseResult",
    "MAVLinkMessage",
    "BadData",
    "UnknownMessage",
    "Header",
    "backend",
    "backend_info",
    "native_available",
    "__version__",
]

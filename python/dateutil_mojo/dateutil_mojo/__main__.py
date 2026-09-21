"""`python -m dateutil_mojo` — quickstart smoke for end users and CI.

Parses a handful of formats through the public API, prints the results and
a checksum, and reports the active backend. Both the native and the
forced-fallback (DATEUTIL_MOJO_DISABLE_NATIVE=1) runs must print the same
datetimes and checksum.
"""

from __future__ import annotations

from datetime import datetime

import dateutil_mojo

DEFAULT = datetime(2000, 1, 1, 12, 0, 0)

CASES = [
    "2025-07-08T14:30:00+02:00",
    "Tue, 08 Jul 2025 14:30:00 +0200",
    "07/08/2025",
    "March 1st, 2025",
    "20250708",
    "spam 2025-07-08 2:30 PM eggs",
    "2025-07-08T14:30:00Z",
]


def main() -> None:
    results = [dateutil_mojo.parse(c, default=DEFAULT, fuzzy=True) for c in CASES]
    for c, r in zip(CASES, results):
        print(f"{c!r:45} -> {r.isoformat()}")
    col = dateutil_mojo.parse_column(
        ["2025-07-08", "2025-07-09 12:00", "07/10/2025"], default=DEFAULT
    )
    for r in col:
        print(r.isoformat())
    total = sum(
        r.toordinal() * 86400 + r.hour * 3600 + r.minute * 60 + r.second
        for r in results + col
    )
    print(f"checksum: {total}")
    info = dateutil_mojo.backend_info()
    print(f"native: {info['native_available']} ({info.get('native_source') or info.get('error')})")


if __name__ == "__main__":
    main()

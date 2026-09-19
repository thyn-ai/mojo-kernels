"""`python -m croniter_mojo` — quickstart smoke for end users and CI.

Runs a handful of get_next/get_prev calls (including DST boundaries), prints
the results and a checksum, and reports the active backend. Both the native
and the forced-fallback (CRONITER_MOJO_DISABLE_NATIVE=1) runs must print the
same datetimes and checksum.
"""

from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo

import croniter_mojo


def main() -> None:
    et = ZoneInfo("America/New_York")
    results = [
        croniter_mojo.get_next("0 9 * * mon-fri", datetime(2026, 9, 19, 14, 30, 45)),
        croniter_mojo.get_next("0 0 29 2 *", datetime(2026, 9, 19)),
        croniter_mojo.get_next("0 0 * * 5#3", datetime(2026, 9, 19)),
        croniter_mojo.get_prev("0 0 L * *", datetime(2026, 9, 19)),
        croniter_mojo.get_next("30 2 * * *", datetime(2026, 3, 7, 12, 0, tzinfo=et)),
        croniter_mojo.get_next("30 1 * * *", datetime(2026, 10, 31, 12, 0, tzinfo=et)),
        croniter_mojo.get_prev("45 1 * * *", datetime(2026, 11, 1, 1, 30, tzinfo=et, fold=1)),
    ]
    for r in results:
        print(r.isoformat())
    total = sum(r.toordinal() * 86400 + r.hour * 3600 + r.minute * 60 for r in results)
    print(f"checksum: {total}")
    info = croniter_mojo.backend_info()
    print(f"native: {info['native_available']} ({info.get('native_source') or info.get('error')})")


if __name__ == "__main__":
    main()

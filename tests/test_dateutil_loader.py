"""Loader/backend tests for dateutil_mojo: ABI handshake, env overrides,
backend selection, and the __main__ quickstart smoke.

Run on both backends by scripts/test_all_dateutil.sh (native, then
DATEUTIL_MOJO_DISABLE_NATIVE=1); the backend-conditional assertions adapt.
"""

from __future__ import annotations

import os
import subprocess
import sys
from datetime import datetime

import pytest

import dateutil_mojo

DEFAULT = datetime(2000, 2, 15, 4, 5, 6, 789)

FAST_PATH = [
    "2025-07-08T14:30:00+02:00",
    "2025-07-08",
    "Tue, 08 Jul 2025 14:30:00 +0200",
    "08 Jul 2025 14:30:00",
    "July 8, 2025",
    "07/08/2025",
    "20250708",
    "12:30:45.123456",
]


def _norm(dt: datetime):
    return (
        dt.year, dt.month, dt.day, dt.hour, dt.minute, dt.second,
        dt.microsecond, dt.fold, dt.tzname(), dt.utcoffset(),
    )


def test_backend_info_shape():
    info = dateutil_mojo.backend_info()
    assert isinstance(info["native_available"], bool)
    assert info["abi_version_expected"] == 1
    if os.environ.get("DATEUTIL_MOJO_DISABLE_NATIVE") == "1":
        assert info["native_available"] is False
        assert info["disabled_by_env"] is True
    else:
        assert info["native_available"] is True
        assert info["abi_version_native"] == 1
        assert info["native_source"]


def test_native_available_consistency():
    assert dateutil_mojo.native_available() == dateutil_mojo.backend_info()[
        "native_available"
    ]


def test_disable_native_env(monkeypatch):
    # _load checks the env var before the cache, so toggling works live.
    import dateutil_mojo._native as native_mod

    monkeypatch.setenv("DATEUTIL_MOJO_DISABLE_NATIVE", "1")
    assert native_mod.native_available() is False
    info = native_mod.backend_info()
    assert info["disabled_by_env"] is True
    assert info["native_available"] is False
    monkeypatch.undo()
    # After undo the outer run's env is restored (set in the fallback run).
    assert native_mod.native_available() is (
        os.environ.get("DATEUTIL_MOJO_DISABLE_NATIVE") != "1"
    )


def test_public_api_surface():
    for name in (
        "parse", "parse_column", "ParserError", "tzutc", "tzoffset",
        "tzlocal", "backend_info", "native_available", "__version__",
    ):
        assert hasattr(dateutil_mojo, name), name


def test_parse_result_types():
    dt = dateutil_mojo.parse("2025-07-08T14:30:00+02:00", default=DEFAULT)
    assert isinstance(dt, datetime)
    assert dt.utcoffset().total_seconds() == 7200
    dt2 = dateutil_mojo.parse("2025-07-08T14:30:00Z", default=DEFAULT)
    assert dt2.tzname() == "UTC"
    assert dateutil_mojo.tzutc() == dateutil_mojo.tzoffset(None, 0)


def test_parse_column_fast_path():
    got = dateutil_mojo.parse_column(FAST_PATH, default=DEFAULT)
    exp = [dateutil_mojo._parser.parse(s, default=DEFAULT) for s in FAST_PATH]
    assert [_norm(g) for g in got] == [_norm(e) for e in exp]


def test_parse_errors_match_reference():
    from dateutil_mojo._parser import ParserError

    for s in ("not a date", "2011-13-01", ""):
        with pytest.raises(ParserError):
            dateutil_mojo.parse(s, default=DEFAULT)


def test_parserinfo_unsupported():
    with pytest.raises(NotImplementedError):
        dateutil_mojo.parse("2025-07-08", object())


def test_type_validation():
    with pytest.raises(TypeError):
        dateutil_mojo.parse(12345)
    with pytest.raises(TypeError):
        dateutil_mojo.parse("2025-07-08", bogus_kwarg=1)


def test_main_smoke_both_backends():
    env = dict(os.environ)
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    env["PYTHONPATH"] = os.path.join(repo_root, "python", "dateutil_mojo")

    def run(extra_env):
        e = dict(env)
        e.update(extra_env)
        out = subprocess.run(
            [sys.executable, "-m", "dateutil_mojo"],
            capture_output=True, text=True, env=e, cwd=repo_root,
        )
        assert out.returncode == 0, out.stderr
        return out.stdout

    native_out = run({"DATEUTIL_MOJO_DISABLE_NATIVE": "0"})
    fallback_out = run({"DATEUTIL_MOJO_DISABLE_NATIVE": "1"})

    def checksum(text):
        for line in text.splitlines():
            if line.startswith("checksum: "):
                return int(line.split(":", 1)[1])
        raise AssertionError(f"no checksum in output:\n{text}")

    assert checksum(native_out) == checksum(fallback_out)
    # identical parse results, only the backend line may differ
    native_lines = [ln for ln in native_out.splitlines() if not ln.startswith("native:")]
    fallback_lines = [ln for ln in fallback_out.splitlines() if not ln.startswith("native:")]
    assert native_lines == fallback_lines
    if os.environ.get("DATEUTIL_MOJO_TEST_NATIVE") == "1":
        assert "native: True" in native_out
    assert "native: False" in fallback_out

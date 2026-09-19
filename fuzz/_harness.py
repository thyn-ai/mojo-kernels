"""Shared plumbing for the differential fuzz harnesses in this directory.

Each harness decodes an arbitrary byte string into a structured test case,
runs that case through the native Mojo kernel and the vendored fallback (and
the reference package where one is installed), and raises ``Divergence`` when
the backends disagree beyond the tolerance the package documents. Decoding
is *total*: ``ByteCursor`` returns zeros once the input is exhausted, so every
byte string -- including anything libFuzzer mutates -- is a valid case and no
input is ever rejected before it reaches the code under test.

One ``test_one_input(data)`` entry point serves both modes:

* **atheris** (coverage-guided, Linux x86_64):
  ``python fuzz/fuzz_<name>.py [libFuzzer flags] [corpus dirs]``
* **regression** (any platform, no fuzzer installed):
  ``python fuzz/fuzz_<name>.py --regression [FILE|DIR ...]`` replays every
  seed under ``fuzz/corpus/<name>/`` (the default) and exits non-zero on the
  first divergence. ``tests/test_fuzz_regression_*.py`` runs the same replay
  inside the normal pytest suites.

Known divergences
-----------------
A fuzzer that finds a real bug must keep running so it can find the next
one. Each harness therefore carries a list of ``KnownIssue`` entries, one per
open GitHub issue, each with a *predicate* (which inputs the issue covers)
and a *shape* (exactly how the backends are allowed to disagree there). A
divergence that matches both is reported as that issue and does not fail the
run; anything else fails. Seeds named ``known-issue-<key>-*.bin`` are the
minimised reproducers: the regression replay requires each of them to still
reproduce its issue, so a fix cannot land without removing the entry (and
closing the issue), and a regression cannot land without failing the replay.
"""

from __future__ import annotations

import argparse
import struct
import sys
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

FUZZ_DIR = Path(__file__).resolve().parent
REPO_ROOT = FUZZ_DIR.parent


class Divergence(AssertionError):
    """The kernel, the fallback and/or the reference disagree on a case."""


@dataclass(frozen=True)
class KnownIssue:
    """An open, tracked divergence: which cases it covers and how it looks.

    ``key`` is the stable identifier used in seed filenames
    (``known-issue-<key>-<n>.bin``) and in harness output; ``url`` is the
    GitHub issue. ``applies(case)`` says whether the case is inside the
    issue's documented input class; the harness's comparison then checks
    that the observed disagreement has exactly the documented shape.
    """

    key: str
    url: str
    title: str
    applies: Callable[[object], bool]


class ByteCursor:
    """Total little-endian decoder over an arbitrary byte string.

    Reads past the end yield zero bytes, so any input decodes to a case;
    ``remaining()`` lets a harness scale case sizes with the input length.
    """

    def __init__(self, data: bytes) -> None:
        self._data = bytes(data)
        self._pos = 0

    def remaining(self) -> int:
        return max(0, len(self._data) - self._pos)

    def take(self, n: int) -> bytes:
        chunk = self._data[self._pos : self._pos + n]
        self._pos += n
        return chunk + b"\0" * (n - len(chunk))

    def u8(self) -> int:
        return self.take(1)[0]

    def u16(self) -> int:
        return struct.unpack("<H", self.take(2))[0]

    def f64(self) -> float:
        """A raw IEEE-754 double: NaN, infinities, subnormals and huge
        magnitudes are all reachable, which is the point of this reader."""
        return struct.unpack("<d", self.take(8))[0]

    def unit(self) -> float:
        """A fraction in [0, 1] with 16-bit resolution (0.0 and 1.0 included)."""
        return self.u16() / 65535.0

    def int_in(self, lo: int, hi: int) -> int:
        """Uniform-ish integer in the closed range [lo, hi]."""
        if hi <= lo:
            return lo
        return lo + self.u16() % (hi - lo + 1)

    def choice(self, seq: Sequence):
        return seq[self.int_in(0, len(seq) - 1)]

    def flag(self) -> bool:
        return bool(self.u8() & 1)


class ByteWriter:
    """Inverse of ``ByteCursor`` for hand-crafted seeds (minimised reproducers)."""

    def __init__(self) -> None:
        self._parts: list[bytes] = []

    def u8(self, v: int) -> ByteWriter:
        self._parts.append(struct.pack("<B", v))
        return self

    def u16(self, v: int) -> ByteWriter:
        self._parts.append(struct.pack("<H", v))
        return self

    def f64(self, v: float) -> ByteWriter:
        self._parts.append(struct.pack("<d", v))
        return self

    def unit(self, v: float) -> ByteWriter:
        return self.u16(round(min(max(v, 0.0), 1.0) * 65535))

    def int_in(self, v: int, lo: int, hi: int) -> ByteWriter:
        if hi <= lo:
            return self
        if not lo <= v <= hi:
            raise ValueError(f"{v} outside [{lo}, {hi}]")
        return self.u16(v - lo)

    def choice(self, index: int, seq: Sequence) -> ByteWriter:
        return self.int_in(index, 0, len(seq) - 1)

    def flag(self, v: bool) -> ByteWriter:
        return self.u8(1 if v else 0)

    def bytes(self) -> bytes:
        return b"".join(self._parts)


def seed_files(paths: Iterable[Path]) -> list[Path]:
    """Expand files and directories into a sorted list of seed files."""
    out: list[Path] = []
    for path in paths:
        if path.is_dir():
            out.extend(p for p in sorted(path.iterdir()) if p.is_file())
        elif path.is_file():
            out.append(path)
        else:
            raise FileNotFoundError(path)
    return out


def expected_issue_key(seed: Path) -> str | None:
    """The issue key a ``known-issue-<key>-<n>.bin`` seed must reproduce."""
    stem = seed.stem
    if not stem.startswith("known-issue-"):
        return None
    rest = stem[len("known-issue-") :]
    key, _, _ = rest.rpartition("-")
    return key or rest


def replay_seed(
    seed: Path,
    test_one_input: Callable[[bytes], str | None],
    known_issues: Sequence[KnownIssue],
    strict: bool = True,
) -> str | None:
    """Replay one seed with the strict known-issue rules; returns the outcome.

    Raises ``Divergence`` (or lets the harness's exception through) when the
    case fails, and ``AssertionError`` when a ``known-issue-*`` seed no longer
    reproduces the issue it is named after. ``strict=False`` waives that last
    rule for replays without the native kernel, where a native-vs-fallback
    divergence cannot reproduce by construction.
    """
    outcome = test_one_input(seed.read_bytes())
    expected = expected_issue_key(seed)
    if expected is not None and strict:
        by_key = {issue.key: issue for issue in known_issues}
        if expected not in by_key:
            raise AssertionError(
                f"{seed.name}: no KnownIssue with key {expected!r} is registered; "
                "either restore the entry or rename the seed"
            )
        if outcome != expected:
            raise AssertionError(
                f"{seed.name} no longer reproduces known issue {expected!r} "
                f"({by_key[expected].url}); observed outcome: {outcome!r}. "
                "If the issue is fixed, delete this seed, remove the KnownIssue "
                "entry and close the issue in the same change."
            )
    return outcome


def run_regression(
    name: str,
    paths: Sequence[Path],
    test_one_input: Callable[[bytes], str | None],
    known_issues: Sequence[KnownIssue],
    banner: str = "",
    strict: bool = True,
) -> int:
    """Replay every seed; print a summary; return a process exit code."""
    seeds = seed_files(paths)
    print(f"== {name}: regression replay of {len(seeds)} seed(s) ==")
    if banner:
        print(banner)
    if not strict:
        print("native kernel unavailable: known-issue seeds are replayed but not required to reproduce")
    if not seeds:
        print("error: no seed files found", file=sys.stderr)
        return 2
    hits: dict[str, int] = {}
    for seed in seeds:
        try:
            outcome = replay_seed(seed, test_one_input, known_issues, strict=strict)
        except Exception as exc:  # report, then fail the run
            print(f"FAIL {seed}: {type(exc).__name__}: {exc}", file=sys.stderr)
            return 1
        if outcome is not None:
            hits[outcome] = hits.get(outcome, 0) + 1
    print(f"ok: {len(seeds)} seed(s) replayed, no unexplained divergence")
    for issue in known_issues:
        print(
            f"  known issue {issue.key!r} reproduced by {hits.get(issue.key, 0)} "
            f"seed(s): {issue.title} ({issue.url})"
        )
    return 0


def main(
    name: str,
    test_one_input: Callable[[bytes], str | None],
    known_issues: Sequence[KnownIssue],
    default_corpus: Path,
    argv: Sequence[str] | None = None,
    banner: str = "",
    require_native: Callable[[], str | None] | None = None,
) -> int:
    """Command-line entry point shared by every harness.

    With ``--regression`` the seeds are replayed in-process. Otherwise the
    remaining arguments are handed to atheris/libFuzzer verbatim
    (``-max_total_time=60``, ``-runs=N``, corpus directories, ...).
    ``require_native`` returns a reason string when the native kernel is not
    loadable; fuzzing refuses to start then, because a run that compares the
    fallback with itself finds nothing.
    """
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(
        prog=f"fuzz_{name}.py",
        description=f"Differential fuzz harness for {name} (atheris or --regression).",
        add_help=False,
    )
    parser.add_argument("--regression", action="store_true")
    parser.add_argument("-h", "--help", action="store_true")
    opts, rest = parser.parse_known_args(argv)
    if opts.help:
        parser.print_help()
        print(__doc__)
        return 0
    native_missing = require_native() if require_native is not None else None
    if opts.regression:
        paths = [Path(p) for p in rest] or [default_corpus]
        return run_regression(
            name, paths, test_one_input, known_issues, banner, strict=native_missing is None
        )

    try:
        import atheris
    except ImportError:
        print(
            "error: atheris is not installed. Coverage-guided fuzzing runs in the "
            "pixi `fuzz` environment on Linux x86_64 (`pixi run -e fuzz fuzz-"
            f"{name}`); on other platforms replay the seed corpus with "
            f"`python fuzz/fuzz_{name}.py --regression`.",
            file=sys.stderr,
        )
        return 2
    if native_missing:
        print(
            f"error: native kernel unavailable ({native_missing}); build it first",
            file=sys.stderr,
        )
        return 2
    if banner:
        print(banner, file=sys.stderr)

    def fuzz_target(data: bytes) -> None:
        test_one_input(bytes(data))  # a known-issue outcome is not a crash

    atheris.Setup([sys.argv[0], *rest], fuzz_target)
    atheris.Fuzz()
    return 0

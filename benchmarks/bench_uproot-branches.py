#!/usr/bin/env python3
"""Reproducible benchmark: uproot vs uproot_mojo (native kernel and fallback).

Workload: a self-contained ROOT file written here with uproot.recreate
(zlib-compressed): vector<float64>, vector<int32> and std::string branches,
50,000 entries over 10 baskets. Two file-level cells measure the full
path from file path to arrays:

  * cold -- one call in a freshly started Python process (median of 5).
  * warm -- repeated calls in one process (median of 5); uproot's file
    handle and metadata stay open, ours re-opens per call.

A third cell isolates the object deserialization hot loop this kernel
replaces (uproot's interpreted per-entry container readers,
uproot/containers.py AsVector/AsString): a synthetic ROOT-native
std::vector<std::string> basket (collection header + TStrings per entry)
deserialized per entry by uproot's own Python models vs one kernel call.

Correctness is asserted (offsets and content bytes exactly equal to
uproot's) before any timing happens, so the numbers always come from a
verified-correct build. Timings are the median of 5 runs.

Run from the repository root (build the kernel first):

    PYTHONPATH=python/uproot-branches_mojo python benchmarks/bench_uproot-branches.py
"""

from __future__ import annotations

import os
import platform
import statistics
import struct
import subprocess
import sys
import tempfile
import time

import numpy as np

N_RUNS = 5
N_ENTRIES = 50_000
N_BASKETS = 10

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(REPO, "python", "uproot-branches_mojo"))

import uproot_mojo  # noqa: E402


def machine_info() -> list[str]:
    lines = [
        f"- date: {time.strftime('%Y-%m-%d')}",
        f"- machine: {platform.platform()} ({platform.machine()})",
    ]
    try:
        chip = subprocess.run(
            ["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True
        ).stdout.strip()
        if chip:
            lines.append(f"- cpu: {chip}")
    except OSError:
        pass
    import awkward
    import cramjam
    import uproot

    lines.append(
        f"- python: {platform.python_version()}, numpy: {np.__version__}, "
        f"uproot: {uproot.__version__}, awkward: {awkward.__version__}, "
        f"cramjam: {cramjam.__version__}"
    )
    info = uproot_mojo.backend_info()
    lines.append(f"- backend: {info}")
    return lines


def make_fixture(path: str) -> None:
    import uproot

    rng = np.random.default_rng(20260919)
    per_basket = N_ENTRIES // N_BASKETS
    with uproot.recreate(path, compression=uproot.ZLIB(4)) as f:
        f.mktree(
            "events",
            {"pt": "var * float64", "id": "var * int32", "label": "string"},
        )
        for basket in range(N_BASKETS):
            pt = [
                rng.normal(25.0, 10.0, size=int(n)).tolist()
                for n in rng.integers(0, 16, size=per_basket)
            ]
            ids = [
                rng.integers(-2**31, 2**31 - 1, size=int(n)).tolist()
                for n in rng.integers(0, 8, size=per_basket)
            ]
            labels = [
                "".join(
                    chr(c)
                    for c in rng.integers(ord("a"), ord("z") + 1, size=int(n))
                )
                for n in rng.integers(0, 24, size=per_basket)
            ]
            f["events"].extend({"pt": pt, "id": ids, "label": labels})


def verify(path: str) -> None:
    import uproot

    tree = uproot.open(path)["events"]
    for branch, dtype in (("pt", "float64"), ("id", "int32")):
        ours = uproot_mojo.read_branch(path, branch, dtype=dtype)
        ref = tree[branch].array(library="np", interpretation=None)
        got = ours.to_list()
        want = [list(map(float, entry)) for entry in ref]
        assert len(got) == len(want) and all(
            np.array_equal(np.asarray(g), np.asarray(w)) for g, w in zip(got, want)
        ), f"{branch}: value mismatch vs uproot"
    labels = uproot_mojo.read_branch(path, "label")
    assert labels.to_list() == tree["label"].array(library="np").tolist()


COLD_OURS = """
import time, sys
sys.path.insert(0, {pkg!r})
import uproot_mojo
t0 = time.perf_counter()
r = uproot_mojo.read_branch({path!r}, "pt", dtype="float64")
n = sum(len(e) for e in r.to_list())
print(f"{{time.perf_counter() - t0:.6f}} {{n}}")
"""

COLD_UPROOT = """
import time
import uproot
t0 = time.perf_counter()
arr = uproot.open({path!r})["events"]["pt"].array()
n = sum(len(e) for e in arr)
print(f"{{time.perf_counter() - t0:.6f}} {{n}}")
"""


def cold_median(script: str) -> tuple[float, int]:
    times = []
    count = None
    env = dict(os.environ, PYTHONNOUSERSITE="1")
    # Inherit the parent's PYTHONPATH so oracle/runtime deps installed
    # outside the pixi env remain importable in the fresh process.
    pkg = os.path.join(REPO, "python", "uproot-branches_mojo")
    env["PYTHONPATH"] = os.pathsep.join(
        [pkg, os.environ.get("PYTHONPATH", "")]
    ).rstrip(os.pathsep)
    for _ in range(N_RUNS):
        out = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
            check=True,
            env=env,
        ).stdout.split()
        times.append(float(out[0]))
        count = int(out[1])
    return statistics.median(times), count


def warm_median(fn) -> float:
    fn()  # warm-up
    times = []
    for _ in range(N_RUNS):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    return statistics.median(times)


def _tstring(raw: bytes) -> bytes:
    if len(raw) < 255:
        return bytes([len(raw)]) + raw
    return b"\xff" + struct.pack(">I", len(raw)) + raw


def make_vector_string_basket():
    """Synthetic ROOT-native vector<string> basket (50k entries x 0-5 strings)."""
    rng = np.random.default_rng(20260919)
    entries = []
    for _ in range(N_ENTRIES):
        n = int(rng.integers(0, 6))
        strings = [
            "".join(chr(c) for c in rng.integers(ord("a"), ord("z") + 1, size=int(m)))
            for m in rng.integers(1, 20, size=n)
        ]
        entries.append(strings)
    blobs = []
    borders = np.zeros(N_ENTRIES + 1, dtype=np.int64)
    pos = 0
    for i, strings in enumerate(entries):
        payload = struct.pack(">hI", 3, len(strings)) + b"".join(
            _tstring(s.encode()) for s in strings
        )
        blob = struct.pack(">I", 0x40000000 | len(payload)) + payload
        blobs.append(blob)
        pos += len(blob)
        borders[i + 1] = pos
    return b"".join(blobs), borders, entries


def bench_vector_string() -> tuple[float, float]:
    from uproot.containers import AsString, AsVector
    from uproot.source.chunk import Chunk
    from uproot.source.cursor import Cursor

    from uproot_mojo import _fallback

    data, borders, expected = make_vector_string_basket()
    model = AsVector(True, AsString(False))

    try:
        from uproot_mojo import _native

        _native._load()
        walk = _native.walk_basket
        backend = "native"
    except Exception:
        walk = _fallback.walk_basket
        backend = "fallback"

    class _File:
        file_path = "bench-vector-string.root"

    def uproot_side():
        out = []
        for i in range(N_ENTRIES):
            vector = model.read(
                Chunk.wrap(None, data[borders[i] : borders[i + 1]]),
                Cursor(0),
                {"reading": True},
                _File(),
                _File(),
                None,
            )
            out.append([vector[j] for j in range(len(vector))])
        return out

    def ours():
        return walk(data, borders, _fallback.MODE_AUTO_STR, 0, "bench basket")

    detected, offsets, content, string_offsets = ours()
    assert detected == _fallback.MODE_STR_VEC
    got = uproot_mojo.JaggedStringArray(offsets, string_offsets, np.frombuffer(content, np.uint8), backend)
    assert got.to_list() == expected
    want = uproot_side()
    assert want == expected

    t_uproot = warm_median(uproot_side)
    t_ours = warm_median(ours)
    return t_uproot, t_ours, backend


def main() -> None:
    path = os.path.join(tempfile.mkdtemp(prefix="uproot-mojo-bench-"), "bench.root")
    print("building fixture:", path)
    make_fixture(path)
    size_mb = os.path.getsize(path) / 1e6
    print(f"fixture size: {size_mb:.2f} MiB, {N_ENTRIES} entries x {N_BASKETS} baskets")

    print("verifying correctness vs uproot ...")
    verify(path)

    pkg = os.path.join(REPO, "python", "uproot-branches_mojo")
    t_cold_ours, n1 = cold_median(COLD_OURS.format(pkg=pkg, path=path))
    t_cold_uproot, n2 = cold_median(COLD_UPROOT.format(path=path))
    assert n1 == n2

    import uproot

    # Warm steady state: the full file -> array operation repeated (uproot
    # re-opens per call, matching uproot_mojo.read_branch's stateless path).
    t_warm_ours = warm_median(
        lambda: uproot_mojo.read_branch(path, "pt", dtype="float64")
    )
    t_warm_uproot = warm_median(lambda: uproot.open(path)["events"]["pt"].array())
    # Context: repeated .array() on an already-open TBranch is memoized by
    # uproot (no re-decompression), so it is not a like-for-like comparison.
    _branch = uproot.open(path)["events"]["pt"]
    _branch.array()
    t_warm_uproot_cached = warm_median(lambda: _branch.array())

    t_vecstr_uproot, t_vecstr_ours, vecstr_backend = bench_vector_string()

    print()
    print("machine / versions:")
    for line in machine_info():
        print(" ", line)
    print()
    print(f"median of {N_RUNS} runs; file -> array for vector<float64> ({n1} items total)")
    print("| cell | uproot (ms) | uproot-mojo (ms) | speedup |")
    print("|---|---|---|---|")
    print(
        f"| cold (fresh process, one call) | {t_cold_uproot * 1e3:,.2f} | "
        f"{t_cold_ours * 1e3:,.2f} | {t_cold_uproot / t_cold_ours:.1f}x |"
    )
    print(
        f"| warm (steady-state: full open + read per call) | {t_warm_uproot * 1e3:,.2f} | "
        f"{t_warm_ours * 1e3:,.2f} | {t_warm_uproot / t_warm_ours:.1f}x |"
    )
    print(
        f"| warm, uproot memoized repeat on open TBranch "
        f"(no re-decompression; not like-for-like) | {t_warm_uproot_cached * 1e3:,.2f} | "
        f"(re-reads every call) {t_warm_ours * 1e3:,.2f} | n/a |"
    )
    print(
        f"| vector<string> basket walk, {N_ENTRIES} entries "
        f"(uproot per-entry Python vs one {vecstr_backend} walk) | {t_vecstr_uproot * 1e3:,.2f} | "
        f"{t_vecstr_ours * 1e3:,.2f} | {t_vecstr_uproot / t_vecstr_ours:.1f}x |"
    )


if __name__ == "__main__":
    main()

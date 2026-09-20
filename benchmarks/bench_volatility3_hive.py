#!/usr/bin/env python3
"""Reproducible benchmark: volatility3 (pip oracle) vs vol_mojo.

Two cells mirror the accelerated paths:

- ``hive walk``: the oracle's ``RegistryHive.visit_nodes`` over a generated
  ~20k-node valid hive (see tests/test_volatility3_hive_fixtures.py for the
  generator — no copyrighted hives), vs ``vol_mojo.walk_hive`` on the same
  flat image. The oracle harness (synthetic _CMHIVE/_HMAP superstructure,
  fake nt_symbols ISF) is built once, outside the timed region.
- ``pool scan``: the oracle's ``PoolHeaderScanner.__call__`` over an 8 MiB
  buffer with the 13 builtin constraints, vs ``vol_mojo.scan_pool_headers``
  on the same buffer and constraints (x64 layout, Vista+ semantics,
  alignment 0x10).

Every cell is timed two ways:

- **cold**: first call in a fresh interpreter (import + dlopen + first
  touch), measured inside a spawned subprocess (fixtures are pre-generated
  to /tmp so the subprocess only reads bytes);
- **warm**: median of 5 steady-state calls in this process.

Correctness is gated before timing: both cells are deterministic, so the
oracle and vol_mojo outputs must be exactly equal (tuple-for-tuple /
hit-for-hit, order included) — the full gate is the differential suite.
Timings are single-threaded wall clock.

Run from the repository root (the oracle comes from .oracle-volatility3):

    PYTHONPATH="python/vol_mojo:.oracle-volatility3" pixi run python benchmarks/bench_volatility3_hive.py
"""

from __future__ import annotations

import os
import platform
import statistics
import subprocess
import sys
import tempfile
import time

N_RUNS = 5

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "tests"))
sys.path.insert(0, os.path.join(REPO_ROOT, "python", "vol_mojo"))

from test_volatility3_hive_fixtures import (  # noqa: E402
    HiveBuilder,
    add_management,
    fake_nt_isf_json,
    oracle_pool_scanner_factory,
    put_header_x64,
)

# Workload constants.
HIVE_CHILDREN = 100
HIVE_GRANDCHILDREN = 199  # 100 * 199 + 100 + 1 = 20_001 key nodes
POOL_SIZE = 8 * 1024 * 1024
POOL_HEADERS = 2000

WORK = os.path.join(tempfile.gettempdir(), "vol_mojo_bench")
HIVE_DAT = os.path.join(WORK, "bench_hive.dat")
HIVE_IMG = os.path.join(WORK, "bench_hive.img")
HIVE_ISF = os.path.join(WORK, "bench_hive.isf.json")
POOL_BIN = os.path.join(WORK, "bench_pool.bin")


def machine_info() -> str:
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
    lines.append(f"- python: {platform.python_version()}")
    try:
        mojo = subprocess.run(
            ["mojo", "--version"], capture_output=True, text=True
        ).stdout.strip()
        lines.append(f"- mojo: {mojo}")
    except OSError:
        pass
    return "\n".join(lines)


def build_hive_dat() -> bytes:
    """Deterministic ~20k-node hive: ROOT -> 100 children -> 199 each."""
    b = HiveBuilder()
    child_idxs = []
    for c in range(HIVE_CHILDREN):
        leaves = [
            b.add_cell(b.nk(f"LeafKey-{c:03d}-{g:04d}".encode()))
            for g in range(HIVE_GRANDCHILDREN)
        ]
        # Fan the leaves out through ri -> lh chunks of 40.
        lhs = [
            b.add_cell(
                b.index(b"lh", [(cell, 0x1000 + i) for i, cell in enumerate(leaves[k : k + 40])])
            )
            for k in range(0, HIVE_GRANDCHILDREN, 40)
        ]
        ri = b.add_cell(b.index(b"ri", lhs))
        child_idxs.append(
            b.add_cell(
                b.nk(
                    f"ChildKey-{c:03d}".encode(),
                    subkey_count=HIVE_GRANDCHILDREN,
                    subkey_list=ri,
                )
            )
        )
    lhs = [
        b.add_cell(b.index(b"lh", [(cell, 0x5000 + i) for i, cell in enumerate(child_idxs[k : k + 40])]))
        for k in range(0, HIVE_CHILDREN, 40)
    ]
    ri = b.add_cell(b.index(b"ri", lhs))
    root = b.add_cell(b.nk(b"ROOT", flags=0x2C, subkey_count=HIVE_CHILDREN, subkey_list=ri))
    dat, _ = b.build_dat(root)
    return dat


def build_pool_bin() -> bytes:
    """Deterministic 8 MiB buffer with planted pool headers + decoy tags."""
    buf = bytearray(b"\x00" * POOL_SIZE)
    tags = [
        (b"Proc", 0x41, 2), (b"File", 0x10, 2), (b"Thre", 0x40, 2),
        (b"Muta", 0x08, 2), (b"Driv", 0x10, 2), (b"MmLd", 0x05, 2),
        (b"Symb", 0x05, 1), (b"CM10", 0x40, 1), (b"AtmT", 0x10, 1),
        (b"Pro\xe3", 0x41, 2), (b"Thr\xe5", 0x40, 2), (b"Fil\xe5", 0x10, 2),
        (b"Mut\xe1", 0x08, 2), (b"Dri\xf6", 0x10, 2), (b"Sym\xe2", 0x05, 1),
    ]
    stride = POOL_SIZE // (POOL_HEADERS + 1) & ~0xF
    for i in range(POOL_HEADERS):
        off = stride * (i + 1)
        tag, block, ptype = tags[i % len(tags)]
        if i % 7 == 3:
            block = 0x01  # undersized -> rejected
        if i % 11 == 5:
            ptype = 0  # free pool
        put_header_x64(buf, off, i & 0xFF, block, ptype, tag)
        # decoy tag bytes with no valid header nearby
        if i % 5 == 2:
            buf[off + 0x800 : off + 0x804] = tags[(i + 1) % len(tags)][0]
    return bytes(buf)


def prepare_fixtures() -> None:
    os.makedirs(WORK, exist_ok=True)
    dat = build_hive_dat()
    with open(HIVE_DAT, "wb") as f:
        f.write(dat)
    img, _ = add_management(dat, len(dat) - 0x1000)
    with open(HIVE_IMG, "wb") as f:
        f.write(img)
    with open(HIVE_ISF, "w") as f:
        f.write(fake_nt_isf_json())
    with open(POOL_BIN, "wb") as f:
        f.write(build_pool_bin())


# ---------------------------------------------------------------------------
# Oracle (volatility3) paths — measured the way plugins drive them.
# ---------------------------------------------------------------------------


def make_oracle_hive():
    from test_volatility3_hive_fixtures import _make_hive_layer

    return _make_hive_layer(HIVE_IMG, (os.path.getsize(HIVE_DAT) + 0xFFF) & ~0xFFF, HIVE_ISF)


def oracle_hive_walk(hive):
    tuples = []
    hive.visit_nodes(
        lambda node: tuples.append(
            (node.vol.offset, node.vol.type_name.split("!")[-1], node.get_name())
        )
    )
    return tuples


def make_oracle_pool_args():
    from volatility3.framework.symbols.windows.extensions import pool as _p  # noqa: F401
    from volatility3.plugins.windows import poolscanner

    return poolscanner.PoolScanner.builtin_constraints("nt_symbols")


def make_oracle_pool_run(constraints):
    """The oracle scanner, built once; the callable times only the scan."""
    return oracle_pool_scanner_factory(
        POOL_BIN, constraints, "poolheader-x64", True, 0x10
    )


# ---------------------------------------------------------------------------
# Timing helpers.
# ---------------------------------------------------------------------------


def time_median(fn, n_runs: int = N_RUNS) -> float:
    fn()  # warmup (also the steady-state verifier)
    samples = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        fn()
        samples.append(time.perf_counter() - t0)
    return statistics.median(samples)


def time_cold(snippet: str, env_extra: dict | None = None) -> float:
    """First-call latency in a fresh interpreter (import + dlopen + call)."""
    env = dict(os.environ)
    env["PYTHONPATH"] = os.path.join(REPO_ROOT, "python", "vol_mojo") + os.pathsep + env.get(
        "PYTHONPATH", ""
    )
    if env_extra:
        env.update(env_extra)
    code = (
        "import time\n"
        "t0 = time.perf_counter()\n"
        + snippet
        + "\nprint(f'{time.perf_counter() - t0:.6f}')"
    )
    out = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        env=env,
        check=True,
        cwd=REPO_ROOT,
    )
    return float(out.stdout.strip().splitlines()[-1])


def main() -> None:
    import vol_mojo

    info = vol_mojo.backend_info()
    print("== environment ==")
    print(machine_info())
    print(
        f"- vol_mojo backend: {'native' if info['native_available'] else 'FALLBACK'} "
        f"({info.get('native_source') or info.get('error')})"
    )
    print(f"- warm timings: median of {N_RUNS}; cold: first call in a fresh process")
    if not info["native_available"]:
        sys.exit(
            "native kernel unavailable; refusing to benchmark the fallback as 'vol_mojo'"
        )

    import volatility3  # noqa: F401  (oracle presence check)

    print("- generating fixtures (deterministic, no downloads) ...")
    prepare_fixtures()
    hive_dat = open(HIVE_DAT, "rb").read()
    pool_bin = open(POOL_BIN, "rb").read()
    n_nodes_expected = HIVE_CHILDREN * HIVE_GRANDCHILDREN + HIVE_CHILDREN + 1
    print(f"- hive: {len(hive_dat)} bytes, {n_nodes_expected} key nodes; "
          f"pool buffer: {len(pool_bin)} bytes, {POOL_HEADERS} planted headers + decoys")

    print("\n== correctness gate ==")
    oracle_hive = make_oracle_hive()
    expected_walk = oracle_hive_walk(oracle_hive)
    got_walk = vol_mojo.walk_hive(hive_dat)
    assert got_walk == expected_walk, "hive walk mismatch"
    print(f"  hive walk: {len(got_walk)} tuples, exact match [OK]")

    constraints = make_oracle_pool_args()
    oracle_pool_run = make_oracle_pool_run(constraints)
    expected_scan = oracle_pool_run(pool_bin)
    got_scan = vol_mojo.scan_pool_headers(
        pool_bin, constraints, alignment=0x10, layout="x64", vista_semantics=True
    )
    assert got_scan == expected_scan, "pool scan mismatch"
    print(f"  pool scan: {len(got_scan)} passing headers, exact match [OK]")

    rows: list[tuple[str, float, float, float]] = []  # label, oracle warm, ours cold, ours warm

    print("\n== warm steady-state (median of 5, seconds) ==")
    print(f"{'cell':>10} | {'volatility3 oracle':>18} | {'vol_mojo':>10} | {'speedup':>8}")
    print(f"{'-' * 10}-+-{'-' * 18}-+-{'-' * 10}-+-{'-' * 8}")
    for label, oracle_fn, our_fn in [
        ("hive walk", lambda: oracle_hive_walk(oracle_hive), lambda: vol_mojo.walk_hive(hive_dat)),
        (
            "pool scan",
            lambda: oracle_pool_run(pool_bin),
            lambda: vol_mojo.scan_pool_headers(
                pool_bin, constraints, alignment=0x10, layout="x64", vista_semantics=True
            ),
        ),
    ]:
        t_oracle = time_median(oracle_fn)
        t_ours = time_median(our_fn)
        rows.append((label, t_oracle, 0.0, t_ours))
        print(f"{label:>10} | {t_oracle:>18.4f} | {t_ours:>10.6f} | {t_oracle / t_ours:>7.1f}x")

    print("\n== cold first-call (fresh interpreter, seconds) ==")
    walk_call = (
        "import vol_mojo\n"
        f"vol_mojo.walk_hive(open({HIVE_DAT!r}, 'rb').read())\n"
    )
    # Same workload shape as the quickstart (two well-known Microsoft pool
    # tags, size floors, nonpaged-or-free / paged-or-free).
    scan_call = (
        "import vol_mojo\n"
        f"data = open({POOL_BIN!r}, 'rb').read()\n"
        "cs = [\n"
        " vol_mojo.PoolConstraint(b'Proc', size=(600, None), page_type=vol_mojo.PAGE_TYPE_NONPAGED | vol_mojo.PAGE_TYPE_FREE),\n"
        " vol_mojo.PoolConstraint(b'CM10', size=(800, None), page_type=vol_mojo.PAGE_TYPE_PAGED | vol_mojo.PAGE_TYPE_FREE),\n"
        "]\n"
        "vol_mojo.scan_pool_headers(data, cs, alignment=0x10)\n"
    )
    colds = []
    for label, snippet in [("hive walk", walk_call), ("pool scan", scan_call)]:
        t_cold = time_cold(snippet)
        colds.append(t_cold)
        print(f"  {label:>10}: {t_cold:.4f} s")
    rows = [(label, t_o, t_c, t_w) for (label, t_o, _, t_w), t_c in zip(rows, colds)]

    # Fallback context rows: what non-native platforms get.
    os.environ["VOL_MOJO_DISABLE_NATIVE"] = "1"
    try:
        import vol_mojo._native as nat

        nat._LIB, nat._LIB_SOURCE = None, None
        fb_walk = time_median(lambda: vol_mojo.walk_hive(hive_dat))
        fb_scan = time_median(
            lambda: vol_mojo.scan_pool_headers(
                pool_bin, constraints, alignment=0x10, layout="x64", vista_semantics=True
            )
        )
    finally:
        del os.environ["VOL_MOJO_DISABLE_NATIVE"]
        nat._LIB, nat._LIB_SOURCE = None, None
    print("\n== pure-Python fallback context (warm, median of 5, seconds) ==")
    print(f"  hive walk: {fb_walk:.4f}")
    print(f"  pool scan: {fb_scan:.4f}")

    print("\n== README paste block ==")
    print("| workload | volatility3 2.28.2 warm (s) | vol-mojo cold (s) | vol-mojo warm (s) | warm speedup |")
    print("|---|---:|---:|---:|---:|")
    for label, t_o, t_c, t_w in rows:
        print(f"| {label} | {t_o:.4f} | {t_c:.4f} | {t_w:.6f} | {t_o / t_w:.0f}x |")
    print(f"| hive walk — Python fallback | {rows[0][1]:.4f} | — | {fb_walk:.4f} | {rows[0][1] / fb_walk:.1f}x |")
    print(f"| pool scan — Python fallback | {rows[1][1]:.4f} | — | {fb_scan:.4f} | {rows[1][1] / fb_scan:.1f}x |")


if __name__ == "__main__":
    main()

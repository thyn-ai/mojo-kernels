#!/usr/bin/env python3
"""Reproducible benchmark: stdlib ipaddress vs ipaddress_mojo (batch API).

Inputs are deterministic synthetic workloads generated locally (seeded RNG;
no network, no system files). Correctness against the stdlib oracle is
asserted before any timing happens, so the numbers below always come from a
verified-correct build.

Three workloads, each timed warm (median of N_RUNS in-process repetitions),
plus a cold-start table (median of COLD_RUNS fresh OS processes):

* bulk membership (the ACL/log-analytics hot loop): for M addresses, is the
  address in ANY of N networks — stdlib `any(a in n for n in nets)` vs
  `ipaddress_mojo.contains_many(nets, addrs)` on both backends.
* bulk parse: N address strings parsed by `ipaddress.ip_address` vs
  `ipaddress_mojo.parse_many` on both backends.
* bulk collapse: N random networks collapsed by
  `ipaddress.collapse_addresses` vs `ipaddress_mojo.collapse_batch`.

Run from the repository root:

    PYTHONPATH=python/ipaddress_mojo python benchmarks/bench_ipaddress.py
"""

from __future__ import annotations

import os
import platform
import statistics
import subprocess
import sys
import time

N_RUNS = 7  # warm timings: median over this many repetitions
COLD_RUNS = 7  # cold timings: median over this many fresh processes

# Membership sizes: (n_networks, n_addresses). The stdlib row is only
# measured where the quadratic scan finishes in reasonable time.
MEM_SMALL = (10_000, 10_000)  # 1e8 membership checks — the stdlib baseline
MEM_BIG = (1_000_000, 1_000_000)  # production scale for the kernel
PARSE_N = 1_000_000
COLLAPSE_N = 1_000_000


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


def cold_time(code: str, env_extra: dict | None = None) -> float:
    """Median wall time of `python -c code` in a fresh process."""
    env = dict(os.environ)
    repo = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    env["PYTHONPATH"] = os.path.join(repo, "python", "ipaddress_mojo") + os.pathsep + env.get(
        "PYTHONPATH", ""
    )
    if env_extra:
        env.update(env_extra)
    samples = []
    for _ in range(COLD_RUNS):
        t0 = time.perf_counter()
        subprocess.run(
            [sys.executable, "-c", code],
            env=env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        samples.append(time.perf_counter() - t0)
    return statistics.median(samples)


def median(f, n=N_RUNS):
    samples = []
    for _ in range(n):
        t0 = time.perf_counter()
        f()
        samples.append(time.perf_counter() - t0)
    return statistics.median(samples)


def fmt(t: float) -> str:
    if t >= 1:
        return f"{t:,.3f} s"
    if t >= 1e-3:
        return f"{t * 1e3:,.3f} ms"
    return f"{t * 1e6:,.1f} µs"


def make_membership(rng, n_nets, n_addrs, v4=True):
    import ipaddress_mojo as ipm

    if v4:
        nets = [ipm.IPv4Network((rng.randrange(0, 2**32), rng.randrange(8, 33)), strict=False)
                for _ in range(n_nets)]
        addrs = [ipm.IPv4Address(rng.randrange(0, 2**32)) for _ in range(n_addrs)]
    else:
        nets = [ipm.IPv6Network((rng.randrange(0, 2**64), rng.randrange(64, 129)), strict=False)
                for _ in range(n_nets)]
        addrs = [ipm.IPv6Address(rng.randrange(0, 2**64)) for _ in range(n_addrs)]
    return nets, addrs


def main() -> None:
    import random

    import ipaddress as ref
    import ipaddress_mojo as ipm
    from ipaddress_mojo import _native

    rng = random.Random(20260920)

    print("== machine ==")
    print(machine_info())
    print(f"- warm: median of {N_RUNS}; cold: median of {COLD_RUNS} fresh processes")
    info = _native.backend_info()
    print(f"- backend: native_available={info['native_available']} ({info['native_source']})")

    # ------------------------------------------------------------------
    # Correctness gate (must pass before any timing).
    # ------------------------------------------------------------------
    nets, addrs = make_membership(rng, 1000, 2000, v4=True)
    got = list(ipm.contains_many(nets, addrs))
    want = [any(a in n for n in [ref.IPv4Network((int(x.network_address), x.prefixlen))
                                 for x in nets])
            for a in [ref.IPv4Address(int(a)) for a in addrs]]
    assert got == want, "contains_many parity gate failed"
    strs = [".".join(str(rng.randrange(256)) for _ in range(4)) for _ in range(5000)]
    assert [int(a) for a in ipm.parse_many(strs)] == [int(ref.ip_address(s)) for s in strs]
    cn = [ipm.IPv4Network((rng.randrange(0, 2**32), rng.randrange(16, 33)), strict=False)
          for _ in range(5000)]
    assert [str(x) for x in ipm.collapse_batch(cn)] == [
        str(x) for x in ref.collapse_addresses(
            [ref.IPv4Network((int(n.network_address), n.prefixlen)) for n in cn])
    ]
    print("correctness gate: OK")

    # ------------------------------------------------------------------
    # 1. Bulk membership
    # ------------------------------------------------------------------
    print("\n== bulk membership: warm (per call) ==")
    print(f"{'workload':>38} | {'stdlib any()':>14} | {'native':>12} | {'fallback':>12} | {'speedup':>9}")

    for n_nets, n_addrs, stdlib_too in [(MEM_SMALL[0], MEM_SMALL[1], False),
                                        (MEM_BIG[0], MEM_BIG[1], True)]:
        nets, addrs = make_membership(rng, n_nets, n_addrs, v4=True)
        t_nat = median(lambda: ipm.contains_many(nets, addrs))
        # fallback: force off native for this measurement
        os.environ["IPADDRESS_MOJO_DISABLE_NATIVE"] = "1"
        _native._LIB = None
        t_fb = median(lambda: ipm.contains_many(nets, addrs))
        del os.environ["IPADDRESS_MOJO_DISABLE_NATIVE"]
        _native._LIB = None
        if not stdlib_too:
            ref_nets = [ref.IPv4Network((int(x.network_address), x.prefixlen)) for x in nets]
            ref_addrs = [ref.IPv4Address(int(a)) for a in addrs]

            def stdlib_scan():
                return sum(1 for a in ref_addrs if any(a in n for n in ref_nets))

            t_std = median(stdlib_scan, n=3)
            speed = f"{t_std / t_nat:,.1f}x"
            std_cell = fmt(t_std)
        else:
            speed = "—"
            std_cell = "> 1 h (not run)"
        print(f"{f'v4: {n_addrs:,} addrs x {n_nets:,} nets':>38} | {std_cell:>14} | "
              f"{fmt(t_nat):>12} | {fmt(t_fb):>12} | {speed:>9}")

    # ------------------------------------------------------------------
    # 2. Bulk parse
    # ------------------------------------------------------------------
    print("\n== bulk parse: warm (per call) ==")
    print(f"{'workload':>38} | {'stdlib loop':>14} | {'native':>12} | {'fallback':>12} | {'speedup':>9}")
    v4_strs = [".".join(str(rng.randrange(256)) for _ in range(4)) for _ in range(PARSE_N)]
    v6_strs = [":".join("%x" % rng.randrange(0x10000) for _ in range(8)) for _ in range(PARSE_N)]
    for label, strs in [(f"v4: {PARSE_N:,} addresses", v4_strs),
                        (f"v6: {PARSE_N:,} addresses", v6_strs)]:
        t_nat = median(lambda: ipm.parse_many(strs))
        os.environ["IPADDRESS_MOJO_DISABLE_NATIVE"] = "1"
        _native._LIB = None
        t_fb = median(lambda: ipm.parse_many(strs))
        del os.environ["IPADDRESS_MOJO_DISABLE_NATIVE"]
        _native._LIB = None
        t_std = median(lambda: [ref.ip_address(s) for s in strs], n=3)
        print(f"{label:>38} | {fmt(t_std):>14} | {fmt(t_nat):>12} | {fmt(t_fb):>12} | "
              f"{t_std / t_nat:>8.1f}x")

    # ------------------------------------------------------------------
    # 3. Bulk collapse
    # ------------------------------------------------------------------
    print("\n== bulk collapse: warm (per call) ==")
    print(f"{'workload':>38} | {'stdlib':>14} | {'native':>12} | {'fallback':>12} | {'speedup':>9}")
    cn4 = [ipm.IPv4Network((rng.randrange(0, 2**32) & (0xFFFFFFFF << rng.randrange(0, 12)),
                            rng.randrange(12, 33)), strict=False)
           for _ in range(COLLAPSE_N)]
    cn6 = [ipm.IPv6Network((rng.randrange(0, 2**64) & ((2**64 - 1) << rng.randrange(0, 32)),
                            rng.randrange(48, 129)), strict=False)
           for _ in range(COLLAPSE_N)]
    for label, nets, ref_mk in [(f"v4: {COLLAPSE_N:,} networks", cn4, ref.IPv4Network),
                                (f"v6: {COLLAPSE_N:,} networks", cn6, ref.IPv6Network)]:
        t_nat = median(lambda: ipm.collapse_batch(nets))
        os.environ["IPADDRESS_MOJO_DISABLE_NATIVE"] = "1"
        _native._LIB = None
        t_fb = median(lambda: ipm.collapse_batch(nets))
        del os.environ["IPADDRESS_MOJO_DISABLE_NATIVE"]
        _native._LIB = None
        ref_nets = [ref_mk((int(n.network_address), n.prefixlen)) for n in nets]
        t_std = median(lambda: list(ref.collapse_addresses(ref_nets)), n=3)
        print(f"{label:>38} | {fmt(t_std):>14} | {fmt(t_nat):>12} | {fmt(t_fb):>12} | "
              f"{t_std / t_nat:>8.1f}x")

    # ------------------------------------------------------------------
    # 4. Cold start (fresh process: import + first call)
    # ------------------------------------------------------------------
    print("\n== cold start (fresh process: import + 1k-item call) ==")
    cold_setup = (
        "nets=[ipm.IPv4Network((i*256+0x0A000000)%2**32,24) for i in range(1000)];"
        "addrs=[ipm.IPv4Address(i*64) for i in range(1000)];"
        "ipm.contains_many(nets,addrs);"
        "ipm.parse_many(['10.1.2.3','::1','2001:db8::1'])"
    )
    cold_std = (
        "import ipaddress as ip;"
        "nets=[ip.IPv4Network((i*256+0x0A000000)%2**32,24) for i in range(1000)];"
        "addrs=[ip.IPv4Address(i*64) for i in range(1000)];"
        "[any(a in n for n in nets) for a in addrs];"
        "[ip.ip_address(s) for s in ['10.1.2.3','::1','2001:db8::1']]"
    )
    t_std_cold = cold_time(cold_std)
    t_nat_cold = cold_time("import ipaddress_mojo as ipm;" + cold_setup)
    t_fb_cold = cold_time("import ipaddress_mojo as ipm;" + cold_setup,
                          {"IPADDRESS_MOJO_DISABLE_NATIVE": "1"})
    print(f"stdlib: {fmt(t_std_cold)} | native: {fmt(t_nat_cold)} | fallback: {fmt(t_fb_cold)}")


if __name__ == "__main__":
    main()

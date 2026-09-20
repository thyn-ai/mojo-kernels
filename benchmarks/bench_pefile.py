#!/usr/bin/env python3
"""Reproducible benchmark: pefile vs pefile_mojo (checksum + import parsing).

Inputs are synthetic PE images generated locally (tests/test_pefile_fixtures
builder — deterministic, no system binaries, no network). Correctness against
the oracle (PyPI pefile) is asserted before any timing happens, so the numbers
below always come from a verified-correct build.

Two workloads, each timed cold (first call in a fresh OS process, median of
COLD_RUNS subprocesses) and warm (median of N_RUNS in-process repetitions):

* checksum: pefile's `PE(data).generate_checksum()` end-to-end (includes its
  mandatory full parse + re-serialization), pefile's `pe.generate_checksum()`
  with the PE object pre-built (still includes write()), and
  `pefile_mojo.generate_checksum(data, offset)` on both backends.
* imports: pefile's scoped import parse `PE(data, fast_load=True)` +
  `parse_data_directories(directories=[1])` vs `pefile_mojo.parse_imports(data)`
  on both backends. (pefile's full-parse construction is also shown for
  reference; it is a superset of the job.)

Run from the repository root:

    PYTHONPATH=python/pefile_mojo python benchmarks/bench_pefile.py
"""

from __future__ import annotations

import os
import platform
import statistics
import subprocess
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tests"))

N_RUNS = 7  # warm timings: median over this many repetitions
COLD_RUNS = 7  # cold timings: median over this many fresh processes
CHECKSUM_SIZES = [64 * 1024, 1024 * 1024, 8 * 1024 * 1024]
IMPORT_DLLS = 64
IMPORT_FUNCS = 56  # 64*(56+1 ordinal) entries/table x2 tables < pefile's 8192 cap


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
    env["PYTHONPATH"] = os.path.join(repo, "python", "pefile_mojo") + os.pathsep + os.path.join(
        repo, "tests"
    ) + os.pathsep + env.get("PYTHONPATH", "")
    if env_extra:
        env.update(env_extra)
    samples = []
    for _ in range(COLD_RUNS):
        t0 = time.perf_counter()
        subprocess.run(
            [sys.executable, "-c", code], env=env, check=True,
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
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


def main() -> None:
    import pefile

    import pefile_mojo
    from test_pefile_fixtures import make_big_pe

    info = pefile_mojo.backend_info()
    print("== environment ==")
    print(machine_info())
    print(
        f"- pefile_mojo backend: {'native' if info['native_available'] else 'FALLBACK'} "
        f"({info.get('native_source') or info.get('error')})"
    )
    print(f"- pefile (oracle): {pefile.__version__}")
    print(f"- warm: median of {N_RUNS}; cold: median of {COLD_RUNS} fresh processes")
    if not info["native_available"]:
        sys.exit("native kernel unavailable; refusing to benchmark the fallback as 'pefile_mojo'")

    from pefile_mojo import _reference

    # -- workloads ----------------------------------------------------------
    ck_blobs = {
        size: make_big_pe(n_dlls=4, n_funcs=4, bits=32, text_bytes=size - 1536)
        for size in CHECKSUM_SIZES
    }
    imp_blob = make_big_pe(n_dlls=IMPORT_DLLS, n_funcs=IMPORT_FUNCS, bits=64)

    # -- correctness gate ---------------------------------------------------
    print("\n== correctness gate (bit-exact vs pefile) ==")
    for size, blob in ck_blobs.items():
        pe = pefile.PE(data=blob)
        off = pefile_mojo.checksum_field_offset(blob)
        want = pe.generate_checksum()
        got_native = pefile_mojo.generate_checksum(blob, off)
        got_fallback = _reference.checksum(blob, off)
        ok = want == got_native == got_fallback
        print(f"  checksum {size:>9,} B: oracle={want} native={got_native} "
              f"fallback={got_fallback} [{'OK' if ok else 'FAIL'}]")
        if not ok:
            sys.exit("correctness gate failed (checksum)")

    def oracle_imports(blob):
        pe = pefile.PE(data=blob)
        return [
            (d.dll, [(s.name, s.ordinal, s.hint, s.address, s.bound) for s in d.imports])
            for d in pe.DIRECTORY_ENTRY_IMPORT
        ]

    def ours_imports(blob):
        return [
            (d.dll, [(s.name, s.ordinal, s.hint, s.address, s.bound) for s in d.imports])
            for d in pefile_mojo.parse_imports(blob)
        ]

    ref_rows = _reference.parse_imports(pefile_mojo.core._parse_headers(imp_blob))
    ours = [(dll, [(s["name"], s["ordinal"], s["hint"], s["address"], s["bound"]) for s in syms])
            for dll, syms in ref_rows]
    want_imp = oracle_imports(imp_blob)
    got_imp = ours_imports(imp_blob)
    ok = want_imp == got_imp == ours
    n_syms = sum(len(d.imports) for d in pefile.PE(data=imp_blob).DIRECTORY_ENTRY_IMPORT)
    print(f"  imports {IMPORT_DLLS} DLLs x {IMPORT_FUNCS + 1} ({n_syms} symbols): "
          f"[{'OK' if ok else 'FAIL'}]")
    if not ok:
        sys.exit("correctness gate failed (imports)")

    # -- checksum warm ------------------------------------------------------
    print("\n== checksum: warm (median per call) ==")
    print(f"{'size':>10} | {'pefile e2e':>12} | {'pefile cs-only':>15} | "
          f"{'mojo native':>12} | {'mojo fallback':>14} | {'speedup*':>9}")
    ck_rows = []
    for size, blob in ck_blobs.items():
        off = pefile_mojo.checksum_field_offset(blob)
        t_e2e = median(lambda: pefile.PE(data=blob).generate_checksum())
        pe = pefile.PE(data=blob)  # pre-built: generate_checksum still re-serializes
        t_cs = median(lambda: pe.generate_checksum())
        t_nat = median(lambda: pefile_mojo.generate_checksum(blob, off))
        t_fb = median(lambda: _reference.checksum(blob, off))
        ck_rows.append((size, t_e2e, t_cs, t_nat, t_fb))
        print(f"{size:>10,} | {1e3*t_e2e:>10.3f}ms | {1e3*t_cs:>13.3f}ms | "
              f"{1e3*t_nat:>10.4f}ms | {1e3*t_fb:>12.3f}ms | {t_cs/t_nat:>8.1f}x")
    print("* vs pefile generate_checksum with pre-built PE (the conservative comparison)")

    # -- imports warm -------------------------------------------------------
    print("\n== imports: warm (median per call) ==")

    def oracle_scoped():
        pe = pefile.PE(data=imp_blob, fast_load=True)
        pe.parse_data_directories(directories=[1])
        return pe

    t_scoped = median(oracle_scoped)
    t_full = median(lambda: pefile.PE(data=imp_blob))
    t_nat = median(lambda: pefile_mojo.parse_imports(imp_blob))
    t_fb = median(
        lambda: _reference.parse_imports(pefile_mojo.core._parse_headers(imp_blob))
    )
    print(f"  pefile scoped (fast_load + parse import dir): {1e3*t_scoped:>9.3f} ms")
    print(f"  pefile full parse (superset of the job)     : {1e3*t_full:>9.3f} ms")
    print(f"  pefile_mojo native                          : {1e3*t_nat:>9.4f} ms "
          f"({t_scoped/t_nat:.1f}x vs scoped)")
    print(f"  pefile_mojo fallback                        : {1e3*t_fb:>9.3f} ms "
          f"({t_scoped/t_fb:.2f}x vs scoped)")

    # -- cold ---------------------------------------------------------------
    print("\n== cold: import + first call in a fresh process (median) ==")
    setup_ck = (
        "import pefile, pefile_mojo; from test_pefile_fixtures import make_big_pe; "
        f"b = make_big_pe(n_dlls=4, n_funcs=4, bits=32, text_bytes={1024 * 1024 - 1536}); "
        "off = pefile_mojo.checksum_field_offset(b); "
    )
    cold_ck_oracle = cold_time(
        setup_ck + "pefile.PE(data=b).generate_checksum()"
    )
    cold_ck_mojo = cold_time(setup_ck + "pefile_mojo.generate_checksum(b, off)")
    cold_ck_mojo_fb = cold_time(
        setup_ck + "pefile_mojo.generate_checksum(b, off)",
        env_extra={"PEFILE_MOJO_DISABLE_NATIVE": "1"},
    )
    setup_imp = (
        "import pefile, pefile_mojo; from test_pefile_fixtures import make_big_pe; "
        f"b = make_big_pe(n_dlls={IMPORT_DLLS}, n_funcs={IMPORT_FUNCS}, bits=64); "
    )
    cold_imp_oracle = cold_time(
        setup_imp
        + "pe = pefile.PE(data=b, fast_load=True); pe.parse_data_directories(directories=[1])"
    )
    cold_imp_mojo = cold_time(setup_imp + "pefile_mojo.parse_imports(b)")
    cold_imp_mojo_fb = cold_time(
        setup_imp + "pefile_mojo.parse_imports(b)",
        env_extra={"PEFILE_MOJO_DISABLE_NATIVE": "1"},
    )
    print(f"  checksum 1 MiB: pefile {1e3*cold_ck_oracle:.1f} ms | "
          f"mojo native {1e3*cold_ck_mojo:.1f} ms | mojo fallback {1e3*cold_ck_mojo_fb:.1f} ms")
    print(f"  imports {IMPORT_DLLS} DLLs: pefile {1e3*cold_imp_oracle:.1f} ms | "
          f"mojo native {1e3*cold_imp_mojo:.1f} ms | mojo fallback {1e3*cold_imp_mojo_fb:.1f} ms")

    # -- README paste block ---------------------------------------------------
    print("\n== README paste block ==")
    print("### Checksum (warm, median per call)")
    print("| image size | pefile PE+generate_checksum | pefile generate_checksum (pre-built PE) "
          "| pefile_mojo native | pefile_mojo fallback | speedup (vs pre-built) |")
    print("|---:|---:|---:|---:|---:|---:|")
    for size, t_e2e, t_cs, t_n, t_f in ck_rows:
        print(f"| {size/1024:,.0f} KiB | {1e3*t_e2e:.3f} ms | {1e3*t_cs:.3f} ms "
              f"| {1e3*t_n:.4f} ms | {1e3*t_f:.3f} ms | {t_cs/t_n:.1f}x |")
    print("\n### Import parse (warm, median per call)")
    print(f"| workload ({n_syms} imported symbols) | time | speedup vs pefile scoped |")
    print("|---|---:|---:|")
    print(f"| pefile scoped import parse | {1e3*t_scoped:.3f} ms | 1.0x |")
    print(f"| pefile full PE parse (superset) | {1e3*t_full:.3f} ms | "
          f"{t_scoped/t_full:.2f}x |")
    print(f"| pefile_mojo native | {1e3*t_nat:.4f} ms | {t_scoped/t_nat:.1f}x |")
    print(f"| pefile_mojo fallback | {1e3*t_fb:.3f} ms | {t_scoped/t_fb:.2f}x |")
    print("\n### Cold start (fresh process: interpreter import + first call, median)")
    print("| workload | pefile | pefile_mojo native | pefile_mojo fallback |")
    print("|---|---:|---:|---:|")
    print(f"| checksum, 1 MiB image | {1e3*cold_ck_oracle:.1f} ms "
          f"| {1e3*cold_ck_mojo:.1f} ms | {1e3*cold_ck_mojo_fb:.1f} ms |")
    print(f"| import parse, {IMPORT_DLLS} DLLs | {1e3*cold_imp_oracle:.1f} ms "
          f"| {1e3*cold_imp_mojo:.1f} ms | {1e3*cold_imp_mojo_fb:.1f} ms |")


if __name__ == "__main__":
    main()

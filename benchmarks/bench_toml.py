#!/usr/bin/env python3
"""Reproducible benchmark: tomllib vs tomlkit vs toml_mojo (native + fallback).

Documents are generated locally from fixed seeds (no network, no datasets):
a small ~1 KB pyproject-style config, a ~64 KB mixed document, and a ~1 MB
mixed document (tables, dotted keys, arrays, inline tables, all scalar types,
datetimes, multiline strings, comments).

- warm steady-state: per-call latency, median of 5 runs (each run averages
  enough iterations for ~150 ms), per backend and document.
- cold first-call: the very first `loads()` in a fresh Python process
  (includes the native library load + ABI handshake for toml_mojo), median
  of 5 processes.

Correctness is asserted (strict structural equality vs tomllib, floats
bit-exact) before any timing happens, so the numbers below always come from
a verified-correct build.

Run from the repository root:

    PYTHONPATH=python/toml_mojo pixi run python benchmarks/bench_toml.py
"""

from __future__ import annotations

import os
import platform
import random
import statistics
import string
import subprocess
import sys
import tempfile
import time

SEED = 20260919
N_RUNS = 5

# ---------------------------------------------------------------------------
# Document generators (deterministic)
# ---------------------------------------------------------------------------


def _small_config() -> str:
    return """\
# Example project configuration
[project]
name = "example-app"
version = "1.4.2"
description = "A small TOML document with all common value shapes"
requires-python = ">=3.11"
authors = [{name = "Algenta"}]
keywords = ["toml", "config", "parser"]
released = 2026-09-19T09:30:00Z

[project.optional-dependencies]
test = ["pytest>=8", "coverage"]
docs = ["sphinx"]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.example]
enabled = true
retries = 3
timeout = 2.5
ratio = 0.125
big = 12345678901234567890
mask = 0xDEADBEEF
when = 2026-09-19
at = 09:30:00.123456
paths = ["/usr/local/bin", "/opt/bin"]
note = \"\"\"multi
line
string\"\"\"

[tool.example.env]
dev = {debug = true, level = 5}
prod = {debug = false, level = 1}
"""


def _rand_text(rng: random.Random, n: int) -> str:
    pool = string.ascii_letters + string.digits + "    .,;:!?-_()[]{}"
    return "".join(rng.choice(pool) for _ in range(n))


def _mixed_doc(target_bytes: int, seed: int) -> str:
    rng = random.Random(seed)
    out: list[str] = ["# generated mixed TOML document", "root_key = 1"]
    size = len(out[0]) + len(out[1]) + 2
    t = 0
    while size < target_bytes:
        t += 1
        lines = [f"[section_{t}]", f"name_{t} = \"{_rand_text(rng, 24)}\""]
        for i in range(rng.randint(4, 14)):
            kind = rng.random()
            if kind < 0.25:
                lines.append(f"int_{t}_{i} = {rng.randint(-10**12, 10**12)}")
            elif kind < 0.4:
                lines.append(f"float_{t}_{i} = {rng.uniform(-1e4, 1e4):.6f}e{rng.randint(-9, 9)}")
            elif kind < 0.5:
                lines.append(
                    f"arr_{t}_{i} = [{', '.join(str(rng.randint(0, 999)) for _ in range(rng.randint(0, 8)))}]"
                )
            elif kind < 0.6:
                lines.append(
                    f"dt_{t}_{i} = {rng.randint(1970, 2030):04d}-{rng.randint(1, 12):02d}-{rng.randint(1, 28):02d}T"
                    f"{rng.randint(0, 23):02d}:{rng.randint(0, 59):02d}:{rng.randint(0, 59):02d}"
                    f"{rng.choice(['Z', '+02:00', '-05:30', ''])}"
                )
            elif kind < 0.7:
                lines.append(
                    f"inl_{t}_{i} = {{a = {rng.randint(0, 9)}, b = \"{_rand_text(rng, 8)}\", c.d = true}}"
                )
            elif kind < 0.8:
                lines.append(f"str_{t}_{i} = \"{_rand_text(rng, rng.randint(0, 60))}\"")
            elif kind < 0.9:
                lines.append(f"flag_{t}_{i} = {rng.choice(['true', 'false'])}")
            else:
                lines.append(f"hex_{t}_{i} = 0x{rng.getrandbits(32):08X}")
        if rng.random() < 0.15:
            lines.append(f"ml_{t} = \"\"\"{_rand_text(rng, 80)}\n{_rand_text(rng, 40)}\"\"\"")
        if rng.random() < 0.2:
            lines.append(f"# comment about section {t}")
        block = "\n".join(lines)
        out.append(block)
        size += len(block) + 1
    return "\n".join(out) + "\n"


# ---------------------------------------------------------------------------
# Timing helpers
# ---------------------------------------------------------------------------


def time_warm(fn, doc: str, min_run_time: float = 0.15) -> float:
    """Median per-call seconds over 5 runs of ~min_run_time each."""
    t0 = time.perf_counter()
    fn(doc)
    dt = time.perf_counter() - t0
    n = max(1, int(min_run_time / max(dt, 1e-6)))
    samples = []
    for _ in range(N_RUNS):
        t0 = time.perf_counter()
        for _ in range(n):
            fn(doc)
        samples.append((time.perf_counter() - t0) / n)
    return statistics.median(samples)


def time_cold(module: str, doc_path: str) -> float:
    """Median first-`loads` latency across 5 fresh Python processes."""
    code = (
        "import time, sys, {m} as mod\n"
        "doc = open(sys.argv[1], encoding='utf-8').read()\n"
        "t0 = time.perf_counter()\n"
        "mod.loads(doc)\n"
        "print(time.perf_counter() - t0)\n"
    ).format(m=module)
    samples = []
    env = dict(os.environ)
    for _ in range(N_RUNS):
        r = subprocess.run(
            [sys.executable, "-c", code, doc_path],
            capture_output=True,
            text=True,
            env=env,
            check=True,
        )
        samples.append(float(r.stdout.strip()))
    return statistics.median(samples)


def strict_eq(a, b) -> bool:
    import math
    import struct

    if type(a) is not type(b):
        return False
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(strict_eq(a[k], b[k]) for k in a)
    if isinstance(a, list):
        return len(a) == len(b) and all(strict_eq(x, y) for x, y in zip(a, b))
    if isinstance(a, float):
        if math.isnan(a) or math.isnan(b):
            return math.isnan(a) and math.isnan(b)
        return struct.pack("<d", a) == struct.pack("<d", b)
    return a == b


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


def main() -> None:
    import tomllib

    import toml_mojo

    try:
        import tomlkit
    except ImportError:
        tomlkit = None

    info = toml_mojo.backend_info()
    print("== environment ==")
    print(machine_info())
    print(
        f"- toml_mojo backend: {'native' if info['native_available'] else 'FALLBACK'} "
        f"({info.get('native_source') or info.get('error')})"
    )
    if tomlkit is not None:
        print(f"- tomlkit: {getattr(tomlkit, '__version__', 'unknown')}")
    print(f"- seeds: docs from SEED={SEED}; runs: median of {N_RUNS}")
    if not info["native_available"]:
        sys.exit("native kernel unavailable; refusing to benchmark the fallback as 'toml_mojo'")

    docs = {
        "small (~1 KB config)": _small_config(),
        "medium (~64 KB)": _mixed_doc(64 * 1024, SEED + 1),
        "large (~1 MB)": _mixed_doc(1024 * 1024, SEED + 2),
    }

    print("\n== correctness gate (strict equality vs tomllib, floats bit-exact) ==")
    for name, doc in docs.items():
        ref = tomllib.loads(doc)
        ours = toml_mojo.loads(doc)
        ok = strict_eq(ours, ref)
        print(f"  {name:>22}: {len(doc):>9,} bytes  [{'OK' if ok else 'FAIL'}]")
        if not ok:
            sys.exit(f"correctness gate failed for {name}")
        if tomlkit is not None:
            tk = tomlkit.loads(doc)
            if not strict_eq(tk.unwrap(), ref):
                sys.exit(f"tomlkit unwrap disagrees with tomllib for {name}")

    backends = [("tomllib", tomllib.loads)]
    if tomlkit is not None:
        backends.append(("tomlkit", tomlkit.loads))
    backends.append(("toml_mojo (native)", toml_mojo.loads))
    # Measured last, with the fallback forced for exactly its own block:
    # the fallback *is* tomllib plus the wrapper's resolver overhead.
    backends.append(("toml_mojo (fallback)", None))

    print("\n== cold first-call (fresh process, median of 5) ==")
    with tempfile.TemporaryDirectory() as td:
        for name, doc in docs.items():
            path = os.path.join(td, "doc.toml")
            with open(path, "w", encoding="utf-8") as f:
                f.write(doc)
            row = [f"{name:>22}"]
            for label, _ in backends:
                if "fallback" in label:
                    continue  # fallback cold == tomllib cold plus one env lookup
                module = {"tomllib": "tomllib", "tomlkit": "tomlkit"}.get(label, "toml_mojo")
                ms = 1e3 * time_cold(module, path)
                row.append(f"{ms:>10.3f} ms")
            print("  " + " | ".join(row))

    print("\n== warm steady-state (per call, median of 5 runs) ==")
    header = f"{'document':>22} | {'backend':>22} | {'ms/call':>10} | {'MB/s':>9} | {'vs tomllib':>10}"
    print(header)
    print(f"{'-' * 22}-+-{'-' * 22}-+-{'-' * 10}-+-{'-' * 9}-+-{'-' * 10}")
    readme_rows = []
    for name, doc in docs.items():
        base = None
        for label, fn in backends:
            if label == "toml_mojo (fallback)":
                os.environ["TOML_MOJO_DISABLE_NATIVE"] = "1"
                try:
                    sec = time_warm(toml_mojo.loads, doc)
                finally:
                    del os.environ["TOML_MOJO_DISABLE_NATIVE"]
            else:
                sec = time_warm(fn, doc)
            if label == "tomllib":
                base = sec
            ms = sec * 1e3
            mbps = len(doc) / sec / 1e6
            speedup = (base / sec) if base else 1.0
            print(f"{name:>22} | {label:>22} | {ms:>10.4f} | {mbps:>9.1f} | {speedup:>9.2f}x")
            readme_rows.append((name, label, ms, mbps, speedup))

    print("\n== README paste block ==")
    print("| document | backend | ms/call | MB/s | vs tomllib |")
    print("|---|---:|---:|---:|---:|")
    for name, label, ms, mbps, speedup in readme_rows:
        print(f"| {name} | {label} | {ms:.4f} | {mbps:.1f} | {speedup:.2f}x |")


if __name__ == "__main__":
    main()

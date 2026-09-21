#!/usr/bin/env python3
"""Reproducible benchmark: PyPI python-slugify vs slugify_mojo (native + fallback).

Correctness is gated first: every corpus text must produce a byte-identical
slug on the native and fallback backends, or the benchmark aborts. Timings:

- cold first-call: wall time of import + table load + first slugify() in a
  fresh process, median of 5 launches.
- warm steady-state: per-call latency over each workload, median of 5
  batches, for the oracle, the native kernel and the forced fallback;
  slugify_column is timed as a full column pass.

The corpus is generated locally from fixed seeds (same style as the
differential suite) — no network, no datasets.

Run from the repository root, e.g.:

    PYTHONPATH=python/slugify_mojo /tmp/slugify-mojo-venv/bin/python benchmarks/bench_slugify.py
"""

from __future__ import annotations

import os
import platform
import random
import statistics
import subprocess
import sys
import time

N_RUNS = 5  # median over this many batches / launches

_rng = random.Random(20260920)

ASCII_TITLES = [
    "The Quick Brown Fox Jumps Over the Lazy Dog!",
    "Top 10 Ways to Optimize Your CMS Pipeline",
    "Hello World: A Beginner's Guide to Slugs",
    "Summer Sale 2026 — Up to 50% Off Everything",
    "How to Cook the Perfect Steak (Step by Step)",
] * 4  # 20 short ASCII titles

ACCENTED_TITLES = [
    "Déjà Vu — Café naïve résumé à la carte",
    "Árvíztűrő tükörfúrógép: Hungarian Showcase",
    "Crème brûlée, piñata, señor, Zürich",
    "Œuvres complètes de Molière, tome ½",
    "İstanbul'da kahvaltı nerede yenir?",
] * 4  # 20 short accented titles

MIXED_SCRIPTS = [
    "Москва — столица России, 日本語のテキスト, Ελληνικά",
    "Программирование на Python для начинающих",
    "中文标题：如何提高网站流量和搜索引擎排名",
    "مرحبا بالعالم، هذا اختبار للنص العربي",
    "한국어 제목과 English Words 함께 사용하기",
] * 4  # 20 mixed-script titles

LONG_TEXT = (
    "The Quick Brown Fox — Déjà Vu! Café &amp; Restaurant №5 "
    "предлагает 日本語のメニュー και ελληνικά πιάτα 😀 "
) * 12  # ~1000 chars

ENTITY_TEXTS = [
    "Fish &amp; Chips &lt;tag&gt; &#65;&#x42; &eacute;tude",
    "Tom &amp; Jerry &#8212; The &quot;Movie&quot; &#169;",
    "AT&amp;T &gt; Verizon &amp; T-Mobile &#8211; chart",
    "&Aacute;&Eacute;&Iacute;&Oacute;&Uacute; &#x1F600; test",
] * 5  # 20 entity-heavy titles


def _all_workloads() -> dict[str, list[str]]:
    return {
        "ascii": ASCII_TITLES,
        "accented": ACCENTED_TITLES,
        "mixed": MIXED_SCRIPTS,
        "entities": ENTITY_TEXTS,
    }


COLD_SNIPPETS = {
    "oracle": (
        "import time; t0=time.perf_counter();"
        "from slugify import slugify;"
        "slugify('Déjà Vu — Café naïve!');"
        "print(f'{time.perf_counter()-t0:.6f}')"
    ),
    "native": (
        "import time; t0=time.perf_counter();"
        "import slugify_mojo;"
        "slugify_mojo.slugify('Déjà Vu — Café naïve!');"
        "print(f'{time.perf_counter()-t0:.6f}')"
    ),
    "fallback": (
        "import time; t0=time.perf_counter();"
        "import slugify_mojo;"
        "slugify_mojo.slugify('Déjà Vu — Café naïve!');"
        "print(f'{time.perf_counter()-t0:.6f}')"
    ),
}


def cold_start(which: str) -> float:
    """Median seconds for import + first slugify in a fresh process."""
    env = dict(os.environ)
    env["PYTHONNOUSERSITE"] = "1"
    if which == "fallback":
        env["SLUGIFY_MOJO_DISABLE_NATIVE"] = "1"
    samples = []
    for _ in range(N_RUNS):
        out = subprocess.run(
            [sys.executable, "-c", COLD_SNIPPETS[which]],
            capture_output=True,
            text=True,
            env=env,
            check=True,
        )
        samples.append(float(out.stdout.strip()))
    return statistics.median(samples)


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
    from slugify import slugify as oracle_slugify

    import slugify_mojo

    info = slugify_mojo.backend_info()
    print("== environment ==")
    print(machine_info())
    print(
        f"- slugify_mojo backend: {'native' if info['native_available'] else 'FALLBACK'} "
        f"({info.get('native_source') or info.get('error')})"
    )
    workloads = _all_workloads()
    total = sum(len(v) for v in workloads.values()) + 1
    print(f"- corpus: {total} texts across {len(workloads)} workloads + 1 long text")
    print(f"- runs: median of {N_RUNS}")
    if not info["native_available"]:
        sys.exit("native kernel unavailable; refusing to benchmark the fallback as 'native'")

    print("\n== correctness gate (byte-exact vs oracle) ==")
    corpus = [t for texts in workloads.values() for t in texts] + [LONG_TEXT]
    for disable, label in ((False, "native"), (True, "fallback")):
        os.environ["SLUGIFY_MOJO_DISABLE_NATIVE"] = "1" if disable else "0"
        for text in corpus:
            ref = oracle_slugify(text)
            ours = slugify_mojo.slugify(text)
            if ours != ref:
                sys.exit(f"correctness gate failed ({label}) for {text[:50]!r}: {ours} != {ref}")
        batch = slugify_mojo.slugify_column(corpus)
        refs = [oracle_slugify(t) for t in corpus]
        if batch != refs:
            sys.exit(f"correctness gate failed ({label}) for slugify_column")
        print(f"  {label}: all {len(corpus)} texts + slugify_column byte-identical")
    os.environ["SLUGIFY_MOJO_DISABLE_NATIVE"] = "0"

    def time_calls(fn, texts, inner_repeats=1) -> float:
        """Median seconds for one full pass over `texts`."""
        samples = []
        for _ in range(N_RUNS):
            t0 = time.perf_counter()
            for _ in range(inner_repeats):
                for text in texts:
                    fn(text)
            samples.append((time.perf_counter() - t0) / inner_repeats)
        return statistics.median(samples)

    print("\n== warm steady-state: slugify() latency per workload ==")
    rows = []
    for name, texts in workloads.items():
        inner = max(1, 2000 // len(texts))
        t_oracle = time_calls(oracle_slugify, texts, inner)
        t_native = time_calls(slugify_mojo.slugify, texts, inner)
        os.environ["SLUGIFY_MOJO_DISABLE_NATIVE"] = "1"
        t_fallback = time_calls(slugify_mojo.slugify, texts, inner)
        os.environ["SLUGIFY_MOJO_DISABLE_NATIVE"] = "0"
        per_oracle, per_native, per_fallback = (
            1e6 * t / len(texts) for t in (t_oracle, t_native, t_fallback)
        )
        rows.append((name, len(texts), per_oracle, per_native, per_fallback))
        print(
            f"  {name:9s}: oracle {per_oracle:7.2f} µs | native {per_native:6.2f} µs "
            f"({per_oracle/per_native:5.1f}x) | fallback {per_fallback:6.2f} µs "
            f"({per_oracle/per_fallback:5.2f}x)"
        )

    print("\n== warm steady-state: slugify() on the long (~1000 chars) text ==")
    lo = time_calls(oracle_slugify, [LONG_TEXT], 50)
    ln = time_calls(slugify_mojo.slugify, [LONG_TEXT], 50)
    os.environ["SLUGIFY_MOJO_DISABLE_NATIVE"] = "1"
    lf = time_calls(slugify_mojo.slugify, [LONG_TEXT], 50)
    os.environ["SLUGIFY_MOJO_DISABLE_NATIVE"] = "0"
    print(f"  oracle:  {1e6*lo:8.1f} µs/call")
    print(f"  native:  {1e6*ln:8.1f} µs/call  ({lo/ln:.1f}x)")
    print(f"  fallback:{1e6*lf:8.1f} µs/call  ({lo/lf:.2f}x)")

    print("\n== warm steady-state: slugify_column() over 1000 short titles ==")
    _base = ASCII_TITLES + ACCENTED_TITLES + MIXED_SCRIPTS + ENTITY_TEXTS  # 80
    column = (_base * 13)[:1000]
    assert len(column) == 1000

    def oracle_column(texts):
        for t in texts:
            oracle_slugify(t)

    samples = []
    for _ in range(N_RUNS):
        t0 = time.perf_counter()
        oracle_column(column)
        samples.append(time.perf_counter() - t0)
    co = statistics.median(samples)
    samples = []
    for _ in range(N_RUNS):
        t0 = time.perf_counter()
        slugify_mojo.slugify_column(column)
        samples.append(time.perf_counter() - t0)
    cn = statistics.median(samples)
    os.environ["SLUGIFY_MOJO_DISABLE_NATIVE"] = "1"
    samples = []
    for _ in range(N_RUNS):
        t0 = time.perf_counter()
        slugify_mojo.slugify_column(column)
        samples.append(time.perf_counter() - t0)
    cf = statistics.median(samples)
    os.environ["SLUGIFY_MOJO_DISABLE_NATIVE"] = "0"
    print(f"  oracle loop: {1e3*co:8.2f} ms/column")
    print(f"  native batch:{1e3*cn:8.2f} ms/column  ({co/cn:.1f}x)")
    print(f"  fallback:    {1e3*cf:8.2f} ms/column  ({co/cf:.2f}x)")

    print("\n== cold first-call (fresh process: import + table + first slugify) ==")
    c_oracle = cold_start("oracle")
    c_native = cold_start("native")
    c_fallback = cold_start("fallback")
    print(f"  oracle:  {1e3*c_oracle:8.1f} ms")
    print(f"  native:  {1e3*c_native:8.1f} ms  ({c_oracle/c_native:.2f}x)")
    print(f"  fallback:{1e3*c_fallback:8.1f} ms  ({c_oracle/c_fallback:.2f}x)")

    print("\n== README paste block ==")
    print("Warm steady-state `slugify()` (median of 5 batches):")
    print()
    print("| workload | PyPI python-slugify | slugify-mojo (native) | slugify-mojo (fallback) | native speedup |")
    print("|---|---:|---:|---:|---:|")
    for name, n, per_oracle, per_native, per_fallback in rows:
        print(
            f"| {n}× {name} titles, per call | {per_oracle:.2f} µs | {per_native:.2f} µs "
            f"| {per_fallback:.2f} µs | {per_oracle/per_native:.1f}x |"
        )
    print(
        f"| long text ({len(LONG_TEXT)} chars), per call | {1e6*lo:.1f} µs | {1e6*ln:.1f} µs "
        f"| {1e6*lf:.1f} µs | {lo/ln:.1f}x |"
    )
    print(
        f"| slugify_column, 1000 mixed titles | {1e3*co:.2f} ms | {1e3*cn:.2f} ms "
        f"| {1e3*cf:.2f} ms | {co/cn:.1f}x |"
    )
    print()
    print("Cold first-call (fresh process, median of 5):")
    print()
    print("| package | import + first slugify |")
    print("|---|---:|")
    print(f"| PyPI python-slugify | {1e3*c_oracle:.1f} ms |")
    print(f"| slugify-mojo (native) | {1e3*c_native:.1f} ms |")
    print(f"| slugify-mojo (fallback) | {1e3*c_fallback:.1f} ms |")


if __name__ == "__main__":
    main()

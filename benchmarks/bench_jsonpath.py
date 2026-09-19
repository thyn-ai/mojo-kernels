#!/usr/bin/env python3
"""Reproducible benchmark: jsonpath_ng (ext) vs jsonpath_mojo.

Datasets are generated locally from fixed seeds (no network, no datasets):
store-shaped JSON documents with 100 / 2,000 / 20,000 book records. The query
battery exercises the whole supported scope: child chains, wildcards,
recursive descent, slices with steps, field unions, and filter scripts
(comparisons, `&` conjunctions, arith, existence).

Method: correctness is asserted (exact (value, path) parity with the oracle)
before any timing happens, so the numbers below always come from a
verified-correct build. COLD = wall time of the first call on a fresh
interpreter state for that (dataset, backend) pair (includes one-time
expression parsing setup). WARM = median of 5 runs of the full battery.
The oracle is invoked as `jsonpath_ng.ext.parse(expr).find(data)` per query —
exactly how the library is used; jsonpath_mojo is invoked as
`jsonpath_mojo.find(expr, data)`.

Run from the repository root (oracle must be importable, see
scripts/test_all_jsonpath.sh):

    PYTHONPATH=python/jsonpath_mojo:/tmp/jporacle-lib python benchmarks/bench_jsonpath.py
"""

from __future__ import annotations

import os
import platform
import random
import statistics
import subprocess
import sys
import time

N_RUNS = 5
SEED = 20260919
BOOK_COUNTS = [100, 2_000, 20_000]
QUERIES = [
    "$.store.book[0].title",
    "$.store.book[*].author",
    "$..author",
    "$..price",
    "$..*",
    "$.store.book[1:500:7].title",
    "$.store.book[::-1].title",
    "$['store','expensive']",
    "$.store.book[?(@.price < 10)].title",
    "$.store.book[?(@.price < 20 & @.category == 'fiction')].title",
    "$.store.book[?(@.price * 2 > 30)].price",
    "$.store.book[?(@.isbn)].title",
    "$..book[?(@.price > 15)].author",
    "$.store.book[?(@.category == 'fiction' & @.price > 5 & @.isbn)].title",
]

CATEGORIES = ["fiction", "reference", "sci-fi", "history", "tech"]
AUTHORS = [
    "Nigel Rees", "Evelyn Waugh", "Herman Melville", "J. R. R. Tolkien",
    "Ursula K. Le Guin", "Octavia Butler", "Ted Chiang", "Ada Palmer",
]


def make_doc(seed: int, n_books: int) -> dict:
    rng = random.Random(seed)
    books = []
    for i in range(n_books):
        book = {
            "category": rng.choice(CATEGORIES),
            "author": rng.choice(AUTHORS),
            "title": f"Book title number {i}",
            "price": round(rng.uniform(3.0, 60.0), 2),
        }
        if rng.random() < 0.7:
            book["isbn"] = f"0-{rng.randint(100, 999)}-{rng.randint(10000, 99999)}-X"
        books.append(book)
    return {
        "store": {
            "book": books,
            "bicycle": {"color": "red", "price": 19.95},
        },
        "expensive": 15,
    }


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


def check_parity(doc) -> None:
    """Every backend must agree with the oracle exactly, on every query."""
    from jsonpath_ng.ext import parse as oracle_parse

    import jsonpath_mojo

    for expr in QUERIES:
        want = [(m.value, str(m.full_path)) for m in oracle_parse(expr).find(doc)]
        got_native = jsonpath_mojo.find(expr, doc)
        assert got_native == want, f"native parity failed on {expr!r}"
        os.environ["JSONPATH_MOJO_DISABLE_NATIVE"] = "1"
        try:
            got_fallback = jsonpath_mojo.find(expr, doc)
        finally:
            del os.environ["JSONPATH_MOJO_DISABLE_NATIVE"]
        assert got_fallback == want, f"fallback parity failed on {expr!r}"


def bench(doc, oracle_find, our_find) -> dict:
    """Cold (first-call) and warm (median-of-5 battery) timings per backend."""
    # Guard the native timing pass against a stray fallback override leaked
    # from the surrounding environment (or a prior failed parity check).
    os.environ.pop("JSONPATH_MOJO_DISABLE_NATIVE", None)
    results = {}
    for name, fn in (("oracle", oracle_find), ("mojo-native", our_find), ("mojo-fallback", None)):
        if fn is None:
            os.environ["JSONPATH_MOJO_DISABLE_NATIVE"] = "1"
            fn = our_find
        try:
            t0 = time.perf_counter()
            for expr in QUERIES:
                fn(expr, doc)
            cold = time.perf_counter() - t0
            samples = []
            for _ in range(N_RUNS):
                t0 = time.perf_counter()
                for expr in QUERIES:
                    fn(expr, doc)
                samples.append(time.perf_counter() - t0)
            results[name] = (cold, statistics.median(samples))
        finally:
            if name == "mojo-fallback":
                del os.environ["JSONPATH_MOJO_DISABLE_NATIVE"]
    return results


def main() -> None:
    from jsonpath_ng.ext import parse as oracle_parse

    import jsonpath_mojo

    info = jsonpath_mojo.backend_info()
    print("== environment ==")
    print(machine_info())
    try:
        import jsonpath_ng  # noqa: F401
        from importlib.metadata import version

        print(f"- jsonpath_ng: {version('jsonpath-ng')}")
    except Exception:  # noqa: BLE001
        pass
    print(f"- jsonpath_mojo backend: {'native' if info['native_available'] else 'FALLBACK'} "
          f"({info['native_source']})")
    print(f"- runs: cold = first battery call; warm = median of {N_RUNS} battery runs "
          f"({len(QUERIES)} queries per battery)")
    print()

    def oracle_find(expr, doc):
        return oracle_parse(expr).find(doc)

    def our_find(expr, doc):
        return jsonpath_mojo.find(expr, doc)

    for n_books in BOOK_COUNTS:
        doc = make_doc(SEED, n_books)
        check_parity(doc)
        res = bench(doc, oracle_find, our_find)
        print(f"== dataset: {n_books} books ==")
        print(f"{'backend':<16}{'cold (s)':>12}{'warm (s)':>12}{'speedup (warm)':>16}")
        base = res["oracle"][1]
        for name, (cold, warm) in res.items():
            speedup = f"{base / warm:,.1f}x" if warm > 0 else "n/a"
            print(f"{name:<16}{cold:>12.4f}{warm:>12.4f}{speedup:>16}")
        print()


if __name__ == "__main__":
    main()

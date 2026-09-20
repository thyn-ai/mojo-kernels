#!/usr/bin/env python3
"""Reproducible benchmark: pip jmespath vs jmespath_mojo.

Documents are generated locally from a fixed seed (no network, no datasets):

- **big-doc scan**: one ~0.85 MB AWS-describe-instances-style document
  (400 reservations x 10 instances), queried with filter/flatten/projection/
  aggregation/sort expressions — the shape JMESPath exists for.
- **small docs**: 5,000 log-record-style documents (~200 bytes each), one
  search per document — measures the per-call JSON-boundary cost honestly.
- **boundary floor**: the same big document with a trivial path expression
  (`a.b.c`-shaped) — the pure serialization round trip.

Correctness is asserted (structural equality vs pip jmespath) before any
timing happens, so the numbers always come from a verified-correct build.

The wrapper serializes the document with ``marshal.dumps`` (C-speed binary
walk, exact types) per call; the kernel parses, evaluates and returns the
result as JSON. That boundary is included in every number below — it is the
honest cost of a per-call native round trip.

Two timing modes per cell:

- **warm**: median of 5 batches in one process (steady state).
- **cold**: median of 7 fresh processes, timing only the first search call
  after imports (includes dlopen + runtime init for the native kernel).

Run from the repository root (the oracle comes from the pixi environment,
the wrapper from PYTHONPATH):

    PYTHONPATH=python/jmespath_mojo pixi run python benchmarks/bench_jmespath.py
"""

from __future__ import annotations

import json
import os
import platform
import random
import statistics
import subprocess
import sys
import time

SEED = 20260919
N_WARM_RUNS = 5
N_COLD_RUNS = 7
BIG_RESERVATIONS = 400
BIG_INSTANCES = 10
N_SMALL_DOCS = 5_000
SMALL_BATCH = 200  # searches per warm batch in the small-doc cell

BIG_QUERIES = {
    "filter+flatten+project": (
        "reservations[].instances[?state=='running'].id | []"
    ),
    "filter+and+project": (
        "reservations[].instances[?state=='running' && type=='m5.large'].id | []"
    ),
    "flatten+avg": "avg(reservations[].instances[].metrics.vcpus)",
    "sort_by+pipe": "sort_by(reservations[].instances[], &metrics.vcpus)[*].id | [0]",
    "sort 32k floats+pick": (
        "sort(reservations[].instances[].metrics.cpu[]) | [16000]"
    ),
    "max of 32k floats": "max(reservations[].instances[].metrics.cpu[])",
    "sum of 32k floats": "sum(reservations[].instances[].metrics.cpu[])",
    "object-wildcard": "reservations[].tags.*",
    "merge+pipe": "merge(reservations[].tags | [0], `{\"owner\": \"bench\"}`)",
}

SMALL_QUERY = "detail.user.roles[?active].name | [0]"
FLOOR_QUERY = "reservations[0].instances[0].id"


def make_big_doc(seed: int) -> dict:
    rng = random.Random(seed)
    states = ["running", "running", "running", "stopped", "pending", "terminated"]
    types = ["m5.large", "m5.xlarge", "t3.micro", "c6i.2xl"]
    reservations = []
    for r in range(BIG_RESERVATIONS):
        instances = []
        for i in range(BIG_INSTANCES):
            instances.append(
                {
                    "id": f"i-{r:04d}{i:04d}",
                    "state": rng.choice(states),
                    "type": rng.choice(types),
                    "az": rng.choice(["a", "b", "c"]),
                    "metrics": {
                        "vcpus": rng.choice([2, 4, 8, 16]),
                        "cpu": [round(rng.uniform(0, 100), 2) for _ in range(8)],
                    },
                    "tags": {"env": rng.choice(["dev", "prod"]), "team": f"t{rng.randint(0, 9)}"},
                }
            )
        reservations.append(
            {
                "id": f"r-{r:05d}",
                "owner": f"acct-{rng.randint(100, 999)}",
                "instances": instances,
                "tags": {"request": f"req-{r}", "spot": rng.choice([True, False])},
            }
        )
    return {"reservations": reservations}


def make_small_docs(seed: int) -> list[dict]:
    rng = random.Random(seed)
    docs = []
    for i in range(N_SMALL_DOCS):
        docs.append(
            {
                "ts": f"2026-09-19T10:{i % 60:02d}:{i % 60:02d}Z",
                "level": rng.choice(["info", "warn", "error"]),
                "detail": {
                    "user": {
                        "id": rng.randint(1, 10_000),
                        "roles": [
                            {"name": rng.choice(["admin", "dev", "ops", "viewer"]),
                             "active": rng.choice([True, False])}
                            for _ in range(3)
                        ],
                    },
                    "latency_ms": round(rng.uniform(0.5, 900), 1),
                },
            }
        )
    return docs


def _struct_equal(a, b) -> bool:
    if type(a) is not type(b):
        return False
    if isinstance(a, float):
        return repr(a) == repr(b) or (a != a and b != b)
    if isinstance(a, list):
        return len(a) == len(b) and all(_struct_equal(x, y) for x, y in zip(a, b))
    if isinstance(a, dict):
        return list(a.keys()) == list(b.keys()) and all(
            _struct_equal(a[k], b[k]) for k in a
        )
    return a == b


def check_parity(expr, data) -> None:
    ours = jmespath_mojo.search(expr, data)
    ref = jmespath.search(expr, data)
    if not _struct_equal(ours, ref):
        sys.exit(
            f"correctness gate failed for {expr!r}:\n  ours: {ours!r}\n  ref:  {ref!r}"
        )


def time_warm(search, expr, data, batch: int) -> float:
    """Median seconds over N_WARM_RUNS batches of `batch` searches."""
    samples = []
    for _ in range(N_WARM_RUNS):
        t0 = time.perf_counter()
        for _ in range(batch):
            search(expr, data)
        samples.append(time.perf_counter() - t0)
    return statistics.median(samples)


def time_warm_many(search, expr, docs) -> float:
    """Median seconds over N_WARM_RUNS runs of one search per document."""
    samples = []
    for _ in range(N_WARM_RUNS):
        t0 = time.perf_counter()
        for doc in docs:
            search(expr, doc)
        samples.append(time.perf_counter() - t0)
    return statistics.median(samples)


COLD_SNIPPET = """
import json, sys, time
sys.path.insert(0, {wrapper!r})
sys.path.insert(0, {oracle!r})
import {mod}
payload = json.load(open({docfile!r}))
expr, _ = payload["expr"], None
data = payload["data"]
t0 = time.perf_counter()
{mod}.search(expr, data)
print(time.perf_counter() - t0)
"""


def time_cold(module: str, expr: str, data) -> float:
    """Median seconds for the first search call in N_COLD_RUNS fresh processes."""
    wrapper = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "python", "jmespath_mojo"))
    oracle = os.environ.get("JMESPATH_ORACLE_PATH", "/tmp/jm-oracle")
    docfile = os.path.join(os.environ.get("TMPDIR", "/tmp"), f"bench_jm_cold_{os.getpid()}.json")
    with open(docfile, "w") as f:
        json.dump({"expr": expr, "data": data}, f)
    samples = []
    try:
        for _ in range(N_COLD_RUNS):
            snippet = COLD_SNIPPET.format(
                wrapper=wrapper, oracle=oracle, mod=module, docfile=docfile
            )
            out = subprocess.run(
                [sys.executable, "-c", snippet],
                capture_output=True,
                text=True,
                check=True,
            )
            samples.append(float(out.stdout.strip()))
    finally:
        os.unlink(docfile)
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
    global jmespath, jmespath_mojo
    import jmespath
    import jmespath_mojo

    info = jmespath_mojo.backend_info()
    print("== environment ==")
    print(machine_info())
    print(
        f"- jmespath_mojo backend: {'native' if info['native_available'] else 'FALLBACK'} "
        f"({info.get('native_source') or info.get('error')})"
    )
    print(f"- jmespath oracle: {jmespath.__version__}")
    print(f"- seeds: {SEED}; warm: median of {N_WARM_RUNS}; cold: median of {N_COLD_RUNS} fresh processes")
    if not info["native_available"]:
        sys.exit("native kernel unavailable; refusing to benchmark the fallback as 'jmespath_mojo'")

    print("\n== building documents ==")
    t0 = time.perf_counter()
    big_doc = make_big_doc(SEED)
    big_json_len = len(json.dumps(big_doc))
    small_docs = make_small_docs(SEED)
    print(f"- big doc: {big_json_len / 1e6:.2f} MB JSON "
          f"({BIG_RESERVATIONS}x{BIG_INSTANCES} instances)")
    print(f"- small docs: {len(small_docs)} docs")

    print("\n== correctness gate (structural equality vs pip jmespath) ==")
    for name, expr in list(BIG_QUERIES.items()) + [("floor", FLOOR_QUERY)]:
        check_parity(expr, big_doc)
        print(f"  OK {name}: {expr!r}")
    for doc in small_docs[:50]:
        check_parity(SMALL_QUERY, doc)
    print(f"  OK small docs (first 50): {SMALL_QUERY!r}")

    rows = []
    print("\n== big doc: warm steady-state ==")
    print(f"{'expression':>28} | {'jmespath ms':>11} | {'mojo ms':>9} | {'speedup':>8}")
    print(f"{'-' * 28}-+-{'-' * 11}-+-{'-' * 9}-+-{'-' * 8}")
    for name, expr in BIG_QUERIES.items():
        t_ref = time_warm(jmespath.search, expr, big_doc, 5)
        t_ours = time_warm(jmespath_mojo.search, expr, big_doc, 5)
        ms_ref, ms_ours = 1e3 * t_ref / 5, 1e3 * t_ours / 5
        rows.append((name, ms_ref, ms_ours, ms_ref / ms_ours))
        print(f"{name:>28} | {ms_ref:>11.3f} | {ms_ours:>9.3f} | {ms_ref / ms_ours:>7.2f}x")

    floor = {}
    t_ref = time_warm(jmespath.search, FLOOR_QUERY, big_doc, 5)
    t_ours = time_warm(jmespath_mojo.search, FLOOR_QUERY, big_doc, 5)
    floor["ref_ms"], floor["ours_ms"] = 1e3 * t_ref / 5, 1e3 * t_ours / 5
    print(f"{'boundary floor (simple path)':>28} | {floor['ref_ms']:>11.3f} "
          f"| {floor['ours_ms']:>9.3f} | {floor['ref_ms'] / floor['ours_ms']:>7.2f}x")

    print("\n== big doc: cold first call (fresh process, median of 7) ==")
    cold_expr = BIG_QUERIES["filter+flatten+project"]
    c_ref = time_cold("jmespath", cold_expr, big_doc)
    c_ours = time_cold("jmespath_mojo", cold_expr, big_doc)
    print(f"  {cold_expr!r}")
    print(f"  jmespath: {1e3 * c_ref:>8.3f} ms   jmespath_mojo: {1e3 * c_ours:>8.3f} ms")

    print("\n== small docs (5k x 1 search each): warm ==")
    t_ref = time_warm_many(jmespath.search, SMALL_QUERY, small_docs)
    t_ours = time_warm_many(jmespath_mojo.search, SMALL_QUERY, small_docs)
    per_ref, per_ours = 1e6 * t_ref / len(small_docs), 1e6 * t_ours / len(small_docs)
    print(f"  jmespath: {per_ref:>7.2f} us/call   jmespath_mojo: {per_ours:>7.2f} us/call"
          f"   ratio: {per_ref / per_ours:>5.2f}x")

    print("\n== README paste block ==")
    print("| workload (big doc, 0.85 MB) | pip jmespath | jmespath_mojo | speedup |")
    print("|---|---:|---:|---:|")
    for name, ms_ref, ms_ours, speedup in rows:
        print(f"| {name} (warm) | {ms_ref:.3f} ms | {ms_ours:.3f} ms | {speedup:.2f}x |")
    print(f"| boundary floor (warm) | {floor['ref_ms']:.3f} ms | {floor['ours_ms']:.3f} ms "
          f"| {floor['ref_ms'] / floor['ours_ms']:.2f}x |")
    print(f"| filter+flatten+project (cold, first call) | {1e3 * c_ref:.3f} ms "
          f"| {1e3 * c_ours:.3f} ms | {c_ref / c_ours:.2f}x |")
    print(f"| small docs 5k x 1 search (warm) | {per_ref:.2f} us/call "
          f"| {per_ours:.2f} us/call | {per_ref / per_ours:.2f}x |")


if __name__ == "__main__":
    main()

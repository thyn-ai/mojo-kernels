#!/usr/bin/env python3
"""Reproducible benchmark: jsonschema (PyPI) vs jsonschema_mojo.

Three deterministic, seeded scenarios (no network, no datasets):

A) records: 10,000 order records validated against one shared schema
   (the service hot loop: per-node Python walk over many small documents,
   including two `pattern` checks deferred to Python's re);
B) big-document: one ~3 MB deeply nested document validated once per call
   (the batch/document hot loop);
C) error-heavy: 2,000 invalid records producing ~4 errors each (the
   error-record path).

Cold = fresh Validator + first validate (schema gate + native compile +
first walk), median of 5. Warm = steady-state, median of 5 runs. Correctness
(pass/fail + sorted failing-path multiset vs the oracle) is asserted before
any timing happens, so the numbers below always come from a verified-correct
build.

Run from the repository root:

    PYTHONPATH=python/jsonschema_mojo python benchmarks/bench_jsonschema.py
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

ORDER_SCHEMA = {
    "type": "object",
    "required": ["order_id", "email", "total", "items"],
    "properties": {
        "order_id": {"type": "string", "pattern": "^ORD-[0-9]{6}$"},
        "email": {"type": "string", "minLength": 3, "pattern": "@"},
        "total": {"type": "number", "minimum": 0},
        "currency": {"enum": ["USD", "EUR", "GBP", "JPY"]},
        "items": {
            "type": "array",
            "minItems": 1,
            "items": {
                "type": "object",
                "required": ["sku", "qty", "price"],
                "properties": {
                    "sku": {"type": "string", "minLength": 1},
                    "qty": {"type": "integer", "minimum": 1},
                    "price": {"type": "number", "minimum": 0},
                },
                "additionalProperties": False,
            },
        },
        "address": {
            "type": "object",
            "properties": {
                "street": {"type": "string", "minLength": 1},
                "zip": {"type": "string", "pattern": "^[0-9]{5}$"},
            },
            "required": ["street", "zip"],
            "additionalProperties": False,
        },
        "vip": {"type": "boolean"},
    },
    "additionalProperties": False,
}

BIG_SCHEMA = {
    "type": "array",
    "items": {
        "type": "array",
        "items": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "v": {"type": "integer", "minimum": 0, "maximum": 100},
                    "s": {"type": "string", "maxLength": 16},
                },
                "required": ["v", "s"],
                "additionalProperties": False,
            },
        },
    },
}


def make_order(rng: random.Random, i: int) -> dict:
    return {
        "order_id": f"ORD-{i:06d}",
        "email": f"user{i}@example.com",
        "total": round(rng.uniform(0, 500), 2),
        "currency": rng.choice(["USD", "EUR", "GBP", "JPY"]),
        "items": [
            {
                "sku": f"SKU-{rng.randint(0, 9999):04d}",
                "qty": rng.randint(1, 5),
                "price": round(rng.uniform(1, 100), 2),
            }
            for _ in range(rng.randint(1, 4))
        ],
        "address": {
            "street": f"{rng.randint(1, 999)} Main St",
            "zip": f"{rng.randint(0, 99999):05d}",
        },
        "vip": rng.random() < 0.2,
    }


def make_orders(n: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    return [make_order(rng, i) for i in range(n)]


def make_bad_orders(n: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    bad = []
    for i in range(n):
        order = make_order(rng, i)
        order["total"] = -1.0  # minimum violation
        order["email"] = "not-an-email"  # pattern violation
        del order["order_id"]  # required violation
        order["extra_field"] = True  # additionalProperties violation
        bad.append(order)
    return bad


def make_big_document(seed: int) -> list:
    rng = random.Random(seed)
    return [
        [
            [
                {
                    "v": rng.randint(0, 100),
                    "s": "".join(rng.choices("abcdefghij", k=rng.randint(1, 16))),
                }
                for _ in range(25)
            ]
            for _ in range(30)
        ]
        for _ in range(30)
    ]


def median_of(fn, n=N_RUNS) -> float:
    samples = []
    for _ in range(n):
        t0 = time.perf_counter()
        fn()
        samples.append(time.perf_counter() - t0)
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
    from jsonschema import Draft202012Validator as OracleValidator

    import jsonschema_mojo
    from jsonschema_mojo import Validator

    info = jsonschema_mojo.backend_info()
    print("== environment ==")
    print(machine_info())
    print(
        f"- jsonschema_mojo backend: {'native' if info['native_available'] else 'FALLBACK'} "
        f"({info.get('native_source') or info.get('error')})"
    )
    import jsonschema

    import importlib.metadata

    print(f"- oracle: jsonschema {importlib.metadata.version('jsonschema')} (PyPI)")
    print(f"- seed: {SEED}; runs: median of {N_RUNS}")
    if not info["native_available"]:
        sys.exit(
            "native kernel unavailable; refusing to benchmark the fallback as 'jsonschema_mojo'"
        )

    scenarios = [
        ("A: 10k order records", ORDER_SCHEMA, make_orders(10_000, SEED), 10_000, "is_valid"),
        ("B: big nested document", BIG_SCHEMA, [make_big_document(SEED + 1)], 1, "is_valid"),
        ("C: 2k invalid records", ORDER_SCHEMA, make_bad_orders(2_000, SEED + 2), 2_000, "iter_errors"),
    ]

    print("\n== correctness gate (pass/fail + failing-path multiset vs oracle) ==")
    for name, schema, instances, _, _mode in scenarios:
        ours, oracle = Validator(schema), OracleValidator(schema)
        for inst in instances[:25]:
            our_keys = sorted(
                (e.json_path, e.validator or "") for e in ours.iter_errors(inst)
            )
            ref_keys = sorted(
                (e.json_path, e.validator or "") for e in oracle.iter_errors(inst)
            )
            assert our_keys == ref_keys, (name, inst, our_keys, ref_keys)
            assert ours.is_valid(inst) == oracle.is_valid(inst)
        print(f"  {name}: OK (25/25 instances parity)")

    def timed_call(validator, inst, mode):
        if mode == "is_valid":
            validator.is_valid(inst)
        else:
            list(validator.iter_errors(inst))

    rows = []
    for name, schema, instances, n_docs, mode in scenarios:
        # Cold: fresh validator + first call, median of N_RUNS.
        def cold_ours():
            timed_call(Validator(schema), instances[0], mode)

        def cold_ref():
            timed_call(OracleValidator(schema), instances[0], mode)

        t_cold_ours = median_of(cold_ours)
        t_cold_ref = median_of(cold_ref)

        # Warm: steady-state over the whole batch, median of N_RUNS.
        ours = Validator(schema)
        oracle = OracleValidator(schema)

        def warm_ours():
            for inst in instances:
                timed_call(ours, inst, mode)

        def warm_ref():
            for inst in instances:
                timed_call(oracle, inst, mode)

        t_warm_ours = median_of(warm_ours)
        t_warm_ref = median_of(warm_ref)
        rows.append(
            (
                name,
                t_cold_ref * 1e3,
                t_cold_ours * 1e3,
                t_warm_ref * 1e3 / n_docs,
                t_warm_ours * 1e3 / n_docs,
                n_docs / t_warm_ref,
                n_docs / t_warm_ours,
            )
        )

    print("\n== results ==")
    header = (
        f"{'scenario':>24} | {'cold ref ms':>11} | {'cold mojo ms':>12} | "
        f"{'warm ref ms/doc':>15} | {'warm mojo ms/doc':>16} | {'speedup':>8}"
    )
    print(header)
    print("-" * len(header))
    for name, cr, co, wr, wo, tr, to in rows:
        print(
            f"{name:>24} | {cr:>11.3f} | {co:>12.3f} | "
            f"{wr:>15.4f} | {wo:>16.4f} | {wr / wo:>7.1f}x"
        )

    print("\nwarm throughput (docs/sec):")
    for name, cr, co, wr, wo, tr, to in rows:
        print(f"  {name:>24}: ref {tr:>12,.0f}  mojo {to:>12,.0f}")

    # Fallback (pure-Python) warm numbers for honesty.
    print("\nfallback backend (forced pure-Python), warm ms/doc:")
    os.environ["JSONSCHEMA_MOJO_DISABLE_NATIVE"] = "1"
    import importlib

    importlib.reload(jsonschema_mojo._native)
    importlib.reload(jsonschema_mojo.core)
    importlib.reload(jsonschema_mojo)
    from jsonschema_mojo import Validator as FallbackValidator

    for name, schema, instances, n_docs, mode in scenarios:
        ours = FallbackValidator(schema)
        assert ours.backend == "fallback"
        if mode == "is_valid":
            t = median_of(lambda: [ours.is_valid(inst) for inst in instances])
        else:
            t = median_of(lambda: [list(ours.iter_errors(inst)) for inst in instances])
        print(f"  {name:>24}: {t * 1e3 / n_docs:.4f}")

    print("\n== README paste block ==")
    print("| scenario | cold jsonschema (ms) | cold jsonschema_mojo (ms) | "
          "warm jsonschema (ms/doc) | warm jsonschema_mojo (ms/doc) | warm speedup |")
    print("|---|---:|---:|---:|---:|---:|")
    for name, cr, co, wr, wo, tr, to in rows:
        print(f"| {name} | {cr:.3f} | {co:.3f} | {wr:.4f} | {wo:.4f} | {wr / wo:.1f}x |")


if __name__ == "__main__":
    main()

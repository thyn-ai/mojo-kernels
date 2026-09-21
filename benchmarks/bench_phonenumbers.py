#!/usr/bin/env python3
"""Reproducible benchmark: PyPI phonenumbers vs phonenumbers_mojo (native + fallback).

Correctness is gated first: every corpus number must produce identical
results (parsed fields, validity, all three formats) on the native and
fallback backends as the oracle, or the benchmark aborts. Timings:

- cold first-call: wall time of import + metadata load + first parse() in a
  fresh process, median of 5 launches.
- warm steady-state: per-call latency for parse/is_valid_number and
  format_number over the corpus, median of 5 batches, for the oracle, the
  native kernel and the forced fallback; plus validate_column batch
  throughput.

The corpus is the libphonenumber golden example numbers for the 20 top
regions plus generated variants — no network, no datasets.

Run from the repository root, e.g.:

    PYTHONPATH=python/phonenumbers_mojo .oracle-phonenumbers/bin/python benchmarks/bench_phonenumbers.py
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

TOP_REGIONS = [
    "US", "GB", "DE", "FR", "IN", "CN", "JP", "BR", "RU", "AU",
    "CA", "IT", "ES", "MX", "KR", "NL", "SE", "CH", "PL", "TR",
]
REGION_CC = {
    "US": "1", "GB": "44", "DE": "49", "FR": "33", "IN": "91", "CN": "86",
    "JP": "81", "BR": "55", "RU": "7", "AU": "61", "CA": "1", "IT": "39",
    "ES": "34", "MX": "52", "KR": "82", "NL": "31", "SE": "46", "CH": "41",
    "PL": "48", "TR": "90",
}


def build_corpus() -> list[tuple[str, str | None]]:
    """Golden example numbers for the top regions + generated variants."""
    from defusedxml import ElementTree as ET
    from pathlib import Path

    # Parses the repo-vendored libphonenumber metadata to build the corpus --
    # a repo file, not untrusted input.
    root = ET.parse(
        Path(__file__).resolve().parents[1]
        / "kernels/phonenumbers/data/PhoneNumberMetadata.xml"
    ).getroot()
    cases: list[tuple[str, str | None]] = []
    wanted = set(TOP_REGIONS)
    for terr in root.iter("territory"):
        rid = terr.get("id")
        if rid not in wanted:
            continue
        for el in terr:
            ex = el.find("exampleNumber")
            if ex is not None and ex.text:
                nsn = "".join(ex.text.split())
                cases.append((f"+{terr.get('countryCode')} {nsn}", None))
    rng = random.Random(7)
    for region in TOP_REGIONS:
        cc = REGION_CC[region]
        for _ in range(30):
            digits = "".join(rng.choice("0123456789") for _ in range(rng.randrange(6, 13)))
            form = rng.randrange(4)
            if form == 0:
                text = f"+{cc} {digits}"
            elif form == 1:
                text = f"+{cc}-{digits}"
            elif form == 2:
                text = digits
            else:
                text = f"{cc} {digits}"
            cases.append((text, region if rng.random() < 0.85 else None))
    rng.shuffle(cases)
    return cases


COLD_SNIPPETS = {
    "oracle": (
        "import time; t0=time.perf_counter();"
        "import phonenumbers;"
        "phonenumbers.parse('+1 212-555-1234');"
        "print(f'{time.perf_counter()-t0:.6f}')"
    ),
    "native": (
        "import time; t0=time.perf_counter();"
        "import phonenumbers_mojo;"
        "phonenumbers_mojo.parse('+1 212-555-1234');"
        "print(f'{time.perf_counter()-t0:.6f}')"
    ),
    "fallback": (
        "import time; t0=time.perf_counter();"
        "import phonenumbers_mojo;"
        "phonenumbers_mojo.parse('+1 212-555-1234');"
        "print(f'{time.perf_counter()-t0:.6f}')"
    ),
}


def cold_start(which: str) -> float:
    """Median seconds for import + first parse in a fresh process."""
    env = dict(os.environ)
    env["PYTHONNOUSERSITE"] = "1"
    if which == "fallback":
        env["PHONENUMBERS_MOJO_DISABLE_NATIVE"] = "1"
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
    import phonenumbers as oracle
    import phonenumbers_mojo as mine

    info = mine.backend_info()
    print("== environment ==")
    print(machine_info())
    print(
        f"- phonenumbers_mojo backend: {'native' if info['native_available'] else 'FALLBACK'} "
        f"({info.get('native_source') or info.get('error')})"
    )
    print(f"- oracle: phonenumbers {oracle.__version__}")
    if not info["native_available"]:
        sys.exit("native kernel unavailable; refusing to benchmark the fallback as 'native'")

    corpus = build_corpus()
    print(f"- corpus: {len(corpus)} numbers ({len(TOP_REGIONS)} regions)")
    print(f"- runs: median of {N_RUNS}")

    print("\n== correctness gate (identical vs oracle on both backends) ==")
    formats = (0, 1, 2)
    for disable, label in ((False, "native"), (True, "fallback")):
        os.environ["PHONENUMBERS_MOJO_DISABLE_NATIVE"] = "1" if disable else "0"
        for text, region in corpus:
            try:
                o = oracle.parse(text, region)
            except oracle.NumberParseException as e:
                o_res = ("ERR", e.error_type, str(e))
            else:
                o_res = (
                    "OK",
                    o.country_code, o.national_number, o.extension,
                    o.italian_leading_zero, o.number_of_leading_zeros,
                    oracle.is_valid_number(o),
                    tuple(oracle.format_number(o, f) for f in formats),
                )
            try:
                m = mine.parse(text, region)
            except mine.NumberParseException as e:
                m_res = ("ERR", e.error_type, str(e))
            else:
                m_res = (
                    "OK",
                    m.country_code, m.national_number, m.extension,
                    m.italian_leading_zero, m.number_of_leading_zeros,
                    mine.is_valid_number(m),
                    tuple(mine.format_number(m, f) for f in formats),
                )
            if o_res != m_res:
                sys.exit(f"correctness gate failed ({label}) for {text!r} {region}: {m_res} != {o_res}")
        print(f"  {label}: all {len(corpus)} numbers identical")
    os.environ["PHONENUMBERS_MOJO_DISABLE_NATIVE"] = "0"

    # Prepare parsed objects for the format/valid loops (skip unparseable rows).
    parsed, mine_parsed = [], []
    for t, r in corpus:
        try:
            o = oracle.parse(t, r)
            m = mine.parse(t, r)
        except (oracle.NumberParseException, mine.NumberParseException):
            continue
        parsed.append(o)
        mine_parsed.append(m)

    def time_calls(fn, repeats=3) -> float:
        """Median seconds for one full pass over `repeats` x corpus calls."""
        samples = []
        for _ in range(N_RUNS):
            t0 = time.perf_counter()
            for _ in range(repeats):
                fn()
            samples.append((time.perf_counter() - t0) / repeats)
        return statistics.median(samples)

    print("\n== warm steady-state: parse() over the corpus ==")

    def oracle_parse_loop():
        for t, r in corpus:
            try:
                oracle.parse(t, r)
            except oracle.NumberParseException:
                pass

    def mine_parse_loop():
        for t, r in corpus:
            try:
                mine.parse(t, r)
            except mine.NumberParseException:
                pass

    t_oracle = time_calls(oracle_parse_loop)
    t_native = time_calls(mine_parse_loop)
    os.environ["PHONENUMBERS_MOJO_DISABLE_NATIVE"] = "1"
    t_fallback = time_calls(mine_parse_loop)
    os.environ["PHONENUMBERS_MOJO_DISABLE_NATIVE"] = "0"
    po, pn, pf = (1e6 * t / len(corpus) for t in (t_oracle, t_native, t_fallback))
    print(f"  oracle (PyPI phonenumbers):  {po:8.2f} µs/call")
    print(f"  phonenumbers_mojo native:  {pn:8.2f} µs/call  ({po/pn:.1f}x)")
    print(f"  phonenumbers_mojo fallback:{pf:8.2f} µs/call  ({po/pf:.2f}x)")

    print("\n== warm steady-state: is_valid_number() over parsed corpus ==")
    vo = time_calls(lambda: [oracle.is_valid_number(n) for n in parsed])
    vn = time_calls(lambda: [mine.is_valid_number(n) for n in mine_parsed])
    os.environ["PHONENUMBERS_MOJO_DISABLE_NATIVE"] = "1"
    vf = time_calls(lambda: [mine.is_valid_number(n) for n in mine_parsed])
    os.environ["PHONENUMBERS_MOJO_DISABLE_NATIVE"] = "0"
    vo_, vn_, vf_ = (1e6 * t / len(corpus) for t in (vo, vn, vf))
    print(f"  oracle:  {vo_:8.2f} µs/call")
    print(f"  native:  {vn_:8.2f} µs/call  ({vo_/vn_:.1f}x)")
    print(f"  fallback:{vf_:8.2f} µs/call  ({vo_/vf_:.2f}x)")

    print("\n== warm steady-state: format_number(NATIONAL) over parsed corpus ==")
    fo = time_calls(lambda: [oracle.format_number(n, 2) for n in parsed])
    fn_ = time_calls(lambda: [mine.format_number(n, 2) for n in mine_parsed])
    os.environ["PHONENUMBERS_MOJO_DISABLE_NATIVE"] = "1"
    ff = time_calls(lambda: [mine.format_number(n, 2) for n in mine_parsed])
    os.environ["PHONENUMBERS_MOJO_DISABLE_NATIVE"] = "0"
    fo_, fn2, ff_ = (1e6 * t / len(corpus) for t in (fo, fn_, ff))
    print(f"  oracle:  {fo_:8.2f} µs/call")
    print(f"  native:  {fn2:8.2f} µs/call  ({fo_/fn2:.1f}x)")
    print(f"  fallback:{ff_:8.2f} µs/call  ({fo_/ff_:.2f}x)")

    print("\n== warm steady-state: validate_column batch (one call for the whole corpus) ==")
    strings = [t for t, _ in corpus]
    regions = [r for _, r in corpus]
    # single-region batch (the common pipeline shape)
    us_strings = [t for t, r in corpus if r in (None, "US")]

    def oracle_col():
        out = []
        for t in us_strings:
            try:
                out.append(oracle.is_valid_number(oracle.parse(t, "US")))
            except oracle.NumberParseException:
                out.append(False)
        return out

    def mine_col():
        return mine.validate_column(us_strings, "US")

    bo = time_calls(oracle_col, repeats=3)
    bn = time_calls(mine_col, repeats=3)
    os.environ["PHONENUMBERS_MOJO_DISABLE_NATIVE"] = "1"
    bf = time_calls(mine_col, repeats=3)
    os.environ["PHONENUMBERS_MOJO_DISABLE_NATIVE"] = "0"
    bo_, bn_, bf_ = (1e6 * t / len(us_strings) for t in (bo, bn, bf))
    print(f"  oracle per-item loop: {bo_:8.2f} µs/number")
    print(f"  native validate_column: {bn_:8.2f} µs/number  ({bo_/bn_:.1f}x)")
    print(f"  fallback validate_column: {bf_:8.2f} µs/number  ({bo_/bf_:.2f}x)")

    print("\n== cold first-call (fresh process: import + metadata + first parse) ==")
    c_oracle = cold_start("oracle")
    c_native = cold_start("native")
    c_fallback = cold_start("fallback")
    print(f"  oracle:  {1e3*c_oracle:8.2f} ms")
    print(f"  native:  {1e3*c_native:8.2f} ms  ({c_oracle/c_native:.2f}x)")
    print(f"  fallback:{1e3*c_fallback:8.2f} ms  ({c_oracle/c_fallback:.2f}x)")

    print("\n== README paste block ==")
    print("Warm steady-state (median of 5 batches, this machine):")
    print()
    print("| workload | PyPI phonenumbers | phonenumbers-mojo (native) | phonenumbers-mojo (fallback) | native speedup |")
    print("|---|---:|---:|---:|---:|")
    print(f"| `parse()`, per call | {po:.2f} µs | {pn:.2f} µs | {pf:.2f} µs | {po/pn:.1f}x |")
    print(f"| `is_valid_number()`, per call | {vo_:.2f} µs | {vn_:.2f} µs | {vf_:.2f} µs | {vo_/vn_:.1f}x |")
    print(f"| `format_number(NATIONAL)`, per call | {fo_:.2f} µs | {fn2:.2f} µs | {ff_:.2f} µs | {fo_/fn2:.1f}x |")
    print(f"| `validate_column` batch, per number | {bo_:.2f} µs | {bn_:.2f} µs | {bf_:.2f} µs | {bo_/bn_:.1f}x |")
    print()
    print("Cold first-call (fresh process, median of 5):")
    print()
    print("| package | import + first parse |")
    print("|---|---:|")
    print(f"| PyPI phonenumbers | {1e3*c_oracle:.2f} ms |")
    print(f"| phonenumbers-mojo (native) | {1e3*c_native:.2f} ms |")
    print(f"| phonenumbers-mojo (fallback) | {1e3*c_fallback:.2f} ms |")


if __name__ == "__main__":
    main()

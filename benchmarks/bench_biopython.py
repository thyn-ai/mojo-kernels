#!/usr/bin/env python3
"""Reproducible benchmark: Bio.SeqIO (biopython) vs bio_mojo, FASTA + GenBank.

Corpora are generated locally from fixed seeds (no network, no datasets) and
written to a temporary directory, so timings use real files on disk — the
primary end-user input shape:

  FASTA:   10k-record (1.6 MB) and 150k-record (24 MB) seeded corpora
  GenBank: 300-record (0.3 MB) and 3k-record (2.8 MB) seeded corpora
           (features exercise the whole supported location/qualifier subset)

Correctness is asserted (record-for-record vs Bio.SeqIO) before any timing
happens, so the numbers always come from a verified-correct build.

Per cell we report:
  cold — first parse in a fresh process state (kernel dlopen + ABI
         handshake + first buffer scan), single measurement
  warm — median of 5 subsequent full parses

Run from the repository root:

    PYTHONPATH=python/bio_mojo pixi run python benchmarks/bench_biopython.py
"""

from __future__ import annotations

import io
import os
import platform
import statistics
import subprocess
import sys
import tempfile
import time

N_WARM_RUNS = 5
FASTA_SIZES = [10_000, 150_000]
GB_SIZES = [300, 3_000]
SEED = 20260919


def _rand_seq(rng, length):
    alphabet = "ACGTUNRYMKSWHBVDacgtun" + ".-*"
    return "".join(rng.choice(alphabet) for _ in range(length))


def make_fasta_corpus(seed: int, n_records: int) -> str:
    import random

    rng = random.Random(seed)
    out = []
    for i in range(n_records):
        out.append(f">seq{i:07d} seeded benchmark record {i}")
        total = rng.randint(60, 240)
        seq = _rand_seq(rng, total)
        for pos in range(0, len(seq), 80):
            out.append(seq[pos : pos + 80])
    return "\n".join(out) + "\n"


_GB_TYPES = ("source", "gene", "CDS", "mRNA", "tRNA", "rRNA", "exon", "misc_feature")
_GB_LOCS = (
    "{a}..{b}", "<{a}..>{b}", "complement({a}..{b})",
    "join({a}..{b},{c}..{d})", "complement(join({a}..{b},{c}..{d}))",
    "order({a}..{b},{c}..{d})", "{a}^{a1}",
)


def make_gb_corpus(seed: int, n_records: int) -> str:
    import random

    rng = random.Random(seed)
    out = []
    for i in range(n_records):
        name = f"SC{i:06d}"[:16]
        length = rng.randint(50, 5000)
        out.append(
            f"LOCUS       {name:<16} {length:>5} bp    DNA             "
            f"PLN       01-JAN-2000"
        )
        out.append(f"DEFINITION  Seeded benchmark record {i}.")
        out.append(f"ACCESSION   AC{i:06d}")
        out.append(f"VERSION     AC{i:06d}.{1 + i % 3}  GI:{1000 + i}")
        out.append("FEATURES             Location/Qualifiers")
        for _ in range(rng.randint(1, 5)):
            ftype = rng.choice(_GB_TYPES)
            vals = sorted(rng.sample(range(5, 45), 6))
            loc = rng.choice(_GB_LOCS).format(
                a=vals[0], b=vals[1], c=vals[2], d=vals[3], e=vals[4], f=vals[5],
                a1=vals[0] + 1,
            )
            out.append(f"     {ftype:<16}{loc}")
            if ftype == "source":
                out.append('                     /organism="Syntheticus constructus"')
            if ftype == "CDS":
                prot = "".join(rng.choice("ACDEFGHIKLMNPQRSTVWY") for _ in range(30))
                out.append(f'                     /translation="{prot[:15]}')
                out.append(f'                     {prot[15:]}"')
            out.append(f'                     /note="benchmark feature {i}"')
        out.append("ORIGIN")
        seq = _rand_seq(rng, min(length, rng.randint(200, 900)))
        for pos in range(0, len(seq), 60):
            chunk = seq[pos : pos + 60]
            groups = " ".join(chunk[g : g + 10] for g in range(0, len(chunk), 10))
            out.append(f"{pos + 1:>9} {groups}")
        out.append("//")
    return "\n".join(out) + "\n"


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


def _fasta_tuples(records):
    return [(r.id, r.name, r.description, str(r.seq)) for r in records]


def _gb_tuples(records):
    return [
        (r.id, r.name, r.description, str(r.seq), len(r.features),
         [(f.type, str(f.location), f.qualifiers) for f in r.features])
        for r in records
    ]


def bench_format(label, path, n_records, ours_parse, oracle_parse, to_tuples):
    """Correctness gate, then cold + warm (median of N_WARM_RUNS) timings."""
    # Correctness gate (untimed).
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ours = to_tuples(ours_parse(str(path)))
        oracle = to_tuples(oracle_parse(str(path)))
    if ours != oracle:
        sys.exit(f"correctness gate failed for {label}: records differ")
    n_bytes = os.path.getsize(path)

    def timed_ours():
        return to_tuples(ours_parse(str(path)))

    def timed_oracle():
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return to_tuples(oracle_parse(str(path)))

    # Cold: first call in this process (kernel dlopen + ABI handshake + scan).
    t0 = time.perf_counter()
    timed_ours()
    cold_ours = time.perf_counter() - t0
    t0 = time.perf_counter()
    timed_oracle()
    cold_oracle = time.perf_counter() - t0

    # Warm: median of N_WARM_RUNS full parses.
    warm_ours, warm_oracle = [], []
    for _ in range(N_WARM_RUNS):
        t0 = time.perf_counter()
        timed_ours()
        warm_ours.append(time.perf_counter() - t0)
        t0 = time.perf_counter()
        timed_oracle()
        warm_oracle.append(time.perf_counter() - t0)
    w_ours, w_oracle = statistics.median(warm_ours), statistics.median(warm_oracle)
    return {
        "label": label,
        "records": n_records,
        "bytes": n_bytes,
        "cold_ours": cold_ours,
        "cold_oracle": cold_oracle,
        "warm_ours": w_ours,
        "warm_oracle": w_oracle,
    }


def main() -> None:
    import bio_mojo
    from Bio import SeqIO

    info = bio_mojo.backend_info()
    print("== environment ==")
    print(machine_info())
    print(
        f"- bio_mojo backend: {'native' if info['native_available'] else 'FALLBACK'} "
        f"({info.get('native_source') or info.get('error')})"
    )
    import Bio

    print(f"- biopython (oracle): {Bio.__version__}")
    print(f"- seeds: {SEED}+size; warm = median of {N_WARM_RUNS}; cold = first call")
    if not info["native_available"]:
        sys.exit("native kernel unavailable; refusing to benchmark the fallback as 'bio_mojo'")

    rows = []
    with tempfile.TemporaryDirectory(prefix="bio-mojo-bench-") as tmp:
        for size in FASTA_SIZES:
            path = os.path.join(tmp, f"fasta-{size}.fa")
            with open(path, "w") as fh:
                fh.write(make_fasta_corpus(SEED + size, size))
            rows.append(
                bench_format(
                    f"FASTA {size:,}",
                    path,
                    size,
                    bio_mojo.parse_fasta,
                    lambda p: SeqIO.parse(p, "fasta"),
                    _fasta_tuples,
                )
            )
        for size in GB_SIZES:
            path = os.path.join(tmp, f"gb-{size}.gbk")
            with open(path, "w") as fh:
                fh.write(make_gb_corpus(SEED + size, size))
            rows.append(
                bench_format(
                    f"GenBank {size:,}",
                    path,
                    size,
                    bio_mojo.parse_genbank,
                    lambda p: SeqIO.parse(p, "genbank"),
                    _gb_tuples,
                )
            )

    print("\n== full-file parse (whole corpus into records) ==")
    header = (
        f"{'corpus':>16} | {'size':>7} | {'cold SeqIO':>11} | {'cold bio_mojo':>13} "
        f"| {'speedup':>7} | {'warm SeqIO':>11} | {'warm bio_mojo':>13} | {'speedup':>7}"
    )
    print(header)
    print("-" * len(header))
    for r in rows:
        print(
            f"{r['label']:>16} | {r['bytes'] / 1e6:>6.1f}M "
            f"| {1e3 * r['cold_oracle']:>9.1f}ms | {1e3 * r['cold_ours']:>11.1f}ms "
            f"| {r['cold_oracle'] / r['cold_ours']:>6.1f}x "
            f"| {1e3 * r['warm_oracle']:>9.1f}ms | {1e3 * r['warm_ours']:>11.1f}ms "
            f"| {r['warm_oracle'] / r['warm_ours']:>6.1f}x"
        )

    print("\n== warm throughput ==")
    for r in rows:
        mb_s = r["bytes"] / 1e6 / r["warm_ours"]
        rec_s = r["records"] / r["warm_ours"]
        print(f"{r['label']:>16}: {mb_s:>7.1f} MB/s, {rec_s:>12,.0f} records/s")

    print("\n== README paste block ==")
    print("| corpus | file size | records | cold Bio.SeqIO | cold bio_mojo | cold speedup | warm Bio.SeqIO | warm bio_mojo | warm speedup |")
    print("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for r in rows:
        print(
            f"| {r['label']} | {r['bytes'] / 1e6:.1f} MB | {r['records']:,} "
            f"| {1e3 * r['cold_oracle']:.1f} ms | {1e3 * r['cold_ours']:.1f} ms "
            f"| {r['cold_oracle'] / r['cold_ours']:.1f}x "
            f"| {1e3 * r['warm_oracle']:.1f} ms | {1e3 * r['warm_ours']:.1f} ms "
            f"| {r['warm_oracle'] / r['warm_ours']:.1f}x |"
        )
    print("\nWarm bio_mojo throughput: " + ", ".join(
        f"{r['label']} {r['bytes'] / 1e6 / r['warm_ours']:.0f} MB/s" for r in rows
    ))


if __name__ == "__main__":
    main()

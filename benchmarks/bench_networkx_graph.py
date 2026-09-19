#!/usr/bin/env python3
"""Reproducible benchmark: networkx vs nx_mojo (betweenness + dijkstra).

Graphs are generated locally from fixed seeds (no network, no datasets):
Erdos-Renyi G(n, m) graphs for betweenness and a 50k-node graph for
single-source Dijkstra. Timings report a cold first call and the median of
5 warm calls, per engine, interleaved so machine load hits both engines
alike. Correctness is asserted (dicts vs networkx within 1e-12, in practice
bit-exact) before any timing happens, so the numbers below always come from
a verified-correct build.

Run from the repository root (oracle on PYTHONPATH, kernel built):

    PYTHONPATH=build/nx-test-oracle:python/nx_mojo \\
        ~/.pixi/bin/pixi run python benchmarks/bench_networkx_graph.py
"""

from __future__ import annotations

import platform
import statistics
import subprocess
import sys
import time

import networkx as nx

import nx_mojo
from nx_mojo.core import _to_csr

N_RUNS = 5  # warm-run count; we report the median
ATOL = 1e-12

BC_CASES = [
    # (label, n, m, seed, directed, weight attr)
    ("BC unweighted n=300", 300, 1500, 101, False, None),
    ("BC unweighted n=1000", 1000, 5000, 103, False, None),
    ("BC weighted n=1000", 1000, 5000, 103, False, "weight"),
    ("BC unweighted n=1000 (directed)", 1000, 5000, 107, True, None),
]

DIJKSTRA_CASES = [
    # (label, n, m, seed, weight attr)
    ("dijkstra n=50k (no weight attr)", 50_000, 250_000, 109, None),
    ("dijkstra n=50k weighted", 50_000, 250_000, 109, "weight"),
]

N_DIJKSTRA_SOURCES = 3  # sources per warm run; timings are per call


def make_bc_graph(n, m, seed, directed, weight):
    G = nx.gnm_random_graph(n, m, seed=seed, directed=directed)
    if weight:
        for u, v in G.edges():
            G[u][v][weight] = ((u * 31 + v * 17 + seed) % 13) + 0.5
    return G


def make_dijkstra_graph(n, m, seed, weight):
    G = nx.gnm_random_graph(n, m, seed=seed)
    if weight:
        for u, v in G.edges():
            G[u][v][weight] = ((u * 31 + v * 17 + seed) % 13) * 0.25 + 0.25
    return G


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
    import numpy

    lines.append(
        f"- python: {platform.python_version()}, numpy: {numpy.__version__}, "
        f"networkx: {nx.__version__}"
    )
    try:
        mojo = subprocess.run(
            ["mojo", "--version"], capture_output=True, text=True
        ).stdout.strip()
        lines.append(f"- mojo: {mojo}")
    except OSError:
        pass
    return "\n".join(lines)


def time_calls(fn, args_list, n_runs):
    """(cold, [warm samples]) in seconds, one entry per warm run."""
    t0 = time.perf_counter()
    fn(*args_list[0])
    cold = time.perf_counter() - t0
    warm = []
    for args in args_list:
        t0 = time.perf_counter()
        fn(*args)
        warm.append(time.perf_counter() - t0)
    return cold, warm


def main() -> None:
    info = nx_mojo.backend_info()
    print("== environment ==")
    print(machine_info())
    print(
        f"- nx_mojo backend: {'native' if info['native_available'] else 'FALLBACK'} "
        f"({info.get('native_source') or info.get('error')})"
    )
    if not info["native_available"]:
        sys.exit("native kernel unavailable; refusing to benchmark the fallback as 'nx_mojo'")
    print(f"- warm runs: median of {N_RUNS}; cold = first call on the graph")

    rows = []

    print("\n== correctness gate (dicts vs networkx, tolerance 1e-12) ==")
    for label, n, m, seed, directed, weight in BC_CASES:
        G = make_bc_graph(n, m, seed, directed, weight)
        ours = nx_mojo.betweenness_centrality(G, weight=weight)
        ref = nx.betweenness_centrality(G, weight=weight)
        worst = max(abs(ours[v] - ref[v]) for v in ref)
        status = "OK" if worst <= ATOL else "FAIL"
        print(f"  {label:>32}: max|diff| = {worst:.3e}  [{status}]")
        if worst > ATOL:
            sys.exit(f"correctness gate failed: {label}")
    dijkstra_graphs = {}
    for label, n, m, seed, weight in DIJKSTRA_CASES:
        G = make_dijkstra_graph(n, m, seed, weight)
        dijkstra_graphs[label] = (G, weight)
        ours_d, ours_p = nx_mojo.single_source_dijkstra(G, 0, weight=weight or "weight")
        ref_d, ref_p = nx.single_source_dijkstra(G, 0, weight=weight or "weight")
        worst = max(abs(ours_d[v] - ref_d[v]) for v in ref_d)
        paths_ok = ours_p == ref_p
        status = "OK" if worst <= ATOL and paths_ok else "FAIL"
        print(f"  {label:>32}: max|diff| = {worst:.3e}, paths equal: {paths_ok}  [{status}]")
        if status != "OK":
            sys.exit(f"correctness gate failed: {label}")

    print("\n== betweenness_centrality (seconds; cold first call / warm median of 5) ==")
    header = f"{'case':>32} | {'engine':>8} | {'cold':>9} | {'warm med':>9} | {'speedup (warm)':>14}"
    print(header)
    print(f"{'-' * 32}-+-{'-' * 8}-+-{'-' * 9}-+-{'-' * 9}-+-{'-' * 14}")
    for label, n, m, seed, directed, weight in BC_CASES:
        G = make_bc_graph(n, m, seed, directed, weight)
        # Interleave engines so machine load hits both alike.
        args = [(G,)] * N_RUNS
        cold_o, warm_o = time_calls(
            lambda g: nx_mojo.betweenness_centrality(g, weight=weight), args, N_RUNS
        )
        cold_n, warm_n = time_calls(
            lambda g: nx.betweenness_centrality(g, weight=weight), args, N_RUNS
        )
        med_o, med_n = statistics.median(warm_o), statistics.median(warm_n)
        speedup = med_n / med_o
        print(f"{label:>32} | {'networkx':>8} | {cold_n:>9.3f} | {med_n:>9.3f} | {'1.0x':>14}")
        print(f"{'':>32} | {'nx_mojo':>8} | {cold_o:>9.4f} | {med_o:>9.4f} | {speedup:>13.1f}x")
        rows.append((label, cold_n, med_n, cold_o, med_o, speedup))

    print("\n== single_source_dijkstra (seconds; per call, cold / warm median of 5) ==")
    print(header)
    print(f"{'-' * 32}-+-{'-' * 8}-+-{'-' * 9}-+-{'-' * 9}-+-{'-' * 14}")
    for label, n, m, seed, weight in DIJKSTRA_CASES:
        G, weight = dijkstra_graphs[label]
        sources = list(G.nodes)[:N_RUNS]
        args = [(G, s) for s in sources]
        cold_o, warm_o = time_calls(
            lambda g, s: nx_mojo.single_source_dijkstra(g, s, weight=weight or "weight"),
            args,
            N_RUNS,
        )
        cold_n, warm_n = time_calls(
            lambda g, s: nx.single_source_dijkstra(g, s, weight=weight or "weight"),
            args,
            N_RUNS,
        )
        med_o, med_n = statistics.median(warm_o), statistics.median(warm_n)
        speedup = med_n / med_o
        print(f"{label:>32} | {'networkx':>8} | {cold_n:>9.3f} | {med_n:>9.3f} | {'1.0x':>14}")
        print(f"{'':>32} | {'nx_mojo':>8} | {cold_o:>9.4f} | {med_o:>9.4f} | {speedup:>13.1f}x")
        rows.append((label, cold_n, med_n, cold_o, med_o, speedup))

    # Conversion share: how much of an nx_mojo call is networkx->CSR
    # conversion (pure Python) vs the native kernel.
    print("\n== nx_mojo per-call cost split (median of 5, seconds) ==")
    print(f"{'case':>32} | {'CSR convert':>11} | {'total call':>10} | {'convert %':>9}")
    print(f"{'-' * 32}-+-{'-' * 11}-+-{'-' * 10}-+-{'-' * 9}")
    for label, n, m, seed, weight in DIJKSTRA_CASES:
        G, weight = dijkstra_graphs[label]
        samples = []
        for _ in range(N_RUNS):
            t0 = time.perf_counter()
            _to_csr(G, weight or "weight")
            samples.append(time.perf_counter() - t0)
        conv = statistics.median(samples)
        total = next(r for r in rows if r[0] == label)[4]
        print(f"{label:>32} | {conv:>11.4f} | {total:>10.4f} | {conv / total:>8.0%}")

    print("\n== README paste block ==")
    print("| case | networkx cold (s) | networkx warm (s) | nx_mojo cold (s) | nx_mojo warm (s) | speedup (warm) |")
    print("|---|---:|---:|---:|---:|---:|")
    for label, cold_n, med_n, cold_o, med_o, speedup in rows:
        print(
            f"| {label} | {cold_n:.3f} | {med_n:.3f} | {cold_o:.4f} | {med_o:.4f} "
            f"| {speedup:.1f}x |"
        )


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""End-user quickstart for nx-mojo: install the wheel, run this file.

    pip install nx-mojo           # or: pip install dist/*.whl from the repo
    python quickstart.py

Computes betweenness centrality and a Dijkstra shortest-path tree on the
Zachary karate club with both the native backend and (when
NX_MOJO_DISABLE_NATIVE=1) the pure-Python fallback, and prints checksums
that must be identical on either backend.
"""

from __future__ import annotations

import networkx as nx

import nx_mojo


def main() -> None:
    info = nx_mojo.backend_info()
    print(f"nx-mojo {nx_mojo.__version__}")
    print(f"backend: {'native' if info['native_available'] else 'pure-Python fallback'}")

    G = nx.karate_club_graph()
    bc = nx_mojo.betweenness_centrality(G)
    # Order-independent checksum of the full result dict.
    bc_checksum = sum(v * (i + 1) for i, (k, v) in enumerate(sorted(bc.items())))
    print(f"betweenness checksum: {bc_checksum:.12e}")
    print(f"most central node: {max(bc, key=bc.get)} ({max(bc.values()):.6f})")

    dist, paths = nx_mojo.single_source_dijkstra(G, 0, weight="weight")
    dist_checksum = sum(v * (i + 1) for i, (k, v) in enumerate(sorted(dist.items())))
    path_checksum = sum(len(p) * (i + 1) for i, (k, p) in enumerate(sorted(paths.items())))
    print(f"dijkstra dist checksum: {dist_checksum:.12e}")
    print(f"dijkstra path checksum: {path_checksum:.12e}")

    # Cross-check against networkx itself (both must agree to all digits).
    ref_bc = nx.betweenness_centrality(G)
    ref_dist, ref_paths = nx.single_source_dijkstra(G, 0, weight="weight")
    assert max(abs(bc[v] - ref_bc[v]) for v in ref_bc) <= 1e-12
    assert max(abs(dist[v] - ref_dist[v]) for v in ref_dist) <= 1e-12
    assert paths == ref_paths
    print("parity with networkx: OK (1e-12)")

    # The installed wheel registers an official networkx backend; dispatch
    # through networkx's own machinery must reach this package.
    dispatched = nx.betweenness_centrality(G, backend="nx_mojo")
    assert dispatched == bc
    print("networkx backend dispatch (backend='nx_mojo'): OK")


if __name__ == "__main__":
    main()

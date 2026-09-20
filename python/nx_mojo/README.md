# nx-mojo

[NetworkX](https://networkx.org/) graph algorithms accelerated by a clean-room
Mojo kernel — with networkx-identical results, and a vendored pure-Python
fallback for platforms without a native build (including Windows — tested there in CI: [`windows-fallback`](https://github.com/thyn-ai/mojo-kernels/actions/workflows/windows-fallback.yml)).

```python
import networkx as nx
import nx_mojo

G = nx.karate_club_graph()

nx_mojo.betweenness_centrality(G)            # same dict as nx.betweenness_centrality
nx_mojo.single_source_dijkstra(G, source=0)  # same (dist, paths) as nx.single_source_dijkstra
```

- **Same results**: `betweenness_centrality` and `single_source_dijkstra`
  take the same arguments, return the same shapes (dicts keyed by the
  original node labels), and raise the same networkx exceptions
  (`NodeNotFound`, `NetworkXNoPath`, `ValueError` on negative weights) as
  their `networkx` counterparts. The differential test suite asserts
  agreement with pip `networkx==3.5` within **1e-12** on both the native and
  fallback backends — in practice the agreement is **bit-for-bit**, including
  Dijkstra paths, which are compared exactly. Parity is also re-verified
  against the latest published networkx (3.6.1) on both backends.
- **Much faster betweenness**: a compiled Brandes inner loop (CSR + native
  priority queue) instead of per-edge Python dict work — **20x-40x** on
  300-1000 node graphs (Apple M4 Max; full method and numbers below).
- **No toolchain needed**: per-platform wheels ship the compiled kernel.
  Everywhere else the package transparently uses its pure-Python fallback.
- Force the fallback with `NX_MOJO_DISABLE_NATIVE=1`; inspect the active
  backend with `nx_mojo.backend_info()`.

## Scope

Exact, deterministic computation of the two networkx functions, on the
graph types reachable via CSR:

- simple undirected/directed graphs (`nx.Graph` / `nx.DiGraph`), with or
  without self-loops;
- `betweenness_centrality(G, k=None, normalized=True|False, weight=None|<attr>)`:
  unweighted (BFS Brandes) and weighted (Dijkstra Brandes), normalized and
  raw, `endpoints=False`;
- `single_source_dijkstra(G, source, target=None, cutoff=None, weight=<attr>)`:
  non-negative weights; distances **and** paths match the oracle.

Documented exclusions — these raise `NotImplementedError` (never silently
compute something else):

- multigraphs (`nx.MultiGraph` / `nx.MultiDiGraph`, i.e. parallel edges);
- `k`-sampled approximate betweenness and `endpoints=True`;
- callable `weight` functions (weights must be a numeric edge attribute,
  addressed by name; missing attributes default to 1, like the oracle);
- node/edge attributes other than the weight attribute are ignored.

All arithmetic is IEEE-754 float64 in the oracle's operation order (integer
weights convert exactly; the kernel is built with `--fp-mode contract=off`
so no FMA contraction can drift a single ulp from CPython's evaluation).
Distances are returned as floats (the oracle returns ints for
integer-weighted graphs; the values are equal).

## Official networkx backend registration (the pitch)

networkx 3.3+ discovers out-of-tree backends through the
`networkx.backends` entry point. This package registers itself, so once
installed you can dispatch through networkx's own machinery — no imports of
`nx_mojo` required:

```python
import networkx as nx

G = nx.karate_club_graph()
nx.betweenness_centrality(G, backend="nx_mojo")      # dispatched to the Mojo kernel
nx.single_source_dijkstra(G, 0, backend="nx_mojo")

# Or prioritize it for every dispatchable call:
#   export NETWORKX_BACKEND_PRIORITY=nx_mojo
# Out-of-scope inputs (multigraphs, k=..., endpoints=True, callable weights)
# are declined via `can_run`, so networkx transparently falls back to its
# own implementation for those.
```

The registration is the standard two entry points (in `pyproject.toml`):

```toml
[project.entry-points."networkx.backends"]
nx_mojo = "nx_mojo.backend:BackendInterface"

[project.entry-points."networkx.backend_info"]
nx_mojo = "nx_mojo.backend:get_info"
```

`BackendInterface` converts nothing (`convert_from_nx` returns the graph as
-is — the kernel consumes networkx adjacency directly) and returns
networkx-shaped results unchanged.

## Benchmarks

Measured on this machine (Apple M4 Max, macOS 26.6.2 arm64, python 3.12.14,
numpy 2.5.3, networkx 3.5, Mojo 1.1.0) on 2026-09-19, with
`benchmarks/bench_networkx_graph.py`: seeded G(n, m) graphs, cold first call
and median of 5 warm calls per engine, interleaved so machine load hits
both engines alike. Correctness (1e-12; bit-exact in this run) is asserted
before any timing.

| case | networkx cold (s) | networkx warm (s) | nx_mojo cold (s) | nx_mojo warm (s) | speedup (warm) |
|---|---:|---:|---:|---:|---:|
| BC unweighted n=300 | 0.096 | 0.095 | 0.0036 | 0.0033 | 28.5x |
| BC unweighted n=1000 | 1.428 | 1.673 | 0.0409 | 0.0422 | 39.6x |
| BC weighted n=1000 | 6.242 | 4.950 | 0.2501 | 0.2492 | 19.9x |
| BC unweighted n=1000 (directed) | 1.464 | 1.489 | 0.0398 | 0.0372 | 40.0x |
| dijkstra n=50k (no weight attr) | 0.257 | 0.386 | 0.4160 | 0.3166 | 1.2x |
| dijkstra n=50k weighted | 0.820 | 0.741 | 0.3062 | 0.3510 | 2.1x |

Betweenness is the flagship win: the native Brandes loop dominates even
with graph conversion included in every call. Single-source Dijkstra is
conversion-bound (the networkx→CSR conversion is pure Python and accounts
for ~40-60% of our per-call time on the 50k-node graph; the native kernel
itself answers in ~9 ms), so the honest speedup is 1.2x-2.1x per call —
nx_mojo never claims the fallback's numbers as its own.

## How it works

```
networkx graph ──▶ CSR (adjacency order preserved) ──▶ libnxgraphmojo
                  (Python, per call)                    (Mojo kernel: Brandes BFS/Dijkstra
                                                         path counting + accumulation,
                                                         Dijkstra with parent tracking)
```

- `kernels/networkx-graph/src/nxgraphmojo.mojo` — clean-room kernel written
  from the published algorithms (Brandes 2001; Dijkstra). Exported C ABI v1
  with an ABI-version handshake; the wrapper falls back if the handshake
  fails.
- `nx_mojo/_reference.py` — vendored pure-Python implementation of the same
  algorithms over the same CSR, used whenever the native library is
  unavailable. Both backends are differential-tested against pip `networkx`.
- The kernel reproduces the oracle's observable float behaviour exactly:
  FIFO BFS order, (distance, push-counter) heap tie-breaks, pop-time sigma
  updates for weighted counting, and reverse-discovery accumulation with
  strictly separated multiply/add rounding.

Source, benchmarks, and development: <https://github.com/thyn-ai/mojo-kernels>

License: Apache-2.0, © 2026 Algenta

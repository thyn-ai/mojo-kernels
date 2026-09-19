"""networkx-compatible graph algorithms, API-compatible with `networkx` itself.

`betweenness_centrality` and `single_source_dijkstra` accept the same
positional/keyword arguments, return the same shapes (dicts keyed by the
original node labels), and raise the same networkx exceptions as their
`networkx` counterparts — computed by a Mojo kernel where the platform
supports it (macOS arm64 / Linux x86_64), with a vendored pure-Python
fallback everywhere else.

Both backends share the graph conversion, rescaling, and error handling in
this module, so they cannot disagree about anything outside the inner loop;
the differential test suite asserts agreement with `networkx` within 1e-12
(in practice bit-for-bit) on both paths.

Scope (documented exclusions — out-of-scope inputs raise NotImplementedError
rather than silently computing something else):

  * simple graphs only: `nx.Graph` / `nx.DiGraph` (no parallel edges, so no
    `nx.MultiGraph` / `nx.MultiDiGraph`);
  * betweenness_centrality: exact computation only (no `k` sampling) and no
    endpoint inclusion (`endpoints=False`);
  * weights: a numeric edge attribute addressed by name (string `weight`
    parameter); callable weight functions are not supported. All arithmetic
    is IEEE-754 float64 (integer weights are converted exactly);
  * node/edge attributes other than the weight attribute are ignored.
"""

from __future__ import annotations

from itertools import chain

import networkx as nx
import numpy as np

from nx_mojo import _reference
from nx_mojo._native import (
    DIJKSTRA_CONTRADICTORY_PATHS,
    NativeGraph,
    NativeUnavailable,
)

__all__ = ["betweenness_centrality", "single_source_dijkstra"]

_INT32_MAX = 2**31 - 1


def _check_graph(G, what: str) -> None:
    """Reject out-of-scope graph types with a clear error (never silent)."""
    if G.is_multigraph():
        raise NotImplementedError(
            f"nx_mojo.{what} supports simple graphs only (nx.Graph / nx.DiGraph); "
            "multigraphs with parallel edges are not supported"
        )


def _to_csr(G, weight_attr):
    """Convert a networkx graph to CSR, preserving adjacency iteration order.

    Node ids are assigned in `G` iteration order; each node's arc list keeps
    the order of `G[v].items()`, which is exactly the order the oracle's
    traversals visit them. Returns (nodes, row_ptr, col, weight) as NumPy
    arrays (weight None when `weight_attr` is None); the native loader makes
    contiguous copies where needed and the fallback consumes them read-only.
    """
    nodes = list(G)
    n = len(nodes)
    if n > _INT32_MAX:
        raise ValueError(f"graph too large for nx_mojo (>{_INT32_MAX} nodes)")
    # G._adj is the same adjacency mapping networkx's own traversals use
    # (`G_succ = G._adj  # For speed-up`); iterating it per node gives every
    # arc in oracle order, for directed and undirected graphs alike.
    adj = G._adj
    get_adj = adj.__getitem__
    counts = np.fromiter(map(len, map(get_adj, nodes)), dtype=np.int64, count=n)
    row_ptr = np.empty(n + 1, dtype=np.int64)
    row_ptr[0] = 0
    np.cumsum(counts, out=row_ptr[1:])
    nnz = int(row_ptr[-1])
    # Fast path: node labels already are 0..n-1 (very common), so the
    # label->id map can be skipped in the hot loop.
    if nodes == list(range(n)):
        col = np.fromiter(
            chain.from_iterable(map(get_adj, nodes)), dtype=np.int32, count=nnz
        )
    else:
        index = {v: i for i, v in enumerate(nodes)}
        col = np.fromiter(
            map(index.__getitem__, chain.from_iterable(map(get_adj, nodes))),
            dtype=np.int32,
            count=nnz,
        )
    if weight_attr is None:
        return nodes, row_ptr, col, None
    # Weights: one data.get(attr, 1) per arc, like the oracle's weight
    # function. This loop is the unavoidable per-arc Python cost; the bulk
    # float64 conversion happens once in np.asarray.
    raw_weight = []
    w_append = raw_weight.append
    for v in nodes:
        for data in adj[v].values():
            w_append(data.get(weight_attr, 1))
    try:
        weight = np.asarray(raw_weight, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"non-numeric {weight_attr!r} edge weight: {exc}"
        ) from exc
    return nodes, row_ptr, col, weight


def _rescale(bc: list[float], n: int, *, normalized: bool, directed: bool) -> list[float]:
    """Oracle-identical rescaling for exact, endpoints-excluded betweenness."""
    # Mirrors networkx._rescale with k=None, endpoints=False: N counts the
    # valid (s, t) pairs with s != t that could pass through a node.
    N = n - 1
    if N < 2:
        return bc  # no rescaling necessary: b=0 for all nodes
    if normalized:
        # Divide by the number of valid (s, t) node pairs.
        scale = 1 / (N * (N - 1))
    else:
        # Undirected graphs count each unordered pair once (the raw sums
        # count every ordered pair), directed graphs are used as-is.
        correction = 1 if directed else 2
        scale = N / (N * correction)
    if scale != 1:
        for i in range(len(bc)):
            bc[i] *= scale
    return bc


def betweenness_centrality(G, k=None, normalized=True, weight=None, endpoints=False, seed=None):
    """Compute the shortest-path betweenness centrality for nodes.

    Drop-in replacement for `networkx.betweenness_centrality` restricted to
    exact computation (`k=None`) without endpoint inclusion; see the module
    docstring for the full scope. Returns a dict keyed by node, in `G`
    iteration order, matching the oracle's values within 1e-12 (in practice
    bit-for-bit).
    """
    if k is not None:
        raise NotImplementedError(
            "nx_mojo.betweenness_centrality supports exact computation only "
            "(k=None); sampled approximation is not supported"
        )
    if endpoints:
        raise NotImplementedError(
            "nx_mojo.betweenness_centrality supports endpoints=False only"
        )
    if callable(weight):
        raise NotImplementedError(
            "nx_mojo.betweenness_centrality requires `weight` to be an edge "
            "attribute name; callable weight functions are not supported"
        )
    _check_graph(G, "betweenness_centrality")
    del seed  # only consumed by networkx when k is not None
    n = G.number_of_nodes()
    if n == 0:
        return {}
    nodes, row_ptr, col, weights = _to_csr(G, weight)
    weighted = weight is not None
    raw: list[float] | None = None
    try:
        graph = NativeGraph(n, G.is_directed(), row_ptr, col, weights, weighted)
        try:
            raw = graph.betweenness(weighted).tolist()
        finally:
            graph.close()
    except NativeUnavailable:
        raw = None
    if raw is None:
        raw = _reference.brandes_raw(n, row_ptr, col, weights)
    bc = _rescale(raw, n, normalized=normalized, directed=G.is_directed())
    return {v: bc[i] for i, v in enumerate(nodes)}


def single_source_dijkstra(G, source, target=None, cutoff=None, weight="weight"):
    """Find shortest weighted paths and lengths from a source node.

    Drop-in replacement for `networkx.single_source_dijkstra` (see the module
    docstring for scope). Returns `(distance, path)`: two dicts keyed by node
    when `target` is None, otherwise the scalar distance and the node list
    for `target`. Distances are always float (the oracle returns ints for
    integer-weighted graphs; the values are equal).
    """
    _check_graph(G, "single_source_dijkstra")
    if callable(weight):
        raise NotImplementedError(
            "nx_mojo.single_source_dijkstra requires `weight` to be an edge "
            "attribute name; callable weight functions are not supported"
        )
    if source not in G:
        raise nx.NodeNotFound(f"Node {source} not found in graph")
    if target == source:  # networkx: `if target in sources`
        return (0, [target])
    nodes, row_ptr, col, weights = _to_csr(G, weight)
    index = {v: i for i, v in enumerate(nodes)}
    source_id = index[source]
    target_id = index[target] if target is not None and target in index else None
    # A target that is not a node can never be finalized; it takes the same
    # NetworkXNoPath path as an unreachable target below.
    try:
        graph = NativeGraph(len(nodes), G.is_directed(), row_ptr, col, weights, True)
        try:
            rc, dist_arr, parent_arr = graph.dijkstra(source_id, target_id, cutoff)
        finally:
            graph.close()
        if rc == DIJKSTRA_CONTRADICTORY_PATHS:
            raise ValueError("Contradictory paths found:", "negative weights?")
        if rc < 0:
            raise NativeUnavailable(f"native dijkstra rejected the call (status {rc})")
        dist_items = (
            (i, float(d)) for i, d in enumerate(dist_arr) if d != -1.0
        )
        parent_get = parent_arr.__getitem__
    except NativeUnavailable:
        dist, parent = _reference.dijkstra(
            len(nodes), row_ptr, col, weights, source_id, target_id, cutoff
        )
        dist_items = iter(dist.items())
        parent_get = parent.__getitem__
    if target is not None and target_id is None:
        raise nx.NetworkXNoPath(f"No path to {target}.")
    dist_out: dict = {}
    paths_out: dict = {}
    for i, d in dist_items:
        dist_out[nodes[i]] = d
        labels = [nodes[i]]
        j = i
        while j != source_id:
            j = parent_get(j)
            labels.append(nodes[j])
        labels.reverse()
        paths_out[nodes[i]] = labels
    if target_id is not None:
        try:
            return (dist_out[nodes[target_id]], paths_out[nodes[target_id]])
        except KeyError:
            raise nx.NetworkXNoPath(f"No path to {target}.") from None
    return (dist_out, paths_out)

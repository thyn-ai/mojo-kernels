"""Vendored pure-Python reference implementations for the fallback backend.

This is the fallback path used when the native Mojo kernel is unavailable
(unsupported platform, missing shared library, ABI mismatch, or
``NX_MOJO_DISABLE_NATIVE=1``). It is a clean-room implementation of Brandes'
betweenness algorithm (Brandes, "A Faster Algorithm for Betweenness
Centrality", 2001) and Dijkstra's shortest-path algorithm over CSR arrays,
written to be observably identical to the networkx oracle: same traversal
order, same priority-queue tie-break, same IEEE-754 float64 operation order,
so results agree bit-for-bit in practice (the differential suite asserts
agreement within 1e-12 on both backends).

Graph conversion, rescaling, and error handling are shared with the native
path in `nx_mojo.core`, so the two backends can never disagree about
anything outside the inner loop.
"""

from __future__ import annotations

from heapq import heappop, heappush

__all__ = ["brandes_raw", "dijkstra"]


def brandes_raw(
    n: int,
    row_ptr: list[int],
    col: list[int],
    weight: list[float] | None,
) -> list[float]:
    """Raw (unscaled) Brandes betweenness sums for every node.

    `weight` is None for the unweighted BFS variant; otherwise it holds one
    non-negative arc weight per CSR entry and the weighted (Dijkstra-based)
    variant is used.
    """
    bc = [0.0] * n
    for s in range(n):
        if weight is None:
            stack, preds, sigma = _bfs_count(n, row_ptr, col, s)
        else:
            stack, preds, sigma = _dijkstra_count(n, row_ptr, col, weight, s)
        # Dependency accumulation in reverse discovery order.
        delta = dict.fromkeys(stack, 0)
        while stack:
            w = stack.pop()
            coeff = (1 + delta[w]) / sigma[w]
            for v in preds[w]:
                delta[v] += sigma[v] * coeff
            if w != s:
                bc[w] += delta[w]
    return bc


def _bfs_count(n: int, row_ptr: list[int], col: list[int], s: int):
    """Shortest-path counting BFS; returns (S, P, sigma) in discovery order."""
    preds = [[] for _ in range(n)]
    sigma = [0.0] * n
    dist = [-1] * n
    sigma[s] = 1.0
    dist[s] = 0
    stack: list[int] = []
    queue = [s]
    head = 0
    while head < len(queue):
        v = queue[head]
        head += 1
        stack.append(v)
        dv = dist[v]
        sigmav = sigma[v]
        for a in range(row_ptr[v], row_ptr[v + 1]):
            w = col[a]
            if dist[w] < 0:
                queue.append(w)
                dist[w] = dv + 1
            if dist[w] == dv + 1:  # shortest path: count it
                sigma[w] += sigmav
                preds[w].append(v)
    return stack, preds, sigma


def _dijkstra_count(
    n: int,
    row_ptr: list[int],
    col: list[int],
    weight: list[float],
    s: int,
):
    """Weighted path counting; returns (S, P, sigma) in finalization order.

    Each heap entry carries the predecessor that produced its distance, and
    sigma[v] += sigma[pred] runs when the entry is popped and finalized
    (this leaves sigma[s] exactly twice the usual count; the accumulation
    only uses ratios sigma[v]/sigma[w], which are unaffected because
    doubling is exact in binary floating point).
    """
    preds = [[] for _ in range(n)]
    sigma = [0.0] * n
    seen: dict[int, float] = {s: 0}
    done: set[int] = set()
    sigma[s] = 1.0
    stack: list[int] = []
    counter = 0
    heap: list[tuple[float, int, int, int]] = [(0.0, counter, s, s)]
    while heap:
        dist, _, pred, v = heappop(heap)
        if v in done:
            continue  # already searched this node
        sigma[v] += sigma[pred]  # count paths
        done.add(v)
        stack.append(v)
        for a in range(row_ptr[v], row_ptr[v + 1]):
            w = col[a]
            vw_dist = dist + weight[a]
            if w not in done and (w not in seen or vw_dist < seen[w]):
                seen[w] = vw_dist
                counter += 1
                heappush(heap, (vw_dist, counter, v, w))
                sigma[w] = 0.0
                preds[w] = [v]
            elif vw_dist == seen[w]:  # handle equal paths
                # `w` is either finalized (which implies it has a `seen`
                # entry) or was seen with vw_dist >= seen[w]; either way
                # seen[w] exists here.
                sigma[w] += sigma[v]
                preds[w].append(v)
    return stack, preds, sigma


def dijkstra(
    n: int,
    row_ptr: list[int],
    col: list[int],
    weight: list[float],
    source: int,
    target: int | None,
    cutoff: float | None,
) -> tuple[dict[int, float], dict[int, int]]:
    """Single-source Dijkstra with first-discovery parent tracking.

    Returns (dist, parent): dist maps finalized nodes to their shortest
    distance, parent maps each finalized non-source node to the predecessor
    through which its best distance was first achieved. Equal-distance
    alternative parents are not recorded (matching the oracle, which only
    tracks them in a `pred` structure that single_source_dijkstra does not
    return). Raises ValueError on contradictory paths (negative weights).
    """
    dist: dict[int, float] = {}
    seen: dict[int, float] = {source: 0}
    parent: dict[int, int] = {}
    counter = 0
    fringe: list[tuple[float, int, int]] = [(0.0, counter, source)]
    while fringe:
        d, _, v = heappop(fringe)
        if v in dist:
            continue  # already searched this node
        dist[v] = d
        if target is not None and v == target:
            break
        for a in range(row_ptr[v], row_ptr[v + 1]):
            u = col[a]
            vu_dist = dist[v] + weight[a]
            if cutoff is not None and vu_dist > cutoff:
                continue
            if u in dist:
                if vu_dist < dist[u]:
                    raise ValueError("Contradictory paths found:", "negative weights?")
                # Equal-distance parent of a finalized node: pred-only, not
                # tracked here (same as the oracle's single_source_dijkstra).
            elif u not in seen or vu_dist < seen[u]:
                seen[u] = vu_dist
                counter += 1
                heappush(fringe, (vu_dist, counter, u))
                parent[u] = v
            # elif vu_dist == seen[u]: pred-only, not tracked.
    return dist, parent

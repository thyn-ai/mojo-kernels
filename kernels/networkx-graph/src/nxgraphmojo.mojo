"""Clean-room graph kernels: Brandes betweenness centrality and Dijkstra
single-source shortest paths on CSR graphs.

Written fresh from the published algorithms (Ulrik Brandes, "A Faster
Algorithm for Betweenness Centrality", 2001; Dijkstra's algorithm in its
priority-queue formulation). No third-party Mojo code is used or adapted.

The kernel deliberately reproduces the observable IEEE-754 float64 behaviour
of the networkx oracle (differential-tested against pip networkx 3.5):

  * BFS / Dijkstra visit order follows the CSR neighbour order, which the
    wrapper builds in networkx adjacency-iteration order.
  * The priority queue is ordered by (distance, push-counter), so equal
    distances pop in first-in-first-out order, matching heapq tuples.
  * Weighted path counting increments sigma[v] at pop time from the stored
    predecessor of the popped heap entry (sigma[s] ends up exactly doubled;
    the ratio sigma[v]/sigma[w] used by the accumulation is unaffected
    bit-for-bit, because doubling is exact in binary floating point).
  * Accumulation processes nodes in reverse discovery order and computes
    coeff = (1 + delta[w]) / sigma[w] once per node, then
    delta[v] += sigma[v] * coeff per predecessor.

Exported C ABI (v1):

    int32_t  nxgraphmojo_abi_version(void)
    void*    nxgraphmojo_graph_create(int64_t n, int32_t directed,
                                      const int64_t* row_ptr,
                                      const int32_t* col,
                                      const double* weight,
                                      int32_t has_weight)
    void     nxgraphmojo_graph_destroy(void* handle)
    int32_t  nxgraphmojo_betweenness(void* handle, int32_t use_weights,
                                     double* out_bc)
    int64_t  nxgraphmojo_dijkstra(void* handle, int64_t source,
                                  int32_t has_target, int64_t target,
                                  int32_t has_cutoff, double cutoff,
                                  double* out_dist, int64_t* out_parent)

Graphs are simple (no parallel arcs per ordered pair is not required by the
kernel itself, but the wrapper only ever submits simple graphs); self-loops
are allowed and follow the same relaxation rules as any other arc.
"""

from std.memory import Pointer, unsafe_memcpy
from std.memory.alloc import unsafe_alloc
from std.origin import MutUntrackedOrigin

comptime ABI_VERSION: Int32 = 1

# C-side pointer spellings (untracked origin: the caller owns the lifetime of
# anything passed in; the library owns what it allocates).
comptime F64Ptr = Pointer[Float64, MutUntrackedOrigin]
comptime I32Ptr = Pointer[Int32, MutUntrackedOrigin]
comptime I64Ptr = Pointer[Int64, MutUntrackedOrigin]
comptime U8Ptr = Pointer[UInt8, MutUntrackedOrigin]
comptime Handle = Optional[Pointer[UInt8, MutUntrackedOrigin]]

# nxgraphmojo_betweenness status codes.
comptime BC_OK: Int32 = 0
comptime BC_NULL_HANDLE: Int32 = 1
comptime BC_NO_WEIGHTS: Int32 = 2

# nxgraphmojo_dijkstra negative status codes (non-negative values are the
# number of finalized nodes).
comptime DIJKSTRA_NULL_HANDLE: Int64 = -1
comptime DIJKSTRA_BAD_NODE: Int64 = -2
comptime DIJKSTRA_CONTRADICTORY_PATHS: Int64 = -3
comptime DIJKSTRA_NO_WEIGHTS: Int64 = -4


struct NxGraph(Copyable, Movable):
    """Owned, native copy of one CSR graph (with optional arc weights)."""

    var n: Int64
    var directed: Int32
    var nnz: Int64
    var has_weight: Int32
    var row_ptr: I64Ptr  # [n + 1]
    var col: I32Ptr  # [nnz]
    var weight: F64Ptr  # [nnz] when has_weight != 0, else a 1-slot dummy

    def __init__(
        out self,
        n: Int64,
        directed: Int32,
        nnz: Int64,
        has_weight: Int32,
        row_ptr: I64Ptr,
        col: I32Ptr,
        weight: F64Ptr,
    ):
        self.n = n
        self.directed = directed
        self.nnz = nnz
        self.has_weight = has_weight
        self.row_ptr = row_ptr
        self.col = col
        self.weight = weight


# --------------------------------------------------------------------------
# Binary min-heap ordered by (distance, push counter), matching the pop
# order of Python heapq over (dist, count, ...) tuples.
# --------------------------------------------------------------------------


struct MinHeap(Copyable, Movable):
    """Parallel-array binary heap; capacity is fixed at construction."""

    var dist: F64Ptr
    var count: I64Ptr
    var pred: I32Ptr
    var node: I32Ptr
    var size: Int64
    var capacity: Int64

    def __init__(out self, capacity: Int64):
        self.capacity = capacity
        self.size = 0
        self.dist = unsafe_alloc[Float64](Int(capacity))
        self.count = unsafe_alloc[Int64](Int(capacity))
        self.pred = unsafe_alloc[Int32](Int(capacity))
        self.node = unsafe_alloc[Int32](Int(capacity))

    def destroy(mut self):
        self.dist.unsafe_free()
        self.count.unsafe_free()
        self.pred.unsafe_free()
        self.node.unsafe_free()

    @always_inline
    def _less(self, i: Int, j: Int) -> Bool:
        """Total order: distance first, then push counter (FIFO tie-break)."""
        var di = self.dist[unsafe_offset=i]
        var dj = self.dist[unsafe_offset=j]
        if di != dj:
            return di < dj
        return self.count[unsafe_offset=i] < self.count[unsafe_offset=j]

    def push(mut self, dist: Float64, count: Int64, pred: Int32, node: Int32):
        var i = Int(self.size)
        self.dist[unsafe_offset=i] = dist
        self.count[unsafe_offset=i] = count
        self.pred[unsafe_offset=i] = pred
        self.node[unsafe_offset=i] = node
        self.size += 1
        # Sift up.
        while i > 0:
            var parent = (i - 1) >> 1
            if self._less(i, parent):
                self._swap(i, parent)
                i = parent
            else:
                break

    def pop(mut self) -> Tuple[Float64, Int64, Int32, Int32]:
        """Remove and return the minimum entry. Undefined when empty."""
        var top_dist = self.dist[unsafe_offset=0]
        var top_count = self.count[unsafe_offset=0]
        var top_pred = self.pred[unsafe_offset=0]
        var top_node = self.node[unsafe_offset=0]
        self.size -= 1
        var last = Int(self.size)
        if last > 0:
            self.dist[unsafe_offset=0] = self.dist[unsafe_offset=last]
            self.count[unsafe_offset=0] = self.count[unsafe_offset=last]
            self.pred[unsafe_offset=0] = self.pred[unsafe_offset=last]
            self.node[unsafe_offset=0] = self.node[unsafe_offset=last]
            # Sift down.
            var i = 0
            while True:
                var left = 2 * i + 1
                var right = left + 1
                var smallest = i
                if left < last and self._less(left, smallest):
                    smallest = left
                if right < last and self._less(right, smallest):
                    smallest = right
                if smallest == i:
                    break
                self._swap(i, smallest)
                i = smallest
        return {top_dist, top_count, top_pred, top_node}

    def _swap(mut self, i: Int, j: Int):
        var td = self.dist[unsafe_offset=i]
        self.dist[unsafe_offset=i] = self.dist[unsafe_offset=j]
        self.dist[unsafe_offset=j] = td
        var tc = self.count[unsafe_offset=i]
        self.count[unsafe_offset=i] = self.count[unsafe_offset=j]
        self.count[unsafe_offset=j] = tc
        var tp = self.pred[unsafe_offset=i]
        self.pred[unsafe_offset=i] = self.pred[unsafe_offset=j]
        self.pred[unsafe_offset=j] = tp
        var tn = self.node[unsafe_offset=i]
        self.node[unsafe_offset=i] = self.node[unsafe_offset=j]
        self.node[unsafe_offset=j] = tn


# --------------------------------------------------------------------------
# Per-source scratch shared by BFS and Dijkstra path counting.
# --------------------------------------------------------------------------


struct BrandesWorkspace:
    """Reusable scratch for the all-sources Brandes loop.

    Predecessor lists are stored as per-node linked lists in a per-source
    arena (`pnode`/`pnext`/`phead`): prepending reverses insertion order,
    which does not affect results — for a fixed node w every predecessor v
    is distinct, so the accumulation additions `delta[v] += sigma[v]*coeff`
    target distinct accumulators in any iteration order. The order that
    matters (reverse discovery order over w) is preserved exactly.
    """

    var n: Int
    var nnz: Int
    var sigma: F64Ptr  # [n]
    var delta: F64Ptr  # [n]
    var dist: I32Ptr  # [n] BFS levels (-1 = unreached)
    var queue: I32Ptr  # [n] BFS FIFO
    var stack: I32Ptr  # [n] discovery order S
    var phead: I64Ptr  # [n] head of predecessor list (-1 = empty)
    var pnode: I32Ptr  # [nnz + 1] predecessor arena
    var pnext: I64Ptr  # [nnz + 1]
    var seen: F64Ptr  # [n] Dijkstra tentative distances
    var in_seen: U8Ptr  # [n]
    var done: U8Ptr  # [n] finalized

    def __init__(out self, n: Int, nnz: Int):
        self.n = n
        self.nnz = nnz
        var arena = nnz + 1
        self.sigma = unsafe_alloc[Float64](n)
        self.delta = unsafe_alloc[Float64](n)
        self.dist = unsafe_alloc[Int32](n)
        self.queue = unsafe_alloc[Int32](n)
        self.stack = unsafe_alloc[Int32](n)
        self.phead = unsafe_alloc[Int64](n)
        self.pnode = unsafe_alloc[Int32](arena)
        self.pnext = unsafe_alloc[Int64](arena)
        self.seen = unsafe_alloc[Float64](n)
        self.in_seen = unsafe_alloc[UInt8](n)
        self.done = unsafe_alloc[UInt8](n)

    def destroy(mut self):
        self.sigma.unsafe_free()
        self.delta.unsafe_free()
        self.dist.unsafe_free()
        self.queue.unsafe_free()
        self.stack.unsafe_free()
        self.phead.unsafe_free()
        self.pnode.unsafe_free()
        self.pnext.unsafe_free()
        self.seen.unsafe_free()
        self.in_seen.unsafe_free()
        self.done.unsafe_free()

    def reset(mut self):
        for i in range(self.n):
            self.sigma[unsafe_offset=i] = 0.0
            self.delta[unsafe_offset=i] = 0.0
            self.dist[unsafe_offset=i] = -1
            self.phead[unsafe_offset=i] = -1
            self.seen[unsafe_offset=i] = 0.0
            self.in_seen[unsafe_offset=i] = 0
            self.done[unsafe_offset=i] = 0

    @always_inline
    def push_pred(mut self, w: Int, v: Int, mut arena_count: Int):
        """Prepend predecessor v to P[w] (arena slot arena_count)."""
        self.pnode[unsafe_offset=arena_count] = Int32(v)
        self.pnext[unsafe_offset=arena_count] = self.phead[unsafe_offset=w]
        self.phead[unsafe_offset=w] = Int64(arena_count)
        arena_count += 1

# --------------------------------------------------------------------------
# ABI
# --------------------------------------------------------------------------


@export
def nxgraphmojo_abi_version() abi("C") -> Int32:
    return ABI_VERSION


@export
def nxgraphmojo_graph_create(
    n: Int64,
    directed: Int32,
    row_ptr: I64Ptr,
    col: I32Ptr,
    weight: F64Ptr,
    has_weight: Int32,
) abi("C") -> Handle:
    """Copy the caller's CSR buffers into a native graph; NULL on invalid input.

    `weight` must point to at least one slot even when `has_weight` == 0
    (it is never read in that case). Rejects negative sizes, non-monotonic
    row offsets, and column indices outside [0, n); n == 0 is allowed.
    """
    if n < 0:
        return None
    var nnz = Int64(0)
    if n > 0:
        nnz = row_ptr[unsafe_offset=Int(n)]
        if nnz < 0:
            return None
        var prev = Int64(0)
        if row_ptr[unsafe_offset=0] != 0:
            return None
        for i in range(1, Int(n) + 1):
            var cur = row_ptr[unsafe_offset=i]
            if cur < prev:
                return None
            prev = cur
    var rp_copy = unsafe_alloc[Int64](Int(n) + 1)
    unsafe_memcpy(dest=rp_copy, src=row_ptr, count=Int(n) + 1)
    var col_copy = unsafe_alloc[Int32](Int(nnz) + 1)
    var w_copy = unsafe_alloc[Float64](Int(nnz) + 1)
    if nnz > 0:
        for a in range(Int(nnz)):
            var c = col[unsafe_offset=a]
            if c < 0 or Int64(c) >= n:
                rp_copy.unsafe_free()
                col_copy.unsafe_free()
                w_copy.unsafe_free()
                return None
            col_copy[unsafe_offset=a] = c
        if has_weight != 0:
            unsafe_memcpy(dest=w_copy, src=weight, count=Int(nnz))

    var g = unsafe_alloc[NxGraph](1)
    g[] = NxGraph(
        n, directed, nnz, has_weight, rp_copy, col_copy, w_copy
    )
    return g.unsafe_bitcast[UInt8]()


@export
def nxgraphmojo_graph_destroy(handle: Handle) abi("C"):
    if not handle:
        return
    var g = handle.value().unsafe_bitcast[NxGraph]()
    g[].row_ptr.unsafe_free()
    g[].col.unsafe_free()
    g[].weight.unsafe_free()
    g.unsafe_free()


def _bfs_count(g: NxGraph, s: Int, mut ws: BrandesWorkspace) -> Int:
    """Unweighted shortest-path counting BFS from s; returns len(S).

    Mirrors the oracle's level-order traversal: neighbours are scanned in
    CSR order, discovered nodes are enqueued FIFO, and a node is appended
    to S when dequeued.
    """
    var row_ptr = g.row_ptr
    var col = g.col
    var arena_count = 0
    var head = 0
    var tail = 0
    var scount = 0
    ws.dist[unsafe_offset=s] = 0
    ws.sigma[unsafe_offset=s] = 1.0
    ws.queue[unsafe_offset=tail] = Int32(s)
    tail += 1
    while head < tail:
        var v = Int(ws.queue[unsafe_offset=head])
        head += 1
        ws.stack[unsafe_offset=scount] = Int32(v)
        scount += 1
        var dv = ws.dist[unsafe_offset=v]
        var sigmav = ws.sigma[unsafe_offset=v]
        var a = Int(row_ptr[unsafe_offset=v])
        var end = Int(row_ptr[unsafe_offset=v + 1])
        while a < end:
            var w = Int(col[unsafe_offset=a])
            if ws.dist[unsafe_offset=w] < 0:
                ws.queue[unsafe_offset=tail] = Int32(w)
                tail += 1
                ws.dist[unsafe_offset=w] = dv + 1
            if ws.dist[unsafe_offset=w] == dv + 1:
                ws.sigma[unsafe_offset=w] += sigmav
                ws.push_pred(w, v, arena_count)
            a += 1
    return scount


def _dijkstra_count(
    g: NxGraph, s: Int, mut ws: BrandesWorkspace, mut heap: MinHeap
) -> Int:
    """Weighted shortest-path counting from s; returns len(S).

    Mirrors the oracle's pop-time path counting: each heap entry carries
    the predecessor that produced its (best) distance, and
    sigma[v] += sigma[pred] runs when the entry is popped and finalized.
    Equal-distance relaxations of an unfinalized node add to sigma and the
    predecessor list immediately; improvements reset both.
    """
    var row_ptr = g.row_ptr
    var col = g.col
    var weight = g.weight
    var arena_count = 0
    var counter = Int64(0)
    var scount = 0
    ws.sigma[unsafe_offset=s] = 1.0
    ws.seen[unsafe_offset=s] = 0.0
    ws.in_seen[unsafe_offset=s] = 1
    heap.push(0.0, counter, Int32(s), Int32(s))
    counter += 1
    while heap.size > 0:
        var popped = heap.pop()
        var d = popped[0]
        var pred = Int(popped[2])
        var v = Int(popped[3])
        if ws.done[unsafe_offset=v] != 0:
            continue
        ws.sigma[unsafe_offset=v] += ws.sigma[unsafe_offset=pred]
        ws.done[unsafe_offset=v] = 1
        ws.stack[unsafe_offset=scount] = Int32(v)
        scount += 1
        var a = Int(row_ptr[unsafe_offset=v])
        var end = Int(row_ptr[unsafe_offset=v + 1])
        while a < end:
            var w = Int(col[unsafe_offset=a])
            var vw_dist = d + weight[unsafe_offset=a]
            if ws.done[unsafe_offset=w] == 0 and (
                ws.in_seen[unsafe_offset=w] == 0
                or vw_dist < ws.seen[unsafe_offset=w]
            ):
                ws.seen[unsafe_offset=w] = vw_dist
                ws.in_seen[unsafe_offset=w] = 1
                heap.push(vw_dist, counter, Int32(v), Int32(w))
                counter += 1
                ws.sigma[unsafe_offset=w] = 0.0
                ws.phead[unsafe_offset=w] = -1
                ws.push_pred(w, v, arena_count)
            elif ws.in_seen[unsafe_offset=w] != 0 and vw_dist == ws.seen[
                unsafe_offset=w
            ]:
                ws.sigma[unsafe_offset=w] += ws.sigma[unsafe_offset=v]
                ws.push_pred(w, v, arena_count)
            a += 1
    return scount


@export
def nxgraphmojo_betweenness(
    handle: Handle,
    use_weights: Int32,
    out_bc: F64Ptr,
) abi("C") -> Int32:
    """Raw (unscaled) Brandes betweenness sums into out_bc[n].

    The wrapper applies the normalization/directed rescaling; this kernel
    always returns the sum of pair-dependencies over all sources in node
    order. Returns 0 on success, 1 on a NULL handle, 2 when weights were
    requested for a graph created without them.
    """
    if not handle:
        return BC_NULL_HANDLE
    var g = handle.value().unsafe_bitcast[NxGraph]()
    var n = Int(g[].n)
    for i in range(n):
        out_bc[unsafe_offset=i] = 0.0
    if n == 0:
        return BC_OK
    if use_weights != 0 and g[].has_weight == 0:
        return BC_NO_WEIGHTS

    var ws = BrandesWorkspace(n, Int(g[].nnz))
    var heap = MinHeap(Int64(g[].nnz) + 1)
    for s in range(n):
        ws.reset()
        var scount: Int
        if use_weights != 0:
            heap.size = 0
            scount = _dijkstra_count(g[], s, ws, heap)
        else:
            scount = _bfs_count(g[], s, ws)
        # Accumulate over S in reverse discovery order.
        var i = scount - 1
        while i >= 0:
            var w = Int(ws.stack[unsafe_offset=i])
            var coeff = (1.0 + ws.delta[unsafe_offset=w]) / ws.sigma[
                unsafe_offset=w
            ]
            var p = ws.phead[unsafe_offset=w]
            while p != -1:
                var v = Int(ws.pnode[unsafe_offset=Int(p)])
                # delta[v] += sigma[v] * coeff — the multiply must round
                # separately from the add, exactly like the oracle's
                # CPython evaluation. The kernel is built with
                # `--fp-mode contract=off` so LLVM cannot fuse this into a
                # single-rounded FMA (which drifts by ~1 ulp per update).
                ws.delta[unsafe_offset=v] += (
                    ws.sigma[unsafe_offset=v] * coeff
                )
                p = ws.pnext[unsafe_offset=Int(p)]
            if w != s:
                out_bc[unsafe_offset=w] += ws.delta[unsafe_offset=w]
            i -= 1
    heap.destroy()
    ws.destroy()
    return BC_OK


@export
def nxgraphmojo_dijkstra(
    handle: Handle,
    source: Int64,
    has_target: Int32,
    target: Int64,
    has_cutoff: Int32,
    cutoff: Float64,
    out_dist: F64Ptr,
    out_parent: I64Ptr,
) abi("C") -> Int64:
    """Single-source Dijkstra with first-discovery parent tracking.

    out_dist receives -1.0 for unfinalized nodes and the (non-negative)
    shortest distance otherwise; out_parent receives -1 for the source and
    for unreached nodes, and the discovering predecessor otherwise. Equal
    -distance alternative parents are not recorded (the oracle only tracks
    them in `pred`, which single_source_dijkstra does not return). Returns
    the number of finalized nodes (>= 0) or a negative status: -1 NULL
    handle, -2 source/target out of range, -3 contradictory paths (negative
    weights), -4 the graph has no weights.
    """
    if not handle:
        return DIJKSTRA_NULL_HANDLE
    var g = handle.value().unsafe_bitcast[NxGraph]()
    var n = Int(g[].n)
    if source < 0 or source >= g[].n:
        return DIJKSTRA_BAD_NODE
    if has_target != 0 and (target < 0 or target >= g[].n):
        return DIJKSTRA_BAD_NODE
    if g[].has_weight == 0:
        return DIJKSTRA_NO_WEIGHTS
    var row_ptr = g[].row_ptr
    var col = g[].col
    var weight = g[].weight

    var seen = unsafe_alloc[Float64](n)
    var in_seen = unsafe_alloc[UInt8](n)
    for i in range(n):
        out_dist[unsafe_offset=i] = -1.0
        out_parent[unsafe_offset=i] = -1
        seen[unsafe_offset=i] = 0.0
        in_seen[unsafe_offset=i] = 0

    var heap = MinHeap(g[].nnz + 1)
    var counter = Int64(0)
    var finalized = Int64(0)
    var s = Int(source)
    seen[unsafe_offset=s] = 0.0
    in_seen[unsafe_offset=s] = 1
    heap.push(0.0, counter, Int32(s), Int32(s))
    counter += 1
    var status = finalized
    var stop = False
    while heap.size > 0 and not stop:
        var popped = heap.pop()
        var d = popped[0]
        var v = Int(popped[3])
        if out_dist[unsafe_offset=v] != -1.0:
            continue  # already finalized
        out_dist[unsafe_offset=v] = d
        finalized += 1
        if has_target != 0 and v == Int(target):
            break
        var a = Int(row_ptr[unsafe_offset=v])
        var end = Int(row_ptr[unsafe_offset=v + 1])
        while a < end:
            var u = Int(col[unsafe_offset=a])
            var vu_dist = d + weight[unsafe_offset=a]
            if has_cutoff != 0 and vu_dist > cutoff:
                a += 1
                continue
            if out_dist[unsafe_offset=u] != -1.0:
                if vu_dist < out_dist[unsafe_offset=u]:
                    status = DIJKSTRA_CONTRADICTORY_PATHS
                    stop = True
                    break
                # Equal-distance parent of a finalized node: only recorded
                # in the oracle's `pred`, which this call does not return.
            elif in_seen[unsafe_offset=u] == 0 or vu_dist < seen[
                unsafe_offset=u
            ]:
                seen[unsafe_offset=u] = vu_dist
                in_seen[unsafe_offset=u] = 1
                heap.push(vu_dist, counter, Int32(v), Int32(u))
                counter += 1
                out_parent[unsafe_offset=u] = Int64(v)
            # elif vu_dist == seen[u]: pred-only, not tracked.
            a += 1
    heap.destroy()
    seen.unsafe_free()
    in_seen.unsafe_free()
    if status < 0:
        return status
    return finalized

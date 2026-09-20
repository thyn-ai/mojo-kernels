"""Clean-room MOT metrics accumulator (MOTA / MOTP / IDF1 / MOSTLY_*) for
MOTChallenge-style event streams.

Written fresh from the published metric definitions (Bernardin & Stiefelhagen,
"Evaluating Multiple Object Tracking Performance: The CLEAR MOT Metrics",
EURASIP JIVP 2008; Ristani et al., "Performance Measures and a Data Set for
Multi-Target, Multi-Camera Tracking", ECCVW 2016, for the ID measures) and the
publicly documented event semantics of the reference implementation it is
checked against (py-motmetrics 1.4.0: per-frame carry-forward of established
tracks, then a minimum-cost assignment for the remainder, with MATCH / SWITCH
/ TRANSFER / ASCEND / MIGRATE / MISS / FP events). No third-party Mojo code is
used or adapted.

The per-frame assignment uses the shortest-augmenting-path rectangular
assignment algorithm as described in pages 1685-1686 of

    D.F. Crouse. On implementing 2D rectangular assignment algorithms.
    IEEE Transactions on Aerospace and Electronic Systems 52(4):1679-1696,
    August 2016. doi: 10.1109/TAES.2016.140952

replicating the scan and tie-breaking order of the reference solver (reverse
initial column order, strict-less relaxation, prefer-unassigned-column on
ties, swap-remove of visited columns) so that assignments — and therefore the
accumulated event counts — agree with the oracle even on tie-heavy integer
cost matrices. Non-edges are substituted by the same large finite penalty the
reference wrapper uses (2 * min(shape) * (max|finite cost| + 1) + 1), which
keeps every matrix feasible and never changes the optimal finite matching.

IDF1 needs only the optimal value of the global identity assignment, not the
assignment itself: with the reference's cost construction every optimal
solution has idfp = predictions - idtp and idfn = objects - idtp, where
idtp is the maximum total co-occurrence count over object/hypothesis
matchings. That maximum is unique even when the matching is not, so IDF1 is
computed here as a max-weight matching over the sparse co-occurrence counts
(frames with a finite object/hypothesis distance), which is exactly the
reference value while avoiding the (objects + hypotheses)^2 dense matrix.

All state lives in this kernel: the Python wrapper streams each frame's
ids + distance matrix through `motmojo_update` and reads the accumulated
counters once from `motmojo_finalize`. Frame ids and object/hypothesis ids
are signed 64-bit integers (MOTChallenge format).

Exported C ABI (v1):

    int32_t  motmojo_abi_version(void)
    void*    motmojo_create(double max_switch_time)
    int32_t  motmojo_update(handle, oids, n_oids, hids, n_hids, dists, frameid)
    int32_t  motmojo_finalize(handle, dst, capacity)
    void     motmojo_destroy(handle)
"""

from std.collections import Dict, List
from std.math import inf, isfinite
from std.memory import Pointer
from std.memory.alloc import unsafe_alloc
from std.origin import MutUntrackedOrigin

comptime ABI_VERSION: Int32 = 1

# Number of float64 slots `motmojo_finalize` writes; mirrored by the wrapper.
comptime COUNTS_LEN: Int64 = 21

# C-side pointer spelling (untracked origin: the caller owns every buffer
# passed in; the library owns only what it allocates).
comptime F64Ptr = Pointer[Float64, MutUntrackedOrigin]
comptime I64Ptr = Pointer[Int64, MutUntrackedOrigin]
comptime Handle = Optional[Pointer[UInt8, MutUntrackedOrigin]]

comptime POS_INF = inf[DType.float64]()


struct PairKey(Copyable, Movable, Hashable, Equatable):
    """(object id, hypothesis id) dictionary key."""

    var a: Int64
    var b: Int64

    def __init__(out self, a: Int64, b: Int64):
        self.a = a
        self.b = b

    def __eq__(self, other: PairKey) -> Bool:
        return self.a == other.a and self.b == other.b

    def __hash__(self) -> UInt:
        var h1 = hash(self.a)
        var h2 = hash(self.b)
        return UInt(h1 ^ (h2 << 1) ^ (h2 >> 3))


struct ObjStats(Copyable, Movable):
    """Per ground-truth-object running tallies.

    `present` counts appearances (one per frame the id is passed to update);
    `ocs` counts distinct frame ids (the IDF1 co-occurrence base); `matched`
    counts MATCH+SWITCH events; the remaining fields drive the fragmentation
    count: a fragmentation is a tracked -> not-tracked transition that is
    later closed by another tracked period inside the object's tracked span
    (a trailing miss at end of life is not a fragmentation).
    """

    var present: Int64
    var ocs: Int64
    var matched: Int64
    var frag: Int64
    var last_ocs_frame: Int64
    var ever_matched: Bool
    var prev_miss: Bool
    var pending: Bool

    def __init__(out self):
        self.present = 0
        self.ocs = 0
        self.matched = 0
        self.frag = 0
        self.last_ocs_frame = Int64.MIN
        self.ever_matched = False
        self.prev_miss = False
        self.pending = False


struct HypStats(Copyable, Movable):
    """Per-hypothesis tallies (`present` per frame, `hcs` per distinct frame)."""

    var present: Int64
    var hcs: Int64
    var last_frame: Int64

    def __init__(out self):
        self.present = 0
        self.hcs = 0
        self.last_frame = Int64.MIN


struct MotState(Copyable, Movable):
    """Whole accumulation state for one event stream."""

    var max_switch_time: Float64
    # Tracking state carried across frames.
    var m: Dict[Int64, Int64]  # object id -> currently paired hypothesis id
    var res_m: Dict[Int64, Int64]  # hypothesis id -> object id it last resulted from
    var last_match: Dict[Int64, Int64]  # object id -> frame of its last match
    var last_occ: Dict[Int64, Int64]  # object id -> frame it was last present
    var hyp_hist: Dict[Int64, Int64]  # hypothesis id -> frame it last matched
    # Metric tallies.
    var obj: Dict[Int64, ObjStats]
    var hyp: Dict[Int64, HypStats]
    var ex: Dict[PairKey, Int64]  # (object, hypothesis) -> finite-distance frames
    var frames: Dict[Int64, Bool]  # distinct frame ids seen
    var matches: Int64
    var switches: Int64
    var misses: Int64
    var fps: Int64
    var transfers: Int64
    var ascends: Int64
    var migrates: Int64
    var dist_sum: Float64  # sum of match/switch distances, in event order

    def __init__(out self, max_switch_time: Float64):
        self.max_switch_time = max_switch_time
        self.m = Dict[Int64, Int64]()
        self.res_m = Dict[Int64, Int64]()
        self.last_match = Dict[Int64, Int64]()
        self.last_occ = Dict[Int64, Int64]()
        self.hyp_hist = Dict[Int64, Int64]()
        self.obj = Dict[Int64, ObjStats]()
        self.hyp = Dict[Int64, HypStats]()
        self.ex = Dict[PairKey, Int64]()
        self.frames = Dict[Int64, Bool]()
        self.matches = 0
        self.switches = 0
        self.misses = 0
        self.fps = 0
        self.transfers = 0
        self.ascends = 0
        self.migrates = 0
        self.dist_sum = 0.0


def _abs_i64(v: Int64) -> Int64:
    return v if v >= 0 else -v


def _bump_ex(mut ex: Dict[PairKey, Int64], key: PairKey):
    if key in ex:
        try:
            ex[key] += 1
        except:
            pass
    else:
        ex[key.copy()] = 1


def _lsap_solve(cost: F64Ptr, nr: Int, nc: Int, mut col4row: List[Int64]):
    """Minimum-cost assignment of all `nr` rows to distinct columns.

    Requires nr <= nc and a fully finite cost matrix. Shortest augmenting
    path per row (Crouse 2016, pp. 1685-1686) with the reference solver's
    exact scan order and tie rules: the candidate column list starts in
    reverse index order (so a constant cost matrix yields the identity), a
    strict less-than relaxes path costs, and on a tie the column that would
    extend the matching (currently unassigned) is preferred. col4row must
    have length >= nr; on return col4row[i] is row i's assigned column.
    """
    var u = List[Float64](length=nr, fill=0.0)
    var v = List[Float64](length=nc, fill=0.0)
    var row4col = List[Int64](length=nc, fill=-1)
    for i in range(nr):
        col4row[i] = -1
    var path = List[Int64](length=nc, fill=-1)
    var spc = List[Float64](length=nc, fill=0.0)
    var remaining = List[Int64](length=nc, fill=0)
    var sr = List[Bool](length=nr, fill=False)
    var sc = List[Bool](length=nc, fill=False)
    for cur_row in range(nr):
        for it in range(nc):
            remaining[it] = Int64(nc - it - 1)
        for j in range(nc):
            spc[j] = POS_INF
            path[j] = -1
            sc[j] = False
        for i in range(nr):
            sr[i] = False
        var min_val = 0.0
        var sink = Int64(-1)
        var i = Int64(cur_row)
        var num_remaining = nc
        while sink == -1:
            var index = -1
            var lowest = POS_INF
            sr[Int(i)] = True
            for it in range(num_remaining):
                var j = remaining[it]
                var r = min_val + cost[unsafe_offset=Int(i) * nc + Int(j)] - u[Int(i)] - v[Int(j)]
                if r < spc[Int(j)]:
                    path[Int(j)] = i
                    spc[Int(j)] = r
                if spc[Int(j)] < lowest or (
                    spc[Int(j)] == lowest and row4col[Int(j)] == -1
                ):
                    lowest = spc[Int(j)]
                    index = it
            min_val = lowest
            var j = remaining[index]
            if row4col[Int(j)] == -1:
                sink = j
            else:
                i = row4col[Int(j)]
            sc[Int(j)] = True
            remaining[index] = remaining[num_remaining - 1]
            num_remaining -= 1
        u[cur_row] += min_val
        for r2 in range(nr):
            if sr[r2] and r2 != cur_row:
                u[r2] += min_val - spc[Int(col4row[r2])]
        for j2 in range(nc):
            if sc[j2]:
                v[j2] -= min_val - spc[j2]
        var j3 = sink
        while True:
            var i2 = path[Int(j3)]
            row4col[Int(j3)] = i2
            var tmp = col4row[Int(i2)]
            col4row[Int(i2)] = j3
            j3 = tmp
            if i2 == Int64(cur_row):
                break


@export
def motmojo_abi_version() abi("C") -> Int32:
    return ABI_VERSION


@export
def motmojo_create(max_switch_time: Float64) abi("C") -> Handle:
    """Create an accumulator. `max_switch_time` bounds the frame-id distance
    over which a re-appearing object may still produce SWITCH events (the
    reference default is +inf: no bound). NaN/negative values are rejected
    (NULL handle); +inf is allowed."""
    if max_switch_time != max_switch_time or max_switch_time < 0.0:
        return None
    var s = unsafe_alloc[MotState](1)
    s[] = MotState(max_switch_time)
    return s.unsafe_bitcast[UInt8]()


@export
def motmojo_update(
    handle: Handle,
    oids: I64Ptr,
    n_oids: Int64,
    hids: I64Ptr,
    n_hids: Int64,
    dists: F64Ptr,
    frameid: Int64,
) abi("C") -> Int32:
    """Accumulate one frame. `dists` is the row-major n_oids x n_hids distance
    matrix; NaN and +/-inf entries signal do-not-pair constellations. Object
    and hypothesis ids must each be unique within the frame (MOTChallenge
    format). Returns 0, 1 on a NULL handle, 2 on invalid sizes."""
    if not handle:
        return 1
    if n_oids < 0 or n_hids < 0:
        return 2
    var s = handle.value().unsafe_bitcast[MotState]()
    var no = Int(n_oids)
    var nh = Int(n_hids)

    s[].frames[frameid] = True

    # Presence tallies (per appearance) and distinct-frame bases for IDF1.
    for i in range(no):
        var oid = oids[unsafe_offset=i]
        var st = s[].obj.get(oid, ObjStats())
        st.present += 1
        if st.last_ocs_frame != frameid:
            st.ocs += 1
        st.last_ocs_frame = frameid
        s[].obj[oid] = st.copy()
    for j in range(nh):
        var hid = hids[unsafe_offset=j]
        var hs = s[].hyp.get(hid, HypStats())
        hs.present += 1
        if hs.last_frame != frameid:
            hs.hcs += 1
        hs.last_frame = frameid
        s[].hyp[hid] = hs.copy()
    # Co-occurrence counts: one per finite distance entry (the IDF1 base).
    for i in range(no):
        var oid = oids[unsafe_offset=i]
        for j in range(nh):
            if isfinite(dists[unsafe_offset=i * nh + j]):
                _bump_ex(s[].ex, PairKey(oid, hids[unsafe_offset=j]))

    var omask = List[Bool](length=no, fill=False)
    var hmask = List[Bool](length=nh, fill=False)

    if no > 0 and nh > 0:
        # 1. Carry forward established tracks (object array order).
        for i in range(no):
            var oid = oids[unsafe_offset=i]
            if not (oid in s[].m):
                continue
            var hprev = s[].m.get(oid, Int64(0))
            for j in range(nh):
                if not hmask[j] and hids[unsafe_offset=j] == hprev:
                    var d = dists[unsafe_offset=i * nh + j]
                    if isfinite(d):
                        var hid = hids[unsafe_offset=j]
                        omask[i] = True
                        hmask[j] = True
                        s[].m[oid] = hid
                        s[].matches += 1
                        s[].dist_sum += d
                        var st = s[].obj.get(oid, ObjStats())
                        st.matched += 1
                        s[].obj[oid] = st.copy()
                        s[].last_match[oid] = frameid
                        s[].hyp_hist[hid] = frameid
                    break
        # 2. Minimum-cost assignment for the rest. Masked-out rows/columns
        #    and non-edges become the reference's large finite penalty.
        var any_valid = False
        var c = 0.0
        for i in range(no):
            if omask[i]:
                continue
            for j in range(nh):
                if hmask[j]:
                    continue
                var d = dists[unsafe_offset=i * nh + j]
                if isfinite(d):
                    any_valid = True
                    var ad = d if d >= 0.0 else -d
                    if ad > c:
                        c = ad
        if any_valid:
            c += 1.0
            var r = no if no < nh else nh
            var large = 2.0 * Float64(r) * c + 1.0
            var transposed = nh < no
            var nr = nh if transposed else no
            var nc = no if transposed else nh
            var cost_buf = unsafe_alloc[Float64](nr * nc)
            for a in range(nr):
                for b in range(nc):
                    var i = b if transposed else a
                    var j = a if transposed else b
                    var d = dists[unsafe_offset=i * nh + j]
                    var valid = (
                        isfinite(d) and not omask[i] and not hmask[j]
                    )
                    cost_buf[unsafe_offset=a * nc + b] = d if valid else large
            var col4row = List[Int64](length=nr, fill=-1)
            _lsap_solve(cost_buf, nr, nc, col4row)
            cost_buf.unsafe_free()
            # Original row indices ascending in both layouts.
            var row_of = List[Int64](length=no, fill=-1)
            var col_of = List[Int64](length=no, fill=-1)
            if transposed:
                for j in range(nh):
                    var i = Int(col4row[j])
                    row_of[i] = Int64(j)
                    col_of[i] = 1  # marks "assigned"
            else:
                for i in range(no):
                    row_of[i] = col4row[i]
                    col_of[i] = 1
            for i in range(no):
                if col_of[i] < 0:
                    continue
                var j = Int(row_of[i])
                var d = dists[unsafe_offset=i * nh + j]
                if not isfinite(d) or omask[i] or hmask[j]:
                    continue
                var o = oids[unsafe_offset=i]
                var h = hids[unsafe_offset=j]
                var is_switch = False
                if o in s[].m and s[].m.get(o, Int64(0)) != h:
                    var gap = _abs_i64(frameid - s[].last_occ.get(o, Int64(0)))
                    is_switch = Float64(gap) <= s[].max_switch_time
                if is_switch:
                    if not (h in s[].hyp_hist):
                        s[].ascends += 1
                    s[].switches += 1
                else:
                    s[].matches += 1
                if h in s[].res_m and s[].res_m.get(h, Int64(0)) != o:
                    if not (o in s[].last_match):
                        s[].migrates += 1
                    s[].transfers += 1
                s[].hyp_hist[h] = frameid
                s[].last_match[o] = frameid
                s[].m[o] = h
                s[].res_m[h] = o
                omask[i] = True
                hmask[j] = True
                s[].dist_sum += d
                var st = s[].obj.get(o, ObjStats())
                st.matched += 1
                s[].obj[o] = st.copy()

    # 3./4. Remaining objects are misses, remaining hypotheses false alarms.
    for i in range(no):
        if not omask[i]:
            s[].misses += 1
    for j in range(nh):
        if not hmask[j]:
            s[].fps += 1

    # 5. Fragmentation state and occurrence bookkeeping.
    for i in range(no):
        var oid = oids[unsafe_offset=i]
        var st = s[].obj.get(oid, ObjStats())
        var miss = not omask[i]
        if not st.ever_matched:
            if not miss:
                st.ever_matched = True
                st.prev_miss = False
                st.pending = False
        else:
            if miss:
                if not st.prev_miss:
                    st.pending = True
                st.prev_miss = True
            else:
                if st.pending:
                    st.frag += 1
                    st.pending = False
                st.prev_miss = False
        s[].obj[oid] = st.copy()
        s[].last_occ[oid] = frameid
    return 0


@export
def motmojo_finalize(handle: Handle, dst: F64Ptr, capacity: Int64) abi("C") -> Int32:
    """Write the accumulated counters as float64 (exact for counts < 2^53).
    Layout: 0 num_frames, 1 num_objects, 2 num_predictions, 3 num_matches,
    4 num_switches, 5 num_misses, 6 num_false_positives, 7 num_transfer,
    8 num_ascend, 9 num_migrate, 10 dist_sum, 11 num_unique_objects,
    12 mostly_tracked, 13 partially_tracked, 14 mostly_lost,
    15 num_fragmentations, 16 idtp, 17 idfp, 18 idfn, 19 ocs_sum, 20 hcs_sum.
    Returns 0, 1 on a NULL handle, 2 if capacity < COUNTS_LEN."""
    if not handle:
        return 1
    if capacity < COUNTS_LEN:
        return 2
    var s = handle.value().unsafe_bitcast[MotState]()

    var num_objects = Int64(0)
    var ocs_sum = Int64(0)
    var frag = Int64(0)
    var mt = Int64(0)
    var pt = Int64(0)
    var ml = Int64(0)
    var oid_list = List[Int64]()
    for kv in s[].obj.items():
        var st = kv.value.copy()
        num_objects += st.present
        ocs_sum += st.ocs
        frag += st.frag
        oid_list.append(kv.key.copy())
        var ratio = Float64(st.matched) / Float64(st.present)
        if ratio >= 0.8:
            mt += 1
        elif ratio >= 0.2:
            pt += 1
        else:
            ml += 1
    var num_predictions = Int64(0)
    var hcs_sum = Int64(0)
    var hyp_list = List[Int64]()
    for kv in s[].hyp.items():
        num_predictions += kv.value.present
        hcs_sum += kv.value.hcs
        hyp_list.append(kv.key.copy())

    # IDF1: max-weight matching over the co-occurrence counts. The weight of
    # the optimal matching (idtp) is unique even when the matching is not.
    var idtp = Int64(0)
    var no_u = len(oid_list)
    var nh_u = len(hyp_list)
    if no_u > 0 and nh_u > 0:
        var transposed = nh_u < no_u
        var nr = nh_u if transposed else no_u
        var nc = no_u if transposed else nh_u
        var cost_buf = unsafe_alloc[Float64](nr * nc)
        for a in range(nr):
            for b in range(nc):
                var i = b if transposed else a
                var j = a if transposed else b
                var w = s[].ex.get(
                    PairKey(oid_list[i], hyp_list[j]), Int64(0)
                )
                cost_buf[unsafe_offset=a * nc + b] = Float64(-w)
        var col4row = List[Int64](length=nr, fill=-1)
        _lsap_solve(cost_buf, nr, nc, col4row)
        cost_buf.unsafe_free()
        for a in range(nr):
            var b = Int(col4row[a])
            var i = b if transposed else a
            var j = a if transposed else b
            idtp += s[].ex.get(PairKey(oid_list[i], hyp_list[j]), Int64(0))

    dst[unsafe_offset=0] = Float64(Int64(len(s[].frames)))
    dst[unsafe_offset=1] = Float64(num_objects)
    dst[unsafe_offset=2] = Float64(num_predictions)
    dst[unsafe_offset=3] = Float64(s[].matches)
    dst[unsafe_offset=4] = Float64(s[].switches)
    dst[unsafe_offset=5] = Float64(s[].misses)
    dst[unsafe_offset=6] = Float64(s[].fps)
    dst[unsafe_offset=7] = Float64(s[].transfers)
    dst[unsafe_offset=8] = Float64(s[].ascends)
    dst[unsafe_offset=9] = Float64(s[].migrates)
    dst[unsafe_offset=10] = s[].dist_sum
    dst[unsafe_offset=11] = Float64(Int64(len(s[].obj)))
    dst[unsafe_offset=12] = Float64(mt)
    dst[unsafe_offset=13] = Float64(pt)
    dst[unsafe_offset=14] = Float64(ml)
    dst[unsafe_offset=15] = Float64(frag)
    dst[unsafe_offset=16] = Float64(idtp)
    dst[unsafe_offset=17] = Float64(hcs_sum - idtp)
    dst[unsafe_offset=18] = Float64(ocs_sum - idtp)
    dst[unsafe_offset=19] = Float64(ocs_sum)
    dst[unsafe_offset=20] = Float64(hcs_sum)
    return 0


@export
def motmojo_destroy(handle: Handle) abi("C"):
    """Release an accumulator. Reassigning a fresh empty state drops every
    dictionary's heap storage before the shell is freed."""
    if not handle:
        return
    var s = handle.value().unsafe_bitcast[MotState]()
    s[] = MotState(0.0)
    s.unsafe_free()

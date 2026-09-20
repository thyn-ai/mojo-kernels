"""Vendored pure-Python MOT metrics accumulator (fallback path).

Used when the native Mojo kernel is unavailable (unsupported platform,
missing shared library, ABI mismatch, or ``MOTMETRICS_MOJO_DISABLE_NATIVE=1``).
It is a clean-room implementation of the published metric definitions
(Bernardin & Stiefelhagen, CLEAR MOT, EURASIP JIVP 2008; Ristani et al.,
ECCVW 2016, for the ID measures) written to be observably identical to the
reference stack the differential suite checks against (py-motmetrics 1.4.0 on
the scipy assignment solver): same per-frame event semantics (carry-forward of
established tracks in object order, then a minimum-cost assignment for the
remainder; MATCH / SWITCH / TRANSFER / ASCEND / MIGRATE / MISS / FP events),
same large-penalty substitution for non-edges (2 * min(shape) *
(max|finite cost| + 1) + 1), and the same assignment algorithm with the same
scan and tie-breaking order (the shortest-augmenting-path algorithm of
Crouse 2016, pp. 1685-1686, as the reference solver implements it), so event
counts agree exactly, including on tie-heavy integer cost matrices.

IDF1 uses the fact that in the reference's global-assignment construction the
reported quantities satisfy idfp = predictions - idtp and idfn = objects -
idtp for EVERY optimal assignment; idtp (the maximum total co-occurrence
count) is unique even when the matching is not. It is therefore computed as a
max-weight matching over the sparse co-occurrence counts, which is exactly the
reference value.

All arithmetic is IEEE-754 float64 in the same operation order as the Mojo
kernel (Python floats are C doubles; only adds/subtracts/compares in the
solver), so this fallback and the native backend agree bit-for-bit.
"""

from __future__ import annotations

import math

POS_INF = float("inf")


def _lsap_solve(cost: list[float], nr: int, nc: int) -> list[int]:
    """Minimum-cost assignment of all `nr` rows to distinct columns.

    Requires nr <= nc and a fully finite flat row-major cost matrix. Shortest
    augmenting path per row (Crouse 2016, pp. 1685-1686) with the reference
    solver's exact scan order and tie rules: candidate columns start in
    reverse index order (a constant cost matrix yields the identity), a
    strict less-than relaxes path costs, and on a tie the column that would
    extend the matching (currently unassigned) is preferred. Returns
    col4row[0..nr).
    """
    u = [0.0] * nr
    v = [0.0] * nc
    row4col = [-1] * nc
    col4row = [-1] * nr
    path = [-1] * nc
    spc = [0.0] * nc
    remaining = [0] * nc
    sr = [False] * nr
    sc = [False] * nc
    for cur in range(nr):
        for it in range(nc):
            remaining[it] = nc - it - 1
        for j in range(nc):
            spc[j] = POS_INF
            path[j] = -1
            sc[j] = False
        for i in range(nr):
            sr[i] = False
        min_val = 0.0
        sink = -1
        i = cur
        num_remaining = nc
        while sink == -1:
            index = -1
            lowest = POS_INF
            sr[i] = True
            for it in range(num_remaining):
                j = remaining[it]
                r = min_val + cost[i * nc + j] - u[i] - v[j]
                if r < spc[j]:
                    path[j] = i
                    spc[j] = r
                if spc[j] < lowest or (spc[j] == lowest and row4col[j] == -1):
                    lowest = spc[j]
                    index = it
            min_val = lowest
            j = remaining[index]
            if row4col[j] == -1:
                sink = j
            else:
                i = row4col[j]
            sc[j] = True
            remaining[index] = remaining[num_remaining - 1]
            num_remaining -= 1
        u[cur] += min_val
        for r2 in range(nr):
            if sr[r2] and r2 != cur:
                u[r2] += min_val - spc[col4row[r2]]
        for j2 in range(nc):
            if sc[j2]:
                v[j2] -= min_val - spc[j2]
        j3 = sink
        while True:
            i2 = path[j3]
            row4col[j3] = i2
            col4row[i2], j3 = j3, col4row[i2]
            if i2 == cur:
                break
    return col4row


class RefAccumulator:
    """Pure-Python event-stream accumulator with the kernel's semantics.

    Object/hypothesis/frame ids are Python ints (MOTChallenge format);
    `dists` is a sequence of `n_oids` rows of `n_hids` floats where NaN and
    +/-inf signal do-not-pair constellations.
    """

    __slots__ = (
        "max_switch_time",
        "m",
        "res_m",
        "last_match",
        "last_occ",
        "hyp_hist",
        "obj",
        "hyp",
        "ex",
        "frames",
        "matches",
        "switches",
        "misses",
        "fps",
        "transfers",
        "ascends",
        "migrates",
        "dist_sum",
    )

    def __init__(self, max_switch_time: float) -> None:
        self.max_switch_time = float(max_switch_time)
        self.m: dict[int, int] = {}
        self.res_m: dict[int, int] = {}
        self.last_match: dict[int, int] = {}
        self.last_occ: dict[int, int] = {}
        self.hyp_hist: dict[int, int] = {}
        # oid -> [present, ocs, matched, frag, last_ocs_frame,
        #         ever_matched, prev_miss, pending]
        self.obj: dict[int, list] = {}
        # hid -> [present, hcs, last_frame]
        self.hyp: dict[int, list] = {}
        self.ex: dict[tuple[int, int], int] = {}
        self.frames: set[int] = set()
        self.matches = 0
        self.switches = 0
        self.misses = 0
        self.fps = 0
        self.transfers = 0
        self.ascends = 0
        self.migrates = 0
        self.dist_sum = 0.0

    def update(
        self,
        oids: list[int],
        hids: list[int],
        dists: list[list[float]],
        frameid: int,
    ) -> None:
        no = len(oids)
        nh = len(hids)
        self.frames.add(frameid)

        for i in range(no):
            st = self.obj.get(oids[i])
            if st is None:
                # present, ocs, matched, frag, last_ocs_frame,
                # ever_matched, prev_miss, pending
                st = [0, 0, 0, 0, None, False, False, False]
                self.obj[oids[i]] = st
            st[0] += 1
            if st[4] != frameid:
                st[1] += 1
            st[4] = frameid
        for j in range(nh):
            hs = self.hyp.get(hids[j])
            if hs is None:
                hs = [0, 0, None]
                self.hyp[hids[j]] = hs
            hs[0] += 1
            if hs[2] != frameid:
                hs[1] += 1
            hs[2] = frameid
        for i in range(no):
            row = dists[i]
            o = oids[i]
            for j in range(nh):
                if math.isfinite(row[j]):
                    key = (o, hids[j])
                    self.ex[key] = self.ex.get(key, 0) + 1

        omask = [False] * no
        hmask = [False] * nh

        if no > 0 and nh > 0:
            # 1. Carry forward established tracks (object array order).
            for i in range(no):
                hprev = self.m.get(oids[i])
                if hprev is None:
                    continue
                for j in range(nh):
                    if not hmask[j] and hids[j] == hprev:
                        d = dists[i][j]
                        if math.isfinite(d):
                            omask[i] = True
                            hmask[j] = True
                            self.m[oids[i]] = hids[j]
                            self.matches += 1
                            self.dist_sum += d
                            self.obj[oids[i]][2] += 1
                            self.last_match[oids[i]] = frameid
                            self.hyp_hist[hids[j]] = frameid
                        break
            # 2. Minimum-cost assignment for the rest.
            any_valid = False
            c = 0.0
            for i in range(no):
                if omask[i]:
                    continue
                for j in range(nh):
                    if hmask[j]:
                        continue
                    d = dists[i][j]
                    if math.isfinite(d):
                        any_valid = True
                        ad = d if d >= 0.0 else -d
                        if ad > c:
                            c = ad
            if any_valid:
                c += 1.0
                large = 2.0 * float(min(no, nh)) * c + 1.0
                transposed = nh < no
                nr = nh if transposed else no
                nc = no if transposed else nh
                cost = [0.0] * (nr * nc)
                for a in range(nr):
                    base = a * nc
                    for b in range(nc):
                        i, j = (b, a) if transposed else (a, b)
                        d = dists[i][j]
                        valid = math.isfinite(d) and not omask[i] and not hmask[j]
                        cost[base + b] = d if valid else large
                col4row = _lsap_solve(cost, nr, nc)
                # Pairs in ascending original row order, either layout.
                pairs = [0] * no
                assigned = [False] * no
                if transposed:
                    for j in range(nh):
                        pairs[col4row[j]] = j
                        assigned[col4row[j]] = True
                else:
                    for i in range(no):
                        pairs[i] = col4row[i]
                        assigned[i] = True
                for i in range(no):
                    if not assigned[i]:
                        continue
                    j = pairs[i]
                    d = dists[i][j]
                    if not math.isfinite(d) or omask[i] or hmask[j]:
                        continue
                    o = oids[i]
                    h = hids[j]
                    is_switch = (
                        o in self.m
                        and self.m[o] != h
                        and abs(frameid - self.last_occ.get(o, 0))
                        <= self.max_switch_time
                    )
                    if is_switch:
                        if h not in self.hyp_hist:
                            self.ascends += 1
                        self.switches += 1
                    else:
                        self.matches += 1
                    if h in self.res_m and self.res_m[h] != o:
                        if o not in self.last_match:
                            self.migrates += 1
                        self.transfers += 1
                    self.hyp_hist[h] = frameid
                    self.last_match[o] = frameid
                    self.m[o] = h
                    self.res_m[h] = o
                    omask[i] = True
                    hmask[j] = True
                    self.dist_sum += d
                    self.obj[o][2] += 1

        for i in range(no):
            if not omask[i]:
                self.misses += 1
        for j in range(nh):
            if not hmask[j]:
                self.fps += 1

        for i in range(no):
            st = self.obj[oids[i]]
            miss = not omask[i]
            if not st[5]:  # ever_matched
                if not miss:
                    st[5] = True
                    st[6] = False
                    st[7] = False
            else:
                if miss:
                    if not st[6]:
                        st[7] = True
                    st[6] = True
                else:
                    if st[7]:
                        st[3] += 1
                        st[7] = False
                    st[6] = False
            self.last_occ[oids[i]] = frameid

    def counts(self) -> dict[str, float]:
        """The accumulated counters (same layout as the native finalize)."""
        num_objects = 0
        ocs_sum = 0
        frag = 0
        mt = pt = ml = 0
        oids = list(self.obj.keys())
        for st in self.obj.values():
            num_objects += st[0]
            ocs_sum += st[1]
            frag += st[3]
            ratio = st[2] / st[0]
            if ratio >= 0.8:
                mt += 1
            elif ratio >= 0.2:
                pt += 1
            else:
                ml += 1
        num_predictions = 0
        hcs_sum = 0
        hids = list(self.hyp.keys())
        for hs in self.hyp.values():
            num_predictions += hs[0]
            hcs_sum += hs[1]

        # IDF1: max-weight matching over the co-occurrence counts; the
        # optimal weight (idtp) is unique even when the matching is not.
        idtp = 0
        no_u = len(oids)
        nh_u = len(hids)
        if no_u > 0 and nh_u > 0:
            transposed = nh_u < no_u
            nr = nh_u if transposed else no_u
            nc = no_u if transposed else nh_u
            cost = [0.0] * (nr * nc)
            for a in range(nr):
                base = a * nc
                for b in range(nc):
                    i, j = (b, a) if transposed else (a, b)
                    cost[base + b] = float(-self.ex.get((oids[i], hids[j]), 0))
            col4row = _lsap_solve(cost, nr, nc)
            for a in range(nr):
                b = col4row[a]
                i, j = (b, a) if transposed else (a, b)
                idtp += self.ex.get((oids[i], hids[j]), 0)

        return {
            "num_frames": len(self.frames),
            "num_objects": num_objects,
            "num_predictions": num_predictions,
            "num_matches": self.matches,
            "num_switches": self.switches,
            "num_misses": self.misses,
            "num_false_positives": self.fps,
            "num_transfer": self.transfers,
            "num_ascend": self.ascends,
            "num_migrate": self.migrates,
            "dist_sum": self.dist_sum,
            "num_unique_objects": len(self.obj),
            "mostly_tracked": mt,
            "partially_tracked": pt,
            "mostly_lost": ml,
            "num_fragmentations": frag,
            "idtp": idtp,
            "idfp": hcs_sum - idtp,
            "idfn": ocs_sum - idtp,
            "ocs_sum": ocs_sum,
            "hcs_sum": hcs_sum,
        }

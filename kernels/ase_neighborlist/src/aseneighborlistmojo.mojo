"""Clean-room periodic neighbor list via cell-linked lists (binning).

Written fresh from the textbook linked-cell algorithm for molecular
neighbor searches (Allen & Tildesley, "Computer Simulation of Liquids",
1987, §5.3.2; Frenkel & Smit, "Understanding Molecular Simulation",
2002, App. F). No third-party code is used or adapted; this file was
authored from the algorithm definition against the *observable* behavior
of ASE's `primitive_neighbor_list` (pair/set contract only).

Algorithm
---------
1. Fractional coordinates f_i = r_i @ inv(cell). For periodic axes the
   fractional coordinate is wrapped into [0, 1); the removed integer part
   k_i is kept so the reported shift vector S refers to the *original*
   positions (ASE convention: D = pos[j] - pos[i] + S @ cell).
2. The cell is binned along its three lattice axes. Bin widths are chosen
   so the *perpendicular* lattice-plane height per bin is >= cmax (the
   largest pair cutoff). Because |Δf_d| <= |Δr| / h_d for any displacement
   Δr (h_d = perpendicular height of axis d, i.e. 1/|column d of inv(cell)|),
   every pair within cmax sits in bins at most K_d = floor(cmax*nbins_d/h_d)+1
   apart along axis d. Searching m_d in [-K_d, +K_d] is therefore complete
   for arbitrary triclinic cells and arbitrarily skewed bin aspect ratios;
   the exact distance test filters the overestimate. For periodic axes the
   target bin index wraps (t = (c + m) mod nbins) and the quotient
   q = (c + m) div nbins is the image shift of that offset; small cells
   (cutoff > h_d) correctly yield multiple shift vectors S per atom pair.
3. Distance test: |Δr|^2 < rc^2 with rc the per-pair cutoff (strict <,
   matching the oracle; rc <= 0 pairs never match).

Emission contract (matches the observable ASE behavior):
  * bothways=1: every (i, j, S) with i != j or S != 0 is emitted once;
    (i, i, 0) is emitted iff self_interaction=1. Self-images (i, i, S != 0)
    are always emitted. This mirrors ase.neighborlist.primitive_neighbor_list.
  * bothways=0 (classic ase.neighborlist.NeighborList reduction): emit iff
    lex(S) > 0 (first nonzero component positive), or S == 0 and i < j;
    (i, i, 0) iff self_interaction=1. Verified against the NeighborList
    class with bothways=False.

Exported C ABI (batch-shaped: one call builds one whole list):

    int32_t aseneighborlistmojo_abi_version(void)
    int32_t aseneighborlistmojo_build(
        n, pos, cell, pbc,
        cutoff_mode, atom_types, n_types, cut_matrix, radii,
        self_interaction, bothways, max_nbins,
        out_i, out_j, out_S, capacity, out_count)

        pos         [3n]  Cartesian positions (may lie outside the cell)
        cell        [9]   cell vectors, row-major
        pbc         [3]   0/1 per axis
        cutoff_mode 0: rc(A,B) = cut_matrix[type[A]*n_types + type[B]]
                  1: rc(A,B) = radii[A] + radii[B]
        atom_types  [n]   type index per atom (matrix mode), in [0, n_types)
        cut_matrix  [n_types*n_types] symmetric pair cutoffs (matrix mode)
        radii       [n]   per-atom radii (radii mode)
        self_interaction / bothways   0/1 flags
        max_nbins   <=0 means unlimited; otherwise the bin grid is shrunk
                    until nbins_0*nbins_1*nbins_2 <= max_nbins (results are
                    unchanged — K_d grows to compensate)
        out_i/out_j [capacity] pair indices
        out_S       [3*capacity] shift vectors
        out_count   [1] total number of pairs found

    Returns 0 on success, 1 when capacity was too small (out_count holds
    the exact required capacity; output buffers hold the first `capacity`
    pairs), 2 on invalid geometry (singular cell on a periodic axis, or a
    bin grid above the internal safety ceiling), 3 on invalid arguments.

Determinism: bins are filled by a stable counting sort (atoms in index
order) and searched in ascending (i, m0, m1, m2, j) order, so the emitted
pair stream is byte-identical across runs and platforms. The Python
wrapper sorts (i, j, S) lexicographically regardless.
"""

from std.math import floor
from std.memory import Pointer
from std.memory.alloc import unsafe_alloc
from std.origin import MutUntrackedOrigin

comptime ABI_VERSION: Int32 = 1

comptime CUTOFF_MATRIX: Int32 = 0
comptime CUTOFF_RADII: Int32 = 1

# Hard ceiling for the bin grid when max_nbins <= 0 ("unlimited").
comptime NBINS_SAFETY_CEILING: Int = 1 << 31

comptime F64Ptr = Pointer[Float64, MutUntrackedOrigin]
comptime I32Ptr = Pointer[Int32, MutUntrackedOrigin]
comptime I64Ptr = Pointer[Int64, MutUntrackedOrigin]


@export
def aseneighborlistmojo_abi_version() abi("C") -> Int32:
    return ABI_VERSION


def floordiv(a: Int, b: Int) -> Int:
    """Floor division for possibly-negative numerators (b > 0)."""
    var q = a // b
    var r = a - q * b
    if r != 0 and ((r < 0) != (b < 0)):
        q -= 1
    return q


@export
def aseneighborlistmojo_build(
    n: Int64,
    pos: F64Ptr,
    cell: F64Ptr,
    pbc: I32Ptr,
    cutoff_mode: Int32,
    atom_types: I32Ptr,
    n_types: Int64,
    cut_matrix: F64Ptr,
    radii: F64Ptr,
    self_interaction: Int32,
    bothways: Int32,
    max_nbins: Int64,
    out_i: I64Ptr,
    out_j: I64Ptr,
    out_S: I64Ptr,
    capacity: Int64,
    out_count: I64Ptr,
) abi("C") -> Int32:
    if n < 0 or capacity < 0 or n_types <= 0:
        return 3
    if cutoff_mode != CUTOFF_MATRIX and cutoff_mode != CUTOFF_RADII:
        return 3

    var N = Int(n)
    var T = Int(n_types)
    out_count[unsafe_offset=0] = 0
    if N == 0:
        return 0

    # --- cell and its inverse -------------------------------------------
    var c00 = cell[unsafe_offset=0]
    var c01 = cell[unsafe_offset=1]
    var c02 = cell[unsafe_offset=2]
    var c10 = cell[unsafe_offset=3]
    var c11 = cell[unsafe_offset=4]
    var c12 = cell[unsafe_offset=5]
    var c20 = cell[unsafe_offset=6]
    var c21 = cell[unsafe_offset=7]
    var c22 = cell[unsafe_offset=8]

    var det = (
        c00 * (c11 * c22 - c12 * c21)
        - c01 * (c10 * c22 - c12 * c20)
        + c02 * (c10 * c21 - c11 * c20)
    )
    var any_pbc = pbc[unsafe_offset=0] != 0 or pbc[
        unsafe_offset=1
    ] != 0 or pbc[unsafe_offset=2] != 0

    # Inverse cell (rows). For a singular cell with no periodic axis the
    # search falls back to Cartesian binning (identity), which leaves the
    # Euclidean distances — and therefore the emitted (i, j, S=0) sets —
    # unchanged. A singular cell on a periodic axis is invalid geometry.
    var i00 = Float64(1.0)
    var i01 = Float64(0.0)
    var i02 = Float64(0.0)
    var i10 = Float64(0.0)
    var i11 = Float64(1.0)
    var i12 = Float64(0.0)
    var i20 = Float64(0.0)
    var i21 = Float64(0.0)
    var i22 = Float64(1.0)
    var singular = not (det > 1e-300 or det < -1e-300)
    if singular:
        if any_pbc:
            return 2
        # Cartesian mode: the metric below must be the identity as well.
        c00 = 1.0
        c01 = 0.0
        c02 = 0.0
        c10 = 0.0
        c11 = 1.0
        c12 = 0.0
        c20 = 0.0
        c21 = 0.0
        c22 = 1.0
    else:
        var inv_det = 1.0 / det
        i00 = (c11 * c22 - c12 * c21) * inv_det
        i01 = (c02 * c21 - c01 * c22) * inv_det
        i02 = (c01 * c12 - c02 * c11) * inv_det
        i10 = (c12 * c20 - c10 * c22) * inv_det
        i11 = (c00 * c22 - c02 * c20) * inv_det
        i12 = (c02 * c10 - c00 * c12) * inv_det
        i20 = (c10 * c21 - c11 * c20) * inv_det
        i21 = (c01 * c20 - c00 * c21) * inv_det
        i22 = (c00 * c11 - c01 * c10) * inv_det

    # Perpendicular lattice-plane heights h_d = 1/|column d of inv(cell)|.
    var h0 = 1.0 / (i00 * i00 + i10 * i10 + i20 * i20) ** 0.5
    var h1 = 1.0 / (i01 * i01 + i11 * i11 + i21 * i21) ** 0.5
    var h2 = 1.0 / (i02 * i02 + i12 * i12 + i22 * i22) ** 0.5

    # --- per-atom fractional coordinates, wrap integers, types ----------
    var fw = unsafe_alloc[Float64](3 * N)  # wrapped fractional coords
    var kk = unsafe_alloc[Int64](3 * N)  # wrap integers per axis
    var types = unsafe_alloc[Int](N)
    var fmin0 = Float64(0.0)
    var fmin1 = Float64(0.0)
    var fmin2 = Float64(0.0)
    var fmax0 = Float64(0.0)
    var fmax1 = Float64(0.0)
    var fmax2 = Float64(0.0)
    var cmax = Float64(0.0)
    for a in range(N):
        var x = pos[unsafe_offset=3 * a]
        var y = pos[unsafe_offset=3 * a + 1]
        var z = pos[unsafe_offset=3 * a + 2]
        var f0 = x * i00 + y * i10 + z * i20
        var f1 = x * i01 + y * i11 + z * i21
        var f2 = x * i02 + y * i12 + z * i22
        var k0 = Int64(0)
        var k1 = Int64(0)
        var k2 = Int64(0)
        if pbc[unsafe_offset=0] != 0:
            var fl = floor(f0)
            k0 = Int64(fl)
            f0 = f0 - fl
        if pbc[unsafe_offset=1] != 0:
            var fl = floor(f1)
            k1 = Int64(fl)
            f1 = f1 - fl
        if pbc[unsafe_offset=2] != 0:
            var fl = floor(f2)
            k2 = Int64(fl)
            f2 = f2 - fl
        fw[unsafe_offset=3 * a] = f0
        fw[unsafe_offset=3 * a + 1] = f1
        fw[unsafe_offset=3 * a + 2] = f2
        kk[unsafe_offset=3 * a] = k0
        kk[unsafe_offset=3 * a + 1] = k1
        kk[unsafe_offset=3 * a + 2] = k2
        if a == 0:
            fmin0 = f0
            fmax0 = f0
            fmin1 = f1
            fmax1 = f1
            fmin2 = f2
            fmax2 = f2
        else:
            fmin0 = fmin0 if fmin0 < f0 else f0
            fmax0 = fmax0 if fmax0 > f0 else f0
            fmin1 = fmin1 if fmin1 < f1 else f1
            fmax1 = fmax1 if fmax1 > f1 else f1
            fmin2 = fmin2 if fmin2 < f2 else f2
            fmax2 = fmax2 if fmax2 > f2 else f2
        var t = Int(atom_types[unsafe_offset=a])
        if t < 0 or t >= T:
            kk.unsafe_free()
            fw.unsafe_free()
            types.unsafe_free()
            return 3
        types[unsafe_offset=a] = t

    # Largest pair cutoff (bin sizing upper bound; per-pair tests are exact).
    if cutoff_mode == CUTOFF_MATRIX:
        for e in range(T * T):
            var v = cut_matrix[unsafe_offset=e]
            cmax = cmax if cmax > v else v
    else:
        for a in range(N):
            var v = radii[unsafe_offset=a] + radii[unsafe_offset=a]
            cmax = cmax if cmax > v else v
    if not (cmax > 0.0):
        # No pair cutoff can ever match (strict < with rc <= 0).
        kk.unsafe_free()
        fw.unsafe_free()
        types.unsafe_free()
        return 0

    # --- bin grid ---------------------------------------------------------
    var range0 = fmax0 - fmin0
    var range1 = fmax1 - fmin1
    var range2 = fmax2 - fmin2
    var nb0 = Int(1)
    var nb1 = Int(1)
    var nb2 = Int(1)
    if pbc[unsafe_offset=0] != 0:
        nb0 = max(1, Int(floor(h0 / cmax)))
    elif range0 > 0.0:
        nb0 = max(1, Int(floor(range0 * h0 / cmax)))
    if pbc[unsafe_offset=1] != 0:
        nb1 = max(1, Int(floor(h1 / cmax)))
    elif range1 > 0.0:
        nb1 = max(1, Int(floor(range1 * h1 / cmax)))
    if pbc[unsafe_offset=2] != 0:
        nb2 = max(1, Int(floor(h2 / cmax)))
    elif range2 > 0.0:
        nb2 = max(1, Int(floor(range2 * h2 / cmax)))

    var ceiling = Int(max_nbins)
    if ceiling <= 0:
        ceiling = NBINS_SAFETY_CEILING
    var product = nb0 * nb1 * nb2
    if product > NBINS_SAFETY_CEILING:
        kk.unsafe_free()
        fw.unsafe_free()
        types.unsafe_free()
        return 2
    # Shrink the grid until it fits the memory cap. Shrinking only ever
    # widens the per-axis search ranges K_d below; results are unchanged.
    var shrink_rounds = 0
    while product > ceiling and shrink_rounds < 64:
        var s = (Float64(ceiling) / Float64(product)) ** (1.0 / 3.0)
        nb0 = max(1, Int(floor(Float64(nb0) * s)))
        nb1 = max(1, Int(floor(Float64(nb1) * s)))
        nb2 = max(1, Int(floor(Float64(nb2) * s)))
        product = nb0 * nb1 * nb2
        shrink_rounds += 1
    if product > NBINS_SAFETY_CEILING or product <= 0:
        kk.unsafe_free()
        fw.unsafe_free()
        types.unsafe_free()
        return 2

    # Bin widths (fractional) and per-axis search ranges.
    var w0 = Float64(1.0) / Float64(nb0)
    var w1 = Float64(1.0) / Float64(nb1)
    var w2 = Float64(1.0) / Float64(nb2)
    if pbc[unsafe_offset=0] == 0:
        w0 = range0 / Float64(nb0) if range0 > 0.0 else Float64(1.0)
    if pbc[unsafe_offset=1] == 0:
        w1 = range1 / Float64(nb1) if range1 > 0.0 else Float64(1.0)
    if pbc[unsafe_offset=2] == 0:
        w2 = range2 / Float64(nb2) if range2 > 0.0 else Float64(1.0)
    var kx0 = Int(floor(cmax / (h0 * w0))) + 1
    var kx1 = Int(floor(cmax / (h1 * w1))) + 1
    var kx2 = Int(floor(cmax / (h2 * w2))) + 1

    # --- stable counting sort of atoms into bins --------------------------
    var counts = unsafe_alloc[Int64](product)
    for b in range(product):
        counts[unsafe_offset=b] = 0
    var bin_of = unsafe_alloc[Int](N)
    for a in range(N):
        var f0 = fw[unsafe_offset=3 * a]
        var f1 = fw[unsafe_offset=3 * a + 1]
        var f2 = fw[unsafe_offset=3 * a + 2]
        var b0 = Int(0)
        var b1 = Int(0)
        var b2 = Int(0)
        if pbc[unsafe_offset=0] != 0:
            b0 = Int(floor(f0 * Float64(nb0)))
        elif range0 > 0.0:
            b0 = Int(floor((f0 - fmin0) / w0))
        if pbc[unsafe_offset=1] != 0:
            b1 = Int(floor(f1 * Float64(nb1)))
        elif range1 > 0.0:
            b1 = Int(floor((f1 - fmin1) / w1))
        if pbc[unsafe_offset=2] != 0:
            b2 = Int(floor(f2 * Float64(nb2)))
        elif range2 > 0.0:
            b2 = Int(floor((f2 - fmin2) / w2))
        b0 = min(max(b0, 0), nb0 - 1)
        b1 = min(max(b1, 0), nb1 - 1)
        b2 = min(max(b2, 0), nb2 - 1)
        var bin = (b2 * nb1 + b1) * nb0 + b0
        bin_of[unsafe_offset=a] = bin
        counts[unsafe_offset=bin] += 1
    var starts = unsafe_alloc[Int64](product + 1)
    starts[unsafe_offset=0] = 0
    for b in range(product):
        starts[unsafe_offset=b + 1] = starts[unsafe_offset=b] + counts[
            unsafe_offset=b
        ]
    var members = unsafe_alloc[Int64](N)
    var cursor = unsafe_alloc[Int64](product)
    for b in range(product):
        cursor[unsafe_offset=b] = starts[unsafe_offset=b]
    for a in range(N):
        var bin = bin_of[unsafe_offset=a]
        members[unsafe_offset=Int(cursor[unsafe_offset=bin])] = Int64(a)
        cursor[unsafe_offset=bin] += 1

    # --- neighbor search --------------------------------------------------
    var total = Int64(0)
    var cap = Int(capacity)
    var selfint = self_interaction != 0
    var both = bothways != 0

    for a in range(N):
        var bin = bin_of[unsafe_offset=a]
        var c0 = bin % nb0
        var c1 = (bin // nb0) % nb1
        var c2 = bin // (nb0 * nb1)
        var fa0 = fw[unsafe_offset=3 * a]
        var fa1 = fw[unsafe_offset=3 * a + 1]
        var fa2 = fw[unsafe_offset=3 * a + 2]
        var ka0 = kk[unsafe_offset=3 * a]
        var ka1 = kk[unsafe_offset=3 * a + 1]
        var ka2 = kk[unsafe_offset=3 * a + 2]
        var ta = types[unsafe_offset=a]
        var ra = Float64(0.0)
        if cutoff_mode == CUTOFF_RADII:
            ra = radii[unsafe_offset=a]
        for m0 in range(-kx0, kx0 + 1):
            var s0 = c0 + m0
            var q0 = Int64(0)
            var t0 = s0
            if pbc[unsafe_offset=0] != 0:
                var qq0 = floordiv(s0, nb0)
                q0 = Int64(qq0)
                t0 = s0 - qq0 * nb0
            elif t0 < 0 or t0 >= nb0:
                continue
            for m1 in range(-kx1, kx1 + 1):
                var s1 = c1 + m1
                var q1 = Int64(0)
                var t1 = s1
                if pbc[unsafe_offset=1] != 0:
                    var qq1 = floordiv(s1, nb1)
                    q1 = Int64(qq1)
                    t1 = s1 - qq1 * nb1
                elif t1 < 0 or t1 >= nb1:
                    continue
                for m2 in range(-kx2, kx2 + 1):
                    var s2 = c2 + m2
                    var q2 = Int64(0)
                    var t2 = s2
                    if pbc[unsafe_offset=2] != 0:
                        var qq2 = floordiv(s2, nb2)
                        q2 = Int64(qq2)
                        t2 = s2 - qq2 * nb2
                    elif t2 < 0 or t2 >= nb2:
                        continue
                    var tbin = (t2 * nb1 + t1) * nb0 + t0
                    var bstart = Int(starts[unsafe_offset=tbin])
                    var bend = Int(starts[unsafe_offset=tbin + 1])
                    for bp in range(bstart, bend):
                        var b = Int(members[unsafe_offset=bp])
                        var df0 = fw[unsafe_offset=3 * b] + Float64(q0) - fa0
                        var df1 = fw[unsafe_offset=3 * b + 1] + Float64(q1) - fa1
                        var df2 = fw[unsafe_offset=3 * b + 2] + Float64(q2) - fa2
                        var dr0 = df0 * c00 + df1 * c10 + df2 * c20
                        var dr1 = df0 * c01 + df1 * c11 + df2 * c21
                        var dr2 = df0 * c02 + df1 * c12 + df2 * c22
                        var dd = dr0 * dr0 + dr1 * dr1 + dr2 * dr2
                        var rc = Float64(0.0)
                        if cutoff_mode == CUTOFF_MATRIX:
                            rc = cut_matrix[
                                unsafe_offset=ta * T + types[unsafe_offset=b]
                            ]
                        else:
                            rc = ra + radii[unsafe_offset=b]
                        if rc <= 0.0 or not (dd < rc * rc):
                            continue
                        var s_out0 = q0 - kk[unsafe_offset=3 * b] + ka0
                        var s_out1 = q1 - kk[unsafe_offset=3 * b + 1] + ka1
                        var s_out2 = q2 - kk[unsafe_offset=3 * b + 2] + ka2
                        var emit = True
                        if (
                            a == b
                            and s_out0 == 0
                            and s_out1 == 0
                            and s_out2 == 0
                        ):
                            emit = selfint
                        elif not both:
                            var lexpos = s_out0 > 0 or (
                                s_out0 == 0
                                and (s_out1 > 0 or (s_out1 == 0 and s_out2 > 0))
                            )
                            emit = lexpos or (
                                s_out0 == 0
                                and s_out1 == 0
                                and s_out2 == 0
                                and a < b
                            )
                        if not emit:
                            continue
                        if total < Int64(cap):
                            var w = Int(total)
                            out_i[unsafe_offset=w] = Int64(a)
                            out_j[unsafe_offset=w] = Int64(b)
                            out_S[unsafe_offset=3 * w] = s_out0
                            out_S[unsafe_offset=3 * w + 1] = s_out1
                            out_S[unsafe_offset=3 * w + 2] = s_out2
                        total += 1

    cursor.unsafe_free()
    members.unsafe_free()
    starts.unsafe_free()
    bin_of.unsafe_free()
    counts.unsafe_free()
    types.unsafe_free()
    kk.unsafe_free()
    fw.unsafe_free()

    out_count[unsafe_offset=0] = total
    if total > Int64(cap):
        return 1
    return 0

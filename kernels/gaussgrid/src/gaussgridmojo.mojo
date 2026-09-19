"""Clean-room contracted-Cartesian-Gaussian-on-grid evaluator.

Written fresh from the textbook definition of a contracted Cartesian
Gaussian basis function (Taketa, Huzinaga, O-ohata, J. Phys. Soc. Jap.
21, 2313 (1966), "THO"):

    bf(r) = N_c * (x-cx)^l (y-cy)^m (z-cz)^n *
            sum_p  w_p * exp(-alpha_p * |r-c|^2)

where w_p = coef_p * N_p folds the primitive normalization constant N_p
(THO eq. 2.2) and N_c is the contracted-function normalization. Both
normalization constants are computed by the Python wrapper (it owns the
basis-set data and the validation); this kernel receives them as flat
arrays and evaluates only the embarrassingly parallel grid loop. No
third-party Mojo code is used or adapted.

Exported C ABI (batch-shaped: one call evaluates one whole 3-D grid):

    int32_t gaussgridmojo_abi_version(void)
    int32_t gaussgridmojo_eval(
        n_bf, bf_offsets, bf_l, bf_m, bf_n, bf_cx, bf_cy, bf_cz, bf_norm,
        prim_alpha, prim_w,
        ax, ay, az, nx, ny, nz,
        n_mo, mo_coeff, mode, out)

        bf_offsets  [n_bf+1]  CSR row offsets into the primitive arrays
        bf_l/m/n    [n_bf]    Cartesian powers (l, m, n)
        bf_cx/y/z   [n_bf]    centers (same length unit as the axes)
        bf_norm     [n_bf]    contracted normalization N_c
        prim_alpha  [n_prims] Gaussian exponents
        prim_w      [n_prims] w_p = coef_p * N_p
        ax/ay/az    [nx]/[ny]/[nz]  grid-point coordinates per axis
        mo_coeff    [n_mo * n_bf]   MO coefficient rows (row-major)
        mode        0 = wavefunction: out = sum_bf mo_coeff[0,bf] * bf(r)
                    1 = density:      out = sum_mo (sum_bf coeff*bf(r))^2
        out         [nx*ny*nz] C-order, k (z) fastest — cclib's layout

The axis-separable form of the primitive exponential is exact algebra:

    exp(-a*|r-c|^2) = exp(-a*dx^2) * exp(-a*dy^2) * exp(-a*dz^2)

so the per-primitive axis factors are precomputed once per call
(O(n_prims * (nx+ny+nz)) exponentials instead of O(n_prims*nx*ny*nz)),
and the hot loop is one SIMD fused multiply-add per (point, primitive).
The grid row under update stays cache-resident while all basis functions
are accumulated into it. All arithmetic is IEEE-754 float64; basis
functions with an exactly zero MO coefficient are skipped, which is the
same result the reference adds (0.0 * value == 0.0).
"""

from std.math import exp
from std.memory import Pointer
from std.memory.alloc import unsafe_alloc
from std.origin import MutUntrackedOrigin
from std.sys import simd_width_of

comptime ABI_VERSION: Int32 = 1

comptime MODE_WAVEFUNCTION: Int32 = 0
comptime MODE_DENSITY: Int32 = 1

# Native SIMD width for float64 on the build target (2 on NEON, 4 on AVX2).
comptime WIDTH = simd_width_of[DType.float64]()

# C-side pointer spellings (untracked origin: the caller owns the lifetime
# of anything passed in; the library owns what it allocates).
comptime F64Ptr = Pointer[Float64, MutUntrackedOrigin]
comptime I32Ptr = Pointer[Int32, MutUntrackedOrigin]
comptime I64Ptr = Pointer[Int64, MutUntrackedOrigin]


@export
def gaussgridmojo_abi_version() abi("C") -> Int32:
    return ABI_VERSION


@export
def gaussgridmojo_eval(
    n_bf: Int64,
    bf_offsets: I64Ptr,
    bf_l: I32Ptr,
    bf_m: I32Ptr,
    bf_n: I32Ptr,
    bf_cx: F64Ptr,
    bf_cy: F64Ptr,
    bf_cz: F64Ptr,
    bf_norm: F64Ptr,
    prim_alpha: F64Ptr,
    prim_w: F64Ptr,
    ax: F64Ptr,
    ay: F64Ptr,
    az: F64Ptr,
    nx: Int64,
    ny: Int64,
    nz: Int64,
    n_mo: Int64,
    mo_coeff: F64Ptr,
    mode: Int32,
    out_grid: F64Ptr,
) abi("C") -> Int32:
    """Evaluate MO amplitude(s) or the summed density on the whole grid.

    Returns 0 on success, 1 on invalid sizes, 2 on an invalid mode.
    All input buffers must hold the documented number of float64/int32/
    int64 slots; `out` must hold nx*ny*nz float64 slots and is fully
    overwritten.
    """
    if n_bf <= 0 or nx <= 0 or ny <= 0 or nz <= 0 or n_mo <= 0:
        return 1
    if mode == MODE_WAVEFUNCTION and n_mo != 1:
        return 2
    if mode != MODE_WAVEFUNCTION and mode != MODE_DENSITY:
        return 2

    var NBF = Int(n_bf)
    var NX = Int(nx)
    var NY = Int(ny)
    var NZ = Int(nz)
    var NMO = Int(n_mo)
    var npts = NX * NY * NZ

    var total_prims = Int(bf_offsets[unsafe_offset=NBF])
    if total_prims < 0:
        return 1

    # One arena for every precomputed factor:
    #   DXL/DYM/DZN [NBF x N?]  per-axis Cartesian powers (x-cx)^l etc.
    #   EX/EY       [P x N?]    per-primitive exp(-a*dx^2), exp(-a*dy^2)
    #   GEZ         [P x NZ]    per-primitive exp(-a*dz^2) * (z-cz)^n
    #   PSI         [NZ]        cache-resident grid-row accumulator
    var n_axis = NX + NY + NZ
    var arena_len = NBF * n_axis + total_prims * n_axis + NZ
    if total_prims == 0 or NBF == 0:
        # No primitives contribute: the result is identically zero.
        for p in range(npts):
            out_grid[unsafe_offset=p] = 0.0
        return 0
    var arena = unsafe_alloc[Float64](arena_len)
    var DXL = arena
    var DYM = DXL.unsafe_offset(NBF * NX)
    var DZN = DYM.unsafe_offset(NBF * NY)
    var EX = DZN.unsafe_offset(NBF * NZ)
    var EY = EX.unsafe_offset(total_prims * NX)
    var GEZ = EY.unsafe_offset(total_prims * NY)
    var PSI = GEZ.unsafe_offset(total_prims * NZ)
    # Per-MO list of basis functions with a non-zero coefficient.
    var active = unsafe_alloc[Int64](NBF)

    # Per-basis-function axis powers: DXL[b,i] = (ax[i]-cx)^l, etc.
    for b in range(NBF):
        var cx = bf_cx[unsafe_offset=b]
        var cy = bf_cy[unsafe_offset=b]
        var cz = bf_cz[unsafe_offset=b]
        var l = Int(bf_l[unsafe_offset=b])
        var m = Int(bf_m[unsafe_offset=b])
        var n = Int(bf_n[unsafe_offset=b])
        for i in range(NX):
            var d = ax[unsafe_offset=i] - cx
            var p = Float64(1.0)
            for _ in range(l):
                p *= d
            DXL[unsafe_offset=b * NX + i] = p
        for j in range(NY):
            var d = ay[unsafe_offset=j] - cy
            var p = Float64(1.0)
            for _ in range(m):
                p *= d
            DYM[unsafe_offset=b * NY + j] = p
        for k in range(NZ):
            var d = az[unsafe_offset=k] - cz
            var p = Float64(1.0)
            for _ in range(n):
                p *= d
            DZN[unsafe_offset=b * NZ + k] = p

    # Per-primitive Gaussian axis factors; the z factor folds in the
    # Cartesian power of its basis function (GEZ), so the hot loop is a
    # single fused multiply-add per (point, primitive).
    for b in range(NBF):
        var cx = bf_cx[unsafe_offset=b]
        var cy = bf_cy[unsafe_offset=b]
        var cz = bf_cz[unsafe_offset=b]
        var o0 = Int(bf_offsets[unsafe_offset=b])
        var o1 = Int(bf_offsets[unsafe_offset=b + 1])
        for p in range(o0, o1):
            var a = prim_alpha[unsafe_offset=p]
            for i in range(NX):
                var d = ax[unsafe_offset=i] - cx
                EX[unsafe_offset=p * NX + i] = exp(-a * d * d)
            for j in range(NY):
                var d = ay[unsafe_offset=j] - cy
                EY[unsafe_offset=p * NY + j] = exp(-a * d * d)
            for k in range(NZ):
                var d = az[unsafe_offset=k] - cz
                GEZ[unsafe_offset=p * NZ + k] = exp(-a * d * d) * DZN[
                    unsafe_offset=b * NZ + k
                ]

    # Main loop: one MO at a time; for each (x, y) grid line accumulate
    # every active basis function into the cache-resident z row, then
    # store the wavefunction row or accumulate its square into `out`.
    if mode == MODE_DENSITY:
        for p in range(npts):
            out_grid[unsafe_offset=p] = 0.0

    for mo in range(NMO):
        var crow = mo_coeff.unsafe_offset(mo * NBF)
        var n_active = 0
        for b in range(NBF):
            if crow[unsafe_offset=b] != 0.0:
                active[unsafe_offset=n_active] = Int64(b)
                n_active += 1

        for i in range(NX):
            for j in range(NY):
                var base = (i * NY + j) * NZ
                var row = out_grid.unsafe_offset(base)
                if mode == MODE_DENSITY:
                    row = PSI
                # Zero the z row.
                var k0 = 0
                while k0 + WIDTH <= NZ:
                    row.unsafe_store(k0, SIMD[DType.float64, WIDTH](0.0))
                    k0 += WIDTH
                while k0 < NZ:
                    row[unsafe_offset=k0] = 0.0
                    k0 += 1

                for ai in range(n_active):
                    var b = Int(active[unsafe_offset=ai])
                    var pre = (
                        crow[unsafe_offset=b]
                        * bf_norm[unsafe_offset=b]
                        * DXL[unsafe_offset=b * NX + i]
                        * DYM[unsafe_offset=b * NY + j]
                    )
                    var o0 = Int(bf_offsets[unsafe_offset=b])
                    var o1 = Int(bf_offsets[unsafe_offset=b + 1])
                    for p in range(o0, o1):
                        var s = (
                            pre
                            * prim_w[unsafe_offset=p]
                            * EX[unsafe_offset=p * NX + i]
                            * EY[unsafe_offset=p * NY + j]
                        )
                        var sv = SIMD[DType.float64, WIDTH](s)
                        var gez = GEZ.unsafe_offset(p * NZ)
                        var k = 0
                        while k + WIDTH <= NZ:
                            row.unsafe_store(
                                k,
                                row.unsafe_load[width=WIDTH](k)
                                + sv * gez.unsafe_load[width=WIDTH](k),
                            )
                            k += WIDTH
                        while k < NZ:
                            row[unsafe_offset=k] += s * GEZ[
                                unsafe_offset=p * NZ + k
                            ]
                            k += 1

                if mode == MODE_DENSITY:
                    var out_row = out_grid.unsafe_offset(base)
                    var k = 0
                    while k + WIDTH <= NZ:
                        var v = PSI.unsafe_load[width=WIDTH](k)
                        out_row.unsafe_store(
                            k, out_row.unsafe_load[width=WIDTH](k) + v * v
                        )
                        k += WIDTH
                    while k < NZ:
                        var v = PSI[unsafe_offset=k]
                        out_row[unsafe_offset=k] += v * v
                        k += 1

    active.unsafe_free()
    arena.unsafe_free()
    return 0

"""Clean-room Konno-Ohmachi spectral smoothing kernel.

Written fresh from the published algorithm (Konno & Ohmachi, 1998, BSSA
88(1):228-241): the smoothing window around center frequency fc is

    W(f, fc) = (sin(x)/x)^4   with   x = b * log10(f / fc)

with the limiting value W = 1 at f == fc, and the smoothed spectrum is the
windowed weighted mean (normalized) or plain weighted sum (unnormalized) of
the samples. No third-party Mojo code is used or adapted; the implementation
was validated black-box against the ObsPy package (the differential oracle)
without reading its sources. Observed edge semantics mirrored here:

  * samples at exactly f == 0 carry zero window weight;
  * centers at exactly f == 0 pass the input value through unchanged;
  * `count` is the number of filter applications, clamped to at least 1.

Exported C ABI (batch-shaped: whole buffers in, whole buffers out):

    int32_t konnomojo_abi_version(void)
    int32_t konnomojo_smooth_loop_f64(spectra, freqs, n_spectra, n_freqs,
                                      bandwidth, count, normalize, out)
    int32_t konnomojo_smooth_loop_f32(spectra, freqs, n_spectra, n_freqs,
                                      bandwidth, count, normalize, out)
    int32_t konnomojo_window_matrix_f64(freqs, n_freqs, bandwidth, out_w)
    int32_t konnomojo_window_matrix_f32(freqs, n_freqs, bandwidth, out_w)

`smooth_loop` is the fused window+reduction path: for every center it
evaluates the window and both reductions in one pass over the samples,
without materializing the O(n^2) window matrix. `window_matrix` fills the
raw (unnormalized) n x n window matrix W[i, j] = W(f_i, f_j) for callers
that prefer a matrix-multiplication reduction; row normalization, matrix
powers and the matmul itself stay with the caller (BLAS is already native).

All arithmetic is IEEE-754 in the input dtype. The window argument is
evaluated as x = b * (log10(f_i) - log10(f_j)) with log10(f) precomputed
once per call; this differs from the literal b*log10(f_i/f_j) by at most a
few ulps and removes the per-element log10 from the O(n^2) inner loop (the
differential suite documents the measured agreement with the oracle).
Accumulation is one SIMD lane per sample with a horizontal reduce, again a
documented-tolerance (not bit-exact) match to NumPy's pairwise summation.
"""

from std.math import log10, sin
from std.memory import Pointer
from std.memory.alloc import unsafe_alloc
from std.origin import MutUntrackedOrigin
from std.sys import simd_width_of

comptime ABI_VERSION: Int32 = 1

# Native SIMD widths on the build target (2 on NEON, 4 on AVX2 for f64).
comptime WIDTH64 = simd_width_of[DType.float64]()
comptime WIDTH32 = simd_width_of[DType.float32]()

# C-side pointer spellings (untracked origin: the caller owns the lifetime of
# anything passed in; the library owns what it allocates).
comptime F64Ptr = Pointer[Float64, MutUntrackedOrigin]
comptime F32Ptr = Pointer[Float32, MutUntrackedOrigin]


@export
def konnomojo_abi_version() abi("C") -> Int32:
    return ABI_VERSION


def _smooth_loop_f64(
    spectra: F64Ptr,
    freqs: F64Ptr,
    n_spectra: Int,
    n_freqs: Int,
    bandwidth: Float64,
    count: Int,
    normalize: Bool,
    smoothed: F64Ptr,
):
    """Fused window+reduction, float64, one spectrum per row of `spectra`."""
    var logf = unsafe_alloc[Float64](n_freqs)
    var cur = unsafe_alloc[Float64](n_freqs)
    var wbuf = unsafe_alloc[Float64](n_freqs)
    for i in range(n_freqs):
        logf[unsafe_offset=i] = log10(freqs[unsafe_offset=i])

    var bv = SIMD[DType.float64, WIDTH64](bandwidth)
    var one = SIMD[DType.float64, WIDTH64](1.0)
    var zero = SIMD[DType.float64, WIDTH64](0.0)

    for s in range(n_spectra):
        var row_in = spectra.unsafe_offset(s * n_freqs)
        var row_out = smoothed.unsafe_offset(s * n_freqs)
        for i in range(n_freqs):
            cur[unsafe_offset=i] = row_in[unsafe_offset=i]
        for _app in range(count):
            for j in range(n_freqs):
                var fc = freqs[unsafe_offset=j]
                if fc == 0.0:
                    # Zero-frequency centers pass the input through unchanged.
                    row_out[unsafe_offset=j] = cur[unsafe_offset=j]
                    continue
                var xj = SIMD[DType.float64, WIDTH64](
                    bandwidth * logf[unsafe_offset=j]
                )
                var fcv = SIMD[DType.float64, WIDTH64](fc)
                var acc_w = SIMD[DType.float64, WIDTH64](0.0)
                var acc_num = SIMD[DType.float64, WIDTH64](0.0)
                var i = 0
                while i + WIDTH64 <= n_freqs:
                    var f = freqs.unsafe_load[width=WIDTH64](i)
                    var x = bv * logf.unsafe_load[width=WIDTH64](i) - xj
                    var t = sin(x) / x
                    var w = t * t
                    w = w * w
                    # Limiting value 1 wherever f == fc (also fixes the 0/0).
                    w = f.eq(fcv).select(one, w)
                    # Samples at exactly f == 0 carry no weight.
                    w = f.eq(zero).select(zero, w)
                    if normalize:
                        wbuf.unsafe_store(i, w)
                        acc_w += w
                    else:
                        acc_num += w * cur.unsafe_load[width=WIDTH64](i)
                    i += WIDTH64
                var sum_w = acc_w.reduce_add()
                var num = acc_num.reduce_add()
                while i < n_freqs:
                    var f = freqs[unsafe_offset=i]
                    var x = bandwidth * logf[unsafe_offset=i] - xj[0]
                    var t = sin(x) / x
                    var w = t * t
                    w = w * w
                    if f == fc:
                        w = 1.0
                    if f == 0.0:
                        w = 0.0
                    if normalize:
                        wbuf[unsafe_offset=i] = w
                        sum_w += w
                    else:
                        num += w * cur[unsafe_offset=i]
                    i += 1
                if normalize:
                    # Oracle op order: normalize the window first, then reduce.
                    var inv = SIMD[DType.float64, WIDTH64](sum_w)
                    var acc2 = SIMD[DType.float64, WIDTH64](0.0)
                    var k = 0
                    while k + WIDTH64 <= n_freqs:
                        var wn = wbuf.unsafe_load[width=WIDTH64](k) / inv
                        acc2 += wn * cur.unsafe_load[width=WIDTH64](k)
                        k += WIDTH64
                    var num_n = acc2.reduce_add()
                    while k < n_freqs:
                        num_n += (wbuf[unsafe_offset=k] / sum_w) * cur[
                            unsafe_offset=k
                        ]
                        k += 1
                    row_out[unsafe_offset=j] = num_n
                else:
                    row_out[unsafe_offset=j] = num
            if _app + 1 < count:
                for i in range(n_freqs):
                    cur[unsafe_offset=i] = row_out[unsafe_offset=i]

    logf.unsafe_free()
    cur.unsafe_free()
    wbuf.unsafe_free()


def _smooth_loop_f32(
    spectra: F32Ptr,
    freqs: F32Ptr,
    n_spectra: Int,
    n_freqs: Int,
    bandwidth: Float32,
    count: Int,
    normalize: Bool,
    smoothed: F32Ptr,
):
    """Fused window+reduction, float32 (all arithmetic in float32)."""
    var logf = unsafe_alloc[Float32](n_freqs)
    var cur = unsafe_alloc[Float32](n_freqs)
    var wbuf = unsafe_alloc[Float32](n_freqs)
    for i in range(n_freqs):
        logf[unsafe_offset=i] = log10(freqs[unsafe_offset=i])

    var bv = SIMD[DType.float32, WIDTH32](bandwidth)
    var one = SIMD[DType.float32, WIDTH32](1.0)
    var zero = SIMD[DType.float32, WIDTH32](0.0)

    for s in range(n_spectra):
        var row_in = spectra.unsafe_offset(s * n_freqs)
        var row_out = smoothed.unsafe_offset(s * n_freqs)
        for i in range(n_freqs):
            cur[unsafe_offset=i] = row_in[unsafe_offset=i]
        for _app in range(count):
            for j in range(n_freqs):
                var fc = freqs[unsafe_offset=j]
                if fc == 0.0:
                    row_out[unsafe_offset=j] = cur[unsafe_offset=j]
                    continue
                var xj = SIMD[DType.float32, WIDTH32](
                    bandwidth * logf[unsafe_offset=j]
                )
                var fcv = SIMD[DType.float32, WIDTH32](fc)
                var acc_w = SIMD[DType.float32, WIDTH32](0.0)
                var acc_num = SIMD[DType.float32, WIDTH32](0.0)
                var i = 0
                while i + WIDTH32 <= n_freqs:
                    var f = freqs.unsafe_load[width=WIDTH32](i)
                    var x = bv * logf.unsafe_load[width=WIDTH32](i) - xj
                    var t = sin(x) / x
                    var w = t * t
                    w = w * w
                    w = f.eq(fcv).select(one, w)
                    w = f.eq(zero).select(zero, w)
                    if normalize:
                        wbuf.unsafe_store(i, w)
                        acc_w += w
                    else:
                        acc_num += w * cur.unsafe_load[width=WIDTH32](i)
                    i += WIDTH32
                var sum_w = acc_w.reduce_add()
                var num = acc_num.reduce_add()
                while i < n_freqs:
                    var f = freqs[unsafe_offset=i]
                    var x = bandwidth * logf[unsafe_offset=i] - xj[0]
                    var t = sin(x) / x
                    var w = t * t
                    w = w * w
                    if f == fc:
                        w = 1.0
                    if f == 0.0:
                        w = 0.0
                    if normalize:
                        wbuf[unsafe_offset=i] = w
                        sum_w += w
                    else:
                        num += w * cur[unsafe_offset=i]
                    i += 1
                if normalize:
                    var inv = SIMD[DType.float32, WIDTH32](sum_w)
                    var acc2 = SIMD[DType.float32, WIDTH32](0.0)
                    var k = 0
                    while k + WIDTH32 <= n_freqs:
                        var wn = wbuf.unsafe_load[width=WIDTH32](k) / inv
                        acc2 += wn * cur.unsafe_load[width=WIDTH32](k)
                        k += WIDTH32
                    var num_n = acc2.reduce_add()
                    while k < n_freqs:
                        num_n += (wbuf[unsafe_offset=k] / sum_w) * cur[
                            unsafe_offset=k
                        ]
                        k += 1
                    row_out[unsafe_offset=j] = num_n
                else:
                    row_out[unsafe_offset=j] = num
            if _app + 1 < count:
                for i in range(n_freqs):
                    cur[unsafe_offset=i] = row_out[unsafe_offset=i]

    logf.unsafe_free()
    cur.unsafe_free()
    wbuf.unsafe_free()


@export
def konnomojo_smooth_loop_f64(
    spectra: F64Ptr,
    freqs: F64Ptr,
    n_spectra: Int64,
    n_freqs: Int64,
    bandwidth: Float64,
    count: Int64,
    normalize: Int32,
    smoothed: F64Ptr,
) abi("C") -> Int32:
    """Loop-path smoothing for `n_spectra` rows of length `n_freqs`.

    `out` must hold n_spectra * n_freqs float64 slots. Returns 0 on success,
    1 on a non-positive size. `count` below 1 is clamped to one application.
    """
    if n_spectra < 0 or n_freqs < 0:
        return 1
    if n_spectra == 0 or n_freqs == 0:
        return 0
    var apps = Int(count)
    if apps < 1:
        apps = 1
    _smooth_loop_f64(
        spectra,
        freqs,
        Int(n_spectra),
        Int(n_freqs),
        bandwidth,
        apps,
        normalize != 0,
        smoothed,
    )
    return 0


@export
def konnomojo_smooth_loop_f32(
    spectra: F32Ptr,
    freqs: F32Ptr,
    n_spectra: Int64,
    n_freqs: Int64,
    bandwidth: Float32,
    count: Int64,
    normalize: Int32,
    smoothed: F32Ptr,
) abi("C") -> Int32:
    """Float32 twin of `konnomojo_smooth_loop_f64`."""
    if n_spectra < 0 or n_freqs < 0:
        return 1
    if n_spectra == 0 or n_freqs == 0:
        return 0
    var apps = Int(count)
    if apps < 1:
        apps = 1
    _smooth_loop_f32(
        spectra,
        freqs,
        Int(n_spectra),
        Int(n_freqs),
        bandwidth,
        apps,
        normalize != 0,
        smoothed,
    )
    return 0


@export
def konnomojo_window_matrix_f64(
    freqs: F64Ptr, n_freqs: Int64, bandwidth: Float64, out_w: F64Ptr
) abi("C") -> Int32:
    """Raw window matrix W[i, j] = W(f_i, f_j), row-major n x n (float64).

    Applies the f == fc limiting value and the zero-sample mask, but no row
    normalization (the caller owns that, plus matrix powers and matmul).
    """
    if n_freqs < 0:
        return 1
    if n_freqs == 0:
        return 0
    var n = Int(n_freqs)
    var logf = unsafe_alloc[Float64](n)
    for i in range(n):
        logf[unsafe_offset=i] = log10(freqs[unsafe_offset=i])
    var bv = SIMD[DType.float64, WIDTH64](bandwidth)
    var one = SIMD[DType.float64, WIDTH64](1.0)
    for i in range(n):
        var fi = freqs[unsafe_offset=i]
        var row = out_w.unsafe_offset(i * n)
        if fi == 0.0:
            for j in range(n):
                row[unsafe_offset=j] = 0.0
            continue
        var xi = SIMD[DType.float64, WIDTH64](bandwidth * logf[unsafe_offset=i])
        var fiv = SIMD[DType.float64, WIDTH64](fi)
        var j = 0
        while j + WIDTH64 <= n:
            var f = freqs.unsafe_load[width=WIDTH64](j)
            var x = xi - bv * logf.unsafe_load[width=WIDTH64](j)
            var t = sin(x) / x
            var w = t * t
            w = w * w
            w = f.eq(fiv).select(one, w)
            row.unsafe_store(j, w)
            j += WIDTH64
        while j < n:
            var f = freqs[unsafe_offset=j]
            var x = bandwidth * (logf[unsafe_offset=i] - logf[unsafe_offset=j])
            var t = sin(x) / x
            var w = t * t
            w = w * w
            if f == fi:
                w = 1.0
            row[unsafe_offset=j] = w
            j += 1
    logf.unsafe_free()
    return 0


@export
def konnomojo_window_matrix_f32(
    freqs: F32Ptr, n_freqs: Int64, bandwidth: Float32, out_w: F32Ptr
) abi("C") -> Int32:
    """Float32 twin of `konnomojo_window_matrix_f64`."""
    if n_freqs < 0:
        return 1
    if n_freqs == 0:
        return 0
    var n = Int(n_freqs)
    var logf = unsafe_alloc[Float32](n)
    for i in range(n):
        logf[unsafe_offset=i] = log10(freqs[unsafe_offset=i])
    var bv = SIMD[DType.float32, WIDTH32](bandwidth)
    var one = SIMD[DType.float32, WIDTH32](1.0)
    for i in range(n):
        var fi = freqs[unsafe_offset=i]
        var row = out_w.unsafe_offset(i * n)
        if fi == 0.0:
            for j in range(n):
                row[unsafe_offset=j] = 0.0
            continue
        var xi = SIMD[DType.float32, WIDTH32](bandwidth * logf[unsafe_offset=i])
        var fiv = SIMD[DType.float32, WIDTH32](fi)
        var j = 0
        while j + WIDTH32 <= n:
            var f = freqs.unsafe_load[width=WIDTH32](j)
            var x = xi - bv * logf.unsafe_load[width=WIDTH32](j)
            var t = sin(x) / x
            var w = t * t
            w = w * w
            w = f.eq(fiv).select(one, w)
            row.unsafe_store(j, w)
            j += WIDTH32
        while j < n:
            var f = freqs[unsafe_offset=j]
            var x = bandwidth * (logf[unsafe_offset=i] - logf[unsafe_offset=j])
            var t = sin(x) / x
            var w = t * t
            w = w * w
            if f == fi:
                w = 1.0
            row[unsafe_offset=j] = w
            j += 1
    logf.unsafe_free()
    return 0

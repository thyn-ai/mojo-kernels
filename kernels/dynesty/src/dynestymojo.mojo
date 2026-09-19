"""Clean-room nested-sampling bounding/proposal kernel.

Written fresh from the published nested-sampling literature:

  - Skilling (2006), "Nested sampling for general Bayesian computation"
    (constrained-prior sampling with a shrinking prior volume),
  - Mukherjee, Parkinson & Liddle (2006), "A nested sampling algorithm for
    cosmological model selection" (ellipsoidal rejection sampling),
  - Feroz, Hobson & Bridges (2009, MULTINEST) (covariance bounding scaled
    to contain every live point, times an enlargement factor).

No third-party code is used or adapted. This kernel implements exactly two
pieces of machinery — the likelihood callback, prior transform, and all
evidence bookkeeping live in the Python wrapper:

  1. Ellipsoid fit (`dynestymojo_fit_ellipsoid`): from the n live points
     in the unit hypercube, compute the sample mean and covariance
     (ddof = 1), scale the covariance so the ellipsoid contains every live
     point (maximum Mahalanobis distance), multiply by the caller's
     enlargement factor, and return the Cholesky factor of the final
     bounding matrix together with the ellipsoid's log-volume.

  2. Batch proposal (`dynestymojo_propose_batch`): draw up to `max_out`
     candidate unit-cube points, uniformly from the ellipsoid intersected
     with the unit hypercube (mode 0) or uniformly from the whole cube
     (mode 1). Uniform-in-ellipsoid draws use the textbook construction
     x = mean + r * L @ g/|g| with g Gaussian and r = u^(1/ndim); draws
     landing outside the cube are rejected. Candidates within one batch
     are iid, so the wrapper's "first candidate above the likelihood
     threshold" is an exact draw from the constrained prior — batching
     changes interpreter overhead, not the sampling distribution.

Deterministic RNG: xoshiro256** seeded via splitmix64 (both public-domain
algorithms), Box-Muller for the Gaussian direction draws, and a Lanczos
lgamma (Numerical Recipes coefficients) for the unit-ball volume. One
seeded state per `dynestymojo_rng_create` handle: same seed -> same stream
-> bit-reproducible runs. All arithmetic is IEEE-754 float64.

Exported C ABI (v1):

    int32_t  dynestymojo_abi_version(void)
    void*    dynestymojo_rng_create(uint64_t seed)
    void     dynestymojo_rng_destroy(void* rng)
    int32_t  dynestymojo_fit_ellipsoid(int64_t ndim, int64_t n,
                                       const double* U, double enlarge,
                                       double* out_mean, double* out_chol,
                                       double* out_logvol)
    int64_t  dynestymojo_propose_batch(void* rng, int64_t ndim,
                                       const double* mean,
                                       const double* chol,
                                       int32_t mode, int64_t max_out,
                                       double* out)

    fit_ellipsoid status: 0 = ok, 1 = invalid arguments,
                          2 = covariance numerically singular (caller
                              should fall back to whole-cube sampling).
    propose_batch returns the number of candidates written (0..max_out);
    0 means the ellipsoid barely intersects the cube and the caller
    should retry in cube mode.
"""

from std.math import cos, exp, log, sqrt
from std.memory import Pointer
from std.memory.alloc import unsafe_alloc
from std.origin import MutUntrackedOrigin

comptime ABI_VERSION: Int32 = 1

comptime MODE_ELLIPSOID: Int32 = 0
comptime MODE_CUBE: Int32 = 1

# C-side pointer spellings (untracked origin: the caller owns the lifetime
# of anything passed in; the library owns what it allocates).
comptime F64Ptr = Pointer[Float64, MutUntrackedOrigin]
comptime U64Ptr = Pointer[UInt64, MutUntrackedOrigin]
comptime Handle = Optional[Pointer[UInt8, MutUntrackedOrigin]]

comptime PI: Float64 = 3.14159265358979323846264338327950288
comptime TWO_PI: Float64 = 6.28318530717958647692528676655900576

# Cap on internal rejection attempts per propose_batch call (mode 0):
# generous for sane bounds, bounded so a degenerate ellipsoid cannot hang.
comptime MAX_ATTEMPTS_FACTOR: Int64 = 4096


# --- deterministic RNG: splitmix64 seeding into a xoshiro256** state ---


def _splitmix64(state: U64Ptr) -> UInt64:
    state[] = state[] + 0x9E3779B97F4A7C15
    var z = state[]
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9
    z = (z ^ (z >> 27)) * 0x94D049BB133111EB
    return z ^ (z >> 31)


def _rotl(x: UInt64, k: UInt64) -> UInt64:
    return (x << k) | (x >> (64 - k))


def _xoshiro_next(s: U64Ptr) -> UInt64:
    var result = _rotl(s[unsafe_offset=1] * 5, 7) * 9
    var t = s[unsafe_offset=1] << 17
    s[unsafe_offset=2] = s[unsafe_offset=2] ^ s[unsafe_offset=0]
    s[unsafe_offset=3] = s[unsafe_offset=3] ^ s[unsafe_offset=1]
    s[unsafe_offset=1] = s[unsafe_offset=1] ^ s[unsafe_offset=2]
    s[unsafe_offset=0] = s[unsafe_offset=0] ^ s[unsafe_offset=3]
    s[unsafe_offset=2] = s[unsafe_offset=2] ^ t
    s[unsafe_offset=3] = _rotl(s[unsafe_offset=3], 45)
    return result


def _next_f64(s: U64Ptr) -> Float64:
    """Uniform on [0, 1): top 53 bits scaled by 2^-53."""
    return Float64(_xoshiro_next(s) >> 11) * (1.0 / 9007199254740992.0)


def _next_gaussian(s: U64Ptr) -> Float64:
    """One standard normal draw (Box-Muller; the partner draw is discarded
    — the stream stays trivially reproducible and the cost is noise)."""
    var u1 = _next_f64(s)
    while u1 <= 0.0:
        u1 = _next_f64(s)
    var u2 = _next_f64(s)
    return sqrt(-2.0 * log(u1)) * cos(TWO_PI * u2)


def _lgamma(z: Float64) -> Float64:
    """Log-gamma, Lanczos approximation (Numerical Recipes gammln form).
    Only called with z = ndim/2 + 1 >= 1.5, well inside its accurate range
    (~2e-10 relative). Coefficients inlined: no comptime arrays in Mojo 1.1."""
    var ser = 1.000000000190015
    ser += 76.18009172947146 / (z + 1.0)
    ser += -86.50532032941677 / (z + 2.0)
    ser += 24.01409824083091 / (z + 3.0)
    ser += -1.231739572450155 / (z + 4.0)
    ser += 0.1208650973866179e-2 / (z + 5.0)
    ser += -0.5395239384953e-5 / (z + 6.0)
    var tmp = z + 5.5
    tmp -= (z + 0.5) * log(tmp)
    return -tmp + log(2.5066282746310005 * ser / z)


def _log_unit_ball_volume(ndim: Int) -> Float64:
    """log of the ndim-dimensional unit-ball volume
    pi^(ndim/2) / Gamma(ndim/2 + 1)."""
    var half = Float64(ndim) * 0.5
    return half * log(PI) - _lgamma(half + 1.0)


@export
def dynestymojo_abi_version() abi("C") -> Int32:
    return ABI_VERSION


@export
def dynestymojo_rng_create(seed: UInt64) abi("C") -> Handle:
    """Allocate a seeded xoshiro256** state (4 x uint64, splitmix64-seeded)."""
    var mix = unsafe_alloc[UInt64](1)
    mix[] = seed
    var s = unsafe_alloc[UInt64](4)
    for i in range(4):
        s[unsafe_offset=i] = _splitmix64(mix)
    if (
        s[unsafe_offset=0] == 0
        and s[unsafe_offset=1] == 0
        and s[unsafe_offset=2] == 0
        and s[unsafe_offset=3] == 0
    ):
        # The all-zero state is the one forbidden xoshiro256** state.
        s[unsafe_offset=0] = 0x9E3779B97F4A7C15
    mix.unsafe_free()
    return s.unsafe_bitcast[UInt8]()


@export
def dynestymojo_rng_destroy(handle: Handle) abi("C"):
    if handle:
        var s = handle.value().unsafe_bitcast[UInt64]()
        s.unsafe_free()


@export
def dynestymojo_fit_ellipsoid(
    ndim: Int64,
    n: Int64,
    u_points: F64Ptr,
    enlarge: Float64,
    out_mean: F64Ptr,
    out_chol: F64Ptr,
    out_logvol: F64Ptr,
) abi("C") -> Int32:
    """Fit the enlarged, all-containing bounding ellipsoid of the live points.

    u_points is n*ndim row-major (one live point per row). out_chol is
    ndim*ndim row-major and holds the lower-triangular Cholesky factor L
    of the bounding matrix A = enlarge * max_mahalanobis^2 * cov (zeros
    above the diagonal), so the ellipsoid is {x : (x-mean)^T A^-1 (x-mean)
    <= 1}. Returns 0 on success, 1 on invalid arguments, 2 when the
    covariance is numerically singular even after diagonal jitter.
    """
    if ndim <= 0 or n <= 1 or enlarge < 1.0:
        return 1
    var d = Int(ndim)
    var nn = Int(n)

    # Workspace: cov (d*d) + chol (d*d) + z (d).
    var arena = unsafe_alloc[Float64](2 * d * d + d)
    var cov = arena
    var chol = arena.unsafe_offset(d * d)
    var z = arena.unsafe_offset(2 * d * d)

    # Sample mean.
    for j in range(d):
        out_mean[unsafe_offset=j] = 0.0
    for i in range(nn):
        var row = u_points.unsafe_offset(i * d)
        for j in range(d):
            out_mean[unsafe_offset=j] += row[unsafe_offset=j]
    for j in range(d):
        out_mean[unsafe_offset=j] /= Float64(nn)

    # Centered covariance with ddof = 1 (upper triangle, mirrored below).
    for j in range(d):
        for k in range(j, d):
            var acc = 0.0
            for i in range(nn):
                var row = u_points.unsafe_offset(i * d)
                acc += (row[unsafe_offset=j] - out_mean[unsafe_offset=j]) * (
                    row[unsafe_offset=k] - out_mean[unsafe_offset=k]
                )
            cov[unsafe_offset=j * d + k] = acc / Float64(nn - 1)
            cov[unsafe_offset=k * d + j] = acc / Float64(nn - 1)

    var trace = 0.0
    for j in range(d):
        trace += cov[unsafe_offset=j * d + j]
    var jitter_base = 1e-12 * trace / Float64(d)
    if jitter_base <= 0.0:
        jitter_base = 1e-300

    # Cholesky with escalating diagonal jitter (duplicate or collinear
    # points make the raw covariance singular; a tiny nudge fixes the
    # common case, status 2 covers the rest).
    var ok = False
    var attempt = 0
    while attempt < 7 and not ok:
        var jitter = 0.0
        if attempt > 0:
            jitter = jitter_base
            for _ in range(attempt - 1):
                jitter *= 10.0
        ok = True
        var j = 0
        while j < d and ok:
            var k = 0
            while k <= j and ok:
                var acc = cov[unsafe_offset=j * d + k]
                if j == k:
                    acc += jitter
                for p in range(k):
                    acc -= chol[unsafe_offset=j * d + p] * chol[
                        unsafe_offset=k * d + p
                    ]
                if j == k:
                    if acc <= 0.0:
                        ok = False
                    else:
                        chol[unsafe_offset=j * d + j] = sqrt(acc)
                else:
                    chol[unsafe_offset=j * d + k] = acc / chol[
                        unsafe_offset=k * d + k
                    ]
                k += 1
            # Zero the strict upper triangle for a clean ABI contract.
            if ok:
                for k2 in range(j + 1, d):
                    chol[unsafe_offset=j * d + k2] = 0.0
            j += 1
        attempt += 1
    if not ok:
        arena.unsafe_free()
        return 2

    # Max squared Mahalanobis distance of any live point: solve L z = d.
    var dmax2 = 0.0
    for i in range(nn):
        var row = u_points.unsafe_offset(i * d)
        for j in range(d):
            var acc = row[unsafe_offset=j] - out_mean[unsafe_offset=j]
            for p in range(j):
                acc -= chol[unsafe_offset=j * d + p] * z[unsafe_offset=p]
            z[unsafe_offset=j] = acc / chol[unsafe_offset=j * d + j]
        var d2 = 0.0
        for j in range(d):
            d2 += z[unsafe_offset=j] * z[unsafe_offset=j]
        if d2 > dmax2:
            dmax2 = d2
    if not (dmax2 > 0.0):  # all live points identical, or NaN upstream
        arena.unsafe_free()
        return 2

    # Final bounding matrix A = enlarge * dmax2 * cov; its Cholesky factor
    # is sqrt(enlarge * dmax2) * L.
    var scale = sqrt(enlarge * dmax2)
    var half_logdet = 0.0
    for j in range(d):
        for k in range(j + 1):
            out_chol[unsafe_offset=j * d + k] = chol[
                unsafe_offset=j * d + k
            ] * scale
        for k in range(j + 1, d):
            out_chol[unsafe_offset=j * d + k] = 0.0
        half_logdet += log(chol[unsafe_offset=j * d + j] * scale)

    # log-volume = log(unit-ball volume) + log|L_A|.
    out_logvol[] = _log_unit_ball_volume(d) + half_logdet

    arena.unsafe_free()
    return 0


@export
def dynestymojo_propose_batch(
    handle: Handle,
    ndim: Int64,
    mean: F64Ptr,
    chol: F64Ptr,
    mode: Int32,
    max_out: Int64,
    out_points: F64Ptr,
) abi("C") -> Int64:
    """Write up to max_out candidate unit-cube points (row-major) into
    out_points.

    mode 0: uniform over {ellipsoid(mean, chol) intersect [0,1)^ndim} via
    Gaussian-direction draws plus cube rejection; mode 1: uniform over the
    whole cube (mean/chol ignored). Returns the number of points written;
    0 in mode 0 means the ellipsoid/cube intersection is too small to hit
    within the attempt cap and the caller should switch to cube mode.
    """
    if not handle or ndim <= 0 or max_out <= 0:
        return 0
    if mode != MODE_ELLIPSOID and mode != MODE_CUBE:
        return 0
    var d = Int(ndim)
    var s = handle.value().unsafe_bitcast[UInt64]()

    var g = unsafe_alloc[Float64](d)
    var count = Int64(0)

    if mode == MODE_CUBE:
        while count < max_out:
            var row = out_points.unsafe_offset(Int(count) * d)
            for j in range(d):
                row[unsafe_offset=j] = _next_f64(s)
            count += 1
        g.unsafe_free()
        return count

    var max_attempts = max_out * MAX_ATTEMPTS_FACTOR
    var attempts = Int64(0)
    var inv_d = 1.0 / Float64(d)
    while count < max_out and attempts < max_attempts:
        attempts += 1
        # Random direction: g / |g| with g standard normal.
        var norm2 = 0.0
        for j in range(d):
            var gj = _next_gaussian(s)
            g[unsafe_offset=j] = gj
            norm2 += gj * gj
        if norm2 <= 0.0:
            continue
        var inv_norm = 1.0 / sqrt(norm2)
        # Radius: u^(1/d) gives a uniform-in-ball radial profile. u == 0
        # would collapse the draw onto the mean; redraw instead.
        var u = _next_f64(s)
        if u <= 0.0:
            continue
        var r = exp(log(u) * inv_d)
        # x = mean + r * L @ (g/|g|), L lower-triangular row-major.
        var row = out_points.unsafe_offset(Int(count) * d)
        var inside = True
        var j = 0
        while j < d and inside:
            var acc = 0.0
            for k in range(j + 1):
                acc += chol[unsafe_offset=j * d + k] * (
                    r * g[unsafe_offset=k] * inv_norm
                )
            var x = mean[unsafe_offset=j] + acc
            if x < 0.0 or x >= 1.0:
                inside = False
            else:
                row[unsafe_offset=j] = x
            j += 1
        if inside:
            count += 1

    g.unsafe_free()
    return count

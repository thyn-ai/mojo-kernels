"""Clean-room SPADE surrogate-dithering + p-value-spectrum kernels.

Written fresh from the published algorithms:

  * Spike dithering (elephant ``spike_train_surrogates.dither_spikes``
    semantics, the surrogate generation behind SPADE's p-value spectrum;
    Torre et al., Front. Comput. Neurosci. 7:132, 2013): each spike is
    displaced by an independent uniform offset in [-dither, +dither).
    Two modes are implemented:

      - method 0 (plain dither): spikes are re-drawn independently per
        surrogate; with edges_mode 0 spikes landing outside
        (t_start, t_stop) are dropped, with edges_mode 1 they are clamped
        to the range ends. Binning follows elephant's
        ``conversion.BinnedSpikeTrain`` (tolerance=None) semantics:
        bin = trunc((t - t_start) / bin_size), spikes whose bin index
        equals n_bins (i.e. landing exactly on t_stop) are discarded.
      - method 1 (refractory dither): the dither range of each spike is
        shrunk so it cannot enter the refractory period of its (current)
        neighbours; the refractory period is min(given, smallest ISI).
        Spikes are perturbed in a uniform random order (Fisher-Yates).

    Random numbers come from an in-kernel xoshiro256** generator seeded
    per (surrogate, train) stream via SplitMix64 — the output is fully
    deterministic for a given seed. RNG *sequence* parity with any other
    implementation is impossible by construction; the contract is
    distributional equivalence plus same-seed reproducibility.

  * P-value spectrum (elephant ``spade._get_pvalue_spec`` semantics):
    given the per-surrogate matrix of maximal pattern occurrences
    max_occs[surrogate, size, duration], for each (pattern size,
    duration) column the occurrence histogram over unit bins
    [min_occ, ..., max+1] is built, tail sums give the survival counts,
    and dividing by n_surr yields the p-value entries
    [size, occurrence, (duration,) p_value] in the reference emission
    order (size ascending, duration ascending, occurrence ascending).

Exported C ABI (batch-shaped: whole surrogate set / whole spectrum per
call, so FFI overhead is amortized to noise):

    int32_t elephantsurrogatesmojo_abi_version(void)

    int32_t elephantsurrogatesmojo_dither(
        n_trains, spike_counts[n_trains], spike_times[total_spikes],
        t_start, t_stop, bin_size, n_bins, dither, refractory_period,
        method, edges_mode, n_surrogates, seed,
        out_buf[n_surrogates * n_trains * n_bins])

        method      0 = plain dither, 1 = refractory dither
        edges_mode  method 0 only: 0 = drop outside, 1 = clamp to ends
        out         uint8 0/1 occupancy (bool layout), fully overwritten

    int64_t elephantsurrogatesmojo_pvalue_spec_count(
        n_rows, n_sizes, winlen, max_occs, min_occ)

    int32_t elephantsurrogatesmojo_pvalue_spec_fill(
        n_rows, n_sizes, winlen, max_occs, min_spikes, min_occ, n_surr,
        out_size, out_occ, out_dur, out_p)

        max_occs    float64[n_rows * n_sizes * winlen], C-order
        n_surr      p-value denominator (surrogate count the spectrum was
                    collected from; may exceed n_rows)
        out_*       int32/int32/int32/float64 arrays with room for the
                    count returned by the count call
"""

from std.memory import Pointer
from std.memory.alloc import unsafe_alloc
from std.origin import MutUntrackedOrigin

comptime ABI_VERSION: Int32 = 1

comptime METHOD_PLAIN: Int32 = 0
comptime METHOD_REFRACTORY: Int32 = 1

comptime EDGES_DROP: Int32 = 0
comptime EDGES_CLAMP: Int32 = 1

# C-side pointer spellings (untracked origin: the caller owns the lifetime
# of anything passed in; the library owns what it allocates).
comptime F64Ptr = Pointer[Float64, MutUntrackedOrigin]
comptime I32Ptr = Pointer[Int32, MutUntrackedOrigin]
comptime I64Ptr = Pointer[Int64, MutUntrackedOrigin]
comptime U8Ptr = Pointer[UInt8, MutUntrackedOrigin]

comptime INV_2POW53: Float64 = 1.0 / 9007199254740992.0
comptime SPLITMIX_INCREMENT: UInt64 = 0x9E3779B97F4A7C15


def splitmix64(mut state: UInt64) -> UInt64:
    """One SplitMix64 step; advances `state` and returns the variate."""
    state += SPLITMIX_INCREMENT
    var z = state
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9
    z = (z ^ (z >> 27)) * 0x94D049BB133111EB
    return z ^ (z >> 31)


def rotl64(x: UInt64, k: UInt64) -> UInt64:
    return (x << k) | (x >> (64 - k))


struct Xoshiro256StarStar:
    """xoshiro256** stream; deterministic, seedable per (surrogate, train)."""

    var s0: UInt64
    var s1: UInt64
    var s2: UInt64
    var s3: UInt64

    def __init__(out self, seed: UInt64, stream: UInt64):
        var state = seed ^ (stream * SPLITMIX_INCREMENT)
        self.s0 = splitmix64(state)
        self.s1 = splitmix64(state)
        self.s2 = splitmix64(state)
        self.s3 = splitmix64(state)

    def next(mut self) -> UInt64:
        var result = rotl64(self.s1 * 5, 7) * 9
        var t = self.s1 << 17
        self.s2 ^= self.s0
        self.s3 ^= self.s1
        self.s1 ^= self.s2
        self.s0 ^= self.s3
        self.s2 ^= t
        self.s3 = rotl64(self.s3, 45)
        return result

    def uniform(mut self) -> Float64:
        """Uniform double in [0, 1) from the top 53 bits."""
        return Float64(self.next() >> 11) * INV_2POW53


@export
def elephantsurrogatesmojo_abi_version() abi("C") -> Int32:
    return ABI_VERSION


def bin_index(t: Float64, t_start: Float64, bin_size: Float64) -> Int:
    """elephant BinnedSpikeTrain (tolerance=None) bin index: truncation."""
    return Int((t - t_start) / bin_size)


@export
def elephantsurrogatesmojo_dither(
    n_trains: Int64,
    spike_counts: I64Ptr,
    spike_times: F64Ptr,
    t_start: Float64,
    t_stop: Float64,
    bin_size: Float64,
    n_bins: Int64,
    dither: Float64,
    refractory_period: Float64,
    method: Int32,
    edges_mode: Int32,
    n_surrogates: Int64,
    seed: UInt64,
    out_buf: U8Ptr,
) abi("C") -> Int32:
    """Generate n_surrogates dithered binned surrogates for every train.

    Returns 0 on success, 1 on invalid arguments. `out` must hold
    n_surrogates*n_trains*n_bins uint8 slots and is fully overwritten
    (zeroed, then occupied bins set to 1).
    """
    if n_trains <= 0 or n_surrogates <= 0 or n_bins <= 0:
        return 1
    if bin_size <= 0.0 or dither < 0.0 or t_start >= t_stop:
        return 1
    if method != METHOD_PLAIN and method != METHOD_REFRACTORY:
        return 1
    if edges_mode != EDGES_DROP and edges_mode != EDGES_CLAMP:
        return 1

    var NTR = Int(n_trains)
    var NB = Int(n_bins)
    var NS = Int(n_surrogates)

    # Zero the whole output buffer first.
    var total = NS * NTR * NB
    for i in range(total):
        out_buf[i] = 0

    # Scratch buffers sized by the longest train (refractory path).
    var max_count: Int = 0
    for i in range(NTR):
        var c = Int(spike_counts[i])
        if c > max_count:
            max_count = c
    var cur = unsafe_alloc[Float64](max(max_count, 1))
    var perm = unsafe_alloc[Int64](max(max_count, 1))

    var base: Int = 0  # running offset into spike_times
    for i in range(NTR):
        var n = Int(spike_counts[i])
        for s in range(NS):
            var rng = Xoshiro256StarStar(seed, UInt64(s) * UInt64(NTR) + UInt64(i))
            var out_row = (s * NTR + i) * NB
            if method == METHOD_PLAIN:
                for j in range(n):
                    var t = spike_times[base + j]
                    var tp = t + 2.0 * dither * rng.uniform() - dither
                    if edges_mode == EDGES_DROP:
                        if tp <= t_start or tp >= t_stop:
                            continue
                    else:
                        if tp < t_start:
                            tp = t_start
                        elif tp > t_stop:
                            tp = t_stop
                    var b = bin_index(tp, t_start, bin_size)
                    if b >= 0 and b < NB:
                        out_buf[out_row + b] = 1
            else:
                # Effective refractory period: min(given, smallest ISI).
                var refr = refractory_period
                for j in range(1, n):
                    var isi = spike_times[base + j] - spike_times[base + j - 1]
                    if isi < refr:
                        refr = isi
                for j in range(n):
                    cur[j] = spike_times[base + j]
                    perm[j] = Int64(j)
                # Fisher-Yates: uniform random perturbation order.
                for jj in range(n - 1, 0, -1):
                    var k = Int(rng.uniform() * Float64(jj + 1))
                    if k > jj:
                        k = jj
                    var tmp = perm[jj]
                    perm[jj] = perm[k]
                    perm[k] = tmp
                for jj in range(n):
                    var idx = Int(perm[jj])
                    var spike = cur[idx]
                    var prev_spike = t_start - refr
                    if idx > 0:
                        prev_spike = cur[idx - 1]
                    var next_spike = t_stop + refr
                    if idx < n - 1:
                        next_spike = cur[idx + 1]
                    var prev_dither = dither
                    if spike - prev_spike - refr < prev_dither:
                        prev_dither = spike - prev_spike - refr
                    var next_dither = dither
                    if next_spike - spike - refr < next_dither:
                        next_dither = next_spike - spike - refr
                    var dt = (prev_dither + next_dither) * rng.uniform() - prev_dither
                    cur[idx] = spike + dt
                for j in range(n):
                    var b = bin_index(cur[j], t_start, bin_size)
                    if b >= 0 and b < NB:
                        out_buf[out_row + b] = 1
        base += n

    cur.free()
    perm.free()
    return 0


def int16_wrap(m: Float64) -> Int:
    """numpy scalar `.astype(np.int16)` semantics: trunc, then mod 2^16."""
    var mi = Int(m)
    var w = mi % 65536
    if w > 32767:
        w -= 65536
    elif w < -32768:
        w += 65536
    return w


def column_entry_count(
    max_occs: F64Ptr,
    n_surr: Int,
    col_stride: Int,
    col_offset: Int,
    min_occ: Int,
) -> Int:
    """Number of p-value entries one (size, duration) column emits.

    Mirrors np.histogram(col, bins=np.arange(min_occ, int16(max(col)) + 2)):
    entries == number of unit bins == max(n_edges - 1, 0).
    """
    var m = max_occs[col_offset]
    for j in range(1, n_surr):
        var v = max_occs[j * col_stride + col_offset]
        if v > m:
            m = v
    var n_edges = int16_wrap(m) + 2 - min_occ
    if n_edges <= 1:
        return 0
    return n_edges - 1


@export
def elephantsurrogatesmojo_pvalue_spec_count(
    n_rows: Int64,
    n_sizes: Int64,
    winlen: Int64,
    max_occs: F64Ptr,
    min_occ: Int64,
) abi("C") -> Int64:
    """Total number of spectrum entries (for output allocation). -1 on error."""
    if n_rows <= 0 or n_sizes <= 0 or winlen <= 0:
        return -1
    var NS = Int(n_rows)
    var NZ = Int(n_sizes)
    var NW = Int(winlen)
    var MO = Int(min_occ)
    var col_stride = NZ * NW  # C-order row stride of the (surr, size, dur) cube
    var total: Int = 0
    for size_id in range(NZ):
        for dur in range(NW):
            total += column_entry_count(
                max_occs, NS, col_stride, size_id * NW + dur, MO
            )
    return Int64(total)


@export
def elephantsurrogatesmojo_pvalue_spec_fill(
    n_rows: Int64,
    n_sizes: Int64,
    winlen: Int64,
    max_occs: F64Ptr,
    min_spikes: Int64,
    min_occ: Int64,
    n_surr: Int64,
    out_size: I32Ptr,
    out_occ: I32Ptr,
    out_dur: I32Ptr,
    out_p: F64Ptr,
) abi("C") -> Int32:
    """Fill the p-value spectrum entries. Returns 0 on success, 1 on error.

    The caller must allocate room for `..._pvalue_spec_count(...)` entries.
    Emission order: size ascending, duration ascending, occurrence
    ascending — the reference order. `n_rows` is the number of surrogate
    rows in `max_occs`; `n_surr` is only the p-value denominator (the
    reference divides by the given surrogate count, not the row count).
    """
    if n_rows <= 0 or n_sizes <= 0 or winlen <= 0 or n_surr <= 0:
        return 1
    var NS = Int(n_rows)
    var NSURR = Int(n_surr)
    var NZ = Int(n_sizes)
    var NW = Int(winlen)
    var MS = Int(min_spikes)
    var MO = Int(min_occ)
    var col_stride = NZ * NW  # C-order row stride of the (surr, size, dur) cube

    var pos: Int = 0
    for size_id in range(NZ):
        for dur in range(NW):
            var col_offset = size_id * NW + dur
            var nb = column_entry_count(max_occs, NS, col_stride, col_offset, MO)
            if nb <= 0:
                continue
            # Unit-bin histogram counts[k] = #{col in [MO+k, MO+k+1)} with
            # the last bin closed on the right (np.histogram semantics).
            var counts = unsafe_alloc[Int64](nb)
            for k in range(nb):
                counts[k] = 0
            var hi = Float64(MO + nb)  # right edge of the last bin
            for j in range(NS):
                var v = max_occs[j * col_stride + col_offset]
                if v >= Float64(MO) and v <= hi:
                    var k = Int(v) - MO
                    if k >= nb:
                        k = nb - 1  # v exactly on the closed right edge
                    counts[k] += 1
            # Tail sums divided by n_surr give the p-values (float64).
            # The reverse cumsum runs over k descending, but entries are
            # written at pos + k so emission stays occurrence-ascending,
            # exactly the reference order.
            var acc: Int64 = 0
            for k in range(nb - 1, -1, -1):
                acc += counts[k]
                out_size[pos + k] = Int32(MS + size_id)
                # np.uint16 cast of the bin's left edge, mirrored.
                out_occ[pos + k] = Int32((MO + k) & 0xFFFF)
                out_dur[pos + k] = Int32(dur)
                out_p[pos + k] = Float64(acc) / Float64(NSURR)
            pos += nb
            counts.free()
    return 0

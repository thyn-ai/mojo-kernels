"""Differential tests: obspy_mojo must match obspy's konno_ohmachi_smoothing.

Run twice by `scripts/test_all_obspy_konno.sh`: once against the native Mojo
kernel and once with OBSPY_MOJO_DISABLE_NATIVE=1 (forced pure-Python
fallback). Both backends must agree with the installed obspy package (the
oracle, tested with obspy 1.5.1) within the documented tolerance:

    float64: rtol=1e-9, atol=1e-12   (measured agreement: <= 3e-14 relative)
    float32: rtol=1e-4, atol=1e-6    (measured agreement: <= 5e-6 relative)

The tolerance is not bit-exact by design: the native kernel evaluates
log10/sin with SIMD and accumulates one SIMD lane per sample (NumPy uses
pairwise summation), so the last few ulps differ. Everything else — branch
decisions, edge semantics, dtypes, errors — is mirrored exactly.

Everything here is generated locally from explicit seeds — no network, no
randomness without a fixed seed — so the suite is reproducible on any
machine.
"""

from __future__ import annotations

import numpy as np
import pytest
from obspy.signal.konnoohmachismoothing import (
    konno_ohmachi_smoothing as oracle,
)

import obspy_mojo
from obspy_mojo import konno_ohmachi_smoothing as ours

RTOL64, ATOL64 = 1e-9, 1e-12
RTOL32, ATOL32 = 1e-4, 1e-6


def make_freqs(n: int, fmin: float = 0.1, fmax: float = 100.0) -> np.ndarray:
    """Log-spaced frequencies, the standard seismology grid."""
    return np.logspace(np.log10(fmin), np.log10(fmax), n)


def make_spectra(seed: int, shape, scale: float = 100.0, dtype=np.float64) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return (rng.random(shape) * scale).astype(dtype)


def assert_smoothing_close(actual: np.ndarray, expected: np.ndarray) -> None:
    assert isinstance(actual, np.ndarray), f"expected np.ndarray, got {type(actual)}"
    assert actual.shape == expected.shape, (actual.shape, expected.shape)
    assert actual.dtype == expected.dtype, (actual.dtype, expected.dtype)
    if expected.dtype == np.float32:
        rtol, atol = RTOL32, ATOL32
    else:
        rtol, atol = RTOL64, ATOL64
    np.testing.assert_allclose(actual, expected, rtol=rtol, atol=atol, equal_nan=True)


# --------------------------------------------------------------------------
# loop path (the default single-spectrum path): sizes x bandwidth x normalize
# --------------------------------------------------------------------------

SIZES = [8, 64, 257, 1024]
BANDWIDTHS = [pytest.param({}, id="b40-default"),
              pytest.param({"bandwidth": 10.0}, id="b10"),
              pytest.param({"bandwidth": 80.0}, id="b80")]
NORMALIZE = [pytest.param(True, id="normalize"),
             pytest.param(False, id="unnormalized")]


@pytest.mark.parametrize("n", SIZES)
@pytest.mark.parametrize("bkw", BANDWIDTHS)
@pytest.mark.parametrize("normalize", NORMALIZE)
def test_loop_path_seeded_parity(n, bkw, normalize):
    freqs = make_freqs(n)
    spectra = make_spectra(1000 + n, n)
    kw = dict(normalize=normalize, **bkw)
    assert_smoothing_close(ours(spectra, freqs, **kw), oracle(spectra, freqs, **kw))


@pytest.mark.parametrize("n", SIZES)
def test_loop_path_second_seed(n):
    freqs = make_freqs(n, fmin=0.01, fmax=50.0)
    spectra = make_spectra(2000 + n, n, scale=1e6)
    assert_smoothing_close(
        ours(spectra, freqs, normalize=True), oracle(spectra, freqs, normalize=True)
    )


# --------------------------------------------------------------------------
# matrix path (2-D input): the row-normalized matrix reduction
# --------------------------------------------------------------------------

@pytest.mark.parametrize("n", SIZES)
@pytest.mark.parametrize("normalize", NORMALIZE)
def test_matrix_path_2d_parity(n, normalize):
    freqs = make_freqs(n)
    spectra = make_spectra(3000 + n, (5, n))
    assert_smoothing_close(
        ours(spectra, freqs, normalize=normalize),
        oracle(spectra, freqs, normalize=normalize),
    )


def test_matrix_path_single_row_2d():
    # A (1, n) 2-D input takes the matrix path, which differs measurably from
    # the loop path; both must match the oracle's own choice.
    freqs = make_freqs(257)
    spectra = make_spectra(3100, (1, 257))
    assert_smoothing_close(
        ours(spectra, freqs, normalize=True), oracle(spectra, freqs, normalize=True)
    )


def test_matrix_path_3d_parity():
    freqs = make_freqs(64, fmin=1.0)
    spectra = make_spectra(3200, (2, 3, 64), scale=10.0)
    assert_smoothing_close(
        ours(spectra, freqs, normalize=True), oracle(spectra, freqs, normalize=True)
    )


# --------------------------------------------------------------------------
# count semantics: number of filter applications, clamped to >= 1
# --------------------------------------------------------------------------

@pytest.mark.parametrize("count", [1, 2, 3, 5])
@pytest.mark.parametrize("normalize", NORMALIZE)
def test_count_applications_1d(count, normalize):
    freqs = make_freqs(257)
    spectra = make_spectra(4000 + count, 257)
    kw = dict(count=count, normalize=normalize)
    assert_smoothing_close(ours(spectra, freqs, **kw), oracle(spectra, freqs, **kw))


@pytest.mark.parametrize("count", [1, 2, 3])
@pytest.mark.parametrize("normalize", NORMALIZE)
def test_count_applications_2d(count, normalize):
    freqs = make_freqs(257)
    spectra = make_spectra(4100 + count, (3, 257))
    kw = dict(count=count, normalize=normalize)
    assert_smoothing_close(ours(spectra, freqs, **kw), oracle(spectra, freqs, **kw))


@pytest.mark.parametrize("count", [0, -1, -10])
def test_count_clamped_to_one(count):
    freqs = make_freqs(128)
    spectra = make_spectra(4200, 128)
    spectra_2d = make_spectra(4201, (4, 128))
    for arr in (spectra, spectra_2d):
        kw = dict(count=count, normalize=True)
        assert_smoothing_close(ours(arr, freqs, **kw), oracle(arr, freqs, **kw))


def test_count_with_enforce_no_matrix():
    # count > 1 on the loop path = repeated per-center applications.
    freqs = make_freqs(257)
    spectra = make_spectra(4300, 257)
    kw = dict(count=3, enforce_no_matrix=True, normalize=True)
    assert_smoothing_close(ours(spectra, freqs, **kw), oracle(spectra, freqs, **kw))


# --------------------------------------------------------------------------
# memory-branch decision: matrix estimate vs max_memory_usage (SI megabytes)
# --------------------------------------------------------------------------

@pytest.mark.parametrize("mem", [0, 1, 2, 512])
@pytest.mark.parametrize("normalize", NORMALIZE)
def test_memory_branch_2d(mem, normalize):
    # n=400 -> 400^2 * 8 B = 1.28 MB: loop at mem <= 1, matrix at mem >= 2.
    freqs = make_freqs(400)
    spectra = make_spectra(4400, (3, 400))
    kw = dict(max_memory_usage=mem, normalize=normalize)
    assert_smoothing_close(ours(spectra, freqs, **kw), oracle(spectra, freqs, **kw))


@pytest.mark.parametrize("mem", [0, 2])
def test_memory_branch_1d_count2(mem):
    freqs = make_freqs(400)
    spectra = make_spectra(4500, 400)
    kw = dict(count=2, max_memory_usage=mem, normalize=True)
    assert_smoothing_close(ours(spectra, freqs, **kw), oracle(spectra, freqs, **kw))


def test_memory_threshold_boundary():
    # est = (n^2 + n + 2 * spectra.shape[0]) * itemsize / 2^20 < max_memory_usage
    # with a strict '<' at the exact boundary (probed black-box from the oracle).
    freqs = make_freqs(400)
    spectra = make_spectra(4600, (2, 400))
    exact = (400**2 + 400 + 2 * 2) * 8 / 1048576
    for mem in (np.nextafter(exact, -np.inf), exact, np.nextafter(exact, np.inf)):
        kw = dict(max_memory_usage=mem, normalize=True)
        assert_smoothing_close(ours(spectra, freqs, **kw), oracle(spectra, freqs, **kw))


# --------------------------------------------------------------------------
# edge frequencies: exact zeros, duplicates, negatives (NaN poisoning)
# --------------------------------------------------------------------------

ZERO_POSITIONS = [pytest.param(0, id="first"), pytest.param(31, id="middle"),
                  pytest.param(63, id="last")]


@pytest.mark.parametrize("pos", ZERO_POSITIONS)
@pytest.mark.parametrize("normalize", NORMALIZE)
def test_zero_frequency_loop(pos, normalize):
    freqs = make_freqs(64)
    freqs[pos] = 0.0
    spectra = make_spectra(5000 + pos, 64)
    kw = dict(normalize=normalize)
    assert_smoothing_close(ours(spectra, freqs, **kw), oracle(spectra, freqs, **kw))


@pytest.mark.parametrize("pos", ZERO_POSITIONS)
@pytest.mark.parametrize("normalize", NORMALIZE)
def test_zero_frequency_matrix(pos, normalize):
    freqs = make_freqs(64)
    freqs[pos] = 0.0
    spectra = make_spectra(5100 + pos, (4, 64))
    kw = dict(normalize=normalize)
    assert_smoothing_close(ours(spectra, freqs, **kw), oracle(spectra, freqs, **kw))


def test_zero_frequency_count2_matrix():
    freqs = make_freqs(64)
    freqs[31] = 0.0
    spectra = make_spectra(5200, (3, 64))
    kw = dict(count=2, normalize=True)
    assert_smoothing_close(ours(spectra, freqs, **kw), oracle(spectra, freqs, **kw))


def test_zero_frequency_passthrough_value():
    # Centers at exactly f == 0 return the input sample bit-for-bit.
    freqs = make_freqs(64)
    freqs[17] = 0.0
    spectra = make_spectra(5300, 64)
    for kw in (dict(normalize=True), dict(normalize=False)):
        out = ours(spectra, freqs, **kw)
        assert out[17] == spectra[17]


def test_duplicate_frequencies():
    freqs = make_freqs(64)
    freqs[11] = freqs[10]
    spectra = make_spectra(5400, 64)
    kw = dict(normalize=True)
    assert_smoothing_close(ours(spectra, freqs, **kw), oracle(spectra, freqs, **kw))
    kw2d = dict(normalize=False)
    two_d = make_spectra(5401, (2, 64))
    assert_smoothing_close(ours(two_d, freqs, **kw2d), oracle(two_d, freqs, **kw2d))


def test_negative_frequencies_poison_with_nan():
    freqs = make_freqs(64)
    freqs[7] = -4.0
    spectra = make_spectra(5500, 64)
    out_ours = ours(spectra, freqs, normalize=True)
    out_oracle = oracle(spectra, freqs, normalize=True)
    assert np.isnan(out_oracle).all()
    assert_smoothing_close(out_ours, out_oracle)  # equal_nan=True


def test_denormal_frequency_behaves_like_positive():
    freqs = make_freqs(64)
    freqs[0] = 1e-300
    spectra = make_spectra(5600, 64)
    kw = dict(normalize=True)
    assert_smoothing_close(ours(spectra, freqs, **kw), oracle(spectra, freqs, **kw))


# --------------------------------------------------------------------------
# dtypes: float32 end-to-end, mixed-dtype warning + cast
# --------------------------------------------------------------------------

@pytest.mark.parametrize("normalize", NORMALIZE)
def test_float32_loop(normalize):
    freqs = make_freqs(128).astype(np.float32)
    spectra = make_spectra(6000, 128, dtype=np.float32)
    kw = dict(normalize=normalize)
    assert_smoothing_close(ours(spectra, freqs, **kw), oracle(spectra, freqs, **kw))


@pytest.mark.parametrize("normalize", NORMALIZE)
def test_float32_matrix_and_count(normalize):
    freqs = make_freqs(128).astype(np.float32)
    spectra = make_spectra(6100, (3, 128), dtype=np.float32)
    for count in (1, 2):
        kw = dict(count=count, normalize=normalize)
        assert_smoothing_close(ours(spectra, freqs, **kw), oracle(spectra, freqs, **kw))


def test_float32_memory_branch():
    # n=400 float32 -> 0.64 MB threshold.
    freqs = make_freqs(400).astype(np.float32)
    spectra = make_spectra(6200, (2, 400), dtype=np.float32)
    for mem in (0.5, 0.64, 1.0):
        kw = dict(max_memory_usage=mem, normalize=True)
        assert_smoothing_close(ours(spectra, freqs, **kw), oracle(spectra, freqs, **kw))


def test_mixed_dtype_warns_and_casts():
    freqs = make_freqs(64).astype(np.float32)
    spectra = make_spectra(6300, 64)  # float64
    with pytest.warns(UserWarning, match="should have the same dtype"):
        out_ours = ours(spectra, freqs, normalize=True)
    with pytest.warns(UserWarning, match="should have the same dtype"):
        out_oracle = oracle(spectra, freqs, normalize=True)
    assert out_ours.dtype == np.float64 == out_oracle.dtype
    assert_smoothing_close(out_ours, out_oracle)
    # and the other way around
    with pytest.warns(UserWarning):
        out_ours2 = ours(spectra.astype(np.float32), freqs.astype(np.float64))
    with pytest.warns(UserWarning):
        out_oracle2 = oracle(spectra.astype(np.float32), freqs.astype(np.float64))
    assert_smoothing_close(out_ours2, out_oracle2)


def test_fortran_ordered_input():
    freqs = make_freqs(128)
    spectra = np.asfortranarray(make_spectra(6400, (5, 128)))
    kw = dict(normalize=True)
    assert_smoothing_close(ours(spectra, freqs, **kw), oracle(spectra, freqs, **kw))


# --------------------------------------------------------------------------
# validation and error parity
# --------------------------------------------------------------------------

def test_dtype_validation_errors():
    freqs = make_freqs(8)
    spectra = make_spectra(7000, 8)
    with pytest.raises(ValueError, match="`spectra` needs to have a dtype of float32/64"):
        ours(spectra.astype(np.int64), freqs)
    with pytest.raises(ValueError, match="`frequencies` needs to have a dtype of float32/64"):
        ours(spectra, freqs.astype(np.int64))
    with pytest.raises(ValueError, match="`spectra` needs to have a dtype"):
        ours(spectra.astype(bool), freqs)
    with pytest.raises(ValueError, match="`spectra` needs to have a dtype"):
        ours(spectra.astype(np.float16), freqs)
    with pytest.raises(ValueError, match="`spectra` needs to have a dtype"):
        ours(spectra.astype(np.complex128), freqs)


def test_list_input_raises_attribute_error():
    with pytest.raises(AttributeError):
        ours([1.0, 2.0, 3.0], make_freqs(3))
    with pytest.raises(AttributeError):
        oracle([1.0, 2.0, 3.0], make_freqs(3))


def test_length_mismatch_raises_value_error():
    with pytest.raises(ValueError):
        ours(make_spectra(7100, 16), make_freqs(8))
    with pytest.raises(ValueError):
        ours(make_spectra(7101, (2, 16)), make_freqs(8))


def test_count_type_errors():
    freqs = make_freqs(32)
    spectra = make_spectra(7200, 32)
    with pytest.raises(TypeError):
        ours(spectra, freqs, count=2.5)
    with pytest.raises(TypeError):
        oracle(spectra, freqs, count=2.5)
    with pytest.raises(TypeError):
        ours(spectra, freqs, count="3")
    with pytest.raises(TypeError):
        oracle(spectra, freqs, count="3")
    # bool is an int: count=True == count=1
    assert_smoothing_close(
        ours(spectra, freqs, count=True), oracle(spectra, freqs, count=True)
    )


def test_3d_loop_path_raises_index_error():
    freqs = make_freqs(16, fmin=1.0)
    spectra = make_spectra(7300, (2, 3, 16))
    with pytest.raises(IndexError):
        ours(spectra, freqs, enforce_no_matrix=True)
    with pytest.raises(IndexError):
        oracle(spectra, freqs, enforce_no_matrix=True)


def test_bandwidth_types_and_sign():
    freqs = make_freqs(64)
    spectra = make_spectra(7400, 64)
    for b in (40, np.float32(40), 40.0, -40.0):
        kw = dict(bandwidth=b, normalize=True)
        assert_smoothing_close(ours(spectra, freqs, **kw), oracle(spectra, freqs, **kw))


def test_bandwidth_zero_poisons_with_nan():
    freqs = make_freqs(64)
    spectra = make_spectra(7500, 64)
    out_ours = ours(spectra, freqs, bandwidth=0.0, normalize=True)
    out_oracle = oracle(spectra, freqs, bandwidth=0.0, normalize=True)
    assert np.isnan(out_oracle).all()
    assert_smoothing_close(out_ours, out_oracle)


def test_empty_inputs():
    assert_smoothing_close(ours(np.empty(0), np.empty(0)), oracle(np.empty(0), np.empty(0)))
    s0 = np.zeros((0, 5))
    f5 = make_freqs(5, fmin=1.0)
    assert_smoothing_close(ours(s0, f5), oracle(s0, f5))
    s2 = np.zeros((2, 0))
    assert_smoothing_close(ours(s2, np.empty(0)), oracle(s2, np.empty(0)))


def test_single_frequency():
    out_ours = ours(np.array([3.0]), np.array([2.0]), normalize=True)
    out_oracle = oracle(np.array([3.0]), np.array([2.0]), normalize=True)
    assert_smoothing_close(out_ours, out_oracle)


# --------------------------------------------------------------------------
# backend sanity: the suite runs once per backend (see scripts/test_all_obspy_konno.sh)
# --------------------------------------------------------------------------

def test_expected_backend_is_serving():
    import os

    info = obspy_mojo.backend_info()
    if os.environ.get("OBSPY_MOJO_DISABLE_NATIVE") == "1":
        assert info["native_available"] is False
    else:
        # The native run requires a built kernel (scripts/test_all_obspy_konno.sh
        # builds it first).
        assert info["native_available"] is True, info.get("error")

"""Differential tests: elephant_mojo vs the elephant reference (pip oracle).

Run twice by `scripts/test_all_elephant_surrogates.sh`: once against the
native Mojo kernel and once with ELEPHANT_MOJO_DISABLE_NATIVE=1 (forced
NumPy fallback). The parity contract is split by function, mirroring the
nature of the two kernels:

- ``pvalue_spectrum`` is DETERMINISTIC: it must reproduce elephant's
  ``spade._get_pvalue_spec`` output BIT-EXACTLY (same signature integers,
  same float64 p-values compared with ``==``) on both backends. Also
  verified end-to-end through elephant's real ``spade.pvalue_spectrum``
  pipeline with a pinned surrogate generator.
- ``dither`` is STOCHASTIC: RNG-sequence parity across implementations is
  impossible by construction, so the contract is (a) same-seed
  reproducibility within a backend, and (b) statistical equivalence with
  the reference (`spade._generate_binned_surrogates` /
  `spike_train_surrogates.dither_spikes`): KS tests on displacement and
  survivor-count distributions (p > 1e-3), per-bin occupancy
  two-proportion z-tests with Bonferroni correction, and exact invariant
  checks (rate, order, refractory separation, shapes). Fixed seeds make
  every statistic reproducible.

The oracle is the published PyPI release (elephant==1.2.1, installed by
the test runner); tests that need it skip cleanly when it is absent.
"""

from __future__ import annotations

import os
import random
import subprocess
import sys
import textwrap

import numpy as np
import pytest

import elephant_mojo
from elephant_mojo import dither, pvalue_spectrum

elephant = pytest.importorskip("elephant", reason="test oracle elephant not installed")
import quantities as pq  # noqa: E402
import neo  # noqa: E402
from scipy import stats  # noqa: E402

import elephant.spade as espade  # noqa: E402
from elephant.spike_train_surrogates import dither_spikes  # noqa: E402

T_START = 0.0
T_STOP = 1000.0
BIN_SIZE = 5.0
N_BINS = int(T_STOP / BIN_SIZE)  # 200
DITHER = 15.0

# Statistical thresholds (documented in the module docstring).
KS_P_MIN = 1e-3
SURVIVOR_FRAC_TOL = 0.02
# Family-wise alpha for the per-bin occupancy z-tests, Bonferroni-corrected
# over bins. 0.001 (not 0.05): these gates hunt SEMANTIC divergences (wrong
# dither width, off-by-one bins, wrong edge handling), which produce z >> 10,
# so a 0.1% family-wise rate keeps full power while a fixed seeded draw sits
# far from the threshold. All oracle draws are fully seeded (numpy AND
# stdlib random — elephant's refractory path uses random.random()), so the
# statistics are deterministic across machines; the margin is insurance
# against toolchain-version wobble only.
OCCUPANCY_ALPHA = 0.001


def _neo_trains(trains_ms: list[np.ndarray]) -> list[neo.SpikeTrain]:
    return [
        neo.SpikeTrain(t * pq.ms, t_start=T_START * pq.ms, t_stop=T_STOP * pq.ms)
        for t in trains_ms
    ]


def _oracle_binned_surrogates(trains_ms, n_surr, method="dither_spikes", seed=0):
    """Reference path: spade's _generate_binned_surrogates -> bool arrays."""
    np.random.seed(seed)
    random.seed(seed)  # some oracle paths draw from stdlib random too
    sts = _neo_trains(trains_ms)
    out = [
        bst.to_bool_array()
        for _, bst in espade._generate_binned_surrogates(
            sts,
            bin_size=BIN_SIZE * pq.ms,
            dither=DITHER * pq.ms,
            surr_method=method,
            n_surrogates=n_surr,
        )
    ]
    return np.stack(out)


def _bin_times_trunc(times_ms: np.ndarray) -> np.ndarray:
    """Verified BinnedSpikeTrain(tolerance=None) rule: trunc, drop == n_bins."""
    b = ((times_ms - T_START) / BIN_SIZE).astype(np.int64)
    return b[(b >= 0) & (b < N_BINS)]


def _two_proportion_z(count_a, n_a, count_b, n_b):
    p_pool = (count_a + count_b) / (n_a + n_b)
    se = np.sqrt(p_pool * (1.0 - p_pool) * (1.0 / n_a + 1.0 / n_b))
    se = np.maximum(se, 1e-12)
    return np.abs(count_a / n_a - count_b / n_b) / se


# ---------------------------------------------------------------------------
# pvalue_spectrum: bit-exact parity with spade._get_pvalue_spec
# ---------------------------------------------------------------------------


def _ref_pvalue_spec(max_occs, min_spikes, max_spikes, min_occ, n_surr, winlen, spectrum):
    return espade._get_pvalue_spec(
        np.asarray(max_occs, dtype=np.float64).copy(),
        min_spikes,
        max_spikes,
        min_occ,
        n_surr,
        winlen,
        spectrum,
    )


def _assert_entries_bit_exact(mine, ref, spectrum):
    assert len(mine) == len(ref)
    for m, r in zip(mine, ref):
        if spectrum == "#":
            assert m[0] == int(r[0]) and m[1] == int(r[1])
            assert m[2] == float(r[2])  # exact float64 equality, no tolerance
        else:
            assert m[0] == int(r[0]) and m[1] == int(r[1]) and m[2] == int(r[2])
            assert m[3] == float(r[3])


@pytest.mark.parametrize("seed", range(12))
def test_pvalue_spectrum_random_matrices_bit_exact(seed):
    rng = np.random.default_rng(seed)
    n_surr = int(rng.integers(1, 30))
    n_sizes = int(rng.integers(1, 5))
    min_spikes = int(rng.integers(1, 4))
    max_spikes = min_spikes + n_sizes - 1
    min_occ = int(rng.integers(0, 5))
    max_occs = rng.integers(0, 9, size=(n_surr, n_sizes)).astype(np.float64)
    ref = _ref_pvalue_spec(max_occs, min_spikes, max_spikes, min_occ, n_surr, 1, "#")
    mine = pvalue_spectrum(
        max_occs, min_spikes, max_spikes, min_occ, n_surr=n_surr, spectrum="#"
    )
    _assert_entries_bit_exact(mine, ref, "#")


@pytest.mark.parametrize("seed", range(8))
def test_pvalue_spectrum_random_cubes_bit_exact(seed):
    rng = np.random.default_rng(1000 + seed)
    n_surr = int(rng.integers(1, 20))
    n_sizes = int(rng.integers(1, 4))
    winlen = int(rng.integers(1, 4))
    min_spikes = int(rng.integers(1, 4))
    max_spikes = min_spikes + n_sizes - 1
    min_occ = int(rng.integers(0, 4))
    max_occs = rng.integers(0, 7, size=(n_surr, n_sizes, winlen)).astype(np.float64)
    ref = _ref_pvalue_spec(
        max_occs, min_spikes, max_spikes, min_occ, n_surr, winlen, "3d#"
    )
    mine = pvalue_spectrum(
        max_occs,
        min_spikes,
        max_spikes,
        min_occ,
        n_surr=n_surr,
        winlen=winlen,
        spectrum="3d#",
    )
    _assert_entries_bit_exact(mine, ref, "3d#")


def test_pvalue_spectrum_degenerate_columns_bit_exact():
    # Columns whose maximum is below min_occ (even all-zero) emit no entries;
    # values on bin edges exercise the closed last bin.
    max_occs = np.array(
        [
            [0.0, 0.0, 5.0],
            [0.0, 1.0, 5.0],
            [2.0, 0.0, 6.0],
            [0.0, 0.0, 7.0],
        ]
    )
    ref = _ref_pvalue_spec(max_occs, 2, 4, 2, 4, 1, "#")
    mine = pvalue_spectrum(max_occs, 2, 4, 2, n_surr=4)
    _assert_entries_bit_exact(mine, ref, "#")
    # min_occ = 0 includes the zero bin in the histogram.
    ref0 = _ref_pvalue_spec(max_occs, 2, 4, 0, 4, 1, "#")
    mine0 = pvalue_spectrum(max_occs, 2, 4, 0, n_surr=4)
    _assert_entries_bit_exact(mine0, ref0, "#")


def test_pvalue_spectrum_large_occurrences_bit_exact():
    rng = np.random.default_rng(77)
    max_occs = rng.integers(0, 30000, size=(40, 3)).astype(np.float64)
    ref = _ref_pvalue_spec(max_occs, 2, 4, 1, 40, 1, "#")
    mine = pvalue_spectrum(max_occs, 2, 4, 1, n_surr=40)
    _assert_entries_bit_exact(mine, ref, "#")


def test_pvalue_spectrum_single_surrogate_and_n_surr_override():
    max_occs = np.array([[3.0, 0.0]])
    ref = _ref_pvalue_spec(max_occs, 2, 3, 1, 1, 1, "#")
    mine = pvalue_spectrum(max_occs, 2, 3, 1)
    _assert_entries_bit_exact(mine, ref, "#")
    # The reference divides by the *given* n_surr, not by the row count.
    ref10 = _ref_pvalue_spec(max_occs, 2, 3, 1, 10, 1, "#")
    mine10 = pvalue_spectrum(max_occs, 2, 3, 1, n_surr=10)
    _assert_entries_bit_exact(mine10, ref10, "#")


def test_oracle_mining_extension_starts_beside_numpy():
    """elephant's compiled `fim` mining extension must start in a process
    that has already imported this environment's numpy.

    Regression test for the abort behind the end-to-end test below. On macOS
    the oracle wheel vendors its own copy of LLVM's OpenMP runtime for `fim`,
    and libomp aborts the whole interpreter ("OMP: Error #15") the moment a
    second copy initialises — which is what happened while the environment's
    OpenBLAS was the OpenMP build (pixi.toml pins the pthreads build on
    osx-arm64 for this reason; Linux wheels vendor GCC's libgomp, which
    coexists with libomp). Inside pytest that abort is uncatchable and output
    capture swallows the runtime's message, so the check runs in a child
    interpreter and reports its stderr. The mining call is the one
    `spade.concepts_mining` makes for every surrogate, on the smallest binned
    matrix that reaches `fim`: identical transactions short-circuit before it,
    so the third window differs from the other two.
    """
    child = textwrap.dedent(
        """
        import numpy as np
        import quantities as pq
        from elephant import conversion as conv
        import elephant.spade as espade

        assert espade.HAVE_FIM, "oracle has no compiled fim extension"
        # Two neurons, four bins: windows 0 and 2 hold both neurons, window 3
        # only the first, so the transactions differ and fim is called.
        binned = np.array([[1, 0, 1, 1], [1, 0, 1, 0]], dtype=bool)
        bst = conv.BinnedSpikeTrain(
            binned, bin_size=1 * pq.ms, t_start=0 * pq.ms, t_stop=4 * pq.ms, tolerance=None
        )
        concepts, _ = espade.concepts_mining(
            bst, 1 * pq.ms, 1, min_spikes=2, max_spikes=2, min_occ=2, min_neu=1, report="#"
        )
        print(concepts.tolist())
        """
    )
    proc = subprocess.run([sys.executable, "-c", child], capture_output=True, text=True, timeout=300)
    assert proc.returncode == 0, (
        f"oracle mining did not survive in a child interpreter "
        f"(returncode {proc.returncode}); stderr:\n{proc.stderr}"
    )
    # The size-2 pattern {neuron 0, neuron 1} occurs twice: one spectrum entry.
    assert proc.stdout.strip() == "[[2, 2, 1]]", proc.stdout


def test_pvalue_spectrum_end_to_end_through_elephant_pipeline(monkeypatch):
    """Bit-exact inside elephant's real spade.pvalue_spectrum pipeline.

    The surrogate generator is pinned to elephant_mojo's dither output, so
    both sides mine identical surrogates; the pipeline's spectrum must
    equal ours entry for entry.
    """
    rng = np.random.default_rng(5)
    trains = [np.sort(rng.uniform(T_START, T_STOP, 50)) for _ in range(4)]
    n_surr = 24
    fixed = dither(
        trains, BIN_SIZE, DITHER, n_surrogates=n_surr, t_stop=T_STOP, seed=1234
    )
    sts = _neo_trains(trains)

    def fake_generator(spiketrains, bin_size, dither, surr_method, n_surrogates, **kw):
        from elephant import conversion as conv

        for surr_id in range(n_surrogates):
            yield surr_id, conv.BinnedSpikeTrain(
                fixed[surr_id],
                bin_size=bin_size,
                t_start=spiketrains[0].t_start,
                t_stop=spiketrains[0].t_stop,
                tolerance=None,
            )

    monkeypatch.setattr(espade, "_generate_binned_surrogates", fake_generator)
    min_spikes, max_spikes, min_occ, winlen = 2, 4, 2, 1
    pv_ref = espade.pvalue_spectrum(
        sts,
        bin_size=BIN_SIZE * pq.ms,
        winlen=winlen,
        dither=DITHER * pq.ms,
        n_surr=n_surr,
        min_spikes=min_spikes,
        max_spikes=max_spikes,
        min_occ=min_occ,
        spectrum="#",
    )

    # Rebuild the pipeline's intermediate max_occs from the same fixed
    # surrogates with elephant's own mining + max-occurrence reduction.
    from elephant import conversion as conv

    max_occs = np.zeros((n_surr, max_spikes - min_spikes + 1))
    for surr_id in range(n_surr):
        bst = conv.BinnedSpikeTrain(
            fixed[surr_id],
            bin_size=BIN_SIZE * pq.ms,
            t_start=T_START * pq.ms,
            t_stop=T_STOP * pq.ms,
            tolerance=None,
        )
        surr_concepts = espade.concepts_mining(
            bst,
            BIN_SIZE * pq.ms,
            winlen,
            min_spikes=min_spikes,
            max_spikes=max_spikes,
            min_occ=min_occ,
            min_neu=1,
            report="#",
        )[0][:, :-1]
        max_occs[surr_id] = espade._get_max_occ(
            surr_concepts, min_spikes, max_spikes, winlen, "#"
        )
    mine = pvalue_spectrum(
        max_occs, min_spikes, max_spikes, min_occ, n_surr=n_surr, spectrum="#"
    )
    _assert_entries_bit_exact(mine, pv_ref, "#")
    assert len(mine) > 0  # the fixture must actually exercise the path


def test_pvalue_spectrum_validation_errors():
    with pytest.raises(ValueError, match="Invalid spectrum"):
        pvalue_spectrum(np.zeros((2, 2)), 2, 3, 1, spectrum="4d#")
    with pytest.raises(ValueError, match="2-D"):
        pvalue_spectrum(np.zeros((2, 2, 1)), 2, 3, 1, spectrum="#")
    with pytest.raises(ValueError, match="3-D"):
        pvalue_spectrum(np.zeros((2, 2)), 2, 3, 1, spectrum="3d#")
    with pytest.raises(ValueError, match="size axis"):
        pvalue_spectrum(np.zeros((2, 4)), 2, 3, 1)
    with pytest.raises(ValueError, match="max_spikes"):
        pvalue_spectrum(np.zeros((2, 2)), 4, 3, 1)
    with pytest.raises(ValueError, match="non-finite"):
        pvalue_spectrum(np.array([[np.inf, 0.0]]), 2, 3, 1)
    with pytest.raises(ValueError, match="duration axis"):
        pvalue_spectrum(np.zeros((2, 2, 3)), 2, 3, 1, winlen=2, spectrum="3d#")
    with pytest.raises(ValueError, match="at least one surrogate"):
        pvalue_spectrum(np.zeros((0, 2)), 2, 3, 1)


# ---------------------------------------------------------------------------
# dither: determinism, invariants, and statistical equivalence
# ---------------------------------------------------------------------------


def test_dither_shape_dtype_and_rate_invariant():
    rng = np.random.default_rng(9)
    trains = [np.sort(rng.uniform(T_START, T_STOP, 30)) for _ in range(3)]
    out = dither(trains, BIN_SIZE, DITHER, n_surrogates=7, t_stop=T_STOP, seed=1)
    assert out.shape == (7, 3, N_BINS)
    assert out.dtype == bool
    # Interior spikes with ISI > 2*dither + bin_size: no edge losses and no
    # bin collisions are possible, so every spike survives in its own bin.
    interior = [
        np.array([100.0, 200.0, 350.0, 500.0, 650.0, 800.0]),
        np.array([150.0, 400.0, 700.0]),
    ]
    out2 = dither(interior, BIN_SIZE, DITHER, n_surrogates=5, t_stop=T_STOP, seed=2)
    for i, t in enumerate(interior):
        assert (out2[:, i, :].sum(axis=1) == len(t)).all()


def test_dither_empty_train_and_single_spike():
    out = dither([[], [500.0]], BIN_SIZE, DITHER, n_surrogates=4, t_stop=T_STOP, seed=3)
    assert out.shape == (4, 2, N_BINS)
    assert not out[:, 0, :].any()
    assert out[:, 1, :].sum(axis=1).min() == 1  # center spike never drops


def test_dither_seed_reproducibility():
    trains = [np.array([100.0, 250.0, 700.0])]
    a = dither(trains, BIN_SIZE, DITHER, n_surrogates=8, t_stop=T_STOP, seed=42)
    b = dither(trains, BIN_SIZE, DITHER, n_surrogates=8, t_stop=T_STOP, seed=42)
    assert np.array_equal(a, b)  # same backend, same seed -> identical
    c = dither(trains, BIN_SIZE, DITHER, n_surrogates=8, t_stop=T_STOP, seed=43)
    assert not np.array_equal(a, c)
    d = dither(trains, BIN_SIZE, DITHER, n_surrogates=8, t_stop=T_STOP, seed=None)
    e = dither(trains, BIN_SIZE, DITHER, n_surrogates=8, t_stop=T_STOP, seed=None)
    assert not np.array_equal(d, e)  # OS-entropy seeds


def test_dither_non_integer_ratio_truncates_and_drops_t_stop_spike():
    # (t_stop - t_start) / bin_size truncates; a spike exactly on t_stop is
    # discarded by the binning (verified reference semantics).
    out = dither(
        [[6.0]], 4.0, 0.0, n_surrogates=2, t_start=0.0, t_stop=6.0, seed=0
    )
    assert out.shape == (2, 1, 1)
    assert not out.any()  # bin index 1 == n_bins -> discarded
    out0 = dither([[0.0, 3.9]], 4.0, 0.0, n_surrogates=2, t_stop=6.0, seed=0)
    assert out0.shape == (2, 1, 1)
    assert out0[:, 0, 0].all()  # both spikes land in bin 0


@pytest.mark.parametrize("position", [8.0, 500.0, 992.0])
def test_dither_displacement_distribution_matches_oracle(position):
    """Single-spike displacement: occupied-bin distribution vs reference."""
    n_surr = 8000
    np.random.seed(0)
    random.seed(0)  # defense: some oracle paths draw from stdlib random
    oracle_bins = np.empty(n_surr)
    st = _neo_trains([[position]])[0]
    for k in range(n_surr):
        s = np.asarray(dither_spikes(st, dither=DITHER * pq.ms)[0].magnitude)
        b = _bin_times_trunc(s)
        oracle_bins[k] = b[0] if b.size else -1  # -1: dropped at the edges
    mine = dither([[position]], BIN_SIZE, DITHER, n_surrogates=n_surr, t_stop=T_STOP, seed=99)
    mine_bins = np.array(
        [r[0] if r.size else -1 for r in (np.flatnonzero(row[0]) for row in mine)]
    )
    oracle_surv = oracle_bins[oracle_bins >= 0].astype(float)
    mine_surv = mine_bins[mine_bins >= 0].astype(float)
    ks = stats.ks_2samp(oracle_surv, mine_surv)
    assert ks.pvalue > KS_P_MIN, f"position {position}: {ks}"
    oracle_drop = (oracle_bins < 0).mean()
    mine_drop = (mine_bins < 0).mean()
    assert abs(oracle_drop - mine_drop) < SURVIVOR_FRAC_TOL


def test_dither_survivor_counts_and_occupancy_match_oracle():
    rng = np.random.default_rng(11)
    trains = [np.sort(rng.uniform(T_START, T_STOP, 40)) for _ in range(4)]
    n_surr = 2000
    oracle = _oracle_binned_surrogates(trains, n_surr, seed=0)
    mine = dither(trains, BIN_SIZE, DITHER, n_surrogates=n_surr, t_stop=T_STOP, seed=77)
    oc = oracle.sum(axis=2).astype(float)
    mc = mine.sum(axis=2).astype(float)
    for i in range(len(trains)):
        ks = stats.ks_2samp(oc[:, i], mc[:, i])
        assert ks.pvalue > KS_P_MIN, f"train {i}: {ks}"
        se = oc[:, i].std() / np.sqrt(n_surr) * 2  # ~99% two-sided mean band
        assert abs(oc[:, i].mean() - mc[:, i].mean()) < 4 * se + 0.05
    # Per-bin occupancy: two-proportion z-test, Bonferroni over all bins.
    z = _two_proportion_z(oracle.sum(axis=0), n_surr, mine.sum(axis=0), n_surr)
    zcrit = stats.norm.ppf(1.0 - OCCUPANCY_ALPHA / (2.0 * oracle[0].size))
    assert z.max() < zcrit, f"occupancy mismatch: max z {z.max():.2f} >= {zcrit:.2f}"


@pytest.mark.parametrize("position", [8.0, 992.0])
def test_dither_edges_false_clamps_like_oracle(position):
    """edges=False clamps to [t_start, t_stop]; a spike clamped exactly onto
    t_stop is then discarded by the binning (verified reference semantics),
    so both sides drop at the same rate."""
    n_surr = 8000
    np.random.seed(0)
    random.seed(0)  # defense: some oracle paths draw from stdlib random
    st = _neo_trains([[position]])[0]
    oracle_bins = np.empty(n_surr)
    for k in range(n_surr):
        s = np.asarray(
            dither_spikes(st, dither=DITHER * pq.ms, edges=False)[0].magnitude
        )
        b = _bin_times_trunc(s)  # bin == n_bins (t_stop) drops out here
        oracle_bins[k] = b[0] if b.size else -1
    mine = dither(
        [[position]], BIN_SIZE, DITHER, n_surrogates=n_surr, t_stop=T_STOP,
        edges=False, seed=17,
    )
    rows = [np.flatnonzero(row[0]) for row in mine]
    mine_bins = np.array([r[0] if r.size else -1 for r in rows])
    oracle_surv = oracle_bins[oracle_bins >= 0]
    mine_surv = mine_bins[mine_bins >= 0].astype(float)
    ks = stats.ks_2samp(oracle_surv, mine_surv)
    assert ks.pvalue > KS_P_MIN, ks
    # The clamps must be exercised on both sides at matching rates.
    edge_bin = 0 if position < T_STOP / 2 else N_BINS - 1
    assert (
        abs((oracle_bins == edge_bin).mean() - (mine_bins == edge_bin).mean())
        < SURVIVOR_FRAC_TOL
    )
    assert abs((oracle_bins < 0).mean() - (mine_bins < 0).mean()) < SURVIVOR_FRAC_TOL


def test_dither_refractory_matches_oracle_distribution():
    rng = np.random.default_rng(13)
    train = np.sort(rng.uniform(T_START, T_STOP, 30))
    n_surr = 2500
    refr_ms = 5.0
    np.random.seed(0)
    # elephant's refractory path draws the per-spike displacement with the
    # stdlib `random` module (random.random()), not numpy's RNG — seed both,
    # otherwise the oracle draw differs between processes and the occupancy
    # statistic flakes at the family-wise threshold (observed on CI).
    random.seed(0)
    st = _neo_trains([train])[0]
    oracle = np.zeros((n_surr, N_BINS), dtype=bool)
    for k in range(n_surr):
        s = np.asarray(
            dither_spikes(st, dither=DITHER * pq.ms, refractory_period=refr_ms * pq.ms)[
                0
            ].magnitude
        )
        oracle[k, _bin_times_trunc(s)] = True
    mine = dither(
        [train],
        BIN_SIZE,
        DITHER,
        n_surrogates=n_surr,
        t_stop=T_STOP,
        method="dither_spikes_with_refractory_period",
        refractory_period=refr_ms,
        seed=5,
    )[:, 0, :]
    oc = oracle.sum(axis=1).astype(float)
    mc = mine.sum(axis=1).astype(float)
    ks = stats.ks_2samp(oc, mc)
    assert ks.pvalue > KS_P_MIN, ks
    z = _two_proportion_z(oracle.sum(axis=0), n_surr, mine.sum(axis=0), n_surr)
    zcrit = stats.norm.ppf(1.0 - OCCUPANCY_ALPHA / (2.0 * N_BINS))
    assert z.max() < zcrit, f"occupancy mismatch: max z {z.max():.2f} >= {zcrit:.2f}"


def test_dither_refractory_separation_invariant():
    """With effective refractory >= bin_size the two spikes never share a bin."""
    refr_ms = 5.0  # == bin_size; original ISI 20 ms -> effective refr 5 ms
    trains = [[490.0, 510.0]]
    out = dither(
        trains,
        BIN_SIZE,
        DITHER,
        n_surrogates=3000,
        t_stop=T_STOP,
        method="dither_spikes_with_refractory_period",
        refractory_period=refr_ms,
        seed=21,
    )
    assert (out.sum(axis=2) == 2).all()  # neither spike ever drops or merges


def test_dither_validation_errors():
    good = [[100.0, 200.0]]
    with pytest.raises(ValueError, match="t_stop is required"):
        dither(good, BIN_SIZE, DITHER)
    with pytest.raises(ValueError, match="bin_size"):
        dither(good, 0.0, DITHER, t_stop=T_STOP)
    with pytest.raises(ValueError, match="dither"):
        dither(good, BIN_SIZE, -1.0, t_stop=T_STOP)
    with pytest.raises(ValueError, match="t_start must be smaller"):
        dither(good, BIN_SIZE, DITHER, t_start=10.0, t_stop=10.0)
    with pytest.raises(ValueError, match="n_surrogates"):
        dither(good, BIN_SIZE, DITHER, n_surrogates=0, t_stop=T_STOP)
    with pytest.raises(ValueError, match="non-empty list"):
        dither([], BIN_SIZE, DITHER, t_stop=T_STOP)
    with pytest.raises(ValueError, match="1-D"):
        dither([np.zeros((2, 2))], BIN_SIZE, DITHER, t_stop=T_STOP)
    with pytest.raises(ValueError, match="sorted ascending"):
        dither([[300.0, 100.0]], BIN_SIZE, DITHER, t_stop=T_STOP)
    with pytest.raises(ValueError, match="outside"):
        dither([[100.0, 2000.0]], BIN_SIZE, DITHER, t_stop=T_STOP)
    with pytest.raises(ValueError, match="method"):
        dither(good, BIN_SIZE, DITHER, t_stop=T_STOP, method="shuffle")
    with pytest.raises(ValueError, match="refractory_period is required"):
        dither(
            good,
            BIN_SIZE,
            DITHER,
            t_stop=T_STOP,
            method="dither_spikes_with_refractory_period",
        )
    with pytest.raises(ValueError, match="refractory_period must be None"):
        dither(good, BIN_SIZE, DITHER, t_stop=T_STOP, refractory_period=5.0)
    with pytest.raises(ValueError, match="seed"):
        dither(good, BIN_SIZE, DITHER, t_stop=T_STOP, seed=-1)
    with pytest.raises(ValueError, match="edges"):
        dither(good, BIN_SIZE, DITHER, t_stop=T_STOP, edges="yes")


def test_backend_indicator_matches_environment():
    forced = os.environ.get("ELEPHANT_MOJO_DISABLE_NATIVE") == "1"
    assert elephant_mojo.native_available() is (not forced)

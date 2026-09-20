"""Differential tests: motmetrics_mojo must match py-motmetrics value-for-value.

Run twice by `scripts/test_all_motmetrics.sh`: once against the native Mojo
kernel and once with MOTMETRICS_MOJO_DISABLE_NATIVE=1 (forced pure-Python
fallback). Both backends must agree with the oracle: every count exactly
(counts, MOSTLY_*, idtp/idfp/idfn, and the single-division ratios MOTA /
precision / recall / IDP / IDR / IDF1 are integer-derived and compared
exactly), and MOTP within the documented 1e-10 (its distance sum is the only
floating-point accumulation; the oracle's pairwise summation can differ from
the sequential one by ~1 ulp — measured agreement is ~1e-15).

The oracle is the published PyPI package `motmetrics` 1.4.0 (on its default
scipy assignment-solver path), exercised through MOTAccumulator.update +
MetricsHost.compute — the exact public API users call.

Everything here is generated locally from explicit seeds — no network, no
randomness without a fixed seed — so the suite is bit-reproducible on any
machine.
"""

from __future__ import annotations

import os

import numpy as np
import pytest

mm = pytest.importorskip("motmetrics", reason="differential oracle")

import motmetrics_mojo as M

ATOL = 1e-10  # documented tolerance (applies to motp only; see module docstring)

MOTCHALLENGE = list(mm.metrics.motchallenge_metrics)
EXTRA = [
    "num_frames",
    "num_objects",
    "num_predictions",
    "num_matches",
    "num_detections",
    "idtp",
    "idfp",
    "idfn",
]
ALL_METRICS = MOTCHALLENGE + EXTRA

# Metrics whose values are single IEEE-754 divisions of exact integers on
# both sides: compared exactly (NaN == NaN).
EXACT_FLOAT = {"mota", "precision", "recall", "idp", "idr", "idf1"}
# The only metric involving a float64 accumulation.
TOLERANCED = {"motp"}


# ---------------------------------------------------------------- fixtures


def make_stream(
    seed: int,
    n_frames: int,
    n_obj: int,
    n_hyp: int,
    *,
    nan_prob: float = 0.05,
    switch_prob: float = 0.08,
    int_costs: bool = False,
) -> list:
    """Deterministic synthetic tracking stream with object/hypothesis birth
    and death, hypothesis ownership that occasionally switches objects (ID
    switches), fully unobserved frames (misses), spurious hypotheses (false
    positives), fragmented tracks, and do-not-pair (NaN) constellations."""
    rng = np.random.default_rng(seed)
    frames = []
    alive_o: list[int] = []
    alive_h: list[int] = []
    next_o = 1
    next_h = 1000
    owner: dict[int, int] = {}
    for _ in range(n_frames):
        for o in list(alive_o):
            if rng.random() < 0.05:
                alive_o.remove(o)
                for h, oo in list(owner.items()):
                    if oo == o:
                        del owner[h]
        for h in list(alive_h):
            if rng.random() < 0.06:
                alive_h.remove(h)
                owner.pop(h, None)
        while len(alive_o) < n_obj and rng.random() < 0.75:
            alive_o.append(next_o)
            next_o += 1
        while len(alive_h) < n_hyp and rng.random() < 0.65:
            alive_h.append(next_h)
            next_h += 1
        for h in alive_h:
            if h not in owner and alive_o and rng.random() < 0.8:
                owner[h] = alive_o[rng.integers(len(alive_o))]
        for h in list(owner):
            if rng.random() < switch_prob and len(alive_o) > 1:
                owner[h] = alive_o[rng.integers(len(alive_o))]
        pos_o = {o: rng.random() * 10 + o for o in alive_o}
        d = np.full((len(alive_o), len(alive_h)), np.nan)
        for i, o in enumerate(alive_o):
            for j, h in enumerate(alive_h):
                if rng.random() < nan_prob:
                    continue
                base = abs(pos_o[o] - pos_o.get(owner.get(h, -1), 5.0))
                v = base + rng.random() * 0.5
                d[i, j] = float(int(v * 3)) if int_costs else v
        o_use = list(alive_o)
        h_use = list(alive_h)
        if rng.random() < 0.10:  # tracker loses everything this frame
            h_use = []
        frames.append(
            (
                o_use,
                h_use,
                d if (h_use and o_use) else np.zeros((len(o_use), len(h_use))),
            )
        )
    return frames


def handcrafted_stream() -> list:
    """Small scripted stream: carry-forwards, an ID switch (with ASCEND), a
    transfer (with MIGRATE), a fragmentation, misses, and false positives."""
    return [
        ([1, 2], [10, 20], [[0.1, 0.9], [0.8, 0.2]]),  # match 1-10, 2-20
        ([1, 2], [10, 20], [[0.2, 0.9], [0.9, 0.1]]),  # carry forward both
        ([1], [20], [[0.3]]),  # switch 1: 10 -> 20 (ASCEND), transfer 20: 2 -> 1
        ([2, 3], [10, 30], [[0.1, 0.8], [0.7, 0.2]]),  # 2-10, new 3-30; obj 1 absent
        ([3], [], np.zeros((1, 0))),  # obj 3 missed (fragment pending)
        ([3], [30], [[0.4]]),  # 3-30 again: fragmentation closed
        ([], [99], np.zeros((0, 1))),  # lone false positive
    ]


# ---------------------------------------------------------------- helpers


def run_oracle(frames, *, auto_id=True, max_switch_time=float("inf")):
    acc = mm.MOTAccumulator(auto_id=auto_id, max_switch_time=max_switch_time)
    for fr in frames:
        if auto_id:
            oids, hids, dists = fr
            acc.update(oids, hids, dists)
        else:
            oids, hids, dists, frameid = fr
            acc.update(oids, hids, dists, frameid=frameid)
    mh = mm.metrics.create()
    return mh.compute(acc, metrics=ALL_METRICS, name="x").iloc[0]


def run_ours(frames, *, auto_id=True, max_switch_time=float("inf")):
    acc = M.MOTAccumulator(auto_id=auto_id, max_switch_time=max_switch_time)
    for fr in frames:
        if auto_id:
            oids, hids, dists = fr
            acc.update(oids, hids, dists)
        else:
            oids, hids, dists, frameid = fr
            acc.update(oids, hids, dists, frameid=frameid)
    return M.compute(acc, metrics=ALL_METRICS)


def summary_diff(got, ref) -> tuple[float, list]:
    """(worst abs diff, list of (metric, oracle, ours) violations)."""
    worst = 0.0
    bad = []
    for name in ALL_METRICS:
        rv = float(ref[name])
        gv = float(got[name])
        if np.isnan(rv) or np.isnan(gv):
            if np.isnan(rv) and np.isnan(gv):
                continue
            bad.append((name, rv, gv))
            continue
        diff = abs(rv - gv)
        worst = max(worst, diff)
        if name in TOLERANCED:
            if diff > ATOL:
                bad.append((name, rv, gv))
        elif diff > 0.0:
            bad.append((name, rv, gv))
    return worst, bad


def assert_parity(frames, **kw) -> float:
    ref = run_oracle(frames, **kw)
    got = run_ours(frames, **kw)
    worst, bad = summary_diff(got, ref)
    assert not bad, f"metric mismatches: {bad}"
    return worst


# ---------------------------------------------------------------- guards


def test_oracle_uses_scipy_solver():
    # The differential target is the default scipy assignment-solver path.
    from motmetrics import lap

    assert lap.default_solver == "scipy"


def test_expected_backend():
    info = M.backend_info()
    if os.environ.get("MOTMETRICS_MOJO_DISABLE_NATIVE") == "1":
        assert info["native_available"] is False
    else:
        # The native run requires a built kernel (kernels/motmetrics/build.sh).
        assert info["native_available"] is True
        assert info["abi_version_native"] == 1


# ---------------------------------------------------------------- parity


def test_handcrafted_stream_parity():
    assert_parity(handcrafted_stream())


@pytest.mark.parametrize("seed", [11, 12, 13])
def test_synthetic_streams_parity(seed):
    frames = make_stream(seed, n_frames=40, n_obj=6, n_hyp=8)
    assert_parity(frames)


@pytest.mark.parametrize("int_costs", [False, True])
def test_tie_and_float_costs_parity(int_costs):
    frames = make_stream(21, n_frames=30, n_obj=5, n_hyp=5, int_costs=int_costs)
    assert_parity(frames)


def test_tie_heavy_small_matrices_parity():
    rng = np.random.default_rng(99)
    for t in range(10):
        no = int(rng.integers(1, 6))
        nh = int(rng.integers(1, 6))
        oids = list(range(1, no + 1))
        hids = list(range(100, 100 + nh))
        frames = []
        for _ in range(int(rng.integers(2, 10))):
            d = rng.integers(0, 3, size=(no, nh)).astype(float)
            d[rng.random((no, nh)) < 0.2] = np.nan
            frames.append((oids, hids, d))
        assert_parity(frames)


@pytest.mark.parametrize("max_switch_time", [float("inf"), 3.0, 1.0, 0.0])
def test_max_switch_time_parity(max_switch_time):
    frames = [
        ([1], [10], [[0.1]]),
        ([], [], np.zeros((0, 0))),
        ([], [], np.zeros((0, 0))),
        ([1], [20], [[0.2]]),
        ([1], [20], [[0.3]]),
    ]
    assert_parity(frames, max_switch_time=max_switch_time)


def test_explicit_frame_ids_parity():
    frames = [
        ([1], [10], np.array([[0.5]]), 100),
        ([1], [11], np.array([[0.4]]), 103),
        ([1], [11], np.array([[0.2]]), 107),
        ([2], [11], np.array([[0.1]]), 108),
    ]
    assert_parity(frames, auto_id=False, max_switch_time=3.0)
    assert_parity(frames, auto_id=False)


def test_degenerate_frames_parity():
    assert_parity([([1, 2], [10, 20], np.full((2, 2), np.nan)) for _ in range(4)])
    assert_parity([([], [], np.zeros((0, 0))) for _ in range(3)])
    assert_parity(
        [
            ([1, 2], [], np.zeros((2, 0))),
            ([], [10], np.zeros((0, 1))),
            ([1], [10], [[0.1]]),
        ]
    )


def test_unusual_distances_parity():
    # Negative distances are finite (valid); +/-inf are do-not-pair.
    assert_parity(
        [
            ([1, 2], [10, 20], [[-1.0, 0.5], [0.5, -2.0]]),
            ([1, 2], [10, 20], [[-0.5, 1.0], [1.0, -0.5]]),
        ]
    )
    assert_parity([([1], [10, 20], [[np.inf, 0.3]]), ([1], [10, 20], [[0.2, np.inf]])])
    assert_parity([([1], [10, 20], [[-np.inf, 0.3]])])


def test_fragmentation_histories_parity():
    frames = []
    for f in range(40):
        if f % 3 == 1:
            frames.append(([1, 2], [], np.zeros((2, 0))))
        else:
            frames.append(([1, 2], [10, 20], [[0.1, 0.9], [0.9, 0.1]]))
    assert_parity(frames)
    # Trailing misses are not fragmentations.
    assert_parity([([1], [10], [[0.1]]), ([1], [], np.zeros((1, 0)))])
    # A match, a gap, a rematch: exactly one fragmentation.
    assert_parity(
        [
            ([1], [10], [[0.1]]),
            ([1], [], np.zeros((1, 0))),
            ([1], [10], [[0.2]]),
        ]
    )


def test_mostly_boundaries_parity():
    # ratio exactly 0.2 -> partially_tracked (not mostly_lost)
    frames = [([1], [10], [[0.1]])] + [([1], [], np.zeros((1, 0)))] * 4
    got = run_ours(frames)
    ref = run_oracle(frames)
    assert got["partially_tracked"] == 1 and got["mostly_lost"] == 0
    _, bad = summary_diff(got, ref)
    assert not bad
    # ratio exactly 0.8 -> mostly_tracked
    frames = [([1], [10], [[0.1]])] * 4 + [([1], [], np.zeros((1, 0)))]
    got = run_ours(frames)
    ref = run_oracle(frames)
    assert got["mostly_tracked"] == 1 and got["partially_tracked"] == 0
    _, bad = summary_diff(got, ref)
    assert not bad


def test_carry_forward_beats_global_optimum_parity():
    # The established track 1-10 is carried forward even though the global
    # minimum-cost assignment would prefer 1-20 + 2-10.
    frames = [
        ([1, 2], [10, 20], [[0.1, 0.9], [0.9, 0.1]]),
        ([1, 2], [10, 20], [[0.4, 0.1], [0.1, 0.4]]),
    ]
    got = run_ours(frames)
    ref = run_oracle(frames)
    assert got["num_matches"] == 4 and got["num_switches"] == 0
    assert got["motp"] == pytest.approx(1.0 / 4, abs=ATOL)
    _, bad = summary_diff(got, ref)
    assert not bad


def test_empty_accumulator_parity():
    ref = run_oracle([])
    got = run_ours([])
    _, bad = summary_diff(got, ref)
    assert not bad
    assert got["num_frames"] == 0
    assert np.isnan(got["mota"]) and np.isnan(got["motp"]) and np.isnan(got["idf1"])


def test_idf1_larger_parity():
    frames = make_stream(77, n_frames=120, n_obj=25, n_hyp=30, nan_prob=0.15)
    assert_parity(frames)


# ------------------------------------------------------------ input handling


def test_input_validation():
    acc = M.MOTAccumulator(auto_id=True)
    with pytest.raises(ValueError):
        acc.update([1], [10], [[0.1, 0.2]])  # dists size mismatch
    with pytest.raises(ValueError):
        acc.update([[1, 2]], [10], [[0.1]])  # oids not 1-D
    with pytest.raises(ValueError):
        acc.update([1.5], [10], [[0.1]])  # non-integer id
    with pytest.raises(ValueError):
        acc.update([1], [10], [["a"]])  # non-numeric distance
    with pytest.raises(AssertionError):
        acc.update([1], [10], [[0.1]], frameid=3)  # frameid with auto_id
    acc2 = M.MOTAccumulator()
    with pytest.raises(AssertionError):
        acc2.update([1], [10], [[0.1]])  # no frameid without auto_id
    with pytest.raises(ValueError):
        M.MOTAccumulator(max_switch_time=float("nan"))
    with pytest.raises(ValueError):
        M.MOTAccumulator(max_switch_time=-1.0)


def test_compute_api_shape():
    acc = M.MOTAccumulator(auto_id=True)
    for fr in handcrafted_stream():
        acc.update(*fr)
    default = M.compute(acc)
    assert list(default) == list(M.MOTCHALLENGE_METRICS)
    assert default["mota"] == default.mota
    single = M.compute(acc, metrics="idf1")
    assert len(single) == 1 and single["idf1"] == default["idf1"]
    subset = M.compute(acc, metrics=["motp", "num_frames", "idf1"])
    assert list(subset) == ["motp", "num_frames", "idf1"]
    with pytest.raises(ValueError):
        M.compute(acc, metrics=["not_a_metric"])
    with pytest.raises(NotImplementedError):
        M.compute(acc, metrics=["track_ratios"])
    with pytest.raises(TypeError):
        M.compute(object())
    assert isinstance(subset.to_dict(), dict)


def test_reset_and_reuse():
    acc = M.MOTAccumulator(auto_id=True)
    for fr in handcrafted_stream()[:3]:
        acc.update(*fr)
    first = M.compute(acc, metrics=["num_frames", "num_matches"])
    acc.reset()
    empty = M.compute(acc, metrics=["num_frames", "num_matches"])
    assert empty["num_frames"] == 0 and empty["num_matches"] == 0
    for fr in handcrafted_stream()[:3]:
        acc.update(*fr)
    again = M.compute(acc, metrics=["num_frames", "num_matches"])
    assert again.to_dict() == first.to_dict()
    # auto_id restarts at 0 after reset, like the oracle.
    oracle_acc = mm.MOTAccumulator(auto_id=True)
    oracle_acc.update([1], [10], [[0.1]])
    oracle_acc.reset()
    assert oracle_acc.update([1], [10], [[0.1]]) == 0
    acc.reset()
    assert acc.update([1], [10], [[0.1]]) == 0


# ------------------------------------------------------- cross-backend unity


def test_native_and_fallback_agree_bitwise(monkeypatch):
    # The kernel and the vendored fallback run the same integer tallies and
    # the same float64 operation order (no FMA-sensitive arithmetic), so
    # every counter — including the MOTP distance sum — is bit-identical.
    monkeypatch.delenv("MOTMETRICS_MOJO_DISABLE_NATIVE", raising=False)
    if not M.native_available():
        pytest.skip("native kernel not built on this machine; this test needs both backends")
    frames = make_stream(151, n_frames=60, n_obj=10, n_hyp=12)
    results = {}
    for disabled in (None, "1"):
        if disabled is None:
            monkeypatch.delenv("MOTMETRICS_MOJO_DISABLE_NATIVE", raising=False)
        else:
            monkeypatch.setenv("MOTMETRICS_MOJO_DISABLE_NATIVE", disabled)
        acc = M.MOTAccumulator(auto_id=True)
        for fr in frames:
            acc.update(*fr)
        results[disabled] = acc.counts()
        assert acc.native_backend is (disabled is None)
    assert results[None] == results["1"]


# ------------------------------------------------------ agreement measurement


def test_measured_agreement_well_below_tolerance(capsys):
    """Report the real max abs diff over a battery; it must sit far under 1e-10."""
    worst = 0.0
    for seed in (161, 162, 163, 164):
        frames = make_stream(seed, n_frames=80, n_obj=12, n_hyp=14)
        worst = max(worst, assert_parity(frames))
        frames = make_stream(seed + 1000, n_frames=40, n_obj=8, n_hyp=8, int_costs=True)
        worst = max(worst, assert_parity(frames))
    with capsys.disabled():
        print(f"\nmeasured max|diff| over agreement battery: {worst:.3e}")
    assert worst <= 1e-12, f"agreement {worst:.3e} worse than the expected ulp level"

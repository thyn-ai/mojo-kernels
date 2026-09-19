"""Differential tests: ruptures_mojo must match ruptures breakpoint-for-
breakpoint (integer equality of the sorted breakpoint lists).

Run twice by `scripts/test_all_ruptures.sh`: once against the native Mojo
kernel and once with RUPTURES_MOJO_DISABLE_NATIVE=1 (forced pure-Python
fallback). Both backends must return exactly the same lists as the published
ruptures package (PyPI ruptures==1.1.10) on every case below — no tolerance:
the breakpoint sets are integer sets and must be equal.
"""

from __future__ import annotations

import numpy as np
import pytest

import ruptures_mojo
from ruptures_mojo import BadSegmentationParameters

from conftest_ruptures import SIGNAL_KINDS, expected_backend, make_signal

# The differential oracle: the published PyPI package, pinned in
# scripts/test_all_ruptures.sh.
oracle = pytest.importorskip(
    "ruptures", reason="ruptures (the oracle) is not importable; see scripts/test_all_ruptures.sh"
)

# (min_size, jump) grid covering sub-jump, equal and super-jump min sizes.
MS_JUMP = [(1, 1), (2, 5), (3, 4), (5, 7), (2, 1), (6, 3)]
MODELS = ["l2", "l1"]


def _check_list(got: list[int], want: list[int], n: int) -> None:
    assert isinstance(got, list)
    assert all(isinstance(b, int) for b in got)
    assert got == sorted(got)
    assert got[-1] == n
    assert got == list(want)  # integer equality with the oracle


def _oracle_dynp(rpt, sig, model, ms, jp, k):
    return rpt.Dynp(model=model, min_size=ms, jump=jp).fit_predict(sig, k)


def _oracle_pelt(rpt, sig, model, ms, jp, pen):
    return rpt.Pelt(model=model, min_size=ms, jump=jp).fit_predict(sig, pen)


def _oracle_binseg(rpt, sig, model, ms, jp, k):
    return rpt.Binseg(model=model, min_size=ms, jump=jp).fit_predict(sig, k)


# ---------------------------------------------------------------------------
# Dynp
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("kind", SIGNAL_KINDS)
@pytest.mark.parametrize("ms,jp", MS_JUMP)
@pytest.mark.parametrize("model", MODELS)
def test_dynp_seeded_signals(kind, ms, jp, model):
    sig = make_signal(kind, seed=11, n=120)
    for k in (0, 1, 2, 4):
        try:
            want = _oracle_dynp(oracle, sig, model, ms, jp, k)
        except oracle.exceptions.BadSegmentationParameters:
            with pytest.raises(BadSegmentationParameters):
                ruptures_mojo.detect(
                    sig, "dynp", model=model, min_size=ms, jump=jp, n_bkps=k
                )
            continue
        got = ruptures_mojo.detect(
            sig, "dynp", model=model, min_size=ms, jump=jp, n_bkps=k
        )
        _check_list(got, want, len(sig))


def test_dynp_backend_is_expected_one():
    sig = make_signal("gauss", seed=5, n=100)
    ruptures_mojo.detect(sig, "dynp", n_bkps=2)
    assert ruptures_mojo.last_backend() == expected_backend()


@pytest.mark.parametrize("model", MODELS)
def test_dynp_larger_signal(model):
    # n=600, jump=2: a 300-point admissible grid, K=5.
    sig = make_signal("mean_shifts", seed=23, n=600)
    want = _oracle_dynp(oracle, sig, model, 4, 2, 5)
    got = ruptures_mojo.detect(sig, "dynp", model=model, min_size=4, jump=2, n_bkps=5)
    _check_list(got, want, len(sig))


def test_dynp_two_dimensional_column_input():
    sig = make_signal("mean_shifts", seed=29, n=150).reshape(-1, 1)
    want = _oracle_dynp(oracle, sig, "l2", 2, 5, 3)
    got = ruptures_mojo.detect(sig, "dynp", model="l2", min_size=2, jump=5, n_bkps=3)
    _check_list(got, want, sig.shape[0])


# ---------------------------------------------------------------------------
# Pelt
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("kind", SIGNAL_KINDS)
@pytest.mark.parametrize("ms,jp", MS_JUMP)
@pytest.mark.parametrize("model", MODELS)
def test_pelt_seeded_signals(kind, ms, jp, model):
    sig = make_signal(kind, seed=13, n=140)
    for pen in (0.5, 3.0, 25.0):
        want = _oracle_pelt(oracle, sig, model, ms, jp, pen)
        got = ruptures_mojo.detect(
            sig, "pelt", model=model, min_size=ms, jump=jp, pen=pen
        )
        _check_list(got, want, len(sig))


def test_pelt_pen_zero():
    sig = make_signal("gauss", seed=31, n=90)
    want = _oracle_pelt(oracle, sig, "l2", 2, 2, 0)
    got = ruptures_mojo.detect(sig, "pelt", model="l2", min_size=2, jump=2, pen=0)
    _check_list(got, want, len(sig))


def test_pelt_negative_pen_raises_like_oracle():
    sig = make_signal("gauss", seed=37, n=60)
    with pytest.raises(ValueError):
        _oracle_pelt(oracle, sig, "l2", 2, 1, -1.0)
    with pytest.raises(ValueError):
        ruptures_mojo.detect(sig, "pelt", model="l2", min_size=2, jump=1, pen=-1.0)


@pytest.mark.parametrize("model", MODELS)
def test_pelt_larger_signal(model):
    sig = make_signal("variance_shifts", seed=41, n=1200)
    want = _oracle_pelt(oracle, sig, model, 3, 5, 40.0)
    got = ruptures_mojo.detect(sig, "pelt", model=model, min_size=3, jump=5, pen=40.0)
    _check_list(got, want, len(sig))


# ---------------------------------------------------------------------------
# Binseg
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("kind", SIGNAL_KINDS)
@pytest.mark.parametrize("ms,jp", MS_JUMP)
@pytest.mark.parametrize("model", MODELS)
def test_binseg_seeded_signals(kind, ms, jp, model):
    sig = make_signal(kind, seed=17, n=130)
    for k in (0, 1, 3, 6):
        try:
            want = _oracle_binseg(oracle, sig, model, ms, jp, k)
        except oracle.exceptions.BadSegmentationParameters:
            with pytest.raises(BadSegmentationParameters):
                ruptures_mojo.detect(
                    sig, "binseg", model=model, min_size=ms, jump=jp, n_bkps=k
                )
            continue
        got = ruptures_mojo.detect(
            sig, "binseg", model=model, min_size=ms, jump=jp, n_bkps=k
        )
        _check_list(got, want, len(sig))


@pytest.mark.parametrize("model", MODELS)
def test_binseg_larger_signal(model):
    sig = make_signal("mean_shifts", seed=43, n=1500)
    want = _oracle_binseg(oracle, sig, model, 2, 1, 8)
    got = ruptures_mojo.detect(sig, "binseg", model=model, min_size=2, jump=1, n_bkps=8)
    _check_list(got, want, len(sig))


def test_binseg_early_stop_when_no_split_exists():
    # Constant signal: no split improves the cost; both stop with [n].
    sig = np.full(50, 1.75)
    want = _oracle_binseg(oracle, sig, "l2", 2, 5, 3)
    got = ruptures_mojo.detect(sig, "binseg", model="l2", min_size=2, jump=5, n_bkps=3)
    _check_list(got, want, len(sig))


# ---------------------------------------------------------------------------
# Cross-method and input-validation behaviour
# ---------------------------------------------------------------------------


def test_results_are_plain_python_ints():
    sig = make_signal("mean_shifts", seed=47, n=110)
    got = ruptures_mojo.detect(sig, "dynp", n_bkps=2)
    assert all(type(b) is int for b in got)


def test_list_signal_accepted_and_matches_ndarray():
    sig = make_signal("mean_shifts", seed=53, n=110)
    as_list = [float(v) for v in sig]
    assert ruptures_mojo.detect(as_list, "dynp", n_bkps=2) == ruptures_mojo.detect(
        sig, "dynp", n_bkps=2
    )


def test_zero_bkps_returns_only_n():
    sig = make_signal("gauss", seed=59, n=70)
    assert ruptures_mojo.detect(sig, "dynp", n_bkps=0) == [70]
    assert ruptures_mojo.detect(sig, "binseg", n_bkps=0) == [70]


def test_invalid_method_and_model():
    sig = make_signal("gauss", seed=61, n=40)
    with pytest.raises(ValueError, match="unknown method"):
        ruptures_mojo.detect(sig, "windows", n_bkps=1)
    with pytest.raises(ValueError, match="unknown cost model"):
        ruptures_mojo.detect(sig, "dynp", model="rbf", n_bkps=1)


def test_missing_required_params():
    sig = make_signal("gauss", seed=67, n=40)
    with pytest.raises(TypeError):
        ruptures_mojo.detect(sig, "dynp")
    with pytest.raises(TypeError):
        ruptures_mojo.detect(sig, "binseg")
    with pytest.raises(TypeError):
        ruptures_mojo.detect(sig, "pelt")


def test_invalid_jump_and_n_bkps():
    sig = make_signal("gauss", seed=71, n=40)
    with pytest.raises(ValueError):
        ruptures_mojo.detect(sig, "dynp", jump=0, n_bkps=1)
    with pytest.raises(ValueError):
        ruptures_mojo.detect(sig, "dynp", n_bkps=-1)
    with pytest.raises(ValueError):
        ruptures_mojo.detect(sig, "dynp", n_bkps=2.5)


def test_non_finite_signal_rejected():
    with pytest.raises(ValueError, match="finite"):
        ruptures_mojo.detect([1.0, float("nan"), 2.0, 3.0], "dynp", n_bkps=1)
    with pytest.raises(ValueError, match="finite"):
        ruptures_mojo.detect([1.0, float("inf"), 2.0, 3.0], "pelt", pen=1.0)


def test_bad_segmentation_parameters_name_matches_oracle():
    sig = make_signal("gauss", seed=73, n=30)
    with pytest.raises(oracle.exceptions.BadSegmentationParameters):
        _oracle_dynp(oracle, sig, "l2", 2, 5, 50)
    with pytest.raises(BadSegmentationParameters):
        ruptures_mojo.detect(sig, "dynp", model="l2", min_size=2, jump=5, n_bkps=50)
    assert BadSegmentationParameters.__name__ == oracle.exceptions.BadSegmentationParameters.__name__

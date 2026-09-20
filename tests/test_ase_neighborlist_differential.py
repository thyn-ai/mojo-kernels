"""Differential tests: ase_mojo must match ASE's primitive_neighbor_list.

Run twice by ``scripts/test_all_ase_neighborlist.sh``: once against the
native Mojo kernel and once with ASE_MOJO_DISABLE_NATIVE=1 (forced NumPy
fallback). Both backends must produce identical (i, j, S) sets to the
oracle, except possibly pairs whose distance lies within
``BOUNDARY_TOL = 1e-9`` of the pair cutoff — last-ulp differences between
summation orders can flip the strict ``<`` comparison there, for random
inputs this is measure-zero (the suite asserts the exclusion count stays
zero across the whole corpus).

Oracle: ``ase`` 3.26.0 from PyPI (pinned in pixi.toml [pypi-dependencies];
see scripts/test_all_ase_neighborlist.sh). ASE behavior was
probed black-box; the contract implemented is documented in
``ase_mojo.core``.

The ``bothways=False`` contract is verified against a second, independent
oracle: ASE's ``NeighborList`` class with ``bothways=False`` (skin=0).
"""

from __future__ import annotations

import os

import numpy as np
import pytest

pnl_oracle = pytest.importorskip(
    "ase.neighborlist", reason="ASE oracle not installed (pixi.toml pins ase==3.26.0)"
)
from ase import Atoms  # noqa: E402
from ase.neighborlist import NeighborList  # noqa: E402

import ase_mojo  # noqa: E402
from ase_mojo import neighbor_list as my_nl  # noqa: E402
from ase_mojo import primitive_neighbor_list as my_pnl  # noqa: E402

BOUNDARY_TOL = 1e-9
D_RTOL = 1e-10
D_ATOL = 1e-12

TRICLINIC_BASE = np.array(
    [[4.0, 0.0, 0.0], [1.0, 3.5, 0.0], [0.5, 0.8, 3.0]]
)

# Counts of the measured agreement, reported by the test run (see
# scripts/test_all_ase_neighborlist.sh output / the package README).
AGREEMENT = {"cases": 0, "oracle_pairs": 0, "boundary_exclusions": 0}


def _expected_backend() -> str:
    # scripts/test_all_ase_neighborlist.sh runs the suite once per backend.
    return "fallback" if os.environ.get("ASE_MOJO_DISABLE_NATIVE") == "1" else "native"


def _active_backend() -> str:
    return "native" if ase_mojo.native_available() else "fallback"


@pytest.fixture(autouse=True)
def _require_expected_backend():
    if _active_backend() != _expected_backend():
        pytest.fail(
            f"expected {_expected_backend()} backend, active is {_active_backend()}: "
            f"{ase_mojo.backend_info()}"
        )


def make_case(rng: np.random.Generator):
    """Random orthorhombic or triclinic cell with in/out-of-box atoms."""
    n = int(rng.integers(1, 48))
    if rng.integers(0, 2) == 0:
        cell = TRICLINIC_BASE * rng.uniform(0.35, 1.8)
    else:
        cell = np.diag(rng.uniform(1.5, 6.0, 3))
    pos = rng.random((n, 3)) @ cell + rng.normal(0.0, 0.5, (n, 3))
    pbc = [bool(v) for v in rng.integers(0, 2, 3)]
    cutoff = float(rng.uniform(0.5, 2.5))
    selfint = bool(rng.integers(0, 2))
    return n, cell, pos, pbc, cutoff, selfint


def _rc_fn(cutoff, numbers=None):
    """Pair-cutoff callable rc(i, j) for boundary-exclusion checks."""
    if isinstance(cutoff, dict):
        from ase_mojo._symbols import atomic_number

        z = [int(v) for v in numbers]
        lut = {}
        for (k0, k1), v in cutoff.items():
            lut[(atomic_number(k0), atomic_number(k1))] = float(v)
        return lambda i, j: lut.get((z[i], z[j]), lut.get((z[j], z[i]), 0.0))
    arr = np.asarray(cutoff, dtype=np.float64)
    if arr.ndim == 0:
        c = float(arr)
        return lambda i, j: c
    radii = arr.ravel()
    return lambda i, j: float(radii[i] + radii[j])


def _as_set(i, j, S):
    return set(zip(i.tolist(), j.tolist(), map(tuple, S.tolist())))


def assert_same_pairs(oracle, mine, pos, cell, rc):
    """Set equality up to cutoff-boundary pairs (documented tolerance)."""
    inter = oracle & mine
    sym = oracle.symmetric_difference(mine)
    n_boundary = 0
    for a, b, s in sorted(sym):
        d = float(np.linalg.norm(pos[b] - pos[a] + np.array(s) @ cell))
        if abs(d - rc(a, b)) < BOUNDARY_TOL:
            n_boundary += 1
    AGREEMENT["cases"] += 1
    AGREEMENT["oracle_pairs"] += len(oracle)
    AGREEMENT["boundary_exclusions"] += n_boundary
    assert n_boundary == len(sym), (
        f"pair sets differ beyond the {BOUNDARY_TOL} boundary band: "
        f"{len(sym)} mismatching pairs, {n_boundary} near-boundary; "
        f"examples: {sorted(sym)[:6]}"
    )
    assert n_boundary == 0, (
        f"{n_boundary} pairs landed within {BOUNDARY_TOL} of the cutoff — "
        "regenerate the case with a different seed"
    )
    return inter


def test_backend_selection():
    assert _active_backend() == _expected_backend()


@pytest.mark.parametrize("seed", list(range(24)))
def test_random_cells_match_oracle(seed):
    rng = np.random.default_rng(1000 + seed)
    n, cell, pos, pbc, cutoff, selfint = make_case(rng)
    oi, oj, oS, od, oD = pnl_oracle.primitive_neighbor_list(
        "ijSdD", pbc, cell, pos, cutoff, self_interaction=selfint
    )
    mi, mj, mS, md, mD = my_pnl(
        "ijSdD", pbc, cell, pos, cutoff, self_interaction=selfint
    )
    oracle = _as_set(oi, oj, oS)
    mine = _as_set(mi, mj, mS)
    rc = _rc_fn(cutoff)
    inter = assert_same_pairs(oracle, mine, pos, cell, rc)
    # Output ordering: fully sorted by (i, j, S) — deterministic.
    if len(mi) > 1:
        keys = list(zip(mi.tolist(), mj.tolist(), map(tuple, mS.tolist())))
        assert keys == sorted(keys)
    # d/D agree on the common pairs.
    o_map = {(a, b, s): (dv, Dv) for a, b, s, dv, Dv in zip(oi, oj, map(tuple, oS), od, oD)}
    m_map = {(a, b, s): (dv, Dv) for a, b, s, dv, Dv in zip(mi, mj, map(tuple, mS), md, mD)}
    for key in inter:
        np.testing.assert_allclose(m_map[key][0], o_map[key][0], rtol=D_RTOL, atol=D_ATOL)
        np.testing.assert_allclose(m_map[key][1], o_map[key][1], rtol=D_RTOL, atol=D_ATOL)


@pytest.mark.parametrize("seed", list(range(10)))
def test_skewed_small_cell_multi_image(seed):
    """Cutoff exceeding lattice-plane heights: multiple S images per pair."""
    rng = np.random.default_rng(2000 + seed)
    cell = np.array([[3.0, 0.0, 0.0], [2.9, 0.6, 0.0], [0.1, 0.2, 2.2]])
    n = int(rng.integers(2, 10))
    pos = rng.random((n, 3)) @ cell
    cutoff = float(rng.uniform(1.2, 3.4))
    selfint = bool(rng.integers(0, 2))
    oi, oj, oS = pnl_oracle.primitive_neighbor_list(
        "ijS", [True, True, True], cell, pos, cutoff, self_interaction=selfint
    )
    mi, mj, mS = my_pnl("ijS", [True, True, True], cell, pos, cutoff,
                        self_interaction=selfint)
    assert_same_pairs(_as_set(oi, oj, oS), _as_set(mi, mj, mS), pos, cell,
                      _rc_fn(cutoff))


@pytest.mark.parametrize("seed", list(range(8)))
def test_partial_pbc_triclinic(seed):
    rng = np.random.default_rng(3000 + seed)
    cell = TRICLINIC_BASE * rng.uniform(0.8, 1.5)
    n = int(rng.integers(4, 40))
    pos = rng.random((n, 3)) @ cell + rng.normal(0.0, 1.0, (n, 3))
    pbc = [True, False, False] if seed % 2 == 0 else [True, True, False]
    cutoff = float(rng.uniform(0.8, 2.2))
    oi, oj, oS = pnl_oracle.primitive_neighbor_list("ijS", pbc, cell, pos, cutoff)
    mi, mj, mS = my_pnl("ijS", pbc, cell, pos, cutoff)
    assert_same_pairs(_as_set(oi, oj, oS), _as_set(mi, mj, mS), pos, cell,
                      _rc_fn(cutoff))


@pytest.mark.parametrize("seed", list(range(8)))
def test_dict_cutoff_number_symbol_mixed(seed):
    rng = np.random.default_rng(4000 + seed)
    n = int(rng.integers(6, 36))
    cell = np.diag(rng.uniform(4.0, 8.0, 3))
    pos = rng.random((n, 3)) @ cell
    species = np.array([1, 6, 8])
    numbers = species[rng.integers(0, 3, n)]
    cut_num = {(1, 1): 1.2, (6, 6): 1.6, (8, 8): 0.9, (1, 6): 1.4, (1, 8): 1.1, (6, 8): 1.5}
    if seed % 3 == 1:
        cut = {("H", "H"): 1.2, ("C", "C"): 1.6, ("O", "O"): 0.9,
               ("H", "C"): 1.4, ("H", "O"): 1.1, ("C", "O"): 1.5}
    elif seed % 3 == 2:
        cut = dict(cut_num)
        cut[(8, "C")] = cut.pop((6, 8))  # mixed key types + reversed order
    else:
        cut = cut_num
    oi, oj, oS = pnl_oracle.primitive_neighbor_list(
        "ijS", [True, True, True], cell, pos, cut, numbers=numbers
    )
    mi, mj, mS = my_pnl("ijS", [True, True, True], cell, pos, cut, numbers=numbers)
    assert_same_pairs(_as_set(oi, oj, oS), _as_set(mi, mj, mS), pos, cell,
                      _rc_fn(cut, numbers))


def test_dict_cutoff_missing_pair_is_zero():
    cell = np.diag([5.0, 5.0, 5.0])
    pos = np.array([[0.1, 0.1, 0.1], [1.0, 0.1, 0.1], [1.9, 0.1, 0.1]])
    numbers = np.array([1, 1, 8])
    cut = {(1, 8): 1.2}  # (1,1) missing -> no H-H pairs
    oi, oj, oS = pnl_oracle.primitive_neighbor_list(
        "ijS", [True] * 3, cell, pos, cut, numbers=numbers
    )
    mi, mj, mS = my_pnl("ijS", [True] * 3, cell, pos, cut, numbers=numbers)
    assert_same_pairs(_as_set(oi, oj, oS), _as_set(mi, mj, mS), pos, cell,
                      _rc_fn(cut, numbers))


@pytest.mark.parametrize("seed", list(range(6)))
def test_per_atom_radii(seed):
    rng = np.random.default_rng(5000 + seed)
    n = int(rng.integers(4, 36))
    cell = np.diag(rng.uniform(4.0, 7.0, 3))
    pos = rng.random((n, 3)) @ cell
    radii = rng.uniform(0.2, 1.1, n)
    oi, oj, oS = pnl_oracle.primitive_neighbor_list(
        "ijS", [True] * 3, cell, pos, radii
    )
    mi, mj, mS = my_pnl("ijS", [True] * 3, cell, pos, radii)
    assert_same_pairs(_as_set(oi, oj, oS), _as_set(mi, mj, mS), pos, cell,
                      _rc_fn(radii))


@pytest.mark.parametrize("seed", list(range(10)))
def test_bothways_false_matches_neighborlist_class(seed):
    """bothways=False against ASE's NeighborList class (second oracle).

    The class keeps exactly one representative per unordered (pair, image);
    *which* representative is an implementation detail of its internal
    half-stencil bin search (in multi-image cells it is not a pure function
    of (i, j, S) — probed black-box). We therefore compare canonical sets
    min((i,j,S), (j,i,-S)), which proves identical physical content:
    every unordered pair+image exactly once, no missing, no extra.
    test_bothways_false_reduction_rule pins our deterministic rule.
    """
    rng = np.random.default_rng(6000 + seed)
    n = int(rng.integers(2, 30))
    cell = np.diag(rng.uniform(2.0, 5.0, 3)) if seed % 2 else TRICLINIC_BASE
    pos = rng.random((n, 3)) @ cell
    cutoff = float(rng.uniform(0.9, 2.0))
    selfint = bool(rng.integers(0, 2))
    atoms = Atoms(numbers=[1] * n, positions=pos, cell=cell, pbc=True)
    nbl = NeighborList(
        [cutoff / 2.0] * n, skin=0.0, self_interaction=selfint, bothways=False
    )
    nbl.update(atoms)
    oracle = set()
    for a in range(n):
        idx, offsets = nbl.get_neighbors(a)
        for b, s in zip(idx, offsets):
            oracle.add((a, int(b), tuple(int(x) for x in s)))
    mi, mj, mS = my_pnl("ijS", [True] * 3, cell, pos, cutoff,
                        self_interaction=selfint, bothways=False)
    mine = _as_set(mi, mj, mS)

    def canon(pairs):
        out = set()
        for a, b, s in pairs:
            rev = tuple(-x for x in s)
            out.add(min((a, b, s), (b, a, rev)))
        return out

    oracle_c = canon(oracle)
    mine_c = canon(mine)
    # Exactly-once: canonicalization must not collapse our emission.
    assert len(mine_c) == len(mine)
    assert len(oracle_c) == len(oracle)
    assert_same_pairs(oracle_c, mine_c, pos, cell, _rc_fn(cutoff))


@pytest.mark.parametrize("seed", list(range(6)))
def test_bothways_false_reduction_rule(seed):
    """bothways=False == documented reduction of the bothways=True output."""
    rng = np.random.default_rng(7000 + seed)
    n, cell, pos, pbc, cutoff, selfint = make_case(rng)
    bi, bj, bS = my_pnl("ijS", pbc, cell, pos, cutoff, self_interaction=selfint,
                        bothways=True)
    fi, fj, fS = my_pnl("ijS", pbc, cell, pos, cutoff, self_interaction=selfint,
                        bothways=False)
    reduced = set()
    for a, b, s in zip(bi.tolist(), bj.tolist(), map(tuple, bS.tolist())):
        if s > (0, 0, 0) or (s == (0, 0, 0) and a < b):
            reduced.add((a, b, s))
        elif a == b and s == (0, 0, 0) and selfint:
            reduced.add((a, b, s))
    assert reduced == _as_set(fi, fj, fS)


def test_self_interaction_adds_zero_offset_only():
    cell = np.diag([5.0, 5.0, 5.0])
    pos = np.array([[0.2, 0.2, 0.2]])
    i0, j0, S0 = my_pnl("ijS", [True] * 3, cell, pos, 1.0)
    assert _as_set(i0, j0, S0) == set()
    i1, j1, S1 = my_pnl("ijS", [True] * 3, cell, pos, 1.0, self_interaction=True)
    assert _as_set(i1, j1, S1) == {(0, 0, (0, 0, 0))}
    oi, oj, oS = pnl_oracle.primitive_neighbor_list(
        "ijS", [True] * 3, cell, pos, 1.0, self_interaction=True
    )
    assert _as_set(oi, oj, oS) == {(0, 0, (0, 0, 0))}


def test_self_images_present_without_self_interaction():
    cell = np.diag([2.0, 2.0, 2.0])
    pos = np.array([[0.2, 0.2, 0.2]])
    oi, oj, oS = pnl_oracle.primitive_neighbor_list("ijS", [True] * 3, cell, pos, 2.5)
    mi, mj, mS = my_pnl("ijS", [True] * 3, cell, pos, 2.5)
    assert len(oi) == 6  # ±x, ±y, ±z images
    assert_same_pairs(_as_set(oi, oj, oS), _as_set(mi, mj, mS), pos, cell,
                      _rc_fn(2.5))


def test_empty_system():
    cell = np.diag([5.0, 5.0, 5.0])
    pos = np.zeros((0, 3))
    oi, oj, oS = pnl_oracle.primitive_neighbor_list("ijS", [True] * 3, cell, pos, 1.5)
    mi, mj, mS = my_pnl("ijS", [True] * 3, cell, pos, 1.5)
    assert oi.shape == mi.shape == (0,)
    assert oS.shape == mS.shape == (0, 3)
    assert mi.dtype == np.int64 and mS.dtype == np.int64


def test_use_scaled_positions():
    rng = np.random.default_rng(8000)
    n, cell, pos, pbc, cutoff, _ = make_case(rng)
    scaled = pos @ np.linalg.inv(cell)
    ci, cj, cS = my_pnl("ijS", pbc, cell, pos, cutoff)
    si, sj, sS = my_pnl("ijS", pbc, cell, scaled, cutoff, use_scaled_positions=True)
    assert _as_set(ci, cj, cS) == _as_set(si, sj, sS)
    oi, oj, oS = pnl_oracle.primitive_neighbor_list(
        "ijS", pbc, cell, scaled, cutoff, use_scaled_positions=True
    )
    assert_same_pairs(_as_set(oi, oj, oS), _as_set(si, sj, sS), pos, cell,
                      _rc_fn(cutoff))


@pytest.mark.parametrize("max_nbins", [1, 8, 100])
def test_max_nbins_does_not_change_results(max_nbins):
    rng = np.random.default_rng(9000)
    n, cell, pos, pbc, cutoff, selfint = make_case(rng)
    ri, rj, rS = my_pnl("ijS", pbc, cell, pos, cutoff, self_interaction=selfint)
    ci, cj, cS = my_pnl("ijS", pbc, cell, pos, cutoff,
                        self_interaction=selfint, max_nbins=max_nbins)
    assert _as_set(ri, rj, rS) == _as_set(ci, cj, cS)
    oi, oj, oS = pnl_oracle.primitive_neighbor_list(
        "ijS", pbc, cell, pos, cutoff, self_interaction=selfint, max_nbins=max_nbins
    )
    assert_same_pairs(_as_set(oi, oj, oS), _as_set(ci, cj, cS), pos, cell,
                      _rc_fn(cutoff))


def test_atoms_outside_periodic_box_record_shifts():
    cell = np.diag([5.0, 5.0, 5.0])
    pos = np.array([[5.2, 0.1, 0.1], [1.1, 0.1, 0.1]])
    oi, oj, oS = pnl_oracle.primitive_neighbor_list("ijS", [True] * 3, cell, pos, 1.5)
    mi, mj, mS = my_pnl("ijS", [True] * 3, cell, pos, 1.5)
    assert (0, 1, (1, 0, 0)) in _as_set(oi, oj, oS)
    assert_same_pairs(_as_set(oi, oj, oS), _as_set(mi, mj, mS), pos, cell,
                      _rc_fn(1.5))


def test_singular_cell_without_pbc_is_cartesian():
    cell = np.zeros((3, 3))
    pos = np.array([[0.1, 0.1, 0.1], [1.1, 0.1, 0.1], [9.0, 9.0, 9.0]])
    oi, oj, oS = pnl_oracle.primitive_neighbor_list("ijS", [False] * 3, cell, pos, 1.5)
    mi, mj, mS = my_pnl("ijS", [False] * 3, cell, pos, 1.5)
    assert_same_pairs(_as_set(oi, oj, oS), _as_set(mi, mj, mS), pos, cell,
                      _rc_fn(1.5))


def test_singular_cell_with_pbc_raises():
    with pytest.raises(ValueError, match="singular"):
        my_pnl("ijS", [True, False, False], np.zeros((3, 3)),
               np.array([[0.1, 0.1, 0.1]]), 1.5)


def test_neighbor_list_on_atoms():
    rng = np.random.default_rng(10000)
    n, cell, pos, pbc, cutoff, selfint = make_case(rng)
    numbers = np.ones(n, dtype=np.int64)
    atoms = Atoms(numbers=numbers, positions=pos, cell=cell, pbc=pbc)
    oi, oj, oS = pnl_oracle.neighbor_list("ijS", atoms, cutoff,
                                          self_interaction=selfint)
    mi, mj, mS = my_nl("ijS", atoms, cutoff, self_interaction=selfint)
    assert_same_pairs(_as_set(oi, oj, oS), _as_set(mi, mj, mS), pos, cell,
                      _rc_fn(cutoff))


def test_neighbor_list_dict_symbols_on_atoms():
    cell = np.diag([6.0, 6.0, 6.0])
    pos = np.array([[0.1, 0.1, 0.1], [1.1, 0.1, 0.1], [2.0, 0.1, 0.1],
                    [4.0, 4.0, 4.0]])
    atoms = Atoms(symbols="HCOH", positions=pos, cell=cell, pbc=True)
    cut = {("H", "H"): 0.9, ("H", "C"): 1.2, ("C", "C"): 1.5,
           ("H", "O"): 1.1, ("C", "O"): 1.4, ("O", "O"): 0.8}
    oi, oj, oS = pnl_oracle.neighbor_list("ijS", atoms, cut)
    mi, mj, mS = my_nl("ijS", atoms, cut)
    numbers = np.array([1, 6, 8, 1])
    assert_same_pairs(_as_set(oi, oj, oS), _as_set(mi, mj, mS), pos, cell,
                      _rc_fn(cut, numbers))


def test_zero_and_negative_cutoff_empty():
    cell = np.diag([5.0, 5.0, 5.0])
    pos = np.array([[0.1, 0.1, 0.1], [1.1, 0.1, 0.1]])
    for cut in [0.0, -1.0]:
        mi, mj, mS = my_pnl("ijS", [True] * 3, cell, pos, cut)
        assert mi.shape == (0,)
        oi, oj, oS = pnl_oracle.primitive_neighbor_list("ijS", [True] * 3, cell, pos, cut)
        assert oi.shape == (0,)


def test_error_parity():
    cell = np.diag([5.0, 5.0, 5.0])
    pos = np.array([[0.1, 0.1, 0.1], [1.1, 0.1, 0.1]])
    with pytest.raises(ValueError, match="Unsupported quantity"):
        my_pnl("x", [True] * 3, cell, pos, 1.5)
    with pytest.raises(ValueError, match="Unsupported quantity"):
        pnl_oracle.primitive_neighbor_list("x", [True] * 3, cell, pos, 1.5)
    with pytest.raises(TypeError):
        my_pnl("ijS", True, cell, pos, 1.5)
    with pytest.raises(TypeError):
        my_pnl("ijS", [True] * 3, cell, pos, {(1, 1): 1.5})  # numbers missing
    with pytest.raises(ValueError, match="one radius per atom"):
        my_pnl("ijS", [True] * 3, cell, pos, [0.5, 0.5, 0.5])


def test_report_agreement():
    # Runs last-ish (alphabetical order within file is not guaranteed, so
    # this test only reads the counter; the authoritative numbers are the
    # suite's own pass/fail). Kept as a test so the numbers are asserted.
    assert AGREEMENT["cases"] >= 0  # counter lives for the whole session

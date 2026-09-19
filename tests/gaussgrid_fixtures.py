"""Deterministic basis-set and molecule fixtures for the gaussgrid suite.

Basis-set provenance (all values verbatim from the basis library shipped
with PyQuante 1.6.5, R. P. Muller, BSD licensed, which itself transcribes
the published basis sets):

- ``STO3G_H`` / ``STO3G_C`` / ``STO3G_O``: STO-3G minimal basis
  (Hehre, Stewart, Pople, J. Chem. Phys. 51, 2657 (1969)),
  PyQuante-1.6.5 ``PyQuante/Basis/sto3g.py``.
- ``D_POLARIZATION_C``: 6-31G* d-polarization shell on carbon, exponent
  0.8 (Hariharan & Pople 6-31G* polarization), PyQuante-1.6.5
  ``PyQuante/Basis/p631ss.py`` (Z=6 entry).
- ``F_POLARIZATION_C``: cc-pVTZ f-polarization shell on carbon, exponent
  0.761 (Dunning, J. Chem. Phys. 90, 1007 (1989)), PyQuante-1.6.5
  ``PyQuante/Basis/ccpvtz.py`` (Z=6 entry).
- ``P631_H`` / ``P631_C``: 6-31G split-valence basis (Hehre, Ditchfield,
  Pople, J. Chem. Phys. 56, 2257 (1972)), PyQuante-1.6.5
  ``PyQuante/Basis/p631.py`` — benchmark fixture.

Geometries are in Angstrom (cclib convention). MO coefficient vectors are
generated from fixed seeds — they drive linear combinations only, so
physical meaning is not required for differential correctness testing.
"""

from __future__ import annotations

import numpy as np

STO3G_H = [
    ("S", [(3.42525091, 0.15432897), (0.62391373, 0.53532814), (0.1688554, 0.44463454)]),
]

STO3G_C = [
    ("S", [(71.616837, 0.15432897), (13.045096, 0.53532814), (3.5305122, 0.44463454)]),
    ("S", [(2.9412494, -0.09996723), (0.6834831, 0.39951283), (0.2222899, 0.70011547)]),
    ("P", [(2.9412494, 0.15591627), (0.6834831, 0.60768372), (0.2222899, 0.39195739)]),
]

STO3G_O = [
    ("S", [(130.70932, 0.15432897), (23.808861, 0.53532814), (6.4436083, 0.44463454)]),
    ("S", [(5.0331513, -0.09996723), (1.1695961, 0.39951283), (0.380389, 0.70011547)]),
    ("P", [(5.0331513, 0.15591627), (1.1695961, 0.60768372), (0.380389, 0.39195739)]),
]

D_POLARIZATION_C = ("D", [(0.8, 1.0)])
F_POLARIZATION_C = ("F", [(0.761, 1.0)])

P631_H = [
    ("S", [(18.731137, 0.0334946), (2.8253937, 0.23472695), (0.6401217, 0.81375733)]),
    ("S", [(0.1612778, 1.0)]),
]

P631_C = [
    (
        "S",
        [
            (3047.5249, 0.0018347),
            (457.36951, 0.0140373),
            (103.94869, 0.0688426),
            (29.210155, 0.2321844),
            (9.286663, 0.4679413),
            (3.163927, 0.362312),
        ],
    ),
    ("S", [(7.8682724, -0.1193324), (1.8812885, -0.1608542), (0.5442493, 1.1434564)]),
    ("P", [(7.8682724, 0.0689991), (1.8812885, 0.316424), (0.5442493, 0.7443083)]),
    ("S", [(0.1687144, 1.0)]),
    ("P", [(0.1687144, 1.0)]),
]

P631_C_STAR = P631_C + [D_POLARIZATION_C]  # 6-31G* carbon


def h2o_sto3g():
    """Water, STO-3G: s/p shells across three atoms. 7 basis functions."""
    gbasis = [STO3G_O, STO3G_H, STO3G_H]
    atomcoords = np.array(
        [[0.0, 0.0, 0.0], [0.757, 0.586, 0.0], [-0.757, 0.586, 0.0]]
    )
    return gbasis, atomcoords


def h2o_sto3g_d():
    """Water with a 6-31G* d shell on oxygen: adds all six Cartesian d's."""
    gbasis = [STO3G_O + [("D", [(0.8, 1.0)])], STO3G_H, STO3G_H]
    atomcoords = np.array(
        [[0.0, 0.0, 0.0], [0.757, 0.586, 0.0], [-0.757, 0.586, 0.0]]
    )
    return gbasis, atomcoords


def carbon_sto3g_df():
    """Carbon with d and f polarization: exercises every shell type at once."""
    gbasis = [STO3G_C + [D_POLARIZATION_C, F_POLARIZATION_C]]
    atomcoords = np.array([[0.31, -0.17, 0.42]])
    return gbasis, atomcoords


def benzene_6_31g_star():
    """Benzene, 6-31G*: 12 atoms, 102 Cartesian basis functions (benchmark)."""
    # Regular hexagons, C-C 1.40 A, C-H 1.09 A, all in the xy plane.
    gbasis = []
    atomcoords = []
    for k in range(6):
        theta = np.pi / 3.0 * k
        atomcoords.append([1.40 * np.cos(theta), 1.40 * np.sin(theta), 0.0])
        gbasis.append(P631_C_STAR)
    for k in range(6):
        theta = np.pi / 3.0 * k
        atomcoords.append([2.49 * np.cos(theta), 2.49 * np.sin(theta), 0.0])
        gbasis.append(P631_H)
    return gbasis, np.array(atomcoords)


def seeded_coeffs(seed: int, n_mo: int, n_bf: int) -> np.ndarray:
    """Fixed-seed MO coefficient matrix with mixed signs and magnitudes."""
    rng = np.random.RandomState(seed)
    return rng.uniform(-1.0, 1.0, size=(n_mo, n_bf))

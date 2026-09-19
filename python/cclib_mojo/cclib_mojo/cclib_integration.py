"""cclib integration: drop-in equivalents of cclib's volume functions.

These functions mirror the signatures and semantics of
``cclib.method.volume.wavefunction``, ``electrondensity_spin`` and
``electrondensity`` — without monkeypatching cclib. They duck-type the two
inputs (any object with the ``ccdata.gbasis``/``ccdata.atomcoords``
attributes and any object with the ``Volume`` attributes
``origin``/``topcorner``/``spacing``/``data``), evaluate with cclib_mojo,
and return a copy of the volume whose ``data`` attribute holds the result.

Usage with a real parsed logfile::

    import cclib
    from cclib.method.volume import Volume
    from cclib_mojo import cclib_integration

    data = cclib.io.ccread("calc.log")
    vol = Volume(origin=(-5, -5, -5), topcorner=(5, 5, 5),
                 spacing=(0.2, 0.2, 0.2))
    wf = cclib_integration.wavefunction(data, vol, data.mocoeffs[0][3])
    wf.writeascube("mo4.cube")       # same layout cclib would write

No cclib import happens here; cclib is an optional, user-side dependency.
"""

from __future__ import annotations

import copy

import numpy as np

from cclib_mojo.core import density_on_grid, wavefunction_on_grid


def _grid_spec(volume) -> tuple[np.ndarray, np.ndarray, tuple[int, int, int]]:
    """(origin, spacing, shape) from a cclib-Volume-like object."""
    origin = np.asarray(volume.origin, dtype=np.float64)
    spacing = np.asarray(volume.spacing, dtype=np.float64)
    shape = tuple(int(d) for d in volume.data.shape)
    return origin, spacing, shape


def wavefunction(ccdata, volume, mocoeffs):
    """cclib ``wavefunction`` equivalent: one MO's amplitude on the volume.

    ``mocoeffs`` is a 1-D coefficient vector (e.g. ``ccdata.mocoeffs[0][3]``).
    Returns a copy of ``volume`` with ``data`` replaced.
    """
    origin, spacing, shape = _grid_spec(volume)
    grid = wavefunction_on_grid(
        ccdata.gbasis, ccdata.atomcoords[-1], mocoeffs, origin, spacing, shape
    )
    wavefn = copy.copy(volume)
    wavefn.data = grid
    return wavefn


def electrondensity_spin(ccdata, volume, mocoeffslist: list) -> object:
    """cclib ``electrondensity_spin`` equivalent: sum of squared MOs.

    ``mocoeffslist`` is a length-1 list of MO coefficient matrices, exactly
    as cclib expects (``[ccdata.mocoeffs[0][1:2]]``).
    """
    if len(mocoeffslist) != 1:
        raise ValueError("mocoeffslist input to the function should have length of 1.")
    origin, spacing, shape = _grid_spec(volume)
    grid = np.zeros(shape, dtype=np.float64)
    for mocoeffs in mocoeffslist:
        grid += density_on_grid(
            ccdata.gbasis, ccdata.atomcoords[-1], mocoeffs, origin, spacing, shape
        )
    density = copy.copy(volume)
    density.data = grid
    return density


def electrondensity(ccdata, volume, mocoeffslist: list) -> object:
    """cclib ``electrondensity`` equivalent.

    One spin set (restricted) doubles the spin density; two spin sets
    (unrestricted) sums them — the same convention as cclib.
    """
    if len(mocoeffslist) == 2:
        alpha = electrondensity_spin(ccdata, volume, [mocoeffslist[0]])
        beta = electrondensity_spin(ccdata, volume, [mocoeffslist[1]])
        alpha.data = alpha.data + beta.data
        return alpha
    edens = electrondensity_spin(ccdata, volume, [mocoeffslist[0]])
    edens.data *= 2
    return edens

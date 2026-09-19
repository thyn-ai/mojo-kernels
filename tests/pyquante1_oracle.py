"""Differential oracle: PyQuante 1.6.5 amplitude path, transcribed to Python 3.

PyQuante 1.6.5 (the version cclib's ``_found_pyquante`` backend imports) is
Python-2 only, so it cannot execute in this project's toolchain. This module
is a faithful Python-3 transcription of its pure-Python evaluation path —
same formulas, same operation order, same per-point interpreter cost
profile — used ONLY as the test/benchmark oracle. It is not part of the
shipped package.

Upstream sources (PyQuante 1.6.5, Copyright (c) 2004 Richard P. Muller,
modified BSD license; cclib, BSD 3-Clause):

- ``PyQuante/PGBF.py``   : primitive amplitude and THO eq. 2.2 norm
- ``PyQuante/CGBF.py``   : contracted amplitude, normalization via
                           self-overlap (``contracted_gto`` semantics)
- ``PyQuante/pyints.py`` : fact2, dist2, binomial_prefactor, overlap_1D,
                           gaussian_product_center, overlap (THO eq. 2.12)
- ``cclib/method/volume.py`` : sym2powerlist, getbfs flow, pyamp loop,
                           wavefunction / electrondensity_spin accumulation

The math itself is textbook Gaussian-basis algebra (Taketa, Huzinaga,
O-ohata, J. Phys. Soc. Jap. 21, 2313 (1966)).
"""

from __future__ import annotations

import math

import numpy as np

# cclib/method/volume.py sym2powerlist, verbatim.
SYM2POWERLIST = {
    "S": [(0, 0, 0)],
    "P": [(1, 0, 0), (0, 1, 0), (0, 0, 1)],
    "D": [(2, 0, 0), (0, 2, 0), (0, 0, 2), (1, 1, 0), (0, 1, 1), (1, 0, 1)],
    "F": [
        (3, 0, 0),
        (2, 1, 0),
        (2, 0, 1),
        (1, 2, 0),
        (1, 1, 1),
        (1, 0, 2),
        (0, 3, 0),
        (0, 2, 1),
        (0, 1, 2),
        (0, 0, 3),
    ],
}

# cclib/parser/utils.py convertor values used on this path.
ANG2BOHR = 1.8897261245  # convertor(x, "Angstrom", "bohr") for centers
BOHR2ANG = 0.5291772109  # convertor(1, "bohr", "Angstrom") for grid axes


# ---------------------------------------------------------------- pyints ---


def fact(i: int) -> int:
    "Normal factorial (pyints.fact)."
    val = 1
    while i > 1:
        val = i * val
        i = i - 1
    return val


def fact2(i: int) -> int:
    "Double factorial (!!) function = 1*3*5*...*i (pyints.fact2)."
    val = 1
    while i > 0:
        val = i * val
        i = i - 2
    return val


def dist2(A, B) -> float:
    "pyints.dist2."
    return pow(A[0] - B[0], 2) + pow(A[1] - B[1], 2) + pow(A[2] - B[2], 2)


def binomial(a: int, b: int) -> float:
    "Binomial coefficient (pyints.binomial)."
    return fact(a) / fact(b) / fact(a - b)


def binomial_prefactor(s: int, ia: int, ib: int, xpa: float, xpb: float) -> float:
    "From Augspurger and Dykstra (pyints.binomial_prefactor)."
    total = 0
    for t in range(s + 1):
        if s - ia <= t <= ib:
            total = total + binomial(ia, s - t) * binomial(ib, t) * pow(
                xpa, ia - s + t
            ) * pow(xpb, ib - t)
    return total


def overlap_1D(l1: int, l2: int, PAx: float, PBx: float, gamma: float) -> float:
    "Taken from THO eq. 2.12 (pyints.overlap_1D)."
    total = 0
    for i in range(1 + int(math.floor(0.5 * (l1 + l2)))):
        total = total + binomial_prefactor(2 * i, l1, l2, PAx, PBx) * fact2(
            2 * i - 1
        ) / pow(2 * gamma, i)
    return total


def gaussian_product_center(alpha1: float, A, alpha2: float, B):
    "pyints.gaussian_product_center."
    gamma = alpha1 + alpha2
    return (
        (alpha1 * A[0] + alpha2 * B[0]) / gamma,
        (alpha1 * A[1] + alpha2 * B[1]) / gamma,
        (alpha1 * A[2] + alpha2 * B[2]) / gamma,
    )


def overlap(alpha1: float, powers1, A, alpha2: float, powers2, B) -> float:
    "Taken from THO eq. 2.12 (pyints.overlap)."
    l1, m1, n1 = powers1
    l2, m2, n2 = powers2
    rab2 = dist2(A, B)
    gamma = alpha1 + alpha2
    P = gaussian_product_center(alpha1, A, alpha2, B)

    pre = pow(math.pi / gamma, 1.5) * math.exp(-alpha1 * alpha2 * rab2 / gamma)

    wx = overlap_1D(l1, l2, P[0] - A[0], P[0] - B[0], gamma)
    wy = overlap_1D(m1, m2, P[1] - A[1], P[1] - B[1], gamma)
    wz = overlap_1D(n1, n2, P[2] - A[2], P[2] - B[2], gamma)
    return pre * wx * wy * wz


# ------------------------------------------------------------------ PGBF ---


class PGBF:
    "Class for Primitive Gaussian Basis Functions (PGBF.py amplitude path)."

    def __init__(self, exponent, origin, powers=(0, 0, 0)):
        self.exp = float(exponent)
        self.origin = tuple(float(i) for i in origin)
        self.powers = powers
        self.coef = 1.0
        # PyQuante 1.6.5 normalizes primitives at construction (C side);
        # PGBF.normalize is the same THO eq. 2.2 formula.
        self.normalize()

    def amp(self, x, y, z):
        "Compute the amplitude of the PGBF at point x,y,z (PGBF.amp)."
        i, j, k = self.powers
        x0, y0, z0 = self.origin
        return (
            self.norm
            * self.coef
            * pow(x - x0, i)
            * pow(y - y0, j)
            * pow(z - z0, k)
            * math.exp(-self.exp * dist2((x, y, z), (x0, y0, z0)))
        )

    def normalize(self):
        "Normalize basis function. From THO eq. 2.2 (PGBF.normalize)."
        l, m, n = self.powers
        alpha = self.exp
        self.norm = math.sqrt(
            pow(2, 2 * (l + m + n) + 1.5)
            * pow(alpha, l + m + n + 1.5)
            / fact2(2 * l - 1)
            / fact2(2 * m - 1)
            / fact2(2 * n - 1)
            / pow(math.pi, 1.5)
        )

    def overlap(self, other):
        "Compute overlap element with another PGBF (PGBF.overlap)."
        return (
            self.norm
            * other.norm
            * overlap(self.exp, self.powers, self.origin, other.exp, other.powers, other.origin)
        )


# ------------------------------------------------------------------ CGBF ---


class CGBF:
    "Class for a contracted Gaussian basis function (CGBF.py amplitude path)."

    def __init__(self, origin, powers=(0, 0, 0)):
        self.origin = tuple(float(i) for i in origin)
        self.powers = powers
        self.norm = 1.0
        self.prims = []

    def add_primitive(self, exponent, coefficient):
        "Add a primitive BF to this contracted set (CGBF.add_primitive)."
        pbf = PGBF(exponent, self.origin, self.powers)
        pbf.coef = coefficient
        self.prims.append(pbf)

    def normalize(self):
        "Normalize the CGBF (contracted_gto_normalize semantics)."
        S = 0.0
        for ipbf in self.prims:
            for jpbf in self.prims:
                S += ipbf.coef * jpbf.coef * ipbf.overlap(jpbf)
        self.norm = self.norm / math.sqrt(S)

    def amp(self, x, y, z):
        "Compute the amplitude of the CGBF at point x,y,z (CGBF.amp)."
        val = 0.0
        for prim in self.prims:
            val += prim.amp(x, y, z)
        return self.norm * val


# ------------------------------------------------------- cclib volume.py ---


def getbfs(gbasis, atomcoords_ang):
    """cclib's getbfs flow: cclib gbasis + Angstrom coords -> list of CGBF.

    Centers are converted Angstrom -> bohr exactly as cclib's pyquante2
    bridge does (convertor(x, "Angstrom", "bohr") = x * 1.8897261245).
    """
    bfs = []
    for i, atom_shells in enumerate(gbasis):
        center = (
            float(atomcoords_ang[i][0]) * ANG2BOHR,
            float(atomcoords_ang[i][1]) * ANG2BOHR,
            float(atomcoords_ang[i][2]) * ANG2BOHR,
        )
        for sym, prims in atom_shells:
            for power in SYM2POWERLIST[sym]:
                bf = CGBF(center, power)
                for expnt, coef in prims:
                    bf.add_primitive(expnt, coef)
                bf.normalize()
                bfs.append(bf)
    return bfs


def pyamp(bfs, bs, points):
    """cclib's pyamp hot loop: one basis function on all grid points."""
    mesh_vals = np.zeros(len(points))
    for i in range(len(points)):
        mesh_vals[i] = bfs[bs].amp(points[i][0], points[i][1], points[i][2])
    return mesh_vals


def grid_axes(origin, step, shape):
    """cclib's getGrid: Angstrom axes divided by bohr->Angstrom constant."""
    nx, ny, nz = shape
    ax = (origin[0] + np.arange(nx, dtype=np.float64) * step[0]) / BOHR2ANG
    ay = (origin[1] + np.arange(ny, dtype=np.float64) * step[1]) / BOHR2ANG
    az = (origin[2] + np.arange(nz, dtype=np.float64) * step[2]) / BOHR2ANG
    return ax, ay, az


def gridpoints(origin, step, shape):
    """cclib's gridpoints ordering: x outermost, z innermost (C-order)."""
    ax, ay, az = grid_axes(origin, step, shape)
    return np.asanyarray(
        tuple((xp, yp, zp) for xp in ax for yp in ay for zp in az)
    )


def wavefunction(gbasis, atomcoords_ang, mocoeffs, origin, step, shape):
    """cclib's Volume.wavefunction accumulation, PyQuante1 amplitude path."""
    bfs = getbfs(gbasis, atomcoords_ang)
    points = gridpoints(origin, step, shape)
    data = np.zeros(int(np.prod(shape)), "d")
    for bs in range(len(bfs)):
        if abs(mocoeffs[bs]) > 0.0:
            data += pyamp(bfs, bs, points) * mocoeffs[bs]
    return data.reshape(shape)


def electrondensity_spin(gbasis, atomcoords_ang, mocoeffs, origin, step, shape):
    """cclib's electrondensity_spin accumulation for one spin set."""
    bfs = getbfs(gbasis, atomcoords_ang)
    points = gridpoints(origin, step, shape)
    npts = int(np.prod(shape))
    density = np.zeros(npts, "d")
    for mocoeff in mocoeffs:
        wavefn = np.zeros(npts, "d")
        for bs in range(len(bfs)):
            if abs(mocoeff[bs]) > 0.0:
                wavefn += pyamp(bfs, bs, points) * mocoeff[bs]
        density += wavefn**2
    return density.reshape(shape)

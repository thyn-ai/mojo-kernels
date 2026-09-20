"""Runnable ase-mojo quickstart (also the wheel smoke test in CI).

Builds a small triclinic configuration, computes the neighbor list with
ase_mojo, verifies it against an O(n^2) brute-force minimum-image search,
and prints a checksum. Works on both backends (native and fallback).

Run from anywhere after `pip install ase-mojo`:

    python quickstart.py
"""

import os
import sys

# When run from the source tree (python/ase_mojo/quickstart.py), the
# interpreter prepends this script's own directory to sys.path, which would
# shadow the *installed* wheel with the source checkout. Remove exactly
# that auto-inserted entry (PYTHONPATH entries are intentionally kept, so
# the script also works in repo development mode).
_HERE = os.path.dirname(os.path.abspath(__file__))
if sys.path and os.path.abspath(sys.path[0] or os.getcwd()) == _HERE:
    sys.path.pop(0)

import numpy as np  # noqa: E402

import ase_mojo  # noqa: E402
from ase_mojo import primitive_neighbor_list  # noqa: E402


def main() -> None:
    rng = np.random.default_rng(20260919)
    cell = np.array([[4.0, 0.0, 0.0], [1.0, 3.5, 0.0], [0.5, 0.8, 3.0]])
    pos = rng.random((16, 3)) @ cell
    cutoff = 1.6

    i, j, S, d = primitive_neighbor_list(
        "ijSd", [True, True, True], cell, pos, cutoff
    )

    # Brute-force minimum-image verification (independent of the package).
    want = set()
    for a in range(16):
        for b in range(16):
            if a == b:
                continue
            for s0 in (-1, 0, 1):
                for s1 in (-1, 0, 1):
                    for s2 in (-1, 0, 1):
                        s = (s0, s1, s2)
                        D = pos[b] - pos[a] + np.array(s) @ cell
                        if np.linalg.norm(D) < cutoff:
                            want.add((a, b, s))
    got = set(zip(i.tolist(), j.tolist(), map(tuple, S.tolist())))
    assert got == want, "quickstart pair set does not match brute force"
    assert np.all(d < cutoff)

    info = ase_mojo.backend_info()
    backend = "native" if info["native_available"] else "fallback"
    checksum = float(d.sum()) if len(d) else 0.0
    print(f"ase-mojo quickstart (backend: {backend})")
    print(f"  atoms: 16, cutoff: {cutoff} A, pairs: {len(i)}")
    print(f"  distance checksum: {checksum:.12e}")


if __name__ == "__main__":
    main()

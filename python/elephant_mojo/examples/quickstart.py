#!/usr/bin/env python3
"""elephant-mojo quickstart: surrogate dithering + a p-value spectrum.

Self-contained: no elephant install and no Mojo toolchain required. Run
after `pip install elephant-mojo`:

    python quickstart.py

It prints the active backend (native Mojo kernel or NumPy fallback), a
seeded surrogate-generation summary, and a deterministic p-value-spectrum
checksum. With ELEPHANT_MOJO_DISABLE_NATIVE=1 the fallback is forced; the
spectrum checksum is identical on either backend (it is a deterministic
function), while the stochastic dither summary differs between backends
by construction (same distributions, different RNG sequences).
"""

import numpy as np

from elephant_mojo import backend_info, dither, pvalue_spectrum

# Two tiny spike trains (seconds), sorted ascending within [0, 1].
spiketrains = [
    np.array([0.012, 0.105, 0.240, 0.402, 0.555, 0.731, 0.908]),
    np.array([0.040, 0.180, 0.333, 0.510, 0.666, 0.845]),
]

surrogates = dither(
    spiketrains,
    bin_size=0.005,  # 5 ms bins -> 200 bins over [0, 1) s
    dither=0.015,  # 15 ms dither half-width
    n_surrogates=100,
    t_start=0.0,
    t_stop=1.0,
    seed=42,
)
print("surrogate block:", surrogates.shape, surrogates.dtype)
print("mean survivors per train:", surrogates.sum(axis=2).mean(axis=0).tolist())

# A fixed per-surrogate maximal-occurrence matrix (n_surr=100, sizes 2..4):
# in a real analysis this comes from mining each surrogate (README recipe).
rng = np.random.default_rng(20260919)
max_occs = rng.integers(0, 8, size=(100, 3)).astype(np.float64)

pv = pvalue_spectrum(max_occs, min_spikes=2, max_spikes=4, min_occ=2)
print("spectrum entries:", len(pv), "| first:", pv[0])

info = backend_info()
backend = "native" if info["native_available"] else "fallback"
checksum = float(sum(e[-1] for e in pv))
print(f"backend: {backend}")
print(f"spectrum checksum: {checksum:.17e}")

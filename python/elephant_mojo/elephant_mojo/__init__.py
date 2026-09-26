"""elephant-mojo: fast SPADE surrogate dithering and p-value spectra.

A drop-in accelerator for the surrogate machinery behind
`elephant <https://python-elephant.org/>`_'s SPADE analysis, powered by a
Mojo kernel where the platform supports it (macOS arm64, Linux x86_64),
with a vendored NumPy fallback everywhere else (including Windows):

    from elephant_mojo import dither, pvalue_spectrum

    # Binned surrogates, the dither_spikes + binning pipeline of SPADE:
    surrogates = dither(spiketrains, bin_size=5.0, dither=15.0,
                        n_surrogates=100, t_stop=1000.0, seed=42)

    # P-value spectrum from per-surrogate maximal pattern occurrences:
    pv = pvalue_spectrum(max_occs, min_spikes=2, max_spikes=4, min_occ=2)

Set ELEPHANT_MOJO_DISABLE_NATIVE=1 to force the pure-Python fallback.
"""

from elephant_mojo._native import backend_info, native_available
from elephant_mojo.core import (
    METHOD_DITHER_SPIKES,
    METHOD_REFRACTORY_PERIOD,
    DitherError,
    PValueSpectrumError,
    dither,
    pvalue_spectrum,
)

__version__ = "0.1.5"  # x-release-please-version
__all__ = [
    "DitherError",
    "METHOD_DITHER_SPIKES",
    "METHOD_REFRACTORY_PERIOD",
    "PValueSpectrumError",
    "backend_info",
    "dither",
    "native_available",
    "pvalue_spectrum",
    "__version__",
]

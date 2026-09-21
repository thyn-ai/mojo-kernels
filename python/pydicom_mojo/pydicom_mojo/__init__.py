"""pydicom-mojo: Mojo-accelerated DICOM RLE Lossless codec, byte-compatible
with pydicom's pure-Python PackBits codec (decode AND encode).

Powered by a clean-room Mojo kernel where the platform supports it (macOS
arm64, Linux x86_64), with a vendored pure-Python fallback everywhere else
(including Windows). Both backends are byte-identical to the pydicom
oracle.

    import pydicom_mojo

    # Decode one RLE frame (including its 64-byte header) to little-endian,
    # planar configuration 1 pixel bytes:
    raw = pydicom_mojo.decode_frame(rle_frame, rows=512, columns=512,
                                    samples_per_pixel=1, bits_allocated=16)

    # Encode one frame of little-endian interleaved pixel bytes:
    rle_frame = pydicom_mojo.encode_frame(raw_le, columns=512,
                                          samples_per_pixel=1, bits_allocated=16)

    # Segment-level PackBits is exposed too:
    plane = pydicom_mojo.decode_segment(segment_bytes)
    segment_bytes = pydicom_mojo.encode_segment(plane, columns=512)

To use it inside pydicom's own pixel data pipelines, register the plugin
(see pydicom_mojo.plugin):

    from pydicom.pixels.decoders import RLELosslessDecoder
    from pydicom.pixels.encoders import RLELosslessEncoder
    RLELosslessDecoder.add_plugin("pydicom_mojo", ("pydicom_mojo.plugin", "decode_frame"))
    RLELosslessEncoder.add_plugin("pydicom_mojo", ("pydicom_mojo.plugin", "encode_frame"))

Set PYDICOM_MOJO_DISABLE_NATIVE=1 to force the pure-Python fallback.
"""

from __future__ import annotations

from pydicom_mojo._native import backend_info, native_available
from pydicom_mojo._reference import RleCodecError
from pydicom_mojo.core import (
    decode_frame,
    decode_segment,
    encode_frame,
    encode_segment,
    get_backend,
)

__version__ = "0.1.3"  # x-release-please-version
__all__ = [
    "decode_frame",
    "encode_frame",
    "decode_segment",
    "encode_segment",
    "get_backend",
    "backend_info",
    "native_available",
    "RleCodecError",
    "__version__",
]

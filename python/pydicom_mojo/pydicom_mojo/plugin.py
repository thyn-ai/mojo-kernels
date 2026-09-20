"""pydicom v3 decoder/encoder plugin entry points for pydicom-mojo.

Register with pydicom's out-of-tree plugin system (the same seam
`pylibjpeg` uses) to make pydicom's own pixel data pipelines use the
Mojo-accelerated RLE codec:

    from pydicom.pixels.decoders import RLELosslessDecoder
    from pydicom.pixels.encoders import RLELosslessEncoder

    RLELosslessDecoder.add_plugin(
        "pydicom_mojo", ("pydicom_mojo.plugin", "decode_frame")
    )
    RLELosslessEncoder.add_plugin(
        "pydicom_mojo", ("pydicom_mojo.plugin", "encode_frame")
    )

    ds.decompress(decoding_plugin="pydicom_mojo")   # or the default attempt
    ds.compress(RLELossless, encoding_plugin="pydicom_mojo")

This module deliberately does NOT import pydicom: the dependency tables are
keyed by the transfer syntax UID string (pydicom's `UID` is a `str`
subclass, so membership and equality checks match), and the runner objects
are duck-typed. pydicom is only ever the caller.
"""

from __future__ import annotations

from pydicom_mojo.core import decode_frame as _decode_frame_impl
from pydicom_mojo.core import encode_frame as _encode_frame_impl

# 1.2.840.10008.1.2.5 = RLE Lossless. No dependencies beyond this package.
RLE_LOSSLESS_UID = "1.2.840.10008.1.2.5"

DECODER_DEPENDENCIES = {RLE_LOSSLESS_UID: ()}
ENCODER_DEPENDENCIES = {RLE_LOSSLESS_UID: ()}


def is_available(uid) -> bool:
    """Return True if this plugin supports the transfer syntax `uid`."""
    return str(uid) == RLE_LOSSLESS_UID


def decode_frame(src: bytes, runner) -> bytes:
    """Decoding-plugin entry point (pydicom v3 `DecodeFunction` seam).

    Mirrors pydicom.pixels.decoders.rle._decode_frame: reads the frame
    geometry from the runner, honours the optional `rle_segment_order`
    runner option ("<" or ">"), and marks the runner's planar
    configuration as 1 after a successful decode.
    """
    frame = _decode_frame_impl(
        src,
        runner.rows,
        runner.columns,
        runner.samples_per_pixel,
        runner.bits_allocated,
        runner.get_option("rle_segment_order", ">"),
    )
    # Update the runner options to ensure the reshaping is correct
    # Only do this if we successfully decoded the frame
    runner.set_option("planar_configuration", 1)
    return frame


def encode_frame(src: bytes, runner) -> bytes:
    """Encoding-plugin entry point (pydicom v3 `EncodeFunction` seam).

    Mirrors pydicom.pixels.encoders.native._encode_frame: honours the
    optional `byteorder` runner option ("<" default, ">" unsupported).
    """
    return _encode_frame_impl(
        src,
        runner.columns,
        runner.samples_per_pixel,
        runner.bits_allocated,
        runner.get_option("byteorder", "<"),
    )

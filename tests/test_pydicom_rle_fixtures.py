"""Shared fixtures for the pydicom-rle differential suite.

The oracle is the published PyPI package `pydicom` (no NumPy required: the
suite works at the buffer level). Everything else is generated locally from
fixed seeds — no network, no datasets — so the suite is bit-reproducible on
any machine.

Oracle callables mirror pydicom 3.0.2's plugin-internal functions:

  * ``oracle_decode_frame`` -> pydicom.pixels.decoders.rle._rle_decode_frame
  * ``oracle_encode_frame`` -> pydicom.pixels.encoders.native._encode_frame
    (driven through a minimal stand-in for EncodeRunner)
"""

from __future__ import annotations

import random
import struct

import pydicom
from pydicom.dataset import Dataset
from pydicom.pixels.decoders.rle import (
    _rle_decode_frame,
    _rle_decode_segment,
    _rle_parse_header,
)
from pydicom.pixels.encoders.native import _encode_frame, _encode_segment
from pydicom.uid import ExplicitVRLittleEndian, RLELossless, generate_uid

PYDICOM_VERSION = pydicom.__version__

# SOP Class UIDs used by the generated datasets.
CT_IMAGE_STORAGE = "1.2.840.10008.5.1.4.1.1.2"
SEGMENTATION_STORAGE = "1.2.840.10008.5.1.4.1.1.66.4"


class FakeEncodeRunner:
    """The slice of pydicom's EncodeRunner the native encoder reads."""

    def __init__(self, columns: int, samples_per_pixel: int, bits_allocated: int, byteorder: str = "<"):
        self.columns = columns
        self.samples_per_pixel = samples_per_pixel
        self.bits_allocated = bits_allocated
        self._byteorder = byteorder

    def get_option(self, name, default=None):
        if name == "byteorder":
            return self._byteorder
        return default


def oracle_decode_segment(src: bytes) -> bytes:
    return bytes(_rle_decode_segment(src))


def oracle_encode_segment(src: bytes, columns: int) -> bytes:
    return bytes(_encode_segment(src, columns))


def oracle_decode_frame(
    src: bytes, rows: int, columns: int, samples_per_pixel: int,
    bits_allocated: int, segment_order: str = ">",
) -> bytes:
    return bytes(
        _rle_decode_frame(src, rows, columns, samples_per_pixel, bits_allocated, segment_order)
    )


def oracle_encode_frame(
    src: bytes, columns: int, samples_per_pixel: int, bits_allocated: int, byteorder: str = "<"
) -> bytes:
    return bytes(_encode_frame(src, FakeEncodeRunner(columns, samples_per_pixel, bits_allocated, byteorder)))


def oracle_parse_header(header: bytes) -> list[int]:
    return _rle_parse_header(header)


# ---------------------------------------------------------------------------
# Synthetic image content (deterministic, seeded)
# ---------------------------------------------------------------------------


def ct_like_frame(rows: int, columns: int, seed: int) -> bytes:
    """16-bit LE CT-like slice: smooth low-frequency structures plus fine
    noise (values 0..4095, stored 16-bit). Moderately compressible."""
    rng = random.Random(seed)
    cx, cy = rows * 0.5 + rng.uniform(-4, 4), columns * 0.5 + rng.uniform(-4, 4)
    radius = min(rows, columns) * rng.uniform(0.28, 0.36)
    out = bytearray(rows * columns * 2)
    for y in range(rows):
        for x in range(columns):
            dist2 = (y - cy) ** 2 + (x - cx) ** 2
            if dist2 < radius * radius:
                value = 900 + int(140 * (1 - dist2 / (radius * radius)))
            else:
                value = 60
            value += rng.randrange(24)  # fine acquisition noise
            out[2 * (y * columns + x)] = value & 0xFF
            out[2 * (y * columns + x) + 1] = (value >> 8) & 0xFF
    return bytes(out)


def seg_like_frame(rows: int, columns: int, seed: int) -> bytes:
    """8-bit binary segmentation-like frame: a handful of filled blobs on a
    zero background. Extremely compressible (long runs)."""
    rng = random.Random(seed)
    out = bytearray(rows * columns)
    for _ in range(rng.randrange(2, 5)):
        cy, cx = rng.randrange(rows), rng.randrange(columns)
        ry, rx = rng.randrange(2, max(3, rows // 4)), rng.randrange(2, max(3, columns // 4))
        label = rng.randrange(1, 8)
        for y in range(max(0, cy - ry), min(rows, cy + ry)):
            for x in range(max(0, cx - rx), min(columns, cx + rx)):
                if ((y - cy) / ry) ** 2 + ((x - cx) / rx) ** 2 <= 1.0:
                    out[y * columns + x] = label
    return bytes(out)


def rgb_like_frame(rows: int, columns: int, seed: int) -> bytes:
    """8-bit interleaved RGB frame: smooth colour gradients plus noise."""
    rng = random.Random(seed)
    out = bytearray(rows * columns * 3)
    for y in range(rows):
        for x in range(columns):
            i = 3 * (y * columns + x)
            out[i] = (x * 255 // max(columns - 1, 1) + rng.randrange(6)) & 0xFF
            out[i + 1] = (y * 255 // max(rows - 1, 1) + rng.randrange(6)) & 0xFF
            out[i + 2] = ((x + y) * 7 + rng.randrange(10)) & 0xFF
    return bytes(out)


# ---------------------------------------------------------------------------
# DICOM datasets built with pydicom itself
# ---------------------------------------------------------------------------


def make_dataset(
    rows: int,
    columns: int,
    samples_per_pixel: int,
    bits_allocated: int,
    pixel_data: bytes,
    number_of_frames: int = 1,
    segmentation: bool = False,
) -> Dataset:
    """A minimal native-transfer-syntax dataset carrying `pixel_data`."""
    ds = Dataset()
    ds.SOPClassUID = SEGMENTATION_STORAGE if segmentation else CT_IMAGE_STORAGE
    ds.SOPInstanceUID = generate_uid()
    ds.Modality = "SEG" if segmentation else "CT"
    ds.Rows = rows
    ds.Columns = columns
    ds.BitsAllocated = bits_allocated
    ds.BitsStored = bits_allocated
    ds.HighBit = bits_allocated - 1
    ds.PixelRepresentation = 0
    ds.SamplesPerPixel = samples_per_pixel
    if samples_per_pixel > 1:
        ds.PlanarConfiguration = 0
        ds.PhotometricInterpretation = "RGB"
    else:
        ds.PhotometricInterpretation = "MONOCHROME2"
    if number_of_frames > 1:
        ds.NumberOfFrames = str(number_of_frames)
    ds.file_meta = Dataset()
    ds.file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
    ds.file_meta.MediaStorageSOPClassUID = ds.SOPClassUID
    ds.file_meta.MediaStorageSOPInstanceUID = ds.SOPInstanceUID
    ds.PixelData = pixel_data
    return ds


def oracle_compress(ds: Dataset) -> Dataset:
    """Compress the dataset's Pixel Data to RLE Lossless with the oracle's
    own native encoder plugin (in place)."""
    ds.compress(RLELossless, encoding_plugin="pydicom")
    return ds


def split_frames(ds: Dataset) -> list[bytes]:
    """The encapsulated RLE frames of a compressed dataset, as raw bytes."""
    from pydicom.encaps import generate_frames

    number_of_frames = int(getattr(ds, "NumberOfFrames", 1))
    return [bytes(frame) for frame in generate_frames(ds.PixelData, number_of_frames=number_of_frames)]


def rle_frame_header(nr_segments: int, offsets: list[int]) -> bytes:
    """A well-formed 64-byte RLE header."""
    values = [nr_segments] + list(offsets)
    values += [0] * (16 - len(values))
    return struct.pack("<16L", *values)

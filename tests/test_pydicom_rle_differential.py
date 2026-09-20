"""Differential tests: pydicom_mojo must match the pydicom oracle byte-for-byte.

Run twice by scripts/test_all_pydicom_rle.sh: once against the native Mojo
kernel and once with PYDICOM_MOJO_DISABLE_NATIVE=1 (forced pure-Python
fallback). Both backends must agree with pydicom EXACTLY (byte-equal, zero
tolerance) on every segment, frame, and dataset.

The oracle is the published PyPI package, pinned to pydicom==3.0.2; it is
never a runtime dependency of pydicom_mojo. DICOM fixtures are generated
with pydicom itself (its own native RLE encoder produces the compressed
datasets); synthetic pixel content comes from the seeded builders in
test_pydicom_rle_fixtures.py.
"""

from __future__ import annotations

import os
import random
import struct
import warnings

import pytest

import pydicom_mojo

from test_pydicom_rle_fixtures import (
    PYDICOM_VERSION,
    ct_like_frame,
    make_dataset,
    oracle_compress,
    oracle_decode_frame,
    oracle_decode_segment,
    oracle_encode_frame,
    oracle_encode_segment,
    oracle_parse_header,
    rgb_like_frame,
    rle_frame_header,
    seg_like_frame,
    split_frames,
)

# (samples_per_pixel, bits_allocated) geometries seen in real DICOM RLE.
GEOMETRIES = [
    pytest.param(1, 8, id="mono8"),
    pytest.param(1, 16, id="mono16"),
    pytest.param(1, 32, id="mono32"),
    pytest.param(3, 8, id="rgb8"),
    pytest.param(3, 16, id="rgb16"),
]


def _expected_backend() -> str:
    # scripts/test_all_pydicom_rle.sh runs the suite once per backend.
    return "fallback" if os.environ.get("PYDICOM_MOJO_DISABLE_NATIVE") == "1" else "native"


def _rng_bytes(rng: random.Random, n: int) -> bytes:
    return bytes(rng.randrange(256) for _ in range(n))


# ---------------------------------------------------------------------------
# Segment decode: oracle-pinned edge cases and fuzz (Mode A)
# ---------------------------------------------------------------------------

SEGMENT_EDGE_CASES = [
    pytest.param(b"", id="empty"),
    pytest.param(b"\x80", id="noop-only"),
    pytest.param(b"\x00\x41", id="one-literal"),
    pytest.param(b"\xfe\x07", id="replicate-3"),
    pytest.param(b"\x81\x09", id="replicate-128"),
    pytest.param(b"\xfe", id="replicate-missing-value"),
    pytest.param(b"\x05\x01\x02", id="literal-truncated"),
    pytest.param(b"\x80\x00\x41\x80", id="noop-around-literal"),
    pytest.param(b"\x7f" + bytes(range(128)), id="max-literal"),
    pytest.param(b"\x81", id="replicate128-missing-value"),
    pytest.param(bytes([255, 0]), id="replicate-2-of-zero"),
]


@pytest.mark.parametrize("blob", SEGMENT_EDGE_CASES)
def test_segment_decode_edge_cases_match_oracle(blob):
    assert pydicom_mojo.decode_segment(blob) == oracle_decode_segment(blob)
    assert pydicom_mojo.get_backend() == _expected_backend()


@pytest.mark.parametrize("size", (1, 2, 3, 17, 64, 255, 1024))
def test_segment_decode_random_packet_soup_matches_oracle(size):
    # Arbitrary bytes: exercises truncation, no-ops, and every packet kind.
    rng = random.Random(f"seg-soup-{size}")
    for _ in range(40):
        blob = _rng_bytes(rng, rng.randrange(size + 1))
        assert pydicom_mojo.decode_segment(blob) == oracle_decode_segment(blob)


# ---------------------------------------------------------------------------
# Segment encode: oracle-pinned packetization (Mode B) + round trips
# ---------------------------------------------------------------------------

ROW_PACKET_CASES = [
    (b"\x2a", bytes([0, 0x2A])),                # singleton -> literal
    (b"\x2a\x2a", bytes([0xFF, 0x2A])),         # 2-run -> replicate
    (b"\x01\x02", bytes([1, 1, 2, 0])),         # literal pair, odd -> 0x00 pad
    (b"\x07\x07\x07", bytes([0xFE, 7])),        # 3-run
    (b"\x09" * 128, bytes([0x81, 9])),          # exact 128-run
    (b"\x09" * 129, bytes([0x81, 9, 0, 9])),    # 128 + leftover 1 -> literal
    (b"\x09" * 130, bytes([0x81, 9, 0xFF, 9])),  # 128 + 2 -> replicate
    (b"\x09" * 256, bytes([0x81, 9, 0x81, 9])),  # two full runs
    (b"\x09" * 300, bytes([0x81, 9, 0x81, 9, 0xD5, 9])),  # 128+128+44
    (  # mixed packets, odd length -> trailing 0x00 pad
        bytes([1, 2, 3, 3, 4, 5, 5, 5, 6]),
        bytes([1, 1, 2, 0xFF, 3, 0, 4, 0xFE, 5, 0, 6, 0]),
    ),
]


@pytest.mark.parametrize("row, expected", ROW_PACKET_CASES)
def test_segment_encode_packetization_matches_oracle(row, expected):
    # Oracle-pinned outputs AND byte equality with the oracle encoder.
    assert oracle_encode_segment(row, len(row)) == expected
    assert pydicom_mojo.encode_segment(row, len(row)) == expected
    assert pydicom_mojo.get_backend() == _expected_backend()


def test_segment_encode_even_padding_matches_oracle():
    # Odd-length encoded segments get one trailing 0x00 byte.
    assert pydicom_mojo.encode_segment(b"\x01\x02\x03", 2) == oracle_encode_segment(b"\x01\x02\x03", 2)
    assert len(pydicom_mojo.encode_segment(b"\x01\x02\x03", 2)) % 2 == 0
    assert pydicom_mojo.encode_segment(b"abc", 100) == oracle_encode_segment(b"abc", 100)
    # The padding byte is decode-safe: it is a literal header with no payload.
    padded = pydicom_mojo.encode_segment(b"\x01\x02\x03", 2)
    assert pydicom_mojo.decode_segment(padded) == b"\x01\x02\x03"


def test_segment_encode_columns_edge_semantics_match_oracle():
    with pytest.raises(ValueError, match=r"range\(\) arg 3 must not be zero"):
        pydicom_mojo.encode_segment(b"abc", 0)
    with pytest.raises(ValueError, match=r"range\(\) arg 3 must not be zero"):
        pydicom_mojo.encode_segment(b"", 0)
    # Negative columns: the oracle's range() is empty -> empty segment.
    assert pydicom_mojo.encode_segment(b"abc", -1) == oracle_encode_segment(b"abc", -1) == b""
    assert pydicom_mojo.encode_segment(b"", 5) == oracle_encode_segment(b"", 5) == b""


@pytest.mark.parametrize("size", (0, 1, 2, 5, 100, 997, 4096))
@pytest.mark.parametrize("columns", (1, 3, 16, 128, 4096))
def test_segment_encode_fuzz_matches_oracle(size, columns):
    rng = random.Random(f"seg-enc-{size}-{columns}")
    # Runs-heavy content: packbits adversarial (long runs, run boundaries).
    payload = bytes(rng.choice((0, 0, 0, 1, 7, 128, 255, rng.randrange(256))) for _ in range(size))
    mine = pydicom_mojo.encode_segment(payload, columns)
    oracle = oracle_encode_segment(payload, columns)
    assert mine == oracle
    # Both sides decode back to the payload (oracle decoder included).
    assert oracle_decode_segment(mine) == payload
    assert pydicom_mojo.decode_segment(mine) == payload


def test_segment_round_trip_128_boundaries():
    rng = random.Random("seg-128")
    for run_len in (126, 127, 128, 129, 130, 255, 256, 257, 384, 511, 513):
        payload = bytes([7]) * run_len + _rng_bytes(rng, 13) + bytes([9]) * run_len
        for columns in (len(payload), 97):
            mine = pydicom_mojo.encode_segment(payload, columns)
            assert mine == oracle_encode_segment(payload, columns)
            assert pydicom_mojo.decode_segment(mine) == payload


# ---------------------------------------------------------------------------
# Header parsing
# ---------------------------------------------------------------------------


def test_header_short_raises_like_oracle():
    for length in (0, 4, 63):
        header = bytes(length)
        with pytest.raises(ValueError, match="The RLE header can only be 64 bytes long"):
            oracle_parse_header(header)
        with pytest.raises(ValueError, match="The RLE header can only be 64 bytes long"):
            pydicom_mojo.decode_frame(header, 1, 1, 1, 8)


def test_header_too_many_segments_raises_like_oracle():
    header = struct.pack("<16L", 16, *([64] * 15))
    with pytest.raises(ValueError, match="invalid number of segments"):
        oracle_parse_header(header)
    with pytest.raises(ValueError, match=r"invalid number of segments \(16\)"):
        pydicom_mojo.decode_frame(header, 1, 1, 1, 8)


def test_header_segment_count_mismatch_raises_like_oracle():
    # 2 segments in the header, but 1x8-bit geometry expects 1.
    frame = rle_frame_header(2, [64, 70]) + b"\x00\x01" * 12
    with pytest.raises(ValueError, match=r"doesn't match the expected amount \(2 vs. 1"):
        oracle_decode_frame(frame, 2, 2, 1, 8)
    with pytest.raises(ValueError, match=r"doesn't match the expected amount \(2 vs. 1"):
        pydicom_mojo.decode_frame(frame, 2, 2, 1, 8)


# ---------------------------------------------------------------------------
# Frame decode: geometry matrix, segment orders, error/warning parity
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("samples,bits", GEOMETRIES)
@pytest.mark.parametrize("rows,columns", ((1, 1), (5, 3), (16, 16), (31, 47)))
def test_frame_round_trip_matches_oracle(samples, bits, rows, columns):
    rng = random.Random(f"frame-{samples}-{bits}-{rows}-{columns}")
    nbytes = rows * columns * samples * (bits // 8)
    native_pixels = _rng_bytes(rng, nbytes)
    encoded = pydicom_mojo.encode_frame(native_pixels, columns, samples, bits)
    assert encoded == oracle_encode_frame(native_pixels, columns, samples, bits)
    expected = oracle_decode_frame(encoded, rows, columns, samples, bits)
    actual = pydicom_mojo.decode_frame(encoded, rows, columns, samples, bits)
    assert actual == expected
    assert pydicom_mojo.get_backend() == _expected_backend()


@pytest.mark.parametrize("samples,bits", GEOMETRIES)
def test_frame_decode_synthetic_content_matches_oracle(samples, bits):
    # Image-like content from the seeded builders (not uniform-random).
    rows, columns = 48, 64
    if samples == 1 and bits == 16:
        native = ct_like_frame(rows, columns, seed=1)
    elif samples == 1 and bits == 8:
        native = seg_like_frame(rows, columns, seed=2)
    else:
        rng = random.Random(f"synth-{samples}-{bits}")
        native = _rng_bytes(rng, rows * columns * samples * (bits // 8))
    encoded = oracle_encode_frame(native, columns, samples, bits)
    assert pydicom_mojo.decode_frame(encoded, rows, columns, samples, bits) == oracle_decode_frame(
        encoded, rows, columns, samples, bits
    )


def test_frame_decode_little_endian_segment_order_matches_oracle():
    rows, columns = 4, 6
    rng = random.Random("le-order")
    native = _rng_bytes(rng, rows * columns * 2)
    encoded = oracle_encode_frame(native, columns, 1, 16)
    for order in ("<", ">", "non-conformant-garbage"):
        expected = oracle_decode_frame(encoded, rows, columns, 1, 16, order)
        actual = pydicom_mojo.decode_frame(encoded, rows, columns, 1, 16, order)
        assert actual == expected


def test_frame_decode_big_endian_segment_layout_pinned():
    # Hand-built 16-bit frame: MSB segment first ('>' order, DICOM default).
    # Decodes to LE pixels 0x0102, 0x0304, 0x0506, 0x0708.
    seg0 = bytes([3]) + bytes([1, 3, 5, 7])  # MSBs
    seg1 = bytes([3]) + bytes([2, 4, 6, 8])  # LSBs
    frame = rle_frame_header(2, [64, 64 + len(seg0)]) + seg0 + seg1
    expected = bytes.fromhex("0201040306050807")
    assert oracle_decode_frame(frame, 2, 2, 1, 16, ">") == expected
    assert pydicom_mojo.decode_frame(frame, 2, 2, 1, 16, ">") == expected
    # '<' order reads the first segment as the LSB plane instead.
    expected_le = bytes.fromhex("0102030405060708")
    assert oracle_decode_frame(frame, 2, 2, 1, 16, "<") == expected_le
    assert pydicom_mojo.decode_frame(frame, 2, 2, 1, 16, "<") == expected_le


def test_frame_decode_bits_not_multiple_of_8_raises_like_oracle():
    with pytest.raises(NotImplementedError, match="Unable to decode RLE encoded pixel data with 12 bits"):
        oracle_decode_frame(bytes(64), 2, 2, 1, 12)
    with pytest.raises(NotImplementedError, match="Unable to decode RLE encoded pixel data with 12 bits"):
        pydicom_mojo.decode_frame(bytes(64), 2, 2, 1, 12)


def test_frame_decode_short_segment_raises_like_oracle():
    seg0 = bytes([3]) + bytes([1, 3, 5, 7])
    seg1 = bytes([1]) + bytes([2, 4])  # decodes to 2 bytes, needs 4
    frame = rle_frame_header(2, [64, 64 + len(seg0)]) + seg0 + seg1
    with pytest.raises(ValueError, match=r"doesn't match the expected amount \(2 vs. 4 bytes\)"):
        oracle_decode_frame(frame, 2, 2, 1, 16)
    with pytest.raises(ValueError, match=r"doesn't match the expected amount \(2 vs. 4 bytes\)"):
        pydicom_mojo.decode_frame(frame, 2, 2, 1, 16)


def test_frame_decode_overlong_segment_warns_and_truncates_like_oracle():
    seg0 = bytes([3]) + bytes([1, 3, 5, 7])
    seg1 = bytes([4]) + bytes([2, 4, 6, 8]) + b"\x99"  # 5 bytes, needs 4
    frame = rle_frame_header(2, [64, 64 + len(seg0)]) + seg0 + seg1
    with warnings.catch_warnings(record=True) as caught_oracle:
        warnings.simplefilter("always")
        expected = oracle_decode_frame(frame, 2, 2, 1, 16)
    with warnings.catch_warnings(record=True) as caught_mine:
        warnings.simplefilter("always")
        actual = pydicom_mojo.decode_frame(frame, 2, 2, 1, 16)
    assert actual == expected
    assert [str(w.message) for w in caught_mine] == [str(w.message) for w in caught_oracle]
    assert any("non-conformant padding" in str(w.message) for w in caught_mine)


def test_frame_decode_overlong_segment_large_replicate_tail():
    # Over-long via replicate packets (forces the native buffer-growth path).
    rows, columns = 8, 8
    seg_msb = bytes([63]) + bytes(64)  # exactly 64 bytes decoded
    seg_lsb = bytes([63]) + bytes(64) + bytes([0x81, 0xFF])  # +128 bytes tail
    frame = rle_frame_header(2, [64, 64 + len(seg_msb)]) + seg_msb + seg_lsb
    with warnings.catch_warnings(record=True) as caught_oracle:
        warnings.simplefilter("always")
        expected = oracle_decode_frame(frame, rows, columns, 1, 16)
    with warnings.catch_warnings(record=True) as caught_mine:
        warnings.simplefilter("always")
        actual = pydicom_mojo.decode_frame(frame, rows, columns, 1, 16)
    assert actual == expected
    assert [str(w.message) for w in caught_mine] == [str(w.message) for w in caught_oracle]


def test_frame_decode_empty_segments_where_geometry_allows():
    # rows*columns == 0: every segment decodes to 0 bytes; garbage offsets
    # that slice empty are fine — mirrors the oracle's degenerate path.
    frame = rle_frame_header(1, [64])  # no segment payload at all
    expected = oracle_decode_frame(frame, 0, 5, 1, 8)
    actual = pydicom_mojo.decode_frame(frame, 0, 5, 1, 8)
    assert actual == expected == b""


# ---------------------------------------------------------------------------
# Frame encode
# ---------------------------------------------------------------------------


def test_frame_encode_header_layout_pinned():
    # 16-bit 2x2 LE pixels -> 2 segments, offsets 64 and 70 (oracle-pinned).
    # Each plane is two rows of 2 distinct bytes: two 2-byte literal packets.
    le_pixels = bytes([2, 1, 4, 3, 6, 5, 8, 7])
    expected = (
        bytes.fromhex("020000004000000046000000" + "00" * 52)
        + bytes([1, 1, 3, 1, 5, 7])  # MSB plane: rows [1,3], [5,7]
        + bytes([1, 2, 4, 1, 6, 8])  # LSB plane: rows [2,4], [6,8]
    )
    assert oracle_encode_frame(le_pixels, 2, 1, 16) == expected
    assert pydicom_mojo.encode_frame(le_pixels, 2, 1, 16) == expected


def test_frame_encode_too_many_segments_raises_like_oracle():
    with pytest.raises(ValueError, match="maximum of 15 segments"):
        oracle_encode_frame(bytes(64), 2, 4, 32)
    with pytest.raises(ValueError, match="maximum of 15 segments"):
        pydicom_mojo.encode_frame(bytes(64), 2, 4, 32)


def test_frame_encode_big_endian_byteorder_raises_like_oracle():
    with pytest.raises(ValueError, match=r"Unsupported option \"byteorder = '>'\""):
        oracle_encode_frame(bytes(8), 2, 1, 16, byteorder=">")
    with pytest.raises(ValueError, match=r"Unsupported option \"byteorder = '>'\""):
        pydicom_mojo.encode_frame(bytes(8), 2, 1, 16, byteorder=">")


@pytest.mark.parametrize("samples,bits", GEOMETRIES)
def test_frame_encode_fuzz_matches_oracle(samples, bits):
    rng = random.Random(f"enc-fuzz-{samples}-{bits}")
    rows, columns = 23, 37
    nbytes = rows * columns * samples * (bits // 8)
    # Mix of noisy and run-heavy content.
    for kind in ("noise", "flat", "mixed"):
        if kind == "noise":
            native = _rng_bytes(rng, nbytes)
        elif kind == "flat":
            native = bytes(nbytes)
        else:
            block = _rng_bytes(rng, 64)
            native = (block * (nbytes // 64 + 1))[:nbytes]
        assert pydicom_mojo.encode_frame(native, columns, samples, bits) == oracle_encode_frame(
            native, columns, samples, bits
        )


def test_frame_encode_lossless_round_trip_through_both_codecs():
    rng = random.Random("rt-both")
    for samples, bits in ((1, 8), (1, 16), (3, 8), (1, 32), (3, 16)):
        rows, columns = 9, 11
        nbytes = rows * columns * samples * (bits // 8)
        native = _rng_bytes(rng, nbytes)
        encoded = pydicom_mojo.encode_frame(native, columns, samples, bits)
        decoded = pydicom_mojo.decode_frame(encoded, rows, columns, samples, bits)
        # decode output is planar configuration 1; re-interleave to compare.
        npix = rows * columns
        ba = bits // 8
        plane = npix * ba
        interleaved = bytearray(nbytes)
        for s in range(samples):
            for i in range(npix):
                interleaved[(i * samples + s) * ba : (i * samples + s + 1) * ba] = decoded[
                    s * plane + i * ba : s * plane + (i + 1) * ba
                ]
        assert bytes(interleaved) == native


# ---------------------------------------------------------------------------
# End-to-end through pydicom's own pipelines (datasets built with pydicom)
# ---------------------------------------------------------------------------


def _registered_plugins():
    from pydicom.pixels.decoders import RLELosslessDecoder
    from pydicom.pixels.encoders import RLELosslessEncoder

    RLELosslessDecoder.add_plugin("pydicom_mojo", ("pydicom_mojo.plugin", "decode_frame"))
    RLELosslessEncoder.add_plugin("pydicom_mojo", ("pydicom_mojo.plugin", "encode_frame"))
    return RLELosslessDecoder, RLELosslessEncoder


def _remove_plugins(decoder, encoder):
    for manager in (decoder, encoder):
        try:
            manager.remove_plugin("pydicom_mojo")
        except ValueError:
            pass


PIPELINE_CASES = [
    pytest.param(48, 64, 1, 16, 3, False, id="ct16-x3"),
    pytest.param(64, 64, 1, 8, 4, True, id="seg8-x4"),
    pytest.param(32, 32, 3, 8, 2, False, id="rgb8-x2"),
    pytest.param(17, 19, 1, 8, 1, False, id="odd-dims"),
]


@pytest.mark.parametrize("rows,columns,samples,bits,frames,segmentation", PIPELINE_CASES)
def test_pydicom_pipeline_parity(rows, columns, samples, bits, frames, segmentation):
    decoder, encoder = _registered_plugins()
    try:
        assert "pydicom_mojo" in decoder.available_plugins
        assert "pydicom_mojo" in encoder.available_plugins

        if samples == 1 and bits == 16:
            frame_bytes = b"".join(ct_like_frame(rows, columns, seed=100 + f) for f in range(frames))
        elif samples == 1 and bits == 8:
            frame_bytes = b"".join(seg_like_frame(rows, columns, seed=200 + f) for f in range(frames))
        else:
            frame_bytes = b"".join(rgb_like_frame(rows, columns, seed=300 + f) for f in range(frames))

        ds = make_dataset(rows, columns, samples, bits, frame_bytes, frames, segmentation)
        oracle_compress(ds)

        # Decode through pydicom's decoder pipeline with each plugin.
        raw_oracle, _ = decoder.as_buffer(ds, raw=True, decoding_plugin="pydicom")
        raw_mojo, _ = decoder.as_buffer(ds, raw=True, decoding_plugin="pydicom_mojo")
        assert bytes(raw_mojo) == bytes(raw_oracle)

        # Encode through pydicom's encoder pipeline with each plugin.
        import copy

        ds_native = make_dataset(rows, columns, samples, bits, frame_bytes, frames, segmentation)
        ds_oracle = copy.deepcopy(ds_native)
        ds_oracle.compress("1.2.840.10008.1.2.5", encoding_plugin="pydicom")
        ds_mojo = copy.deepcopy(ds_native)
        ds_mojo.compress("1.2.840.10008.1.2.5", encoding_plugin="pydicom_mojo")
        assert ds_mojo.PixelData == ds_oracle.PixelData

        # Every encapsulated frame also matches at the standalone API level.
        for frame in split_frames(ds):
            assert pydicom_mojo.decode_frame(frame, rows, columns, samples, bits) == oracle_decode_frame(
                frame, rows, columns, samples, bits
            )
    finally:
        _remove_plugins(decoder, encoder)


def test_plugin_registration_rejected_duplicate():
    decoder, encoder = _registered_plugins()
    try:
        with pytest.raises(ValueError, match="already has a plugin named 'pydicom_mojo'"):
            decoder.add_plugin("pydicom_mojo", ("pydicom_mojo.plugin", "decode_frame"))
    finally:
        _remove_plugins(decoder, encoder)


def test_plugin_dependency_tables():
    from pydicom.uid import RLELossless

    from pydicom_mojo import plugin

    assert RLELossless in plugin.DECODER_DEPENDENCIES
    assert RLELossless in plugin.ENCODER_DEPENDENCIES
    assert plugin.is_available(RLELossless) is True
    assert plugin.is_available("1.2.840.10008.1.2") is False  # Explicit VR LE


def test_oracle_version_pinned():
    # The differential claims are pinned to this exact oracle release.
    assert PYDICOM_VERSION == "3.0.2"

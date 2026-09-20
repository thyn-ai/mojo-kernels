"""Differential tests: pdf_mojo must match the pypdf oracle byte-for-byte.

Run twice by scripts/test_all_pypdf_filters.sh: once against the native
Mojo kernel and once with PDF_MOJO_DISABLE_NATIVE=1 (forced pure-Python
fallback). Both backends must agree with pypdf EXACTLY (byte-equal, zero
tolerance) on every stream.

The oracle is the published PyPI package, pinned to pypdf==6.19.0 in
pixi.toml [pypi-dependencies]; it is never a runtime dependency of pdf_mojo.
The predicted/LZW streams are synthetic: built locally from fixed seeds by
the fresh encoders in test_pypdf_filters_fixtures.py.
"""

from __future__ import annotations

import os
import random

import pytest

import pdf_mojo
from pdf_mojo import PdfFilterError

from test_pypdf_filters_fixtures import (
    PYPDF_VERSION,
    lzw_encode,
    oracle_flate_png,
    oracle_lzw,
    png_bpp,
    png_predict_encode,
    png_row_len,
    tiff_predict_encode,
)

# (columns, colors, bpc) combinations seen in real PDFs, plus sub-byte
# bit depths that exercise the floor(colors*bpc/8) == 0 edge.
PNG_CONFIGS = [
    pytest.param(1, 1, 8, id="c1-g8"),
    pytest.param(7, 1, 8, id="c7-g8"),
    pytest.param(16, 3, 8, id="c16-rgb8"),
    pytest.param(5, 4, 8, id="c5-rgba8"),
    pytest.param(255, 3, 8, id="c255-rgb8"),
    pytest.param(3, 1, 16, id="c3-g16"),
    pytest.param(9, 2, 16, id="c9-ga16"),
    pytest.param(4, 4, 16, id="c4-cmyk16"),
    pytest.param(11, 1, 4, id="c11-g4"),
    pytest.param(13, 2, 4, id="c13-x2-bpc4"),
    pytest.param(8, 3, 4, id="c8-x3-bpc4"),
    pytest.param(8, 1, 2, id="c8-g2-bpp0"),
    pytest.param(17, 1, 1, id="c17-g1-bpp0"),
    pytest.param(3, 3, 1, id="c3-x3-bpc1-bpp0"),
]

PNG_PREDICTORS = [10, 11, 12, 13, 14, 15]


def _expected_backend() -> str:
    # scripts/test_all_pypdf_filters.sh runs the suite once per backend.
    return "fallback" if os.environ.get("PDF_MOJO_DISABLE_NATIVE") == "1" else "native"


def _rng_bytes(rng: random.Random, n: int) -> bytes:
    return bytes(rng.randrange(256) for _ in range(n))


def _random_png_stream(rng: random.Random, columns: int, colors: int, bpc: int, rows: int):
    """Random predicted stream with valid per-row filter bytes (0-4)."""
    row_len = png_row_len(columns, colors, bpc)
    filters = [rng.randrange(5) for _ in range(rows)]
    raw = bytearray()
    for f in filters:
        raw.append(f)
        raw += _rng_bytes(rng, row_len)
    return bytes(raw), row_len


# ---------------------------------------------------------------------------
# PNG / TIFF predictors: decode-compare against the oracle (Mode A)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("columns,colors,bpc", PNG_CONFIGS)
@pytest.mark.parametrize("predictor", PNG_PREDICTORS)
def test_png_predicted_streams_match_oracle(columns, colors, bpc, predictor):
    rng = random.Random(f"png-{predictor}-{columns}-{colors}-{bpc}")
    for rows in (1, 2, 5, 33):
        predicted, _ = _random_png_stream(rng, columns, colors, bpc, rows)
        expected = oracle_flate_png(predicted, columns, colors, bpc, predictor)
        actual = pdf_mojo.decode_png_prediction(predicted, columns, colors, bpc, predictor)
        assert actual == expected
    assert pdf_mojo.get_backend() == _expected_backend()


@pytest.mark.parametrize("columns,colors,bpc", PNG_CONFIGS)
def test_tiff_predictor_streams_match_oracle(columns, colors, bpc):
    rng = random.Random(f"tiff-{columns}-{colors}-{bpc}")
    row_len = png_row_len(columns, colors, bpc)
    for rows in (1, 3, 17):
        predicted = _rng_bytes(rng, row_len * rows)
        expected = oracle_flate_png(predicted, columns, colors, bpc, 2)
        actual = pdf_mojo.decode_png_prediction(predicted, columns, colors, bpc, 2)
        assert actual == expected


def test_predictor_1_passthrough():
    rng = random.Random("p1")
    for n in (0, 1, 7, 100, 1023):  # any length, ragged included
        data = _rng_bytes(rng, n)
        expected = oracle_flate_png(data, 4, 1, 8, 1)
        assert pdf_mojo.decode_png_prediction(data, 4, 1, 8, 1) == expected


def test_png_ragged_rows_zero_padded_like_oracle():
    rng = random.Random("ragged")
    for columns, colors, bpc in (16, 3, 8), (9, 1, 8), (4, 1, 16):
        row_len = png_row_len(columns, colors, bpc)
        for drop in (1, 2, row_len - 1, row_len):  # drop raw bytes of the last row
            predicted, _ = _random_png_stream(rng, columns, colors, bpc, 4)
            truncated = predicted[:-drop]
            expected = oracle_flate_png(truncated, columns, colors, bpc, 15)
            actual = pdf_mojo.decode_png_prediction(truncated, columns, colors, bpc, 15)
            assert actual == expected
            assert len(actual) == 4 * row_len  # padded to whole rows


def test_tiff_ragged_tail_matches_oracle():
    rng = random.Random("tiff-ragged")
    columns, colors, bpc = 16, 3, 8
    row_len = png_row_len(columns, colors, bpc)
    for extra in (1, 2, row_len - 1):
        predicted = _rng_bytes(rng, 3 * row_len + extra)
        expected = oracle_flate_png(predicted, columns, colors, bpc, 2)
        actual = pdf_mojo.decode_png_prediction(predicted, columns, colors, bpc, 2)
        assert actual == expected
        assert len(actual) == len(predicted)  # never padded


def test_png_empty_stream():
    for predictor in (1, 2, 10, 15):
        expected = oracle_flate_png(b"", 8, 1, 8, predictor)
        assert pdf_mojo.decode_png_prediction(b"", 8, 1, 8, predictor) == expected == b""


# ---------------------------------------------------------------------------
# PNG / TIFF predictors: round-trip through the fresh encoder (Mode B)
# ---------------------------------------------------------------------------

RT_CONFIGS = [c for c in PNG_CONFIGS if png_bpp(c.values[1], c.values[2]) >= 1]


@pytest.mark.parametrize("columns,colors,bpc", RT_CONFIGS, ids=[c.id for c in RT_CONFIGS])
def test_png_round_trip_through_encoder(columns, colors, bpc):
    rng = random.Random(f"rt-{columns}-{colors}-{bpc}")
    row_len = png_row_len(columns, colors, bpc)
    for rows in (1, 9, 40):
        original = _rng_bytes(rng, rows * row_len)
        filters = [rng.randrange(5) for _ in range(rows)]
        predicted = png_predict_encode(original, columns, colors, bpc, filters)
        # Fresh encoder -> oracle decode must round-trip (validates the encoder)...
        assert oracle_flate_png(predicted, columns, colors, bpc, 15) == original
        # ...and our decoder must round-trip the same bytes.
        assert pdf_mojo.decode_png_prediction(predicted, columns, colors, bpc, 15) == original


@pytest.mark.parametrize("columns,colors,bpc", RT_CONFIGS, ids=[c.id for c in RT_CONFIGS])
def test_tiff_round_trip_through_encoder(columns, colors, bpc):
    rng = random.Random(f"rtt-{columns}-{colors}-{bpc}")
    row_len = png_row_len(columns, colors, bpc)
    original = _rng_bytes(rng, 12 * row_len)
    predicted = tiff_predict_encode(original, columns, colors, bpc)
    assert oracle_flate_png(predicted, columns, colors, bpc, 2) == original
    assert pdf_mojo.decode_png_prediction(predicted, columns, colors, bpc, 2) == original


# ---------------------------------------------------------------------------
# PNG error handling
# ---------------------------------------------------------------------------


def test_png_filter_byte_above_4_raises_on_both_sides():
    from pypdf.errors import PdfReadError

    predicted = bytes([5, 10, 20, 30])
    with pytest.raises(PdfReadError):
        oracle_flate_png(predicted, 3, 1, 8, 15)
    with pytest.raises(PdfFilterError):
        pdf_mojo.decode_png_prediction(predicted, 3, 1, 8, 15)
    predicted9 = bytes([9, 10, 20, 30])
    with pytest.raises(PdfReadError):
        oracle_flate_png(predicted9, 3, 1, 8, 15)
    with pytest.raises(PdfFilterError):
        pdf_mojo.decode_png_prediction(predicted9, 3, 1, 8, 15)


def test_invalid_png_parameters_raise_value_error():
    with pytest.raises(ValueError):
        pdf_mojo.decode_png_prediction(b"\x00\x01", 0, 1, 8, 12)  # columns
    with pytest.raises(ValueError):
        pdf_mojo.decode_png_prediction(b"\x00\x01", 1, 0, 8, 12)  # colors
    with pytest.raises(ValueError):
        pdf_mojo.decode_png_prediction(b"\x00\x01", 1, 1, 3, 12)  # bpc
    with pytest.raises(ValueError):
        pdf_mojo.decode_png_prediction(b"\x00\x01", 1, 1, 8, 5)  # predictor
    with pytest.raises(ValueError):
        pdf_mojo.decode_png_prediction(b"\x00\x01", 1, 1, 8, 16)  # predictor
    with pytest.raises(TypeError):
        pdf_mojo.decode_png_prediction("not-bytes", 1, 1, 8, 12)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# LZW
# ---------------------------------------------------------------------------

LZW_PAYLOADS = [
    pytest.param(b"", id="empty"),
    pytest.param(b"A", id="one-byte"),
    pytest.param(b"TOBEORNOTTOBEORTOBEORNOT", id="classic"),
    pytest.param(b"ab" * 4000, id="kwkwk-heavy"),
    pytest.param(b"\x00" * 50000, id="extreme-rle"),
    pytest.param(bytes(range(256)) * 40, id="all-bytes"),
    pytest.param(b"the quick brown fox jumps over the lazy dog. " * 500, id="text"),
]


@pytest.mark.parametrize("early_change", (0, 1))
@pytest.mark.parametrize("payload", LZW_PAYLOADS)
def test_lzw_encoded_streams_match_oracle(payload, early_change):
    encoded = lzw_encode(payload, early_change=early_change)
    assert oracle_lzw(encoded, early_change) == payload  # encoder sanity
    assert pdf_mojo.decode_lzw(encoded, early_change) == payload
    assert pdf_mojo.get_backend() == _expected_backend()


@pytest.mark.parametrize("early_change", (0, 1))
@pytest.mark.parametrize("size", (1, 10, 300, 2000, 9000, 70000))
def test_lzw_random_payloads_match_oracle(size, early_change):
    rng = random.Random(f"lzw-rand-{size}")
    payload = _rng_bytes(rng, size)
    encoded = lzw_encode(payload, early_change=early_change)
    assert oracle_lzw(encoded, early_change) == payload
    assert pdf_mojo.decode_lzw(encoded, early_change) == payload


@pytest.mark.parametrize("early_change", (0, 1))
@pytest.mark.parametrize("clear_interval", (1, 100, 2000))
def test_lzw_mid_stream_clears_match_oracle(clear_interval, early_change):
    rng = random.Random(f"lzw-clear-{clear_interval}")
    payload = _rng_bytes(rng, 20000)
    encoded = lzw_encode(payload, early_change=early_change, clear_interval=clear_interval)
    assert oracle_lzw(encoded, early_change) == payload
    assert pdf_mojo.decode_lzw(encoded, early_change) == payload


@pytest.mark.parametrize("early_change", (0, 1))
def test_lzw_truncated_no_eod_matches_oracle(early_change):
    rng = random.Random("lzw-trunc")
    payload = _rng_bytes(rng, 5000)
    encoded = lzw_encode(payload, early_change=early_change, emit_eod=False)
    expected = oracle_lzw(encoded, early_change)
    actual = pdf_mojo.decode_lzw(encoded, early_change)
    assert actual == expected
    # pypdf decodes the zero padding bits as literal 0x00 codes; the payload
    # is a prefix of the result.
    assert expected.startswith(payload)
    assert len(expected) - len(payload) <= 1


@pytest.mark.parametrize("early_change", (0, 1))
def test_lzw_garbage_after_eod_ignored(early_change):
    payload = b"garbage-after-eod test payload " * 20
    encoded = lzw_encode(payload, early_change=early_change) + b"\xde\xad\xbe\xef"
    assert oracle_lzw(encoded, early_change) == payload
    assert pdf_mojo.decode_lzw(encoded, early_change) == payload


def test_lzw_empty_input():
    assert oracle_lzw(b"", 1) == b""
    assert pdf_mojo.decode_lzw(b"", 1) == b""
    assert pdf_mojo.decode_lzw(b"", 0) == b""


def test_lzw_non_literal_first_code_raises_on_both_sides():
    from pypdf.errors import PdfStreamError

    def stream(codes, width=9):
        bits = "".join(f"{c:0{width}b}" for c in codes)
        bits += "0" * ((8 - len(bits) % 8) % 8)
        return int(bits, 2).to_bytes(len(bits) // 8, "big")

    for codes in ([300, 257], [258, 257]):
        bad = stream(codes)
        with pytest.raises(PdfStreamError):
            oracle_lzw(bad, 1)
        with pytest.raises(PdfFilterError):
            pdf_mojo.decode_lzw(bad, 1)


def test_lzw_lenient_kwkwk_code_above_next_matches_oracle():
    # The oracle decodes ANY code >= next-free-entry as prev + prev[0];
    # pin that leniency byte-exactly (here: code 300 when next is 258).
    def stream(codes, width=9):
        bits = "".join(f"{c:0{width}b}" for c in codes)
        bits += "0" * ((8 - len(bits) % 8) % 8)
        return int(bits, 2).to_bytes(len(bits) // 8, "big")

    encoded = stream([256, 65, 300, 257])
    assert oracle_lzw(encoded, 1) == b"AAA"
    assert pdf_mojo.decode_lzw(encoded, 1) == b"AAA"


def test_lzw_width_boundaries_match_oracle():
    # Structured payload that walks the 9->10->11->12 bit transitions with
    # the table filling to the 4096-entry cap, for both EarlyChange values.
    rng = random.Random("lzw-widths")
    words = [_rng_bytes(rng, rng.randrange(1, 5)) for _ in range(800)]
    payload = b" ".join(rng.choice(words) for _ in range(60000))
    for early_change in (0, 1):
        encoded = lzw_encode(payload, early_change=early_change)
        assert oracle_lzw(encoded, early_change) == payload
        assert pdf_mojo.decode_lzw(encoded, early_change) == payload


def test_lzw_width_growth_rule_matches_oracle():
    # Pin the exact width-growth rule: pypdf 6.19.0 grows the code width
    # when the table index reaches (1 << width) - 1 for BOTH EarlyChange 0
    # and 1 (verified at the 9->10 and 10->11 bit transitions by
    # construction: the add after literal 253 makes next == 511, so
    # literal 254 and the EOD are 10-bit codes here).
    def pack(code_widths):
        bits = "".join(f"{c:0{w}b}" for c, w in code_widths)
        bits += "0" * ((8 - len(bits) % 8) % 8)
        return int(bits, 2).to_bytes(len(bits) // 8, "big")

    target = bytes(range(255))
    stream = pack([(256, 9)] + [(v, 9) for v in range(254)] + [(254, 10), (257, 10)])
    for early_change in (0, 1):
        assert oracle_lzw(stream, early_change) == target
        assert pdf_mojo.decode_lzw(stream, early_change) == target


def test_lzw_invalid_early_change_raises():
    with pytest.raises(ValueError):
        pdf_mojo.decode_lzw(b"\x00", 2)
    with pytest.raises(TypeError):
        pdf_mojo.decode_lzw("not-bytes")  # type: ignore[arg-type]


def test_oracle_version_pinned():
    # The differential claims are pinned to this exact oracle release.
    assert PYPDF_VERSION == "6.19.0"

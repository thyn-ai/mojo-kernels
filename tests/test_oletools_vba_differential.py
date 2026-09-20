"""Differential tests: oletools_mojo must match the oletools oracle byte-for-byte.

Run twice by scripts/test_all_oletools_vba.sh: once against the native Mojo
kernel and once with OLETOOLS_MOJO_DISABLE_NATIVE=1 (forced pure-Python
fallback). Both backends must agree with oletools EXACTLY — identical output
bytes, or the identical exception type AND message — on every stream.

The oracle is the published PyPI package, pinned to oletools==0.60.2 (see
scripts/test_all_oletools_vba.sh); it is never a runtime dependency of
oletools_mojo. All streams are synthetic: built locally from fixed seeds per
MS-OVBA 2.4.1 and validated against the oracle directly (oletools ships no
public compressor and no compressed-stream test fixtures in the wheel).
"""

from __future__ import annotations

import os
import random

import pytest

import oletools_mojo
from oletools_mojo._reference import copytoken_bit_count

from test_oletools_vba_fixtures import (
    OLETOOLS_VERSION,
    chunk_header,
    container,
    copy_token,
    literal_chunk,
    oracle_copytoken_help,
    oracle_decompress,
    raw_chunk,
    rng_bytes,
    vba_compress,
    vba_like_source,
)


def _expected_backend() -> str:
    # scripts/test_all_oletools_vba.sh runs the suite once per backend.
    return "fallback" if os.environ.get("OLETOOLS_MOJO_DISABLE_NATIVE") == "1" else "native"


def assert_same_result(stream, label: str = "") -> None:
    """ours(stream) must produce the oracle's output bytes, or an exception
    of the identical type AND message."""
    try:
        expected = oracle_decompress(stream)
        exp_exc = None
    except Exception as exc:  # noqa: BLE001 — comparing error surfaces
        expected, exp_exc = None, exc
    try:
        actual = oletools_mojo.decompress_stream(stream)
        act_exc = None
    except Exception as exc:  # noqa: BLE001
        actual, act_exc = None, exc
    tag = f" [{label}]" if label else ""
    if isinstance(stream, (bytes, bytearray)):
        shown = stream.hex() if len(stream) <= 64 else stream[:64].hex() + "…"
    else:
        shown = repr(stream)
    if exp_exc is None:
        assert act_exc is None, f"oracle OK ({len(expected)}B) but ours raised {act_exc!r}{tag}: {shown}"
        assert actual == expected, f"output mismatch{tag}: {shown}"
    else:
        assert act_exc is not None, (
            f"oracle raised {exp_exc!r} but ours returned {len(actual)}B{tag}: {shown}"
        )
        assert type(act_exc) is type(exp_exc), f"{act_exc!r} vs {exp_exc!r}{tag}: {shown}"
        assert str(act_exc) == str(exp_exc), f"{act_exc!r} vs {exp_exc!r}{tag}: {shown}"


# ---------------------------------------------------------------------------
# Mode B: round-trips through the fresh compressor
# ---------------------------------------------------------------------------

ROUNDTRIP_PAYLOADS = [
    pytest.param(b"A", id="one-byte"),
    pytest.param(b"Hello VBA world!", id="tiny-text"),
    pytest.param(b"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa", id="small-rle"),
    pytest.param(b"\x00" * 5000, id="rle-across-chunks"),
    pytest.param(bytes(range(256)) * 20, id="all-bytes-5k"),
    pytest.param(b"Sub Main()\r\nEnd Sub\r\n" * 400, id="vba-repeated"),
]


@pytest.mark.parametrize("payload", ROUNDTRIP_PAYLOADS)
def test_roundtrip_named_payloads(payload):
    encoded = vba_compress(payload)
    assert oracle_decompress(encoded) == payload  # encoder sanity
    assert oletools_mojo.decompress_stream(encoded) == payload
    assert oletools_mojo.get_backend() == _expected_backend()


def test_roundtrip_empty_payload():
    assert vba_compress(b"") == b"\x01"
    assert oracle_decompress(b"\x01") == b""
    assert oletools_mojo.decompress_stream(b"\x01") == b""


@pytest.mark.parametrize("size", (1, 2, 3, 7, 8, 9, 100, 1000, 3640, 3641, 4095, 4096, 4097, 8191, 8192, 8193, 20000))
def test_roundtrip_incompressible_sizes(size):
    # Random bytes do not compress: exercises RawChunks, including short tails.
    rng = random.Random(f"raw-sizes-{size}")
    payload = rng_bytes(rng, size)
    encoded = vba_compress(payload)
    assert oracle_decompress(encoded) == payload
    assert oletools_mojo.decompress_stream(encoded) == payload


@pytest.mark.parametrize("size", (1, 100, 4096, 4097, 12345))
def test_roundtrip_forced_raw_chunks(size):
    rng = random.Random(f"force-raw-{size}")
    payload = rng_bytes(rng, size)
    encoded = vba_compress(payload, force_raw=True)
    assert oracle_decompress(encoded) == payload
    assert oletools_mojo.decompress_stream(encoded) == payload


def test_roundtrip_vba_like_source_large():
    rng = random.Random("vba-large")
    payload = vba_like_source(rng, 3000)  # ~150 KB of realistic macro text
    encoded = vba_compress(payload)
    assert oracle_decompress(encoded) == payload
    assert oletools_mojo.decompress_stream(encoded) == payload
    assert len(encoded) < len(payload)  # genuinely compressed


def test_roundtrip_text_multi_megabyte():
    rng = random.Random("vba-xl")
    words = [rng_bytes(rng, rng.randrange(2, 9)) for _ in range(500)]
    payload = b" ".join(rng.choice(words) for _ in range(900_000))[: 4 * 1024 * 1024]
    encoded = vba_compress(payload)
    assert oracle_decompress(encoded) == payload
    assert oletools_mojo.decompress_stream(encoded) == payload


# ---------------------------------------------------------------------------
# CopyToken bit geometry: exhaustive over every reachable `difference`
# ---------------------------------------------------------------------------


def _chunk_reaching_difference(
    difference: int, rng, final: tuple[int, int] = (1, 3)
) -> tuple[bytes, bytes]:
    """(stream, expected) for ONE compressed chunk that reaches `difference`
    decompressed bytes, followed by a final CopyToken `final`=(offset, length).

    Small differences use pure literals; large ones fill with RLE copy
    tokens (a chunk holding N literals + one more token overflows the
    4096-byte payload cap around N=3637). The exact output is tracked
    locally (byte-by-byte LZ77 overlap semantics), so the encoder is pinned
    independently of the oracle.
    """
    n_lit = min(difference, 3000)
    fill = difference - n_lit
    if fill in (1, 2):  # copy tokens emit >= 3 bytes; trade literals for fill
        n_lit -= 3 - fill
        fill = 3
    data = rng_bytes(rng, n_lit)
    out = bytearray()  # built up token-by-token during the interleave below

    # Token plan: (kind, value) with value = literal byte or (offset, length)
    # computed against the difference AT that token.
    plan: list[tuple[str, object]] = [("lit", b) for b in data]
    cur = n_lit
    while cur < difference:
        remaining = difference - cur
        bc = max(4, (cur - 1).bit_length())
        max_len = (0xFFFF >> bc) + 3
        m = min(max_len, remaining)
        if remaining - m in (1, 2):
            m = remaining - 3
        plan.append(("copy", (1, m)))
        cur += m
    plan.append(("copy", final))

    # Interleave the plan into 8-slot flag groups, tracking the expected
    # output with the decoder's exact copy semantics.
    payload = bytearray()
    t = 0
    cur = 0  # difference (decompressed bytes so far) at the current token
    while t < len(plan):
        flag_pos = len(payload)
        payload.append(0)
        flags = 0
        for bit in range(8):
            if t >= len(plan):
                break
            kind, value = plan[t]
            if kind == "lit":
                payload.append(value)  # type: ignore[arg-type]
                out.append(value)  # type: ignore[arg-type]
                cur += 1
            else:
                flags |= 1 << bit
                off, ln = value  # type: ignore[misc]
                payload += copy_token(off, ln, cur)
                copy_source = len(out) - off
                for i in range(ln):
                    out.append(out[copy_source + i])
                cur += ln
            t += 1
        payload[flag_pos] = flags
    stream = container(chunk_header(1, len(payload) - 1) + bytes(payload))
    return stream, bytes(out)


@pytest.mark.parametrize("difference", list(range(1, 4097)))
def test_copytoken_at_every_difference(difference):
    rng = random.Random(f"diff-{difference}")
    stream, expected = _chunk_reaching_difference(difference, rng)
    assert oracle_decompress(stream) == expected
    assert oletools_mojo.decompress_stream(stream) == expected


@pytest.mark.parametrize("difference", (1, 2, 3, 15, 16, 17, 255, 256, 257, 4095, 4096))
def test_copytoken_max_offset_and_length(difference):
    # Deepest legal offset (= difference) and a long copy at that position.
    bit_count = max(4, (difference - 1).bit_length())
    max_len = min((0xFFFF >> bit_count) + 3, 64)
    rng = random.Random(f"deep-{difference}")
    stream, expected = _chunk_reaching_difference(
        difference, rng, final=(difference, max_len)
    )
    assert oracle_decompress(stream) == expected
    assert oletools_mojo.decompress_stream(stream) == expected


def test_bit_count_integer_formula_matches_oracle_sweep():
    # The kernel/reference use max(4, ceil_log2_int(d)); the oracle uses
    # float64 math.log(d, 2). They must agree across the whole domain that
    # real streams can reach (a chunk decompresses to at most a few MiB in
    # the wildest malformed cases — far below the float anomaly boundary).
    for d in range(1, 2_000_001):
        assert copytoken_bit_count(d) == oracle_copytoken_help(d, 0)[2], d
    # Documented boundary: the oracle's float computation first diverges at
    # d == 2**29 (an unreachable single-chunk size); pinned here so the
    # unsupported scope is explicit, not accidental.
    assert oracle_copytoken_help(2**29, 0)[2] == 30
    assert copytoken_bit_count(2**29) == 29


# ---------------------------------------------------------------------------
# Pinned edge cases (oracle semantics, one by one)
# ---------------------------------------------------------------------------

EDGE_STREAMS = [
    pytest.param(b"", id="empty->IndexError"),
    pytest.param(b"\x01", id="sig-only"),
    pytest.param(b"\x02abc", id="bad-signature->ValueError"),
    pytest.param(b"\x00", id="zero-sig->ValueError"),
    pytest.param(container(chunk_header(0, 0, signature=0) + b"xx"), id="bad-chunk-signature->ValueError"),
    pytest.param(container(chunk_header(1, 0, signature=0b111) + b"xx"), id="chunk-signature-7->ValueError"),
    pytest.param(container(chunk_header(0, 0xFFE) + b"A" * 5000), id="raw-bad-size->ValueError"),
    pytest.param(container(raw_chunk(bytes(range(256)) * 16)), id="raw-ok"),
    pytest.param(container(raw_chunk(b"AB")), id="raw-truncated-lenient"),
    pytest.param(container(raw_chunk(b"".join([b"R"] * 4096)) + b"\x00"), id="trailing-single-byte->struct.error"),
    # Compressed chunk: literal 'A' then CopyToken(offset=1, length=3).
    pytest.param(container(chunk_header(1, 2) + b"\x02A" + b"\x00\x00"), id="copy-basic"),
    # CopyToken as the very first token (difference == 0 -> math domain).
    pytest.param(container(chunk_header(1, 2) + b"\x01\x00\x00"), id="copytoken-at-zero->ValueError"),
    # CopyToken needs 2 bytes but only 1 remains in the container.
    pytest.param(container(chunk_header(1, 1) + b"\x01\x00"), id="copytoken-truncated->struct.error"),
    # Flag byte consumed, chunk ends before any token ("wasted" flag).
    pytest.param(container(chunk_header(1, 0) + b"\xff"), id="flag-at-end"),
    # 'AB' then CopyToken(offset=3, length=3): negative source wraps.
    pytest.param(container(chunk_header(1, 3) + b"\x04AB\x00\x20"), id="negative-wrap-ok"),
    # 'A' then CopyToken(offset=5, length=3): wrap still out of range.
    pytest.param(container(chunk_header(1, 2) + b"\x02A\x00\x40"), id="negative-wrap->IndexError"),
    # Declared chunk size beyond the container (oracle logs, clamps, decodes).
    pytest.param(container(chunk_header(1, 0xFFF) + b"\x02A\x00\x00"), id="chunk-size-beyond-container"),
    # CopyToken straddling compressed_end into the next chunk's bytes.
    pytest.param(
        container(chunk_header(1, 2) + b"\x02A\x00" + chunk_header(1, 2) + b"\x02B\x00\x00"),
        id="copytoken-straddles-chunk",
    ),
    # Raw chunk followed by a compressed chunk.
    pytest.param(
        container(raw_chunk(b"R" * 4096) + chunk_header(1, 2) + b"\x02A\x00\x00"),
        id="raw-then-compressed",
    ),
    # A compressed chunk decompressing past 4096 bytes (non-spec; oracle allows).
    pytest.param(
        container(chunk_header(1, 5) + b"\x06A\x00\x00\xff\x0f"),
        id="chunk-overdecompresses",
    ),
]


@pytest.mark.parametrize("stream", EDGE_STREAMS)
def test_edge_cases_match_oracle(stream):
    assert_same_result(stream)


def test_chunk_overdecompresses_expected_output():
    # Pin the construction above: literal 'A' + CopyToken(offset=1, len=3)
    # + CopyToken(offset=1, len=4098) -> 'A' * 4102 from ONE chunk.
    stream = container(chunk_header(1, 5) + b"\x06A\x00\x00\xff\x0f")
    expected = b"A" * 4102
    assert oracle_decompress(stream) == expected
    assert oletools_mojo.decompress_stream(stream) == expected


# ---------------------------------------------------------------------------
# Input type handling (identical conversion semantics to the oracle)
# ---------------------------------------------------------------------------


def test_input_types_match_oracle():
    stream = vba_compress(b"bytearray and memoryview inputs")
    assert oletools_mojo.decompress_stream(bytearray(stream)) == oracle_decompress(stream)
    assert oletools_mojo.decompress_stream(memoryview(stream)) == oracle_decompress(stream)
    # str input: bytearray() conversion raises TypeError on both sides.
    with pytest.raises(TypeError) as ours_exc:
        oletools_mojo.decompress_stream("abc")
    with pytest.raises(TypeError) as oracle_exc:
        oracle_decompress("abc")
    assert str(ours_exc.value) == str(oracle_exc.value)
    # int input: bytearray(3) yields 3 NUL bytes -> ValueError on both sides.
    assert_same_result(3)


# ---------------------------------------------------------------------------
# Seeded structured fuzz: deep token-level coverage vs the oracle
# ---------------------------------------------------------------------------


def _fuzz_stream(rng: random.Random) -> bytes:
    mode = rng.randrange(5)
    if mode == 0:  # pure random bytes (usually dies at the signature checks)
        return rng_bytes(rng, rng.randrange(0, 200))
    out = bytearray(b"\x01" if rng.random() < 0.9 else bytes([rng.randrange(256)]))
    for _ in range(rng.randrange(0, 4)):
        kind = rng.randrange(6)
        if kind == 0:  # valid raw chunk (full or short)
            out += raw_chunk(rng_bytes(rng, rng.choice((4096, rng.randrange(1, 4096)))))
        elif kind == 1:  # compressed header + random payload bytes
            n = rng.randrange(1, 100)
            out += chunk_header(1, n - 1) + rng_bytes(rng, n)
        elif kind == 2:  # random 2-byte header + some data
            out += rng_bytes(rng, 2) + rng_bytes(rng, rng.randrange(0, 30))
        elif kind == 3:  # valid literal chunk
            out += literal_chunk(rng_bytes(rng, rng.randrange(1, 200)))
        elif kind == 4:  # copy-token-heavy payload (all-ones flags likely)
            n = rng.randrange(1, 60)
            payload = bytes(rng.choice((0x00, 0xFF, rng.randrange(256))) for _ in range(n))
            out += chunk_header(1, len(payload) - 1) + payload
        else:  # raw header with a wrong size field
            out += chunk_header(0, rng.randrange(0, 0xFFE)) + rng_bytes(rng, rng.randrange(0, 50))
    stream = bytes(out)
    if rng.random() < 0.3 and stream:  # truncate anywhere
        stream = stream[: rng.randrange(len(stream))]
    if rng.random() < 0.3 and stream:  # flip a few bytes
        b = bytearray(stream)
        for _ in range(rng.randrange(1, 4)):
            b[rng.randrange(len(b))] = rng.randrange(256)
        stream = bytes(b)
    return stream


def test_fuzz_structured_streams_match_oracle():
    rng = random.Random("oletools-vba-fuzz-1")
    for i in range(300):
        assert_same_result(_fuzz_stream(rng), label=f"fuzz-{i}")


def test_fuzz_compressed_roundtrip_streams_match_oracle():
    # Fuzz the *payloads* through the fresh compressor (always valid streams,
    # deep in the copy-token space).
    rng = random.Random("oletools-vba-fuzz-2")
    for i in range(60):
        n = rng.choice((rng.randrange(0, 50), rng.randrange(50, 5000), rng.randrange(5000, 40000)))
        alphabet = rng.choice((2, 4, 16, 256))
        payload = bytes(rng.randrange(alphabet) for _ in range(n))
        encoded = vba_compress(payload)
        assert oracle_decompress(encoded) == payload, f"encoder broke on case {i}"
        assert oletools_mojo.decompress_stream(encoded) == payload


def test_oracle_version_pinned():
    # The differential claims are pinned to this exact oracle release.
    assert OLETOOLS_VERSION == "0.60.2"

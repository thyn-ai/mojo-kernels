"""Differential tests: msgpack_mojo must match the published `msgpack` package.

Run twice by `scripts/test_all_msgpack.sh`: once against the native Mojo
kernel and once with MSGPACK_MOJO_DISABLE_NATIVE=1 (forced pure-Python
fallback engine). Both backends must produce byte-identical pack output to
the oracle and equal values on unpack.

Oracle: the published PyPI package `msgpack==1.1.2`, exercised through its
default C backend (`msgpack._cmsgpack`, guarded below); the oracle's
pure-Python fallback (`msgpack.fallback`) is cross-checked on the same
battery. The oracle lives in .oracle-msgpack/ (pip --target; see
scripts/test_all_msgpack.sh) because `pixi run` prunes pip-installed
packages from the pixi env.

Everything here is generated locally from explicit seeds — no network, no
randomness without a fixed seed — so the suite is bit-reproducible on any
machine.
"""

from __future__ import annotations

import datetime as dt
import math
import random
import struct

import pytest

msgpack = pytest.importorskip("msgpack", reason="differential oracle")
from msgpack import fallback as oracle_fallback  # noqa: E402

import msgpack_mojo  # noqa: E402
from msgpack_mojo import ExtType, Timestamp  # noqa: E402

ORACLE_PACK_KW = dict()  # default options unless stated otherwise


def oracle_pack(obj, **kw):
    return msgpack.packb(obj, **kw)


def oracle_unpack(blob, **kw):
    kw.setdefault("strict_map_key", False)
    return msgpack.unpackb(blob, **kw)


def oracle_fallback_pack(obj, **kw):
    return oracle_fallback.Packer(**kw).pack(obj)


def norm(v):
    """Canonicalize values for cross-implementation comparison.

    ExtType/Timestamp are different classes in the two packages; NaN is not
    self-equal. Everything else compares directly.
    """
    if type(v).__name__ == "Timestamp" and hasattr(v, "seconds"):
        return ("Timestamp", v.seconds, v.nanoseconds)
    if type(v).__name__ == "ExtType" and isinstance(v, tuple):
        return ("ExtType", v[0], v[1])
    if isinstance(v, float) and math.isnan(v):
        return "nan"
    # MessagePack arrays unpack as lists by default: tuples and lists are the
    # same wire type, so they normalize identically (use_list=False typing is
    # asserted separately with isinstance checks).
    if isinstance(v, (list, tuple)):
        return ("list", tuple(norm(x) for x in v))
    if isinstance(v, dict):
        return ("dict", tuple((norm(k), norm(x)) for k, x in v.items()))
    return v


def assert_pack_parity(obj, **kw):
    """packb must be byte-exact with the oracle's C backend AND its fallback."""
    ours = msgpack_mojo.packb(obj, **kw)
    ref = oracle_pack(obj, **kw)
    assert ours == ref, f"pack mismatch vs C backend for {obj!r:.80}\n{ours.hex()}\n{ref.hex()}"
    ref_fb = oracle_fallback_pack(obj, **kw)
    assert ours == ref_fb, f"pack mismatch vs oracle fallback for {obj!r:.80}"
    return ours


def assert_unpack_parity(blob, **kw):
    """unpackb must return the same value as the oracle's C backend."""
    ours = norm(msgpack_mojo.unpackb(blob, **kw))
    ref = norm(oracle_unpack(blob, **kw))
    assert ours == ref
    return ours


def assert_roundtrip(obj, **kw):
    """ours -> oracle -> ours, identity at the value level."""
    blob = msgpack_mojo.packb(obj, **kw)
    assert norm(oracle_unpack(blob)) == norm(obj)
    assert norm(msgpack_mojo.unpackb(oracle_pack(obj, **kw))) == norm(obj)
    assert norm(msgpack_mojo.unpackb(blob)) == norm(obj)


# ---------------------------------------------------------------- scalars

INT_BOUNDARY = [
    0, 1, 42, 127, 128, 255, 256, 65535, 65536, 2**31 - 1, 2**31, 2**32 - 1,
    2**32, 2**63 - 1, 2**63, 2**64 - 1,
    -1, -31, -32, -33, -127, -128, -129, -2**15, -2**15 - 1, -2**31,
    -2**31 - 1, -2**63,
]

FLOAT_EXACT = [
    0.0, -0.0, 1.0, -1.0, 1.5, -2.75, 3.141592653589793, 1e300, -1e300,
    5e-324, 2.2250738585072014e-308, float("inf"), float("-inf"), float("nan"),
]


@pytest.mark.parametrize("v", INT_BOUNDARY)
def test_pack_int_boundaries_byte_exact(v):
    assert_pack_parity(v)
    assert_roundtrip(v)


@pytest.mark.parametrize("v", FLOAT_EXACT)
def test_pack_float_byte_exact(v):
    assert_pack_parity(v)
    assert_roundtrip(v)


@pytest.mark.parametrize("seed", range(64))
def test_pack_float_random_bits_byte_exact(seed):
    rng = random.Random(seed)
    v = struct.unpack("<d", rng.getrandbits(64).to_bytes(8, "little"))[0]
    assert_pack_parity(v)
    if v == v:  # NaN unpacks fine but compares unequal; bytes still match
        assert_roundtrip(v)


@pytest.mark.parametrize("seed", range(32))
def test_pack_single_float_random_bits_byte_exact(seed):
    rng = random.Random(10_000 + seed)
    v = struct.unpack("<d", rng.getrandbits(64).to_bytes(8, "little"))[0]
    if math.isnan(v) or math.isinf(v) or abs(v) > 1e38:
        pytest.skip("covered by the fixed f32 battery")
    assert_pack_parity(v, use_single_float=True)


def test_pack_single_float_fixed_battery():
    in_range = [0.0, -0.0, 1.0, -1.0, 1.5, -2.75, 3.141592653589793, 1e-46,
                16777217.0, 0.1, 5e-324, 2.2250738585072014e-308,
                float("inf"), float("-inf"), float("nan")]
    for v in in_range:
        # f32 packing narrows by design: identity is vs the oracle's unpack
        # of the same blob, not vs the original f64 value.
        blob = assert_pack_parity(v, use_single_float=True)
        assert_unpack_parity(blob)
    # Finite doubles outside f32 range saturate to +-inf in the C backend
    # (and in ours); the oracle's own pure-Python fallback raises
    # OverflowError from struct.pack instead, so only the C backend is
    # comparable on this battery.
    for v in [1e300, -1e300, 1e39, -1e39]:
        ours = msgpack_mojo.packb(v, use_single_float=True)
        assert ours == oracle_pack(v, use_single_float=True)
        assert_unpack_parity(ours)


@pytest.mark.parametrize("v", [None, True, False])
def test_pack_nil_bool_byte_exact(v):
    assert_pack_parity(v)
    assert_roundtrip(v)


STR_LEN_BOUNDARY = [0, 1, 31, 32, 33, 255, 256, 65535, 65536]


@pytest.mark.parametrize("n", STR_LEN_BOUNDARY)
def test_pack_str_length_boundaries_byte_exact(n):
    assert_pack_parity("x" * n)
    assert_pack_parity("é" * (n // 2 + 1))  # multi-byte UTF-8
    assert_roundtrip("x" * n)


@pytest.mark.parametrize("n", STR_LEN_BOUNDARY)
def test_pack_bin_length_boundaries_byte_exact(n):
    payload = bytes(range(256)) * (n // 256 + 1)
    assert_pack_parity(payload[:n])
    assert_roundtrip(payload[:n])


def test_pack_unicode_strings():
    for s in ["héllo wörld", "日本語", "\U0001f600\U0001f601", "é" * 100, "mixed \x00 nul"]:
        assert_pack_parity(s)
        assert_roundtrip(s)


# ---------------------------------------------------------------- containers


def test_pack_nested_structures_byte_exact():
    cases = [
        [],
        [[]],
        {},
        [1, "two", 3.0, None, True, b"\x00\xff"],
        {"a": 1, "b": [1, 2, {"c": "d"}], "e": {"f": [None]}},
        [[[[[["deep"]]]]]],
        [list(range(20)), {"k" * i: i for i in range(10)}],
        (1, 2, (3, 4)),
        {str(i): [i, -i, float(i)] for i in range(50)},
    ]
    for obj in cases:
        assert_pack_parity(obj)
        assert_roundtrip(obj)


@pytest.mark.parametrize("seed", range(200))
def test_random_structures_pack_byte_exact_and_roundtrip(seed):
    rng = random.Random(seed)

    def gen(depth):
        r = rng.random()
        if depth >= 5 or r < 0.35:
            t = rng.randrange(6)
            if t == 0:
                return rng.choice([None, True, False])
            if t == 1:
                return rng.choice(
                    [rng.randrange(-2**7, 2**7), rng.randrange(-2**31, 2**31),
                     rng.randrange(-2**63, 2**63), rng.randrange(0, 2**64)]
                )
            if t == 2:
                return struct.unpack(
                    "<d", rng.getrandbits(64).to_bytes(8, "little")
                )[0]
            if t == 3:
                return "".join(
                    rng.choice("abcé日\U0001f600") for _ in range(rng.randrange(40))
                )
            if t == 4:
                return bytes(rng.randrange(256) for _ in range(rng.randrange(40)))
            return rng.randrange(-100, 100)
        t = rng.randrange(3)
        if t == 0:
            return [gen(depth + 1) for _ in range(rng.randrange(7))]
        if t == 1:
            return {
                f"k{i}_{rng.randrange(1000)}": gen(depth + 1)
                for i in range(rng.randrange(7))
            }
        return tuple(gen(depth + 1) for _ in range(rng.randrange(4)))

    obj = gen(0)
    assert_pack_parity(obj)
    assert_roundtrip(obj)


def test_large_payloads_byte_exact():
    cases = [
        bytes(1 << 20),
        "s" * (1 << 20),
        list(range(100_000)),
        {f"key{i}": i for i in range(10_000)},
        [[i, f"s{i}", float(i)] for i in range(5_000)],
    ]
    for obj in cases:
        assert_pack_parity(obj)
        assert_roundtrip(obj)


def test_deep_nesting_roundtrip():
    obj = cur = []
    for _ in range(200):
        nxt = []
        cur.append(nxt)
        cur = nxt
    cur.append("bottom")
    assert_pack_parity(obj)
    assert_roundtrip(obj)


# ---------------------------------------------------------------- ext / timestamp


@pytest.mark.parametrize("n", [1, 2, 3, 4, 8, 16, 17, 255, 256, 65536])
def test_pack_ext_header_boundaries_byte_exact(n):
    ours_obj = ExtType(7, b"d" * n)
    ref_obj = msgpack.ExtType(7, b"d" * n)
    assert msgpack_mojo.packb(ours_obj) == oracle_pack(ref_obj)
    # Same-package round-trip (the oracle packs our ExtType class as a plain
    # tuple; cross-package comparison goes through the wire bytes).
    blob = msgpack_mojo.packb(ours_obj)
    assert norm(msgpack_mojo.unpackb(blob)) == norm(ours_obj)


def test_ext_types_roundtrip():
    for code in (0, 1, 42, 126, 127):
        ours_obj = ExtType(code, bytes(range(code % 5 + 1)))
        ref_obj = msgpack.ExtType(code, bytes(range(code % 5 + 1)))
        assert msgpack_mojo.packb(ours_obj) == oracle_pack(ref_obj)
        assert_unpack_parity(oracle_pack(ref_obj))
        assert norm(msgpack_mojo.unpackb(msgpack_mojo.packb(ours_obj))) == norm(ours_obj)


def test_ext_hook_parity():
    blob = oracle_pack(msgpack.ExtType(9, b"ab"))
    hook = lambda c, d: ("hooked", c, d)  # noqa: E731
    assert msgpack_mojo.unpackb(blob, ext_hook=hook) == oracle_unpack(blob, ext_hook=hook)


def test_ext_code_out_of_range_raises():
    # Wire code 200 (i8 -56) is not a valid ExtType code on unpack.
    with pytest.raises(ValueError):
        msgpack_mojo.unpackb(bytes.fromhex("c701c8aa"))
    with pytest.raises(ValueError):
        oracle_unpack(bytes.fromhex("c701c8aa"))
    with pytest.raises(ValueError):
        ExtType(128, b"")
    with pytest.raises(ValueError):
        msgpack.ExtType(128, b"")


TIMESTAMPS = [
    (0, 0),
    (42, 999),
    (2**32 - 1, 0),
    (2**32, 0),
    (2**34 - 1, 999_999_999),
    (2**34, 123),
    (-1, 1),
    (-2**40, 987_654_321),
]


@pytest.mark.parametrize("sec,nsec", TIMESTAMPS)
def test_timestamp_pack_byte_exact_and_unpack(sec, nsec):
    ours_obj = Timestamp(sec, nsec)
    ref_obj = msgpack.Timestamp(seconds=sec, nanoseconds=nsec)
    assert msgpack_mojo.packb(ours_obj) == oracle_pack(ref_obj)
    blob = oracle_pack(ref_obj)
    for ts_opt in (0, 1, 2):
        assert_unpack_parity(blob, timestamp=ts_opt)
    # timestamp=3 converts to datetime; seconds outside the datetime range
    # raise OverflowError on both sides (checked here explicitly).
    try:
        expected = oracle_unpack(blob, timestamp=3)
    except OverflowError:
        with pytest.raises(OverflowError):
            msgpack_mojo.unpackb(blob, timestamp=3)
    else:
        assert msgpack_mojo.unpackb(blob, timestamp=3) == expected
    assert norm(msgpack_mojo.unpackb(msgpack_mojo.packb(ours_obj))) == norm(ours_obj)


def test_datetime_pack_parity():
    aware = dt.datetime(2020, 3, 14, 15, 9, 26, 535897, tzinfo=dt.timezone.utc)
    off = dt.timezone(dt.timedelta(hours=-5, minutes=30))
    for value in (aware, aware.astimezone(off)):
        assert_pack_parity(value, datetime=True)
        assert_unpack_parity(oracle_pack(value, datetime=True), timestamp=3)
    with pytest.raises(ValueError):
        msgpack_mojo.packb(dt.datetime(2020, 1, 1), datetime=True)
    with pytest.raises(ValueError):
        oracle_pack(dt.datetime(2020, 1, 1), datetime=True)
    with pytest.raises(TypeError):
        msgpack_mojo.packb(aware)  # datetime=False by default
    with pytest.raises(TypeError):
        oracle_pack(aware)


# ---------------------------------------------------------------- pack options


def test_pack_use_bin_type_false():
    # str round-trips to str; bytes go to the legacy raw family and unpack as
    # str (the oracle does the same), so bytes cases compare unpack parity
    # rather than identity.
    for obj in ["a" * 31, "a" * 32, "a" * 255, "a" * 256, "é" * 20]:
        assert_pack_parity(obj, use_bin_type=False)
        assert_roundtrip(obj, use_bin_type=False)
    for obj in [b"\x00" * 31, b"\x00" * 32, b"\x00" * 255, b"\x00" * 256]:
        blob = assert_pack_parity(obj, use_bin_type=False)
        assert_unpack_parity(blob)


def test_pack_strict_types():
    class MyInt(int):
        pass

    class MyStr(str):
        pass

    class MyList(list):
        pass

    for obj in [MyInt(5), MyStr("s"), MyList([1, 2])]:
        assert_pack_parity(obj)  # subclasses pack as base type by default
        with pytest.raises(TypeError):
            msgpack_mojo.packb(obj, strict_types=True)
        with pytest.raises(TypeError):
            oracle_pack(obj, strict_types=True)


def test_pack_default_hook():
    class Weird:
        pass

    hook = lambda o: {"weird": True}  # noqa: E731
    assert_pack_parity(Weird(), default=hook)
    with pytest.raises(TypeError, match="can not serialize"):
        msgpack_mojo.packb(Weird())
    with pytest.raises(TypeError, match="can not serialize"):
        oracle_pack(Weird())


def test_pack_unicode_errors():
    s = "bad surrogate \udcff"
    assert_pack_parity(s, unicode_errors="replace")
    with pytest.raises(UnicodeEncodeError):
        msgpack_mojo.packb(s)
    with pytest.raises(UnicodeEncodeError):
        oracle_pack(s)


def test_pack_bytearray_memoryview():
    assert_pack_parity(bytearray(b"abc"))
    assert_pack_parity(memoryview(b"abc"))
    assert_roundtrip(bytearray(b"abc"))


def test_pack_int_overflow():
    for v in (2**64, -(2**63) - 1, 10**100):
        with pytest.raises(OverflowError):
            msgpack_mojo.packb(v)
        with pytest.raises(OverflowError):
            oracle_pack(v)


def test_pack_unknown_kwarg_rejected():
    with pytest.raises(TypeError):
        msgpack_mojo.packb(1, no_such_option=1)
    with pytest.raises(TypeError):
        oracle_pack(1, no_such_option=1)


# ---------------------------------------------------------------- unpack options


def test_unpack_raw():
    blob = oracle_pack({"k": "v", "b": b"\x01"})
    assert_unpack_parity(blob, raw=True)


def test_unpack_use_list_false():
    blob = oracle_pack([1, [2, 3], {"k": [4]}])
    ours = msgpack_mojo.unpackb(blob, use_list=False)
    ref = oracle_unpack(blob, use_list=False)
    assert norm(ours) == norm(ref)
    assert isinstance(ours, tuple) and isinstance(ours[1], tuple)


def test_unpack_strict_map_key():
    for key in (1, 1.5, None, True, b"bytes-ok"):
        blob = oracle_pack({key: 1})
        if isinstance(key, bytes):
            assert_unpack_parity(blob)  # bytes keys are allowed
            continue
        with pytest.raises(ValueError, match="not allowed for map key"):
            msgpack_mojo.unpackb(blob)
        with pytest.raises(ValueError):
            oracle_unpack(blob, strict_map_key=True)
        assert_unpack_parity(blob, strict_map_key=False)
    # An array key unpacks to an unhashable list: strict mode rejects it with
    # ValueError; non-strict mode fails hashing on both sides.
    blob = oracle_pack({(1, 2): 1})
    with pytest.raises(ValueError, match="not allowed for map key"):
        msgpack_mojo.unpackb(blob)
    with pytest.raises((TypeError, ValueError)):
        msgpack_mojo.unpackb(blob, strict_map_key=False)
    with pytest.raises((TypeError, ValueError)):
        oracle_unpack(blob)


def test_unpack_unicode_errors():
    blob = b"\xd9\x03a\xffb"  # str(3) with an invalid UTF-8 byte
    assert_unpack_parity(blob, unicode_errors="replace")
    with pytest.raises(UnicodeDecodeError):
        msgpack_mojo.unpackb(blob)
    with pytest.raises(UnicodeDecodeError):
        oracle_unpack(blob)


def test_unpack_max_lengths():
    with pytest.raises(ValueError, match="exceeds max_str_len"):
        msgpack_mojo.unpackb(oracle_pack("abc"), max_str_len=2)
    with pytest.raises(ValueError, match="exceeds max_bin_len"):
        msgpack_mojo.unpackb(oracle_pack(b"abc"), max_bin_len=2)
    with pytest.raises(ValueError, match="exceeds max_array_len"):
        msgpack_mojo.unpackb(oracle_pack([1, 2, 3]), max_array_len=2)
    with pytest.raises(ValueError, match="exceeds max_map_len"):
        msgpack_mojo.unpackb(oracle_pack({"a": 1, "b": 2}), max_map_len=1)
    with pytest.raises(ValueError, match="exceeds max_ext_len"):
        msgpack_mojo.unpackb(oracle_pack(msgpack.ExtType(1, b"abcd")), max_ext_len=3)
    for blob, kw in [
        (oracle_pack("abc"), dict(max_str_len=2)),
        (oracle_pack([1, 2, 3]), dict(max_array_len=2)),
    ]:
        with pytest.raises(ValueError):
            oracle_unpack(blob, **kw)
    # Limits exactly at the boundary pass.
    assert msgpack_mojo.unpackb(oracle_pack("abc"), max_str_len=3) == "abc"


def test_unpack_accepts_buffer_types():
    blob = oracle_pack({"k": [1, 2]})
    assert msgpack_mojo.unpackb(bytearray(blob)) == oracle_unpack(blob)
    assert msgpack_mojo.unpackb(memoryview(blob)) == oracle_unpack(blob)


# ---------------------------------------------------------------- error paths


def test_unpack_reserved_prefix_format_error():
    with pytest.raises(msgpack_mojo.FormatError):
        msgpack_mojo.unpackb(b"\xc1")
    with pytest.raises(ValueError):
        oracle_unpack(b"\xc1")
    # FormatError must be a ValueError, like the oracle's.
    assert issubclass(msgpack_mojo.FormatError, ValueError)


def test_unpack_truncated_inputs():
    for blob in (b"", b"\x92\x01", b"\xd9\x05ab", b"\xcb\x00\x00", b"\xc5\x00"):
        with pytest.raises(ValueError):
            msgpack_mojo.unpackb(blob)
        with pytest.raises(ValueError):
            oracle_unpack(blob)


def test_unpack_extra_data():
    with pytest.raises(msgpack_mojo.ExtraData):
        msgpack_mojo.unpackb(b"\x01\x01")
    with pytest.raises(ValueError):
        oracle_unpack(b"\x01\x01")
    assert issubclass(msgpack_mojo.ExtraData, ValueError)


def test_unpack_unhashable_map_key():
    blob = bytes.fromhex("81810102")  # { [1]: 2 }
    with pytest.raises((TypeError, ValueError)):
        msgpack_mojo.unpackb(blob, strict_map_key=False)
    with pytest.raises((TypeError, ValueError)):
        oracle_unpack(blob)


def test_unsupported_hooks_raise_loudly():
    with pytest.raises(NotImplementedError):
        msgpack_mojo.unpackb(b"\x01", object_hook=dict)
    with pytest.raises(NotImplementedError):
        msgpack_mojo.unpackb(b"\x01", list_hook=list)
    with pytest.raises(NotImplementedError):
        msgpack_mojo.unpackb(b"\x01", object_pairs_hook=list)


# ---------------------------------------------------------------- oracle sanity


def test_oracle_c_backend_is_active():
    """The differential suite is only meaningful against the real default."""
    assert "_cmsgpack" in msgpack.Packer.__module__, (
        "oracle msgpack must use its C backend for this suite"
    )


def test_oracle_backends_agree_on_battery():
    """Guard: the oracle's own C and pure-Python backends agree, so comparing
    against either is valid."""
    battery = [None, True, 0, 127, 128, -33, 2**40, -2**40, 1.5, "héllo",
               b"\x00\xff", [1, 2], {"a": [1]}, (1, 2), "x" * 300]
    for obj in battery:
        assert oracle_pack(obj) == oracle_fallback_pack(obj)


def test_dumps_loads_aliases_and_file_io(tmp_path):
    doc = {"project": "msgpack-mojo", "n": [1, 2, 3]}
    assert msgpack_mojo.dumps(doc) == oracle_pack(doc)
    assert msgpack_mojo.loads(msgpack_mojo.dumps(doc)) == doc
    p = tmp_path / "doc.msgpack"
    with open(p, "wb") as f:
        msgpack_mojo.dump(doc, f)
    assert p.read_bytes() == oracle_pack(doc)
    with open(p, "rb") as f:
        assert msgpack_mojo.load(f) == doc

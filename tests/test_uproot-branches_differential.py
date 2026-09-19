"""Differential tests: uproot_mojo must match uproot buffer-for-buffer.

Run twice by scripts/test_all_uproot-branches.sh: once against the native
Mojo kernel and once with UPROOT_MOJO_DISABLE_NATIVE=1 (forced pure-Python
fallback). Both backends must agree with uproot exactly: same offsets, same
content bytes, same values.

File-level fixtures are real ROOT files created here with uproot.recreate
(self-contained, zlib compression). The ROOT-native collection-header
layouts (per-entry 10-byte headers; vector<string>) cannot be written by
uproot.recreate (its writer only emits NumPy-dtype jagged contents and
top-level std::string), so those are validated at the basket-bytes level
against uproot's own container deserializers (uproot.containers.AsVector /
AsString — the interpreted per-entry path this kernel replaces).
"""

from __future__ import annotations

import os
import struct

import numpy as np
import pytest
import uproot
import awkward as ak

import uproot_mojo
from uproot_mojo import _fallback
from uproot_mojo._rootfile import (
    BasketDataError,
    RootFileError,
    UnsupportedBranchError,
)

# ---------------------------------------------------------------------------
# Fixture data (deterministic, no randomness)
# ---------------------------------------------------------------------------

DATA_I32 = [[1, 2, 3], [], [7], list(range(-5, 5)), [2**31 - 1, -2**31], [0]]
DATA_I64 = [[1, 2**40], [], [-2**60], [5] * 4, [0], [2**63 - 1, -2**63, 7]]
DATA_F32 = [[1.5, -2.25], [], [3.0] * 7, [0.5], [1.0, 2.0], [np.inf, -np.inf, np.nan]]
DATA_F64 = [[1.5, -2.25, 1e300], [], [np.pi], [4.5] * 6, [7.25], [1e-300, -0.0]]
DATA_STR = [
    "hello",
    "",
    "x" * 300,  # long TString (0xFF + uint32 length)
    "world",
    "y" * 254,  # longest short TString
    "z" * 255,  # shortest long TString
]


def _write_main_fixture(path):
    with uproot.recreate(path, compression=uproot.ZLIB(4)) as f:
        f.mktree(
            "tree",
            {
                "vi32": "var * int32",
                "vi64": "var * int64",
                "vf32": "var * float32",
                "vf64": "var * float64",
                "s": "string",
            },
        )
        # Two extend calls -> two baskets per branch (multi-basket coverage).
        f["tree"].extend(
            {
                "vi32": DATA_I32[:4],
                "vi64": DATA_I64[:4],
                "vf32": DATA_F32[:4],
                "vf64": DATA_F64[:4],
                "s": DATA_STR[:4],
            }
        )
        f["tree"].extend(
            {
                "vi32": DATA_I32[4:],
                "vi64": DATA_I64[4:],
                "vf32": DATA_F32[4:],
                "vf64": DATA_F64[4:],
                "s": DATA_STR[4:],
            }
        )


def _write_two_trees_fixture(path):
    with uproot.recreate(path, compression=uproot.ZLIB(1)) as f:
        f.mktree("one", {"v": "var * int32"})
        f["one"].extend({"v": [[1], [2, 3]]})
        f.mktree("two", {"v": "var * int32"})
        f["two"].extend({"v": [[9], []]})


def _write_uncompressed_fixture(path):
    with uproot.recreate(path, compression=None) as f:
        f.mktree("tree", {"vi32": "var * int32", "s": "string"})
        f["tree"].extend({"vi32": [[10, 20], [], [30]], "s": ["aa", "", "bbb"]})


@pytest.fixture(scope="session")
def main_file(tmp_path_factory):
    path = tmp_path_factory.mktemp("fixtures") / "main.root"
    _write_main_fixture(path)
    return path


@pytest.fixture(scope="session")
def two_trees_file(tmp_path_factory):
    path = tmp_path_factory.mktemp("fixtures") / "two_trees.root"
    _write_two_trees_fixture(path)
    return path


@pytest.fixture(scope="session")
def uncompressed_file(tmp_path_factory):
    path = tmp_path_factory.mktemp("fixtures") / "uncompressed.root"
    _write_uncompressed_fixture(path)
    return path


def _expected_backend() -> str:
    # scripts/test_all_uproot-branches.sh runs the suite once per backend.
    return "fallback" if os.environ.get("UPROOT_MOJO_DISABLE_NATIVE") == "1" else "native"


def _assert_nested_values_equal(got, want):
    """Exact equality on nested lists, with NaN comparing equal to NaN."""
    assert len(got) == len(want)
    for got_entry, want_entry in zip(got, want):
        assert len(got_entry) == len(want_entry)
        for g, w in zip(got_entry, want_entry):
            if isinstance(w, float) and np.isnan(w):
                assert isinstance(g, float) and np.isnan(g)
            else:
                assert g == w


def _uproot_buffers(path, branch):
    """uproot's own offsets/content buffers for a branch (via awkward).

    Buffer keys are normalized by stripping any "partN-" partition prefix,
    so both partitioned and consolidated arrays work.
    """
    arr = uproot.open(path)["tree"][branch].array(library="ak")
    _form, _length, container = ak.to_buffers(ak.to_packed(arr))
    normalized = {}
    for key, value in container.items():
        short = key.split("-", 1)[1] if key.startswith("part") else key
        normalized[short] = value
    return arr, normalized


# ---------------------------------------------------------------------------
# File-level differential tests vs uproot (uproot.recreate fixtures)
# ---------------------------------------------------------------------------

NUMERIC_CASES = [
    pytest.param("vi32", "int32", DATA_I32, id="vector<int32>"),
    pytest.param("vi64", "int64", DATA_I64, id="vector<int64>"),
    pytest.param("vf32", "float32", DATA_F32, id="vector<float32>"),
    pytest.param("vf64", "float64", DATA_F64, id="vector<float64>"),
]


@pytest.mark.parametrize("branch,dtype,expected", NUMERIC_CASES)
def test_numeric_branch_matches_uproot(main_file, branch, dtype, expected):
    result = uproot_mojo.read_branch(main_file, branch, dtype=dtype)
    assert result.backend == _expected_backend()
    assert result.kind == "numeric"

    arr, container = _uproot_buffers(main_file, branch)
    # Offsets identical to uproot's (int64, cumulative items).
    np.testing.assert_array_equal(result.offsets, container["node0-offsets"])
    # Content dtype and bytes identical to uproot's delivered (native) buffers.
    assert result.content.dtype == container["node1-data"].dtype
    assert result.content.tobytes() == container["node1-data"].tobytes()
    # Values identical (the f32 fixture contains NaN, which compares
    # unequal to itself, hence the NaN-aware helper).
    _assert_nested_values_equal(result.to_list(), expected)
    _assert_nested_values_equal(ak.to_list(result.to_awkward()), ak.to_list(arr))


def test_std_string_branch_matches_uproot(main_file):
    result = uproot_mojo.read_branch(main_file, "s")
    assert result.backend == _expected_backend()
    assert result.kind == "string"

    arr, container = _uproot_buffers(main_file, "s")
    np.testing.assert_array_equal(result.offsets, container["node0-offsets"])
    assert result.content.tobytes() == container["node1-data"].tobytes()
    assert result.to_list() == DATA_STR
    assert ak.to_list(result.to_awkward()) == ak.to_list(arr)


def test_uncompressed_file_matches_uproot(uncompressed_file):
    result = uproot_mojo.read_branch(uncompressed_file, "vi32", dtype="int32")
    assert result.to_list() == [[10, 20], [], [30]]
    arr = uproot.open(uncompressed_file)["tree"]["vi32"].array(library="ak")
    assert ak.to_list(result.to_awkward()) == ak.to_list(arr)
    strings = uproot_mojo.read_branch(uncompressed_file, "s")
    assert strings.to_list() == ["aa", "", "bbb"]


def test_tree_disambiguation(two_trees_file):
    with pytest.raises(UnsupportedBranchError, match="multiple trees"):
        uproot_mojo.read_branch(two_trees_file, "v", dtype="int32")
    one = uproot_mojo.read_branch(two_trees_file, "v", dtype="int32", tree="one")
    two = uproot_mojo.read_branch(two_trees_file, "v", dtype="int32", tree="two")
    assert one.to_list() == [[1], [2, 3]]
    assert two.to_list() == [[9], []]


def test_missing_branch_raises(main_file):
    with pytest.raises(UnsupportedBranchError, match="no TBasket keys"):
        uproot_mojo.read_branch(main_file, "nonexistent", dtype="int32")


def test_counter_branch_out_of_scope(main_file):
    # The auto-generated int32 counter branch is rectilinear (no offsets
    # table): explicitly out of scope.
    with pytest.raises(UnsupportedBranchError, match="entry-offset"):
        uproot_mojo.read_branch(main_file, "nvi32", dtype="int32")


def test_not_a_root_file(tmp_path):
    bogus = tmp_path / "not.root"
    bogus.write_bytes(b"definitely not a root file")
    with pytest.raises(RootFileError):
        uproot_mojo.read_branch(bogus, "vi32", dtype="int32")


def test_numeric_branch_without_dtype_hint(main_file):
    with pytest.raises(UnsupportedBranchError, match="dtype"):
        uproot_mojo.read_branch(main_file, "vi32")


def test_bad_dtype_rejected(main_file):
    with pytest.raises(UnsupportedBranchError, match="unsupported dtype"):
        uproot_mojo.read_branch(main_file, "vi32", dtype="int16")


def test_dtype_accepts_numpy_and_aliases(main_file):
    a = uproot_mojo.read_branch(main_file, "vi32", dtype=np.dtype("i4"))
    b = uproot_mojo.read_branch(main_file, "vi32", dtype="int32")
    np.testing.assert_array_equal(a.offsets, b.offsets)
    assert a.content.tobytes() == b.content.tobytes()


# ---------------------------------------------------------------------------
# Basket-level differential tests vs uproot's container deserializers
# (ROOT-native per-entry collection headers — the objects.py path)
# ---------------------------------------------------------------------------


def _vector_entry(items, dtype):
    payload = struct.pack(">hI", 3, len(items)) + np.asarray(
        items, dtype=np.dtype(dtype).newbyteorder(">")
    ).tobytes()
    return struct.pack(">I", 0x40000000 | len(payload)) + payload


def _tstring(raw: bytes) -> bytes:
    if len(raw) < 255:
        return bytes([len(raw)]) + raw
    return b"\xff" + struct.pack(">I", len(raw)) + raw


def _vector_string_entry(strings):
    payload = struct.pack(">hI", 3, len(strings)) + b"".join(
        _tstring(s.encode("utf-8")) for s in strings
    )
    return struct.pack(">I", 0x40000000 | len(payload)) + payload


def _walk(data: bytes, borders, mode, itemsize=0):
    """Dispatch to the backend under test (native or forced fallback)."""
    if _expected_backend() == "native":
        from uproot_mojo import _native

        return _native.walk_basket(data, borders, mode, itemsize, "synthetic test basket")
    return _fallback.walk_basket(data, borders, mode, itemsize, "synthetic test basket")


class _FakeFile:
    file_path = "synthetic-vector.root"


def _oracle_vector(entry_bytes: bytes, dtype):
    """uproot's own std::vector<T> deserializer (containers.AsVector)."""
    from uproot.containers import AsVector
    from uproot.source.chunk import Chunk
    from uproot.source.cursor import Cursor

    model = AsVector(True, np.dtype(dtype).newbyteorder(">"))
    chunk = Chunk.wrap(None, entry_bytes)
    vector = model.read(
        chunk, Cursor(0), {"reading": True}, _FakeFile(), _FakeFile(), None
    )
    # AsVector.read returns an STLVector; extract the item values.
    return [vector[i] for i in range(len(vector))]


def _oracle_vector_string(entry_bytes: bytes):
    """uproot's own std::vector<std::string> deserializer."""
    from uproot.containers import AsString, AsVector
    from uproot.source.chunk import Chunk
    from uproot.source.cursor import Cursor

    model = AsVector(True, AsString(False))
    chunk = Chunk.wrap(None, entry_bytes)
    vector = model.read(
        chunk, Cursor(0), {"reading": True}, _FakeFile(), _FakeFile(), None
    )
    return [vector[i] for i in range(len(vector))]


VECTOR_B_CASES = [
    pytest.param("int32", [[1, 2, 3], [], [-2**31], [0] * 17, [2**31 - 1]], id="int32"),
    pytest.param("int64", [[2**40, -1], [], [2**63 - 1, -2**63], [5]], id="int64"),
    pytest.param("float32", [[1.5, -2.25], [], [np.pi] * 9], id="float32"),
    pytest.param("float64", [[1e300, -1e-300], [], [np.pi], [0.0] * 33], id="float64"),
]


@pytest.mark.parametrize("dtype_name,entries", VECTOR_B_CASES)
def test_root_native_vector_layout_matches_oracle(dtype_name, entries):
    """Collection-header vector<T> baskets vs uproot's AsVector deserializer."""
    data = b"".join(_vector_entry(e, dtype_name) for e in entries)
    borders = np.zeros(len(entries) + 1, dtype=np.int64)
    pos = 0
    for i, e in enumerate(entries):
        pos += len(_vector_entry(e, dtype_name))
        borders[i + 1] = pos

    detected, offsets, content, string_offsets = _walk(
        data, borders, _fallback.MODE_AUTO_NUM, np.dtype(dtype_name).itemsize
    )
    assert detected == _fallback.MODE_NUM_HDR
    assert string_offsets is None

    oracle_entries = [_oracle_vector(_vector_entry(e, dtype_name), dtype_name) for e in entries]
    # np.concatenate would drop the big-endian byte order; join bytes directly.
    oracle_flat_bytes = b"".join(
        np.asarray(o, dtype=np.dtype(dtype_name).newbyteorder(">")).tobytes()
        for o in oracle_entries
    )
    expected_offsets = np.zeros(len(entries) + 1, dtype=np.int64)
    for i, o in enumerate(oracle_entries):
        expected_offsets[i + 1] = expected_offsets[i] + len(o)

    np.testing.assert_array_equal(offsets, expected_offsets)
    assert content == oracle_flat_bytes


VECTOR_STRING_ENTRIES = [
    ["alpha", "beta", ""],
    [],
    ["x" * 300],  # long TString inside a vector
    ["y" * 254, "z" * 255],
    ["unicode-héllo", "ascii"],
    ["only"],
]


def test_root_native_vector_string_layout_matches_oracle():
    """Collection-header vector<string> baskets vs uproot's AsVector(AsString)."""
    data = b"".join(_vector_string_entry(e) for e in VECTOR_STRING_ENTRIES)
    borders = np.zeros(len(VECTOR_STRING_ENTRIES) + 1, dtype=np.int64)
    pos = 0
    for i, e in enumerate(VECTOR_STRING_ENTRIES):
        pos += len(_vector_string_entry(e))
        borders[i + 1] = pos

    detected, offsets, content, string_offsets = _walk(
        data, borders, _fallback.MODE_AUTO_STR, 0
    )
    assert detected == _fallback.MODE_STR_VEC

    oracle_entries = [
        list(_oracle_vector_string(_vector_string_entry(e)))
        for e in VECTOR_STRING_ENTRIES
    ]
    assert oracle_entries == VECTOR_STRING_ENTRIES  # sanity: oracle reads them back

    expected_item_offsets = np.zeros(len(VECTOR_STRING_ENTRIES) + 1, dtype=np.int64)
    flat = []
    for i, e in enumerate(VECTOR_STRING_ENTRIES):
        expected_item_offsets[i + 1] = expected_item_offsets[i] + len(e)
        flat.extend(e)
    np.testing.assert_array_equal(offsets, expected_item_offsets)

    expected_string_offsets = np.zeros(len(flat) + 1, dtype=np.int64)
    expected_bytes = bytearray()
    for j, s in enumerate(flat):
        expected_bytes += s.encode("utf-8")
        expected_string_offsets[j + 1] = len(expected_bytes)
    np.testing.assert_array_equal(string_offsets, expected_string_offsets)
    assert content == bytes(expected_bytes)


def test_auto_str_detects_std_string_not_vector():
    """A std::string basket (no collection header) must detect as STR_ONE."""
    strings = ["one", "", "t" * 300]
    data = b"".join(_tstring(s.encode("utf-8")) for s in strings)
    borders = np.zeros(len(strings) + 1, dtype=np.int64)
    pos = 0
    for i, s in enumerate(strings):
        pos += len(_tstring(s.encode("utf-8")))
        borders[i + 1] = pos
    detected, offsets, content, string_offsets = _walk(
        data, borders, _fallback.MODE_AUTO_STR, 0
    )
    assert detected == _fallback.MODE_STR_ONE
    assert string_offsets is None
    np.testing.assert_array_equal(offsets, np.array([0, 3, 3, 303], dtype=np.int64))
    assert content == b"one" + b"t" * 300


# ---------------------------------------------------------------------------
# Error handling: corrupt baskets fail with stable codes on both backends
# ---------------------------------------------------------------------------


def _expect_error(data, borders, mode, itemsize, code, match):
    with pytest.raises(BasketDataError, match=match) as excinfo:
        _walk(data, borders, mode, itemsize)
    assert excinfo.value.code == code


def test_corrupt_bytecount_rejected():
    entry = struct.pack(">I", 0x40000000 | 999) + struct.pack(">hI", 3, 1) + b"\x00" * 4
    # bytecount claims 999 bytes after the field; entry is only 10 bytes long.
    borders = np.array([0, len(entry)], dtype=np.int64)
    _expect_error(
        entry, borders, _fallback.MODE_AUTO_NUM, 4, _fallback.ERR_BYTECOUNT, "ERR_BYTECOUNT"
    )


def test_truncated_entry_rejected():
    data = b"\x00\x00\x00\x01"  # 4 bytes; cannot hold an int32x3 entry
    borders = np.array([0, 7], dtype=np.int64)  # border beyond data
    _expect_error(
        data, borders, _fallback.MODE_AUTO_NUM, 4, _fallback.ERR_BORDERS, "ERR_BORDERS"
    )


def test_raw_count_mismatch_rejected():
    data = b"\x00" * 6  # 6 bytes is not a multiple of 4
    borders = np.array([0, 6], dtype=np.int64)
    # 6 % 4 == 2 -> looks like a header candidate; too short for one though.
    _expect_error(
        data, borders, _fallback.MODE_AUTO_NUM, 4, _fallback.ERR_TRUNCATED, "ERR_TRUNCATED"
    )
    _expect_error(
        data, borders, _fallback.MODE_NUM_RAW, 4, _fallback.ERR_COUNT, "ERR_COUNT"
    )


def test_bad_string_prefix_rejected():
    data = b"\x0aabc"  # claims 10 bytes, only 3 follow
    borders = np.array([0, 4], dtype=np.int64)
    # AUTO_STR tries vector first: the 4-byte entry is shorter than a header.
    _expect_error(
        data, borders, _fallback.MODE_AUTO_STR, 0, _fallback.ERR_TRUNCATED, "ERR_TRUNCATED"
    )


def test_vector_string_count_mismatch_rejected():
    # Declares 3 strings but only 1 present; walk cannot land on the border.
    payload = struct.pack(">hI", 3, 3) + _tstring(b"only")
    entry = struct.pack(">I", 0x40000000 | len(payload)) + payload
    borders = np.array([0, len(entry)], dtype=np.int64)
    _expect_error(
        entry, borders, _fallback.MODE_AUTO_STR, 0, _fallback.ERR_STRING, "ERR_STRING"
    )


def test_vector_string_trailing_garbage_rejected():
    payload = struct.pack(">hI", 3, 1) + _tstring(b"one") + b"junk"
    entry = struct.pack(">I", 0x40000000 | len(payload)) + payload
    borders = np.array([0, len(entry)], dtype=np.int64)
    _expect_error(
        entry, borders, _fallback.MODE_STR_VEC, 0, _fallback.ERR_COUNT, "ERR_COUNT"
    )

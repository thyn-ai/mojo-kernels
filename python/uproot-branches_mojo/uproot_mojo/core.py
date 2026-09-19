"""Public API: read vector<T>/string TBranches as offsets+content NumPy arrays.

``read_branch`` locates a branch's baskets in a ROOT file (TKey walk),
decompresses them (zlib via cramjam), and walks the raw basket bytes with
either the native Mojo kernel or the vendored pure-Python fallback. The
result is exactly the buffers uproot delivers for the branch: cumulative
entry offsets and the contiguous (native-endian) content.
"""

from __future__ import annotations

import os
from typing import NamedTuple

import numpy as np

from uproot_mojo import _fallback
from uproot_mojo._native import NativeUnavailable
from uproot_mojo._rootfile import (
    BasketDataError,
    RootFileError,
    UnsupportedBranchError,
    read_branch_baskets,
)

__all__ = [
    "BasketDataError",
    "JaggedArray",
    "JaggedStringArray",
    "RootFileError",
    "UnsupportedBranchError",
    "read_branch",
]

# Supported numeric branch dtypes (big-endian, matching uproot's AsDtype).
_DTYPES = {
    "int32": (np.dtype(">i4"), 4),
    "int64": (np.dtype(">i8"), 8),
    "float32": (np.dtype(">f4"), 4),
    "float64": (np.dtype(">f8"), 8),
}
_NUMPY_TO_NAME = {v[0]: k for k, v in _DTYPES.items()}


class JaggedArray(NamedTuple):
    """offsets+content for vector<T> numeric branches and std::string branches.

    For ``kind == "numeric"``: ``offsets[i]..offsets[i+1]`` selects the items
    of entry ``i`` in ``content`` (native-endian ``dtype``), exactly the
    buffers uproot delivers for the branch. For ``kind == "string"``: content is
    uint8 string bytes (dtype "|u1") and offsets delimit one string per
    entry, exactly uproot's ``AsStrings`` buffers.
    """

    offsets: np.ndarray  # int64[n_entries + 1]
    content: np.ndarray  # native-endian typed (numeric) or uint8 (string)
    kind: str  # "numeric" or "string"
    backend: str  # "native" or "fallback"

    def to_list(self) -> list:
        """Plain-Python values: list of lists (numeric) or list of str."""
        if self.kind == "numeric":
            return [
                self.content[self.offsets[i] : self.offsets[i + 1]].tolist()
                for i in range(len(self.offsets) - 1)
            ]
        return [
            bytes(self.content[self.offsets[i] : self.offsets[i + 1]]).decode("utf-8")
            for i in range(len(self.offsets) - 1)
        ]

    def to_awkward(self):
        """The equivalent awkward Array (requires the optional ``awkward``)."""
        import awkward as ak  # optional dependency, imported lazily

        if self.kind == "numeric":
            form = ak.forms.ListOffsetForm(
                "i64",
                ak.forms.NumpyForm(
                    _NUMPY_TO_NAME[self.content.dtype.newbyteorder(">")],
                    form_key="node1",
                ),
                form_key="node0",
            )
            return ak.from_buffers(form, len(self.offsets) - 1, {
                "node0-offsets": self.offsets,
                "node1-data": self.content,
            })
        form = ak.forms.ListOffsetForm(
            "i64",
            ak.forms.NumpyForm(
                "uint8", parameters={"__array__": "char"}, form_key="node1"
            ),
            parameters={"__array__": "string"},
            form_key="node0",
        )
        return ak.from_buffers(form, len(self.offsets) - 1, {
            "node0-offsets": self.offsets,
            "node1-data": self.content,
        })


class JaggedStringArray(NamedTuple):
    """offsets+content for vector<string> branches (two jagged levels).

    Entry ``i`` owns strings ``offsets[i]..offsets[i+1]``; string ``j`` owns
    bytes ``string_offsets[j]..string_offsets[j+1]`` of ``content``.
    """

    offsets: np.ndarray  # int64[n_entries + 1]
    string_offsets: np.ndarray  # int64[n_strings + 1]
    content: np.ndarray  # uint8 string bytes
    backend: str  # "native" or "fallback"

    def to_list(self) -> list[list[str]]:
        strings = [
            bytes(self.content[self.string_offsets[j] : self.string_offsets[j + 1]]).decode(
                "utf-8"
            )
            for j in range(len(self.string_offsets) - 1)
        ]
        return [
            strings[self.offsets[i] : self.offsets[i + 1]]
            for i in range(len(self.offsets) - 1)
        ]

    def to_awkward(self):
        """The equivalent awkward Array (requires the optional ``awkward``)."""
        import awkward as ak  # optional dependency, imported lazily

        form = ak.forms.ListOffsetForm(
            "i64",
            ak.forms.ListOffsetForm(
                "i64",
                ak.forms.NumpyForm(
                    "uint8", parameters={"__array__": "char"}, form_key="node2"
                ),
                parameters={"__array__": "string"},
                form_key="node1",
            ),
            form_key="node0",
        )
        return ak.from_buffers(form, len(self.offsets) - 1, {
            "node0-offsets": self.offsets,
            "node1-offsets": self.string_offsets,
            "node2-data": self.content,
        })


def _regularize_dtype(dtype) -> tuple[np.dtype, int]:
    """Normalize the user dtype to (big-endian dtype, itemsize)."""
    if dtype is None:
        return None, 0
    key = str(dtype).lower()
    if key in _DTYPES:
        return _DTYPES[key]
    try:
        np_dtype = np.dtype(dtype)
    except TypeError:
        np_dtype = None
    if np_dtype is not None:
        big = np_dtype.newbyteorder(">")
        for candidate, (big_dtype, itemsize) in _DTYPES.items():
            if big_dtype == big:
                return _DTYPES[candidate]
    raise UnsupportedBranchError(
        f"unsupported dtype {dtype!r}: supported numeric branch dtypes are "
        f"{sorted(_DTYPES)} (omit dtype for string branches)"
    )


def _backend_walk():
    """The active basket walker: native kernel if loadable, else fallback."""
    from uproot_mojo import _native

    try:
        _native._load()
    except NativeUnavailable:
        return _fallback.walk_basket, "fallback"
    return _native.walk_basket, "native"


def read_branch(file, branch: str, *, dtype=None, tree=None):
    """Read one vector<T> or string TBranch from a ROOT file.

    Args:
        file: Path to a ROOT file (str or os.PathLike).
        branch: Branch name. Baskets are located by their TKey name.
        dtype: Required for numeric jagged branches: one of "int32", "int64",
            "float32", "float64" (NumPy dtypes accepted too). Omit for
            std::string and vector<string> branches, whose layout is
            validated structurally.
        tree: Tree name, required only when several trees contain a branch
            with this name.

    Returns:
        JaggedArray for vector<int32/int64/float32/float64> and std::string
        branches; JaggedStringArray for vector<string> branches. Offsets and
        content bytes are identical to uproot's interpretation of the same
        branch (verified by the differential test suite).

    Raises:
        RootFileError: not a ROOT file / malformed structure.
        UnsupportedBranchError: branch outside the supported scope.
        BasketDataError: corrupt basket bytes (carries a stable error code).
    """
    if not isinstance(branch, str) or not branch:
        raise ValueError("branch must be a non-empty string")
    if not isinstance(file, (str, os.PathLike)):
        raise TypeError("file must be a path (str or os.PathLike)")

    big_dtype, itemsize = _regularize_dtype(dtype)
    mode = _fallback.MODE_AUTO_NUM if dtype is not None else _fallback.MODE_AUTO_STR

    baskets = read_branch_baskets(os.fspath(file), branch, tree)
    walk, backend = _backend_walk()

    detected_mode = None
    parts_offsets = []
    parts_content = []
    parts_string_offsets = []
    for index, basket in enumerate(baskets):
        where = f"basket {index} of branch {branch!r}"
        try:
            detected, offsets, content, string_offsets = walk(
                basket.data, basket.borders, mode, itemsize, where
            )
        except BasketDataError as exc:
            if dtype is None:
                raise UnsupportedBranchError(
                    f"branch {branch!r} is not a readable std::string or "
                    f"vector<string> branch ({exc}); for numeric jagged "
                    "branches pass dtype= (one of 'int32', 'int64', "
                    "'float32', 'float64')"
                ) from exc
            raise
        if detected_mode is None:
            detected_mode = detected
        elif detected != detected_mode:
            raise BasketDataError(
                _fallback.ERR_MODE,
                f"{where}: entry layout changed between baskets "
                f"({detected_mode} -> {detected})",
            )
        parts_offsets.append(offsets)
        parts_content.append(content)
        if string_offsets is not None:
            parts_string_offsets.append(string_offsets)

    if detected_mode is None:  # pragma: no cover - read_branch_baskets never returns []
        raise UnsupportedBranchError(f"branch {branch!r} has no baskets")

    offsets = _concat_offsets(parts_offsets)
    if detected_mode in (_fallback.MODE_NUM_RAW, _fallback.MODE_NUM_HDR):
        # Baskets store items big-endian (ROOT I/O spec); uproot delivers
        # native-endian arrays, so swap — bytes then match uproot exactly.
        big = np.frombuffer(b"".join(parts_content), dtype=big_dtype)
        content = big.byteswap().view(big_dtype.newbyteorder("="))
        return JaggedArray(offsets, content, "numeric", backend)
    if detected_mode == _fallback.MODE_STR_ONE:
        content = np.frombuffer(b"".join(parts_content), dtype=np.uint8)
        return JaggedArray(offsets, content, "string", backend)
    # vector<string>
    string_offsets = _concat_offsets(parts_string_offsets)
    content = np.frombuffer(b"".join(parts_content), dtype=np.uint8)
    return JaggedStringArray(offsets, string_offsets, content, backend)


def _concat_offsets(parts: list[np.ndarray]) -> np.ndarray:
    """Concatenate per-basket offsets, rebasing each basket by the running total."""
    if len(parts) == 1:
        return parts[0]
    total = sum(len(part) - 1 for part in parts)
    out = np.empty(total + 1, dtype=np.int64)
    acc = 0
    pos = 0
    for part in parts:
        n = len(part) - 1
        out[pos : pos + n] = part[:-1] + acc
        acc += int(part[-1])
        pos += n
    out[total] = acc
    return out

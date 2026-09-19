"""Minimal ROOT TFile/TKey/TBasket reader for split-level-0 branch baskets.

Clean-room implementation of the documented ROOT I/O format (ROOT's
io/doc/TFile spec): file header, TBasket TKey location by validated
structural scan (basket keys are not registered in the file's TKeysList),
ROOT's zlib block compression, and the per-basket entry-offset table. It reads exactly what ``uproot_mojo`` needs — the raw, uncompressed
basket data region plus per-entry byte borders — and nothing more. No
TStreamerInfo/TTree metadata is parsed; the branch's data type is supplied
by the caller (or validated structurally for strings), never guessed.

Scope (documented in the package README):

  * free-standing TBasket keys (embedded baskets are rejected),
  * baskets carrying an entry-offset table (every jagged/string basket
    written by uproot.recreate, and TTree baskets written by CERN ROOT),
  * zlib compression ("ZL" blocks) and stored-uncompressed baskets,
  * single-tree files, or multi-tree files disambiguated with ``tree=``.
"""

from __future__ import annotations

import struct
from typing import NamedTuple

import cramjam
import numpy as np

# TKey fixed part: fNbytes, fVersion, fObjlen, fDatime, fKeylen, fCycle,
# then 32-bit (fVersion < 1000) or 64-bit seeks.
_KEY_HEADER = struct.Struct(">ihiIhh")
# TBasket stream tail inside the key: fVersion, fBufferSize, fNevBufSize,
# fNevBuf, fLast, then one flag byte. It sits at the end of the key header,
# so the basket data starts right after it.
_TBASKET_TAIL = struct.Struct(">Hiiii")
_TBASKET_TAIL_BYTES = _TBASKET_TAIL.size + 1

_FILE_HEADER_SMALL = struct.Struct(">4si")  # magic, fVersion (fBEGIN at 8)


class RootFileError(Exception):
    """The file is not a readable ROOT file within the supported scope."""


class UnsupportedBranchError(Exception):
    """The branch exists but is outside the supported deserialization scope."""


class BasketDataError(Exception):
    """A basket's bytes failed structural validation (with an error code)."""

    def __init__(self, code: int, message: str):
        self.code = code
        super().__init__(message)


class Basket(NamedTuple):
    """One uncompressed basket: raw data region + per-entry byte borders."""

    data: bytes
    borders: np.ndarray  # int64[n_entries + 1], starts of entries + data end
    n_entries: int


def _read_tstring(buf: bytes, pos: int, limit: int) -> tuple[bytes, int]:
    """ROOT TString at pos: 1-byte length, or 0xFF + 4-byte big-endian length."""
    if pos >= limit:
        raise RootFileError("truncated TString in TKey header")
    n = buf[pos]
    pos += 1
    if n == 255:
        if pos + 4 > limit:
            raise RootFileError("truncated long TString in TKey header")
        n = struct.unpack_from(">I", buf, pos)[0]
        pos += 4
    if pos + n > limit:
        raise RootFileError("truncated TString payload in TKey header")
    return buf[pos : pos + n], pos + n


def _decompress_payload(payload: bytes, objlen: int, where: str) -> bytes:
    """Inflate a ROOT-compressed payload: a chain of 9-byte-headed blocks.

    Only zlib ("ZL") blocks are supported, per the package's documented
    scope. An uncompressed payload (objlen == len(payload)) is returned
    verbatim by the caller before reaching this function.
    """
    out = bytearray()
    pos = 0
    while pos < len(payload):
        if pos + 9 > len(payload):
            raise RootFileError(f"truncated compression block header in {where}")
        alg = payload[pos : pos + 2]
        if alg != b"ZL":
            raise UnsupportedBranchError(
                f"unsupported compression {alg!r} in {where}: only zlib ('ZL') "
                "is supported (LZMA 'XZ', LZ4 'L4' and ZSTD 'ZS' are out of scope)"
            )
        csize = payload[pos + 3] | (payload[pos + 4] << 8) | (payload[pos + 5] << 16)
        usize = payload[pos + 6] | (payload[pos + 7] << 8) | (payload[pos + 8] << 16)
        block = payload[pos + 9 : pos + 9 + csize]
        if len(block) != csize:
            raise RootFileError(f"truncated zlib block in {where}")
        inflated = bytes(cramjam.zlib.decompress(block))
        if len(inflated) != usize:
            raise RootFileError(
                f"zlib block in {where} inflated to {len(inflated)} bytes, "
                f"expected {usize}"
            )
        out += inflated
        pos += 9 + csize
    if len(out) != objlen:
        raise RootFileError(
            f"decompressed {where} is {len(out)} bytes, TKey fObjlen says {objlen}"
        )
    return bytes(out)


_TBASKET_SIGNATURE = b"\x07TBasket"


def _candidate_key(buf: bytes, key_start: int, seek_bytes: int, fend: int):
    """Strictly validate a TBasket TKey at key_start, or return None.

    Every structural field must be self-consistent (a coincidental
    "x07TBasket" byte run inside some payload fails these checks): sane
    fNbytes/fKeylen, fVersion matching the seek width, the three TStrings
    ending exactly _TBASKET_TAIL_BYTES before fKeylen, and a TBasket tail
    consistent with fObjlen.
    """
    if key_start < 0 or key_start + _KEY_HEADER.size + seek_bytes > fend:
        return None
    nbytes, kversion, objlen, _datime, keylen, _cycle = _KEY_HEADER.unpack_from(
        buf, key_start
    )
    if nbytes <= 0 or key_start + nbytes > fend:
        return None
    if (kversion > 1000) != (seek_bytes == 16):
        return None
    p = key_start + _KEY_HEADER.size + seek_bytes
    if keylen < p - key_start + 1 + 7 + 1 + 1 + _TBASKET_TAIL_BYTES:
        return None
    if key_start + keylen > fend:
        return None
    try:
        classname, p = _read_tstring(buf, p, key_start + keylen)
        name, p = _read_tstring(buf, p, key_start + keylen)
        title, p = _read_tstring(buf, p, key_start + keylen)
    except RootFileError:
        return None
    if classname != b"TBasket":
        return None
    if p != key_start + keylen - _TBASKET_TAIL_BYTES:
        return None
    _bver, _fbufsize, _fnevbufsize, fnevbuf, flast = _TBASKET_TAIL.unpack_from(
        buf, key_start + keylen - _TBASKET_TAIL_BYTES
    )
    if fnevbuf < 0:
        return None
    data_len = flast - keylen
    if data_len < 0 or data_len > objlen:
        return None
    return {
        "pos": key_start,
        "nbytes": nbytes,
        "objlen": objlen,
        "keylen": keylen,
        "classname": classname,
        "name": name,
        "title": title,
        "data_len": data_len,
        "fnevbuf": fnevbuf,
    }


def _find_tbasket_keys(buf: bytes, fbegin: int, fend: int) -> list[dict]:
    """Locate every free-standing TBasket key in the file by structural scan.

    Basket keys are not registered in the file's TKeysList (only top-level
    objects are), and the linear key chain is broken by free-space gaps, so
    the only metadata-free locator is a full structural scan: every TBasket
    key contains the TString "\\x07TBasket"; each occurrence is validated
    strictly before being accepted. File position order is the basket
    (entry) order: writers append baskets as entries are filled.
    """
    out = []
    pos = fbegin
    while True:
        sig = buf.find(_TBASKET_SIGNATURE, pos, fend)
        if sig < 0:
            break
        pos = sig + 1
        for seek_bytes, back in ((16, 34), (8, 26)):
            info = _candidate_key(buf, sig - back, seek_bytes, fend)
            if info is not None:
                out.append(info)
                break
    out.sort(key=lambda info: info["pos"])
    return out


def _parse_basket(info: dict, buf: bytes, where: str) -> Basket:
    """Decompress one located basket key and extract data + entry borders."""
    pos, nbytes, objlen, keylen = (
        info["pos"],
        info["nbytes"],
        info["objlen"],
        info["keylen"],
    )
    data_len = info["data_len"]
    payload = buf[pos + keylen : pos + nbytes]
    if len(payload) == objlen:
        raw = bytes(payload)
    else:
        raw = _decompress_payload(payload, objlen, where)

    if objlen == data_len:
        raise UnsupportedBranchError(
            f"{where}: no entry-offset table; only jagged/string baskets "
            "with an fEntryOffset table are supported (plain rectilinear "
            "branches are out of scope)"
        )
    table = raw[data_len:]
    if len(table) < 8 or len(table) % 4 != 0:
        raise RootFileError(f"{where}: malformed entry-offset table")
    entries = np.frombuffer(table, dtype=">i4")
    n_entries = len(entries) - 2
    if n_entries != info["fnevbuf"]:
        raise RootFileError(
            f"{where}: entry-offset table implies {n_entries} entries, "
            f"TBasket header says {info['fnevbuf']}"
        )
    # Stored offsets are key-relative (data offset + fKeylen); the last
    # border is the data-region length itself.
    borders = np.empty(n_entries + 1, dtype=np.int64)
    borders[:n_entries] = entries[1 : n_entries + 1].astype(np.int64) - keylen
    borders[n_entries] = data_len
    if n_entries and borders[0] != 0:
        raise RootFileError(f"{where}: first entry offset is not zero")
    return Basket(data=raw[:data_len], borders=borders, n_entries=n_entries)


def read_branch_baskets(path, branch: str, tree: str | None = None) -> list[Basket]:
    """Locate every basket of ``branch`` in the ROOT file at ``path``.

    Baskets are found by structural scan (see ``_find_tbasket_keys``) and
    returned in file position order, which is the basket (entry) order.
    ``tree`` disambiguates same-named branches in multi-tree files (the
    basket key title carries the tree name).
    """
    with open(path, "rb") as handle:
        buf = handle.read()
    if len(buf) < 16 or buf[:4] != b"root":
        raise RootFileError(f"{path!r} is not a ROOT file (missing 'root' magic)")
    (fversion,) = struct.unpack_from(">i", buf, 4)
    (fbegin,) = struct.unpack_from(">i", buf, 8)
    if fversion >= 1_000_000:
        (fend,) = struct.unpack_from(">q", buf, 12)
    else:
        (fend,) = struct.unpack_from(">i", buf, 12)
    fend = min(fend, len(buf))
    if not (0 < fbegin < fend):
        raise RootFileError(
            f"{path!r}: implausible header (fBEGIN={fbegin}, fEND={fend}, "
            f"size={len(buf)})"
        )

    branch_b = branch.encode("utf-8", "surrogateescape")
    tree_b = tree.encode("utf-8", "surrogateescape") if tree is not None else None
    matches = []
    titles = set()
    for info in _find_tbasket_keys(buf, fbegin, fend):
        if info["name"] == branch_b:
            titles.add(info["title"])
            if tree_b is None or info["title"] == tree_b:
                matches.append(info)

    if not matches:
        if titles and tree_b is None:
            raise UnsupportedBranchError(
                f"branch {branch!r} exists under multiple trees {sorted(titles)}; "
                "pass tree= to disambiguate"
            )
        raise UnsupportedBranchError(
            f"no TBasket keys named {branch!r} in {path!r} "
            f"(tree filter: {tree!r}); only free-standing TBasket keys of "
            "split-level-0 branches are supported"
        )
    if tree_b is None and len(titles) > 1:
        raise UnsupportedBranchError(
            f"branch {branch!r} exists under multiple trees {sorted(titles)}; "
            "pass tree= to disambiguate"
        )

    baskets = []
    for index, info in enumerate(matches):
        where = f"basket {index} of branch {branch!r} in {path!r}"
        baskets.append(_parse_basket(info, buf, where))
    return baskets

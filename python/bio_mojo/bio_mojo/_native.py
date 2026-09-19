"""ctypes loader for the bioparse native kernel, with an ABI-version handshake.

Resolution order:

    1. ``$BIO_MOJO_NATIVE_LIB`` (explicit path override, for development)
    2. the shared library bundled in this wheel under ``bio_mojo/_native/``
    3. the repository development build output ``kernels/bioparse/build/``

If the library cannot be found, fails to load, or reports an ABI version this
package does not understand, :class:`NativeUnavailable` is raised and the
caller falls back to the vendored pure-Python parser. Set
``BIO_MOJO_DISABLE_NATIVE=1`` to force that fallback (used by the
differential test suite).

The native parse functions return the same :class:`bio_mojo._raw` structures
as :mod:`bio_mojo._reference`; they only raise ``NativeUnavailable`` (load
problems) or the shared ``ParseError`` (input problems, same codes and
messages as the fallback).

Stable C ABI (v1)::

    int32_t  bioparse_abi_version(void)
    void*    bioparse_fasta_parse(const uint8_t* data, int64_t len,
                                  int32_t* out_err, int64_t* out_err_line)
    int64_t  bioparse_fasta_count(void* h)
    int32_t  bioparse_fasta_arrays(void* h, const uint8_t** titles,
                                   const int64_t** title_offs,
                                   const uint8_t** seqs,
                                   const int64_t** seq_offs)
    void     bioparse_fasta_free(void* h)
    void*    bioparse_gb_parse(const uint8_t* data, int64_t len,
                               int32_t* out_err, int64_t* out_err_line)
    int64_t  bioparse_gb_count(void* h)
    int32_t  bioparse_gb_record_arrays(void* h, ... 11 out-params ...)
    int32_t  bioparse_gb_feature_arrays(void* h, ... 5 out-params ...)
    int32_t  bioparse_gb_qualifier_arrays(void* h, ... 5 out-params ...)
    void     bioparse_gb_free(void* h)
"""

from __future__ import annotations

import ctypes
import os
import sys
import threading

from bio_mojo._raw import (
    RawFasta,
    RawFeature,
    RawGbRecord,
    RawQualifier,
    raise_for_error,
)
from bio_mojo.errors import NativeUnavailable

# Must equal ABI_VERSION in kernels/bioparse/src/bioparse.mojo. A mismatch
# means the installed wheel and the resolved shared library disagree.
ABI_VERSION = 1

_ENV_LIB = "BIO_MOJO_NATIVE_LIB"
_ENV_DISABLE = "BIO_MOJO_DISABLE_NATIVE"

_u8pp = ctypes.POINTER(ctypes.c_void_p)


def _lib_basename() -> str:
    if sys.platform == "darwin":
        return "libbioparse.dylib"
    if sys.platform.startswith("linux"):
        return "libbioparse.so"
    if sys.platform.startswith("win"):
        return "bioparse.dll"  # no Mojo toolchain builds this today
    return "libbioparse.so"


def _candidate_paths() -> list[tuple[str, str]]:
    """(source_label, path) candidates, in resolver order."""
    out: list[tuple[str, str]] = []
    override = os.environ.get(_ENV_LIB)
    if override:
        out.append((f"env {_ENV_LIB}", override))
    here = os.path.dirname(__file__)
    out.append(("bundled in wheel", os.path.join(here, "_native", _lib_basename())))
    out.append(
        (
            "repo-dev build output",
            os.path.abspath(
                os.path.join(
                    here, "..", "..", "..", "kernels", "bioparse", "build", _lib_basename()
                )
            ),
        )
    )
    return out


def _bind_abi(lib: ctypes.CDLL) -> None:
    i32p = ctypes.POINTER(ctypes.c_int32)
    i64p = ctypes.POINTER(ctypes.c_int64)
    parse_args = [ctypes.c_char_p, ctypes.c_int64, i32p, i64p]
    lib.bioparse_abi_version.argtypes = []
    lib.bioparse_abi_version.restype = ctypes.c_int32

    lib.bioparse_fasta_parse.argtypes = parse_args
    lib.bioparse_fasta_parse.restype = ctypes.c_void_p
    lib.bioparse_fasta_count.argtypes = [ctypes.c_void_p]
    lib.bioparse_fasta_count.restype = ctypes.c_int64
    lib.bioparse_fasta_arrays.argtypes = [ctypes.c_void_p, _u8pp, _u8pp, _u8pp, _u8pp]
    lib.bioparse_fasta_arrays.restype = ctypes.c_int32
    lib.bioparse_fasta_free.argtypes = [ctypes.c_void_p]
    lib.bioparse_fasta_free.restype = None

    lib.bioparse_gb_parse.argtypes = parse_args
    lib.bioparse_gb_parse.restype = ctypes.c_void_p
    lib.bioparse_gb_count.argtypes = [ctypes.c_void_p]
    lib.bioparse_gb_count.restype = ctypes.c_int64
    lib.bioparse_gb_record_arrays.argtypes = [ctypes.c_void_p] + [_u8pp] * 11
    lib.bioparse_gb_record_arrays.restype = ctypes.c_int32
    lib.bioparse_gb_feature_arrays.argtypes = [ctypes.c_void_p] + [_u8pp] * 5
    lib.bioparse_gb_feature_arrays.restype = ctypes.c_int32
    lib.bioparse_gb_qualifier_arrays.argtypes = [ctypes.c_void_p] + [_u8pp] * 5
    lib.bioparse_gb_qualifier_arrays.restype = ctypes.c_int32
    lib.bioparse_gb_free.argtypes = [ctypes.c_void_p]
    lib.bioparse_gb_free.restype = None


_LOCK = threading.Lock()
_LIB: ctypes.CDLL | None = None
_LIB_SOURCE: str | None = None
_LOAD_ERROR: str | None = None


def _load() -> ctypes.CDLL:
    """Resolve, dlopen, and ABI-handshake the native kernel. Never caches failure."""
    global _LIB, _LIB_SOURCE, _LOAD_ERROR
    if os.environ.get(_ENV_DISABLE) == "1":
        raise NativeUnavailable(f"native kernel disabled by {_ENV_DISABLE}=1")
    if _LIB is not None:
        return _LIB
    with _LOCK:
        if _LIB is not None:
            return _LIB
        errors: list[str] = []
        for label, path in _candidate_paths():
            try:
                if not path or not os.path.exists(path):
                    continue
                try:
                    lib = ctypes.CDLL(path)
                except OSError as exc:
                    errors.append(f"{label} ({path}): {exc}")
                    continue
                try:
                    _bind_abi(lib)
                    abi = int(lib.bioparse_abi_version())
                except Exception as exc:  # missing/renamed symbol = wrong lib
                    errors.append(f"{label} ({path}): ABI not recognized: {exc}")
                    continue
                if abi != ABI_VERSION:
                    errors.append(
                        f"{label} ({path}): native ABI v{abi} != wrapper ABI v{ABI_VERSION}"
                    )
                    continue
                _LIB, _LIB_SOURCE = lib, f"{label} ({path})"
                _LOAD_ERROR = None
                return lib
            except OSError as exc:
                errors.append(f"{label}: {exc}")
        _LOAD_ERROR = "; ".join(errors) or "no native kernel found on any resolver path"
        raise NativeUnavailable(_LOAD_ERROR)


def native_available() -> bool:
    """True if the native kernel can parse right now. Never raises."""
    try:
        _load()
        return True
    except NativeUnavailable:
        return False


def backend_info() -> dict:
    """Diagnostics for the active backend. Never raises."""
    info = {
        "native_available": False,
        "native_source": None,
        "abi_version_expected": ABI_VERSION,
        "abi_version_native": None,
        "disabled_by_env": os.environ.get(_ENV_DISABLE) == "1",
        "platform": sys.platform,
        "error": None,
    }
    try:
        lib = _load()
    except NativeUnavailable as exc:
        info["error"] = str(exc)
        return info
    info["native_available"] = True
    info["native_source"] = _LIB_SOURCE
    info["abi_version_native"] = int(lib.bioparse_abi_version())
    return info


def _offs(ptr: ctypes.c_void_p) -> ctypes.POINTER(ctypes.c_int64):
    return ctypes.cast(ptr, ctypes.POINTER(ctypes.c_int64))


def _bulk(base: int, offs: list[int]) -> bytes:
    """Copy a whole concat arena out of the kernel's memory in one call."""
    return ctypes.string_at(base, offs[-1]) if offs[-1] else b""


def fasta_parse_bytes(data: bytes) -> RawFasta:
    """Parse a whole FASTA buffer on the native kernel."""
    lib = _load()  # raises NativeUnavailable
    err = ctypes.c_int32(0)
    err_line = ctypes.c_int64(0)
    handle = lib.bioparse_fasta_parse(
        data, len(data), ctypes.byref(err), ctypes.byref(err_line)
    )
    if not handle:
        raise_for_error(err.value, err_line.value)
    try:
        n = lib.bioparse_fasta_count(handle)
        titles, title_offs, seqs, seq_offs = (ctypes.c_void_p() for _ in range(4))
        rc = lib.bioparse_fasta_arrays(
            handle,
            ctypes.byref(titles),
            ctypes.byref(title_offs),
            ctypes.byref(seqs),
            ctypes.byref(seq_offs),
        )
        if rc != 0:
            raise NativeUnavailable(f"native kernel array access failed (rc={rc})")
        to = _offs(title_offs.value)[: n + 1]
        so = _offs(seq_offs.value)[: n + 1]
        return RawFasta(_bulk(titles.value, to), to, _bulk(seqs.value, so), so)
    finally:
        lib.bioparse_fasta_free(handle)


def gb_parse_bytes(data: bytes) -> list[RawGbRecord]:
    """Parse a whole GenBank buffer on the native kernel."""
    lib = _load()  # raises NativeUnavailable
    err = ctypes.c_int32(0)
    err_line = ctypes.c_int64(0)
    handle = lib.bioparse_gb_parse(
        data, len(data), ctypes.byref(err), ctypes.byref(err_line)
    )
    if not handle:
        raise_for_error(err.value, err_line.value)
    try:
        n_rec = lib.bioparse_gb_count(handle)
        rv = [ctypes.c_void_p() for _ in range(11)]
        rc = lib.bioparse_gb_record_arrays(handle, *[ctypes.byref(p) for p in rv])
        if rc != 0:
            raise NativeUnavailable(f"native kernel array access failed (rc={rc})")
        (names, name_offs, defs, def_offs, accs, acc_offs,
         vers, ver_offs, seqs, seq_offs, rec_feat_offs) = rv
        no = _offs(name_offs.value)[: n_rec + 1]
        do = _offs(def_offs.value)[: n_rec + 1]
        ao = _offs(acc_offs.value)[: n_rec + 1]
        vo = _offs(ver_offs.value)[: n_rec + 1]
        so = _offs(seq_offs.value)[: n_rec + 1]
        rfo = _offs(rec_feat_offs.value)[: n_rec + 1]
        n_feat = rfo[n_rec]
        names_b, defs_b = _bulk(names.value, no), _bulk(defs.value, do)
        accs_b, vers_b = _bulk(accs.value, ao), _bulk(vers.value, vo)
        seqs_b = _bulk(seqs.value, so)

        fv = [ctypes.c_void_p() for _ in range(5)]
        rc = lib.bioparse_gb_feature_arrays(handle, *[ctypes.byref(p) for p in fv])
        if rc != 0:
            raise NativeUnavailable(f"native kernel array access failed (rc={rc})")
        keys, key_offs, locs, loc_offs, feat_qual_offs = fv
        ko = _offs(key_offs.value)[: n_feat + 1]
        lo = _offs(loc_offs.value)[: n_feat + 1]
        fqo = _offs(feat_qual_offs.value)[: n_feat + 1]
        n_qual = fqo[n_feat]
        keys_b, locs_b = _bulk(keys.value, ko), _bulk(locs.value, lo)

        qv = [ctypes.c_void_p() for _ in range(5)]
        rc = lib.bioparse_gb_qualifier_arrays(handle, *[ctypes.byref(p) for p in qv])
        if rc != 0:
            raise NativeUnavailable(f"native kernel array access failed (rc={rc})")
        qnames, qname_offs, qvals, qval_offs, qquoted = qv
        qno = _offs(qname_offs.value)[: n_qual + 1]
        qvo = _offs(qval_offs.value)[: n_qual + 1]
        qnames_b, qvals_b = _bulk(qnames.value, qno), _bulk(qvals.value, qvo)
        qq = ctypes.string_at(qquoted.value, n_qual) if n_qual else b""

        quals: list[RawQualifier] = [
            RawQualifier(
                qnames_b[qno[k] : qno[k + 1]],
                qvals_b[qvo[k] : qvo[k + 1]],
                bool(qq[k] & 1),
                bool(qq[k] & 2),
            )
            for k in range(n_qual)
        ]
        feats: list[RawFeature] = [
            RawFeature(
                keys_b[ko[j] : ko[j + 1]],
                locs_b[lo[j] : lo[j + 1]],
                quals[fqo[j] : fqo[j + 1]],
            )
            for j in range(n_feat)
        ]
        return [
            RawGbRecord(
                names_b[no[i] : no[i + 1]],
                defs_b[do[i] : do[i + 1]],
                accs_b[ao[i] : ao[i + 1]],
                vers_b[vo[i] : vo[i + 1]],
                seqs_b[so[i] : so[i + 1]],
                feats[rfo[i] : rfo[i + 1]],
            )
            for i in range(n_rec)
        ]
    finally:
        lib.bioparse_gb_free(handle)

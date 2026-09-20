"""ctypes loader for the capamojo native kernel, with an ABI-version handshake.

Resolution order:

    1. ``$CAPA_MOJO_NATIVE_LIB`` (explicit path override, for development)
    2. the shared library bundled in this wheel under ``capa_mojo/_native/``
    3. the repository development build output ``kernels/capa/build/``

If the library cannot be found, fails to load, or reports an ABI version this
package does not understand, :class:`NativeUnavailable` is raised and the
caller falls back to the pure-Python reference evaluator. Set
``CAPA_MOJO_DISABLE_NATIVE=1`` to force that fallback (used by the
differential test suite).

Stable C ABI (v1)::

    int32_t  capamojo_abi_version(void)
    void*    capamojo_ruleset_create(int64_t n_rules,
                                     const int64_t* rule_node_offsets,
                                     const int32_t* node_op,
                                     const int64_t* node_child_offsets,
                                     const int32_t* node_children,
                                     const int32_t* node_arg,
                                     const uint64_t* node_range_min,
                                     const uint64_t* node_range_max,
                                     int64_t n_count_leaves,
                                     int64_t n_match_groups,
                                     const int64_t* match_group_offsets,
                                     const int32_t* match_group_rules)
    int32_t  capamojo_eval(void* handle, int64_t n_scopes,
                           const int32_t* leaf_counts,
                           const uint64_t* leaf_present_words,
                           uint64_t* out_rule_words)
    void     capamojo_ruleset_destroy(void* handle)
"""

from __future__ import annotations

import ctypes
import os
import sys
import threading

import numpy as np

# Must equal ABI_VERSION in kernels/capa/src/capamojo.mojo. A mismatch means
# the installed wheel and the resolved shared library disagree; fall back.
ABI_VERSION = 1

_ENV_LIB = "CAPA_MOJO_NATIVE_LIB"
_ENV_DISABLE = "CAPA_MOJO_DISABLE_NATIVE"


class NativeUnavailable(RuntimeError):  # noqa: N818
    """The native capamojo kernel could not be found, loaded, or verified."""


def _lib_basename() -> str:
    if sys.platform == "darwin":
        return "libcapamojo.dylib"
    if sys.platform.startswith("linux"):
        return "libcapamojo.so"
    if sys.platform.startswith("win"):
        return "capamojo.dll"  # no Mojo toolchain builds this today
    return "libcapamojo.so"


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
                os.path.join(here, "..", "..", "..", "kernels", "capa", "build", _lib_basename())
            ),
        )
    )
    return out


def _bind_abi(lib: ctypes.CDLL) -> None:
    i32p = ctypes.POINTER(ctypes.c_int32)
    i64p = ctypes.POINTER(ctypes.c_int64)
    u64p = ctypes.POINTER(ctypes.c_uint64)
    lib.capamojo_abi_version.argtypes = []
    lib.capamojo_abi_version.restype = ctypes.c_int32
    lib.capamojo_ruleset_create.argtypes = [
        ctypes.c_int64,  # n_rules
        i64p,  # rule_node_offsets[n_rules+1]
        i32p,  # node_op[total_nodes]
        i64p,  # node_child_offsets[total_nodes+1]
        i32p,  # node_children[total_children]
        i32p,  # node_arg[total_nodes]
        u64p,  # node_range_min[total_nodes]
        u64p,  # node_range_max[total_nodes]
        ctypes.c_int64,  # n_count_leaves
        ctypes.c_int64,  # n_match_groups
        i64p,  # match_group_offsets[n_match_groups+1]
        i32p,  # match_group_rules[total_refs]
    ]
    lib.capamojo_ruleset_create.restype = ctypes.c_void_p
    lib.capamojo_eval.argtypes = [ctypes.c_void_p, ctypes.c_int64, i32p, u64p, u64p]
    lib.capamojo_eval.restype = ctypes.c_int32
    lib.capamojo_ruleset_destroy.argtypes = [ctypes.c_void_p]
    lib.capamojo_ruleset_destroy.restype = None


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
                    abi = int(lib.capamojo_abi_version())
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
    """True if the native kernel can evaluate right now. Never raises."""
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
    info["abi_version_native"] = int(lib.capamojo_abi_version())
    return info


class NativeRuleSet:
    """Owned handle to a native compiled rule set. Not thread-safe to close twice."""

    def __init__(self, compiled) -> None:
        lib = _load()  # raises NativeUnavailable
        i32p = ctypes.POINTER(ctypes.c_int32)
        i64p = ctypes.POINTER(ctypes.c_int64)
        u64p = ctypes.POINTER(ctypes.c_uint64)
        arrays = [
            np.ascontiguousarray(a)
            for a in (
                compiled.rule_node_offsets,
                compiled.node_op,
                compiled.node_child_offsets,
                compiled.node_children,
                compiled.node_arg,
                compiled.node_range_min,
                compiled.node_range_max,
                compiled.match_group_offsets,
                compiled.match_group_rules,
            )
        ]
        handle = lib.capamojo_ruleset_create(
            ctypes.c_int64(len(compiled)),
            arrays[0].ctypes.data_as(i64p),
            arrays[1].ctypes.data_as(i32p),
            arrays[2].ctypes.data_as(i64p),
            arrays[3].ctypes.data_as(i32p),
            arrays[4].ctypes.data_as(i32p),
            arrays[5].ctypes.data_as(u64p),
            arrays[6].ctypes.data_as(u64p),
            ctypes.c_int64(compiled.n_count_leaves),
            ctypes.c_int64(compiled.n_match_groups),
            arrays[7].ctypes.data_as(i64p),
            arrays[8].ctypes.data_as(i32p),
        )
        if not handle:
            raise NativeUnavailable(
                "native kernel rejected the rule set (invalid arrays); "
                "falling back to the pure-Python reference"
            )
        # The kernel copies every buffer; the temporary arrays may be freed.
        self._lib = lib
        self._handle = handle
        self._n_rules = len(compiled)

    def eval(self, counts: np.ndarray, present_words: np.ndarray) -> np.ndarray:
        """Evaluate all rules over one block of <= 64 scopes.

        `counts`: int32[n_slots, n_scopes] leaf-major location counts.
        `present_words`: uint64[n_slots] feature-presence bits.
        Returns uint64[n_rules]: bit i of word r set <=> rule r matched scope i.
        """
        if self._handle is None:
            raise NativeUnavailable("native rule set is closed")
        counts = np.ascontiguousarray(counts, dtype=np.int32)
        present_words = np.ascontiguousarray(present_words, dtype=np.uint64)
        n_scopes = counts.shape[1]
        out = np.zeros(self._n_rules, dtype=np.uint64)
        rc = self._lib.capamojo_eval(
            self._handle,
            ctypes.c_int64(n_scopes),
            counts.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
            present_words.ctypes.data_as(ctypes.POINTER(ctypes.c_uint64)),
            out.ctypes.data_as(ctypes.POINTER(ctypes.c_uint64)),
        )
        if rc != 0:
            raise NativeUnavailable(f"native evaluation failed with status {rc}")
        return out

    def close(self) -> None:
        handle, self._handle = self._handle, None
        if handle and self._lib is not None:
            self._lib.capamojo_ruleset_destroy(handle)

    def __del__(self) -> None:  # best-effort; never raise during GC
        try:
            self.close()
        except Exception:  # noqa: BLE001, S110
            pass

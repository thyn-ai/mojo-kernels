# vol-mojo

**Batch registry-hive walking and pool-header constraint validation with
[volatility3](https://github.com/volatilityfoundation/volatility3)-compatible
semantics** — the two hot loops behind Windows memory-forensics scans —
powered by a Mojo kernel, with a vendored pure-Python fallback:

- `walk_hive(...)` — a batch hive-tree walker producing the same
  `(offset, type, value)` tuples a
  `volatility3.framework.layers.registry.RegistryHive.visit_nodes`
  visitor observes (`framework/layers/registry.py:244`), on flat `.dat`
  hive files (SOFTWARE/SYSTEM/NTUSER hives on disk);
- `scan_pool_headers(...)` — a batch pool-header validator: multi-tag
  scan plus the per-hit size/pool-type/PoolIndex constraint checks of
  `volatility3.plugins.windows.poolscanner.PoolHeaderScanner.__call__`
  (`plugins/windows/poolscanner.py:77-128`), returning the passing
  header offsets.

Prebuilt per-platform wheels mean **no Mojo toolchain is ever required**
on an end user's machine. volatility3 itself is *not* a dependency: it
is the test oracle, never imported at runtime. The package is pure
stdlib + ctypes — there are no third-party runtime dependencies at all.

## The parity contract (read this first)

Both functions are deterministic, and the contract on both backends
(native Mojo and forced fallback) is **exact equality with the
volatility3 2.28.2 oracle** — tuple-for-tuple / hit-for-hit, order
included, asserted by the differential suite over generated fixtures:

1. **`walk_hive` mirrors `visit_nodes` pre-order exactly**: the two
   `SubKeyLists` slots in order; `ri`/`lh`/`lf` index cells flattened
   depth-first in list order (`lh`/`lf` entries are (offset, hash) pairs
   and only the offset is followed); subkey-list entries that do not
   translate, do not decode, or point at non-`nk` cells are **skipped
   silently** — all exactly like the reference. Per visited key node the
   tuple is `(node.vol.offset, "_CM_KEY_NODE", node.get_name())`, i.e.
   the cell index + 4 and the raw name bytes decoded as latin-1. Where
   the reference *raises* on a malformed image (bad root node, truncated
   visited node or index list), `walk_hive` raises `HiveError`.
2. **`scan_pool_headers` mirrors `PoolHeaderScanner.__call__` exactly**:
   leftmost non-overlapping multi-pattern scan with longest match at each
   position (the reference's regex-trie semantics); per hit, the size
   bounds on `alignment * BlockSize`, the `PoolType` bitmask test
   (FREE/NONPAGED/PAGED, with Vista+ or pre-Vista parity), and the
   `PoolIndex` bounds. Header offsets are what the reference yields as
   `header.vol.offset`: the raw tag-offset-minus-4 normalized through the
   scan layer's address mask `(1 << ceil(log2(len(data) - 1))) - 1` — a
   negative raw offset (tag in the first 4 bytes) *wraps* and the wrapped
   header is validated, it is not skipped; a needed field read that then
   falls outside the buffer skips the hit, like the reference's
   `InvalidAddressException` handling. Header layouts: the two bundled
   x64 tables (byte-identical; `poolheader-x64`,
   `poolheader-x64-win7`) and `poolheader-x86`.

## Install

```
pip install vol-mojo
```

Per-platform wheels (macOS arm64, Linux x86_64) carry the native kernel,
self-contained (the Mojo runtime is vendored into the wheel; nothing to
compile, no absolute rpaths). On any other platform — including Windows —
the same wheel API runs on the vendored pure-Python fallback, silently
and correctly. There is no sdist: a source tarball cannot rebuild the
native library.

## Quickstart

```python
from vol_mojo import (
    PoolConstraint, PAGE_TYPE_NONPAGED, PAGE_TYPE_FREE,
    scan_pool_headers, walk_hive,
)

# Walk a flat hive file straight from disk:
tuples = walk_hive(open("SOFTWARE", "rb").read())
# -> [(offset, "_CM_KEY_NODE", name), ...] in visit_nodes pre-order

# Validate pool headers in a memory chunk (x64, Windows 8+):
hits = scan_pool_headers(
    buf,
    [PoolConstraint(b"Proc", size=(600, None),
                    page_type=PAGE_TYPE_NONPAGED | PAGE_TYPE_FREE)],
    alignment=0x10, layout="x64", vista_semantics=True,
)
# -> [(header_offset, b"Proc"), ...] in scan order
```

A runnable version (builds a synthetic hive + pool buffer end to end) is
`examples/quickstart.py` in this package directory:

```
python quickstart.py
```

## API

### `walk_hive(hive, *, root_cell_offset=None, storage_length=None, volatile_storage_length=0)`

- `hive`: bytes-like — the complete flat hive image: 4 KiB `regf` base
  block followed by the hive bins. This is the on-disk layout of Windows
  hive files.
- `root_cell_offset`: root cell index; default reads it from the base
  block like the reference's `root_cell_offset` property (`RootCell`
  when the signature is `regf`, else `0x20`).
- `storage_length`: non-volatile storage length, the invalid-cell bound
  of the reference's address translation; default `len(hive) - 0x1000`.
- `volatile_storage_length`: volatile storage length; flat `.dat` files
  carry none, so the default 0 makes volatile cells get skipped, as the
  reference does with empty volatile storage.
- Returns `[(offset, "_CM_KEY_NODE", name), ...]` in `visit_nodes`
  pre-order. Raises `HiveError` on invalid input and on the malformed
  images where the reference's traversal raises.

### `scan_pool_headers(data, constraints, *, alignment=0x10, layout="x64", vista_semantics=True)`

- `data`: bytes-like buffer to scan.
- `constraints`: `PoolConstraint` rows — or duck-typed equivalents
  exposing `tag`/`size`/`page_type`/`index`, so the oracle's own
  `PoolConstraint` objects work directly. Tags must be unique.
  `size=(min, max)` bounds `alignment * BlockSize`; `page_type` is a
  bitmask of `PAGE_TYPE_PAGED|NONPAGED|FREE`; `index=(min, max)` bounds
  `PoolIndex`. A 0/None bound disables that side, mirroring the
  reference's truthiness tests.
- `alignment`: pool block alignment (0x10 for 64-bit, 0x8 for 32-bit —
  the reference's own values).
- `layout`: `"x64"` covers both bundled x64 header tables; `"x86"` is
  the 32-bit bitfield layout.
- `vista_semantics`: pool-type mapping — True for Vista-and-later
  (paged == odd PoolType), False for pre-Vista (paged == even non-zero
  PoolType), matching the reference's `POOL_HEADER_VISTA` vs
  `POOL_HEADER` classes.
- Returns `[(header_offset, tag), ...]` in scan order. Raises
  `PoolScanError` on invalid input.

Introspection: `vol_mojo.backend_info()` and `vol_mojo.native_available()`
report the active backend. `VOL_MOJO_DISABLE_NATIVE=1` forces the
fallback; `VOL_MOJO_NATIVE_LIB` overrides the native library path.

### volatility3 integration (the component seam)

`vol_mojo.volatility3_integration` duck-types the oracle's objects
without importing volatility3:

- `scan_like_pool_header_scanner(data, constraints, alignment, *, layout="x64", vista_semantics=True)`
  — batch equivalent of `PoolHeaderScanner.__call__`: takes the oracle's
  own `PoolConstraint` objects and returns `(constraint, header_offset)`
  pairs exactly as the oracle yields `(constraint, header)`.
  Inside volatility3 the seam is the scanner instance: a plugin that
  already holds the raw chunk bytes can swap the per-hit validation loop
  for this batch call and keep everything downstream unchanged.
- `walk_hive_tuples(hive_bytes)` — the `visit_nodes` visitor tuples for
  a flat hive file, without a memory image, kernel symbols, or a
  context. Inside volatility3 the seam is the `RegistryHive` layer.

## Benchmark

Measured with `benchmarks/bench_volatility3_hive.py` in this repository
(oracle: volatility3 2.28.2 from PyPI). Workloads: hive walk over a
generated **~20 000-key-node valid hive** (~2 MiB, generator in
`tests/test_volatility3_hive_fixtures.py` — no copyrighted hives); pool
scan over an **8 MiB buffer with 2000 planted headers + decoys** against
the 13 builtin constraints (x64, Vista+ semantics). Median of 5 warm
runs; cold = first call in a fresh interpreter (import + dlopen + first
call). Environment: **Apple M4 Max, macOS 26.6.2 arm64, Python 3.12.14,
Mojo 1.1.0, volatility3 2.28.2**, 2026-09-19.

| workload | volatility3 2.28.2 warm (s) | vol-mojo cold (s) | vol-mojo warm (s) | warm speedup |
|---|---:|---:|---:|---:|
| hive walk | 32.7637 | 0.0735 | 0.012567 | 2607x |
| pool scan | 0.1660 | 0.0417 | 0.006406 | 26x |
| hive walk — Python fallback | 32.7637 | — | 0.0460 | 712.8x |
| pool scan — Python fallback | 0.1660 | — | 0.0223 | 7.4x |

Correctness gate (asserted before every timing run): exact equality with
the oracle on both cells — the full ~20k-tuple walk and every passing
pool-header offset, order included.

Honest framing:

- The oracle's `visit_nodes` pays for the full framework object factory
  per node (template construction, layer mapping, symbol lookups); the
  kernel walks the same bytes directly. The speedup is real but the
  comparison target is the framework walk a plugin performs, not a
  hand-rolled Python parser.
- The pool scan's regex-trie tag pass is the same `re` machinery on
  both sides (the vendored fallback uses it too); the kernel's win is
  the batch FFI shape and the branch-light validation loop.
- The pure-Python fallback keeps exact parity everywhere; it is what
  non-native platforms get, and it is still comfortably faster than the
  framework path for these workloads.

## Scope and limitations

- **Flat `.dat` hive images only.** The walker assumes the on-disk
  identity page mapping (bins follow the 4 KiB base block). Hives
  carved out of a memory image through the in-memory HMAP, compressed
  (`.dat` in transaction-log/dirty states needing log replay), or
  multi-segment volatile storages are out of scope — run volatility3's
  own layer for those. Value-data decoding (`vk` payloads), security
  descriptors (`sk`), and big-data (`db`) chains are likewise not part
  of the walk contract: the tuples describe the key-node tree, exactly
  what `visit_nodes` visits.
- **The walk never mutates** and assumes the input is a tree: a
  pathological/cyclic image trips an iteration guard (status surfaced
  as `HiveError`) where the reference would raise `RecursionError` or
  loop; trees deeper than Python's recursion limit are a divergence by
  construction (the reference crashes there; this package returns the
  tuples).
- **Pool scan = one chunk.** The contract is
  `PoolHeaderScanner.__call__(data, 0)` on a single buffer. The layer
  chunking (`layer.scan` 16 MiB chunks with overlap), the downstream
  object carving (`header.get_object`, top-down/bottom-up), the
  object-type map test, and big-page pool trackers stay in volatility3.
  Buffers must be at least 2 bytes (the reference cannot build a scan
  layer for smaller inputs either).
- **Determinism**: both entry points are pure functions of the input
  bytes — no randomness, no clocks, no global state.
- Single-threaded kernel (determinism first).
- volatility3 is *not* a dependency: inputs are plain bytes, so the
  package also works standalone for any flat-hive or pool-scan task.

## Citing

If you use vol-mojo in academic work, please cite both volatility3 and
this package:

> vol-mojo: Mojo-accelerated registry-hive walking and pool-header
> validation with volatility3-compatible semantics. thyn-ai, 2026.
> https://github.com/thyn-ai/mojo-kernels
> (citation file with DOI forthcoming)

volatility3 itself: The Volatility Foundation,
https://github.com/volatilityfoundation/volatility3 .

## License

Apache-2.0, © 2026 Algenta The kernel and wrapper are clean-room
implementations written against the documented on-disk formats and the
publicly described scanner semantics. volatility3 is used only as the
test/benchmark oracle, never as a runtime dependency, and no
volatility3 code is reproduced in this package.

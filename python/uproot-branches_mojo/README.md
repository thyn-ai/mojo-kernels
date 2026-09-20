# uproot-mojo

**Fast ROOT TBranch reading for `std::vector<T>` and string branches** —
the [uproot](https://github.com/scikit-hep/uproot5) (Scikit-HEP)
deserialization hot path — powered by a Mojo kernel, with a vendored
pure-Python fallback. It reads split-level-0 `std::vector<int32_t>`,
`std::vector<int64_t>`, `std::vector<float>`, `std::vector<double>`,
`std::string` and `std::vector<std::string>` TBranches straight from a ROOT
file and returns exactly the offsets+content buffers uproot delivers —
without parsing TStreamerInfo/TTree metadata at all. Prebuilt per-platform
binaries mean **no Mojo toolchain is ever required** on an end user's
machine, and results match uproot **byte-for-byte** (measured, not
approximate).

Where uproot interprets objects entry by entry in Python (the
`uproot/containers.py` `AsVector`/`AsString` path used for object branches
such as `std::vector<std::string>`), the Mojo kernel walks the whole basket
in one compiled pass — **172.6× faster** on that path (measured; table
below).

## Install

```
pip install uproot-mojo
```

Per-platform wheels (macOS arm64, Linux x86_64) carry the native kernel,
self-contained (the Mojo runtime is vendored into the wheel; nothing to
compile, no absolute rpaths). On any other platform — including Windows, where CI runs this package's fallback suite
([`windows-fallback`](https://github.com/thyn-ai/mojo-kernels/actions/workflows/windows-fallback.yml)) —
the same wheel API runs on the vendored pure-Python fallback, silently and
correctly. There is no sdist: a source tarball cannot rebuild the native
library. Runtime dependencies: `numpy` and `cramjam` (zlib basket
decompression). `uproot`/`awkward` are **not** dependencies (they are the
differential-test oracle); `awkward` is an optional extra only for
`to_awkward()`.

## Quickstart

```python
import uproot_mojo

# vector<float64> branch -> offsets + content, identical to uproot's
result = uproot_mojo.read_branch("events.root", "pt", dtype="float64")
result.offsets     # int64[n_entries + 1]: cumulative items per entry
result.content     # float64[n_items]: the flat item buffer (native-endian)
result.to_list()   # [[1.5, 2.5], [], [4.5], ...]

# std::string branch (no dtype needed; layout is validated structurally)
labels = uproot_mojo.read_branch("events.root", "label")
labels.to_list()   # ["muon", "", "electron", ...]

# std::vector<std::string> branch (ROOT-written files)
tags = uproot_mojo.read_branch("events.root", "tags")
tags.to_list()     # [["b-jet", "central"], [], ...]
```

A runnable demo that writes a fixture with `uproot.recreate`, reads every
branch, and cross-checks against uproot is `quickstart.py` next to this
README in the repository:

```
python python/uproot-branches_mojo/quickstart.py
```

## API

`uproot_mojo.read_branch(file, branch, *, dtype=None, tree=None)`

- `file` — path to a ROOT file.
- `branch` — branch name (baskets are located by their TKey name).
- `dtype` — required for numeric jagged branches: `"int32"`, `"int64"`,
  `"float32"` or `"float64"` (NumPy dtypes accepted). The package never
  guesses a data type from bytes; string layouts are self-describing, so
  `dtype` is omitted (and rejected) for them. This is the one place where
  the API surface is deliberately wider than "branch name only": the type
  of a numeric branch lives in the TTree metadata, which this package
  intentionally does not parse.
- `tree` — tree name, required only when several trees contain a branch
  with the same name (the tree name is the basket key's title).

Returns (both are `NamedTuple`s with `.offsets`, `.content`, `.backend`,
`.to_list()`, `.to_awkward()`):

- `JaggedArray(offsets, content, kind, backend)` for `vector<T>` and
  `std::string` branches — `kind` is `"numeric"` or `"string"`.
- `JaggedStringArray(offsets, string_offsets, content, backend)` for
  `vector<string>` branches (two jagged levels).

Errors are explicit: `RootFileError` (not a ROOT file / malformed),
`UnsupportedBranchError` (outside the supported scope), `BasketDataError`
(corrupt basket bytes, with a stable numeric `.code`). There are no silent
fallbacks between layouts: every basket is fully validated, and a basket
that fails validation raises.

Backend diagnostics: `uproot_mojo.backend_info()` and
`uproot_mojo.native_available()`. Force the fallback with
`UPROOT_MOJO_DISABLE_NATIVE=1` (used by the differential suite); override
the shared-library path with `UPROOT_MOJO_NATIVE_LIB`.

## Supported scope (and what is not)

Supported branch layouts — both are validated per basket:

- **uproot-written jagged layout** (what `uproot.recreate` emits for
  `var * <numeric>`): raw big-endian items per entry, per-entry byte
  borders from the basket's entry-offset table.
- **ROOT-native collection layout** (what CERN ROOT emits for
  `std::vector<T>` and `std::vector<std::string>`): a 10-byte collection
  header per entry (4-byte byte count with `kByteCountMask`, 2-byte
  version, 4-byte item count) followed by the items; strings as
  length-prefixed `TString`s (1-byte length below 255, `0xFF` + 4-byte
  length otherwise). Auto-detection between the two numeric layouts is
  exact (a collection header's 10 bytes are never divisible by the item
  size); between `std::string` and `vector<string>` it is driven by the
  byte-count validation, which must match the entry length exactly.

Supported files: free-standing TBasket keys (located by a validated
structural scan — basket keys are not registered in the file's TKeysList,
and the linear key chain is broken by free-space gaps), baskets carrying
an `fEntryOffset` table (all jagged/string baskets written by
`uproot.recreate`, and TTree baskets written by CERN ROOT), **zlib**
compression (`ZL` blocks, multi-block chains) and stored-uncompressed
baskets.

Not supported (raises `UnsupportedBranchError`, never a silent guess):
rectilinear/counter branches, other item types (`bool`, `int8`, `int16`,
unsigned ints, …), nested vectors (`vector<vector<T>>`), split branches
(`TBranchElement` sub-branches), embedded baskets, baskets without an
entry-offset table, LZMA/LZ4/ZSTD compression, and `TBranch`es inside
`TTree` friends or subdirectories.

Note on `vector<string>` fixtures: `uproot.recreate` cannot write
`std::vector<std::string>` (its writer only emits NumPy-dtype jagged
contents and top-level `std::string`), so that layout is covered at the
basket level: synthetic ROOT-native baskets are deserialized by uproot's
own container models (`uproot.containers.AsVector`/`AsString` — the exact
interpreted path this kernel replaces) and compared buffer-for-buffer.
Numeric and `std::string` branches are covered end-to-end with real
`uproot.recreate` files.

## How it works

1. The wrapper scans the file for TBasket TKeys of the branch (strict
   structural validation — a coincidental `\x07TBasket` byte run inside a
   payload cannot survive the checks), ordered by file position, which is
   the entry order for append-filled trees.
2. Each basket payload is decompressed with cramjam (ROOT's 9-byte-block
   zlib framing), and the entry-offset table yields per-entry byte
   borders.
3. The walker — the Mojo kernel where available, the vendored Python
   walker otherwise — validates and splices every entry into
   offsets+content buffers (batch-shaped C ABI: whole basket in, whole
   buffers out; the kernel allocates nothing and has two passes, scan then
   fill).
4. Numeric content is byteswapped big-endian → native, so the returned
   arrays equal uproot's delivered buffers byte-for-byte (int64 offsets,
   native-endian content).

## Differential tests

`tests/test_uproot-branches_differential.py` runs twice (native, and
`UPROOT_MOJO_DISABLE_NATIVE=1`) and asserts, for every branch of real
`uproot.recreate` fixtures (multi-basket, empty entries, `inf`/`nan`,
254/255/300-char strings, two-tree files, uncompressed files), that
offsets and content bytes are **exactly equal** to uproot's buffers —
parity tolerance is zero. The ROOT-native collection layouts are checked
against uproot's `AsVector`/`AsString` deserializers on synthetic baskets,
including corrupt-input rejection (`ERR_TRUNCATED`, `ERR_BYTECOUNT`,
`ERR_COUNT`, `ERR_STRING`, `ERR_BORDERS`) with identical codes on both
backends. Suite: 30 tests per backend.

## Benchmark

Reproducible: `benchmarks/bench_uproot-branches.py` (median of 5 runs;
fixture built by the script with `uproot.recreate`, 50,000 entries over 10
baskets, 4.41 MiB zlib; correctness asserted before timing). Measured on
an Apple M4 Max, macOS 26.6.2, Python 3.12.14, numpy 2.5.3, uproot 5.7.6,
awkward 2.14.0, cramjam 2.12.1 (2026-09-19).

Native backend (Mojo kernel):

| cell | uproot (ms) | uproot-mojo (ms) | speedup |
|---|---|---|---|
| cold (fresh process, one file→array call) | 429.75 | 45.54 | **9.4×** |
| warm steady-state (full open + read per call) | 15.50 | 17.09 | 0.9× |
| warm, uproot memoized repeat on open TBranch (context only) | 0.02 | (re-reads every call) 17.09 | n/a |
| `vector<string>` basket walk, 50k entries (uproot per-entry Python vs one kernel call) | 296.52 | 1.72 | **172.6×** |

Pure-Python fallback backend (same machine, same fixture):

| cell | uproot (ms) | uproot-mojo fallback (ms) | speedup |
|---|---|---|---|
| cold (fresh process, one file→array call) | 488.66 | 82.61 | **5.9×** |
| warm steady-state (full open + read per call) | 22.33 | 61.26 | 0.4× |
| `vector<string>` basket walk, 50k entries | 429.16 | 131.40 | **3.3×** |

Reading the numbers: the cold path is where the design pays off — uproot
parses and interprets the file's streamer metadata before it can read
anything, while this package goes straight to the baskets. Warm
steady-state is decompression-bound on both sides (zlib inflate is ~14 ms
of the ~16 ms), so parity is the truthful result there; uproot's memoized
re-read on an already-open branch serves from its cache, which this
stateless API never does. The `vector<string>` cell is the object
deserialization hot loop itself (uproot's interpreted per-entry container
readers vs one compiled walk) and shows the kernel's ceiling.

## Provenance

Clean-room implementation: the Mojo kernel and the ROOT file reader were
written fresh from the documented ROOT I/O format (ROOT's
`io/doc/TFile`/`ttree.md` specifications). The differential oracle is the
published `uproot` package (5.7.6). Apache-2.0. By Algenta.

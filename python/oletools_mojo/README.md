# oletools-mojo

Mojo-accelerated MS-OVBA VBA decompression — a drop-in faster replacement
for [oletools](https://pypi.org/project/oletools/)'
`olevba.decompress_stream`, the pure-Python LZ decoder every `VBA_Parser`
macro extraction goes through — with a vendored pure-Python fallback for
platforms without a native build (including Windows — tested there in CI: [`windows-fallback`](https://github.com/thyn-ai/mojo-kernels/actions/workflows/windows-fallback.yml)).

```python
from oletools_mojo import decompress_stream

source = decompress_stream(compressed_container)   # bytes in, bytes out
```

- **Byte-identical to oletools**: the decoder matches the published
  `oletools==0.60.2` oracle byte-for-byte — zero tolerance — on every
  stream in the differential suite, on the native backend AND the forced
  fallback. Error behaviour is matched too: same exception types AND
  messages (`ValueError`/`IndexError`/`struct.error`/`TypeError`). This
  includes the oracle's observable quirks (see below).
- **Much faster**: 68–90x warm on copy-token (LZ) streams and 5x on the
  RawChunk path (Apple M4 Max; full method and numbers below).
- **No toolchain needed**: per-platform wheels ship the compiled kernel
  (macOS arm64, Linux x86_64). Everywhere else the package transparently
  uses its pure-Python fallback — identical output, just slower.
- **Zero dependencies**: bytes in, bytes out. `oletools` is only the test
  oracle (and an optional, user-side integration target), never a runtime
  dependency.
- Force the fallback with `OLETOOLS_MOJO_DISABLE_NATIVE=1`; inspect the
  active backend with `oletools_mojo.backend_info()` /
  `oletools_mojo.get_backend()`.

## API

```python
decompress_stream(compressed_container) -> bytes
```

Accepts `bytearray`, `bytes`, or anything `bytearray()` accepts (identical
conversion semantics to the oracle, including its `TypeError`s) and returns
the decompressed bytes of one MS-OVBA 2.4.1 CompressedContainer.

## olevba integration

`oletools.olevba.VBA_Parser` resolves `decompress_stream` as a module
global at call time, so patching one attribute accelerates all of oletools'
macro extraction (olevba, mraptor, embedded VBA_Parser uses) without
forking it:

```python
import oletools_mojo.olevba_integration as integration
integration.install()          # oletools.olevba now uses oletools_mojo

from oletools.olevba import VBA_Parser
with VBA_Parser("sample.xlsm") as parser:
    for (_, _, filename, code) in parser.extract_macros():
        print(filename)

integration.uninstall()        # restore the stock oletools behaviour
```

`oletools` is imported lazily by `install()`; it is an optional, user-side
dependency. Behaviour is unchanged apart from speed (the patched function
is bit-exact with the stock one, including exception types).

## oletools-compatibility notes (pinned by the differential suite)

The oracle is the published PyPI package, used as a black box. Where the
spec text and the oracle's observable behaviour differ, we match the
oracle:

- Empty input raises `IndexError` (the signature byte is read
  unconditionally); a signature byte other than `0x01` raises `ValueError`.
- Chunk headers are strict 16-bit little-endian reads: a container ending
  in a single trailing byte raises `struct.error`, as does a CopyToken
  whose second byte is past the container end.
- Chunk signature bits must be `0b011`; a RawChunk's size field must encode
  exactly 4098. A short final RawChunk is copied leniently (as many bytes
  as remain).
- A declared CompressedChunkSize beyond the container is tolerated (the
  oracle logs a warning and clamps to the container end).
- CopyToken bit geometry is `bit_count = max(4, ceil(log2(difference)))`
  with exact integer arithmetic — verified to agree with the oracle's
  float64 `math.log(difference, 2)` at **every** difference from 1 to
  2,000,000 (and by construction far beyond; see Unsupported scope).
- A CopyToken at `difference == 0` raises `ValueError` (the oracle's float
  log raises a math-domain error).
- Overlapping copies (offset < length) replicate byte-by-byte (RLE
  semantics), and a CopyToken offset beyond the decompressed length is
  resolved with Python negative-index wrap semantics, re-evaluated against
  the growing output at every byte — the malformed-stream behaviour of the
  oracle, reproduced exactly.

## Unsupported scope

- **One function only**: `decompress_stream`. No OLE/CFB parsing, no VBA
  project parsing, no macro analysis, **no compression API** — those stay
  in oletools (use the integration shim above to accelerate oletools' own
  extraction).
- **Pathological single chunks**: for a CopyToken at `difference >= 2**29`
  (one chunk decompressing to 512 MiB or more — far outside the 4096-byte
  spec maximum and never produced by real producers) the oracle's float64
  bit-count computation has power-of-two rounding anomalies (first at
  `2**29`, where it yields 30 instead of 29) that this package
  intentionally does not reproduce; output for such streams may differ
  from the oracle. The boundary is pinned by a test so it is explicit, not
  accidental.
- **Platforms**: native wheels are built for macOS arm64 and Linux x86_64.
  Everywhere else (Windows included — tested there in CI: [`windows-fallback`](https://github.com/thyn-ai/mojo-kernels/actions/workflows/windows-fallback.yml))
  the package installs fine and uses the vendored pure-Python fallback — identical output, oracle-level
  speed.

## Benchmarks

Measured on this machine (Apple M4 Max, macOS, Python 3.12.14, Mojo 1.1.0,
oletools 0.60.2), protocol: cold first call (doubles as the correctness
gate), warm = median of 5 runs; both sides decode the SAME
CompressedContainer bytes end to end
(`oletools.olevba.decompress_stream` vs `oletools_mojo.decompress_stream`).
Workloads are generated locally from fixed seeds and compressed with the
fresh MS-OVBA compressor in `tests/test_oletools_vba_fixtures.py` (oletools
ships no public compressor). Reproduce with
`python benchmarks/bench_oletools_vba.py` from the repository. Speedup
factors are warm oletools time / warm oletools_mojo time.

<!-- BENCHMARKS:START -->
| stream | decompressed | oletools cold (ms) | oletools warm (ms) | oletools_mojo cold (ms) | oletools_mojo warm (ms) | warm speedup |
|---|---:|---:|---:|---:|---:|---:|
| VBA source ~210 KB | 0.21 MB | 28.95 | 27.88 | 0.582 | 0.349 | 79.8x |
| VBA source ~2.1 MB | 2.14 MB | 318.36 | 299.41 | 3.984 | 3.329 | 89.9x |
| text-like 4 MiB | 4.19 MB | 716.42 | 718.35 | 9.723 | 10.213 | 70.3x |
| RLE 8 MiB | 8.39 MB | 780.13 | 751.60 | 11.927 | 10.996 | 68.4x |
| random 1 MiB (RawChunks) | 1.05 MB | 0.56 | 0.39 | 0.222 | 0.077 | 5.0x |
| mixed 2 MiB | 2.10 MB | 152.27 | 156.16 | 1.798 | 2.005 | 77.9x |
<!-- BENCHMARKS:END -->

The oracle decodes the LZ path one Python-level byte append at a time
(~6–11 MB/s), while the kernel sustains 0.4–0.8 GB/s there; on RawChunks
the oracle's slice-extend is already C-speed, so the residual 5x is the
chunk-walk overhead. The first cold call carries the one-time dlopen/ABI
handshake (sub-millisecond).

## Development

Source, tests, and development: <https://github.com/thyn-ai/mojo-kernels>

- Kernel: `kernels/oletools-vba/` (clean-room Mojo, C ABI v1)
- Wrapper: `python/oletools_mojo/` (`oletools_mojo` import package)
- Differential suite (native + forced fallback):
  `bash scripts/test_all_oletools_vba.sh` — 4171 tests per backend
  (4096-point CopyToken bit-geometry sweep, 2M-point bit-count sweep,
  seeded structured fuzz, pinned edge cases), byte-exact and
  error-exact against the oletools==0.60.2 oracle.
- Quickstart: `python python/oletools_mojo/examples/quickstart.py`

License: Apache-2.0, © 2026 Algenta

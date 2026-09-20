# pypdf-filters-mojo

Mojo-accelerated PNG-predictor and LZW decoders for PDF streams — the two
pure-Python hot loops in [pypdf](https://pypi.org/project/pypdf/)'s
`FlateDecode`/`LZWDecode` filter path — with a vendored pure-Python fallback
for platforms without a native build (including Windows — tested there in CI: [`windows-fallback`](https://github.com/thyn-ai/mojo-kernels/actions/workflows/windows-fallback.yml)).

```python
import zlib
from pdf_mojo import decode_png_prediction, decode_lzw

# Reconstruct an image XObject's flate stream written with PNG predictors:
raw = decode_png_prediction(
    zlib.decompress(stream_data),   # still predictor-coded bytes
    columns=640, colors=3, bits_per_component=8, predictor=15,
)

# Decode an LZW stream (EarlyChange=1 is the PDF default):
raw = decode_lzw(lzw_data, early_change=1)
```

- **Byte-identical to pypdf**: both decoders match the published
  `pypdf==6.19.0` oracle byte-for-byte — zero tolerance — on every stream
  in the differential suite, on the native backend AND the forced
  fallback. This includes pypdf's observable quirks (see below).
- **Much faster**: 12–15x warm end-to-end on multi-MB PNG/TIFF predictor
  streams and 800–1800x on LZW streams (Apple M4 Max; full method and
  numbers below).
- **No toolchain needed**: per-platform wheels ship the compiled kernel
  (macOS arm64, Linux x86_64). Everywhere else the package transparently
  uses its pure-Python fallback — identical output, just slower.
- **Zero dependencies**: bytes in, bytes out. `pypdf` is only the test
  oracle, never a runtime dependency.
- Force the fallback with `PDF_MOJO_DISABLE_NATIVE=1`; inspect the active
  backend with `pdf_mojo.backend_info()` / `pdf_mojo.get_backend()`.

## API

```python
decode_png_prediction(data, columns, colors, bits_per_component, predictor=15) -> bytes
decode_lzw(data, early_change=1) -> bytes
```

`decode_png_prediction` expects the *already inflated* bytes of a
FlateDecode stream (zlib stays in your stack, exactly as in pypdf) and
returns the reconstructed rows. Predictors 1 (none), 2 (TIFF), and 10–15
(PNG) are supported, with `columns >= 1`, `colors >= 1`, and
`bits_per_component` in {1, 2, 4, 8, 16}. Invalid parameters raise
`ValueError`; corrupt streams raise `pdf_mojo.PdfFilterError` (e.g. a PNG
row filter byte > 4).

## pypdf-compatibility notes (pinned by the differential suite)

The oracle is the published PyPI package, used as a black box. Where the
spec and pypdf's observable behaviour differ, we match pypdf:

- bytes-per-pixel is `floor(colors * bits_per_component / 8)` for BOTH the
  TIFF and PNG paths. For sub-byte bpc with few colors this floors to 0,
  which makes the "left neighbour" the byte itself (self-addition); RFC
  2083 would use 1. Real-world producers never emit those combinations.
- The per-row PNG filter byte (0–4) is honoured for any predictor 10–15
  (lenient), a ragged final row is zero-padded to a full row *before*
  filtering, and a filter byte > 4 is an error. TIFF Predictor 2 processes
  a short final row without padding (output length equals input length).
- LZW: decoding stops at the EOD code or when fewer bits than the current
  code width remain (zero padding bits may surface as trailing 0x00
  literals), garbage after EOD is ignored, and any code >= the next free
  table entry decodes as `prev_string + prev_string[0]` (lenient KwKwK).
- LZW code width grows at table index `(1 << width) - 1` for BOTH
  EarlyChange 0 and 1 — pypdf 6.19.0's observable behaviour at the 9→10
  and 10→11 bit transitions (verified by construction). Streams from
  strict EarlyChange=0 producers that grow the width at `(1 << width)`
  decode the way pypdf decodes them.

## Benchmarks

Measured on this machine (Apple M4 Max, macOS, Python 3.12, Mojo 1.1.0,
pypdf 6.19.0), protocol: cold first call (doubles as the correctness
gate), warm = median of 5 runs; both sides end-to-end on identical bytes
(oracle via `pypdf.filters.decode_stream_data`; pdf_mojo via
`zlib.decompress` + kernel for flate streams, raw bytes for LZW).
Reproduce with `python benchmarks/bench_pypdf_filters.py` from the
repository. Speedup factors are warm pypdf time / warm pdf_mojo time.

<!-- BENCHMARKS:START -->
PNG/TIFF predictor reconstruction (end-to-end decode):

| stream | decoded | pypdf cold (ms) | pypdf warm (ms) | pdf_mojo cold (ms) | pdf_mojo warm (ms) | warm speedup |
|---|---:|---:|---:|---:|---:|---:|
| RGB 1024x768x3 bpc=8 | 2.36 MB | 217.21 | 167.92 | 13.096 | 13.030 | 12.9x |
| RGBA 2048x512x4 bpc=8 | 4.19 MB | 287.46 | 283.69 | 20.934 | 20.863 | 13.6x |
| gray16 1600x1200 bpc=16 | 3.84 MB | 294.64 | 276.91 | 23.404 | 23.011 | 12.0x |
| TIFF2 2048x1024x3 bpc=8 | 6.29 MB | 419.42 | 430.20 | 27.207 | 27.868 | 15.4x |

LZW decode:

| stream | decoded | pypdf cold (ms) | pypdf warm (ms) | pdf_mojo cold (ms) | pdf_mojo warm (ms) | warm speedup |
|---|---:|---:|---:|---:|---:|---:|
| text-like 4 MiB | 4.19 MB | 10633.35 | 10371.66 | 13.592 | 12.823 | 808.8x |
| random 1 MiB | 1.05 MB | 5350.35 | 5702.35 | 4.793 | 3.210 | 1776.5x |
| RLE 8 MiB | 8.39 MB | 32504.84 | 29088.83 | 20.357 | 17.417 | 1670.1x |

The zlib inflate is common to both sides, so end-to-end predictor speedups
are compressed by Amdahl's law: kernel-only reconstruction measures ~0.8
GB/s on mixed-filter rows vs pypdf's ~14 MB/s predictor loop (~55x); LZW
has no shared stage — pypdf's bit-at-a-time Python loop runs at 0.2–0.4
MB/s, the kernel at 320–480 MB/s.
<!-- BENCHMARKS:END -->

## Development

Source, tests, and development: <https://github.com/thyn-ai/mojo-kernels>

- Kernel: `kernels/pypdf-filters/` (clean-room Mojo, C ABI v1)
- Wrapper: `python/pypdf_filters_mojo/` (`pdf_mojo` import package)
- Differential suite (native + forced fallback):
  `bash scripts/test_all_pypdf_filters.sh`
- Quickstart: `python python/pypdf_filters_mojo/examples/quickstart.py`

License: Apache-2.0, © 2026 Algenta

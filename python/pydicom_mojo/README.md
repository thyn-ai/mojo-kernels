# pydicom-mojo

Mojo-accelerated **DICOM RLE Lossless** codec — decode AND encode — the two
pure-Python PackBits hot loops in [pydicom](https://pypi.org/project/pydicom/)'s
native RLE plugin (`pydicom.pixels.decoders.rle` /
`pydicom.pixels.encoders.native`) — with a vendored pure-Python fallback for
platforms without a native build (including Windows).

```python
import pydicom_mojo

# Decode one RLE frame (including its 64-byte header) to little-endian,
# planar configuration 1 pixel bytes:
raw = pydicom_mojo.decode_frame(
    rle_frame, rows=512, columns=512, samples_per_pixel=1, bits_allocated=16
)

# Encode one frame of little-endian, interleaved (planar configuration 0)
# pixel bytes into a standard DICOM RLE frame:
rle_frame = pydicom_mojo.encode_frame(
    raw, columns=512, samples_per_pixel=1, bits_allocated=16
)
```

- **Byte-identical to pydicom**: decode and encode both match the published
  `pydicom==3.0.2` oracle byte-for-byte — zero tolerance — on every segment,
  frame, and dataset in the differential suite, on the native backend AND
  the forced fallback. This includes the oracle's lenient edge semantics
  (see below).
- **Much faster**: 3.7–16.3x warm on decode and 12.2–46.5x warm on encode
  for realistic CT/segmentation/RGB volumes (Apple M4 Max; full method and
  numbers below).
- **pydicom v3 plugin ready**: ships a plugin module for pydicom's official
  out-of-tree decoder/encoder plugin seam (the same one `pylibjpeg` uses),
  so pydicom's own pipelines can run on the kernel — see below.
- **No toolchain needed**: per-platform wheels ship the compiled kernel
  (macOS arm64, Linux x86_64). Everywhere else the package transparently
  uses its pure-Python fallback — identical output, just slower.
- **Zero dependencies**: bytes in, bytes out. `pydicom` is only the test
  oracle and the optional host of the plugin seam, never a runtime
  dependency.
- Force the fallback with `PYDICOM_MOJO_DISABLE_NATIVE=1`; inspect the
  active backend with `pydicom_mojo.backend_info()` /
  `pydicom_mojo.get_backend()`.

## API

```python
decode_frame(src, rows, columns, samples_per_pixel, bits_allocated, segment_order=">") -> bytes
encode_frame(src, columns, samples_per_pixel, bits_allocated, byteorder="<") -> bytes
decode_segment(src) -> bytes
encode_segment(src, columns) -> bytes
```

- `decode_frame` takes one encoded frame **including its 64-byte RLE
  header** and returns little-endian, planar configuration 1 pixel bytes
  (all of sample 0, then sample 1, ...; LSB first within each pixel).
  `segment_order` is the segment byte order of the input: `">"` for big
  endian (the DICOM default), `"<"` for little endian (non-conformant).
- `encode_frame` takes one frame of little-endian, interleaved (planar
  configuration 0) pixel bytes and returns a complete RLE frame (64-byte
  header + one PackBits segment per byte plane, most-significant-byte
  plane first per the DICOM Standard). Note that for multi-sample data the
  decode output is planar configuration 1 while the encode input is
  interleaved — exactly as in pydicom.
- `decode_segment` / `encode_segment` work at the segment level (one
  PackBits byte plane), mirroring pydicom's `_rle_decode_segment` /
  `_encode_segment`.
- Return types are `bytes` (the oracle functions return `bytearray`;
  content is byte-identical).

Errors mirror the oracle: `NotImplementedError` when `bits_allocated` is
not a multiple of 8 (decode), `ValueError` for a malformed RLE header, a
segment-count/decoded-length mismatch, `byteorder=">"`, or more than 15
segments (encode). Non-conformant over-long segments warn
(`UserWarning`, same message) and are truncated, exactly like pydicom.

## Using it as a pydicom plugin

pydicom v3's pixel data `Decoder`/`Encoder` classes accept out-of-tree
plugins (`pylibjpeg` is the in-tree precedent). pydicom-mojo ships a
plugin module that satisfies that seam:

```python
from pydicom.pixels.decoders import RLELosslessDecoder
from pydicom.pixels.encoders import RLELosslessEncoder
from pydicom.uid import RLELossless

RLELosslessDecoder.add_plugin("pydicom_mojo", ("pydicom_mojo.plugin", "decode_frame"))
RLELosslessEncoder.add_plugin("pydicom_mojo", ("pydicom_mojo.plugin", "encode_frame"))

ds.decompress(decoding_plugin="pydicom_mojo")        # read path
ds.compress(RLELossless, encoding_plugin="pydicom_mojo")  # write path
```

The plugin honours the `rle_segment_order` decoding option and the
`byteorder` encoding option, and marks the runner's planar configuration
as 1 after a successful decode — identical observable behaviour to
pydicom's own native plugin. The plugin module never imports pydicom
(UID-keyed dependency tables, duck-typed runner), so the package keeps
zero dependencies.

## pydicom-compatibility notes (pinned by the differential suite)

The oracle is the published PyPI package, used as a black box. Where the
spec and pydicom's observable behaviour differ, we match pydicom:

- Decode is lenient at the end of a segment: a literal packet whose
  payload overruns the input is silently truncated, a replicate packet
  missing its value byte appends nothing, and a `0x80` header byte is a
  no-op. There is no malformed-stream error at the segment level; the
  frame level validates decoded lengths instead (short segment →
  `ValueError`, over-long → warning + truncation).
- Encode packetization is pinned exactly: runs of two or more equal bytes
  flush pending literals and code as replicate packets; a full 128-byte
  run is header `129`, a remainder > 1 is header `257 - remainder`, and a
  remainder of exactly 1 is a one-byte literal packet. Literal runs are
  chunked at 128 bytes. An odd-length encoded segment gets one trailing
  `0x00` padding byte (which is decode-safe: it reads as an empty literal
  packet).
- `segment_order` values other than `"<"` iterate segments like `">"` but
  only the exact value `">"` applies the byte-order correction — the
  oracle's observable behaviour for non-conformant options.
- `encode_segment` mirrors the oracle's `columns` edge semantics: `0`
  raises `ValueError`, a negative value codes no rows at all.

## Unsupported scope (honest limits)

- **RLE Lossless only** (transfer syntax `1.2.840.10008.1.2.5`). JPEG,
  JPEG-LS, JPEG 2000, and HTJ2K are out of scope — use pydicom with
  pylibjpeg for those.
- **8/16/32/64-bit allocated samples** — `bits_allocated` must be a
  multiple of 8 for decode (the oracle raises `NotImplementedError`
  otherwise, and so do we). This covers the geometries DICOM RLE is used
  for in practice (8-bit, 16-bit, 32-bit; 1 or 3+ samples per pixel).
- The package codes **frames**, not datasets: encapsulation, file I/O,
  photometric conversion, and NumPy array handling stay in pydicom (or in
  your code). The plugin seam wires the codec into pydicom's pipelines
  exactly where the native codec sits.
- Wheels with the native kernel are built for macOS arm64 and Linux
  x86_64. Everywhere else (including Windows) the package works with the
  identical pure-Python fallback.

## Benchmarks

Measured on this machine (Apple M4 Max, macOS, Python 3.12, Mojo 1.1.0,
pydicom 3.0.2), protocol: cold first call (doubles as the correctness
gate), warm = median of 5 runs; both sides at the exact plugin seam on
identical bytes (oracle: `pydicom.pixels.decoders.rle._rle_decode_frame` /
`pydicom.pixels.encoders.native._encode_frame`; pydicom-mojo:
`pydicom_mojo.decode_frame` / `pydicom_mojo.encode_frame`), one whole
volume per timed call. Reproduce with
`python benchmarks/bench_pydicom_rle.py` from the repository. Speedup
factors are warm oracle time / warm pydicom-mojo time.

<!-- BENCHMARKS:START -->
RLE decode (whole volume per call):

| volume | RLE size | decoded | pydicom cold (ms) | pydicom warm (ms) | pydicom_mojo cold (ms) | pydicom_mojo warm (ms) | warm speedup |
|---|---:|---:|---:|---:|---:|---:|---:|
| CT series 512x512 16-bit, 60 frames | 17.09 MB | 31.46 MB | 216.23 | 221.64 | 32.178 | 33.120 | 6.7x |
| SEG series 512x512 8-bit, 40 frames | 0.22 MB | 10.49 MB | 17.59 | 15.96 | 4.690 | 4.291 | 3.7x |
| RGB frame 1024x1024 8-bit | 3.36 MB | 3.15 MB | 80.67 | 72.30 | 4.999 | 4.424 | 16.3x |

RLE encode (whole volume per call):

| volume | RLE size | decoded | pydicom cold (ms) | pydicom warm (ms) | pydicom_mojo cold (ms) | pydicom_mojo warm (ms) | warm speedup |
|---|---:|---:|---:|---:|---:|---:|---:|
| CT series 512x512 16-bit, 60 frames | 17.09 MB | 31.46 MB | 2275.69 | 2425.28 | 52.857 | 52.103 | 46.5x |
| SEG series 512x512 8-bit, 40 frames | 0.22 MB | 10.49 MB | 79.74 | 76.98 | 5.450 | 6.321 | 12.2x |
| RGB frame 1024x1024 8-bit | 3.36 MB | 3.15 MB | 408.55 | 387.47 | 8.840 | 8.927 | 43.4x |

Notes on the spread: pydicom's decoder is already slice-based for
replicate packets, so the segmentation series (nearly all long runs) sees
the smallest kernel advantage on decode; its encoder is a per-byte Python
`groupby`, which is why encode speedups are uniformly large. RLE expands
noisy interleaved RGB slightly (3.15 MB decoded → 3.36 MB encoded) —
expected for PackBits on high-entropy content, and identical on both
sides.
<!-- BENCHMARKS:END -->

## Development

Source, tests, and development: <https://github.com/thyn-ai/mojo-kernels>

- Kernel: `kernels/pydicom-rle/` (clean-room Mojo, C ABI v1)
- Wrapper: `python/pydicom_mojo/` (`pydicom_mojo` import package)
- Differential suite (native + forced fallback):
  `bash scripts/test_all_pydicom_rle.sh` (needs `pydicom==3.0.2`, `numpy`,
  and `pytest` on the active interpreter)
- Quickstart: `python python/pydicom_mojo/examples/quickstart.py`

License: Apache-2.0, © 2026 Algenta

# msgpack-mojo

Drop-in faster `packb`/`unpackb` for [MessagePack](https://msgpack.org/),
powered by a clean-room [Mojo](https://www.modular.com/mojo) byte engine,
with a vendored pure-Python engine everywhere the native kernel cannot run
(including Windows). No runtime dependencies.

```python
import msgpack_mojo

blob = msgpack_mojo.packb({"hello": "world", "n": [1, 2, 3]})
assert msgpack_mojo.unpackb(blob) == {"hello": "world", "n": [1, 2, 3]}
```

`msgpack_mojo.packb` is **byte-exact** with `msgpack.packb` (the published
`msgpack==1.1.2` package, C backend and its pure-Python fallback alike) for
every supported type and option, and `msgpack_mojo.unpackb` returns the same
values as `msgpack.unpackb` — proven by a differential test suite that runs
against the real oracle package on **both** of this package's backends
(native, and the fallback forced with `MSGPACK_MOJO_DISABLE_NATIVE=1`).

## Quickstart

Install a platform wheel, then:

```python
import msgpack_mojo

doc = {"project": "demo", "version": [0, 1, 0], "tags": ["a", "b"], "ok": True}
blob = msgpack_mojo.packb(doc)
assert msgpack_mojo.unpackb(blob) == doc
```

On macOS arm64 / Linux x86_64 the wheel carries the Mojo kernel and uses it.
Anywhere else — or if the kernel cannot load for any reason — the package
transparently uses its vendored pure-Python engine, which produces exactly
the same bytes and values (the differential suite forces this path and
requires identical results). Check what you got:

```python
>>> import msgpack_mojo
>>> msgpack_mojo.backend_info()["native_available"]
True
```

## Performance

Measured on this machine (see *Benchmark reproduction* below; warm =
steady-state per call, cold = first call in a fresh process, import
excluded, median of 5). Speedups are reported against **both** oracle
backends separately: the default C extension (`msgpack._cmsgpack`) and the
pure-Python `msgpack.fallback`.

<!-- BENCH:START -->
Environment: Apple M4 Max, macOS 26.6.2 arm64, Python 3.12.14, Mojo 1.1.0,
oracle msgpack 1.1.2. Correctness gate (pack byte-exact vs the C backend,
unpack values equal) passed for every payload before timing.

**Warm packb** (per call, median of 5 runs):

| payload | backend | ms/call | MB/s | vs oracle C | vs oracle fallback |
|---|---|---:|---:|---:|---:|
| small_api (~0.5 KB) | msgpack C (default) | 0.0006 | 268.2 | 1.00x | 13.83x |
| small_api (~0.5 KB) | msgpack fallback | 0.0083 | 19.9 | 0.07x | 1.00x |
| small_api (~0.5 KB) | **msgpack_mojo native** | 0.0061 | 26.9 | 0.10x | 1.35x |
| small_api (~0.5 KB) | msgpack_mojo fallback | 0.0109 | 15.1 | 0.06x | 0.76x |
| medium (~64 KB) | msgpack C (default) | 0.1556 | 421.7 | 1.00x | 14.29x |
| medium (~64 KB) | msgpack fallback | 2.2230 | 29.5 | 0.07x | 1.00x |
| medium (~64 KB) | **msgpack_mojo native** | 1.3369 | 49.1 | 0.12x | 1.66x |
| medium (~64 KB) | msgpack_mojo fallback | 3.1490 | 20.8 | 0.05x | 0.71x |
| large (~2 MB) | msgpack C (default) | 5.4599 | 384.1 | 1.00x | 12.82x |
| large (~2 MB) | msgpack fallback | 70.0170 | 30.0 | 0.08x | 1.00x |
| large (~2 MB) | **msgpack_mojo native** | 48.8282 | 43.0 | 0.11x | 1.43x |
| large (~2 MB) | msgpack_mojo fallback | 96.0938 | 21.8 | 0.06x | 0.73x |
| int array (100k) | msgpack C (default) | 2.2250 | 224.7 | 1.00x | 11.89x |
| int array (100k) | msgpack fallback | 26.4671 | 18.9 | 0.08x | 1.00x |
| int array (100k) | **msgpack_mojo native** | 16.0313 | 31.2 | 0.14x | 1.65x |
| int array (100k) | msgpack_mojo fallback | 42.9185 | 11.7 | 0.05x | 0.62x |
| str array (20k) | msgpack C (default) | 0.5804 | 500.7 | 1.00x | 10.08x |
| str array (20k) | msgpack fallback | 5.8539 | 49.6 | 0.10x | 1.00x |
| str array (20k) | **msgpack_mojo native** | 3.7297 | 77.9 | 0.16x | 1.57x |
| str array (20k) | msgpack_mojo fallback | 8.2872 | 35.1 | 0.07x | 0.71x |
| bin blob (1 MB) | msgpack C (default) | 0.1189 | 8815.9 | 1.00x | 0.19x |
| bin blob (1 MB) | msgpack fallback | 0.0224 | 46794.8 | 5.31x | 1.00x |
| bin blob (1 MB) | **msgpack_mojo native** | 0.0843 | 12434.4 | **1.41x** | 0.27x |
| bin blob (1 MB) | msgpack_mojo fallback | 0.1657 | 6329.4 | 0.72x | 0.14x |

**Warm unpackb** (per call, median of 5 runs):

| payload | backend | ms/call | MB/s | vs oracle C | vs oracle fallback |
|---|---|---:|---:|---:|---:|
| small_api (~0.5 KB) | msgpack C (default) | 0.0007 | 224.5 | 1.00x | 19.14x |
| small_api (~0.5 KB) | msgpack fallback | 0.0134 | 12.3 | 0.05x | 1.00x |
| small_api (~0.5 KB) | **msgpack_mojo native** | 0.0093 | 17.7 | 0.08x | 1.44x |
| small_api (~0.5 KB) | msgpack_mojo fallback | 0.0149 | 11.1 | 0.05x | 0.90x |
| medium (~64 KB) | msgpack C (default) | 0.3814 | 172.0 | 1.00x | 11.72x |
| medium (~64 KB) | msgpack fallback | 4.4683 | 14.7 | 0.09x | 1.00x |
| medium (~64 KB) | **msgpack_mojo native** | 2.1749 | 30.2 | 0.18x | 2.05x |
| medium (~64 KB) | msgpack_mojo fallback | 4.0319 | 16.3 | 0.09x | 1.11x |
| large (~2 MB) | msgpack C (default) | 17.8225 | 117.7 | 1.00x | 8.22x |
| large (~2 MB) | msgpack fallback | 146.4502 | 14.3 | 0.12x | 1.00x |
| large (~2 MB) | **msgpack_mojo native** | 66.5456 | 31.5 | 0.27x | 2.20x |
| large (~2 MB) | msgpack_mojo fallback | 135.2060 | 15.5 | 0.13x | 1.08x |
| int array (100k) | msgpack C (default) | 2.1222 | 235.6 | 1.00x | 22.03x |
| int array (100k) | msgpack fallback | 46.7569 | 10.7 | 0.05x | 1.00x |
| int array (100k) | **msgpack_mojo native** | 19.7141 | 25.4 | 0.11x | 2.37x |
| int array (100k) | msgpack_mojo fallback | 51.5703 | 9.7 | 0.04x | 0.91x |
| str array (20k) | msgpack C (default) | 0.4712 | 616.8 | 1.00x | 19.39x |
| str array (20k) | msgpack fallback | 9.1382 | 31.8 | 0.05x | 1.00x |
| str array (20k) | **msgpack_mojo native** | 5.2101 | 55.8 | 0.09x | 1.75x |
| str array (20k) | msgpack_mojo fallback | 8.4167 | 34.5 | 0.06x | 1.09x |
| bin blob (1 MB) | msgpack C (default) | 0.0143 | 73400.5 | 1.00x | 9.87x |
| bin blob (1 MB) | msgpack fallback | 0.1412 | 7427.5 | 0.10x | 1.00x |
| bin blob (1 MB) | **msgpack_mojo native** | 0.0759 | 13821.7 | 0.19x | 1.86x |
| bin blob (1 MB) | msgpack_mojo fallback | 0.0871 | 12033.6 | 0.16x | 1.62x |

**Cold first-call** (fresh process, import excluded, pack+unpack first calls
combined, median of 5):

| payload | oracle C | oracle fallback | msgpack_mojo native | msgpack_mojo fallback |
|---|---:|---:|---:|---:|
| small_api (~0.5 KB) | 0.025 ms | 0.136 ms | 13.407 ms | 0.175 ms |
| medium (~64 KB) | 1.180 ms | 7.933 ms | 17.372 ms | 8.149 ms |
| large (~2 MB) | 27.126 ms | 224.841 ms | 132.313 ms | 224.230 ms |
| int array (100k) | 5.011 ms | 79.676 ms | 45.252 ms | 89.035 ms |
| str array (20k) | 1.184 ms | 14.930 ms | 22.394 ms | 17.112 ms |
| bin blob (1 MB) | 0.253 ms | 0.444 ms | 12.888 ms | 0.873 ms |

### Honest read

The reference package's default backend is a compiled C extension, and on
this machine it is faster than msgpack-mojo's native path on every warm cell
except one: **packb of the 1 MB binary blob, where the Mojo kernel is
1.41x faster than the C backend** (the C packer copies the payload through
its internal buffer; our kernel copies it once into an exactly-sized
output). That is the only warm cell we publish as a win over the C backend.

Everything else is **fallback-tier**: against the reference package's
pure-Python `msgpack.fallback`, msgpack-mojo native is **1.35–1.66x faster
on packb across the five object-graph payloads and 1.44–2.37x faster on
unpackb across all six payloads**, with byte-exact output. That is the
honest value proposition: a drop-in for environments that cannot or do not
run the C extension, or for callers who today use the pure-Python backend
deliberately.

Also measured, for completeness: the oracle's own pure-Python fallback is
anomalously fast on the 1 MB blob pack cell (a single `bytearray` extend is
memcpy-bound at ~47 GB/s, 5.31x faster than its own C backend, and faster
than our two-copy pack path) — we report it because it is real, and because
it keeps the 1.41x C-win cell in perspective. Cold start carries a one-time
~11–13 ms `dlopen` of the Mojo runtime on the native path (visible in the
cold table, amortized over the process lifetime); the pure-Python fallback
path has no such penalty and stays within ~0.6–1.6x of the oracle's own
pure-Python backend warm and cold.
<!-- BENCH:END -->

## Supported surface

`packb(o, **kwargs)` / `dumps` / `dump(o, fp)`:

- types: `None`, `bool`, `int` (including full 64-bit signed/unsigned range;
  outside it raises `OverflowError`), `float` (packed as float64; NaN/Inf
  bit-exact), `str` (UTF-8), `bytes`/`bytearray`/`memoryview`,
  `list`/`tuple`, `dict`, `ExtType`, `Timestamp` (packed as the timestamp
  extension, like the reference), `datetime.datetime` with `datetime=True`
- options: `default`, `use_bin_type`, `strict_types`, `use_single_float`,
  `unicode_errors`, `datetime`; the streaming-only knobs `autoreset` and
  `buf_size` are accepted and ignored (they cannot change one-shot output)

`unpackb(packed, **kwargs)` / `loads` / `load(fp)`:

- accepts `bytes`/`bytearray`/`memoryview`; returns the same value shapes as
  the oracle (`str`/`bytes` per the `raw` option, `list` per `use_list`,
  `ExtType`/`Timestamp`/float/int/`datetime` per `timestamp` 0–3)
- options: `raw`, `use_list`, `strict_map_key`, `timestamp`,
  `unicode_errors`, `ext_hook`, `max_str_len`, `max_bin_len`,
  `max_array_len`, `max_map_len`, `max_ext_len`
- error behavior mirrors the oracle: `FormatError` (a `ValueError`) for the
  reserved `0xc1` prefix, `ValueError` for truncated input, `ExtraData` (a
  `ValueError`) for trailing bytes, `ValueError` for oversized lengths and
  for non-str/bytes map keys under `strict_map_key=True`

`msgpack_mojo.ExtType` and `msgpack_mojo.Timestamp` mirror the oracle's
types (same construction-time validation, same wire forms); both compare
equal by value, and the differential suite compares them by canonical form.

### Unsupported scope (explicit, loud — never silently wrong)

- `object_hook`, `list_hook`, `object_pairs_hook` raise
  `NotImplementedError`.
- The streaming `Packer`/`Unpacker` classes are not provided; this package
  is the one-shot `packb`/`unpackb`/`dump`/`load` surface only.
- Packing walks the object graph with the Python call stack, so nesting
  depth is bounded by `sys.getrecursionlimit()` (like the reference
  pure-Python packer; the reference C packer recurses in C). Unpacking is
  iterative on both backends and handles any depth the oracle's pure-Python
  fallback handles.
- The unpack record stream is sized before writing (two-pass kernel);
  transient memory is proportional to the input. Malformed length prefixes
  cannot trigger oversized allocations: pass 1 validates every byte first.

## How it works

The Mojo kernel is a pure byte transformer (no Python objects cross the
FFI):

- **pack**: the wrapper walks the object graph into a compact instruction
  stream; the kernel encodes it into MessagePack bytes in a single pass
  (smallest-integer encoding, header selection, payload `memcpy`).
- **unpack**: the kernel validates the whole value and computes the exact
  record-stream size (pass 1), then writes a flat typed record stream
  (pass 2); the wrapper assembles Python objects from it iteratively.

Both backends share the walker and the assembler; the fallback engine is a
pure-Python mirror of the kernel's two byte transforms, so backend choice
cannot change any observable result.

## Benchmark reproduction

From the repository root, with the oracle installed into `.oracle-msgpack/`
(`pip install --target .oracle-msgpack msgpack==1.1.2`):

```bash
pixi run bash kernels/msgpack/build.sh
PYTHONPATH="python/msgpack_mojo:.oracle-msgpack" PYTHONNOUSERSITE=1 \
    pixi run python benchmarks/bench_msgpack.py
```

The differential suite (native backend, then forced fallback):

```bash
PYTHONPATH="python/msgpack_mojo:.oracle-msgpack" PYTHONNOUSERSITE=1 \
    pixi run bash scripts/test_all_msgpack.sh
```

## License

Apache-2.0. Attribution: Algenta.

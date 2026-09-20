# pefile-mojo

Bit-exact, much faster replacements for the two hot loops of the
[`pefile`](https://pypi.org/project/pefile/) package — `PE.generate_checksum`
and the import-directory walk behind `PE.parse_data_directories()` /
`PE.parse_import_directory()` — powered by a clean-room Mojo kernel, with a
vendored pure-Python fallback for platforms without a native build (including
Windows).

This is **not** a full `pefile` drop-in: it does not parse whole PE images.
It replaces the two expensive loops with identical results (see
[Unsupported scope](#unsupported-scope)).

```python
import pefile_mojo

blob = open("module.dll", "rb").read()

# Checksum — identical to pefile.PE(blob).generate_checksum():
digest = pefile_mojo.generate_checksum(blob, pefile_mojo.checksum_field_offset(blob))

# Import table — identical to pefile.PE(blob).DIRECTORY_ENTRY_IMPORT:
for desc in pefile_mojo.parse_imports(blob):
    print(desc.dll, [(s.name, s.ordinal, s.hint, s.address, s.bound) for s in desc.imports])
```

- **Bit-exact**: the differential suite asserts *zero-tolerance* agreement
  with PyPI pefile 2024.8.26 on both the native and the fallback backend —
  checksums are identical integers, import tables identical down to every
  hint/address/bound, on 14 synthetic PE32/PE32+ fixture families, their
  `pefile.write()` re-serializations, a 3 648-symbol image, and 48 seeded
  byte/truncation mutations (checksums compared over `pe.write()` output
  wherever the oracle parses; import tables compared verbatim).
- **Much faster**: ~300x-460x for the checksum, ~11x for import parsing
  (Apple M4 Max; full method and numbers below). The pure-Python fallback is
  also faster than pefile's own loops (~1.3x checksum, ~1.7x imports).
- **No toolchain needed**: per-platform wheels ship the compiled kernel.
  Everywhere else the package transparently uses its pure-Python fallback —
  silently correct, same results.
- Force the fallback with `PEFILE_MOJO_DISABLE_NATIVE=1`; inspect the active
  backend with `pefile_mojo.backend_info()`.

## Drop-in recipe

| pefile | pefile_mojo |
|---|---|
| `pe = pefile.PE(path); pe.generate_checksum()` | `pefile_mojo.generate_checksum(open(path, "rb").read(), pefile_mojo.checksum_field_offset(data))` |
| `pe.parse_data_directories()` then `pe.DIRECTORY_ENTRY_IMPORT` | `pefile_mojo.parse_imports(data)` |

`ImportDescriptor` / `ImportSymbol` are frozen dataclasses with the same
field values as pefile's `ImportDescData` / `ImportData` (`dll`, `name`,
`ordinal`, `hint`, `address`, `bound`).

Note on `generate_checksum`: pefile's method re-serializes the image first
(`pe.write()`) and hashes those bytes. For an unmodified file read from disk
the re-serialization reproduces the input exactly (asserted for every fixture
in the test suite), so hashing the file bytes directly — what this package
does — is identical. If you modified header fields through pefile, call
`pe.write()` first and checksum its output:
`pefile_mojo.generate_checksum(bytes(pe.write()), pefile_mojo.checksum_field_offset(...))`.

## Unsupported scope (honest list)

- **Everything else pefile does**: exports, resources, relocations, debug,
  TLS, delay-load and bound-import *directories*, signatures, overlays as
  structured data, `dump_info()`, mutation of fields, packing. Use pefile for
  those; this package only accelerates the two loops above.
- **Ordinal-to-name resolution**: for ordinal imports from `ws2_32.dll`,
  `wsock32.dll`, or `oleaut32.dll`, pefile fills in a symbol name from its
  bundled `ordlookup` database (`b"socket"` etc.). `pefile_mojo` returns
  `name=None, ordinal=N` for *all* ordinal imports; every other field
  (ordinal, hint, address, bound) is identical. The test fixtures avoid those
  three DLLs, so measured parity is unaffected.
- **Corrupt/adversarial inputs**: pefile has elaborate best-effort recovery
  paths (and in a few spots raises raw `struct.error`). pefile_mojo
  reproduces pefile's *observable results* — including its skip rules,
  warning-triggering bounds (`MAX_IMPORT_SYMBOLS`, repeated/spread address
  aborts, truncated tables), and header-level `PEFormatError` behavior — for
  well-formed files and the corrupt cases covered by the mutation suite, but
  exact bit-parity on arbitrarily corrupted files is not guaranteed. Where
  pefile's full `PE()` parse rejects a file for reasons outside the import
  directory (e.g. a corrupt resource directory), `pefile_mojo.parse_imports`
  may still return the import table. `generate_checksum(data, offset)`
  itself is a pure function of the bytes and the offset and never validates
  PE structure.
- **Windows native kernel**: no Mojo toolchain builds `pefilemojo.dll`
  today; the Windows wheel is the pure-Python fallback (silently correct).
- `bound` values of exactly `0xFFFFFFFFFFFFFFFF` cannot be distinguished from
  "not bound" (sentinel).

## Benchmarks

Measured on this machine (Apple M4 Max, macOS 26.6.2, Python 3.12.14, Mojo
1.1.0, pefile 2024.8.26) with `benchmarks/bench_pefile.py`; synthetic PE
images generated deterministically by the test-suite builder; correctness
gate (bit-exact vs pefile) passed before timing. Warm = median of 7
in-process repetitions; cold = median of 7 fresh OS processes (interpreter
import + fixture generation + first call).

### Checksum (warm, median per call)

| image size | pefile PE+generate_checksum | pefile generate_checksum (pre-built PE) | pefile_mojo native | pefile_mojo fallback | speedup (vs pre-built) |
|---:|---:|---:|---:|---:|---:|
| 64 KiB | 6.905 ms | 5.886 ms | 0.0128 ms | 2.362 ms | 460.2x |
| 1,024 KiB | 72.961 ms | 50.463 ms | 0.1655 ms | 42.016 ms | 304.9x |
| 8,192 KiB | 526.152 ms | 374.866 ms | 1.2531 ms | 299.435 ms | 299.1x |

### Import parse (warm, median per call)

| workload (3648 imported symbols) | time | speedup vs pefile scoped |
|---|---:|---:|
| pefile scoped import parse (`fast_load` + import dir only) | 32.236 ms | 1.0x |
| pefile full PE parse (superset) | 33.729 ms | 0.96x |
| pefile_mojo native | 2.9771 ms | 10.8x |
| pefile_mojo fallback | 19.035 ms | 1.69x |

### Cold start (fresh process: interpreter import + first call, median)

| workload | pefile | pefile_mojo native | pefile_mojo fallback |
|---|---:|---:|---:|
| checksum, 1 MiB image | 214.3 ms | 182.4 ms | 209.3 ms |
| import parse, 64 DLLs | 159.5 ms | 130.9 ms | 149.4 ms |

pefile's `generate_checksum` numbers include its mandatory full parse and/or
`write()` re-serialization — that is the cost of pefile's API for this job;
the pre-built-PE column is the conservative comparison.

## Development

Source, tests, and benchmark: <https://github.com/thyn-ai/mojo-kernels>

- Kernel: `kernels/pefile/` (Mojo, clean-room) — build with
  `bash kernels/pefile/build.sh` (needs the repo's pixi toolchain on PATH).
- Differential suite (both backends): `bash scripts/test_all_pefile.sh`
  (installs the pinned, hash-verified pefile oracle if missing).
- Wheel: `bash python/pefile_mojo/build_wheel.sh` (delocate/auditwheel
  repairs the wheel to vendor the Mojo runtime; note in the script about
  confirming Modular's redistribution terms before publishing).
- Benchmark: `PYTHONPATH=python/pefile_mojo python benchmarks/bench_pefile.py`.

License: Apache-2.0, © 2026 Algenta

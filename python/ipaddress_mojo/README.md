# ipaddress-mojo

Drop-in replacement for the CPython 3.12 standard-library [`ipaddress`](https://docs.python.org/3/library/ipaddress.html)
module — same values, same canonical strings, same exception types and
messages — whose three bulk hot loops (batch parse, batch membership, batch
collapse) run on a clean-room Mojo kernel. Platforms without a native build
(including Windows — tested there in CI: [`windows-fallback`](https://github.com/thyn-ai/mojo-kernels/actions/workflows/windows-fallback.yml))
transparently use the vendored pure-Python/NumPy fallback, which is silently
correct on every input.

The target workload is log analytics and ACL tooling: "is this connection's
address in any of these million networks", asked millions of times per
second.

```python
import ipaddress_mojo as ipm

# Drop-in surface (mirrors stdlib ipaddress on CPython 3.12):
net = ipm.ip_network("10.0.0.0/8")            # strict by default, like stdlib
addr = ipm.ip_address("10.1.2.3")
addr in net                                    # True
list(net.subnets(prefixlen_diff=2))            # [10.0.0.0/10, ...]
net.supernet()                                 # 10.0.0.0/7
list(ipm.collapse_addresses([...]))            # canonical minimal cover
list(ipm.summarize_address_range(a, b))

# Vectorized batch API (the Mojo kernel):
addrs = ipm.parse_many(list_of_strings)        # bulk ip_address()
hits = ipm.contains_many(networks, addrs)      # bool per address: in ANY net
merged = ipm.collapse_batch(networks)          # canonical minimal cover
```

- **Exact parity**: the differential suite asserts *zero-tolerance* agreement
  with the stdlib oracle on both backends — values, strings, ordering,
  hashing, and exception types **and messages** for malformed input — over
  ~1,500 tests: the full parse grammar (leading-zero rejection,
  netmask/hostmask forms, embedded v4-in-v6, `%scope` ids), all operators,
  subnets/supernet/overlaps/collapse/summarize, the private/global address
  registries (boundary-swept), and the batch API against the equivalent
  stdlib loops.
- **Much faster at scale**: ~1,660x for bulk membership at 10k×10k
  addresses×networks (the quadratic stdlib scan extrapolates to ~10 hours
  where the kernel takes ~0.3 s at 1M×1M), 1.6-2.5x for bulk parse, 12-35x
  for bulk collapse of a million networks (Apple M4 Max; full method and
  numbers below). `contains_many` replaces the scan with a
  sort + prefix-max + binary search.
- **No toolchain needed**: per-platform wheels ship the compiled kernel.
  Everywhere else the package transparently uses its fallback — silently
  correct, same results.
- Force the fallback with `IPADDRESS_MOJO_DISABLE_NATIVE=1`; inspect the
  active backend with `ipaddress_mojo.backend_info()`.

## API surface

Drop-in (stdlib-identical on CPython 3.12):

| stdlib | ipaddress_mojo |
|---|---|
| `ipaddress.IPv4Address` / `IPv6Address` | `ipaddress_mojo.IPv4Address` / `IPv6Address` |
| `ipaddress.IPv4Network` / `IPv6Network` | `ipaddress_mojo.IPv4Network` / `IPv6Network` |
| `ipaddress.IPv4Interface` / `IPv6Interface` | `ipaddress_mojo.IPv4Interface` / `IPv6Interface` |
| `ip_address` / `ip_network` / `ip_interface` | same names |
| `collapse_addresses` / `summarize_address_range` | same names |
| `get_mixed_type_key` / `v4_int_to_packed` / `v6_int_to_packed` | same names |
| `AddressValueError` / `NetmaskValueError` | the stdlib classes themselves (re-exported) |

Batch (new; both backends, identical results):

- `parse_many(addresses) -> list`: bulk `ip_address()` over a list of
  strings/ints/bytes. The first malformed item raises exactly the exception
  `ip_address()` would raise for it.
- `contains_many(networks, addresses) -> numpy bool array`: element `j` is
  `any(addresses[j] in n for n in networks)`; cross-version pairs never match
  (mirroring `in`).
- `collapse_batch(networks) -> list`: the canonical minimal cover, identical
  to `collapse_addresses`.

Single-object parsing/validation always runs the pure-Python reference (one
FFI call cannot beat in-process parsing, and one code path cannot diverge);
the kernel accelerates the three batch loops.

## Unsupported scope (honest list)

- **Parity is pinned to CPython 3.12** (the version the repo's toolchain and
  CI run). stdlib behavior details — error-message text, the v6 zone-id
  handling, the private/global registries, the prefix-cache quirk that makes
  a cached prefix `24` accept a later `24.0` — are CPython-version-specific;
  other CPython versions may differ in exactly these corners.
- **Equality with stdlib objects**: `ipaddress_mojo.IPv4Address('1.2.3.4') == ipaddress.IPv4Address('1.2.3.4')`
  is `False` (cross-implementation identity, like any re-implementation); all
  *values* compare equal within each implementation. Hash values match the
  stdlib's formula within a process (v4 hashes are salted by Python, like
  the stdlib's own).
- **`pickle`/`copy` of the drop-in objects** is not covered by the
  differential suite (attribute layouts differ from stdlib internals).
- **Windows native kernel**: no Mojo toolchain builds `ipaddressmojo.dll`
  today; the Windows wheel is the pure-Python fallback (silently correct,
  tested on Windows in CI: [`windows-fallback`](https://github.com/thyn-ai/mojo-kernels/actions/workflows/windows-fallback.yml)).
- The batch API requires NumPy (a hard dependency, like the other
  array-returning kernels in this repo).

## Benchmarks

Measured on this machine (Apple M4 Max, macOS 26.6.2, Python 3.12.14, Mojo
1.1.0) with `benchmarks/bench_ipaddress.py`; deterministic seeded workloads;
correctness gate (exact parity vs stdlib) passed before timing. Warm =
median of 7 in-process repetitions; cold = median of 7 fresh OS processes
(interpreter import + first call).

### Bulk membership (warm, per call)

`hits = contains_many(nets, addrs)` vs the stdlib idiom
`any(a in n for n in nets)` per address (quadratic scan).

| workload | stdlib any() | native | fallback | speedup (native vs stdlib) |
|---|---:|---:|---:|---:|
| v4: 10,000 addresses × 10,000 networks | 3.494 s | 2.100 ms | 2.088 ms | **1,663.7x** |
| v4: 1,000,000 addresses × 1,000,000 networks | ~10 h (extrapolated from the row above; not run) | 328.991 ms | 336.020 ms | ~100,000x (est.) |

### Bulk parse (warm, per call)

`parse_many(strings)` vs a list comprehension over `ipaddress.ip_address`.
Batch construction of the result objects in Python dominates this one; the
kernel-side parse itself is >100x.

| workload | stdlib loop | native | fallback | speedup (native vs stdlib) |
|---|---:|---:|---:|---:|
| v4: 1,000,000 addresses | 1.115 s | 708.017 ms | 1.044 s | 1.6x |
| v6: 1,000,000 addresses | 2.977 s | 1.183 s | 3.353 s | 2.5x |

### Bulk collapse (warm, per call)

`collapse_batch(nets)` vs `ipaddress.collapse_addresses`.

| workload | stdlib | native | fallback | speedup (native vs stdlib) |
|---|---:|---:|---:|---:|
| v4: 1,000,000 networks | 9.336 s | 268.474 ms | 1.165 s | 34.8x |
| v6: 1,000,000 networks | 8.610 s | 707.652 ms | 1.399 s | 12.2x |

### Cold start (fresh process: interpreter import + 1k-item call, median)

| stdlib ipaddress | ipaddress_mojo native | ipaddress_mojo fallback |
|---:|---:|---:|
| 160.676 ms | 333.570 ms | 198.322 ms |

Notes:

- The membership win is algorithmic (sort + prefix-max interval stabbing vs
  quadratic scan) and holds on **both** backends — the NumPy fallback uses
  the same structure as the Mojo kernel, so it is nearly as fast here.
- The parse fallback is the pure-Python reference looped; it is comparable
  to stdlib on v4 and ~13% slower on v6 (correctness path, not the fast
  path).
- Cold-start native pays one dlopen of the kernel plus the Mojo runtime
  (~170 ms on this machine); warm calls amortize it to zero.

## Development

Source, tests, and benchmark: <https://github.com/thyn-ai/mojo-kernels>

- Kernel: `kernels/ipaddress/` (Mojo, clean-room) — build with
  `bash kernels/ipaddress/build.sh` (needs the repo's pixi toolchain on PATH).
- Differential suite (both backends): `bash scripts/test_all_ipaddress.sh`
  (the oracle is the interpreter's own stdlib; nothing to install).
- Wheel: `bash python/ipaddress_mojo/build_wheel.sh` (delocate/auditwheel
  repairs the wheel to vendor the Mojo runtime; note in the script about
  confirming Modular's redistribution terms before publishing).
- Benchmark: `PYTHONPATH=python/ipaddress_mojo python benchmarks/bench_ipaddress.py`.

License: Apache-2.0, © 2026 Algenta

# pymavlink-mojo

**Batch MAVLink v1/v2 log decoding — the offline `.tlog` analysis workload —
powered by a clean-room Mojo kernel**, with a vendored pure-Python fallback
for platforms without a native build (including Windows). One call decodes a
whole telemetry log: headers, per-field payloads for the bundled MAVLink
`common.xml` dialect (all 205 messages), CRC-16/MCRF4XX validation, and
pymavlink's robust resync-on-garbage semantics — message-for-message
identical to [pymavlink](https://github.com/ArduPilot/pymavlink)'s own
parser (the path `mavlogdump` uses), and **3–6x faster** end-to-end on
this machine (see Benchmark below).

```python
import pymavlink_mojo

result = pymavlink_mojo.parse_buffer(open("flight.tlog", "rb").read(),
                                     timestamps=True)
for msg in result.messages:                      # MAVLinkMessage | BadData | UnknownMessage
    if msg.get_type() == "ATTITUDE":
        print(msg._timestamp, msg.roll, msg.pitch, msg.yaw)
```

- **Same results**: the differential suite asserts exact equality (no
  numeric tolerance — every compared value is int/float/str/bytes) against
  the published `pymavlink` package on both the native and fallback
  backends: decoded message tuples (type, header, crc, every field value,
  raw frame bytes, tlog timestamps), BAD_DATA records with pymavlink's
  exact reason strings, UNKNOWN records for out-of-dialect frames, and the
  stream-level error count.
- **Much faster**: the whole log is scanned, checksummed and field-decoded
  in one compiled pass — see the measured numbers below.
- **No toolchain needed**: per-platform wheels ship the compiled kernel.
  Everywhere else the package transparently uses its pure-Python fallback,
  silently and correctly.
- Force the fallback with `PYMAVLINK_MOJO_DISABLE_NATIVE=1`; inspect the
  active backend with `pymavlink_mojo.backend_info()`.

## Install

```
pip install pymavlink-mojo
```

Per-platform wheels (macOS arm64, Linux x86_64) carry the native kernel,
self-contained (the Mojo runtime is vendored into the wheel; nothing to
compile, no absolute rpaths). On any other platform — including Windows —
the same wheel API runs on the vendored pure-Python fallback, silently and
correctly. There is no sdist: a source tarball cannot rebuild the native
library.

## Quickstart

```python
import pymavlink_mojo

# .tlog framing (default for MAVProxy/QGroundControl logs): every message
# is preceded by an 8-byte big-endian microsecond timestamp.
result = pymavlink_mojo.parse_buffer(data, timestamps=True)

# Raw MAVLink stream (serial capture, UDP dump, or the payload of a tlog):
result = pymavlink_mojo.parse_buffer(data)                    # timestamps=False

result.messages        # list[MAVLinkMessage | BadData | UnknownMessage]
result.consumed        # bytes fully accounted for (a trailing partial
                       # frame/timestamp unit is left for the next batch)
result.error_count     # BadData records (pymavlink's total_receive_errors)
```

Message objects mirror pymavlink's surface: `msg.get_type()`,
`msg.get_msgId()`, `msg.get_header()`, `msg.get_srcSystem()`,
`msg.get_srcComponent()`, `msg.get_seq()`, `msg.get_crc()`,
`msg.get_msgbuf()`, `msg.get_payload()`, `msg.get_fieldnames()`,
`msg.get_signed()`, `msg.to_dict()` — and field values as plain attributes
(`msg.roll`, `msg.param_id`, ...), with `msg._timestamp` in tlog mode.
`BadData` carries `.data` and pymavlink's exact `.reason` string;
`UnknownMessage` carries the wire `wire_msgid` and the raw frame as `.data`.

A runnable version is `examples/quickstart.py` (self-contained, builds a
synthetic tlog by hand):

```
python examples/quickstart.py
```

## Benchmark

Measured with `benchmarks/bench_pymavlink.py` in this repository (run it
with `PYTHONPATH=python/pymavlink_mojo:<oracle-dir> pixi run python
benchmarks/bench_pymavlink.py`; the pinned oracle must be importable —
`scripts/test_all_pymavlink.sh` provisions it into
`/tmp/pymavlink-mojo-oracle`).
Workload = whole-log batch decode into message objects (headers + payload
fields + CRC + resync), which is what offline tlog analysis tools do. The
oracle runs its real parse loop (`mavutil.mavlogfile.recv_msg` on the same
file, MAVLINK20=1); the raw-stream cell uses pymavlink's own
`MAVLink.parse_buffer` batch API. Corpora are generated with pymavlink's
own message writer from fixed seeds. **Cold** = first decode in the
process (kernel dlopen + ABI handshake + first scan), single measurement;
**warm** = median of 5 runs (3 for the 500k cells). Correctness
(tuple-for-tuple equality plus error count) is asserted before every
timing run. Environment: **Apple M4 Max, macOS 26.6.2 arm64, Python
3.12.5, numpy 2.2.6, Mojo 1.1.0, pymavlink 2.4.47**, 2026-09-20.

| workload | size | cold pymavlink | cold pymavlink-mojo | cold speedup | warm pymavlink | warm pymavlink-mojo | warm speedup |
|---|---:|---:|---:|---:|---:|---:|---:|
| tlog 100k | 5.1 MB | 1635.4 ms | 342.5 ms | 4.8x | 1752.6 ms | 281.0 ms | 6.2x |
| tlog 500k | 25.4 MB | 10002.6 ms | 2299.1 ms | 4.4x | 8740.4 ms | 1703.9 ms | 5.1x |
| tlog 100k noisy | 5.1 MB | 1576.5 ms | 336.1 ms | 4.7x | 1668.8 ms | 335.6 ms | 5.0x |
| raw 500k | 21.4 MB | 6561.8 ms | 2043.7 ms | 3.2x | 6583.4 ms | 1405.5 ms | 4.7x |
| tlog 100k (fallback) | 5.1 MB | 1950.8 ms | 2445.0 ms | 0.8x | 1639.3 ms | 1775.8 ms | 0.9x |

Warm pymavlink-mojo throughput: 14.9–18.1 MB/s (~293k–356k messages/s)
end-to-end on the native backend, including Python message-object
materialization; the vendored pure-Python fallback runs at 2.9 MB/s
(~56k messages/s), the same ballpark as pymavlink itself. The kernel's
scan+CRC+field-decode pass alone runs at ~270 MB/s, so the remaining
per-message cost is Python object construction, which is also where the
remaining speedup over pymavlink comes from — pymavlink's reader runs the
same state machine per message in pure Python (per-chunk `recv` calls,
per-byte CRC-16/MCRF4XX, per-field `struct` unpacking, plus `mavlogfile`'s
`post_message` bookkeeping), while the Mojo kernel does one compiled
single pass over the buffer into flat arenas.

## How it works

```
pip install pymavlink-mojo
        │
        ▼
pymavlink_mojo (thin Python wrapper)
        │  passes the bundled common.xml dialect tables (generated from the
        │  protocol spec; see kernels/pymavlink/gen_tables_pymavlink.py)
        ▼
libpymavmojo.dylib / .so           (Mojo kernel, C ABI v1)
        │  pymavmojo_parse: whole buffer per call — scan, CRC-16/MCRF4XX,
        │  v1+v2 header/payload decode, robust resync, tlog framing
        ▼
flat arenas -> MAVLinkMessage / BadData / UnknownMessage objects
```

- **Batch-shaped C ABI**: one call decodes the whole buffer; FFI overhead
  is per-call, not per-message. Results come back as fixed-stride records
  plus typed spill arenas (int64 / float64 / bytes) that the wrapper slices
  into message objects.
- **Identical semantics, both backends**: the vendored pure-Python fallback
  (`pymavlink_mojo/_reference.py`) implements the same state machine driven
  by the same generated dialect tables and producing the same arena layout,
  so the two backends cannot disagree — the differential suite asserts it.
- **pymavlink-exact resync**: garbage bytes at a frame boundary become
  one-byte `BadData` records ("Bad prefix"); bad-CRC and bad-incompat-flags
  frames become whole-frame `BadData` records with pymavlink's exact reason
  strings; out-of-dialect message ids become `UnknownMessage` records with
  no CRC check (pymavlink's `MAVLink_unknown` behaviour). The tlog reader
  reproduces `mavutil.mavlogfile`'s recv machinery byte-for-byte —
  including its file/parse-buffer aliasing on mixed v1/v2 streams and the
  `scan_timestamp` three-day byte-wise rescan — so corrupt and
  mixed-version logs decode identically.
- **ABI handshake**: the wrapper checks `pymavmojo_abi_version()` before
  parsing; a mismatch falls back cleanly.

## Fallback semantics

There is no Windows Mojo toolchain today, and a shared library can always
go missing — so the wrapper **falls back to a vendored pure-Python
parser**:

- Resolution order: `$PYMAVLINK_MOJO_NATIVE_LIB` → the library bundled in
  the wheel → the repo development build output.
- `PYMAVLINK_MOJO_DISABLE_NATIVE=1` forces the fallback (the test suite
  runs this way as its second pass).
- Inspect what's active: `pymavlink_mojo.backend_info()` and
  `pymavlink_mojo.native_available()`.
- Wheels are **per-platform** (`py3-none-macosx_*_arm64`,
  `py3-none-manylinux_*_x86_64`) and **wheel-only**. Each wheel is
  **self-contained**: `delocate` (macOS) / `auditwheel repair` (Linux)
  vendor the Mojo runtime libraries and rewrite load paths to be
  wheel-relative. (Redistribution terms for Modular's runtime binaries
  should be confirmed with Modular before any public release.) A pure
  `py3-none-any` fallback wheel can be produced with
  `PYMAVLINK_MOJO_ALLOW_PURE_WHEEL=1` (e.g. for Windows).

## Differential tests

```
pixi run bash scripts/test_all_pymavlink.sh   # builds nothing; run
                       # `pixi run bash kernels/pymavlink/build.sh` first.
                       # The script runs the suite twice: once native, once
                       # with PYMAVLINK_MOJO_DISABLE_NATIVE=1.
```

The suite (`tests/test_pymavlink_*.py`) compares pymavlink-mojo against the
published pymavlink package (pinned `pymavlink==2.4.47`, self-provisioned
by the test script) with **exact equality** on: every `common.xml` message
(all 205) across mixed v1/v2 streams, signed frames, injected garbage of
every kind (bad prefixes, bad CRCs, bad incompat flags, unknown ids,
truncated tails), clean and corrupt tlogs (including the
`scan_timestamp` resync and mixed-version aliasing cases), and a tlog
written by pymavlink's own `mavlogfile` writer. Logs are generated with
pymavlink's own message packer from fixed seeds — the suite is
bit-reproducible. 33 tests pass per backend run (one backend-agreement test
runs only in the native pass).

## Scope and limitations

- **Dialect**: the bundled tables cover the complete MAVLink `common.xml`
  message set (all 205 messages, MAVLink 2.0-era definitions including
  extension fields). Frames for message ids outside `common.xml` (e.g.
  ArduPilot's `ardupilotmega.xml`-only messages) are returned as
  `UnknownMessage` records — with no CRC validation, exactly like pymavlink
  running with the `common` dialect. Other dialects are out of scope.
- **Container formats**: the input is the MAVLink wire stream, optionally
  with `.tlog` timestamp framing. ArduPilot DataFlash `.BIN`/`.log` files
  are a *different* container format (pymavlink's `DFReader`, not its
  MAVLink parser) and are out of scope; CSV/text logs likewise.
- **Robust parsing only**: the decoder mirrors pymavlink's
  `robust_parsing=True` path (the log-analysis default): malformed bytes
  yield `BadData` records instead of raising `MAVError`.
- **No signature verification**: signed MAVLink v2 frames are decoded
  (the 13-byte signature block is parsed and skipped) but never verified —
  matching pymavlink parsing without a signing key, so
  `msg.get_signed()` is always False (`msg.get_signature_present()`
  reports the on-wire fact).
- **Timestamp source**: tlog timestamps come from the file, attached with
  pymavlink's exact rules (`scan_timestamp` rescan included). Raw-stream
  parses carry `_timestamp = None` (pymavlink's `MAVLink.parse_buffer` has
  no timestamp concept either).
- pymavlink is *not* a dependency: it is used only as the test/benchmark
  oracle. numpy is the only runtime dependency.
- Single-threaded kernel (determinism first).

## Citing

If you use pymavlink-mojo in academic work, please cite both pymavlink and
this package:

> pymavlink-mojo: batch MAVLink v1/v2 log decoding powered by a Mojo
> kernel. thyn-ai, 2026. https://github.com/thyn-ai/mojo-kernels
> (citation file with DOI forthcoming)

pymavlink itself: A. Tridgell et al., *pymavlink: MAVLink protocol
implementations*, https://github.com/ArduPilot/pymavlink. MAVLink:
L. Meier et al., *MAVLink: Micro Air Vehicle communication protocol*,
https://mavlink.io.

## License

Apache-2.0, © 2026 Algenta. The kernel and wrapper are clean-room
implementations of the published MAVLink serialization specification
(frame layout, CRC-16/MCRF4XX with per-message crc extra, tlog framing).
The bundled dialect tables are protocol facts extracted from the
`common.xml` specification by `kernels/pymavlink/gen_tables_pymavlink.py`.
pymavlink is used only as a test/benchmark reference, never as a runtime
dependency.

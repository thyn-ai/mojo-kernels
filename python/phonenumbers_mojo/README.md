# phonenumbers-mojo

A drop-in faster replacement for the [`phonenumbers`](https://pypi.org/project/phonenumbers/)
package (Python port of Google's libphonenumber), powered by a Mojo kernel
with a pure-Python fallback everywhere else.

```python
import phonenumbers_mojo
from phonenumbers_mojo import PhoneNumberFormat

num = phonenumbers_mojo.parse("+1 212-555-1234")
num.country_code      # 1
num.national_number   # 2125551234
phonenumbers_mojo.is_valid_number(num)                       # True
phonenumbers_mojo.is_possible_number(num)                    # True
phonenumbers_mojo.format_number(num, PhoneNumberFormat.E164) # '+12125551234'
phonenumbers_mojo.format_number(num, PhoneNumberFormat.INTERNATIONAL)  # '+1 212-555-1234'
phonenumbers_mojo.format_number(num, PhoneNumberFormat.NATIONAL)       # '(212) 555-1234'

# batch API for pipelines:
phonenumbers_mojo.validate_column(["+1 212-555-1234", "nope", "+44 20 7946 0018"])
# [True, False, True]
```

The supported surface is `parse`, `is_valid_number`, `is_possible_number`,
`format_number` (E164 / INTERNATIONAL / NATIONAL), the `PhoneNumber`,
`PhoneNumberFormat`, `NumberParseException` and `CountryCodeSource` classes
(with the same constant values), plus the `validate_column(numbers,
region=None) -> list[bool]` batch API. Results are **identical** to the
`phonenumbers` 9.0.39 oracle: same parsed fields, same booleans, same
formatted strings, same `NumberParseException.error_type` and message.

## How it works

All pattern matching runs on a **table-driven Thompson NFA (Pike VM)** over
digit-mask bytecode, compiled offline from the libphonenumber 9.0.39
metadata (`kernels/phonenumbers/gen_tables.py`, clean-room). Validation
descs, possible-length masks, national-prefix rules and format templates for
all 254 regions ship as one compact binary blob inside the wheel
(`phonenumbers_mojo/_data.py`, ~130 KB compressed). Both backends — the
native Mojo kernel and the pure-Python fallback engine — parse this exact
blob and implement the same pinned pipeline (extract → viability →
extension strip → calling-code extraction → national-prefix strip →
length gates), so results agree bit-for-bit on either backend.

- **Native backend** (macOS arm64, Linux x86_64): the Mojo kernel in
  `kernels/phonenumbers/`; loaded via ctypes with an ABI handshake.
- **Fallback backend** (everywhere else, incl. Windows): the pure-Python
  engine `phonenumbers_mojo/_fallback.py`. Set
  `PHONENUMBERS_MOJO_DISABLE_NATIVE=1` to force it.
- Non-ASCII inputs (e.g. full-width digits `＋４４…`, Cyrillic extension
  markers) are routed per-input to the fallback engine on native installs;
  results stay oracle-identical.

The metadata XML is vendored from the libphonenumber project (Apache-2.0
data, © The Libphonenumber Authors; see
`kernels/phonenumbers/data/NOTICE`). No libphonenumber code is used or
adapted.

## Benchmarks

Measured on this machine (Apple M4 Max, macOS 26.6.2 arm64, Python 3.12.5,
Mojo 1.1.0, 2026-09-20), oracle = PyPI `phonenumbers` 9.0.39, corpus = 742
golden + generated numbers across 20 regions, median of 5 batches
(`benchmarks/bench_phonenumbers.py`, correctness-gated: all 742 results
identical to the oracle on both backends):

Warm steady-state:

| workload | PyPI phonenumbers | phonenumbers-mojo (native) | phonenumbers-mojo (fallback) | native speedup |
|---|---:|---:|---:|---:|
| `parse()`, per call | 7.29 µs | 5.17 µs | 47.55 µs | 1.4× |
| `is_valid_number()`, per call | 2.55 µs | 2.13 µs | 15.55 µs | 1.2× |
| `format_number(NATIONAL)`, per call | 3.07 µs | 3.63 µs | 19.64 µs | 0.8× |
| `validate_column` batch, per number | 12.08 µs | 3.61 µs | 79.34 µs | **3.3×** |

Cold first-call (fresh process: import + metadata + first parse, median of 5):

| package | import + first parse |
|---|---:|
| PyPI phonenumbers | 43.40 ms |
| phonenumbers-mojo (native) | 48.09 ms |
| phonenumbers-mojo (fallback) | 35.63 ms |

Honest summary: single-call `parse`/`is_valid_number` are moderately faster
(1.2–1.4×), `format_number` is at parity, and the batch API — the real
pipeline shape — is **3.3× faster** while remaining bit-identical. Cold
start is on par with the oracle (tens of ms either way, dominated by
interpreter/package import; repeated runs vary ±30%). The fallback engine
trades speed for portability (~0.15× of the oracle, correct everywhere).

## Differential testing

`tests/test_phonenumbers_differential.py` runs against the PyPI oracle on
both backends (native, and `PHONENUMBERS_MOJO_DISABLE_NATIVE=1`):

- **Golden corpus**: every `<exampleNumber>` in the libphonenumber 9.0.39
  metadata (all 254 regions, all number types), parsed in national and `+cc`
  form — identical parse fields, validity, possibility and all three formats.
- **Generated corpus**: 1 200 seeded numbers for 20 top regions
  (US, GB, DE, FR, IN, CN, JP, BR, RU, AU, CA, IT, ES, MX, KR, NL, SE, CH,
  PL, TR) with punctuation/extension/IDD variants — identical results and
  identical `NumberParseException` error types and messages.
- **Edge battery**: extension markers, national-prefix transforms (AR/BR/AG),
  RFC3966 (`tel:`) input, vanity numbers, too-long/too-short/invalid-cc
  errors, non-ASCII inputs.

Current status: **4 052 tests pass on the native backend; 4 050 pass (+2
native-only loader skips) on the forced fallback**, 0 mismatches — plus an
extended 37 050-case all-region fuzz (every region, marker variants,
Unicode-digit inputs) with 0 mismatches on each backend. Parity is
exact (no tolerance): identical parsed fields, booleans, formatted strings
and exception `(error_type, message)` everywhere.

## Unsupported scope (honest)

- **API surface**: only the four functions + `validate_column` above. Not
  included: `format_number` RFC3966 output, `PhoneNumberMatcher`,
  `AsYouTypeFormatter`, short-number APIs, geocoding/carrier/timezone
  lookups, `keep_raw_input` (and therefore `raw_input`,
  `country_code_source` is always `0` exactly as the oracle reports for
  parsed numbers), and `PhoneNumber` fields outside the core set.
- **RFC3966**: `tel:` URIs with `;phone-context=` (global-number-digit form)
  and `;isub=` are supported; `phone-context` domain values are parsed per
  the oracle. Other RFC3966 corners (e.g. visual separators inside
  parameters) follow the oracle's behavior on the covered corpus.
- **Unicode digits beyond the common blocks**: the fallback engine folds all
  Unicode decimal digits (via `unicodedata`); the native kernel handles
  ASCII input and defers other scripts to it (silently correct, slower).
- Extensions are capped at 20 digits by the oracle's own grammar; anything
  longer fails identically on both backends.

## Install / build

```bash
pip install dist/phonenumbers_mojo-*.whl          # platform wheel (native kernel)
# or pure-Python anywhere: the same wheel falls back automatically on
# unsupported platforms
```

From source (requires the repo pixi environment with the Mojo toolchain):

```bash
bash kernels/phonenumbers/build.sh                # compile the Mojo kernel
bash python/phonenumbers_mojo/build_wheel.sh      # build + delocate the wheel
scripts/test_all_phonenumbers.sh                  # differential suite, both backends
```

Set `PHONENUMBERS_MOJO_NATIVE_LIB=/path/to/libphonenumbersmojo.{dylib,so}`
to override the kernel resolution during development.

## License and attribution

Apache-2.0. Package and kernel by **Algenta**. Phone metadata:
© The Libphonenumber Authors, Apache-2.0 (`kernels/phonenumbers/data/NOTICE`).

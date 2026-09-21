"""Drop-in phone number parsing/validation/formatting, API-compatible with
the `phonenumbers` package (9.0.39) for the supported surface:

    parse(number, region=None) -> PhoneNumber
    is_valid_number(numobj) -> bool
    is_possible_number(numobj) -> bool
    format_number(numobj, PhoneNumberFormat.{E164,INTERNATIONAL,NATIONAL}) -> str

plus a batch API `validate_column(numbers, region=None) -> list[bool]`.

Execution runs on the native Mojo kernel when its shared library is
available (macOS arm64 / Linux x86_64 wheels) and transparently falls back
to the pure-Python engine in `phonenumbers_mojo._fallback` otherwise. Both
backends run the same pipeline over the same compiled metadata blob; the
differential suite asserts identical results against the PyPI oracle on
both paths.
"""

from __future__ import annotations

from phonenumbers_mojo import _fallback as fb

__all__ = [
    "PhoneNumber",
    "PhoneNumberFormat",
    "NumberParseException",
    "CountryCodeSource",
    "parse",
    "is_valid_number",
    "is_possible_number",
    "format_number",
    "validate_column",
    "backend",
    "backend_info",
    "native_available",
]


class PhoneNumberFormat:
    """Phone number formats (values match the oracle's enum)."""

    E164 = 0
    INTERNATIONAL = 1
    NATIONAL = 2
    RFC3966 = 3  # constant exposed for parity; RFC3966 formatting is unsupported


class CountryCodeSource:
    """Country code source enum (values match the oracle)."""

    FROM_NUMBER_WITH_PLUS_SIGN = 1
    FROM_NUMBER_WITH_IDD = 5
    FROM_NUMBER_WITHOUT_PLUS_SIGN = 10
    FROM_DEFAULT_COUNTRY = 20


class NumberParseException(Exception):
    """Mirror of phonenumbers.NumberParseException.

    `error_type` is one of the class constants; str() renders as
    "(error_type) message", exactly like the oracle.
    """

    INVALID_COUNTRY_CODE = 0
    NOT_A_NUMBER = 1
    TOO_SHORT_AFTER_IDD = 2
    TOO_SHORT_NSN = 3
    TOO_LONG = 4

    def __init__(self, error_type: int, message: str) -> None:
        super().__init__(message)
        self.error_type = error_type
        self._message = message

    @property
    def msg(self) -> str:
        return self._message

    def __str__(self) -> str:
        return f"({self.error_type}) {self._message}"


class PhoneNumber:
    """Mirror of phonenumbers.PhoneNumber for the supported fields."""

    __slots__ = (
        "country_code",
        "national_number",
        "extension",
        "italian_leading_zero",
        "number_of_leading_zeros",
        "country_code_source",
        "preferred_domestic_carrier_code",
        "_nsn",
    )

    def __init__(
        self,
        country_code=None,
        national_number=None,
        extension=None,
        italian_leading_zero=None,
        number_of_leading_zeros=None,
        country_code_source=0,
        preferred_domestic_carrier_code=None,
        _nsn=None,
    ):
        self.country_code = country_code
        self.national_number = national_number
        self.extension = extension
        self.italian_leading_zero = italian_leading_zero
        self.number_of_leading_zeros = number_of_leading_zeros
        self.country_code_source = country_code_source
        self.preferred_domestic_carrier_code = preferred_domestic_carrier_code
        self._nsn = _nsn

    def __repr__(self) -> str:
        return (
            f"PhoneNumber(country_code={self.country_code}, "
            f"national_number={self.national_number}, "
            f"extension={self.extension!r}, "
            f"italian_leading_zero={self.italian_leading_zero}, "
            f"number_of_leading_zeros={self.number_of_leading_zeros}, "
            f"country_code_source={self.country_code_source}, "
            f"preferred_domestic_carrier_code={self.preferred_domestic_carrier_code!r})"
        )

    def __eq__(self, other):
        if not isinstance(other, PhoneNumber):
            return NotImplemented
        return (
            self.country_code == other.country_code
            and self.national_number == other.national_number
            and self.extension == other.extension
            and self.italian_leading_zero == other.italian_leading_zero
            and self.number_of_leading_zeros == other.number_of_leading_zeros
            and self.country_code_source == other.country_code_source
            and self.preferred_domestic_carrier_code
            == other.preferred_domestic_carrier_code
        )

    def __hash__(self):
        return hash(
            (
                self.country_code,
                self.national_number,
                self.extension,
                self.italian_leading_zero,
                self.number_of_leading_zeros,
            )
        )


# ------------------------------------------------------------------ backends

_TABLES: fb.Tables | None = None
_STORE = None  # native store handle, when the kernel is available
_BACKEND: str | None = None


def _tables() -> fb.Tables:
    global _TABLES
    if _TABLES is None:
        _TABLES = fb.Tables.from_default_blob()
    return _TABLES


def _store():
    """Lazily resolve the native kernel; None when unavailable.

    PHONENUMBERS_MOJO_DISABLE_NATIVE=1 is honored at call time (the
    differential suite and users can force the fallback mid-process)."""
    global _STORE, _BACKEND
    import os

    if os.environ.get("PHONENUMBERS_MOJO_DISABLE_NATIVE") == "1":
        if _BACKEND != "fallback":
            _STORE, _BACKEND = None, "fallback"
        return None
    if _BACKEND == "native":
        return _STORE
    if _BACKEND == "fallback" and _STORE is None:
        # Native was unavailable before; do not retry on every call.
        return None
    from phonenumbers_mojo import _native

    try:
        _STORE = _native.NativeStore(_tables_blob())
        _BACKEND = "native"
    except _native.NativeUnavailable:
        _STORE = None
        _BACKEND = "fallback"
    return _STORE


def _tables_blob() -> bytes:
    from phonenumbers_mojo._data import load_blob

    return load_blob()


def backend() -> str:
    """Which backend serves this process: "native" or "fallback"."""
    _store()
    return _BACKEND or "fallback"


def backend_info() -> dict:
    from phonenumbers_mojo import _native

    info = _native.backend_info()
    info["backend"] = backend()
    return info


def native_available() -> bool:
    from phonenumbers_mojo import _native

    return _native.native_available()


def _region_index(region: str | None) -> int:
    if region is None:
        return -1
    idx = _tables().region_by_code.get(region)
    if idx is None or _tables().regions[idx].code == "001":
        return -1
    return idx


def _phone_number_from_parsed(p: fb.Parsed) -> PhoneNumber:
    # Oracle's _set_italian_leading_zeros_for_phone_number: ilz when len>1 and
    # leading '0'; the count covers all but the last zero and is only stored
    # when it exceeds 1 (else None).
    ilz = None
    nlz = None
    if len(p.nsn) > 1 and p.nsn[0] == "0":
        ilz = True
        count = 1
        while count < len(p.nsn) - 1 and p.nsn[count] == "0":
            count += 1
        if count != 1:
            nlz = count
    return PhoneNumber(
        country_code=p.cc,
        national_number=int(p.nsn),
        extension=p.ext,
        italian_leading_zero=ilz,
        number_of_leading_zeros=nlz,
        country_code_source=0,  # the oracle reports 0 for parsed numbers
        _nsn=p.nsn,
    )


def parse(number: str, region: str | None = None) -> PhoneNumber:
    """Parse a phone number string. Raises NumberParseException on failure,
    with the same error_type and message as the oracle."""
    # Non-ASCII input is outside the kernel's scope by design; the
    # pure-Python engine (full Unicode support) handles it transparently.
    store = _store() if (isinstance(number, str) and number.isascii()) else None
    if store is not None:
        from phonenumbers_mojo import _native

        try:
            p = store.parse(number, _region_index(region))
        except _native.ParseFailure as exc:
            raise NumberParseException(exc.error_type, exc.message) from None
        if p is None:
            p = _fallback_parse(number, region)
        return _phone_number_from_parsed(p)
    return _phone_number_from_parsed(_fallback_parse(number, region))


def _fallback_parse(number: str, region: str | None) -> fb.Parsed:
    try:
        return fb.parse(_tables(), number, region)
    except fb.ParseError as exc:
        raise NumberParseException(exc.error_type, exc.message) from None


def _nsn_of(numobj: PhoneNumber) -> str:
    nsn = getattr(numobj, "_nsn", None)
    if nsn is not None:
        return nsn
    # Reconstruct from the public fields (foreign PhoneNumber objects and
    # hand-built instances): leading zeros from the flags.
    digits = str(numobj.national_number)
    zeros = 0
    if numobj.italian_leading_zero:
        zeros = numobj.number_of_leading_zeros or 1
    return "0" * zeros + digits


def is_valid_number(numobj: PhoneNumber) -> bool:
    store = _store()
    nsn = _nsn_of(numobj)
    if store is not None:
        return bool(store.validate(numobj.country_code, nsn)[0])
    return fb.is_valid(_tables(), numobj.country_code, nsn)


def is_possible_number(numobj: PhoneNumber) -> bool:
    store = _store()
    nsn = _nsn_of(numobj)
    if store is not None:
        return bool(store.validate(numobj.country_code, nsn)[1])
    return fb.is_possible(_tables(), numobj.country_code, nsn)


def format_number(numobj: PhoneNumber, num_format: int) -> str:
    store = _store()
    nsn = _nsn_of(numobj)
    if store is not None:
        return store.format(numobj.country_code, nsn, numobj.extension, num_format)
    return fb.format_number(_tables(), numobj.country_code, nsn, numobj.extension, num_format)


def validate_column(numbers, region: str | None = None) -> list[bool]:
    """Batch API: validate a column of raw phone-number strings.

    Returns one bool per input: True when the string parses and is a valid
    number. On the native backend the whole column crosses the FFI once;
    non-ASCII rows are served by the pure-Python engine.
    """
    values = list(numbers)
    store = _store()
    if store is not None:
        rows = store.validate_column(values, _region_index(region))
        if all(r is not None for r in rows):
            return rows  # type: ignore[return-value]
        # Fill kernel-unserved rows (non-ASCII) with the fallback engine.
        tables = _tables()
        out = []
        for v, r in zip(values, rows):
            if r is not None:
                out.append(r)
                continue
            try:
                p = fb.parse(tables, v, region)
                out.append(fb.is_valid(tables, p.cc, p.nsn))
            except fb.ParseError:
                out.append(False)
        return out
    tables = _tables()
    out = []
    for v in values:
        try:
            p = fb.parse(tables, v, region)
            out.append(fb.is_valid(tables, p.cc, p.nsn))
        except fb.ParseError:
            out.append(False)
    return out

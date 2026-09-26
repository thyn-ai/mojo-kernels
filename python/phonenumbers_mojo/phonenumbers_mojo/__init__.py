"""phonenumbers-mojo: a drop-in faster replacement for the `phonenumbers` package.

Same call shapes, same results — powered by a Mojo kernel where the platform
supports it (macOS arm64, Linux x86_64), with a pure-Python fallback
everywhere else (including Windows).

    import phonenumbers_mojo

    num = phonenumbers_mojo.parse("+1 212-555-1234")
    phonenumbers_mojo.is_valid_number(num)                      # True
    phonenumbers_mojo.format_number(num, phonenumbers_mojo.PhoneNumberFormat.INTERNATIONAL)
    # '+1 212-555-1234'

    phonenumbers_mojo.validate_column(["+1 212-555-1234", "nope"])  # [True, False]

Set PHONENUMBERS_MOJO_DISABLE_NATIVE=1 to force the pure-Python fallback.
"""

from phonenumbers_mojo.core import (
    CountryCodeSource,
    NumberParseException,
    PhoneNumber,
    PhoneNumberFormat,
    backend,
    backend_info,
    format_number,
    is_possible_number,
    is_valid_number,
    native_available,
    parse,
    validate_column,
)

__version__ = "0.1.5"  # x-release-please-version
__all__ = [
    "CountryCodeSource",
    "NumberParseException",
    "PhoneNumber",
    "PhoneNumberFormat",
    "backend",
    "backend_info",
    "format_number",
    "is_possible_number",
    "is_valid_number",
    "native_available",
    "parse",
    "validate_column",
    "__version__",
]

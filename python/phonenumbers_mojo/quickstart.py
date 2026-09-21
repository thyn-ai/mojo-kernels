"""Quickstart: parse, validate, format, and batch-validate with phonenumbers_mojo.

Run against an installed wheel (or PYTHONPATH=python/phonenumbers_mojo).
Prints a checksum line so CI can compare native and fallback output exactly.
"""

from __future__ import annotations

import phonenumbers_mojo
from phonenumbers_mojo import PhoneNumberFormat


def main() -> None:
    info = phonenumbers_mojo.backend_info()
    print(f"backend: {phonenumbers_mojo.backend()}")
    print(f"native available: {info['native_available']}")

    numbers = [
        "+1 212-555-1234",
        "011 44 20 7946 0018",      # IDD from the US
        "+39 02 12345678",          # Italian leading zero
        "+54 9 11 2343 0000",       # AR mobile with 9
        "1-800-FLOWERS",            # vanity
        "0211 123456",              # DE national
        "not a number",
    ]
    checksum = 0
    for raw in numbers:
        try:
            num = phonenumbers_mojo.parse(raw, "US" if not raw.startswith("+") else None)
        except phonenumbers_mojo.NumberParseException as e:
            print(f"{raw!r:42} -> ERROR {e}")
            checksum += 1
            continue
        valid = phonenumbers_mojo.is_valid_number(num)
        possible = phonenumbers_mojo.is_possible_number(num)
        e164 = phonenumbers_mojo.format_number(num, PhoneNumberFormat.E164)
        intl = phonenumbers_mojo.format_number(num, PhoneNumberFormat.INTERNATIONAL)
        nat = phonenumbers_mojo.format_number(num, PhoneNumberFormat.NATIONAL)
        checksum += int(num.national_number) % 1000003
        print(
            f"{raw!r:42} -> +{num.country_code} {num.national_number}"
            f" valid={valid} possible={possible}\n"
            f"{'':44}   E164={e164} INTL={intl} NAT={nat}"
        )

    column = ["+1 212-555-1234", "212-555-1234", "+44 20 7946 0018", "123", "+81 3 1234 5678"]
    mask = phonenumbers_mojo.validate_column(column, "US")
    checksum += sum(mask)
    print(f"validate_column({column!r}, 'US') -> {mask}")
    print(f"validation checksum: {checksum}")


if __name__ == "__main__":
    main()

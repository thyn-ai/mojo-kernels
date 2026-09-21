"""Differential tests: phonenumbers_mojo must match PyPI phonenumbers exactly.

The oracle is the published PyPI package (phonenumbers==9.0.39). The corpus
combines:

  - the golden example numbers from the vendored libphonenumber 9.0.39
    metadata XML (every region, every number type with an <exampleNumber>),
  - deterministically generated numbers for 20 top regions (seeded RNG),
  - an edge battery covering extension markers, IDD dialing, RFC3966 input,
    national prefix transforms, and invalid inputs (error-type parity),
  - non-ASCII inputs (the wrapper's fallback escape hatch on native runs).

The suite runs twice via scripts/test_all_phonenumbers.sh: once against the
native Mojo kernel and once with PHONENUMBERS_MOJO_DISABLE_NATIVE=1 (forced
pure-Python fallback). On both backends every result must be identical to
the oracle: parsed fields, validity, possibility, all three formats, and
NumberParseException error_type + message.
"""

from __future__ import annotations

import random
import xml.etree.ElementTree as ET
from pathlib import Path

import phonenumbers as oracle
import phonenumbers_mojo as mine
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
METADATA_XML = REPO_ROOT / "kernels" / "phonenumbers" / "data" / "PhoneNumberMetadata.xml"

TOP_REGIONS = [
    "US", "GB", "DE", "FR", "IN", "CN", "JP", "BR", "RU", "AU",
    "CA", "IT", "ES", "MX", "KR", "NL", "SE", "CH", "PL", "TR",
]
REGION_CC = {
    "US": "1", "GB": "44", "DE": "49", "FR": "33", "IN": "91", "CN": "86",
    "JP": "81", "BR": "55", "RU": "7", "AU": "61", "CA": "1", "IT": "39",
    "ES": "34", "MX": "52", "KR": "82", "NL": "31", "SE": "46", "CH": "41",
    "PL": "48", "TR": "90",
}
FORMATS = (
    oracle.PhoneNumberFormat.E164,
    oracle.PhoneNumberFormat.INTERNATIONAL,
    oracle.PhoneNumberFormat.NATIONAL,
)


def _oracle_full(text, region):
    try:
        o = oracle.parse(text, region)
    except oracle.NumberParseException as e:
        return ("ERR", e.error_type, str(e))
    return (
        "OK",
        (o.country_code, o.national_number, o.extension,
         o.italian_leading_zero, o.number_of_leading_zeros),
        oracle.is_valid_number(o),
        oracle.is_possible_number(o),
        tuple(oracle.format_number(o, f) for f in FORMATS),
    )


def _mine_full(text, region):
    try:
        m = mine.parse(text, region)
    except mine.NumberParseException as e:
        return ("ERR", e.error_type, str(e))
    return (
        "OK",
        (m.country_code, m.national_number, m.extension,
         m.italian_leading_zero, m.number_of_leading_zeros),
        mine.is_valid_number(m),
        mine.is_possible_number(m),
        tuple(mine.format_number(m, f) for f in FORMATS),
    )


def _golden_corpus() -> list[tuple[str, str | None]]:
    """Every <exampleNumber> in the vendored metadata: once as a national
    string with its region, once in +cc form."""
    root = ET.parse(METADATA_XML).getroot()
    cases = []
    for terr in root.iter("territory"):
        rid = terr.get("id")
        cc = terr.get("countryCode")
        for el in terr:
            ex = el.find("exampleNumber")
            if ex is not None and ex.text:
                nsn = "".join(ex.text.split())
                cases.append((nsn, None if rid == "001" else rid))
                cases.append((f"+{cc} {nsn}", None))
    return cases


def _generated_corpus() -> list[tuple[str, str | None]]:
    rng = random.Random(20260920)
    markers = [
        "", " ext 42", " ext. 9", " x123", " X 7", " extn 11", " extension 555",
        " #123", ";ext=88", ",ext. 77", " xt 3", " int 9", "-123#", ",, 445", ", 998",
    ]
    punct = [" ", "-", ".", "(", ")", "  ", "\t"]
    cases = []
    for region in TOP_REGIONS:
        cc = REGION_CC[region]
        for _ in range(60):
            kind = rng.randrange(6)
            digits = "".join(rng.choice("0123456789") for _ in range(rng.randrange(1, 18)))
            if kind == 0:
                text = f"+{cc} {digits}"
            elif kind == 1:
                text = f"+{cc}{digits}"
            elif kind == 2:
                text = digits
            elif kind == 3:
                text = f"{cc}{digits}"
            elif kind == 4:
                text = f"+{cc} " + "".join(d + rng.choice(punct) for d in digits)
            else:
                text = digits
            text += rng.choice(markers)
            cases.append((text, region if rng.random() < 0.8 else None))
    return cases


EDGE_CASES = [
    "", "+", "++", "abc", "()", "1", "12", "123", "+0", "+00", "+999 123456",
    "tel:+1-212-555-1234", "tel:+1-212-555-1234;ext=99", "1-800-FLOWERS",
    "Call +1 212-555-1234 now", "+1 212-555-1234 ext. 123", "+1 212-555-1234x",
    "+1 212-555-1234 x", "011 44 20 7946 0018", "011 44", "011 4", "011 442",
    "011 9999 123456789", "+1 212555123456789012345", "+1 212-555-1234#",
    "(530) 583-6985 x302/x2303", "+1 212-555-1234 12345#", "+390212345678",
    "0812345678", "0268 464 1234", "39 02 12345678", "+1 268 464 1234",
    "1 345 949 1234", "464 1234", "1 999 9999", "0800 123 4567",
    "011 15 2345 6789", "9 11 2345 6789", "0 800 1234567", "011 234 5678",
    "011 234 567", "234 5678", "0 11 2345 6789", "1 2345", "999 123456",
    "44 20 7946 0018", "0 20 7946 0018", "020 7946 0018", "0044 20 7946 0018",
    "10439158,, 445", "2125551234ex 123", "+1 x42 212-555-1234",
    "212-555-1234x5", "212-555-1234, 12", "212-555-1234 123456",
    "1-800-XAMPLEZ", "+1 212-555-XYZA", "212-555-xyzw",
]
EDGE_REGIONS = [None, "US", "IT", "AG", "GB", "CA", "BR", "DE"]


GOLDEN = _golden_corpus()
GENERATED = _generated_corpus()
EDGES = [(t, r) for t in EDGE_CASES for r in EDGE_REGIONS]


@pytest.mark.parametrize("text,region", GOLDEN)
def test_golden_example_numbers(text, region):
    assert _mine_full(text, region) == _oracle_full(text, region)


@pytest.mark.parametrize("text,region", GENERATED)
def test_generated_numbers(text, region):
    assert _mine_full(text, region) == _oracle_full(text, region)


@pytest.mark.parametrize("text,region", EDGES)
def test_edge_cases(text, region):
    assert _mine_full(text, region) == _oracle_full(text, region)


def test_validate_column_matches_oracle():
    values = [t for t, _ in GOLDEN[::17]] + [t for t, _ in GENERATED[::13]]
    values += ["nope", "", "+999 1", "tel:+1-212-555-1234;ext=9"]
    for region in (None, "US", "GB", "IT"):
        expected = []
        for v in values:
            try:
                o = oracle.parse(v, region)
                expected.append(oracle.is_valid_number(o))
            except oracle.NumberParseException:
                expected.append(False)
        assert mine.validate_column(values, region) == expected


def test_non_ascii_inputs_match_oracle():
    """Non-ASCII inputs take the fallback engine on native runs; results must
    still be oracle-identical."""
    cases = [
        "＋1 212-555-1234",  # full-width plus
        "＋４４ ２０ ７９４６ ００１８",  # full-width digits
        "+1 212-555-1234 доб 12",  # Cyrillic extension marker
        "+44 20 7946 0018 extensión 5",
        "ｅｘｔ 42",
        "\u00a0+1 212-555-1234",  # nbsp prefix
    ]
    for text in cases:
        for region in (None, "US", "GB"):
            assert _mine_full(text, region) == _oracle_full(text, region)


def test_error_message_variants():
    """Every NumberParseException message the corpus can produce."""
    cases = [
        (None, "US", 1),
        ("x" * 251, "US", 4),
        ("+999 123456", None, 0),
        ("011 9999 123456789", "US", 0),
        ("2125551234", None, 0),
        ("2125551234", "ZZ", 0),
        ("+1 2", None, 1),
        ("011 44", "US", 2),
        ("+1 212555123456789012345", None, 4),
        ("tel:2125551234;phone-context=", None, 1),
    ]
    for text, region, expected_type in cases:
        with pytest.raises(mine.NumberParseException) as mine_exc:
            mine.parse(text, region)
        with pytest.raises(oracle.NumberParseException) as oracle_exc:
            oracle.parse(text, region)
        assert mine_exc.value.error_type == expected_type
        assert (mine_exc.value.error_type, str(mine_exc.value)) == (
            oracle_exc.value.error_type,
            str(oracle_exc.value),
        )


def test_backend_consistency():
    """The two backends must agree with each other (independent of oracle)."""
    from phonenumbers_mojo import _fallback as fb

    tables = fb.Tables.from_default_blob()
    for text, region in GENERATED[::7] + EDGES[::11]:
        try:
            a = mine.parse(text, region)
        except mine.NumberParseException:
            continue
        try:
            p = fb.parse(tables, text, region)
        except fb.ParseError:
            continue
        assert (a.country_code, a.national_number, a.extension) == (
            p.cc, int(p.nsn), p.ext
        )

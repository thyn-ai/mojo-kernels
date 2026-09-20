"""jsonschema-mojo: a drop-in faster replacement for the `jsonschema` package.

Same call shapes, same pass/fail outcomes, same failing paths — powered by a
Mojo kernel where the platform supports it (macOS arm64, Linux x86_64), with
a vendored pure-Python fallback everywhere else (including Windows).

    import jsonschema_mojo

    jsonschema_mojo.validate({"name": "Ada"}, schema)   # like jsonschema.validate
    validator = jsonschema_mojo.Validator(schema)       # reuse a compiled schema
    for error in validator.iter_errors(instance):
        print(error.json_path, error.message)

Draft 2020-12 subset: type, properties, required, items,
additionalProperties, enum, const, minimum, maximum, exclusiveMinimum,
exclusiveMaximum, minLength, maxLength, pattern, minItems, maxItems,
uniqueItems, minProperties, maxProperties, multipleOf (plus boolean
subschemas). Schemas using other validation keywords raise
UnsupportedSchemaError rather than silently diverging.

Set JSONSCHEMA_MOJO_DISABLE_NATIVE=1 to force the pure-Python fallback.
"""

from jsonschema_mojo._native import backend_info, native_available
from jsonschema_mojo.core import (
    Draft202012Validator,
    UnsupportedSchemaError,
    ValidationError,
    Validator,
    iter_errors,
    validate,
)

__version__ = "0.1.1"  # x-release-please-version
__all__ = [
    "Draft202012Validator",
    "UnsupportedSchemaError",
    "ValidationError",
    "Validator",
    "backend_info",
    "iter_errors",
    "native_available",
    "validate",
    "__version__",
]

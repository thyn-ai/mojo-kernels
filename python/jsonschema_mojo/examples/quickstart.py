"""jsonschema-mojo quickstart: the end-user smoke for a fresh wheel install.

Prints deterministic output; the native backend and the forced pure-Python
fallback (JSONSCHEMA_MOJO_DISABLE_NATIVE=1) must print exactly the same.
"""

import jsonschema_mojo

schema = {
    "type": "object",
    "properties": {
        "name": {"type": "string", "minLength": 1},
        "age": {"type": "integer", "minimum": 0, "maximum": 150},
        "email": {"type": "string", "pattern": "@"},
        "tags": {"type": "array", "items": {"type": "string"}, "uniqueItems": True},
    },
    "required": ["name", "age"],
    "additionalProperties": False,
}

validator = jsonschema_mojo.Validator(schema)

good = {"name": "Ada", "age": 36, "email": "ada@example.com", "tags": ["a", "b"]}
bad = {"name": "", "age": 200, "email": "nope", "tags": ["a", "a"], "extra": 1}

print("package:", jsonschema_mojo.__name__, jsonschema_mojo.__version__)
print("backend:", validator.backend)
print("good valid:", validator.is_valid(good))
print("bad valid:", validator.is_valid(bad))
for error in sorted(
    validator.iter_errors(bad), key=lambda e: (e.json_path, e.validator or "")
):
    print(f"error: {error.json_path} [{error.validator}] {error.message}")
try:
    jsonschema_mojo.validate({"age": -1}, schema)
except jsonschema_mojo.ValidationError as e:
    print("raised:", e.message)

"""Torch-free evidence-scope language shared by every output surface."""

from typing import Final

EVIDENCE_BOUNDARY: Final = (
    "Machine-local CPU evidence; laptop thermal state may affect timings."
)


def scope_json() -> dict[str, object]:
    """Return the canonical schema-1 scope evidence payload."""
    return {"schema_version": 1, "boundary": EVIDENCE_BOUNDARY}


def is_scope_json(value: object) -> bool:
    """Require the exact schema, including integer-not-boolean version typing."""
    return (
        isinstance(value, dict)
        and value.keys() == {"schema_version", "boundary"}
        and type(value.get("schema_version")) is int
        and value["schema_version"] == 1
        and value.get("boundary") == EVIDENCE_BOUNDARY
    )


__all__ = ["EVIDENCE_BOUNDARY", "is_scope_json", "scope_json"]

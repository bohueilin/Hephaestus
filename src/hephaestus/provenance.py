"""Unsigned schema-1 tool-layer run provenance and parent-chain validation."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from hephaestus.bundle import strict_json_loads

_PROVENANCE_KEYS: Final = {
    "schema_version",
    "orchestration_id",
    "run_id",
    "sequence_index",
    "predecessor",
}


@dataclass(frozen=True, slots=True)
class ProvenancePredecessor:
    run_id: str
    manifest_sha256: str

    def as_json(self) -> dict[str, object]:
        return {
            "run_id": self.run_id,
            "manifest_sha256": self.manifest_sha256,
        }


@dataclass(frozen=True, slots=True)
class RunProvenance:
    orchestration_id: str
    run_id: str
    sequence_index: int
    predecessor: ProvenancePredecessor | None

    def __post_init__(self) -> None:
        if not _is_token(self.orchestration_id) or not _is_token(self.run_id):
            raise ValueError("provenance identifiers must be lowercase 256-bit tokens")
        if type(self.sequence_index) is not int or self.sequence_index < 0:
            raise ValueError("provenance sequence index must be nonnegative")
        if self.predecessor is not None and not isinstance(
            self.predecessor, ProvenancePredecessor
        ):
            raise ValueError("invalid provenance predecessor")

    def as_json(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "orchestration_id": self.orchestration_id,
            "run_id": self.run_id,
            "sequence_index": self.sequence_index,
            "predecessor": (
                None if self.predecessor is None else self.predecessor.as_json()
            ),
        }


class ProvenanceError(ValueError):
    """Stored normal-run provenance or a multi-child chain is invalid."""


def parse_run_provenance(value: object) -> RunProvenance:
    """Parse one exact schema-1 provenance object with strict scalar types."""
    if not isinstance(value, dict) or value.keys() != _PROVENANCE_KEYS:
        raise ProvenanceError("provenance:schema")
    if type(value.get("schema_version")) is not int or value["schema_version"] != 1:
        raise ProvenanceError("provenance:schema")
    predecessor_value = value.get("predecessor")
    predecessor: ProvenancePredecessor | None
    if predecessor_value is None:
        predecessor = None
    elif (
        isinstance(predecessor_value, dict)
        and predecessor_value.keys() == {"run_id", "manifest_sha256"}
        and _is_token(predecessor_value.get("run_id"))
        and _is_token(predecessor_value.get("manifest_sha256"))
    ):
        predecessor = ProvenancePredecessor(
            predecessor_value["run_id"],
            predecessor_value["manifest_sha256"],
        )
    else:
        raise ProvenanceError("provenance:predecessor")
    try:
        return RunProvenance(
            value.get("orchestration_id"),  # type: ignore[arg-type]
            value.get("run_id"),  # type: ignore[arg-type]
            value.get("sequence_index"),  # type: ignore[arg-type]
            predecessor,
        )
    except ValueError as error:
        raise ProvenanceError("provenance:schema") from error


def read_run_provenance(bundle: Path) -> RunProvenance:
    """Read and parse one manifested normal child provenance payload."""
    try:
        value = strict_json_loads((bundle / "run_provenance.json").read_bytes())
    except (OSError, UnicodeDecodeError, ValueError) as error:
        raise ProvenanceError("provenance:unreadable") from error
    return parse_run_provenance(value)


def validate_provenance_chain(children: tuple[Path, ...]) -> tuple[RunProvenance, ...]:
    """Bind distinct child IDs and exact ordinal predecessor manifest digests."""
    if not children:
        raise ProvenanceError("provenance:empty")
    records = tuple(read_run_provenance(child) for child in children)
    orchestration_id = records[0].orchestration_id
    if any(record.orchestration_id != orchestration_id for record in records):
        raise ProvenanceError("provenance:orchestration")
    run_ids = tuple(record.run_id for record in records)
    if len(set(run_ids)) != len(run_ids):
        raise ProvenanceError("provenance:duplicate_run_id")
    for index, (_child, record) in enumerate(zip(children, records, strict=True)):
        if record.sequence_index != index:
            raise ProvenanceError("provenance:sequence")
        if index == 0:
            if record.predecessor is not None:
                raise ProvenanceError("provenance:first_predecessor")
            continue
        prior_child = children[index - 1]
        prior_record = records[index - 1]
        try:
            prior_manifest = (prior_child / "manifest.json").read_bytes()
        except OSError as error:
            raise ProvenanceError("provenance:manifest") from error
        expected = ProvenancePredecessor(
            prior_record.run_id,
            hashlib.sha256(prior_manifest).hexdigest(),
        )
        if record.predecessor != expected:
            raise ProvenanceError("provenance:predecessor")
    return records


def _is_token(value: object) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(
        character in "0123456789abcdef" for character in value
    )


__all__ = [
    "ProvenanceError",
    "ProvenancePredecessor",
    "RunProvenance",
    "parse_run_provenance",
    "read_run_provenance",
    "validate_provenance_chain",
]

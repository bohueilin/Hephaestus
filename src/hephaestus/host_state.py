"""Torch-free host-state sampling around one trusted operation."""

from __future__ import annotations

import math
import os
import re
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import PurePosixPath
from typing import Final

from hephaestus.privacy import validate_public_evidence
from hephaestus.provenance import (
    ProvenanceError,
    ProvenancePredecessor,
    RunProvenance,
    parse_run_provenance,
)

HOST_STATE_SCHEMA_VERSION: Final = 1
UNAVAILABLE_REASONS: Final = frozenset(
    {
        "collection_failed",
        "malformed_output",
        "nonfinite_output",
    }
)

_RECORD_KEYS: Final = {
    "schema_version",
    "run_provenance",
    "bundle_relative_path",
    "started_at_utc",
    "ended_at_utc",
    "elapsed_seconds",
    "before",
    "after",
}
_SNAPSHOT_KEYS: Final = {
    "load_average",
}
_METRIC_KEYS: Final = {"value", "unavailable_reason"}
_UTC_PATTERN: Final = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{6}Z$")


@dataclass(frozen=True, slots=True)
class HostStateSnapshot:
    """Immutable, copy-owned host-state evidence for one sampling instant."""

    load_average: tuple[float, float, float] | None
    load_average_unavailable_reason: str | None

    def __post_init__(self) -> None:
        _validate_snapshot_value_pair(
            self.load_average,
            self.load_average_unavailable_reason,
            "load_average",
        )
        if self.load_average is not None:
            if type(self.load_average) is not tuple or len(self.load_average) != 3:
                raise ValueError("invalid host-state load_average")
            for item in self.load_average:
                _finite_nonnegative_float(item, "load_average")

    def as_json(self) -> dict[str, object]:
        """Return a fresh exact-built-in JSON tree for persistence."""
        return {
            "load_average": {
                "value": (
                    None if self.load_average is None else list(self.load_average)
                ),
                "unavailable_reason": self.load_average_unavailable_reason,
            },
        }


@dataclass(frozen=True, slots=True)
class HostStateCapture:
    """Validated before/after capture awaiting a trusted result-path binding."""

    run_provenance: RunProvenance
    started_at_utc: str
    ended_at_utc: str
    elapsed_seconds: float
    before: HostStateSnapshot
    after: HostStateSnapshot

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "run_provenance",
            _clone_exact_provenance(self.run_provenance),
        )
        started = _parse_timestamp(self.started_at_utc)
        ended = _parse_timestamp(self.ended_at_utc)
        if ended < started:
            raise ValueError("invalid host-state UTC chronology")
        _finite_nonnegative_float(self.elapsed_seconds, "elapsed")
        if type(self.before) is not HostStateSnapshot:
            raise ValueError("host-state before must be an exact snapshot DTO")
        if type(self.after) is not HostStateSnapshot:
            raise ValueError("host-state after must be an exact snapshot DTO")


def capture_operation[T](
    operation: Callable[[], T],
    provenance: RunProvenance,
    *,
    sampler: Callable[[], dict[str, object]] = lambda: sample_host_state(),
    monotonic: Callable[[], float] = time.monotonic,
    utc_now: Callable[[], datetime] = lambda: datetime.now(UTC),
) -> tuple[T, HostStateCapture]:
    """Capture host state immediately around a successful operation.

    An exception from ``operation`` escapes unchanged. In that case no after-snapshot
    or completed record is produced.
    """
    owned_provenance = _clone_exact_provenance(provenance)

    before = _snapshot_from_json(sampler())
    started_at_utc = _timestamp_utc(utc_now())
    started = monotonic()
    result = operation()
    ended = monotonic()
    after = _snapshot_from_json(sampler())
    ended_at_utc = _timestamp_utc(utc_now())

    elapsed_seconds = _finite_nonnegative_float(ended - started, "elapsed")
    capture = HostStateCapture(
        run_provenance=owned_provenance,
        started_at_utc=started_at_utc,
        ended_at_utc=ended_at_utc,
        elapsed_seconds=elapsed_seconds,
        before=before,
        after=after,
    )
    return result, capture


def finalize_host_state_record(
    capture: HostStateCapture,
    bundle_relative_path: str,
) -> dict[str, object]:
    """Bind a completed capture to its trusted outer-root-relative run bundle."""
    if type(capture) is not HostStateCapture:
        raise ValueError("host-state finalization requires an exact capture DTO")
    if not _is_run_relative_path(bundle_relative_path):
        raise ValueError("bundle relative path must be a canonical path below runs/")
    record: dict[str, object] = {
        "schema_version": HOST_STATE_SCHEMA_VERSION,
        "run_provenance": capture.run_provenance.as_json(),
        "bundle_relative_path": bundle_relative_path,
        "started_at_utc": capture.started_at_utc,
        "ended_at_utc": capture.ended_at_utc,
        "elapsed_seconds": capture.elapsed_seconds,
        "before": capture.before.as_json(),
        "after": capture.after.as_json(),
    }
    validate_host_state_record(record)
    return record


def sample_host_state() -> dict[str, object]:
    """Collect one identity-free host snapshot with explicit unavailability."""
    snapshot = {"load_average": _load_average()}
    _validate_snapshot(snapshot)
    return snapshot


def validate_host_state_record(record: object) -> None:
    """Validate the exact schema-1 host-state record and its privacy boundary."""
    _validate_exact_json_tree(record)
    if type(record) is not dict or record.keys() != _RECORD_KEYS:
        raise ValueError("invalid host-state record schema")
    if (
        type(record.get("schema_version")) is not int
        or record["schema_version"] != HOST_STATE_SCHEMA_VERSION
    ):
        raise ValueError("invalid host-state record schema")
    try:
        parse_run_provenance(record.get("run_provenance"))
    except ProvenanceError as error:
        raise ValueError("invalid host-state provenance") from error
    if not _is_run_relative_path(record.get("bundle_relative_path")):
        raise ValueError("invalid host-state bundle relative path")

    started = _parse_timestamp(record.get("started_at_utc"))
    ended = _parse_timestamp(record.get("ended_at_utc"))
    if ended < started:
        raise ValueError("invalid host-state UTC chronology")
    _finite_nonnegative_float(record.get("elapsed_seconds"), "elapsed")
    _validate_snapshot(record.get("before"))
    _validate_snapshot(record.get("after"))
    validate_public_evidence(record)


def _clone_exact_provenance(provenance: object) -> RunProvenance:
    if type(provenance) is not RunProvenance:
        raise ValueError("host-state provenance must be trusted RunProvenance")
    orchestration_id = provenance.orchestration_id
    run_id = provenance.run_id
    sequence_index = provenance.sequence_index
    if (
        type(orchestration_id) is not str
        or type(run_id) is not str
        or type(sequence_index) is not int
    ):
        raise ValueError("host-state provenance fields must use exact built-in types")

    predecessor = provenance.predecessor
    predecessor_json: dict[str, object] | None
    if predecessor is None:
        predecessor_json = None
    elif isinstance(predecessor, ProvenancePredecessor):
        predecessor_run_id = predecessor.run_id
        predecessor_manifest_sha256 = predecessor.manifest_sha256
        if (
            type(predecessor_run_id) is not str
            or type(predecessor_manifest_sha256) is not str
        ):
            raise ValueError(
                "host-state provenance fields must use exact built-in types"
            )
        predecessor_json = {
            "run_id": predecessor_run_id,
            "manifest_sha256": predecessor_manifest_sha256,
        }
    else:
        raise ValueError("invalid host-state provenance predecessor")

    provenance_json: dict[str, object] = {
        "schema_version": 1,
        "orchestration_id": orchestration_id,
        "run_id": run_id,
        "sequence_index": sequence_index,
        "predecessor": predecessor_json,
    }
    try:
        return parse_run_provenance(provenance_json)
    except ProvenanceError as error:
        raise ValueError("invalid host-state provenance") from error


def _load_average() -> dict[str, object]:
    try:
        raw = os.getloadavg()
    except (AttributeError, OSError):
        return _unavailable_metric("collection_failed")
    if type(raw) is not tuple or len(raw) != 3:
        return _unavailable_metric("malformed_output")
    try:
        values = [float(value) for value in raw]
    except (TypeError, ValueError, OverflowError):
        return _unavailable_metric("malformed_output")
    if any(not math.isfinite(value) for value in values):
        return _unavailable_metric("nonfinite_output")
    if any(value < 0 for value in values):
        return _unavailable_metric("malformed_output")
    return _available_metric(values)


def _available_metric(value: object) -> dict[str, object]:
    return {"value": value, "unavailable_reason": None}


def _unavailable_metric(reason: str) -> dict[str, object]:
    if reason not in UNAVAILABLE_REASONS:
        raise ValueError("invalid host-state unavailability reason")
    return {"value": None, "unavailable_reason": reason}


def _validate_snapshot(snapshot: object) -> None:
    if type(snapshot) is not dict or snapshot.keys() != _SNAPSHOT_KEYS:
        raise ValueError("invalid host-state snapshot schema")
    _validate_load_average(snapshot.get("load_average"))


def _snapshot_from_json(snapshot: object) -> HostStateSnapshot:
    _validate_exact_json_tree(snapshot)
    _validate_snapshot(snapshot)
    assert type(snapshot) is dict
    load_average = snapshot["load_average"]
    assert type(load_average) is dict
    load_average_value = load_average["value"]
    return HostStateSnapshot(
        load_average=(
            None if load_average_value is None else tuple(load_average_value)
        ),
        load_average_unavailable_reason=load_average["unavailable_reason"],
    )


def _validate_snapshot_value_pair(
    value: object,
    unavailable_reason: object,
    name: str,
) -> None:
    if value is None:
        if (
            type(unavailable_reason) is not str
            or unavailable_reason not in UNAVAILABLE_REASONS
        ):
            raise ValueError(f"invalid host-state {name}")
    elif unavailable_reason is not None:
        raise ValueError(f"invalid host-state {name}")


def _validate_exact_json_tree(value: object) -> None:
    if value is None or type(value) in {bool, int, float, str}:
        return
    if type(value) is list:
        for item in value:
            _validate_exact_json_tree(item)
        return
    if type(value) is dict:
        for key, item in dict.items(value):
            if type(key) is not str:
                raise ValueError(
                    "host-state record must use exact built-in JSON types"
                )
            _validate_exact_json_tree(item)
        return
    raise ValueError("host-state record must use exact built-in JSON types")


def _validate_load_average(metric: object) -> None:
    value = _validate_metric_envelope(metric, "load_average")
    if value is None:
        return
    if type(value) is not list or len(value) != 3:
        raise ValueError("invalid host-state load_average")
    for item in value:
        _finite_nonnegative_float(item, "load_average")


def _validate_metric_envelope(metric: object, name: str) -> object:
    if type(metric) is not dict or metric.keys() != _METRIC_KEYS:
        raise ValueError(f"invalid host-state {name}")
    value = metric.get("value")
    reason = metric.get("unavailable_reason")
    if value is None:
        if type(reason) is not str or reason not in UNAVAILABLE_REASONS:
            raise ValueError(f"invalid host-state {name}")
    elif reason is not None:
        raise ValueError(f"invalid host-state {name}")
    return value


def _timestamp_utc(value: datetime) -> str:
    if not isinstance(value, datetime) or value.utcoffset() != timedelta(0):
        raise ValueError("host-state clock must return an aware UTC datetime")
    return value.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def _parse_timestamp(value: object) -> datetime:
    if type(value) is not str or _UTC_PATTERN.fullmatch(value) is None:
        raise ValueError("invalid host-state UTC timestamp")
    try:
        return datetime.strptime(value, "%Y-%m-%dT%H:%M:%S.%fZ").replace(tzinfo=UTC)
    except ValueError as error:
        raise ValueError("invalid host-state UTC timestamp") from error


def _finite_nonnegative_float(value: object, name: str) -> float:
    if type(value) is not float or not math.isfinite(value) or value < 0:
        raise ValueError(f"invalid host-state {name}")
    return value


def _is_run_relative_path(path: object) -> bool:
    if type(path) is not str or not path or "\\" in path or "\x00" in path:
        return False
    candidate = PurePosixPath(path)
    parts = candidate.parts
    direct_run = len(parts) == 2 and parts[0] == "runs"
    generated_parent_run = len(parts) == 3 and parts[0] != "runs" and parts[1] == "runs"
    return (
        candidate.as_posix() == path
        and not candidate.is_absolute()
        and (direct_run or generated_parent_run)
        and "." not in parts
        and ".." not in parts
    )


__all__ = [
    "HOST_STATE_SCHEMA_VERSION",
    "HostStateCapture",
    "HostStateSnapshot",
    "UNAVAILABLE_REASONS",
    "capture_operation",
    "finalize_host_state_record",
    "sample_host_state",
    "validate_host_state_record",
]

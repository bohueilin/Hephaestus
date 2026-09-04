from __future__ import annotations

import copy
import importlib
import importlib.util
import math
from dataclasses import FrozenInstanceError
from datetime import UTC, datetime

import pytest

from hephaestus.bundle import canonical_json_bytes, strict_json_loads
from hephaestus.provenance import ProvenancePredecessor, RunProvenance

RUN_ID = "1" * 64
ORCHESTRATION_ID = "2" * 64
MANIFEST_DIGEST = "3" * 64


class _PathStringSubclass(str):
    pass


class _DictSubclass(dict):
    pass


class _ListSubclass(list):
    pass


class _StringSubclass(str):
    pass


class _FloatSubclass(float):
    pass


class _SplitViewDict(dict):
    def __init__(self, validation_view: dict[str, object], serialization_view: dict[str, object]):
        super().__init__(serialization_view)
        self._validation_view = validation_view

    def keys(self):
        return self._validation_view.keys()

    def get(self, key, default=None):
        return self._validation_view.get(key, default)

    def __getitem__(self, key):
        return self._validation_view[key]

    def items(self):
        return dict.items(self)


def _host_state():
    assert importlib.util.find_spec("hephaestus.host_state") is not None
    return importlib.import_module("hephaestus.host_state")


def _provenance() -> RunProvenance:
    return RunProvenance(
        orchestration_id=ORCHESTRATION_ID,
        run_id=RUN_ID,
        sequence_index=4,
        predecessor=ProvenancePredecessor(
            run_id="4" * 64,
            manifest_sha256=MANIFEST_DIGEST,
        ),
    )


def _available_snapshot(marker: float) -> dict[str, object]:
    return {
        "load_average": {"value": [marker, marker + 1.0, marker + 2.0], "unavailable_reason": None},
    }


@pytest.mark.parametrize(
    ("getloadavg_result", "expected"),
    (
        (
            (0.25, 1.5, 2.75),
            {"load_average": {"value": [0.25, 1.5, 2.75], "unavailable_reason": None}},
        ),
        (
            OSError("load average unavailable"),
            {
                "load_average": {
                    "value": None,
                    "unavailable_reason": "collection_failed",
                }
            },
        ),
    ),
)
def test_direct_sampler_reads_only_load_average(
    monkeypatch: pytest.MonkeyPatch,
    getloadavg_result: object,
    expected: dict[str, object],
) -> None:
    """A fabricated snapshot that bypasses os.getloadavg must fail this direct boundary."""
    host_state = _host_state()
    calls: list[None] = []

    def getloadavg() -> tuple[float, float, float]:
        calls.append(None)
        if isinstance(getloadavg_result, OSError):
            raise getloadavg_result
        assert type(getloadavg_result) is tuple
        return getloadavg_result

    monkeypatch.setattr(host_state.os, "getloadavg", getloadavg)

    assert host_state.sample_host_state() == expected
    assert calls == [None]


def _captured_record() -> dict[str, object]:
    host_state = _host_state()
    timestamps = iter(
        (
            datetime(2026, 8, 31, 10, 0, 0, 123456, tzinfo=UTC),
            datetime(2026, 8, 31, 10, 0, 2, 654321, tzinfo=UTC),
        )
    )
    monotonic_values = iter((20.0, 21.25))
    snapshots = iter((_available_snapshot(1.0), _available_snapshot(5.0)))
    _, capture = host_state.capture_operation(
        lambda: "result",
        _provenance(),
        sampler=lambda: next(snapshots),
        monotonic=lambda: next(monotonic_values),
        utc_now=lambda: next(timestamps),
    )
    return host_state.finalize_host_state_record(capture, "runs/child-1")


def test_capture_orders_snapshots_around_only_the_operation() -> None:
    """Moving either sample or a clock call inside the operation changes the evidence boundary."""
    host_state = _host_state()
    events: list[str] = []
    snapshots = iter((_available_snapshot(1.0), _available_snapshot(5.0)))
    monotonic_values = iter((100.0, 101.75))
    timestamps = iter(
        (
            datetime(2026, 8, 31, 10, 0, 0, 123456, tzinfo=UTC),
            datetime(2026, 8, 31, 10, 0, 2, 654321, tzinfo=UTC),
        )
    )

    def sample() -> dict[str, object]:
        events.append("sample")
        return next(snapshots)

    def utc_now() -> datetime:
        events.append("utc")
        return next(timestamps)

    def monotonic() -> float:
        events.append("monotonic")
        return next(monotonic_values)

    def operation() -> str:
        events.append("operation")
        return "operation-result"

    result, capture = host_state.capture_operation(
        operation,
        _provenance(),
        sampler=sample,
        monotonic=monotonic,
        utc_now=utc_now,
    )
    record = host_state.finalize_host_state_record(capture, "runs/child-1")

    assert result == "operation-result"
    assert events == [
        "sample",
        "utc",
        "monotonic",
        "operation",
        "monotonic",
        "sample",
        "utc",
    ]
    assert record == {
        "schema_version": 1,
        "run_provenance": _provenance().as_json(),
        "bundle_relative_path": "runs/child-1",
        "started_at_utc": "2026-08-31T10:00:00.123456Z",
        "ended_at_utc": "2026-08-31T10:00:02.654321Z",
        "elapsed_seconds": 1.75,
        "before": _available_snapshot(1.0),
        "after": _available_snapshot(5.0),
    }


def test_capture_preserves_operation_exception_without_after_snapshot() -> None:
    """A failed operation must not be converted into a completed host-state record."""
    host_state = _host_state()
    events: list[str] = []
    failure = LookupError("ordinary operation failure")

    def sample() -> dict[str, object]:
        events.append("sample")
        return _available_snapshot(1.0)

    def operation() -> None:
        events.append("operation")
        raise failure

    with pytest.raises(LookupError) as caught:
        host_state.capture_operation(
            operation,
            _provenance(),
            sampler=sample,
            monotonic=lambda: 10.0,
            utc_now=lambda: datetime(2026, 8, 31, tzinfo=UTC),
        )

    assert caught.value is failure
    assert events == ["sample", "operation"]


def test_capture_requires_explicit_post_operation_bundle_binding() -> None:
    """The bundle path learned from the result is bound only after capture completes."""
    host_state = _host_state()
    snapshots = iter((_available_snapshot(1.0), _available_snapshot(2.0)))
    monotonic_values = iter((4.0, 5.0))
    timestamps = iter(
        (
            datetime(2026, 8, 31, 10, 0, 0, tzinfo=UTC),
            datetime(2026, 8, 31, 10, 0, 1, tzinfo=UTC),
        )
    )

    result, capture = host_state.capture_operation(
        lambda: {"bundle_relative_path": "runs/child-1"},
        _provenance(),
        sampler=lambda: next(snapshots),
        monotonic=lambda: next(monotonic_values),
        utc_now=lambda: next(timestamps),
    )
    record = host_state.finalize_host_state_record(
        capture,
        result["bundle_relative_path"],
    )

    assert type(capture) is host_state.HostStateCapture
    assert record["bundle_relative_path"] == "runs/child-1"
    assert record.keys() == {
        "schema_version",
        "run_provenance",
        "bundle_relative_path",
        "started_at_utc",
        "ended_at_utc",
        "elapsed_seconds",
        "before",
        "after",
    }
    host_state.validate_host_state_record(record)


def test_capture_deeply_owns_sampler_evidence_before_operation_mutation() -> None:
    """The operation cannot rewrite the already-sampled before evidence through aliases."""
    host_state = _host_state()
    before = _available_snapshot(1.0)
    after = _available_snapshot(5.0)

    def operation() -> str:
        before["load_average"]["value"][0] = 99.0
        return "result"

    result, capture = host_state.capture_operation(
        operation,
        _provenance(),
        sampler=iter((before, after)).__next__,
        monotonic=iter((1.0, 2.0)).__next__,
        utc_now=iter(
            (
                datetime(2026, 8, 31, 10, 0, 0, tzinfo=UTC),
                datetime(2026, 8, 31, 10, 0, 1, tzinfo=UTC),
            )
        ).__next__,
    )

    assert result == "result"
    assert type(capture.before) is host_state.HostStateSnapshot
    assert capture.before.as_json() == _available_snapshot(1.0)


def test_capture_is_immutable_and_final_records_have_no_nested_aliases() -> None:
    """Source, capture-access, and prior-record mutations cannot alter captured evidence."""
    host_state = _host_state()
    before = _available_snapshot(1.0)
    after = _available_snapshot(5.0)
    _, capture = host_state.capture_operation(
        lambda: None,
        _provenance(),
        sampler=iter((before, after)).__next__,
        monotonic=iter((1.0, 2.0)).__next__,
        utc_now=iter(
            (
                datetime(2026, 8, 31, 10, 0, 0, tzinfo=UTC),
                datetime(2026, 8, 31, 10, 0, 1, tzinfo=UTC),
            )
        ).__next__,
    )

    access_copy = capture.before.as_json()
    access_copy["load_average"]["value"][0] = 77.0
    before["load_average"]["value"][1] = 88.0
    with pytest.raises(FrozenInstanceError):
        capture.before.load_average = None

    first = host_state.finalize_host_state_record(capture, "runs/child-1")
    second = host_state.finalize_host_state_record(capture, "runs/child-1")
    assert first["before"] is not second["before"]
    assert first["before"]["load_average"] is not second["before"]["load_average"]
    assert first["before"]["load_average"]["value"] is not second["before"]["load_average"]["value"]
    first["before"]["load_average"]["value"][2] = 66.0
    first["after"]["load_average"]["value"][0] = 55.0

    third = host_state.finalize_host_state_record(capture, "runs/child-1")
    assert second == third
    assert third["before"] == _available_snapshot(1.0)
    assert third["after"] == _available_snapshot(5.0)


def test_capture_clones_mutable_predecessor_before_operation_and_stays_stable() -> None:
    """Nested provenance behavior cannot change captured evidence after entry."""
    host_state = _host_state()
    digest_state = {"value": MANIFEST_DIGEST}

    class MutablePredecessor(ProvenancePredecessor):
        def as_json(self) -> dict[str, object]:
            return {
                "run_id": self.run_id,
                "manifest_sha256": digest_state["value"],
            }

    source_predecessor = MutablePredecessor("4" * 64, MANIFEST_DIGEST)
    source = RunProvenance(
        orchestration_id=ORCHESTRATION_ID,
        run_id=RUN_ID,
        sequence_index=4,
        predecessor=source_predecessor,
    )

    def operation() -> None:
        digest_state["value"] = "5" * 64

    _, capture = host_state.capture_operation(
        operation,
        source,
        sampler=lambda: _available_snapshot(1.0),
        monotonic=iter((1.0, 2.0)).__next__,
        utc_now=iter(
            (
                datetime(2026, 8, 31, 10, 0, 0, tzinfo=UTC),
                datetime(2026, 8, 31, 10, 0, 1, tzinfo=UTC),
            )
        ).__next__,
    )
    first = host_state.finalize_host_state_record(capture, "runs/child-1")
    digest_state["value"] = "6" * 64
    second = host_state.finalize_host_state_record(capture, "runs/child-1")

    assert first["run_provenance"]["predecessor"]["manifest_sha256"] == MANIFEST_DIGEST
    assert second == first
    host_state.validate_host_state_record(first)
    host_state.validate_host_state_record(second)


def test_capture_owns_a_cloned_exact_provenance_graph() -> None:
    """Even ordinary trusted provenance is copied into exact capture-owned DTOs."""
    host_state = _host_state()
    source = _provenance()

    _, capture = host_state.capture_operation(
        lambda: None,
        source,
        sampler=lambda: _available_snapshot(1.0),
        monotonic=iter((1.0, 2.0)).__next__,
        utc_now=iter(
            (
                datetime(2026, 8, 31, 10, 0, 0, tzinfo=UTC),
                datetime(2026, 8, 31, 10, 0, 1, tzinfo=UTC),
            )
        ).__next__,
    )

    assert type(capture.run_provenance) is RunProvenance
    assert capture.run_provenance is not source
    assert type(capture.run_provenance.predecessor) is ProvenancePredecessor
    assert capture.run_provenance.predecessor is not source.predecessor


def test_direct_capture_construction_clones_an_exact_nested_predecessor() -> None:
    """Direct DTO construction applies the same deep provenance ownership boundary."""
    host_state = _host_state()

    class PredecessorSubclass(ProvenancePredecessor):
        pass

    source_predecessor = PredecessorSubclass("4" * 64, MANIFEST_DIGEST)
    source = RunProvenance(
        orchestration_id=ORCHESTRATION_ID,
        run_id=RUN_ID,
        sequence_index=4,
        predecessor=source_predecessor,
    )
    _, ordinary = host_state.capture_operation(
        lambda: None,
        _provenance(),
        sampler=lambda: _available_snapshot(1.0),
        monotonic=iter((1.0, 2.0)).__next__,
        utc_now=iter(
            (
                datetime(2026, 8, 31, 10, 0, 0, tzinfo=UTC),
                datetime(2026, 8, 31, 10, 0, 1, tzinfo=UTC),
            )
        ).__next__,
    )

    direct = host_state.HostStateCapture(
        run_provenance=source,
        started_at_utc=ordinary.started_at_utc,
        ended_at_utc=ordinary.ended_at_utc,
        elapsed_seconds=ordinary.elapsed_seconds,
        before=ordinary.before,
        after=ordinary.after,
    )

    assert direct.run_provenance is not source
    assert type(direct.run_provenance.predecessor) is ProvenancePredecessor
    assert direct.run_provenance.predecessor is not source_predecessor


@pytest.mark.parametrize("entrypoint", ("capture_operation", "direct_capture"))
@pytest.mark.parametrize(
    "field",
    (
        "orchestration_id",
        "run_id",
        "predecessor_run_id",
        "predecessor_manifest_sha256",
    ),
)
def test_capture_rejects_provenance_identifier_string_subclasses(
    entrypoint: str,
    field: str,
) -> None:
    """Every authority-bearing provenance identifier must be an exact built-in string."""
    host_state = _host_state()
    predecessor = ProvenancePredecessor(
        run_id=(
            _StringSubclass("4" * 64)
            if field == "predecessor_run_id"
            else "4" * 64
        ),
        manifest_sha256=(
            _StringSubclass(MANIFEST_DIGEST)
            if field == "predecessor_manifest_sha256"
            else MANIFEST_DIGEST
        ),
    )
    source = RunProvenance(
        orchestration_id=(
            _StringSubclass(ORCHESTRATION_ID)
            if field == "orchestration_id"
            else ORCHESTRATION_ID
        ),
        run_id=_StringSubclass(RUN_ID) if field == "run_id" else RUN_ID,
        sequence_index=4,
        predecessor=predecessor,
    )

    with pytest.raises(ValueError, match="provenance.*exact built-in"):
        if entrypoint == "capture_operation":
            host_state.capture_operation(lambda: None, source)
        else:
            _, ordinary = host_state.capture_operation(
                lambda: None,
                _provenance(),
                sampler=lambda: _available_snapshot(1.0),
                monotonic=iter((1.0, 2.0)).__next__,
                utc_now=iter(
                    (
                        datetime(2026, 8, 31, 10, 0, 0, tzinfo=UTC),
                        datetime(2026, 8, 31, 10, 0, 1, tzinfo=UTC),
                    )
                ).__next__,
            )
            host_state.HostStateCapture(
                run_provenance=source,
                started_at_utc=ordinary.started_at_utc,
                ended_at_utc=ordinary.ended_at_utc,
                elapsed_seconds=ordinary.elapsed_seconds,
                before=ordinary.before,
                after=ordinary.after,
            )


@pytest.mark.parametrize(
    "path",
    (
        "runs/child-1",
        "agent-search-20260831T100000.000000Z-mlp_stack/runs/child-1",
        "aa-test-20260831T100000.000000Z-mlp_stack/runs/child-1",
        "planted-demo-20260831T100000.000000Z/runs/child-1",
    ),
)
def test_finalization_accepts_direct_or_one_generated_parent_run_path(path: str) -> None:
    """Direct runs and one generated parent are the exact trusted output topologies."""
    host_state = _host_state()
    _, capture = host_state.capture_operation(
        lambda: None,
        _provenance(),
        sampler=lambda: _available_snapshot(1.0),
        monotonic=iter((1.0, 2.0)).__next__,
        utc_now=iter(
            (
                datetime(2026, 8, 31, 10, 0, 0, tzinfo=UTC),
                datetime(2026, 8, 31, 10, 0, 1, tzinfo=UTC),
            )
        ).__next__,
    )

    record = host_state.finalize_host_state_record(capture, path)

    assert record["bundle_relative_path"] == path
    host_state.validate_host_state_record(record)


@pytest.mark.parametrize(
    "path",
    (
        "/runs/child",
        "outer/agent-search-parent/runs/child",
        "agent-search-parent/nested/runs/child",
        "agent-search-parent/children/child",
        "agent-search-parent/runs",
        "runs/../child",
        "agent-search-parent/runs/../child",
        "agent-search-parent/./runs/child",
        "runs\\child",
        "agent-search-parent\\runs\\child",
        "runs//child",
        "agent-search-parent//runs/child",
        "runs/child/",
        "agent-search-parent/runs/child/",
        "runs/child\x00tail",
        "agent-search-parent/runs/child\x00tail",
        "agent-search-parent\x00/runs/child",
        "child",
        _PathStringSubclass("runs/child"),
    ),
)
def test_finalization_rejects_noncanonical_or_non_run_bundle_paths(path: str) -> None:
    """Persisted binding accepts only canonical relative POSIX children below runs/."""
    host_state = _host_state()
    _, capture = host_state.capture_operation(
        lambda: None,
        _provenance(),
        sampler=lambda: _available_snapshot(1.0),
        monotonic=iter((1.0, 2.0)).__next__,
        utc_now=iter(
            (
                datetime(2026, 8, 31, 10, 0, 0, tzinfo=UTC),
                datetime(2026, 8, 31, 10, 0, 1, tzinfo=UTC),
            )
        ).__next__,
    )

    with pytest.raises(ValueError, match="bundle relative path"):
        host_state.finalize_host_state_record(capture, path)


def test_capture_requires_exact_trusted_provenance_type() -> None:
    """An authority-bearing RunProvenance subclass must not cross the trusted boundary."""
    host_state = _host_state()

    class ProvenanceSubclass(RunProvenance):
        pass

    provenance = ProvenanceSubclass(
        orchestration_id=ORCHESTRATION_ID,
        run_id=RUN_ID,
        sequence_index=0,
        predecessor=None,
    )

    with pytest.raises(ValueError, match="trusted RunProvenance"):
        host_state.capture_operation(lambda: None, provenance)


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        (lambda record: record.update(extra=True), "schema"),
        (lambda record: record.update(schema_version=True), "schema"),
        (lambda record: record.update(elapsed_seconds=-0.1), "elapsed"),
        (lambda record: record.update(elapsed_seconds=float("nan")), "elapsed"),
        (lambda record: record.update(started_at_utc="2026-08-31T10:00:00Z"), "timestamp"),
        (
            lambda record: record.update(ended_at_utc="2026-08-31T09:59:59.999999Z"),
            "chronology",
        ),
        (lambda record: record["run_provenance"].update(run_id="not-trusted"), "provenance"),
        (lambda record: record["before"].update(extra=True), "snapshot"),
        (
            lambda record: record["before"]["load_average"].update(value=[1.0, 2.0]),
            "load_average",
        ),
    ),
)
def test_record_validation_is_exact_and_fail_closed(mutation, message: str) -> None:
    """A wrong key, scalar type, numeric bound, chronology, or reason must fail validation."""
    host_state = _host_state()
    record = copy.deepcopy(_captured_record())
    mutation(record)

    with pytest.raises(ValueError, match=message):
        host_state.validate_host_state_record(record)


@pytest.mark.parametrize(
    "mutation",
    (
        lambda record: record.update(elapsed_seconds=_FloatSubclass(1.25)),
        lambda record: record.update(started_at_utc=_StringSubclass(record["started_at_utc"])),
        lambda record: record["before"]["load_average"].update(
            value=_ListSubclass([1.0, 2.0, 3.0])
        ),
        lambda record: record.update(
            run_provenance=_DictSubclass(record["run_provenance"])
        ),
        lambda record: record.update(before=_DictSubclass(record["before"])),
    ),
)
def test_record_validation_rejects_recursive_json_type_subclasses(mutation) -> None:
    """Every persisted JSON scalar/list/object must be an exact built-in type."""
    host_state = _host_state()
    record = copy.deepcopy(_captured_record())
    mutation(record)

    with pytest.raises(ValueError, match="exact built-in JSON"):
        host_state.validate_host_state_record(record)


def test_split_view_provenance_cannot_validate_before_round_trip_then_fail_after() -> None:
    """Validation and JSON serialization cannot observe different provenance objects."""
    host_state = _host_state()
    record = _captured_record()
    record["run_provenance"] = _SplitViewDict(
        _provenance().as_json(),
        {"schema_version": 99},
    )

    with pytest.raises(ValueError, match="exact built-in JSON"):
        host_state.validate_host_state_record(record)
    round_tripped = strict_json_loads(canonical_json_bytes(record))
    with pytest.raises(ValueError, match="provenance"):
        host_state.validate_host_state_record(round_tripped)


def test_validation_invokes_public_evidence_privacy_boundary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A privacy rejection, including identity/path/raw diagnostics, must fail the record."""
    host_state = _host_state()
    record = _captured_record()

    def reject(_value: object) -> None:
        raise ValueError("public evidence contains private raw stderr identity/path")

    monkeypatch.setattr(host_state, "validate_public_evidence", reject)

    with pytest.raises(ValueError, match="private raw stderr identity/path"):
        host_state.validate_host_state_record(record)


def test_validation_rejects_identity_inside_otherwise_valid_provenance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The shared privacy validator rejects local identity even after strict schema checks."""
    host_state = _host_state()
    privacy = importlib.import_module("hephaestus.privacy")
    record = _captured_record()
    record["run_provenance"]["run_id"] = "abcd" * 16
    monkeypatch.setattr(privacy.getpass, "getuser", lambda: "abcd")
    monkeypatch.setattr(privacy.socket, "gethostname", lambda: "safe-host")

    with pytest.raises(ValueError, match="username"):
        host_state.validate_host_state_record(record)


def test_validation_accepts_only_finite_nonnegative_metric_numbers() -> None:
    """All persisted successful numeric metric values are finite and nonnegative."""
    host_state = _host_state()
    record = _captured_record()

    host_state.validate_host_state_record(record)

    for snapshot_name in ("before", "after"):
        snapshot = record[snapshot_name]
        for metric_name in ("load_average",):
            value = snapshot[metric_name]["value"]
            values = value if isinstance(value, list) else [value]
            assert all(number >= 0 and math.isfinite(number) for number in values)

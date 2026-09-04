from __future__ import annotations

import json
from pathlib import Path

import pytest

import hephaestus.aa_runtime as aa_runtime_module
import hephaestus.demo as demo_module
import hephaestus.host_state as host_state_module
from hephaestus.aa_runtime import TrustedAAOrchestrator
from hephaestus.bundle import verify_manifest
from hephaestus.catalog import WorkloadName
from hephaestus.demo import TrustedDemoOrchestrator
from hephaestus.durability import prepare_attempt_output_root
from hephaestus.host_state import validate_host_state_record
from hephaestus.measure import RunSettings
from hephaestus.search import TrustedSearchOrchestrator
from hephaestus.workflows import run_catalog_action
from tests.test_aa_runtime import _AARunner
from tests.test_search_evidence import _BundleRunner
from tests.test_workflows import CAPABILITIES, CRITERIA, _EvidenceRunner


def _snapshot() -> dict[str, object]:
    return {
        "load_average": {"value": [0.1, 0.2, 0.3], "unavailable_reason": None},
    }


@pytest.fixture(autouse=True)
def _deterministic_host_samples(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(host_state_module, "sample_host_state", _snapshot)


def _rows(root: Path) -> list[dict[str, object]]:
    ledger = root / "host_state.jsonl"
    assert ledger.is_file() and not ledger.is_symlink()
    raw_lines = ledger.read_bytes().splitlines(keepends=True)
    assert raw_lines and all(line.endswith(b"\n") for line in raw_lines)
    rows = [json.loads(line) for line in raw_lines]
    for row in rows:
        validate_host_state_record(row)
    return rows


def _assert_no_nested_host_ledger(parent: Path) -> None:
    assert not tuple(parent.rglob("host_state.jsonl"))


def test_direct_run_places_one_host_row_only_at_supplied_outer_root(
    tmp_path: Path,
) -> None:
    """A direct child is recorded as runs/CHILD without entering its sealed bundle."""
    root = tmp_path / "direct"

    result = run_catalog_action(
        WorkloadName.MLP_STACK,
        "candidate-mlp-default",
        root,
        CRITERIA,
        capability_snapshot=CAPABILITIES,
        runner=_EvidenceRunner(),
    )

    rows = _rows(root)
    assert len(rows) == 1
    assert rows[0]["bundle_relative_path"] == "runs/child-1"
    assert rows[0]["run_provenance"]["sequence_index"] == 0
    assert rows[0]["run_provenance"] == json.loads(
        (result.bundle_path / "run_provenance.json").read_bytes()
    )
    assert not (root / "attempts.jsonl").exists()
    _assert_no_nested_host_ledger(result.bundle_path)
    assert verify_manifest(result.bundle_path).valid


def test_search_places_ordered_rows_at_outer_root_not_manifested_parent(
    tmp_path: Path,
) -> None:
    """Search records PARENT/runs/CHILD paths in proposal order outside its parent."""
    root = tmp_path / "search"
    search = TrustedSearchOrchestrator(
        root,
        CRITERIA,
        RunSettings(schema_version=2, repeats=64),
        capability_snapshot=CAPABILITIES,
        runner=_BundleRunner(
            (
                ("NOT_PROVEN", "graph.recompile_bound"),
                ("NOT_PROVEN", "perf.speedup_proven"),
                ("PROVEN", "all_criteria_passed"),
            )
        ),
    ).optimize(WorkloadName.DYNAMIC_BATCH_TEXT)

    rows = _rows(root)
    parent_name = search.parent_path.name
    assert [row["bundle_relative_path"] for row in rows] == [
        f"{parent_name}/runs/child-1",
        f"{parent_name}/runs/child-2",
        f"{parent_name}/runs/child-3",
    ]
    assert [row["run_provenance"]["sequence_index"] for row in rows] == [0, 1, 2]
    assert not (root / "attempts.jsonl").exists()
    _assert_no_nested_host_ledger(search.parent_path)
    assert verify_manifest(search.parent_path).valid


def test_aa_places_exactly_two_rows_at_outer_root_not_parent_or_children(
    tmp_path: Path,
) -> None:
    """Both A/A runs share one outer ledger and leave the sealed parent unchanged."""
    root = tmp_path / "aa"
    result = TrustedAAOrchestrator(
        root,
        CRITERIA,
        capability_snapshot=CAPABILITIES,
        runner=_AARunner(),
    ).run(WorkloadName.MLP_STACK)

    rows = _rows(root)
    parent_name = result.parent_path.name
    assert [row["bundle_relative_path"] for row in rows] == [
        f"{parent_name}/runs/child-1",
        f"{parent_name}/runs/child-2",
    ]
    assert [row["run_provenance"]["sequence_index"] for row in rows] == [0, 1]
    assert not (root / "attempts.jsonl").exists()
    _assert_no_nested_host_ledger(result.parent_path)
    assert verify_manifest(result.parent_path).valid


def test_demo_places_exactly_four_rows_at_outer_root_in_catalog_sequence(
    tmp_path: Path,
) -> None:
    """Three plants plus control produce four ordered host rows outside the demo tree."""
    root = tmp_path / "demo"
    result = TrustedDemoOrchestrator(
        root,
        CRITERIA,
        capability_snapshot=CAPABILITIES,
        runner=_EvidenceRunner(),
    ).run()

    rows = _rows(root)
    parent_name = result.parent_path.name
    assert [row["bundle_relative_path"] for row in rows] == [
        f"{parent_name}/runs/child-{index}" for index in range(1, 5)
    ]
    assert [row["run_provenance"]["sequence_index"] for row in rows] == [0, 1, 2, 3]
    assert not (root / "attempts.jsonl").exists()
    _assert_no_nested_host_ledger(result.parent_path)
    assert verify_manifest(result.parent_path).valid


def test_durable_aa_uses_preopened_attempt_sink_without_standalone_reopen(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A durable worker-owned root is the sole host sink for both child runs."""
    root = tmp_path / "durable"

    def forbidden_standalone(*args: object, **kwargs: object) -> object:
        del args, kwargs
        raise AssertionError("durable A/A reopened a standalone host writer")

    monkeypatch.setattr(
        aa_runtime_module,
        "prepare_host_state_output_root",
        forbidden_standalone,
        raising=False,
    )
    with prepare_attempt_output_root(root, allow_volatile_output=True) as attempt_root:
        result = TrustedAAOrchestrator(
            attempt_root.workflow_root,
            CRITERIA,
            capability_snapshot=CAPABILITIES,
            runner=_AARunner(),
            _host_state_sink=attempt_root,
        ).run(WorkloadName.MLP_STACK)

    rows = _rows(root)
    assert len(rows) == 2
    assert (root / "attempts.jsonl").read_bytes() == b""
    _assert_no_nested_host_ledger(result.parent_path)


def test_durable_demo_uses_preopened_attempt_sink_without_standalone_reopen(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A durable worker-owned root is the sole host sink for all four demo runs."""
    root = tmp_path / "durable-demo"

    def forbidden_standalone(*args: object, **kwargs: object) -> object:
        del args, kwargs
        raise AssertionError("durable demo reopened a standalone host writer")

    monkeypatch.setattr(
        demo_module,
        "prepare_host_state_output_root",
        forbidden_standalone,
    )
    with prepare_attempt_output_root(root, allow_volatile_output=True) as attempt_root:
        result = TrustedDemoOrchestrator(
            attempt_root.workflow_root,
            CRITERIA,
            capability_snapshot=CAPABILITIES,
            runner=_EvidenceRunner(),
            _host_state_sink=attempt_root,
        ).run()

    rows = _rows(root)
    assert len(rows) == 4
    assert (root / "attempts.jsonl").read_bytes() == b""
    _assert_no_nested_host_ledger(result.parent_path)


@pytest.mark.parametrize("workflow", ("aa", "demo"))
def test_nondurable_orchestrator_resolves_symlinked_outer_root_before_science(
    tmp_path: Path,
    workflow: str,
) -> None:
    """The child producer and pinned standalone sink must share one resolved root identity."""
    real_root = tmp_path / "real-output"
    real_root.mkdir()
    linked_root = tmp_path / "linked-output"
    linked_root.symlink_to(real_root, target_is_directory=True)

    if workflow == "aa":
        result = TrustedAAOrchestrator(
            linked_root,
            CRITERIA,
            capability_snapshot=CAPABILITIES,
            runner=_AARunner(),
        ).run(WorkloadName.MLP_STACK)
        expected_rows = 2
    else:
        result = TrustedDemoOrchestrator(
            linked_root,
            CRITERIA,
            capability_snapshot=CAPABILITIES,
            runner=_EvidenceRunner(),
        ).run()
        expected_rows = 4

    assert result.parent_path.is_relative_to(real_root)
    assert len(_rows(real_root)) == expected_rows
    _assert_no_nested_host_ledger(result.parent_path)

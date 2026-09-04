from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from types import MappingProxyType

import pytest

from hephaestus.bundle import write_json, write_manifest
from hephaestus.catalog import Proposal, WorkloadName, get_catalog_entry
from hephaestus.catalog_runtime import CatalogRuntime
from hephaestus.measure import RunResult, RunSettings
from hephaestus.provenance import RunProvenance


def _snapshot(**overrides: object) -> dict[str, object]:
    values: dict[str, object] = {
        "schema_version": 1,
        "torch_version": "2.13.0",
        "compiler_backends": ["inductor"],
        "inductor_modes": [
            "default",
            "max-autotune-no-cudagraphs",
            "reduce-overhead",
        ],
        "inductor_options": ["epilogue_fusion"],
    }
    values.update(overrides)
    return values


class _CapturingRunner:
    def __init__(self, verdict: str = "NOT_PROVEN", finding: str = "perf.speedup_proven"):
        self.calls: list[dict[str, object]] = []
        self.verdict = verdict
        self.finding = finding

    def __call__(
        self,
        workload_name: str,
        request: object,
        output_root: Path,
        criteria_path: Path,
        settings: RunSettings,
        *,
        input_plan: object = None,
        catalog_metadata: Mapping[str, object] | None = None,
        run_provenance: RunProvenance | None = None,
    ) -> RunResult:
        child = output_root / f"child-{len(self.calls) + 1}"
        child.mkdir(parents=True)
        assert run_provenance is not None
        write_json(child / "run_provenance.json", run_provenance.as_json())
        write_manifest(child)
        self.calls.append(
            {
                "workload_name": workload_name,
                "request": request,
                "output_root": output_root,
                "criteria_path": criteria_path,
                "settings": settings,
                "input_plan": input_plan,
                "catalog_metadata": catalog_metadata,
                "run_provenance": run_provenance,
            }
        )
        return RunResult(
            child,
            self.verdict,
            MappingProxyType(
                {"verdict": self.verdict, "driving_finding": self.finding}
            ),
        )


class _RecordingHostStateSink:
    def __init__(self) -> None:
        self.calls: list[tuple[object, Path]] = []

    def append(self, capture: object, bundle_path: Path) -> None:
        self.calls.append((capture, bundle_path))


def test_runtime_resolves_named_action_to_exact_request_plan_and_metadata(tmp_path: Path) -> None:
    """A proposal must be re-resolved into the reviewed Torch request and trusted plan."""
    runner = _CapturingRunner()
    settings = RunSettings()
    runs = tmp_path / "search" / "runs"
    runtime = CatalogRuntime(
        runs,
        tmp_path / "criteria.yaml",
        settings,
        _host_state_sink=_RecordingHostStateSink(),
        capability_snapshot=_snapshot(),
        runner=runner,
    )
    proposal = Proposal(
        "candidate-dynamic-bucketed",
        WorkloadName.DYNAMIC_BATCH_TEXT,
        "Try the frozen batch buckets while preserving sequence variation.",
    )

    result = runtime.run(proposal)

    assert len(runner.calls) == 1
    call = runner.calls[0]
    request = call["request"]
    assert request.backend == "inductor"  # type: ignore[attr-defined]
    assert request.mode is None  # type: ignore[attr-defined]
    assert request.dynamic is False  # type: ignore[attr-defined]
    assert request.fullgraph is True  # type: ignore[attr-defined]
    assert request.options is None  # type: ignore[attr-defined]
    assert request.disable is False  # type: ignore[attr-defined]
    plan = call["input_plan"]
    assert plan.evidence["dynamic_strategy"] == "bucketed"  # type: ignore[attr-defined]
    assert plan.evidence["bucket_boundaries"] == (2, 4)  # type: ignore[attr-defined]
    metadata = call["catalog_metadata"]
    assert metadata["entry_id"] == "candidate-dynamic-bucketed"  # type: ignore[index]
    assert metadata["requested"]["dynamic"] == "bucketed"  # type: ignore[index]
    assert metadata["effective"]["dynamic"] is False  # type: ignore[index]
    assert call["settings"] is settings
    assert result.bundle_relative_path == "runs/child-1"
    assert result.verdict == "NOT_PROVEN"
    assert result.driving_finding == "perf.speedup_proven"
    assert runtime.receipts[0].proposal == proposal
    assert runtime.receipts[0].result == result


def test_runtime_rejects_forged_proposal_before_harness_invocation(tmp_path: Path) -> None:
    """Mutating a frozen proposal by object internals must not bypass immediate revalidation."""
    runner = _CapturingRunner()
    runtime = CatalogRuntime(
        tmp_path / "search" / "runs",
        tmp_path / "criteria.yaml",
        RunSettings(),
        _host_state_sink=_RecordingHostStateSink(),
        capability_snapshot=_snapshot(),
        runner=runner,
    )
    proposal = Proposal(
        "candidate-mlp-default",
        WorkloadName.MLP_STACK,
        "Try the calibrated native default with a required full graph.",
    )
    object.__setattr__(proposal, "catalog_id", "forged-id")

    with pytest.raises(ValueError, match="unknown catalog"):
        runtime.run(proposal)

    assert runner.calls == []


def test_results_api_rejects_demo_roles_before_harness_invocation(tmp_path: Path) -> None:
    """The runtime object injected into an agent must not execute planted or control IDs."""
    runner = _CapturingRunner()
    runtime = CatalogRuntime(
        tmp_path / "search" / "runs",
        tmp_path / "criteria.yaml",
        RunSettings(),
        _host_state_sink=_RecordingHostStateSink(),
        capability_snapshot=_snapshot(),
        runner=runner,
    )
    entry = get_catalog_entry("planted-eager-fallback")
    proposal = Proposal(entry.catalog_id, entry.workload, entry.rationale)

    with pytest.raises(ValueError, match="candidate"):
        runtime.run(proposal)

    assert runner.calls == []


def test_separate_trusted_demo_path_admits_only_planted_or_control_roles(
    tmp_path: Path,
) -> None:
    """Demo authority must be explicit and must reject ordinary optimizer candidates."""
    runner = _CapturingRunner()
    runtime = CatalogRuntime(
        tmp_path / "demo" / "runs",
        tmp_path / "criteria.yaml",
        RunSettings(),
        _host_state_sink=_RecordingHostStateSink(),
        capability_snapshot=_snapshot(),
        runner=runner,
    )
    planted = get_catalog_entry("planted-eager-fallback")
    candidate = get_catalog_entry("candidate-mlp-default")

    result = runtime.run_demo(
        Proposal(planted.catalog_id, planted.workload, planted.rationale)
    )
    with pytest.raises(ValueError, match="demo"):
        runtime.run_demo(
            Proposal(candidate.catalog_id, candidate.workload, candidate.rationale)
        )

    assert result.bundle_relative_path == "runs/child-1"
    assert len(runner.calls) == 1


def test_runtime_rejects_missing_capability_before_harness_invocation(tmp_path: Path) -> None:
    """A stale capability snapshot must stop execution rather than silently remap a mode."""
    runner = _CapturingRunner()

    with pytest.raises(ValueError, match="mode"):
        CatalogRuntime(
            tmp_path / "search" / "runs",
            tmp_path / "criteria.yaml",
            RunSettings(),
            _host_state_sink=_RecordingHostStateSink(),
            capability_snapshot=_snapshot(inductor_modes=["default"]),
            runner=runner,
        )

    assert runner.calls == []


@pytest.mark.parametrize("forged", ["absolute", "parent", "backslash"])
def test_runtime_rejects_forged_runner_bundle_paths(tmp_path: Path, forged: str) -> None:
    """A harness result cannot escape, alias, or ambiguously name the trusted runs root."""
    runs = tmp_path / "search" / "runs"

    def runner(*args: object, **kwargs: object) -> RunResult:
        del args, kwargs
        runs.mkdir(parents=True, exist_ok=True)
        path = {
            "absolute": tmp_path / "outside",
            "parent": runs,
            "backslash": runs / "bad\\name",
        }[forged]
        path.mkdir(exist_ok=True)
        return RunResult(
            path,
            "PROVEN",
            MappingProxyType(
                {"verdict": "PROVEN", "driving_finding": "all_criteria_passed"}
            ),
        )

    runtime = CatalogRuntime(
        runs,
        tmp_path / "criteria.yaml",
        RunSettings(),
        _host_state_sink=_RecordingHostStateSink(),
        capability_snapshot=_snapshot(),
        runner=runner,
    )
    proposal = Proposal(
        "candidate-mlp-default",
        WorkloadName.MLP_STACK,
        "Try the calibrated native default with a required full graph.",
    )

    with pytest.raises(ValueError, match="path"):
        runtime.run(proposal)

    assert runtime.receipts == ()


def test_runtime_rejects_summary_without_scalar_driving_finding(tmp_path: Path) -> None:
    """A raw or malformed gate summary cannot be projected into an optimizer receipt."""
    runs = tmp_path / "search" / "runs"

    def runner(*args: object, **kwargs: object) -> RunResult:
        del args, kwargs
        child = runs / "child"
        child.mkdir(parents=True)
        return RunResult(child, "PROVEN", MappingProxyType({"verdict": "PROVEN"}))

    runtime = CatalogRuntime(
        runs,
        tmp_path / "criteria.yaml",
        RunSettings(),
        _host_state_sink=_RecordingHostStateSink(),
        capability_snapshot=_snapshot(),
        runner=runner,
    )

    with pytest.raises(ValueError, match="driving"):
        runtime.run(
            Proposal(
                "candidate-mlp-default",
                WorkloadName.MLP_STACK,
                "Try the calibrated native default with a required full graph.",
            )
        )

    assert runtime.receipts == ()


def test_runtime_rejects_duplicate_child_path_before_second_receipt(tmp_path: Path) -> None:
    """Two proposals cannot alias one child bundle and appear as independent evidence."""
    runs = tmp_path / "search" / "runs"
    calls = 0

    def runner(*args: object, **kwargs: object) -> RunResult:
        nonlocal calls
        del args
        calls += 1
        child = runs / "same-child"
        child.mkdir(parents=True, exist_ok=True)
        provenance = kwargs["run_provenance"]
        assert isinstance(provenance, RunProvenance)
        write_json(child / "run_provenance.json", provenance.as_json())
        write_manifest(child)
        verdict = "NOT_PROVEN"
        finding = "graph.recompile_bound"
        return RunResult(
            child,
            verdict,
            MappingProxyType({"verdict": verdict, "driving_finding": finding}),
        )

    runtime = CatalogRuntime(
        runs,
        tmp_path / "criteria.yaml",
        RunSettings(),
        _host_state_sink=_RecordingHostStateSink(),
        capability_snapshot=_snapshot(),
        runner=runner,
    )
    first = get_catalog_entry("candidate-dynamic-static")
    second = get_catalog_entry("candidate-dynamic-true")

    runtime.run(Proposal(first.catalog_id, first.workload, first.rationale))
    with pytest.raises(ValueError, match="duplicate"):
        runtime.run(Proposal(second.catalog_id, second.workload, second.rationale))

    assert calls == 2
    assert len(runtime.receipts) == 1


def test_runtime_requires_an_explicit_private_host_state_sink(tmp_path: Path) -> None:
    """Inferring an outer sink from runs_root.parent would grant undeclared path authority."""
    with pytest.raises(TypeError, match="_host_state_sink"):
        CatalogRuntime(
            tmp_path / "runs",
            tmp_path / "criteria.yaml",
            RunSettings(),
            capability_snapshot=_snapshot(),
            runner=_CapturingRunner(),
        )


def test_runtime_runner_failure_emits_no_completed_host_row_or_receipt(
    tmp_path: Path,
) -> None:
    """An ordinary runner exception remains incomplete operationally and scientifically."""
    sink = _RecordingHostStateSink()
    failure = RuntimeError("runner failed")

    def runner(*args: object, **kwargs: object) -> RunResult:
        del args, kwargs
        raise failure

    runtime = CatalogRuntime(
        tmp_path / "runs",
        tmp_path / "criteria.yaml",
        RunSettings(),
        _host_state_sink=sink,
        capability_snapshot=_snapshot(),
        runner=runner,
    )
    entry = get_catalog_entry("candidate-mlp-default")

    with pytest.raises(RuntimeError) as caught:
        runtime.run(Proposal(entry.catalog_id, entry.workload, entry.rationale))

    assert caught.value is failure
    assert sink.calls == []
    assert runtime.receipts == ()


@pytest.mark.parametrize(
    ("verdict", "finding"),
    (
        ("PROVEN", "all_criteria_passed"),
        ("CONDITIONAL", "graph.no_breaks"),
        ("NOT_PROVEN", "perf.speedup_proven"),
        ("INVALID_EVIDENCE", "accuracy.tolerance"),
    ),
)
def test_runtime_appends_host_state_for_every_normal_scientific_verdict(
    tmp_path: Path,
    verdict: str,
    finding: str,
) -> None:
    """Scientific sign does not control completion of the operational host-state row."""
    sink = _RecordingHostStateSink()
    runtime = CatalogRuntime(
        tmp_path / "runs",
        tmp_path / "criteria.yaml",
        RunSettings(),
        _host_state_sink=sink,
        capability_snapshot=_snapshot(),
        runner=_CapturingRunner(verdict, finding),
    )
    entry = get_catalog_entry("candidate-mlp-default")

    result = runtime.run(Proposal(entry.catalog_id, entry.workload, entry.rationale))

    assert result.verdict == verdict
    assert len(sink.calls) == 1
    assert len(runtime.receipts) == 1

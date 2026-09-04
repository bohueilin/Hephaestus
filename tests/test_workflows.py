from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import MappingProxyType

from hephaestus.bundle import finalize_bundle, write_json, write_manifest
from hephaestus.catalog import WorkloadName
from hephaestus.demo import TrustedDemoOrchestrator, verify_demo_tree
from hephaestus.gate import evaluate_bundle
from hephaestus.input_plan import input_plan_json
from hephaestus.measure import RunResult, RunSettings
from hephaestus.provenance import RunProvenance
from hephaestus.scope import EVIDENCE_BOUNDARY
from hephaestus.search import TrustedSearchOrchestrator, verify_search_tree
from hephaestus.workflows import run_catalog_action
from tests.evidence_helpers import write_normal_child

ROOT = Path(__file__).parents[1]
CRITERIA = ROOT / "gates" / "default.yaml"
CAPABILITIES = {
    "schema_version": 1,
    "torch_version": "2.13.0",
    "compiler_backends": ["inductor"],
    "inductor_modes": ["default", "max-autotune-no-cudagraphs", "reduce-overhead"],
    "inductor_options": ["epilogue_fusion"],
}


class _EvidenceRunner:
    def __init__(self, overrides: Mapping[str, str] | None = None) -> None:
        self.calls: list[dict[str, object]] = []
        self._overrides = {} if overrides is None else dict(overrides)

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
        del criteria_path
        assert catalog_metadata is not None
        catalog_id = catalog_metadata["entry_id"]
        assert isinstance(catalog_id, str)
        expected = {
            "clean-control-mlp": "all_criteria_passed",
            "planted-eager-fallback": "perf.speedup_proven",
            "planted-static-shape-storm": "graph.recompile_bound",
            "planted-graph-break-exposure": "graph.no_breaks",
            "candidate-mlp-default": "perf.speedup_proven",
        }[catalog_id]
        finding = self._overrides.get(catalog_id, expected)
        child = output_root / f"child-{len(self.calls) + 1}"
        child.mkdir(parents=True)
        self.calls.append(
            {
                "workload_name": workload_name,
                "request": request,
                "output_root": output_root,
                "settings": settings,
                "catalog_metadata": catalog_metadata,
            }
        )
        write_normal_child(
            child,
            workload_name=workload_name,
            request=request,
            plan=input_plan,
            metadata=catalog_metadata,
            driving_finding=finding,
            settings=settings,
            run_provenance=run_provenance,
        )
        verdict = json.loads((child / "verdict.json").read_bytes())
        return RunResult(child, verdict["verdict"], MappingProxyType(verdict))


def test_single_run_resolves_candidate_and_returns_real_child_with_frozen_settings(
    tmp_path: Path,
) -> None:
    """CLI-facing execution must use the canonical catalog action, not text-built flags."""
    runner = _EvidenceRunner()

    result = run_catalog_action(
        WorkloadName.MLP_STACK,
        "candidate-mlp-default",
        tmp_path,
        CRITERIA,
        capability_snapshot=CAPABILITIES,
        runner=runner,
    )

    assert len(runner.calls) == 1
    assert runner.calls[0]["settings"] == RunSettings(schema_version=2, repeats=64)
    assert runner.calls[0]["catalog_metadata"]["entry_id"] == "candidate-mlp-default"  # type: ignore[index]
    assert result.bundle_path == tmp_path.resolve() / "runs" / "child-1"
    assert result.verdict == "NOT_PROVEN"
    assert result.driving_finding == "perf.speedup_proven"


def test_single_run_uses_explicit_demo_route_for_demo_roles(tmp_path: Path) -> None:
    """A named planted action must never pass through candidate-only agent authority."""
    runner = _EvidenceRunner()

    result = run_catalog_action(
        WorkloadName.MLP_STACK,
        "planted-eager-fallback",
        tmp_path,
        CRITERIA,
        capability_snapshot=CAPABILITIES,
        runner=runner,
    )

    assert result.verdict == "NOT_PROVEN"
    assert runner.calls[0]["catalog_metadata"]["role"] == "planted"  # type: ignore[index]


def test_single_run_rejects_workload_mismatch_before_creating_artifacts(tmp_path: Path) -> None:
    """A valid ID paired with the wrong workload must fail before harness side effects."""
    runner = _EvidenceRunner()

    try:
        run_catalog_action(
            WorkloadName.TRANSFORMER_BLOCK,
            "candidate-mlp-default",
            tmp_path,
            CRITERIA,
            capability_snapshot=CAPABILITIES,
            runner=runner,
        )
    except ValueError as error:
        assert "workload" in str(error)
    else:
        raise AssertionError("mismatched catalog action was accepted")

    assert runner.calls == []
    assert not tmp_path.joinpath("runs").exists()


def test_demo_runs_three_plants_then_clean_control_and_seals_expected_rows(
    tmp_path: Path,
) -> None:
    """Changing demo order, role count, or expectation source must fail this contract."""
    runner = _EvidenceRunner()

    result = TrustedDemoOrchestrator(
        tmp_path,
        CRITERIA,
        capability_snapshot=CAPABILITIES,
        runner=runner,
    ).run()

    assert [row.catalog_id for row in result.rows] == [
        "planted-eager-fallback",
        "planted-static-shape-storm",
        "planted-graph-break-exposure",
        "clean-control-mlp",
    ]
    assert [row.expected_verdict for row in result.rows] == [
        "NOT_PROVEN",
        "NOT_PROVEN",
        "CONDITIONAL",
        "PROVEN",
    ]
    assert [row.expected_driving_finding for row in result.rows] == [
        "perf.speedup_proven",
        "graph.recompile_bound",
        "graph.no_breaks",
        "all_criteria_passed",
    ]
    assert [row.passed for row in result.rows] == [True, True, True, True]
    assert len(runner.calls) == 4
    assert all(
        call["settings"] == RunSettings(schema_version=2, repeats=64)
        for call in runner.calls
    )
    assert result.passed is True
    assert len({row.bundle_relative_path for row in result.rows}) == 4
    assert verify_demo_tree(result.parent_path) == result
    assert json.loads((result.parent_path / "scope.json").read_bytes()) == {
        "schema_version": 1,
        "boundary": EVIDENCE_BOUNDARY,
    }


def test_sequential_demo_runs_do_not_mutate_the_first_finalized_tree(tmp_path: Path) -> None:
    """Writing an attempts ledger into a manifested tree would change prior demo evidence."""
    first = TrustedDemoOrchestrator(
        tmp_path, CRITERIA, capability_snapshot=CAPABILITIES, runner=_EvidenceRunner()
    ).run()
    before = {
        path.relative_to(first.parent_path).as_posix(): path.read_bytes()
        for path in first.parent_path.rglob("*")
        if path.is_file()
    }

    TrustedDemoOrchestrator(
        tmp_path, CRITERIA, capability_snapshot=CAPABILITIES, runner=_EvidenceRunner()
    ).run()
    after = {
        path.relative_to(first.parent_path).as_posix(): path.read_bytes()
        for path in first.parent_path.rglob("*")
        if path.is_file()
    }

    assert after == before


def test_demo_tree_rejects_rehashed_child_provenance_aliasing(tmp_path: Path) -> None:
    """Four authentic rows cannot alias one run identity or reorder the runtime chain."""
    result = TrustedDemoOrchestrator(
        tmp_path,
        CRITERIA,
        capability_snapshot=CAPABILITIES,
        runner=_EvidenceRunner(),
    ).run()
    first_child = result.parent_path / result.rows[0].bundle_relative_path
    second_child = result.parent_path / result.rows[1].bundle_relative_path
    first_provenance = first_child / "run_provenance.json"
    second_provenance = second_child / "run_provenance.json"
    assert first_provenance.is_file() and second_provenance.is_file()
    cloned = json.loads(first_provenance.read_bytes())
    write_json(second_provenance, cloned)
    write_manifest(second_child)
    write_manifest(result.parent_path)

    checked = verify_demo_tree(result.parent_path)

    assert checked.passed is False
    assert any(mismatch.startswith("provenance:") for mismatch in checked.mismatches)
    manifest = json.loads((result.parent_path / "manifest.json").read_bytes())
    assert "catalog.json" in manifest["files"]
    assert "torch_capabilities.json" in manifest["files"]
    assert "scope.json" in manifest["files"]
    assert "runs/child-4/verdict.json" in manifest["files"]


def test_demo_mismatch_is_preserved_as_failed_evidence(tmp_path: Path) -> None:
    """A plausible-looking unexpected child result must make the demo fail, not tune it."""
    runner = _EvidenceRunner({"planted-eager-fallback": "all_criteria_passed"})

    result = TrustedDemoOrchestrator(
        tmp_path,
        CRITERIA,
        capability_snapshot=CAPABILITIES,
        runner=runner,
    ).run()

    assert result.passed is False
    assert result.rows[0].actual_verdict == "PROVEN"
    assert result.rows[0].passed is False


def test_demo_offline_verifier_rejects_rehashed_row_mismatch(tmp_path: Path) -> None:
    """Editing and rehashing demo.json cannot disconnect a row from its child."""
    result = TrustedDemoOrchestrator(
        tmp_path,
        CRITERIA,
        capability_snapshot=CAPABILITIES,
        runner=_EvidenceRunner(),
    ).run()
    path = result.parent_path / "demo.json"
    payload = json.loads(path.read_bytes())
    payload["rows"][0]["actual_verdict"] = "PROVEN"
    write_json(path, payload)
    write_manifest(result.parent_path)

    checked = verify_demo_tree(result.parent_path)

    assert checked.passed is False
    assert "row:actual_verdict" in checked.mismatches


def test_demo_expectations_are_frozen_against_rehashed_catalog_redefinition(
    tmp_path: Path,
) -> None:
    """Stored catalog edits cannot redefine a missed planted defect into a passing demo."""
    result = TrustedDemoOrchestrator(
        tmp_path,
        CRITERIA,
        capability_snapshot=CAPABILITIES,
        runner=_EvidenceRunner({"planted-eager-fallback": "all_criteria_passed"}),
    ).run()
    assert result.passed is False
    catalog_path = result.parent_path / "catalog.json"
    demo_path = result.parent_path / "demo.json"
    catalog = json.loads(catalog_path.read_bytes())
    planted = next(
        entry
        for entry in catalog["entries"]
        if entry["entry_id"] == "planted-eager-fallback"
    )
    planted["expected_verdict"] = "PROVEN"
    planted["expected_driving_finding"] = "all_criteria_passed"
    demo = json.loads(demo_path.read_bytes())
    demo["rows"][0]["expected_verdict"] = "PROVEN"
    demo["rows"][0]["expected_driving_finding"] = "all_criteria_passed"
    demo["rows"][0]["passed"] = True
    write_json(catalog_path, catalog)
    write_json(demo_path, demo)
    write_manifest(result.parent_path)

    checked = verify_demo_tree(result.parent_path)

    assert checked.passed is False
    assert "catalog:expectation" in checked.mismatches


def test_demo_complete_action_is_frozen_against_coherent_rehashed_tamper(
    tmp_path: Path,
) -> None:
    """A stored plant cannot disable its defect while retaining the frozen v1 ID."""
    result = TrustedDemoOrchestrator(
        tmp_path,
        CRITERIA,
        capability_snapshot=CAPABILITIES,
        runner=_EvidenceRunner(),
    ).run()
    catalog_path = result.parent_path / "catalog.json"
    catalog = json.loads(catalog_path.read_bytes())
    entry = next(
        item
        for item in catalog["entries"]
        if item["entry_id"] == "planted-eager-fallback"
    )
    entry["requested"]["disable"] = False
    entry["effective"]["disable"] = False
    write_json(catalog_path, catalog)

    child = result.parent_path / result.rows[0].bundle_relative_path
    config_path = child / "config.json"
    config = json.loads(config_path.read_bytes())
    config["disable"] = False
    config["catalog"]["requested"]["disable"] = False
    config["catalog"]["effective"]["disable"] = False
    write_json(config_path, config)
    write_manifest(child)
    write_manifest(result.parent_path)

    checked = verify_demo_tree(result.parent_path)

    assert checked.passed is False
    assert "catalog:action" in checked.mismatches


def test_demo_strictly_rejects_unexpected_selected_entry_and_capability_keys(
    tmp_path: Path,
) -> None:
    """Unknown schema fields cannot extend stored demo or capability authority."""
    result = TrustedDemoOrchestrator(
        tmp_path,
        CRITERIA,
        capability_snapshot=CAPABILITIES,
        runner=_EvidenceRunner(),
    ).run()
    capability_path = result.parent_path / "torch_capabilities.json"
    capabilities = json.loads(capability_path.read_bytes())
    capabilities["unexpected"] = True
    write_json(capability_path, capabilities)
    capability_bytes = capability_path.read_bytes()

    catalog_path = result.parent_path / "catalog.json"
    catalog = json.loads(catalog_path.read_bytes())
    catalog["torch_capabilities_sha256"] = hashlib.sha256(capability_bytes).hexdigest()
    entry = next(
        item
        for item in catalog["entries"]
        if item["entry_id"] == "planted-eager-fallback"
    )
    entry["unexpected"] = True
    write_json(catalog_path, catalog)
    write_manifest(result.parent_path)

    checked = verify_demo_tree(result.parent_path)

    assert checked.passed is False
    assert "schema:invalid" in checked.mismatches


def test_demo_schema_and_scope_reject_boolean_version_true(tmp_path: Path) -> None:
    """Demo schema and scope versions must be integers, not equal-valued booleans."""
    result = TrustedDemoOrchestrator(
        tmp_path,
        CRITERIA,
        capability_snapshot=CAPABILITIES,
        runner=_EvidenceRunner(),
    ).run()
    demo_path = result.parent_path / "demo.json"
    demo = json.loads(demo_path.read_bytes())
    demo["schema_version"] = True
    write_json(demo_path, demo)
    scope_path = result.parent_path / "scope.json"
    scope = json.loads(scope_path.read_bytes())
    scope["schema_version"] = True
    write_json(scope_path, scope)
    write_manifest(result.parent_path)

    checked = verify_demo_tree(result.parent_path)

    assert checked.passed is False
    assert "schema:invalid" in checked.mismatches
    assert "scope:invalid" in checked.mismatches


def test_search_parent_manifests_scope_and_rejects_rehashed_scope_tamper(
    tmp_path: Path,
) -> None:
    """Agent evidence must carry the same machine-local boundary as every CLI surface."""
    search = TrustedSearchOrchestrator(
        tmp_path,
        CRITERIA,
        RunSettings(schema_version=2, repeats=64),
        capability_snapshot=CAPABILITIES,
        runner=_EvidenceRunner(
            {"candidate-mlp-default": "all_criteria_passed"}
        ),
    ).optimize(WorkloadName.MLP_STACK)

    assert json.loads((search.parent_path / "scope.json").read_bytes()) == {
        "schema_version": 1,
        "boundary": EVIDENCE_BOUNDARY,
    }
    manifest = json.loads((search.parent_path / "manifest.json").read_bytes())
    assert "scope.json" in manifest["files"]
    write_json(search.parent_path / "scope.json", {"schema_version": 1, "boundary": "forged"})
    write_manifest(search.parent_path)

    integrity = verify_search_tree(search.parent_path)

    assert integrity.valid is False
    assert "scope:invalid" in integrity.mismatches


def test_search_scope_rejects_boolean_schema_version(tmp_path: Path) -> None:
    """Search scope JSON true must not alias integer schema version one."""
    search = TrustedSearchOrchestrator(
        tmp_path,
        CRITERIA,
        RunSettings(schema_version=2, repeats=64),
        capability_snapshot=CAPABILITIES,
        runner=_EvidenceRunner(
            {"candidate-mlp-default": "all_criteria_passed"}
        ),
    ).optimize(WorkloadName.MLP_STACK)
    write_json(
        search.parent_path / "scope.json",
        {"schema_version": True, "boundary": EVIDENCE_BOUNDARY},
    )
    write_manifest(search.parent_path)

    integrity = verify_search_tree(search.parent_path)

    assert integrity.valid is False
    assert "scope:invalid" in integrity.mismatches


def _write_child(
    child: Path,
    *,
    workload_name: str,
    request: object,
    plan: object,
    metadata: Mapping[str, object],
    driving_finding: str,
) -> None:
    plan_payload = input_plan_json(plan)  # type: ignore[arg-type]
    case_indices = plan_payload["compile_sweep_case_indices"]
    assert isinstance(case_indices, tuple)
    case_count = len(case_indices)
    repeats = 31
    origin = datetime(2026, 8, 30, 12, 0, tzinfo=UTC)

    def timestamp(offset: int) -> str:
        return (origin + timedelta(seconds=offset)).isoformat().replace("+00:00", "Z")

    eager_timestamps: list[str] = []
    baseline_timestamps: list[str] = []
    candidate_timestamps: list[str] = []
    for iteration in range(repeats):
        offset = case_count + iteration * 3
        if iteration % 2 == 0:
            eager_timestamps.append(timestamp(offset))
            baseline_timestamps.append(timestamp(offset + 1))
            candidate_timestamps.append(timestamp(offset + 2))
        else:
            candidate_timestamps.append(timestamp(offset))
            baseline_timestamps.append(timestamp(offset + 1))
            eager_timestamps.append(timestamp(offset + 2))

    compiled_seconds = (
        [1.0] * repeats
        if driving_finding == "perf.speedup_proven"
        else [0.8] * repeats
    )
    write_json(child / "env.json", {"boundary": EVIDENCE_BOUNDARY})
    write_json(child / "workload.digest", {"name": workload_name, "sha256": "0" * 64})
    write_json(
        child / "config.json",
        {
            "backend": request.backend,  # type: ignore[attr-defined]
            "mode": request.mode,  # type: ignore[attr-defined]
            "dynamic": request.dynamic,  # type: ignore[attr-defined]
            "fullgraph": request.fullgraph,  # type: ignore[attr-defined]
            "options": None if request.options is None else dict(request.options),  # type: ignore[attr-defined]
            "disable": request.disable,  # type: ignore[attr-defined]
            "catalog": metadata,
        },
    )
    write_json(child / "input_plan.json", plan_payload)
    write_json(
        child / "timings.json",
        {
            "eager_seconds": [1.0] * repeats,
            "eager_timestamps_utc": eager_timestamps,
            "compiled_seconds": compiled_seconds,
            "compiled_timestamps_utc": baseline_timestamps,
            "cold_compile_seconds": 1.0,
            "cold_compile_timestamp_utc": timestamp(0),
            "non_primary_compile_sweep_case_indices": list(range(1, case_count)),
            "non_primary_compile_sweep_seconds": [0.1] * (case_count - 1),
            "non_primary_compile_sweep_timestamps_utc": [
                timestamp(index) for index in range(1, case_count)
            ],
            "aa_baseline_seconds": compiled_seconds,
            "aa_baseline_timestamps_utc": baseline_timestamps,
            "aa_candidate_seconds": compiled_seconds,
            "aa_candidate_timestamps_utc": candidate_timestamps,
            "aa_signed_paired_effects": [0.0] * repeats,
            "aa_bootstrap_absolute_medians": [0.0] * 2000,
            "summary": {},
        },
    )
    write_json(
        child / "accuracy.json",
        {
            "schema_version": 1,
            "dtype": "torch.float32",
            "within_tolerance": True,
            "case_index": 0,
            "atol": 1e-5,
            "rtol": 1e-5,
            "max_absolute_error": 0.0,
            "mismatch": None,
            "cases": [
                {
                    "case_index": index,
                    "within_tolerance": True,
                    "max_absolute_error": 0.0,
                    "mismatch": None,
                }
                for index in range(case_count)
            ],
        },
    )
    recompiles = (
        [{"trigger": f"shape guard {index}"} for index in range(3)]
        if driving_finding == "graph.recompile_bound"
        else []
    )
    graph_breaks = (
        [{"reason": "Tensor.item"}]
        if driving_finding == "graph.no_breaks"
        else []
    )
    write_json(
        child / "dynamo_report.json",
        {"graph_breaks": graph_breaks, "recompiles": recompiles},
    )
    write_json(
        child / "methodology.json",
        {
            "valid": True,
            "warmup_runs": 5,
            "repeats": repeats,
            "bootstrap_samples": 2000,
            "bootstrap_seed": 0,
            "bootstrap_confidence": 0.95,
            "inter_run_spacing_seconds": 0.0,
            "aa_noise_floor": 0.0,
            "aa_effect_formula": "(A-B)/((A+B)/2)",
            "aa_estimator": "p95_absolute_bootstrap_median",
            "aa_pairing": "within_iteration",
            "measurement_schedule": "alternate_eager-A-B__B-A-eager",
            "quantile_method": "linear_interpolation",
        },
    )
    (child / "gate_criteria.yaml").write_bytes(CRITERIA.read_bytes())
    finalize_bundle(child, evaluate_bundle)

from __future__ import annotations

import hashlib
import json
import shutil
from collections.abc import Mapping
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import MappingProxyType

import pytest

import hephaestus.catalog as catalog_module
import hephaestus.search as search_module
from hephaestus.agent import AgentObservation
from hephaestus.bundle import (
    canonical_json_bytes,
    finalize_bundle,
    verify_manifest,
    write_json,
    write_manifest,
)
from hephaestus.catalog import (
    ReadOnlyRunResult,
    WorkloadName,
    catalog_metadata,
    get_catalog_entry,
)
from hephaestus.catalog_runtime import CatalogRuntime
from hephaestus.gate import evaluate_bundle
from hephaestus.input_plan import input_plan_json
from hephaestus.measure import RunResult, RunSettings
from hephaestus.provenance import RunProvenance
from hephaestus.search import (
    SearchRunResult,
    SearchStep,
    SearchTranscript,
    TrustedSearchOrchestrator,
    finalize_search_tree,
    verify_search_tree,
)
from tests.evidence_helpers import write_normal_child

CAPABILITIES = {
    "schema_version": 1,
    "torch_version": "2.13.0",
    "compiler_backends": ["inductor"],
    "inductor_modes": ["default", "max-autotune-no-cudagraphs", "reduce-overhead"],
    "inductor_options": ["epilogue_fusion"],
}


def test_orchestrator_feeds_only_scalar_observations_to_inert_policy(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Trusted orchestration may feed policy only verdict/finding scalar observations."""
    observed: list[tuple[AgentObservation, ...]] = []
    real_transition = search_module.next_proposal

    def inspecting_transition(policy: object, observations: object) -> object:
        assert isinstance(observations, tuple)
        assert all(isinstance(item, AgentObservation) for item in observations)
        observed.append(observations)
        return real_transition(policy, observations)  # type: ignore[arg-type]

    monkeypatch.setattr(search_module, "next_proposal", inspecting_transition)
    search_module.TrustedSearchOrchestrator(
        tmp_path,
        tmp_path / "criteria.yaml",
        RunSettings(schema_version=2, repeats=64),
        capability_snapshot=CAPABILITIES,
        runner=_BundleRunner((('PROVEN', 'all_criteria_passed'),)),
    ).optimize(WorkloadName.MLP_STACK)

    assert observed[0] == ()
    assert observed[-1] == (AgentObservation("PROVEN", "all_criteria_passed"),)


class _BundleRunner:
    def __init__(self, outcomes: tuple[tuple[str, str], ...]) -> None:
        self._outcomes = iter(outcomes)
        self.count = 0

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
        self.count += 1
        expected_verdict, expected_finding = next(self._outcomes)
        child = output_root / f"child-{self.count}"
        child.mkdir(parents=True)
        assert catalog_metadata is not None
        write_normal_child(
            child,
            workload_name=workload_name,
            request=request,
            plan=input_plan,
            metadata=catalog_metadata,
            driving_finding=expected_finding,
            settings=settings,
            run_provenance=run_provenance,
        )
        verdict = json.loads((child / "verdict.json").read_bytes())
        assert verdict["verdict"] == expected_verdict
        assert verdict["driving_finding"] == expected_finding
        return RunResult(
            child,
            expected_verdict,
            MappingProxyType(verdict),
        )


def _bucketed_search(tmp_path: Path) -> SearchRunResult:
    return TrustedSearchOrchestrator(
        tmp_path,
        tmp_path / "criteria.yaml",
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


def test_search_tree_rejects_rehashed_child_provenance_reordering(tmp_path: Path) -> None:
    """Stored transcript order must be bound to distinct chained runtime-issued children."""
    search = _bucketed_search(tmp_path)
    children = [
        search.parent_path / step.result.bundle_relative_path
        for step in search.transcript.steps
    ]
    provenance_paths = [child / "run_provenance.json" for child in children]
    assert all(path.is_file() for path in provenance_paths)
    second = json.loads(provenance_paths[1].read_bytes())
    second["sequence_index"] = 0
    write_json(provenance_paths[1], second)
    write_manifest(children[1])
    write_manifest(search.parent_path)

    verification = verify_search_tree(search.parent_path)

    assert verification.valid is False
    assert any(mismatch.startswith("provenance:") for mismatch in verification.mismatches)


def test_trusted_search_preserves_refusal_then_proven_in_recursive_tree(tmp_path: Path) -> None:
    """Search evidence must contain exact runtime receipts and every child bundle byte."""
    runner = _BundleRunner(
        (
            ("NOT_PROVEN", "graph.recompile_bound"),
            ("PROVEN", "all_criteria_passed"),
        )
    )
    orchestrator = TrustedSearchOrchestrator(
        tmp_path,
        tmp_path / "criteria.yaml",
        RunSettings(schema_version=2, repeats=64),
        capability_snapshot=CAPABILITIES,
        runner=runner,
    )

    search = orchestrator.optimize(WorkloadName.DYNAMIC_BATCH_TEXT)

    assert [step.proposal.catalog_id for step in search.transcript.steps] == [
        "candidate-dynamic-static",
        "candidate-dynamic-true",
    ]
    assert [step.result.verdict for step in search.transcript.steps] == [
        "NOT_PROVEN",
        "PROVEN",
    ]
    assert search.final_result == search.transcript.steps[-1].result
    assert verify_search_tree(search.parent_path).valid is True
    assert verify_manifest(search.parent_path).valid is True
    manifest = json.loads((search.parent_path / "manifest.json").read_bytes())
    assert "catalog.json" in manifest["files"]
    assert "torch_capabilities.json" in manifest["files"]
    assert "transcript.json" in manifest["files"]
    assert "runs/child-1/manifest.json" in manifest["files"]
    assert "runs/child-1/verdict.json" in manifest["files"]
    assert "runs/child-2/timings.json" in manifest["files"]
    catalog = json.loads((search.parent_path / "catalog.json").read_bytes())
    capability_bytes = (search.parent_path / "torch_capabilities.json").read_bytes()
    assert catalog["torch_capabilities_sha256"] == hashlib.sha256(
        capability_bytes
    ).hexdigest()


def test_finalization_rejects_transcript_that_differs_from_private_receipts(
    tmp_path: Path,
) -> None:
    """A caller cannot replace the optimizer trace while retaining authentic child bundles."""
    parent = tmp_path / "search"
    runs = parent / "runs"
    runner = _BundleRunner((('PROVEN', 'all_criteria_passed'),))
    runtime = CatalogRuntime(
        runs,
        tmp_path / "criteria.yaml",
        RunSettings(schema_version=2, repeats=64),
        _host_state_sink=type(
            "HostSink",
            (),
            {"append": lambda self, capture, bundle_path: None},
        )(),
        capability_snapshot=CAPABILITIES,
        runner=runner,
    )
    entry = get_catalog_entry("candidate-dynamic-static")
    proposal = catalog_module.Proposal(entry.catalog_id, entry.workload, entry.rationale)
    authentic_result = runtime.run(proposal)
    transcript = SearchTranscript(
        WorkloadName.DYNAMIC_BATCH_TEXT,
        (SearchStep(proposal, authentic_result),),
    )
    authentic = transcript.steps[0]
    forged = SearchTranscript(
        transcript.workload_name,
        (
            SearchStep(
                authentic.proposal,
                ReadOnlyRunResult(
                    authentic.result.bundle_relative_path,
                    "NOT_PROVEN",
                    "perf.speedup_proven",
                ),
            ),
        ),
    )

    with pytest.raises(ValueError, match="receipt"):
        finalize_search_tree(parent, forged, runtime.receipts)


def test_offline_verifier_rejects_child_verdict_disagreement_even_after_rehash(
    tmp_path: Path,
) -> None:
    """Rehashing a forged child verdict cannot make it agree with the stored transcript."""
    search = TrustedSearchOrchestrator(
        tmp_path,
        tmp_path / "criteria.yaml",
        RunSettings(schema_version=2, repeats=64),
        capability_snapshot=CAPABILITIES,
        runner=_BundleRunner((('PROVEN', 'all_criteria_passed'),)),
    ).optimize(WorkloadName.MLP_STACK)
    child = search.parent_path / search.transcript.steps[0].result.bundle_relative_path
    write_json(
        child / "verdict.json",
        {"verdict": "NOT_PROVEN", "driving_finding": "perf.speedup_proven", "findings": []},
    )
    write_manifest(child)
    write_manifest(search.parent_path)

    integrity = verify_search_tree(search.parent_path)

    assert integrity.valid is False
    assert "receipt:verdict" in integrity.mismatches


def test_offline_verifier_regates_semantically_tampered_child_after_rehash(
    tmp_path: Path,
) -> None:
    """Rehashing altered child evidence must not bypass the pure stored-evidence gate."""
    search = TrustedSearchOrchestrator(
        tmp_path,
        tmp_path / "criteria.yaml",
        RunSettings(schema_version=2, repeats=64),
        capability_snapshot=CAPABILITIES,
        runner=_BundleRunner((('PROVEN', 'all_criteria_passed'),)),
    ).optimize(WorkloadName.MLP_STACK)
    child = search.parent_path / search.transcript.steps[0].result.bundle_relative_path
    write_json(child / "config.json", {"backend": "forged"})
    write_manifest(child)
    write_manifest(search.parent_path)

    integrity = verify_search_tree(search.parent_path)

    assert integrity.valid is False
    assert "child:gate_verdict" in integrity.mismatches


def test_offline_verifier_binds_child_catalog_metadata_to_transcript_proposal(
    tmp_path: Path,
) -> None:
    """A child cannot substitute a planted role/ID for the candidate named by its step."""
    search = TrustedSearchOrchestrator(
        tmp_path,
        tmp_path / "criteria.yaml",
        RunSettings(schema_version=2, repeats=64),
        capability_snapshot=CAPABILITIES,
        runner=_BundleRunner(
            (
                ("NOT_PROVEN", "graph.recompile_bound"),
                ("PROVEN", "all_criteria_passed"),
            )
        ),
    ).optimize(WorkloadName.DYNAMIC_BATCH_TEXT)
    first = search.transcript.steps[0]
    child = search.parent_path / first.result.bundle_relative_path
    config = json.loads((child / "config.json").read_bytes())
    config["catalog"] = catalog_metadata(get_catalog_entry("planted-static-shape-storm"))
    write_json(child / "config.json", config)
    write_manifest(child)
    write_manifest(search.parent_path)

    integrity = verify_search_tree(search.parent_path)

    assert integrity.valid is False
    assert "receipt:catalog_metadata" in integrity.mismatches


def test_offline_verifier_binds_catalog_capability_digest_to_parent_copy(
    tmp_path: Path,
) -> None:
    """A rehashed capability copy must still match the digest recorded by catalog evidence."""
    search = TrustedSearchOrchestrator(
        tmp_path,
        tmp_path / "criteria.yaml",
        RunSettings(schema_version=2, repeats=64),
        capability_snapshot=CAPABILITIES,
        runner=_BundleRunner((('PROVEN', 'all_criteria_passed'),)),
    ).optimize(WorkloadName.MLP_STACK)
    write_json(search.parent_path / "torch_capabilities.json", {"schema_version": 1})
    write_manifest(search.parent_path)

    integrity = verify_search_tree(search.parent_path)

    assert integrity.valid is False
    assert "catalog:capability_digest" in integrity.mismatches


def test_offline_verifier_rejects_unexpected_child_file_after_parent_rehash(
    tmp_path: Path,
) -> None:
    """Parent recursive hashing must not bless a file absent from the child manifest."""
    search = TrustedSearchOrchestrator(
        tmp_path,
        tmp_path / "criteria.yaml",
        RunSettings(schema_version=2, repeats=64),
        capability_snapshot=CAPABILITIES,
        runner=_BundleRunner((('PROVEN', 'all_criteria_passed'),)),
    ).optimize(WorkloadName.MLP_STACK)
    child = search.parent_path / search.transcript.steps[0].result.bundle_relative_path
    write_json(child / "unexpected.json", {"forged": True})
    write_manifest(search.parent_path)

    integrity = verify_search_tree(search.parent_path)

    assert integrity.valid is False
    assert "child:unexpected:unexpected.json" in integrity.mismatches


def test_offline_verifier_rejects_duplicate_transcript_child_paths_before_sets(
    tmp_path: Path,
) -> None:
    """Four refusal steps cannot alias one child and delete the other manifested runs."""
    search = TrustedSearchOrchestrator(
        tmp_path,
        tmp_path / "criteria.yaml",
        RunSettings(schema_version=2, repeats=64),
        capability_snapshot=CAPABILITIES,
        runner=_BundleRunner((('NOT_PROVEN', 'perf.speedup_proven'),) * 4),
    ).optimize(WorkloadName.MLP_STACK)
    transcript_path = search.parent_path / "transcript.json"
    transcript = json.loads(transcript_path.read_bytes())
    original_paths = [
        step["result"]["bundle_relative_path"] for step in transcript["steps"]
    ]
    first_result = dict(transcript["steps"][0]["result"])
    for step in transcript["steps"][1:]:
        step["result"] = dict(first_result)
    write_json(transcript_path, transcript)
    for relative_path in original_paths[1:]:
        shutil.rmtree(search.parent_path / relative_path)
    write_manifest(search.parent_path)

    integrity = verify_search_tree(search.parent_path)

    assert integrity.valid is False
    assert "transcript:duplicate_bundle_path" in integrity.mismatches


def test_offline_verifier_uses_manifested_catalog_after_live_catalog_evolves(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Intact stored evidence must remain valid when the installed live catalog changes."""
    search = TrustedSearchOrchestrator(
        tmp_path,
        tmp_path / "criteria.yaml",
        RunSettings(schema_version=2, repeats=64),
        capability_snapshot=CAPABILITIES,
        runner=_BundleRunner((('PROVEN', 'all_criteria_passed'),)),
    ).optimize(WorkloadName.MLP_STACK)

    def fail_live_lookup(*args: object, **kwargs: object) -> object:
        del args, kwargs
        raise AssertionError("offline verification consulted the live catalog")

    monkeypatch.setattr(catalog_module, "get_catalog_entry", fail_live_lookup)
    monkeypatch.setattr(search_module, "catalog_json", fail_live_lookup)

    integrity = verify_search_tree(search.parent_path)

    assert integrity.valid is True


def test_offline_verifier_rejects_unauthorized_role_in_manifested_catalog(
    tmp_path: Path,
) -> None:
    """Stored catalog evolution cannot grant a planted role to an optimizer transcript."""
    search = TrustedSearchOrchestrator(
        tmp_path,
        tmp_path / "criteria.yaml",
        RunSettings(schema_version=2, repeats=64),
        capability_snapshot=CAPABILITIES,
        runner=_BundleRunner((('PROVEN', 'all_criteria_passed'),)),
    ).optimize(WorkloadName.MLP_STACK)
    catalog_path = search.parent_path / "catalog.json"
    catalog = json.loads(catalog_path.read_bytes())
    entry = next(
        item for item in catalog["entries"] if item["entry_id"] == "candidate-mlp-default"
    )
    entry["role"] = "planted"
    entry["expected_verdict"] = "NOT_PROVEN"
    entry["expected_driving_finding"] = "perf.speedup_proven"
    write_json(catalog_path, catalog)
    write_manifest(search.parent_path)

    integrity = verify_search_tree(search.parent_path)

    assert integrity.valid is False
    assert "catalog:action" in integrity.mismatches


def test_offline_search_rejects_coherently_rehashed_candidate_action(
    tmp_path: Path,
) -> None:
    """Stored catalog authority cannot redefine a frozen v1 candidate action."""
    search = TrustedSearchOrchestrator(
        tmp_path,
        tmp_path / "criteria.yaml",
        RunSettings(schema_version=2, repeats=64),
        capability_snapshot=CAPABILITIES,
        runner=_BundleRunner((('PROVEN', 'all_criteria_passed'),)),
    ).optimize(WorkloadName.MLP_STACK)
    catalog_path = search.parent_path / "catalog.json"
    catalog = json.loads(catalog_path.read_bytes())
    entry = next(
        item for item in catalog["entries"] if item["entry_id"] == "candidate-mlp-default"
    )
    entry["requested"]["mode"] = "reduce-overhead"
    entry["effective"]["mode"] = "reduce-overhead"
    write_json(catalog_path, catalog)

    child = search.parent_path / search.transcript.steps[0].result.bundle_relative_path
    config_path = child / "config.json"
    config = json.loads(config_path.read_bytes())
    config["mode"] = "reduce-overhead"
    config["catalog"]["requested"]["mode"] = "reduce-overhead"
    config["catalog"]["effective"]["mode"] = "reduce-overhead"
    write_json(config_path, config)
    write_manifest(child)
    assert (child / "verdict.json").read_bytes() == canonical_json_bytes(
        evaluate_bundle(child)
    )
    write_manifest(search.parent_path)

    integrity = verify_search_tree(search.parent_path)

    assert integrity.valid is False
    assert "catalog:action" in integrity.mismatches


def test_offline_search_strictly_binds_child_catalog_action_types(
    tmp_path: Path,
) -> None:
    """Float bucket boundaries cannot impersonate the frozen integer child action."""
    search = _bucketed_search(tmp_path)
    child = search.parent_path / search.transcript.steps[-1].result.bundle_relative_path
    config_path = child / "config.json"
    config = json.loads(config_path.read_bytes())
    config["catalog"]["requested"]["bucket_policy"]["boundaries"] = [2.0, 4.0]
    write_json(config_path, config)
    write_manifest(child)
    write_manifest(search.parent_path)

    integrity = verify_search_tree(search.parent_path)

    assert integrity.valid is False
    assert "receipt:catalog_metadata" in integrity.mismatches


def test_offline_search_rejects_float_alias_in_bucketed_input_plan(
    tmp_path: Path,
) -> None:
    """A rehashed bucket plan still requires exact integer boundary types."""
    search = _bucketed_search(tmp_path)
    child = search.parent_path / search.transcript.steps[-1].result.bundle_relative_path
    plan_path = child / "input_plan.json"
    plan = json.loads(plan_path.read_bytes())
    plan["bucket_boundaries"] = [2.0, 4.0]
    write_json(plan_path, plan)
    write_manifest(child)
    write_manifest(search.parent_path)

    integrity = verify_search_tree(search.parent_path)

    assert integrity.valid is False
    assert "child:gate_verdict" in integrity.mismatches


def test_public_search_rejects_nondefault_settings_before_orchestration(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The public search surface cannot override frozen measurement methodology."""

    def forbidden_orchestrator(*args: object, **kwargs: object) -> object:
        del args, kwargs
        raise AssertionError("orchestration started before settings validation")

    monkeypatch.setattr(
        search_module,
        "TrustedSearchOrchestrator",
        forbidden_orchestrator,
    )

    with pytest.raises(ValueError, match="frozen"):
        search_module.run_scripted_search(
            WorkloadName.MLP_STACK,
            tmp_path,
            tmp_path / "criteria.yaml",
            RunSettings(schema_version=2, warmup_runs=0, repeats=64),
        )

    assert not any(tmp_path.iterdir())


def test_public_search_defaults_to_exact_schema_v2_64_settings(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The public default cannot silently retain the superseded 32-repeat contract."""
    captured: list[object] = []
    expected = object()

    class RecordingOrchestrator:
        def __init__(
            self,
            output_root: Path,
            criteria_path: Path,
            settings: RunSettings,
        ) -> None:
            del output_root, criteria_path
            captured.append(settings)

        def optimize(self, workload_name: WorkloadName) -> object:
            captured.append(workload_name)
            return expected

    monkeypatch.setattr(
        search_module,
        "TrustedSearchOrchestrator",
        RecordingOrchestrator,
    )

    result = search_module.run_scripted_search(
        WorkloadName.MLP_STACK,
        tmp_path,
        tmp_path / "criteria.yaml",
    )

    assert result is expected
    assert captured == [
        RunSettings(schema_version=2, repeats=64),
        WorkloadName.MLP_STACK,
    ]


def test_public_search_rejects_superseded_schema_v2_32_settings(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An otherwise-frozen schema-v2 settings object with 32 repeats is not authoritative."""
    monkeypatch.setattr(
        search_module,
        "TrustedSearchOrchestrator",
        lambda *args, **kwargs: pytest.fail("orchestration started"),
    )

    with pytest.raises(ValueError, match="frozen"):
        search_module.run_scripted_search(
            WorkloadName.MLP_STACK,
            tmp_path,
            tmp_path / "criteria.yaml",
            RunSettings(schema_version=2, repeats=32),
        )

    assert not any(tmp_path.iterdir())


def test_public_search_rejects_equal_numeric_settings_alias_before_orchestration(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Integer zero cannot enter orchestration as the frozen float spacing setting."""

    def forbidden_orchestrator(*args: object, **kwargs: object) -> object:
        del args, kwargs
        raise AssertionError("orchestration started before strict settings validation")

    monkeypatch.setattr(
        search_module,
        "TrustedSearchOrchestrator",
        forbidden_orchestrator,
    )

    with pytest.raises(ValueError, match="frozen"):
        search_module.run_scripted_search(
            WorkloadName.MLP_STACK,
            tmp_path,
            tmp_path / "criteria.yaml",
            RunSettings(
                schema_version=2,
                repeats=64,
                inter_run_spacing_seconds=0,
            ),
        )

    assert not any(tmp_path.iterdir())


def test_search_constructor_rejects_equal_numeric_settings_alias(
    tmp_path: Path,
) -> None:
    """Direct orchestrator construction enforces the same strict settings boundary."""
    with pytest.raises(ValueError, match="frozen"):
        TrustedSearchOrchestrator(
            tmp_path,
            tmp_path / "criteria.yaml",
            RunSettings(
                schema_version=2,
                repeats=64,
                inter_run_spacing_seconds=0,
            ),
            capability_snapshot=CAPABILITIES,
            runner=_BundleRunner((('PROVEN', 'all_criteria_passed'),)),
        )

    assert not any(tmp_path.iterdir())


def test_offline_search_rejects_rehashed_nondefault_child_methodology(
    tmp_path: Path,
) -> None:
    """A valid re-gate cannot hide non-frozen settings inside a search child."""
    search = TrustedSearchOrchestrator(
        tmp_path,
        tmp_path / "criteria.yaml",
        RunSettings(schema_version=2, repeats=64),
        capability_snapshot=CAPABILITIES,
        runner=_BundleRunner((('PROVEN', 'all_criteria_passed'),)),
    ).optimize(WorkloadName.MLP_STACK)
    child = search.parent_path / search.transcript.steps[0].result.bundle_relative_path
    methodology_path = child / "methodology.json"
    methodology = json.loads(methodology_path.read_bytes())
    methodology["warmup_runs"] = 0
    write_json(methodology_path, methodology)
    write_manifest(child)
    write_manifest(search.parent_path)

    integrity = verify_search_tree(search.parent_path)

    assert integrity.valid is False
    assert "child:methodology.settings" in integrity.mismatches


def test_offline_verifier_rejects_symlink_and_missing_child(tmp_path: Path) -> None:
    """A link or deleted receipt target must not be treated as a manifested run."""
    search = TrustedSearchOrchestrator(
        tmp_path,
        tmp_path / "criteria.yaml",
        RunSettings(schema_version=2, repeats=64),
        capability_snapshot=CAPABILITIES,
        runner=_BundleRunner((('PROVEN', 'all_criteria_passed'),)),
    ).optimize(WorkloadName.MLP_STACK)
    child = search.parent_path / search.transcript.steps[0].result.bundle_relative_path
    (search.parent_path / "runs" / "linked").symlink_to(child, target_is_directory=True)

    symlink_integrity = verify_search_tree(search.parent_path)

    assert symlink_integrity.valid is False
    assert any("symlink:runs/linked" in mismatch for mismatch in symlink_integrity.mismatches)
    (search.parent_path / "runs" / "linked").unlink()
    (child / "verdict.json").unlink()
    missing_integrity = verify_search_tree(search.parent_path)
    assert missing_integrity.valid is False
    assert any("verdict" in mismatch for mismatch in missing_integrity.mismatches)


def _write_gate_valid_child(
    child: Path,
    workload_name: str,
    request: object,
    plan: object,
    metadata: Mapping[str, object] | None,
    expected_finding: str,
    settings: RunSettings,
) -> None:
    plan_payload = input_plan_json(plan)  # type: ignore[arg-type]
    case_indices = plan_payload["compile_sweep_case_indices"]
    assert isinstance(case_indices, tuple)
    case_count = len(case_indices)
    repeats = settings.repeats
    origin = datetime(2026, 8, 29, 12, 0, tzinfo=UTC)

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
        if expected_finding == "perf.speedup_proven"
        else [0.8] * repeats
    )
    config: dict[str, object] = {
        "backend": request.backend,  # type: ignore[attr-defined]
        "mode": request.mode,  # type: ignore[attr-defined]
        "dynamic": request.dynamic,  # type: ignore[attr-defined]
        "fullgraph": request.fullgraph,  # type: ignore[attr-defined]
        "options": None if request.options is None else dict(request.options),  # type: ignore[attr-defined]
        "disable": request.disable,  # type: ignore[attr-defined]
    }
    assert metadata is not None
    config["catalog"] = metadata
    write_json(child / "env.json", {"boundary": "literal unit evidence"})
    write_json(child / "workload.digest", {"name": workload_name, "sha256": "0" * 64})
    write_json(child / "config.json", config)
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
    accuracy_cases = [
        {
            "case_index": index,
            "within_tolerance": True,
            "max_absolute_error": 0.0,
            "mismatch": None,
        }
        for index in range(case_count)
    ]
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
            "cases": accuracy_cases,
        },
    )
    recompiles = (
        [{"trigger": f"shape guard {index}"} for index in range(3)]
        if expected_finding == "graph.recompile_bound"
        else []
    )
    graph_breaks = (
        [{"reason": "Tensor.item"}]
        if expected_finding == "graph.no_breaks"
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
            "warmup_runs": settings.warmup_runs,
            "repeats": repeats,
            "bootstrap_samples": settings.bootstrap_samples,
            "bootstrap_seed": 0,
            "bootstrap_confidence": 0.95,
            "inter_run_spacing_seconds": settings.inter_run_spacing_seconds,
            "aa_noise_floor": 0.0,
            "aa_effect_formula": "(A-B)/((A+B)/2)",
            "aa_estimator": "p95_absolute_bootstrap_median",
            "aa_pairing": "within_iteration",
            "measurement_schedule": "alternate_eager-A-B__B-A-eager",
            "quantile_method": "linear_interpolation",
        },
    )
    (child / "gate_criteria.yaml").write_bytes(
        (Path(__file__).parents[1] / "gates" / "default.yaml").read_bytes()
    )
    finalize_bundle(child, evaluate_bundle)

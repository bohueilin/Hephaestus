from __future__ import annotations

import hashlib
import inspect
import json
import shutil
from collections.abc import Mapping
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import MappingProxyType

import pytest

import hephaestus.aa as aa_module
import hephaestus.aa_runtime as aa_runtime_module
from hephaestus.aa import (
    AATestResult,
    aa_invalid_distribution_json,
    aa_statistics_json,
    aa_verdict_json,
    compute_null_statistics,
    evaluate_aa_bundle,
)
from hephaestus.aa_runtime import TrustedAAOrchestrator
from hephaestus.bundle import (
    canonical_json_bytes,
    finalize_bundle,
    verify_manifest,
    write_json,
    write_manifest,
)
from hephaestus.catalog import catalog_metadata, get_catalog_entry
from hephaestus.gate import _evaluate_provisional_bundle, evaluate_bundle
from hephaestus.input_plan import input_plan_json
from hephaestus.measure import RunResult, RunSettings
from hephaestus.provenance import RunProvenance
from hephaestus.scope import EVIDENCE_BOUNDARY
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


class _AARunner:
    def __init__(self, *, invalid_child: int | None = None) -> None:
        self.calls: list[dict[str, object]] = []
        self.invalid_child = invalid_child

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
        index = len(self.calls) + 1
        child = output_root / f"child-{index}"
        child.mkdir(parents=True)
        self.calls.append(
            {
                "workload_name": workload_name,
                "request": request,
                "output_root": output_root,
                "criteria_path": criteria_path,
                "settings": settings,
                "input_plan": input_plan,
                "catalog_metadata": catalog_metadata,
            }
        )
        assert catalog_metadata is not None
        write_normal_child(
            child,
            workload_name=workload_name,
            request=request,
            plan=input_plan,
            metadata=catalog_metadata,
            driving_finding="perf.speedup_proven",
            settings=settings,
            run_provenance=run_provenance,
            accuracy_valid=index != self.invalid_child,
        )
        verdict = json.loads((child / "verdict.json").read_bytes())
        return RunResult(
            child,
            verdict["verdict"],
            MappingProxyType(verdict),
        )


class _DifferentActionRunner(_AARunner):
    def __call__(self, *args: object, **kwargs: object) -> RunResult:
        result = super().__call__(*args, **kwargs)  # type: ignore[arg-type]
        if len(self.calls) == 2:
            child = result.bundle_path
            config_path = child / "config.json"
            config = json.loads(config_path.read_bytes())
            config["mode"] = "reduce-overhead"
            config["catalog"] = catalog_metadata(
                get_catalog_entry("candidate-mlp-reduce-overhead")
            )
            write_json(config_path, config)
            write_manifest(child)
            verdict = evaluate_bundle(child)
            write_json(child / "verdict.json", verdict)
            write_manifest(child)
            return RunResult(
                child,
                verdict["verdict"],
                MappingProxyType(verdict),
            )
        return result


class _SecondAppendFails:
    def __init__(self) -> None:
        self.bundle_paths: list[Path] = []

    def append(self, _capture: object, bundle_path: Path) -> None:
        self.bundle_paths.append(bundle_path)
        if len(self.bundle_paths) == 2:
            raise OSError("injected second host-state append failure")


def _orchestrator(tmp_path: Path, runner: _AARunner) -> TrustedAAOrchestrator:
    return TrustedAAOrchestrator(
        tmp_path,
        CRITERIA,
        capability_snapshot=CAPABILITIES,
        runner=runner,
    )


def test_runtime_executes_exactly_two_identical_catalog_actions_with_frozen_settings(
    tmp_path: Path,
) -> None:
    """Changing the config, count, or measurement settings between children must fail."""
    runner = _AARunner()

    result = _orchestrator(tmp_path, runner).run("mlp_stack")

    assert len(runner.calls) == 2
    assert [call["catalog_metadata"]["entry_id"] for call in runner.calls] == [  # type: ignore[index]
        "candidate-mlp-default",
        "candidate-mlp-default",
    ]
    assert all(
        call["settings"] == RunSettings(schema_version=2, repeats=64)
        for call in runner.calls
    )
    assert runner.calls[0]["request"] == runner.calls[1]["request"]
    assert result.verdict == "PASS"
    assert result.driving_finding == "all_criteria_passed"
    assert len(result.child_relative_paths) == len(set(result.child_relative_paths)) == 2
    assert result.child_relative_paths == ("runs/child-1", "runs/child-2")


def test_second_host_state_append_failure_cannot_erase_a_finalized_aa_verdict(
    tmp_path: Path,
) -> None:
    """Appending before the second trusted receipt would erase completed A/A evidence."""
    clean = _orchestrator(tmp_path / "clean", _AARunner()).run("mlp_stack")
    runner = _AARunner()
    sink = _SecondAppendFails()
    result = TrustedAAOrchestrator(
        tmp_path / "failure",
        CRITERIA,
        capability_snapshot=CAPABILITIES,
        runner=runner,
        _host_state_sink=sink,
    ).run("mlp_stack")

    assert len(runner.calls) == 2
    assert [path.name for path in sink.bundle_paths] == ["child-1", "child-2"]
    assert [
        json.loads((result.parent_path / relative / "run_provenance.json").read_bytes())[
            "sequence_index"
        ]
        for relative in result.child_relative_paths
    ] == [0, 1]
    assert (result.parent_path / "aa_distribution.json").is_file()
    assert verify_manifest(result.parent_path).valid
    stored = (result.parent_path / "verdict.json").read_bytes()
    assert stored == canonical_json_bytes(aa_verdict_json(result))
    assert evaluate_aa_bundle(result.parent_path) == result
    assert (result.verdict, result.driving_finding) == (
        clean.verdict,
        clean.driving_finding,
    )


def test_aa_rejects_cloned_children_with_canonical_invalid_distribution(
    tmp_path: Path,
) -> None:
    """Byte-identical timing evidence is valid only when two distinct chained runs produced it."""
    result = _orchestrator(tmp_path, _AARunner()).run("mlp_stack")
    first = result.parent_path / result.child_relative_paths[0]
    second = result.parent_path / result.child_relative_paths[1]
    assert (first / "run_provenance.json").is_file()
    shutil.rmtree(second)
    shutil.copytree(first, second)
    write_json(
        result.parent_path / "aa_distribution.json",
        aa_invalid_distribution_json(schema_version=2),
    )

    checked = _finalize_tampered_aa_parent(result.parent_path)

    assert checked.verdict == "INVALID_EVIDENCE"
    assert checked.driving_finding == "aa.provenance"


def test_aa_requires_equal_child_environment_bytes(tmp_path: Path) -> None:
    """A/A comparisons cannot combine otherwise valid runs from distinct pinned environments."""
    result = _orchestrator(tmp_path, _AARunner()).run("mlp_stack")
    second = result.parent_path / result.child_relative_paths[1]
    env_path = second / "env.json"
    env = json.loads(env_path.read_bytes())
    assert env["os"]["release"]
    env["os"]["release"] = "different-release"
    write_json(env_path, env)
    write_manifest(second)
    aa_test_path = result.parent_path / "aa_test.json"
    aa_test = json.loads(aa_test_path.read_bytes())
    aa_test["children"][1]["env_sha256"] = hashlib.sha256(env_path.read_bytes()).hexdigest()
    aa_test["children"][1]["bundle_manifest_sha256"] = hashlib.sha256(
        (second / "manifest.json").read_bytes()
    ).hexdigest()
    write_json(aa_test_path, aa_test)
    write_json(
        result.parent_path / "aa_distribution.json",
        aa_invalid_distribution_json(schema_version=2),
    )

    checked = _finalize_tampered_aa_parent(result.parent_path)

    assert checked.verdict == "INVALID_EVIDENCE"
    assert checked.driving_finding == "aa.environment"


def test_parent_contains_exact_schema_files_and_recursive_child_manifest(tmp_path: Path) -> None:
    """Dropping scope, raw A/A evidence, or any child byte must break parent integrity."""
    result = _orchestrator(tmp_path, _AARunner()).run("mlp_stack")

    assert {path.name for path in result.parent_path.iterdir()} == {
        "aa_test.json",
        "aa_methodology.json",
        "aa_distribution.json",
        "gate_criteria.yaml",
        "scope.json",
        "runs",
        "verdict.json",
        "manifest.json",
    }
    scope = json.loads((result.parent_path / "scope.json").read_bytes())
    assert scope == {"schema_version": 1, "boundary": EVIDENCE_BOUNDARY}
    manifest = json.loads((result.parent_path / "manifest.json").read_bytes())
    assert "runs/child-1/verdict.json" in manifest["files"]
    assert "runs/child-2/timings.json" in manifest["files"]
    assert evaluate_aa_bundle(result.parent_path) == result


def test_valid_not_proven_children_do_not_fail_the_aa_meta_test(tmp_path: Path) -> None:
    """A normal scientific refusal is valid evidence and must not become process failure."""
    result = _orchestrator(tmp_path, _AARunner()).run("mlp_stack")
    child_verdicts = [
        json.loads((result.parent_path / path / "verdict.json").read_bytes())["verdict"]
        for path in result.child_relative_paths
    ]

    assert child_verdicts == ["NOT_PROVEN", "NOT_PROVEN"]
    assert result.verdict == "PASS"


def test_invalid_child_invalidates_parent(tmp_path: Path) -> None:
    """An accuracy-invalid child cannot contribute timings to the null decision."""
    result = _orchestrator(tmp_path, _AARunner(invalid_child=2)).run("mlp_stack")

    assert result.verdict == "INVALID_EVIDENCE"
    assert result.driving_finding == "child.invalid_evidence"


@pytest.mark.parametrize(
    ("filename", "mutation"),
    (
        ("config.json", lambda value: {**value, "mode": "forged"}),
        (
            "input_plan.json",
            lambda value: {**value, "steady_state_case_index": 99},
        ),
        ("workload.digest", lambda value: {**value, "sha256": "f" * 64}),
    ),
)
def test_unequal_child_identity_evidence_is_rejected_after_rehash(
    tmp_path: Path,
    filename: str,
    mutation: object,
) -> None:
    """Two runs with unequal config, plan, or workload identity are not an A/A pair."""
    result = _orchestrator(tmp_path, _AARunner()).run("mlp_stack")
    child = result.parent_path / result.child_relative_paths[1]
    path = child / filename
    payload = json.loads(path.read_bytes())
    write_json(path, mutation(payload))  # type: ignore[operator]
    write_manifest(child)
    _refresh_aa_child_authority(result.parent_path)
    write_manifest(result.parent_path)

    checked = _evaluate_provisional_semantics(
        result.parent_path,
        canonical_invalid_distribution=True,
    )

    assert checked.verdict == "INVALID_EVIDENCE"
    assert checked.driving_finding == "aa.child_equality"


def test_unequal_child_criteria_is_rejected_after_rehash(tmp_path: Path) -> None:
    """A criteria change between children cannot masquerade as a config null test."""
    result = _orchestrator(tmp_path, _AARunner()).run("mlp_stack")
    child = result.parent_path / result.child_relative_paths[1]
    (child / "gate_criteria.yaml").write_text(
        (child / "gate_criteria.yaml").read_text().replace("1.10", "1.11"),
        encoding="utf-8",
    )
    write_manifest(child)
    _refresh_aa_child_authority(result.parent_path)
    write_manifest(result.parent_path)

    checked = _evaluate_provisional_semantics(
        result.parent_path,
        canonical_invalid_distribution=True,
    )

    assert checked.verdict == "INVALID_EVIDENCE"
    assert checked.driving_finding == "aa.child_equality"


def test_forged_child_verdict_is_rejected_after_recursive_rehash(tmp_path: Path) -> None:
    """Stored child verdict bytes must equal a fresh pure normal-bundle re-gate."""
    result = _orchestrator(tmp_path, _AARunner()).run("mlp_stack")
    child = result.parent_path / result.child_relative_paths[1]
    write_json(
        child / "verdict.json",
        {"verdict": "PROVEN", "driving_finding": "all_criteria_passed", "findings": []},
    )
    write_manifest(child)
    _refresh_aa_child_authority(result.parent_path)
    write_manifest(result.parent_path)

    checked = _evaluate_provisional_semantics(
        result.parent_path,
        canonical_invalid_distribution=True,
    )

    assert checked.verdict == "INVALID_EVIDENCE"
    assert checked.driving_finding == "child.gate_verdict"


def test_rehashed_distribution_tamper_is_recomputed_from_child_raw_series(
    tmp_path: Path,
) -> None:
    """Parent hashes cannot bless a changed stored bootstrap derivation."""
    result = _orchestrator(tmp_path, _AARunner()).run("mlp_stack")
    path = result.parent_path / "aa_distribution.json"
    distribution = json.loads(path.read_bytes())
    distribution["bootstrap_absolute_medians"][0] = 0.5
    write_json(path, distribution)
    write_manifest(result.parent_path)

    checked = _evaluate_provisional_semantics(result.parent_path)

    assert checked.verdict == "INVALID_EVIDENCE"
    assert checked.driving_finding == "aa.bootstrap"


def test_huge_distribution_number_returns_named_invalid_evidence(tmp_path: Path) -> None:
    """Unrepresentable JSON numbers must fail closed instead of escaping evaluation."""
    result = _orchestrator(tmp_path, _AARunner()).run("mlp_stack")
    path = result.parent_path / "aa_distribution.json"
    distribution = json.loads(path.read_bytes())
    distribution["p95_noise_floor"] = 10**400
    write_json(path, distribution)
    write_manifest(result.parent_path)

    checked = evaluate_aa_bundle(result.parent_path)

    assert checked.verdict == "INVALID_EVIDENCE"
    assert checked.driving_finding == "parent.verdict"
    assert "aa_distribution:invalid" in checked.mismatches


def test_finalized_large_finite_aa_bundle_cannot_false_pass(tmp_path: Path) -> None:
    """A finalized tree preserves the declared effect when finite child sums overflow."""
    result = _orchestrator(tmp_path, _AARunner()).run("mlp_stack")
    high = 1.79e308
    low = 1.5e308
    series_a = (high,) * 64
    series_b = (low,) * 64
    for relative, series in zip(
        result.child_relative_paths,
        (series_a, series_b),
        strict=True,
    ):
        child = result.parent_path / relative
        path = child / "timings.json"
        timings = json.loads(path.read_bytes())
        for key in (
            "compiled_seconds",
            "aa_baseline_seconds",
            "aa_candidate_seconds",
        ):
            timings[key] = list(series)
        timings["aa_signed_paired_effects"] = [0.0] * 64
        timings["aa_bootstrap_absolute_medians"] = [0.0] * 2000
        write_json(path, timings)
        (child / "verdict.json").unlink()
        write_manifest(child)
        verdict = _evaluate_provisional_bundle(child)
        assert verdict["verdict"] != "INVALID_EVIDENCE"
        write_json(child / "verdict.json", verdict)
        write_manifest(child)

    _refresh_aa_child_authority(result.parent_path)

    statistics = compute_null_statistics(
        series_a,
        series_b,
        schema_version=2,
    )
    write_json(
        result.parent_path / "aa_distribution.json",
        {
            "schema_version": 2,
            "compiled_seconds_a": list(series_a),
            "compiled_seconds_b": list(series_b),
            **aa_statistics_json(statistics),
        },
    )
    checked = _finalize_tampered_aa_parent(result.parent_path)

    assert checked.verdict == "FAIL"
    assert checked.driving_finding == "methodology.noise_floor"


def test_rehashed_parent_verdict_tamper_is_visible(tmp_path: Path) -> None:
    """A rewritten custom parent verdict cannot survive offline semantic evaluation."""
    result = _orchestrator(tmp_path, _AARunner()).run("mlp_stack")
    write_json(
        result.parent_path / "verdict.json",
        {"schema_version": 1, "verdict": "PASS", "driving_finding": "forged"},
    )
    write_manifest(result.parent_path)

    checked = evaluate_aa_bundle(result.parent_path)

    assert checked.verdict == "INVALID_EVIDENCE"
    assert checked.driving_finding == "parent.verdict"


def test_missing_child_and_symlink_are_rejected(tmp_path: Path) -> None:
    """Unsafe or absent child paths cannot be counted as independent complete runs."""
    result = _orchestrator(tmp_path, _AARunner()).run("mlp_stack")
    child = result.parent_path / result.child_relative_paths[1]
    verdict = child / "verdict.json"
    verdict.unlink()

    missing = evaluate_aa_bundle(result.parent_path)

    assert missing.verdict == "INVALID_EVIDENCE"
    assert missing.driving_finding == "evidence.integrity"

    second = _orchestrator(tmp_path, _AARunner()).run("mlp_stack")
    linked = second.parent_path / "runs" / "linked"
    linked.symlink_to(second.parent_path / second.child_relative_paths[0], target_is_directory=True)

    symlinked = evaluate_aa_bundle(second.parent_path)

    assert symlinked.verdict == "INVALID_EVIDENCE"
    assert symlinked.driving_finding == "evidence.integrity"


def test_stored_parent_verdict_bytes_equal_fresh_offline_evaluation(tmp_path: Path) -> None:
    """Finalization must store exactly the canonical result produced by offline logic."""
    result = _orchestrator(tmp_path, _AARunner()).run("mlp_stack")

    stored = (result.parent_path / "verdict.json").read_bytes()
    first = evaluate_aa_bundle(result.parent_path)
    second = evaluate_aa_bundle(result.parent_path)

    assert stored == canonical_json_bytes(aa_verdict_json(first))
    assert aa_verdict_json(first) == aa_verdict_json(second)


def test_offline_evaluator_uses_stored_candidate_identity_after_live_mapping_evolves(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Historical A/A evidence must not depend on the later live action selection map."""
    result = _orchestrator(tmp_path, _AARunner()).run("mlp_stack")
    evolved = dict(aa_module.AA_CATALOG_IDS)
    evolved["mlp_stack"] = "future-candidate-mlp"
    monkeypatch.setattr(aa_module, "AA_CATALOG_IDS", MappingProxyType(evolved))

    checked = evaluate_aa_bundle(result.parent_path)

    assert checked == result


def test_identical_malformed_child_configs_return_named_invalid_evidence(
    tmp_path: Path,
) -> None:
    """Malformed but byte-equal configs must not escape the pure evaluator as JSON errors."""
    result = _orchestrator(tmp_path, _AARunner()).run("mlp_stack")
    aa_test_path = result.parent_path / "aa_test.json"
    aa_test = json.loads(aa_test_path.read_bytes())
    malformed = b"{"
    for child_record in aa_test["children"]:
        child = result.parent_path / child_record["bundle_relative_path"]
        (child / "config.json").write_bytes(malformed)
        write_manifest(child)
        child_record["config_sha256"] = hashlib.sha256(malformed).hexdigest()
    _refresh_aa_child_authority(result.parent_path, aa_test=aa_test)
    write_manifest(result.parent_path)

    checked = _evaluate_provisional_semantics(
        result.parent_path,
        canonical_invalid_distribution=True,
    )

    assert checked.verdict == "INVALID_EVIDENCE"
    assert checked.driving_finding == "aa.catalog"


def test_aa_complete_action_is_frozen_against_coherent_two_child_tamper(
    tmp_path: Path,
) -> None:
    """Both children cannot replace the frozen default action under its trusted ID."""
    result = _orchestrator(tmp_path, _AARunner()).run("mlp_stack")
    aa_test_path = result.parent_path / "aa_test.json"
    aa_test = json.loads(aa_test_path.read_bytes())
    for child_record in aa_test["children"]:
        child = result.parent_path / child_record["bundle_relative_path"]
        config_path = child / "config.json"
        config = json.loads(config_path.read_bytes())
        config["mode"] = "reduce-overhead"
        config["catalog"]["requested"]["mode"] = "reduce-overhead"
        config["catalog"]["effective"]["mode"] = "reduce-overhead"
        write_json(config_path, config)
        write_manifest(child)
        child_record["config_sha256"] = hashlib.sha256(
            config_path.read_bytes()
        ).hexdigest()
    _refresh_aa_child_authority(result.parent_path, aa_test=aa_test)
    write_manifest(result.parent_path)

    checked = _evaluate_provisional_semantics(
        result.parent_path,
        canonical_invalid_distribution=True,
    )

    assert checked.verdict == "INVALID_EVIDENCE"
    assert checked.driving_finding == "aa.catalog"


def test_runtime_finalizes_invalid_child_before_null_statistics(
    tmp_path: Path,
) -> None:
    """A short invalid child series must produce canonical parent evidence, not raise."""

    class _ShortInvalidRunner(_AARunner):
        def __call__(self, *args: object, **kwargs: object) -> RunResult:
            result = super().__call__(*args, **kwargs)  # type: ignore[arg-type]
            if len(self.calls) == 2:
                timings_path = result.bundle_path / "timings.json"
                timings = json.loads(timings_path.read_bytes())
                timings["compiled_seconds"] = [1.0]
                write_json(timings_path, timings)
                write_manifest(result.bundle_path)
                verdict = evaluate_bundle(result.bundle_path)
                write_json(result.bundle_path / "verdict.json", verdict)
                write_manifest(result.bundle_path)
                return RunResult(
                    result.bundle_path,
                    verdict["verdict"],
                    MappingProxyType(verdict),
                )
            return result

    result = _orchestrator(tmp_path, _ShortInvalidRunner()).run("mlp_stack")

    assert result.verdict == "INVALID_EVIDENCE"
    assert result.driving_finding == "child.invalid_evidence"
    assert result.statistics is None
    assert {path.name for path in result.parent_path.iterdir()} == {
        "aa_test.json",
        "aa_methodology.json",
        "aa_distribution.json",
        "gate_criteria.yaml",
        "scope.json",
        "runs",
        "verdict.json",
        "manifest.json",
    }
    assert evaluate_aa_bundle(result.parent_path) == result
    assert (result.parent_path / "verdict.json").read_bytes() == canonical_json_bytes(
        aa_verdict_json(result)
    )


def test_runtime_preflights_cross_child_action_before_null_statistics(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A coherent but unequal second action must never reach null statistics."""

    def forbidden_statistics(*args: object, **kwargs: object) -> object:
        del args, kwargs
        raise AssertionError("null statistics consumed a non-identical child pair")

    monkeypatch.setattr(
        aa_runtime_module,
        "compute_null_statistics",
        forbidden_statistics,
    )

    result = _orchestrator(tmp_path, _DifferentActionRunner()).run("mlp_stack")

    assert result.verdict == "INVALID_EVIDENCE"
    assert result.driving_finding == "aa.child_equality"
    assert result.statistics is None
    assert evaluate_aa_bundle(result.parent_path) == result


def test_runtime_consumes_preflight_timing_snapshot_without_rereading_children(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A child timing file disappearing after preflight must not crash finalization."""
    real_preflight = aa_runtime_module._preflight_children

    def remove_timing_after_preflight(
        parent: Path,
        child_paths: tuple[str, str],
        **kwargs: object,
    ) -> object:
        snapshot = real_preflight(parent, child_paths, **kwargs)  # type: ignore[arg-type]
        (parent / child_paths[1] / "timings.json").unlink()
        return snapshot

    monkeypatch.setattr(
        aa_runtime_module,
        "_preflight_children",
        remove_timing_after_preflight,
    )

    result = _orchestrator(tmp_path, _AARunner()).run("mlp_stack")

    assert result.verdict == "INVALID_EVIDENCE"
    assert result.statistics is None
    assert evaluate_aa_bundle(result.parent_path) == result
    assert (result.parent_path / "verdict.json").read_bytes() == canonical_json_bytes(
        aa_verdict_json(result)
    )


def test_runtime_finalizes_child_with_unrepresentable_timing_number(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A manifested huge timing value must not reach statistics or escape preflight."""

    class _HugeTimingRunner(_AARunner):
        def __call__(self, *args: object, **kwargs: object) -> RunResult:
            result = super().__call__(*args, **kwargs)  # type: ignore[arg-type]
            if len(self.calls) == 2:
                path = result.bundle_path / "timings.json"
                timings = json.loads(path.read_bytes())
                timings["compiled_seconds"][0] = 10**400
                write_json(path, timings)
                write_manifest(result.bundle_path)
            return result

    def forbidden_statistics(*args: object, **kwargs: object) -> object:
        del args, kwargs
        raise AssertionError("null statistics consumed an invalid numeric series")

    monkeypatch.setattr(
        aa_runtime_module,
        "compute_null_statistics",
        forbidden_statistics,
    )

    result = _orchestrator(tmp_path, _HugeTimingRunner()).run("mlp_stack")

    assert result.verdict == "INVALID_EVIDENCE"
    assert result.statistics is None
    assert evaluate_aa_bundle(result.parent_path) == result
    assert (result.parent_path / "aa_distribution.json").read_bytes() == (
        canonical_json_bytes(aa_invalid_distribution_json(schema_version=2))
    )


def test_runtime_finalizes_derivation_failure_with_canonical_invalid_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A valid timing snapshot whose derivation fails becomes a stable invalid parent."""

    def failed_derivation(*args: object, **kwargs: object) -> object:
        del args, kwargs
        raise ValueError("nonfinite derived statistic")

    monkeypatch.setattr(
        aa_runtime_module,
        "compute_null_statistics",
        failed_derivation,
    )

    result = _orchestrator(tmp_path, _AARunner()).run("mlp_stack")

    assert result.verdict == "INVALID_EVIDENCE"
    assert result.driving_finding == "aa.derivation"
    assert result.statistics is None
    assert (result.parent_path / "aa_distribution.json").read_bytes() == (
        canonical_json_bytes(aa_invalid_distribution_json(schema_version=2))
    )
    assert evaluate_aa_bundle(result.parent_path) == result


def test_valid_children_with_canonical_invalid_distribution_mean_derivation_failure(
    tmp_path: Path,
) -> None:
    """The exact null placeholder has one meaning after both children validate."""
    result = _orchestrator(tmp_path, _AARunner()).run("mlp_stack")
    write_json(
        result.parent_path / "aa_distribution.json",
        aa_invalid_distribution_json(schema_version=2),
    )

    checked = _evaluate_provisional_semantics(result.parent_path)

    assert checked.verdict == "INVALID_EVIDENCE"
    assert checked.driving_finding == "aa.derivation"
    assert checked.statistics is None


def test_valid_children_reject_forged_invalid_distribution_placeholder(
    tmp_path: Path,
) -> None:
    """Only the exact frozen null placeholder can signal a derivation failure."""
    result = _orchestrator(tmp_path, _AARunner()).run("mlp_stack")
    write_json(result.parent_path / "aa_distribution.json", {"forged": True})

    checked = _evaluate_provisional_semantics(result.parent_path)

    assert checked.verdict == "INVALID_EVIDENCE"
    assert checked.driving_finding == "aa.schema"
    assert "aa_distribution:invalid" in checked.mismatches


def test_unequal_child_parent_rejects_noncanonical_invalid_distribution(
    tmp_path: Path,
) -> None:
    """Every preflight-invalid parent requires the same canonical distribution."""

    result = _orchestrator(tmp_path, _DifferentActionRunner()).run("mlp_stack")
    assert result.driving_finding == "aa.child_equality"
    write_json(result.parent_path / "aa_distribution.json", {"forged": True})
    write_manifest(result.parent_path)

    checked = evaluate_aa_bundle(result.parent_path)

    assert checked.verdict == "INVALID_EVIDENCE"
    assert checked.driving_finding == "parent.verdict"
    assert "aa_distribution:invalid" in checked.mismatches


@pytest.mark.parametrize(
    "mutation",
    (
        "unsafe-path",
        "malformed-record",
        "malformed-digest",
        "catalog-id",
        "missing-run",
        "unexpected-run",
    ),
)
def test_child_declaration_and_topology_require_canonical_invalid_distribution(
    tmp_path: Path,
    mutation: str,
) -> None:
    """No child-related invalid path may accept an arbitrary distribution payload."""
    result = _orchestrator(tmp_path, _AARunner()).run("mlp_stack")
    parent = result.parent_path
    aa_test_path = parent / "aa_test.json"
    aa_test = json.loads(aa_test_path.read_bytes())
    if mutation == "unsafe-path":
        aa_test["children"][0]["bundle_relative_path"] = "../escape"
        write_json(aa_test_path, aa_test)
    elif mutation == "malformed-record":
        aa_test["children"][0].pop("config_sha256")
        write_json(aa_test_path, aa_test)
    elif mutation == "malformed-digest":
        aa_test["children"][0]["config_sha256"] = "not-a-digest"
        write_json(aa_test_path, aa_test)
    elif mutation == "catalog-id":
        aa_test["catalog_id"] = "candidate-mlp-reduce-overhead"
        write_json(aa_test_path, aa_test)
    elif mutation == "missing-run":
        shutil.rmtree(parent / result.child_relative_paths[1])
    elif mutation == "unexpected-run":
        (parent / "runs" / "unexpected").mkdir()
    else:
        raise AssertionError(f"unknown mutation: {mutation}")
    write_json(parent / "aa_distribution.json", {"forged": True})

    checked = _finalize_tampered_aa_parent(parent)

    assert checked.verdict == "INVALID_EVIDENCE"
    assert checked.driving_finding == "aa.schema"
    assert "aa_distribution:invalid" in checked.mismatches


@pytest.mark.parametrize("missing", ("aa_test.json", "runs"))
def test_child_authority_root_omission_requires_canonical_invalid_distribution(
    tmp_path: Path,
    missing: str,
) -> None:
    """Missing child declaration/root evidence cannot authorize a forged placeholder."""
    result = _orchestrator(tmp_path, _AARunner()).run("mlp_stack")
    parent = result.parent_path
    target = parent / missing
    if target.is_dir():
        shutil.rmtree(target)
    else:
        target.unlink()
    write_json(parent / "aa_distribution.json", {"forged": True})

    checked = _finalize_tampered_aa_parent(parent)

    assert checked.verdict == "INVALID_EVIDENCE"
    assert checked.driving_finding == "aa.schema"
    assert "aa_distribution:invalid" in checked.mismatches


def test_nonchild_parent_topology_preserves_its_original_finding(
    tmp_path: Path,
) -> None:
    """Unexpected non-child parent evidence remains a parent topology failure."""
    result = _orchestrator(tmp_path, _AARunner()).run("mlp_stack")
    parent = result.parent_path
    (parent / "unexpected.txt").write_bytes(b"unexpected")
    write_json(parent / "aa_distribution.json", {"forged": True})

    checked = _finalize_tampered_aa_parent(parent)

    assert checked.verdict == "INVALID_EVIDENCE"
    assert checked.driving_finding == "parent.topology"
    assert checked.mismatches == ("parent:unexpected:unexpected.txt",)


def test_runtime_finalizes_invalid_child_with_missing_identity_file(
    tmp_path: Path,
) -> None:
    """A finalized invalid child lacking config evidence must not crash its parent."""

    class _MissingConfigRunner(_AARunner):
        def __call__(self, *args: object, **kwargs: object) -> RunResult:
            result = super().__call__(*args, **kwargs)  # type: ignore[arg-type]
            if len(self.calls) == 2:
                child = result.bundle_path
                (child / "config.json").unlink()
                write_manifest(child)
                verdict = evaluate_bundle(child)
                write_json(child / "verdict.json", verdict)
                write_manifest(child)
                return RunResult(
                    child,
                    verdict["verdict"],
                    MappingProxyType(verdict),
                )
            return result

    result = _orchestrator(tmp_path, _MissingConfigRunner()).run("mlp_stack")

    assert result.verdict == "INVALID_EVIDENCE"
    assert result.statistics is None
    assert evaluate_aa_bundle(result.parent_path) == result
    assert (result.parent_path / "verdict.json").read_bytes() == canonical_json_bytes(
        aa_verdict_json(result)
    )


def test_invalid_parent_rejects_forged_pass_verdict_after_rehash(tmp_path: Path) -> None:
    """A semantic child error cannot bypass finalized parent-verdict validation."""
    result = _orchestrator(tmp_path, _AARunner(invalid_child=2)).run("mlp_stack")
    write_json(
        result.parent_path / "verdict.json",
        {
            "schema_version": 1,
            "verdict": "PASS",
            "driving_finding": "all_criteria_passed",
            "findings": [
                {
                    "id": "all_criteria_passed",
                    "status": "PASS",
                    "mismatches": [],
                }
            ],
            "statistics": None,
        },
    )
    write_manifest(result.parent_path)

    checked = evaluate_aa_bundle(result.parent_path)

    assert checked.verdict == "INVALID_EVIDENCE"
    assert checked.driving_finding == "parent.verdict"
    assert "verdict:semantic_mismatch" in checked.mismatches


def test_invalid_parent_rejects_noncanonical_distribution_after_rehash(
    tmp_path: Path,
) -> None:
    """Invalid-child parents accept only the exact frozen invalid distribution."""
    result = _orchestrator(tmp_path, _AARunner(invalid_child=2)).run("mlp_stack")
    write_json(result.parent_path / "aa_distribution.json", {"forged": True})
    write_manifest(result.parent_path)

    checked = evaluate_aa_bundle(result.parent_path)

    assert checked.verdict == "INVALID_EVIDENCE"
    assert checked.driving_finding == "parent.verdict"
    assert "aa_distribution:invalid" in checked.mismatches


def test_aa_child_settings_reject_integer_alias_for_frozen_float(
    tmp_path: Path,
) -> None:
    """Integer zero cannot satisfy the frozen child spacing value of float zero."""
    result = _orchestrator(tmp_path, _AARunner()).run("mlp_stack")
    for relative in result.child_relative_paths:
        child = result.parent_path / relative
        path = child / "methodology.json"
        methodology = json.loads(path.read_bytes())
        methodology["inter_run_spacing_seconds"] = 0
        write_json(path, methodology)
        write_manifest(child)
    _refresh_aa_child_authority(result.parent_path)
    write_json(
        result.parent_path / "aa_distribution.json",
        aa_invalid_distribution_json(schema_version=2),
    )
    (result.parent_path / "verdict.json").unlink()
    write_manifest(result.parent_path)

    checked = aa_module._evaluate_provisional_aa_bundle(result.parent_path)

    assert checked.verdict == "INVALID_EVIDENCE"
    assert checked.driving_finding == "methodology.settings"


def test_public_aa_evaluator_has_no_provisional_bypass(tmp_path: Path) -> None:
    """Only the private trusted finalizer may evaluate verdict-less A/A topology."""
    result = _orchestrator(tmp_path, _AARunner()).run("mlp_stack")
    (result.parent_path / "verdict.json").unlink()
    write_manifest(result.parent_path)

    public = evaluate_aa_bundle(result.parent_path)
    private_evaluator = getattr(aa_module, "_evaluate_provisional_aa_bundle", None)

    assert "_allow_provisional" not in inspect.signature(evaluate_aa_bundle).parameters
    assert public.verdict == "INVALID_EVIDENCE"
    assert public.driving_finding == "parent.topology"
    assert callable(private_evaluator)
    assert private_evaluator(result.parent_path).driving_finding != "parent.topology"


@pytest.mark.parametrize(
    ("filename", "expected_finding"),
    (
        ("aa_methodology.json", "methodology.valid"),
        ("aa_distribution.json", "aa.schema"),
        ("scope.json", "scope.boundary"),
    ),
)
def test_aa_schema_versions_reject_boolean_true(
    tmp_path: Path,
    filename: str,
    expected_finding: str,
) -> None:
    """JSON true must not compare equal to integer schema version one."""
    result = _orchestrator(tmp_path, _AARunner()).run("mlp_stack")
    path = result.parent_path / filename
    payload = json.loads(path.read_bytes())
    payload["schema_version"] = True
    write_json(path, payload)
    write_manifest(result.parent_path)

    checked = _evaluate_provisional_semantics(result.parent_path)

    assert checked.verdict == "INVALID_EVIDENCE"
    assert checked.driving_finding == expected_finding


@pytest.mark.parametrize(
    ("key", "alias"),
    (
        ("independent_runs", 2.0),
        ("repeats", 31.0),
        ("bootstrap_samples", 2000.0),
        ("bootstrap_seed", False),
    ),
)
def test_aa_methodology_rejects_equal_numeric_type_aliases(
    tmp_path: Path,
    key: str,
    alias: object,
) -> None:
    """Equal-valued floats and booleans cannot satisfy frozen methodology types."""
    result = _orchestrator(tmp_path, _AARunner()).run("mlp_stack")
    path = result.parent_path / "aa_methodology.json"
    methodology = json.loads(path.read_bytes())
    methodology[key] = alias
    write_json(path, methodology)
    write_manifest(result.parent_path)

    checked = _evaluate_provisional_semantics(result.parent_path)

    assert checked.verdict == "INVALID_EVIDENCE"
    assert checked.driving_finding == "methodology.valid"


def test_aa_criteria_rejects_boolean_seed_before_child_equality(tmp_path: Path) -> None:
    """A boolean seed must fail the frozen criteria contract before identity comparison."""
    result = _orchestrator(tmp_path, _AARunner()).run("mlp_stack")
    path = result.parent_path / "gate_criteria.yaml"
    path.write_text(
        path.read_text(encoding="utf-8").replace("seed: 0", "seed: false"),
        encoding="utf-8",
    )
    write_manifest(result.parent_path)

    checked = _evaluate_provisional_semantics(result.parent_path)

    assert checked.verdict == "INVALID_EVIDENCE"
    assert checked.driving_finding == "methodology.criteria"


def _write_normal_child(
    child: Path,
    *,
    workload_name: str,
    request: object,
    input_plan: object,
    catalog_metadata: Mapping[str, object] | None,
    accuracy_valid: bool,
) -> None:
    plan = input_plan_json(input_plan)  # type: ignore[arg-type]
    case_indices = plan["compile_sweep_case_indices"]
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

    config: dict[str, object] = {
        "backend": request.backend,  # type: ignore[attr-defined]
        "mode": request.mode,  # type: ignore[attr-defined]
        "dynamic": request.dynamic,  # type: ignore[attr-defined]
        "fullgraph": request.fullgraph,  # type: ignore[attr-defined]
        "options": None if request.options is None else dict(request.options),  # type: ignore[attr-defined]
        "disable": request.disable,  # type: ignore[attr-defined]
        "catalog": catalog_metadata,
    }
    write_json(child / "env.json", {"boundary": EVIDENCE_BOUNDARY})
    write_json(child / "workload.digest", {"name": workload_name, "sha256": "0" * 64})
    write_json(child / "config.json", config)
    write_json(child / "input_plan.json", plan)
    write_json(
        child / "timings.json",
        {
            "eager_seconds": [1.0] * repeats,
            "eager_timestamps_utc": eager_timestamps,
            "compiled_seconds": [1.0] * repeats,
            "compiled_timestamps_utc": baseline_timestamps,
            "cold_compile_seconds": 1.0,
            "cold_compile_timestamp_utc": timestamp(0),
            "non_primary_compile_sweep_case_indices": list(range(1, case_count)),
            "non_primary_compile_sweep_seconds": [0.1] * (case_count - 1),
            "non_primary_compile_sweep_timestamps_utc": [
                timestamp(index) for index in range(1, case_count)
            ],
            "aa_baseline_seconds": [1.0] * repeats,
            "aa_baseline_timestamps_utc": baseline_timestamps,
            "aa_candidate_seconds": [1.0] * repeats,
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
            "within_tolerance": accuracy_valid,
            "case_index": 0,
            "atol": 1e-5,
            "rtol": 1e-5,
            "max_absolute_error": 0.0,
            "mismatch": None if accuracy_valid else {"kind": "value"},
            "cases": [
                {
                    "case_index": index,
                    "within_tolerance": accuracy_valid,
                    "max_absolute_error": 0.0,
                    "mismatch": None if accuracy_valid else {"kind": "value"},
                }
                for index in range(case_count)
            ],
        },
    )
    write_json(child / "dynamo_report.json", {"graph_breaks": [], "recompiles": []})
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


def _evaluate_provisional_semantics(
    parent: Path,
    *,
    canonical_invalid_distribution: bool = False,
) -> AATestResult:
    if canonical_invalid_distribution:
        write_json(
            parent / "aa_distribution.json",
            aa_invalid_distribution_json(schema_version=2),
        )
    (parent / "verdict.json").unlink()
    write_manifest(parent)
    return aa_module._evaluate_provisional_aa_bundle(parent)


def _refresh_aa_child_authority(
    parent: Path,
    *,
    aa_test: dict[str, object] | None = None,
) -> None:
    payload = (
        json.loads((parent / "aa_test.json").read_bytes())
        if aa_test is None
        else aa_test
    )
    children = payload["children"]
    assert isinstance(children, list)
    prior_child: Path | None = None
    prior_run_id: str | None = None
    for record in children:
        assert isinstance(record, dict)
        child = parent / record["bundle_relative_path"]
        provenance_path = child / "run_provenance.json"
        provenance = json.loads(provenance_path.read_bytes())
        if prior_child is not None:
            provenance["predecessor"] = {
                "run_id": prior_run_id,
                "manifest_sha256": hashlib.sha256(
                    (prior_child / "manifest.json").read_bytes()
                ).hexdigest(),
            }
            write_json(provenance_path, provenance)
            write_manifest(child)
        record.update(
            {
                "config_sha256": hashlib.sha256(
                    (child / "config.json").read_bytes()
                ).hexdigest(),
                "env_sha256": hashlib.sha256((child / "env.json").read_bytes()).hexdigest(),
                "input_plan_sha256": hashlib.sha256(
                    (child / "input_plan.json").read_bytes()
                ).hexdigest(),
                "workload_sha256": hashlib.sha256(
                    (child / "workload.digest").read_bytes()
                ).hexdigest(),
                "criteria_sha256": hashlib.sha256(
                    (child / "gate_criteria.yaml").read_bytes()
                ).hexdigest(),
                "bundle_manifest_sha256": hashlib.sha256(
                    (child / "manifest.json").read_bytes()
                ).hexdigest(),
            }
        )
        prior_child = child
        prior_run_id = provenance["run_id"]
    write_json(parent / "aa_test.json", payload)


def _finalize_tampered_aa_parent(parent: Path) -> AATestResult:
    verdict_path = parent / "verdict.json"
    if verdict_path.exists():
        verdict_path.unlink()
    write_manifest(parent)
    provisional = aa_module._evaluate_provisional_aa_bundle(parent)
    write_json(verdict_path, aa_verdict_json(provisional))
    write_manifest(parent)
    return evaluate_aa_bundle(parent)

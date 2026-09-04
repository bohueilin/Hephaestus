import json
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
import yaml

from hephaestus.bundle import canonical_json_bytes, finalize_bundle, write_json, write_manifest
from hephaestus.catalog import catalog_metadata, get_catalog_entry
from hephaestus.evidence_contract import (
    V1_WORKLOAD_SHA256,
    v1_compile_cache_json,
    v1_path_normalization_json,
)
from hephaestus.gate import _evaluate_provisional_bundle, evaluate_bundle
from hephaestus.scope import EVIDENCE_BOUNDARY
from tests.evidence_helpers import cache_metric

CRITERIA = """\
schema_version: 1
minimum_speedup: 1.10
bootstrap:
  samples: 2000
  confidence: 0.95
  seed: 0
compile_budgets_seconds:
  mlp_stack: 30.0
  transformer_block: 60.0
  dynamic_batch_text: 90.0
  graph_break_bait: 30.0
maximum_recompiles:
  mlp_stack: 0
  transformer_block: 0
  dynamic_batch_text: 2
  graph_break_bait: 0
graph_break_policy:
  mode: conditional_if_any
  reasons_required: true
"""


def _write_bundle(
    bundle_dir: Path,
    *,
    eager_seconds: list[float] | None = None,
    compiled_seconds: list[float] | None = None,
    cold_compile_seconds: float = 1.0,
    accuracy_valid: bool = True,
    methodology_valid: bool = True,
    methodology_bootstrap_samples: int = 2000,
    aa_noise_floor: float = 0.0,
    aa_baseline_seconds: list[float] | None = None,
    aa_candidate_seconds: list[float] | None = None,
    stored_aa_signed_effects: list[float] | None = None,
    stored_aa_bootstrap_distribution: list[float] | None = None,
    methodology_overrides: dict[str, object] | None = None,
    graph_breaks: list[dict[str, str]] | None = None,
    recompiles: list[dict[str, str]] | None = None,
    case_count: int = 1,
    failed_accuracy_case: int | None = None,
    config_dynamic: bool | None = False,
    dynamic_strategy: str = "static",
    catalog: dict[str, object] | None = None,
    criteria: str = CRITERIA,
) -> None:
    """Store a complete literal evidence bundle and finalize its manifest."""
    def frozen_series(values: list[float]) -> list[float]:
        if len(values) == 31:
            return values
        if len(values) != 5:
            raise ValueError("fixture timing series must have five or 31 values")
        return (values * 7)[:31]

    compiled = frozen_series(
        compiled_seconds or [0.80, 0.81, 0.79, 0.80, 0.80]
    )
    aa_baseline = (
        compiled
        if aa_baseline_seconds is None
        else frozen_series(aa_baseline_seconds)
    )
    aa_candidate = (
        aa_baseline
        if aa_candidate_seconds is None
        else frozen_series(aa_candidate_seconds)
    )
    aa_signed_effects = (
        [
            0.0
            if baseline == candidate == 0
            else (baseline - candidate) / ((baseline + candidate) / 2)
            for baseline, candidate in zip(aa_baseline, aa_candidate, strict=True)
        ]
        if stored_aa_signed_effects is None
        else frozen_series(stored_aa_signed_effects)
    )
    aa_bootstrap_distribution = (
        [abs(sorted(aa_signed_effects)[len(aa_signed_effects) // 2])] * 2000
        if stored_aa_bootstrap_distribution is None
        else stored_aa_bootstrap_distribution
    )
    timestamp_origin = datetime(2026, 8, 29, 12, 0, tzinfo=UTC)

    def timestamp(offset: int) -> str:
        return (timestamp_origin + timedelta(seconds=offset)).isoformat().replace(
            "+00:00", "Z"
        )

    eager_timestamps: list[str] = []
    baseline_timestamps: list[str] = []
    candidate_timestamps: list[str] = []
    for iteration in range(len(aa_baseline)):
        offset = case_count + 1 + iteration * 3
        if iteration % 2 == 0:
            eager_timestamps.append(timestamp(offset))
            baseline_timestamps.append(timestamp(offset + 1))
            candidate_timestamps.append(timestamp(offset + 2))
        else:
            candidate_timestamps.append(timestamp(offset))
            baseline_timestamps.append(timestamp(offset + 1))
            eager_timestamps.append(timestamp(offset + 2))

    eager = (
        frozen_series(eager_seconds)
        if eager_seconds is not None
        else [1.0] * len(aa_baseline)
    )
    if case_count == 1:
        workload_name = "mlp_stack"
        shapes = [[48, 32]]
    elif case_count == 4:
        workload_name = "dynamic_batch_text"
        shapes = [[2, 24, 32], [1, 48, 32], [3, 16, 32], [4, 12, 32]]
    else:
        raise ValueError("fixture supports only pinned one- or four-case workloads")

    bundle_dir.mkdir()
    (bundle_dir / "gate_criteria.yaml").write_text(criteria)
    write_json(
        bundle_dir / "env.json",
        {
            "schema_version": 1,
            "torch": {
                "version": "2.13.0",
                "git_version": "cf30153c4c131c8164ee7798e5022d810682e2cb",
                "debug": False,
            },
            "python": {"implementation": "CPython", "version": "3.14.7"},
            "os": {"system": "Darwin", "release": "25.5.0"},
            "chip": "arm64",
            "boundary": EVIDENCE_BOUNDARY,
        },
    )
    write_json(
        bundle_dir / "run_provenance.json",
        {
            "schema_version": 1,
            "orchestration_id": "a" * 64,
            "run_id": "b" * 64,
            "sequence_index": 0,
            "predecessor": None,
        },
    )
    write_json(
        bundle_dir / "workload.digest",
        {
            "schema_version": 1,
            "name": workload_name,
            "sha256": V1_WORKLOAD_SHA256[workload_name],
            "accuracy_tolerance": {
                "dtype": "torch.float32",
                "atol": 1e-5,
                "rtol": 1e-5,
            },
        },
    )
    if catalog is None:
        default_id = (
            "planted-static-shape-storm"
            if workload_name == "dynamic_batch_text"
            else "clean-control-mlp"
        )
        catalog = catalog_metadata(get_catalog_entry(default_id))
    config: dict[str, object] = {
        "backend": "inductor",
        "mode": "default",
        "dynamic": config_dynamic,
        "fullgraph": True,
        "options": None,
        "disable": False,
    }
    effective = catalog["effective"]
    assert isinstance(effective, dict)
    config.update(
        {
            key: effective[key]
            for key in ("backend", "mode", "dynamic", "fullgraph", "options", "disable")
        }
    )
    config["dynamic"] = config_dynamic
    config["catalog"] = catalog
    write_json(bundle_dir / "config.json", config)
    write_json(
        bundle_dir / "input_plan.json",
        {
            "schema_version": 1,
            "dynamic_strategy": dynamic_strategy,
            "bucket_axis": None,
            "bucket_boundaries": None,
            "bucket_overflow_rule": None,
            "original_shapes": shapes,
            "effective_shapes": shapes,
            "compile_sweep_case_indices": list(range(case_count)),
            "steady_state_case_index": 0,
        },
    )
    write_json(
        bundle_dir / "timings.json",
        {
            "eager_seconds": eager,
            "eager_timestamps_utc": eager_timestamps,
            "compiled_seconds": compiled,
            "compiled_timestamps_utc": baseline_timestamps,
            "cold_compile_seconds": cold_compile_seconds,
            "cold_compile_timestamp_utc": timestamp(0),
            "warm_cache_compile_seconds": 0.1,
            "warm_cache_compile_timestamp_utc": timestamp(1),
            "non_primary_compile_sweep_case_indices": list(range(1, case_count)),
            "non_primary_compile_sweep_seconds": [0.1] * (case_count - 1),
            "non_primary_compile_sweep_timestamps_utc": [
                timestamp(case_index + 1) for case_index in range(1, case_count)
            ],
            "aa_baseline_seconds": aa_baseline,
            "aa_baseline_timestamps_utc": baseline_timestamps,
            "aa_candidate_seconds": aa_candidate,
            "aa_candidate_timestamps_utc": candidate_timestamps,
            "aa_signed_paired_effects": aa_signed_effects,
            "aa_bootstrap_absolute_medians": aa_bootstrap_distribution,
            "summary": {"speedup_lower_confidence_bound": 999.0},
        },
    )
    failed_case = 0 if not accuracy_valid else failed_accuracy_case
    accuracy_cases = [
        {
            "case_index": case_index,
            "within_tolerance": case_index != failed_case,
            "max_absolute_error": 1.0 if case_index == failed_case else 0.0,
            "mismatch": {"kind": "value"} if case_index == failed_case else None,
        }
        for case_index in range(case_count)
    ]
    write_json(
        bundle_dir / "accuracy.json",
        {
            "schema_version": 1,
            "dtype": "torch.float32",
            "within_tolerance": accuracy_valid,
            "case_index": 0 if failed_case is None else failed_case,
            "atol": 1e-5,
            "rtol": 1e-5,
            "max_absolute_error": 1.0 if failed_case is not None else 0.0,
            "mismatch": {"kind": "value"} if failed_case is not None else None,
            "cases": accuracy_cases,
        },
    )
    methodology: dict[str, object] = {
        "valid": methodology_valid,
        "warmup_runs": 5,
        "repeats": len(aa_baseline),
        "inter_run_spacing_seconds": 0.0,
        "bootstrap_samples": methodology_bootstrap_samples,
        "bootstrap_seed": 0,
        "bootstrap_confidence": 0.95,
        "aa_noise_floor": aa_noise_floor,
        "aa_effect_formula": "(A-B)/((A+B)/2)",
        "aa_estimator": "p95_absolute_bootstrap_median",
        "aa_pairing": "within_iteration",
        "measurement_schedule": "alternate_eager-A-B__B-A-eager",
        "quantile_method": "linear_interpolation",
        "compiler_state_reset": True,
        "compile_cache": v1_compile_cache_json(),
    }
    methodology.update(methodology_overrides or {})
    write_json(bundle_dir / "methodology.json", methodology)
    cold_metric = cache_metric(local_hit=0, local_miss=1)
    warm_metric = cache_metric(local_hit=1, local_miss=0)
    write_json(
        bundle_dir / "dynamo_report.json",
        {
            "schema_version": 1,
            "graph_breaks": graph_breaks or [],
            "recompiles": recompiles or [],
            "compilation_metrics": [warm_metric],
            "log_records": [],
            "cache_evidence": {
                "applicable": True,
                "reason": None,
                "cold_compilation_metrics": [cold_metric],
                "warm_cache_compilation_metrics": [warm_metric],
            },
            "path_normalization": v1_path_normalization_json(),
        },
    )
    finalize_bundle(bundle_dir, _evaluate_provisional_bundle)


def _rewrite_payload(
    bundle: Path, filename: str, mutate: Callable[[dict[str, object]], None]
) -> None:
    payload = json.loads((bundle / filename).read_bytes())
    assert isinstance(payload, dict)
    mutate(payload)
    write_json(bundle / filename, payload)
    (bundle / "verdict.json").unlink()
    write_manifest(bundle)
    write_json(bundle / "verdict.json", _evaluate_provisional_bundle(bundle))
    write_manifest(bundle)


def _finding_ids(verdict: dict[str, object]) -> list[str]:
    return [finding["id"] for finding in verdict["findings"]]  # type: ignore[index]


def test_valid_catalog_metadata_is_accepted_without_changing_gate_semantics(
    tmp_path: Path,
) -> None:
    """Authentic requested/effective metadata must preserve the underlying stored verdict."""
    metadata = catalog_metadata(get_catalog_entry("clean-control-mlp"))
    _write_bundle(
        tmp_path / "bundle",
        catalog=metadata,
        config_dynamic=False,
        dynamic_strategy="static",
    )

    verdict = evaluate_bundle(tmp_path / "bundle")

    assert verdict["verdict"] == "PROVEN"
    assert verdict["driving_finding"] == "all_criteria_passed"


def test_contradictory_catalog_metadata_invalidates_bundle(tmp_path: Path) -> None:
    """Changing requested settings while retaining measured config must be named invalid."""
    metadata = catalog_metadata(get_catalog_entry("clean-control-mlp"))
    metadata["requested"]["dynamic"] = "true"  # type: ignore[index]
    _write_bundle(tmp_path / "bundle", catalog=metadata)

    verdict = evaluate_bundle(tmp_path / "bundle")

    assert verdict == {
        "verdict": "INVALID_EVIDENCE",
        "driving_finding": "methodology.catalog",
        "findings": [{"id": "methodology.catalog", "status": "FAIL"}],
    }


def test_catalog_metadata_must_agree_with_actual_torch_request(tmp_path: Path) -> None:
    """A catalog label cannot describe a different top-level request than Torch received."""
    metadata = catalog_metadata(get_catalog_entry("clean-control-mlp"))
    _write_bundle(tmp_path / "bundle", catalog=metadata)
    _rewrite_payload(
        tmp_path / "bundle",
        "config.json",
        lambda config: config.__setitem__("fullgraph", False),
    )

    verdict = evaluate_bundle(tmp_path / "bundle")

    assert verdict["verdict"] == "INVALID_EVIDENCE"
    assert verdict["driving_finding"] == "methodology.catalog"


def test_catalog_child_gate_does_not_require_entry_to_remain_in_live_catalog(
    tmp_path: Path,
) -> None:
    """A historical bundle must gate from stored metadata after its live ID is removed."""
    metadata = catalog_metadata(get_catalog_entry("clean-control-mlp"))
    metadata["entry_id"] = "historical-clean-control"
    _write_bundle(tmp_path / "bundle", catalog=metadata)

    verdict = evaluate_bundle(tmp_path / "bundle")

    assert verdict["verdict"] == "PROVEN"
    assert verdict["driving_finding"] == "all_criteria_passed"


def test_accuracy_mismatch_invalidates_evidence_without_perf_findings(tmp_path: Path) -> None:
    """A wrong output must make all speed claims meaningless."""
    _write_bundle(tmp_path / "bundle", accuracy_valid=False)

    verdict = evaluate_bundle(tmp_path / "bundle")

    assert verdict["verdict"] == "INVALID_EVIDENCE"
    assert verdict["driving_finding"] == "accuracy.tolerance"
    assert not any(finding_id.startswith("perf.") for finding_id in _finding_ids(verdict))


def test_methodology_violation_invalidates_evidence_without_perf_findings(tmp_path: Path) -> None:
    """An explicitly invalid method must short-circuit the performance gate."""
    _write_bundle(tmp_path / "bundle", methodology_valid=False)

    verdict = evaluate_bundle(tmp_path / "bundle")

    assert verdict["verdict"] == "INVALID_EVIDENCE"
    assert verdict["driving_finding"] == "methodology.valid"
    assert not any(finding_id.startswith("perf.") for finding_id in _finding_ids(verdict))


def test_manifest_mismatch_invalidates_evidence_and_names_the_path(tmp_path: Path) -> None:
    """A tampered timing payload cannot receive a performance verdict."""
    bundle = tmp_path / "bundle"
    _write_bundle(bundle)
    (bundle / "timings.json").write_bytes(b"{}")

    verdict = evaluate_bundle(bundle)

    assert verdict["verdict"] == "INVALID_EVIDENCE"
    assert verdict["driving_finding"] == "evidence.integrity"
    assert verdict["findings"] == [
        {"id": "evidence.integrity", "mismatches": ["changed:timings.json"], "status": "FAIL"}
    ]


def test_speedup_lower_bound_below_threshold_is_not_proven(tmp_path: Path) -> None:
    """A measured speedup whose bootstrap lower bound misses 1.10 must fail hard."""
    _write_bundle(
        tmp_path / "bundle",
        eager_seconds=[1.00, 1.01, 0.99, 1.00, 1.00],
        compiled_seconds=[0.95, 0.96, 0.94, 0.95, 0.95],
    )

    verdict = evaluate_bundle(tmp_path / "bundle")

    assert verdict["verdict"] == "NOT_PROVEN"
    assert verdict["driving_finding"] == "perf.speedup_proven"


def test_nonfinite_derived_speed_statistics_invalidate_complete_bundle(
    tmp_path: Path,
) -> None:
    """Finite raw timings cannot authorize a verdict containing Infinity or NaN."""
    _write_bundle(
        tmp_path / "bundle",
        eager_seconds=[1.79e308] * 5,
        compiled_seconds=[1e-308] * 5,
    )

    verdict = evaluate_bundle(tmp_path / "bundle")

    assert verdict["verdict"] == "INVALID_EVIDENCE"
    assert verdict["driving_finding"] == "methodology.statistics"
    encoded = canonical_json_bytes(verdict)
    assert b"Infinity" not in encoded
    assert b"NaN" not in encoded


def test_nonstandard_nonfinite_json_constant_invalidates_manifested_bundle(
    tmp_path: Path,
) -> None:
    """Rehashing a literal NaN token cannot make non-standard JSON valid evidence."""
    bundle = tmp_path / "bundle"
    _write_bundle(bundle)
    timings_path = bundle / "timings.json"
    original = timings_path.read_bytes()
    tampered = original.replace(
        b'"speedup_lower_confidence_bound":999.0',
        b'"speedup_lower_confidence_bound":NaN',
    )
    assert tampered != original
    timings_path.write_bytes(tampered)
    (bundle / "verdict.json").unlink()
    write_manifest(bundle)
    write_json(bundle / "verdict.json", _evaluate_provisional_bundle(bundle))
    write_manifest(bundle)

    verdict = evaluate_bundle(bundle)

    assert verdict["verdict"] == "INVALID_EVIDENCE"
    assert verdict["driving_finding"] == "methodology.valid"


def test_cold_compile_time_over_workload_budget_is_not_proven(tmp_path: Path) -> None:
    """A fast steady state cannot hide a workload's compile-time cost."""
    _write_bundle(tmp_path / "bundle", cold_compile_seconds=30.01)

    verdict = evaluate_bundle(tmp_path / "bundle")

    assert verdict["verdict"] == "NOT_PROVEN"
    assert verdict["driving_finding"] == "perf.compile_budget"


def test_recompile_count_over_workload_bound_is_not_proven(tmp_path: Path) -> None:
    """The declared dynamic-shape recompile bound is a hard gate."""
    _write_bundle(tmp_path / "bundle", recompiles=[{"trigger": "shape changed"}])

    verdict = evaluate_bundle(tmp_path / "bundle")

    assert verdict["verdict"] == "NOT_PROVEN"
    assert verdict["driving_finding"] == "graph.recompile_bound"


def test_enumerated_graph_break_is_conditional(tmp_path: Path) -> None:
    """A reasoned graph break remains an explicit soft finding rather than a full pass."""
    _write_bundle(tmp_path / "bundle", graph_breaks=[{"reason": "Tensor.item()"}])

    verdict = evaluate_bundle(tmp_path / "bundle")

    assert verdict["verdict"] == "CONDITIONAL"
    assert verdict["driving_finding"] == "graph.no_breaks"


def test_all_hard_criteria_and_zero_graph_breaks_are_proven(tmp_path: Path) -> None:
    """Stored raw timings, not the false stored summary, determine a full pass."""
    _write_bundle(tmp_path / "bundle")

    verdict = evaluate_bundle(tmp_path / "bundle")

    assert verdict["verdict"] == "PROVEN"
    assert verdict["driving_finding"] == "all_criteria_passed"
    assert _finding_ids(verdict) == [
        "perf.speedup_proven",
        "perf.compile_budget",
        "graph.recompile_bound",
        "graph.no_breaks",
    ]


def test_same_stored_evidence_has_a_byte_identical_canonical_verdict(tmp_path: Path) -> None:
    """Offline re-gating must not add wall-clock data or random bootstrap output."""
    bundle = tmp_path / "bundle"
    _write_bundle(bundle)

    first = canonical_json_bytes(evaluate_bundle(bundle))
    second = canonical_json_bytes(evaluate_bundle(bundle))

    assert first == second


def test_noise_floor_must_be_strictly_below_the_configured_effect(tmp_path: Path) -> None:
    """A configured 10% effect is not credible against a larger valid raw A/A floor."""
    _write_bundle(
        tmp_path / "bundle",
        aa_noise_floor=2.0,
        compiled_seconds=[0.0, 0.0, 0.0, 0.0, 0.0],
        aa_baseline_seconds=[0.0, 0.0, 0.0, 0.0, 0.0],
        aa_candidate_seconds=[1.0, 1.0, 1.0, 1.0, 1.0],
    )

    verdict = evaluate_bundle(tmp_path / "bundle")

    assert verdict["verdict"] == "INVALID_EVIDENCE"
    assert verdict["driving_finding"] == "methodology.noise_floor"
    assert not any(finding_id.startswith("perf.") for finding_id in _finding_ids(verdict))


def test_gate_recomputes_raw_aa_effects_before_performance_findings(tmp_path: Path) -> None:
    """Stored signed effects cannot replace the two raw same-config timing series."""
    _write_bundle(
        tmp_path / "bundle",
        stored_aa_signed_effects=[0.01, 0.01, 0.01, 0.01, 0.01],
    )

    verdict = evaluate_bundle(tmp_path / "bundle")

    assert verdict["verdict"] == "INVALID_EVIDENCE"
    assert verdict["driving_finding"] == "methodology.aa_effects"
    assert not any(finding_id.startswith("perf.") for finding_id in _finding_ids(verdict))


def test_gate_preserves_large_finite_paired_effect_before_noise_decision(
    tmp_path: Path,
) -> None:
    """Overflow in A+B cannot erase a stored finite same-config effect."""
    effect = 0.1762917933130699
    _write_bundle(
        tmp_path / "bundle",
        compiled_seconds=[1.79e308] * 5,
        aa_baseline_seconds=[1.79e308] * 5,
        aa_candidate_seconds=[1.5e308] * 5,
        stored_aa_signed_effects=[effect] * 5,
        stored_aa_bootstrap_distribution=[effect] * 2000,
        aa_noise_floor=effect,
    )

    verdict = evaluate_bundle(tmp_path / "bundle")

    assert verdict["verdict"] == "INVALID_EVIDENCE"
    assert verdict["driving_finding"] == "methodology.noise_floor"


def test_gate_recomputes_aa_bootstrap_distribution_before_performance_findings(
    tmp_path: Path,
) -> None:
    """Stored bootstrap medians cannot replace deterministic resampling of raw effects."""
    _write_bundle(
        tmp_path / "bundle",
        stored_aa_bootstrap_distribution=[0.01] * 2000,
    )

    verdict = evaluate_bundle(tmp_path / "bundle")

    assert verdict["verdict"] == "INVALID_EVIDENCE"
    assert verdict["driving_finding"] == "methodology.aa_bootstrap"
    assert not any(finding_id.startswith("perf.") for finding_id in _finding_ids(verdict))


def test_gate_recomputes_bootstrap_p95_before_performance_findings(tmp_path: Path) -> None:
    """Stored floor must exactly equal p95 of the validated bootstrap distribution."""
    _write_bundle(tmp_path / "bundle", aa_noise_floor=0.01)

    verdict = evaluate_bundle(tmp_path / "bundle")

    assert verdict["verdict"] == "INVALID_EVIDENCE"
    assert verdict["driving_finding"] == "methodology.noise_floor"
    assert not any(finding_id.startswith("perf.") for finding_id in _finding_ids(verdict))


def test_isolated_paired_outlier_does_not_exceed_ten_percent_noise_floor(
    tmp_path: Path,
) -> None:
    """One scheduler outlier cannot dominate the bootstrap median of a zero-effect run."""
    baseline = [1.0] * 31
    candidate = [1.0] * 31
    candidate[0] = 2.0
    effects = [0.0] * 31
    effects[0] = -2.0 / 3.0
    _write_bundle(
        tmp_path / "bundle",
        compiled_seconds=baseline,
        aa_baseline_seconds=baseline,
        aa_candidate_seconds=candidate,
        stored_aa_signed_effects=effects,
        stored_aa_bootstrap_distribution=[0.0] * 2000,
        aa_noise_floor=0.0,
    )

    verdict = evaluate_bundle(tmp_path / "bundle")

    assert verdict["driving_finding"] != "methodology.noise_floor"
    assert verdict["verdict"] != "INVALID_EVIDENCE"


def test_systematic_paired_effect_exceeds_ten_percent_noise_floor(tmp_path: Path) -> None:
    """Bootstrap medians must retain a systematic same-config timing difference."""
    baseline = [1.0] * 31
    candidate = [0.8] * 31
    effect = (1.0 - 0.8) / ((1.0 + 0.8) / 2.0)
    _write_bundle(
        tmp_path / "bundle",
        compiled_seconds=baseline,
        aa_baseline_seconds=baseline,
        aa_candidate_seconds=candidate,
        stored_aa_signed_effects=[effect] * 31,
        stored_aa_bootstrap_distribution=[abs(effect)] * 2000,
        aa_noise_floor=abs(effect),
    )

    verdict = evaluate_bundle(tmp_path / "bundle")

    assert verdict["verdict"] == "INVALID_EVIDENCE"
    assert verdict["driving_finding"] == "methodology.noise_floor"
    assert not any(finding_id.startswith("perf.") for finding_id in _finding_ids(verdict))


def test_gate_binds_compiled_performance_timings_to_aa_baseline(tmp_path: Path) -> None:
    """Unrelated zero-noise A/A timings cannot authorize a compiled performance claim."""
    _write_bundle(
        tmp_path / "bundle",
        compiled_seconds=[0.80, 0.81, 0.79, 0.80, 0.80],
        aa_baseline_seconds=[1.0, 1.0, 1.0, 1.0, 1.0],
        aa_candidate_seconds=[1.0, 1.0, 1.0, 1.0, 1.0],
    )

    verdict = evaluate_bundle(tmp_path / "bundle")

    assert verdict["verdict"] == "INVALID_EVIDENCE"
    assert verdict["driving_finding"] == "methodology.aa_baseline"
    assert not any(finding_id.startswith("perf.") for finding_id in _finding_ids(verdict))


def test_gate_requires_declared_bootstrap_samples_to_match_frozen_criteria(
    tmp_path: Path,
) -> None:
    """Stored methodology cannot claim fewer samples than offline gating actually computes."""
    _write_bundle(tmp_path / "bundle", methodology_bootstrap_samples=1999)

    verdict = evaluate_bundle(tmp_path / "bundle")

    assert verdict["verdict"] == "INVALID_EVIDENCE"
    assert verdict["driving_finding"] == "methodology.bootstrap_samples"
    assert not any(finding_id.startswith("perf.") for finding_id in _finding_ids(verdict))


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("original_shapes", [[]]),
        ("compile_sweep_case_indices", [1]),
        ("steady_state_case_index", 1),
        ("steady_state_case_index", False),
    ],
)
def test_gate_rejects_semantically_tampered_input_plan(
    tmp_path: Path, field: str, value: object
) -> None:
    """A re-manifested malformed shape or sweep declaration cannot authorize results."""
    bundle = tmp_path / "bundle"
    _write_bundle(bundle)
    _rewrite_payload(bundle, "input_plan.json", lambda plan: plan.__setitem__(field, value))

    verdict = evaluate_bundle(bundle)

    assert verdict["verdict"] == "INVALID_EVIDENCE"
    assert verdict["driving_finding"] == "methodology.input_plan"
    assert not any(finding_id.startswith("perf.") for finding_id in _finding_ids(verdict))


def test_gate_binds_declared_repeats_to_every_steady_series(tmp_path: Path) -> None:
    """A declared repeat count cannot disagree with otherwise valid raw sample lengths."""
    bundle = tmp_path / "bundle"
    _write_bundle(bundle, methodology_overrides={"repeats": 4})

    verdict = evaluate_bundle(bundle)

    assert verdict["verdict"] == "INVALID_EVIDENCE"
    assert verdict["driving_finding"] == "methodology.repeats"
    assert not any(finding_id.startswith("perf.") for finding_id in _finding_ids(verdict))


def test_gate_rejects_deleted_steady_timestamp(tmp_path: Path) -> None:
    """Every duration must retain its own UTC timestamp for offline schedule checking."""
    bundle = tmp_path / "bundle"
    _write_bundle(bundle)
    _rewrite_payload(bundle, "timings.json", lambda timings: timings["eager_timestamps_utc"].pop())

    verdict = evaluate_bundle(bundle)

    assert verdict["verdict"] == "INVALID_EVIDENCE"
    assert verdict["driving_finding"] == "methodology.timestamps"
    assert not any(finding_id.startswith("perf.") for finding_id in _finding_ids(verdict))


def test_gate_rejects_timestamp_order_inconsistent_with_alternating_schedule(
    tmp_path: Path,
) -> None:
    """Series-local timestamps cannot conceal a non-alternating execution order."""
    bundle = tmp_path / "bundle"
    _write_bundle(bundle)

    def put_baseline_before_eager(timings: dict[str, object]) -> None:
        timestamps = timings["aa_baseline_timestamps_utc"]
        assert isinstance(timestamps, list)
        timestamps[0] = "2026-08-29T12:00:00.500000Z"
        timings["compiled_timestamps_utc"] = timestamps

    _rewrite_payload(bundle, "timings.json", put_baseline_before_eager)

    verdict = evaluate_bundle(bundle)

    assert verdict["verdict"] == "INVALID_EVIDENCE"
    assert verdict["driving_finding"] == "methodology.timestamps"
    assert not any(finding_id.startswith("perf.") for finding_id in _finding_ids(verdict))


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("non_primary_compile_sweep_case_indices", [1, 3]),
        ("non_primary_compile_sweep_seconds", [0.1, 0.1]),
        (
            "non_primary_compile_sweep_timestamps_utc",
            ["2026-08-29T12:00:01Z", "2026-08-29T12:00:02Z"],
        ),
    ],
)
def test_gate_rejects_incomplete_non_primary_compile_sweep(
    tmp_path: Path, field: str, value: object
) -> None:
    """All non-primary cases need matching ordered index, duration, and timestamp evidence."""
    bundle = tmp_path / "bundle"
    _write_bundle(bundle, case_count=4)
    _rewrite_payload(bundle, "timings.json", lambda timings: timings.__setitem__(field, value))

    verdict = evaluate_bundle(bundle)

    assert verdict["verdict"] == "INVALID_EVIDENCE"
    assert verdict["driving_finding"] == "methodology.compile_sweep"
    assert not any(finding_id.startswith("perf.") for finding_id in _finding_ids(verdict))


def test_gate_rejects_non_primary_accuracy_failure_hidden_by_true_aggregate(
    tmp_path: Path,
) -> None:
    """A true summary cannot erase a failed output from a swept non-primary case."""
    bundle = tmp_path / "bundle"
    _write_bundle(bundle, case_count=4, failed_accuracy_case=1)

    verdict = evaluate_bundle(bundle)

    assert verdict["verdict"] == "INVALID_EVIDENCE"
    assert verdict["driving_finding"] == "accuracy.tolerance"
    assert not any(finding_id.startswith("perf.") for finding_id in _finding_ids(verdict))


def test_gate_rejects_missing_non_primary_accuracy_record(tmp_path: Path) -> None:
    """An aggregate cannot be evaluated without one ordered record per swept case."""
    bundle = tmp_path / "bundle"
    _write_bundle(bundle, case_count=4)

    def remove_last_case(accuracy: dict[str, object]) -> None:
        records = accuracy["cases"]
        assert isinstance(records, list)
        records.pop()

    _rewrite_payload(bundle, "accuracy.json", remove_last_case)

    verdict = evaluate_bundle(bundle)

    assert verdict["verdict"] == "INVALID_EVIDENCE"
    assert verdict["driving_finding"] == "accuracy.tolerance"
    assert not any(finding_id.startswith("perf.") for finding_id in _finding_ids(verdict))


def test_gate_rejects_auto_dynamic_outside_the_closed_catalog(tmp_path: Path) -> None:
    """Normal evidence admits only exact catalog-bound effective boolean dynamic values."""
    bundle = tmp_path / "bundle"
    _write_bundle(bundle, config_dynamic=None, dynamic_strategy="auto")

    verdict = evaluate_bundle(bundle)

    assert verdict["verdict"] == "INVALID_EVIDENCE"
    assert verdict["driving_finding"] == "methodology.config"


@pytest.mark.parametrize(
    ("config_dynamic", "dynamic_strategy", "finding"),
    [
        (None, "static", "methodology.config"),
        (False, "auto", "methodology.input_plan"),
        (True, "auto", "methodology.input_plan"),
    ],
)
def test_gate_rejects_auto_dynamic_config_plan_mismatch(
    tmp_path: Path,
    config_dynamic: bool | None,
    dynamic_strategy: str,
    finding: str,
) -> None:
    """Auto identity is valid only when the stored compile dynamic flag is null."""
    bundle = tmp_path / "bundle"
    _write_bundle(
        bundle,
        config_dynamic=config_dynamic,
        dynamic_strategy=dynamic_strategy,
    )

    verdict = evaluate_bundle(bundle)

    assert verdict["verdict"] == "INVALID_EVIDENCE"
    assert verdict["driving_finding"] == finding


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("bootstrap_seed", 1),
        ("bootstrap_confidence", 0.90),
        ("aa_effect_formula", "abs(A-B)"),
        ("aa_estimator", "bare_p95"),
        ("aa_pairing", "sequential_series"),
        ("measurement_schedule", "eager_then_A_then_B"),
        ("quantile_method", "nearest"),
    ],
)
def test_gate_rejects_drifted_aa_methodology_metadata(
    tmp_path: Path,
    field: str,
    value: object,
) -> None:
    """Offline gating must bind every declared estimator and schedule decision."""
    _write_bundle(tmp_path / "bundle", methodology_overrides={field: value})

    verdict = evaluate_bundle(tmp_path / "bundle")

    assert verdict["verdict"] == "INVALID_EVIDENCE"
    assert verdict["driving_finding"] == "methodology.valid"
    assert not any(finding_id.startswith("perf.") for finding_id in _finding_ids(verdict))


def test_gate_criteria_fixture_is_valid_yaml() -> None:
    """The literal criteria stored by each test bundle are parseable evidence."""
    assert yaml.safe_load(CRITERIA)["minimum_speedup"] == 1.10


@pytest.mark.parametrize(
    "drifted_criteria",
    [
        CRITERIA.replace("minimum_speedup: 1.10", "minimum_speedup: 1.09"),
        CRITERIA.replace("samples: 2000", "samples: 1999"),
        CRITERIA.replace("confidence: 0.95", "confidence: 0.94"),
        CRITERIA.replace("seed: 0", "seed: 1"),
        CRITERIA.replace("mlp_stack: 30.0", "mlp_stack: 31.0"),
        CRITERIA.replace("mlp_stack: 0", "mlp_stack: 1"),
        f"{CRITERIA}unrecognized_setting: true\n",
    ],
)
def test_drift_from_any_frozen_v1_criteria_value_invalidates_evidence(
    tmp_path: Path, drifted_criteria: str
) -> None:
    """A self-consistent local criteria copy cannot weaken frozen v1 gate requirements."""
    _write_bundle(tmp_path / "bundle", criteria=drifted_criteria)

    verdict = evaluate_bundle(tmp_path / "bundle")

    assert verdict["verdict"] == "INVALID_EVIDENCE"
    assert verdict["driving_finding"] == "methodology.valid"
    assert not any(finding_id.startswith("perf.") for finding_id in _finding_ids(verdict))


def test_accuracy_mismatch_precedes_malformed_graph_evidence(tmp_path: Path) -> None:
    """Accuracy remains the driver before hard or soft evidence is parsed."""
    _write_bundle(tmp_path / "bundle", accuracy_valid=False, graph_breaks=[{}])  # type: ignore[list-item]

    verdict = evaluate_bundle(tmp_path / "bundle")

    assert verdict["verdict"] == "INVALID_EVIDENCE"
    assert verdict["driving_finding"] == "accuracy.tolerance"
    assert not any(finding_id.startswith("perf.") for finding_id in _finding_ids(verdict))


def test_unreasoned_graph_break_invalidates_evidence_before_hard_failure(tmp_path: Path) -> None:
    """Required graph reasons are validity evidence, even when speedup already fails."""
    _write_bundle(
        tmp_path / "bundle",
        eager_seconds=[1.00, 1.01, 0.99, 1.00, 1.00],
        compiled_seconds=[0.95, 0.96, 0.94, 0.95, 0.95],
        graph_breaks=[{}],  # type: ignore[list-item]
    )

    verdict = evaluate_bundle(tmp_path / "bundle")

    assert verdict["verdict"] == "INVALID_EVIDENCE"
    assert verdict["driving_finding"] == "methodology.valid"
    assert not any(finding_id.startswith("perf.") for finding_id in _finding_ids(verdict))

from __future__ import annotations

import hashlib
import inspect
import json
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

import hephaestus.gate as gate_module
from hephaestus.bundle import canonical_json_bytes, write_json, write_manifest
from hephaestus.catalog import catalog_metadata, get_catalog_entry
from hephaestus.gate import evaluate_bundle
from hephaestus.scope import EVIDENCE_BOUNDARY

CRITERIA_PATH = Path(__file__).parents[1] / "gates" / "default.yaml"
RUN_ID = "b" * 64
ORCHESTRATION_ID = "a" * 64
WORKLOAD_SHA = "eeb39c122ab98b19f2248e8413322164c67a6c14bef6786a8e364a9d55b2c139"

COMPILE_CACHE = {
    "schema_version": 1,
    "policy": "fresh-per-run-local-filesystem-v1",
    "binding": "torch._inductor.utils.fresh_cache",
    "inductor_memory_cleared_between_phases": True,
    "local_fx_graph_cache": True,
    "local_aotautograd_cache": True,
    "remote_caches": False,
    "cpp_precompiled_headers": False,
    "warm_probe": "same-filesystem-cache-after-dynamo-and-inductor-memory-reset",
}
PATH_NORMALIZATION = {
    "schema_version": 1,
    "policy": "semantic-root-tokens-v1",
    "verbatim_fields": ["graph_breaks[].reason", "recompiles[].trigger"],
}
INDUCTOR_CACHE_CONFIG = {
    "fx_graph_cache": True,
    "fx_graph_remote_cache": False,
    "autotune_remote_cache": False,
    "bundled_autotune_remote_cache": False,
    "remote_gemm_autotune_cache": False,
    "force_disable_caches": False,
    "cpp_cache_precompile_headers": False,
}
FUNCTORCH_CACHE_CONFIG = {
    "enable_autograd_cache": True,
    "enable_remote_autograd_cache": False,
}


def _metric(*, local_hit: int, local_miss: int) -> dict[str, object]:
    return {
        "inductor_fx_local_cache_hit_count": local_hit,
        "inductor_fx_local_cache_miss_count": local_miss,
        "inductor_fx_remote_cache_hit_count": 0,
        "inductor_fx_remote_cache_miss_count": 0,
        "aotautograd_local_cache_hit_count": local_hit,
        "aotautograd_local_cache_miss_count": local_miss,
        "aotautograd_remote_cache_hit_count": 0,
        "aotautograd_remote_cache_miss_count": 0,
        "compiler_config": json.dumps({"force_disable_caches": False}, sort_keys=True),
        "inductor_config": json.dumps(INDUCTOR_CACHE_CONFIG, sort_keys=True),
        "functorch_config": json.dumps(FUNCTORCH_CACHE_CONFIG, sort_keys=True),
    }


def _timestamp(origin: datetime, offset: int) -> str:
    return (origin + timedelta(seconds=offset)).isoformat().replace("+00:00", "Z")


def _write_complete_bundle(bundle: Path, *, final: bool = True) -> None:
    bundle.mkdir()
    metadata = catalog_metadata(get_catalog_entry("clean-control-mlp"))
    origin = datetime(2026, 8, 29, 12, 0, tzinfo=UTC)
    eager_timestamps: list[str] = []
    baseline_timestamps: list[str] = []
    candidate_timestamps: list[str] = []
    for iteration in range(31):
        offset = 2 + iteration * 3
        if iteration % 2 == 0:
            eager_timestamps.append(_timestamp(origin, offset))
            baseline_timestamps.append(_timestamp(origin, offset + 1))
            candidate_timestamps.append(_timestamp(origin, offset + 2))
        else:
            candidate_timestamps.append(_timestamp(origin, offset))
            baseline_timestamps.append(_timestamp(origin, offset + 1))
            eager_timestamps.append(_timestamp(origin, offset + 2))

    write_json(
        bundle / "env.json",
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
        bundle / "run_provenance.json",
        {
            "schema_version": 1,
            "orchestration_id": ORCHESTRATION_ID,
            "run_id": RUN_ID,
            "sequence_index": 0,
            "predecessor": None,
        },
    )
    write_json(
        bundle / "workload.digest",
        {
            "schema_version": 1,
            "name": "mlp_stack",
            "sha256": WORKLOAD_SHA,
            "accuracy_tolerance": {
                "dtype": "torch.float32",
                "atol": 1e-5,
                "rtol": 1e-5,
            },
        },
    )
    effective = metadata["effective"]
    assert isinstance(effective, dict)
    write_json(
        bundle / "config.json",
        {
            **{
                key: effective[key]
                for key in ("backend", "mode", "dynamic", "fullgraph", "options", "disable")
            },
            "catalog": metadata,
        },
    )
    write_json(
        bundle / "input_plan.json",
        {
            "schema_version": 1,
            "dynamic_strategy": "static",
            "bucket_axis": None,
            "bucket_boundaries": None,
            "bucket_overflow_rule": None,
            "original_shapes": [[48, 32]],
            "effective_shapes": [[48, 32]],
            "compile_sweep_case_indices": [0],
            "steady_state_case_index": 0,
        },
    )
    write_json(
        bundle / "timings.json",
        {
            "eager_seconds": [1.0] * 31,
            "eager_timestamps_utc": eager_timestamps,
            "cold_compile_seconds": 1.0,
            "cold_compile_timestamp_utc": _timestamp(origin, 0),
            "warm_cache_compile_seconds": 0.1,
            "warm_cache_compile_timestamp_utc": _timestamp(origin, 1),
            "non_primary_compile_sweep_case_indices": [],
            "non_primary_compile_sweep_seconds": [],
            "non_primary_compile_sweep_timestamps_utc": [],
            "compiled_seconds": [0.8] * 31,
            "compiled_timestamps_utc": baseline_timestamps,
            "aa_baseline_seconds": [0.8] * 31,
            "aa_baseline_timestamps_utc": baseline_timestamps,
            "aa_candidate_seconds": [0.8] * 31,
            "aa_candidate_timestamps_utc": candidate_timestamps,
            "aa_signed_paired_effects": [0.0] * 31,
            "aa_bootstrap_absolute_medians": [0.0] * 2000,
            "summary": {},
        },
    )
    write_json(
        bundle / "accuracy.json",
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
                    "case_index": 0,
                    "within_tolerance": True,
                    "max_absolute_error": 0.0,
                    "mismatch": None,
                }
            ],
        },
    )
    write_json(
        bundle / "methodology.json",
        {
            "valid": True,
            "warmup_runs": 5,
            "repeats": 31,
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
            "compiler_state_reset": True,
            "compile_cache": COMPILE_CACHE,
        },
    )
    cold_metric = _metric(local_hit=0, local_miss=1)
    warm_metric = _metric(local_hit=1, local_miss=0)
    write_json(
        bundle / "dynamo_report.json",
        {
            "schema_version": 1,
            "graph_breaks": [],
            "recompiles": [],
            "compilation_metrics": [warm_metric],
            "log_records": [],
            "cache_evidence": {
                "applicable": True,
                "reason": None,
                "cold_compilation_metrics": [cold_metric],
                "warm_cache_compilation_metrics": [warm_metric],
            },
            "path_normalization": PATH_NORMALIZATION,
        },
    )
    (bundle / "gate_criteria.yaml").write_bytes(CRITERIA_PATH.read_bytes())
    write_manifest(bundle)
    if final:
        provisional = gate_module._evaluate_provisional_bundle(bundle)
        write_json(bundle / "verdict.json", provisional)
        write_manifest(bundle)


def _rewrite(bundle: Path, filename: str, mutate: Callable[[dict[str, object]], object]) -> None:
    value = json.loads((bundle / filename).read_bytes())
    assert isinstance(value, dict)
    mutate(value)
    write_json(bundle / filename, value)
    verdict_path = bundle / "verdict.json"
    verdict_path.unlink()
    write_manifest(bundle)
    write_json(verdict_path, gate_module._evaluate_provisional_bundle(bundle))
    write_manifest(bundle)


def test_public_gate_requires_exact_final_topology_and_private_provisional_path(
    tmp_path: Path,
) -> None:
    """Only the normal finalizer may evaluate the exact verdict-less provisional topology."""
    provisional = tmp_path / "provisional"
    _write_complete_bundle(provisional, final=False)

    public = evaluate_bundle(provisional)
    private_evaluator = getattr(gate_module, "_evaluate_provisional_bundle", None)

    assert "_allow_provisional" not in inspect.signature(evaluate_bundle).parameters
    assert public["verdict"] == "INVALID_EVIDENCE"
    assert public["driving_finding"] == "evidence.topology"
    assert callable(private_evaluator)
    assert private_evaluator(provisional)["driving_finding"] != "evidence.topology"


@pytest.mark.parametrize(
    "filename",
    [
        "accuracy.json",
        "config.json",
        "dynamo_report.json",
        "env.json",
        "gate_criteria.yaml",
        "input_plan.json",
        "methodology.json",
        "run_provenance.json",
        "timings.json",
        "verdict.json",
        "workload.digest",
    ],
)
def test_final_normal_bundle_rejects_each_missing_payload(
    tmp_path: Path,
    filename: str,
) -> None:
    bundle = tmp_path / "bundle"
    _write_complete_bundle(bundle)
    (bundle / filename).unlink()
    write_manifest(bundle)

    verdict = evaluate_bundle(bundle)

    assert verdict["verdict"] == "INVALID_EVIDENCE"
    assert verdict["driving_finding"] == "evidence.topology"


def test_final_normal_bundle_rejects_manifested_unexpected_payload(tmp_path: Path) -> None:
    bundle = tmp_path / "bundle"
    _write_complete_bundle(bundle)
    (bundle / "unexpected.txt").write_text("manifested but unauthorized")
    write_manifest(bundle)

    verdict = evaluate_bundle(bundle)

    assert verdict["verdict"] == "INVALID_EVIDENCE"
    assert verdict["driving_finding"] == "evidence.topology"


def test_final_normal_bundle_rejects_semantically_forged_stored_verdict(tmp_path: Path) -> None:
    bundle = tmp_path / "bundle"
    _write_complete_bundle(bundle)
    write_json(
        bundle / "verdict.json",
        {"verdict": "PROVEN", "driving_finding": "forged", "findings": []},
    )
    write_manifest(bundle)

    verdict = evaluate_bundle(bundle)

    assert verdict["verdict"] == "INVALID_EVIDENCE"
    assert verdict["driving_finding"] == "bundle.verdict"


@pytest.mark.parametrize(
    ("filename", "mutation", "finding"),
    [
        (
            "env.json",
            lambda value: value["torch"].__setitem__("git_version", "0" * 40),
            "environment.valid",
        ),
        (
            "workload.digest",
            lambda value: value.__setitem__("sha256", "f" * 64),
            "workload.identity",
        ),
        (
            "workload.digest",
            lambda value: value["accuracy_tolerance"].__setitem__("atol", 999.0),
            "accuracy.contract",
        ),
        (
            "accuracy.json",
            lambda value: (
                value.__setitem__("atol", 999.0),
                value.__setitem__("rtol", 999.0),
            ),
            "accuracy.contract",
        ),
        (
            "config.json",
            lambda value: value.pop("catalog"),
            "methodology.config",
        ),
        (
            "run_provenance.json",
            lambda value: value.__setitem__("run_id", "a" * 63),
            "run.provenance",
        ),
        (
            "methodology.json",
            lambda value: value["compile_cache"].__setitem__("remote_caches", True),
            "methodology.compile_cache",
        ),
        (
            "dynamo_report.json",
            lambda value: value["cache_evidence"]["warm_cache_compilation_metrics"][
                0
            ].__setitem__("inductor_fx_local_cache_hit_count", 0),
            "methodology.cache_evidence",
        ),
        (
            "dynamo_report.json",
            lambda value: value["log_records"].append(
                {
                    "name": "torch._dynamo",
                    "level": "INFO",
                    "message": "/Users/private/x.py",
                }
            ),
            "evidence.privacy",
        ),
    ],
)
def test_normal_contract_tampering_fails_closed_before_scientific_findings(
    tmp_path: Path,
    filename: str,
    mutation: Callable[[dict[str, object]], object],
    finding: str,
) -> None:
    bundle = tmp_path / "bundle"
    _write_complete_bundle(bundle)
    _rewrite(bundle, filename, mutation)

    verdict = evaluate_bundle(bundle)

    assert verdict["verdict"] == "INVALID_EVIDENCE"
    assert verdict["driving_finding"] == finding


def _set_unhashable_accuracy_mismatch(value: dict[str, object]) -> None:
    value["within_tolerance"] = False
    value["mismatch"] = {"kind": []}
    records = value["cases"]
    assert isinstance(records, list) and isinstance(records[0], dict)
    records[0]["within_tolerance"] = False
    records[0]["mismatch"] = {"kind": []}


def _erase_all_passing_accuracy_errors(value: dict[str, object]) -> None:
    value["max_absolute_error"] = None
    records = value["cases"]
    assert isinstance(records, list)
    for record in records:
        assert isinstance(record, dict) and record["within_tolerance"] is True
        record["max_absolute_error"] = None


def test_passing_accuracy_requires_measured_errors_and_exact_aggregate(
    tmp_path: Path,
) -> None:
    """A passing bundle cannot erase every measured error and bind the aggregate to null."""
    bundle = tmp_path / "bundle"
    _write_complete_bundle(bundle)
    _rewrite(bundle, "accuracy.json", _erase_all_passing_accuracy_errors)

    verdict = evaluate_bundle(bundle)

    assert verdict["verdict"] == "INVALID_EVIDENCE"
    assert verdict["driving_finding"] == "accuracy.tolerance"


@pytest.mark.parametrize(
    ("mutation", "finding"),
    [
        (lambda value: value.__setitem__("unexpected", "accepted"), "accuracy.contract"),
        (lambda value: value.__setitem__("dtype", "torch.float64"), "accuracy.contract"),
        (lambda value: value.__setitem__("case_index", 1), "accuracy.tolerance"),
        (_set_unhashable_accuracy_mismatch, "accuracy.tolerance"),
        (
            lambda value: value.__setitem__("diagnostic", "/Users/private/accuracy.py"),
            "evidence.privacy",
        ),
    ],
    ids=(
        "exact-schema",
        "dtype-binding",
        "aggregate-binding",
        "mismatch-schema",
        "privacy",
    ),
)
def test_accuracy_evidence_is_exact_bound_and_public(
    tmp_path: Path,
    mutation: Callable[[dict[str, object]], object],
    finding: str,
) -> None:
    """Accuracy evidence cannot add fields, change its contract, or carry private data."""
    bundle = tmp_path / "bundle"
    _write_complete_bundle(bundle)
    _rewrite(bundle, "accuracy.json", mutation)

    verdict = evaluate_bundle(bundle)

    assert verdict["verdict"] == "INVALID_EVIDENCE"
    assert verdict["driving_finding"] == finding


def test_normal_bundle_literal_is_manifested_and_self_consistent(tmp_path: Path) -> None:
    """The independent fixture must not hide expectations behind producer helpers."""
    bundle = tmp_path / "bundle"
    _write_complete_bundle(bundle)

    assert hashlib.sha256((bundle / "workload.digest").read_bytes()).hexdigest()
    assert (bundle / "verdict.json").read_bytes() == canonical_json_bytes(
        evaluate_bundle(bundle)
    )


def test_cache_config_accepts_torch_213_metric_field_ownership(tmp_path: Path) -> None:
    """Torch records the Inductor cache-disable alias in compiler_config only."""
    bundle = tmp_path / "bundle"
    _write_complete_bundle(bundle)

    def match_torch_metric_layout(value: dict[str, object]) -> None:
        cache = value["cache_evidence"]
        assert isinstance(cache, dict)
        metric_lists = [
            value["compilation_metrics"],
            cache["cold_compilation_metrics"],
            cache["warm_cache_compilation_metrics"],
        ]
        for metrics in metric_lists:
            assert isinstance(metrics, list)
            for metric in metrics:
                assert isinstance(metric, dict)
                config = json.loads(metric["inductor_config"])
                assert isinstance(config, dict)
                config.pop("force_disable_caches")
                metric["inductor_config"] = json.dumps(config, sort_keys=True)

    _rewrite(bundle, "dynamo_report.json", match_torch_metric_layout)

    assert evaluate_bundle(bundle)["verdict"] != "INVALID_EVIDENCE"

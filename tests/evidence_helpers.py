"""Exact schema-1 normal-evidence fixtures shared by recursive-parent tests."""

from __future__ import annotations

import json
from collections.abc import Mapping
from datetime import UTC, datetime, timedelta
from pathlib import Path

from hephaestus.bundle import finalize_bundle, write_json
from hephaestus.evidence_contract import (
    V1_FUNCTORCH_CACHE_PATCH,
    V1_INDUCTOR_CACHE_PATCH,
    V1_WORKLOAD_SHA256,
    v1_compile_cache_json,
    v1_path_normalization_json,
)
from hephaestus.gate import _evaluate_provisional_bundle
from hephaestus.input_plan import input_plan_json
from hephaestus.measure import RunSettings
from hephaestus.provenance import RunProvenance
from hephaestus.scope import EVIDENCE_BOUNDARY

ROOT = Path(__file__).parents[1]


def write_normal_child(
    child: Path,
    *,
    workload_name: str,
    request: object,
    plan: object,
    metadata: Mapping[str, object],
    driving_finding: str,
    settings: RunSettings | None = None,
    run_provenance: RunProvenance | None = None,
    accuracy_valid: bool = True,
) -> None:
    settings = RunSettings() if settings is None else settings
    provenance = run_provenance or RunProvenance(
        orchestration_id="a" * 64,
        run_id="b" * 64,
        sequence_index=0,
        predecessor=None,
    )
    plan_payload = input_plan_json(plan)  # type: ignore[arg-type]
    case_indices = list(plan_payload["compile_sweep_case_indices"])  # type: ignore[arg-type]
    case_count = len(case_indices)
    repeats = settings.repeats
    disabled = request.disable is True  # type: ignore[attr-defined]
    origin = datetime(2026, 8, 29, 12, 0, tzinfo=UTC)

    def timestamp(offset: int) -> str:
        return (origin + timedelta(seconds=offset)).isoformat().replace("+00:00", "Z")

    operational_offset = 1 if disabled else 2
    sweep_offsets = list(range(operational_offset, operational_offset + case_count - 1))
    first_iteration = operational_offset + case_count - 1
    eager_timestamps: list[str] = []
    baseline_timestamps: list[str] = []
    candidate_timestamps: list[str] = []
    for iteration in range(repeats):
        offset = first_iteration + iteration * 3
        if iteration % 2 == 0:
            eager_timestamps.append(timestamp(offset))
            baseline_timestamps.append(timestamp(offset + 1))
            candidate_timestamps.append(timestamp(offset + 2))
        elif settings.schema_version == 2:
            eager_timestamps.append(timestamp(offset))
            candidate_timestamps.append(timestamp(offset + 1))
            baseline_timestamps.append(timestamp(offset + 2))
        else:
            candidate_timestamps.append(timestamp(offset))
            baseline_timestamps.append(timestamp(offset + 1))
            eager_timestamps.append(timestamp(offset + 2))

    compiled_seconds = (
        [1.0] * repeats
        if driving_finding in {"perf.speedup_proven", "accuracy.tolerance"}
        else [0.8] * repeats
    )
    effective = metadata["effective"]
    assert isinstance(effective, Mapping)
    write_json(
        child / "env.json",
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
    write_json(child / "run_provenance.json", provenance.as_json())
    write_json(
        child / "workload.digest",
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
    write_json(
        child / "config.json",
        {
            **{
                key: effective[key]
                for key in ("backend", "mode", "dynamic", "fullgraph", "options", "disable")
            },
            "catalog": dict(metadata),
        },
    )
    write_json(child / "input_plan.json", plan_payload)
    write_json(
        child / "timings.json",
        {
            "eager_seconds": [1.0] * repeats,
            "eager_timestamps_utc": eager_timestamps,
            "cold_compile_seconds": 1.0,
            "cold_compile_timestamp_utc": timestamp(0),
            "warm_cache_compile_seconds": None if disabled else 0.1,
            "warm_cache_compile_timestamp_utc": None if disabled else timestamp(1),
            "non_primary_compile_sweep_case_indices": case_indices[1:],
            "non_primary_compile_sweep_seconds": [0.1] * (case_count - 1),
            "non_primary_compile_sweep_timestamps_utc": [
                timestamp(offset) for offset in sweep_offsets
            ],
            "compiled_seconds": compiled_seconds,
            "compiled_timestamps_utc": baseline_timestamps,
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
    methodology: dict[str, object] = {
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
        "aa_pairing": (
            "within_iteration"
            if settings.schema_version == 1
            else "position_matched_cross_parity"
        ),
        "measurement_schedule": (
            "alternate_eager-A-B__B-A-eager"
            if settings.schema_version == 1
            else "alternate_eager-A-B__eager-B-A"
        ),
        "quantile_method": "linear_interpolation",
        "compiler_state_reset": True,
        "compile_cache": v1_compile_cache_json(),
    }
    if settings.schema_version == 2:
        methodology["schema_version"] = 2
    write_json(child / "methodology.json", methodology)
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
    if disabled:
        metrics: list[dict[str, object]] = []
        cache_evidence: dict[str, object] = {
            "applicable": False,
            "reason": "compiler_disabled",
            "cold_compilation_metrics": [],
            "warm_cache_compilation_metrics": [],
        }
    else:
        cold = cache_metric(local_hit=0, local_miss=1)
        warm = cache_metric(local_hit=1, local_miss=0)
        metrics = [warm]
        cache_evidence = {
            "applicable": True,
            "reason": None,
            "cold_compilation_metrics": [cold],
            "warm_cache_compilation_metrics": [warm],
        }
    write_json(
        child / "dynamo_report.json",
        {
            "schema_version": 1,
            "graph_breaks": graph_breaks,
            "recompiles": recompiles,
            "compilation_metrics": metrics,
            "log_records": [],
            "cache_evidence": cache_evidence,
            "path_normalization": v1_path_normalization_json(),
        },
    )
    (child / "gate_criteria.yaml").write_bytes((ROOT / "gates/default.yaml").read_bytes())
    finalize_bundle(child, _evaluate_provisional_bundle)


def cache_metric(*, local_hit: int, local_miss: int) -> dict[str, object]:
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
        "inductor_config": json.dumps(dict(V1_INDUCTOR_CACHE_PATCH), sort_keys=True),
        "functorch_config": json.dumps(dict(V1_FUNCTORCH_CACHE_PATCH), sort_keys=True),
    }


__all__ = ["cache_metric", "write_normal_child"]

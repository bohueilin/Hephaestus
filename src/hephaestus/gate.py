"""Pure, deterministic verdict evaluation over stored evidence bundles."""

from __future__ import annotations

import hashlib
import json
import math
import random
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any, Final

import yaml

from hephaestus.bundle import canonical_json_bytes, strict_json_loads, verify_manifest
from hephaestus.evidence_contract import (
    V1_ACCURACY_TOLERANCE,
    V1_FUNCTORCH_CACHE_PATCH,
    V1_INDUCTOR_CACHE_PATCH,
    V1_WORKLOAD_SHA256,
    stable_signed_paired_effect,
    strict_json_equal,
    v1_compile_cache_json,
    v1_path_normalization_json,
    v1_run_settings_json,
    v2_run_settings_json,
)
from hephaestus.privacy import validate_public_evidence
from hephaestus.provenance import ProvenanceError, parse_run_provenance
from hephaestus.scope import EVIDENCE_BOUNDARY

FROZEN_V1_CRITERIA: Final[dict[str, object]] = {
    "schema_version": 1,
    "minimum_speedup": 1.10,
    "bootstrap": {"samples": 2000, "confidence": 0.95, "seed": 0},
    "compile_budgets_seconds": {
        "mlp_stack": 30.0,
        "transformer_block": 60.0,
        "dynamic_batch_text": 90.0,
        "graph_break_bait": 30.0,
    },
    "maximum_recompiles": {
        "mlp_stack": 0,
        "transformer_block": 0,
        "dynamic_batch_text": 2,
        "graph_break_bait": 0,
    },
    "graph_break_policy": {"mode": "conditional_if_any", "reasons_required": True},
}

FROZEN_V1_ORIGINAL_SHAPES: Final[dict[str, tuple[tuple[int, ...], ...]]] = {
    "mlp_stack": ((48, 32),),
    "transformer_block": ((16, 128, 32),),
    "dynamic_batch_text": ((2, 24, 32), (1, 48, 32), (3, 16, 32), (4, 12, 32)),
    "graph_break_bait": ((128, 128),),
}

_LEGACY_V1_ACCURACY_MANIFEST_SHA256: Final[frozenset[str]] = frozenset(
    {
        "8210d74300e4e8b7a2a2035a0a8af504b5c97dc2e03c18e9ccdd92b5ddc239d7",
        "c235ccb572c858bb8037f9474f0eaa4fb19e95a9dc0a0738835cc1c42897e641",
        "d15aae605a0037bdca707830da3477c50407c855f7b42a7f7164f1bc7fd6117d",
        "b04f7026eaa57656e2db5e586c73cb1b54efcb34bec7a726c8016fb37c5a6811",
    }
)


def evaluate_bundle(bundle_dir: Path) -> dict[str, object]:
    """Strictly evaluate one finalized stored normal bundle."""
    return _evaluate_normal_bundle(Path(bundle_dir), allow_provisional=False)


def _evaluate_provisional_bundle(bundle_dir: Path) -> dict[str, object]:
    """Evaluate the verdict-less topology only for trusted normal finalization."""
    return _evaluate_normal_bundle(Path(bundle_dir), allow_provisional=True)


def _evaluate_normal_bundle(
    bundle_dir: Path,
    *,
    allow_provisional: bool,
) -> dict[str, object]:
    integrity = verify_manifest(bundle_dir)
    if not integrity.valid:
        finding = {
            "id": "evidence.integrity",
            "status": "FAIL",
            "mismatches": list(integrity.mismatches),
        }
        return _verdict(
            "INVALID_EVIDENCE",
            "evidence.integrity",
            [finding],
        )

    try:
        manifest_sha256 = hashlib.sha256(
            (bundle_dir / "manifest.json").read_bytes()
        ).hexdigest()
    except OSError:
        return _verdict(
            "INVALID_EVIDENCE",
            "evidence.integrity",
            [
                {
                    "id": "evidence.integrity",
                    "status": "FAIL",
                    "mismatches": ["manifest.json:unreadable"],
                }
            ],
        )
    legacy_v1_accuracy = manifest_sha256 in _LEGACY_V1_ACCURACY_MANIFEST_SHA256

    topology = _normal_topology_mismatches(
        bundle_dir,
        allow_provisional=allow_provisional,
    )
    if topology:
        return _verdict(
            "INVALID_EVIDENCE",
            "evidence.topology",
            [
                {
                    "id": "evidence.topology",
                    "status": "FAIL",
                    "mismatches": topology,
                }
            ],
        )

    result = _evaluate_bundle_semantics(
        bundle_dir,
        legacy_v1_accuracy=legacy_v1_accuracy,
    )
    verdict_path = bundle_dir / "verdict.json"
    if verdict_path.exists():
        try:
            stored_verdict = verdict_path.read_bytes()
        except OSError:
            return _verdict(
                "INVALID_EVIDENCE",
                "bundle.verdict",
                [_failure("bundle.verdict")],
            )
        if stored_verdict != canonical_json_bytes(result):
            return _verdict(
                "INVALID_EVIDENCE",
                "bundle.verdict",
                [_failure("bundle.verdict")],
            )
    return result


def _evaluate_bundle_semantics(
    bundle_dir: Path,
    *,
    legacy_v1_accuracy: bool,
) -> dict[str, object]:
    try:
        environment = _json_object(bundle_dir / "env.json")
        _validate_environment(environment)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError):
        return _verdict(
            "INVALID_EVIDENCE",
            "environment.valid",
            [_failure("environment.valid")],
        )
    try:
        provenance = _json_object(bundle_dir / "run_provenance.json")
        parse_run_provenance(provenance)
    except (
        OSError,
        UnicodeDecodeError,
        json.JSONDecodeError,
        ValueError,
        ProvenanceError,
    ):
        return _verdict(
            "INVALID_EVIDENCE",
            "run.provenance",
            [_failure("run.provenance")],
        )

    try:
        criteria = _criteria(bundle_dir / "gate_criteria.yaml")
        methodology = _json_object(bundle_dir / "methodology.json")
        if "schema_version" not in methodology:
            schema_version = 1
        elif (
            type(methodology["schema_version"]) is int
            and methodology["schema_version"] == 2
        ):
            schema_version = 2
        else:
            raise ValueError("unsupported methodology schema version")
        aa_noise_floor = _nonnegative_number(methodology.get("aa_noise_floor"))
        methodology_bootstrap_samples = _nonnegative_integer(
            methodology.get("bootstrap_samples")
        )
        methodology_bootstrap_seed = _nonnegative_integer(
            methodology.get("bootstrap_seed")
        )
        methodology_bootstrap_confidence = _nonnegative_number(
            methodology.get("bootstrap_confidence")
        )
        methodology_repeats = _positive_integer(methodology.get("repeats"))
        _nonnegative_integer(methodology.get("warmup_runs"))
        _nonnegative_number(methodology.get("inter_run_spacing_seconds"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError, yaml.YAMLError):
        return _verdict("INVALID_EVIDENCE", "methodology.valid", [_failure("methodology.valid")])

    if methodology.get("valid") is not True:
        return _verdict("INVALID_EVIDENCE", "methodology.valid", [_failure("methodology.valid")])

    if methodology_bootstrap_samples != criteria["bootstrap"]["samples"]:
        return _verdict(
            "INVALID_EVIDENCE",
            "methodology.bootstrap_samples",
            [_failure("methodology.bootstrap_samples")],
        )
    expected_settings = (
        v1_run_settings_json() if schema_version == 1 else v2_run_settings_json()
    )
    if methodology_repeats != expected_settings["repeats"]:
        return _verdict(
            "INVALID_EVIDENCE",
            "methodology.repeats",
            [_failure("methodology.repeats")],
        )

    frozen_settings = {key: methodology.get(key) for key in expected_settings}
    if not strict_json_equal(frozen_settings, expected_settings):
        return _verdict(
            "INVALID_EVIDENCE",
            "methodology.settings",
            [_failure("methodology.settings")],
        )
    if methodology.get("compiler_state_reset") is not True or not strict_json_equal(
        methodology.get("compile_cache"),
        v1_compile_cache_json(),
    ):
        return _verdict(
            "INVALID_EVIDENCE",
            "methodology.compile_cache",
            [_failure("methodology.compile_cache")],
        )

    expected_methodology = {
        "bootstrap_seed": criteria["bootstrap"]["seed"],
        "bootstrap_confidence": criteria["bootstrap"]["confidence"],
        "aa_effect_formula": "(A-B)/((A+B)/2)",
        "aa_estimator": "p95_absolute_bootstrap_median",
        "aa_pairing": (
            "within_iteration"
            if schema_version == 1
            else "position_matched_cross_parity"
        ),
        "measurement_schedule": (
            "alternate_eager-A-B__B-A-eager"
            if schema_version == 1
            else "alternate_eager-A-B__eager-B-A"
        ),
        "quantile_method": "linear_interpolation",
    }
    actual_methodology = {
        "bootstrap_seed": methodology_bootstrap_seed,
        "bootstrap_confidence": methodology_bootstrap_confidence,
        "aa_effect_formula": methodology.get("aa_effect_formula"),
        "aa_estimator": methodology.get("aa_estimator"),
        "aa_pairing": methodology.get("aa_pairing"),
        "measurement_schedule": methodology.get("measurement_schedule"),
        "quantile_method": methodology.get("quantile_method"),
    }
    if not strict_json_equal(actual_methodology, expected_methodology):
        return _verdict(
            "INVALID_EVIDENCE", "methodology.valid", [_failure("methodology.valid")]
        )

    try:
        workload_payload = _json_object(bundle_dir / "workload.digest")
        workload, accuracy_contract = _validate_workload_identity(workload_payload)
    except _AccuracyContractError:
        return _verdict(
            "INVALID_EVIDENCE",
            "accuracy.contract",
            [_failure("accuracy.contract")],
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError):
        return _verdict(
            "INVALID_EVIDENCE",
            "workload.identity",
            [_failure("workload.identity")],
        )

    try:
        config = _json_object(bundle_dir / "config.json")
        validate_public_evidence(config)
        _validate_config_shape(config)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError):
        return _verdict(
            "INVALID_EVIDENCE",
            "methodology.config",
            [_failure("methodology.config")],
        )

    try:
        input_plan = _json_object(bundle_dir / "input_plan.json")
        sweep_indices = _validate_input_plan(input_plan, workload, config)
    except (KeyError, OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError):
        return _verdict(
            "INVALID_EVIDENCE",
            "methodology.input_plan",
            [_failure("methodology.input_plan")],
        )

    try:
        _validate_catalog_config(config, workload, input_plan)
    except (KeyError, TypeError, ValueError):
        return _verdict(
            "INVALID_EVIDENCE",
            "methodology.catalog",
            [_failure("methodology.catalog")],
        )

    try:
        timings = _json_object(bundle_dir / "timings.json")
        _validate_timing_evidence(
            timings,
            methodology_repeats,
            sweep_indices,
            compiler_disabled=config["disable"] is True,
            schema_version=schema_version,
        )
        _validate_aa_evidence(
            timings,
            aa_noise_floor,
            samples=methodology_bootstrap_samples,
            seed=methodology_bootstrap_seed,
            confidence=methodology_bootstrap_confidence,
            schema_version=schema_version,
        )
    except _EvidenceError as error:
        return _verdict("INVALID_EVIDENCE", error.finding, [_failure(error.finding)])
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError):
        return _verdict("INVALID_EVIDENCE", "methodology.valid", [_failure("methodology.valid")])

    configured_effect = Decimal(str(criteria["minimum_speedup"])) - Decimal("1")
    if configured_effect <= Decimal(str(aa_noise_floor)):
        return _verdict(
            "INVALID_EVIDENCE",
            "methodology.noise_floor",
            [_failure("methodology.noise_floor")],
        )

    try:
        accuracy = _json_object(bundle_dir / "accuracy.json")
        validate_public_evidence(accuracy)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError):
        return _verdict(
            "INVALID_EVIDENCE",
            "evidence.privacy",
            [_failure("evidence.privacy")],
        )
    try:
        _validate_accuracy_contract(
            accuracy,
            accuracy_contract,
            methodology_schema_version=schema_version,
            legacy_v1_accuracy=legacy_v1_accuracy,
        )
    except ValueError:
        return _verdict(
            "INVALID_EVIDENCE",
            "accuracy.contract",
            [_failure("accuracy.contract")],
        )
    try:
        accuracy_valid = _validate_accuracy(accuracy, len(sweep_indices))
    except ValueError:
        return _verdict(
            "INVALID_EVIDENCE",
            "accuracy.tolerance",
            [_failure("accuracy.tolerance")],
        )
    if not accuracy_valid:
        return _verdict("INVALID_EVIDENCE", "accuracy.tolerance", [_failure("accuracy.tolerance")])

    try:
        dynamo = _json_object(bundle_dir / "dynamo_report.json")
        validate_public_evidence(dynamo)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError):
        return _verdict(
            "INVALID_EVIDENCE",
            "evidence.privacy",
            [_failure("evidence.privacy")],
        )

    try:
        _validate_cache_evidence(dynamo, compiler_disabled=config["disable"] is True)
    except (KeyError, OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError):
        return _verdict(
            "INVALID_EVIDENCE",
            "methodology.cache_evidence",
            [_failure("methodology.cache_evidence")],
        )
    try:
        evidence = _hard_evidence(timings, dynamo)
        graph_breaks = _graph_breaks(dynamo)
        compile_budget = criteria["compile_budgets_seconds"][workload]
        recompile_bound = criteria["maximum_recompiles"][workload]
    except (KeyError, OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError):
        return _verdict(
            "INVALID_EVIDENCE",
            "methodology.valid",
            [_failure("methodology.valid")],
        )

    try:
        eager_median = _quantile(evidence["eager_seconds"], 0.5)
        compiled_median = _quantile(evidence["compiled_seconds"], 0.5)
        lower_bound = _bootstrap_lower_bound(
            evidence["eager_seconds"],
            evidence["compiled_seconds"],
            criteria["bootstrap"]["samples"],
            criteria["bootstrap"]["confidence"],
            criteria["bootstrap"]["seed"],
        )
        eager_iqr = _iqr(evidence["eager_seconds"])
        compiled_iqr = _iqr(evidence["compiled_seconds"])
        median_speedup = (
            None
            if compiled_median == 0
            else _finite_derived(eager_median / compiled_median)
        )
    except (ArithmeticError, ValueError):
        return _verdict(
            "INVALID_EVIDENCE",
            "methodology.statistics",
            [_failure("methodology.statistics")],
        )
    speedup_passes = lower_bound is not None and lower_bound >= criteria["minimum_speedup"]
    compile_passes = evidence["cold_compile_seconds"] <= compile_budget
    recompile_passes = len(evidence["recompiles"]) <= recompile_bound
    findings = [
        {
            "id": "perf.speedup_proven",
            "status": "PASS" if speedup_passes else "FAIL",
            "lower_confidence_bound": lower_bound,
            "minimum_speedup": criteria["minimum_speedup"],
        },
        {
            "id": "perf.compile_budget",
            "status": "PASS" if compile_passes else "FAIL",
            "cold_compile_seconds": evidence["cold_compile_seconds"],
            "budget_seconds": compile_budget,
        },
        {
            "id": "graph.recompile_bound",
            "status": "PASS" if recompile_passes else "FAIL",
            "count": len(evidence["recompiles"]),
            "maximum": recompile_bound,
        },
    ]
    measurements = {
        "eager_median_seconds": eager_median,
        "eager_iqr_seconds": eager_iqr,
        "compiled_median_seconds": compiled_median,
        "compiled_iqr_seconds": compiled_iqr,
        "median_speedup": median_speedup,
        "speedup_lower_confidence_bound": lower_bound,
    }

    for finding in findings[:3]:
        if finding["status"] == "FAIL":
            return _verdict("NOT_PROVEN", str(finding["id"]), findings, measurements)

    graph_passes = not graph_breaks
    findings.append(
        {
            "id": "graph.no_breaks",
            "status": "PASS" if graph_passes else "OPEN",
            "count": len(graph_breaks),
        }
    )
    if not graph_passes:
        return _verdict("CONDITIONAL", "graph.no_breaks", findings, measurements)
    return _verdict("PROVEN", "all_criteria_passed", findings, measurements)


def _verdict(
    verdict: str,
    driving_finding: str,
    findings: list[dict[str, object]],
    measurements: dict[str, float | None] | None = None,
) -> dict[str, object]:
    result: dict[str, object] = {
        "verdict": verdict,
        "driving_finding": driving_finding,
        "findings": findings,
    }
    if measurements is not None:
        result["measurements"] = measurements
    return result


def _failure(identifier: str) -> dict[str, object]:
    return {"id": identifier, "status": "FAIL"}


def _criteria(path: Path) -> dict[str, Any]:
    loaded = yaml.safe_load(path.read_text())
    if not _same_frozen_value(loaded, FROZEN_V1_CRITERIA):
        raise ValueError("invalid criteria")
    assert isinstance(loaded, dict)
    return {
        "minimum_speedup": loaded["minimum_speedup"],
        "bootstrap": loaded["bootstrap"],
        "compile_budgets_seconds": loaded["compile_budgets_seconds"],
        "maximum_recompiles": loaded["maximum_recompiles"],
    }


def _same_frozen_value(actual: object, expected: object) -> bool:
    if type(actual) is not type(expected):
        return False
    if isinstance(expected, dict):
        return isinstance(actual, dict) and actual.keys() == expected.keys() and all(
            _same_frozen_value(actual[key], expected[key]) for key in expected
        )
    if isinstance(expected, list):
        return isinstance(actual, list) and len(actual) == len(expected) and all(
            _same_frozen_value(actual_item, expected_item)
            for actual_item, expected_item in zip(actual, expected, strict=True)
        )
    return actual == expected


def _normal_topology_mismatches(
    bundle_dir: Path,
    *,
    allow_provisional: bool,
) -> list[str]:
    expected = {
        "accuracy.json",
        "config.json",
        "dynamo_report.json",
        "env.json",
        "gate_criteria.yaml",
        "input_plan.json",
        "manifest.json",
        "methodology.json",
        "run_provenance.json",
        "timings.json",
        "workload.digest",
    }
    verdict_exists = (bundle_dir / "verdict.json").exists()
    if not allow_provisional or verdict_exists:
        expected.add("verdict.json")
    try:
        actual = {path.name for path in bundle_dir.iterdir()}
    except OSError:
        return ["bundle:missing"]
    return [
        *(f"bundle:missing:{name}" for name in sorted(expected - actual)),
        *(f"bundle:unexpected:{name}" for name in sorted(actual - expected)),
    ]


def _validate_environment(value: dict[str, object]) -> None:
    if value.keys() != {
        "schema_version",
        "torch",
        "python",
        "os",
        "chip",
        "boundary",
    }:
        raise ValueError("invalid environment schema")
    if type(value.get("schema_version")) is not int or value["schema_version"] != 1:
        raise ValueError("invalid environment schema")
    expected_torch = {
        "version": "2.13.0",
        "git_version": "cf30153c4c131c8164ee7798e5022d810682e2cb",
        "debug": False,
    }
    expected_python = {"implementation": "CPython", "version": "3.14.7"}
    if not strict_json_equal(value.get("torch"), expected_torch):
        raise ValueError("environment Torch build is not pinned")
    if not strict_json_equal(value.get("python"), expected_python):
        raise ValueError("environment Python build is not pinned")
    os_value = value.get("os")
    if (
        not isinstance(os_value, dict)
        or os_value.keys() != {"system", "release"}
        or os_value.get("system") != "Darwin"
        or not isinstance(os_value.get("release"), str)
        or not os_value["release"]
    ):
        raise ValueError("environment OS is not pinned")
    if value.get("chip") != "arm64" or value.get("boundary") != EVIDENCE_BOUNDARY:
        raise ValueError("environment machine boundary is not pinned")


def _validate_workload_identity(
    value: dict[str, object],
) -> tuple[str, dict[str, object]]:
    if value.keys() != {
        "schema_version",
        "name",
        "sha256",
        "accuracy_tolerance",
    }:
        raise ValueError("invalid workload identity schema")
    if type(value.get("schema_version")) is not int or value["schema_version"] != 1:
        raise ValueError("invalid workload identity schema")
    name = value.get("name")
    if not isinstance(name, str) or name not in V1_WORKLOAD_SHA256:
        raise ValueError("unknown workload identity")
    if value.get("sha256") != V1_WORKLOAD_SHA256[name]:
        raise ValueError("workload source digest is not frozen")
    tolerance = dict(V1_ACCURACY_TOLERANCE)
    if not strict_json_equal(value.get("accuracy_tolerance"), tolerance):
        raise _AccuracyContractError("workload accuracy tolerance is not frozen")
    return name, tolerance


class _AccuracyContractError(ValueError):
    """The frozen workload and stored accuracy tolerance contracts disagree."""


def _validate_config_shape(config: dict[str, object]) -> None:
    if config.keys() != {
        "backend",
        "mode",
        "dynamic",
        "fullgraph",
        "options",
        "disable",
        "catalog",
    }:
        raise ValueError("invalid compile config schema")
    if not isinstance(config.get("backend"), str) or not config["backend"]:
        raise ValueError("invalid compile backend")
    if config.get("mode") is not None and not isinstance(config["mode"], str):
        raise ValueError("invalid compile mode")
    if type(config.get("dynamic")) is not bool:
        raise ValueError("invalid compile dynamic flag")
    if type(config.get("fullgraph")) is not bool:
        raise ValueError("invalid compile fullgraph flag")
    options = config.get("options")
    if options is not None and not isinstance(options, dict):
        raise ValueError("invalid compile options")
    if type(config.get("disable")) is not bool or not isinstance(config.get("catalog"), dict):
        raise ValueError("invalid compile config")


def _validate_accuracy_contract(
    accuracy: dict[str, object],
    tolerance: dict[str, object],
    *,
    methodology_schema_version: int,
    legacy_v1_accuracy: bool,
) -> None:
    modern_keys = {
        "schema_version",
        "dtype",
        "within_tolerance",
        "case_index",
        "atol",
        "rtol",
        "max_absolute_error",
        "mismatch",
        "cases",
    }
    legacy_keys = modern_keys - {"schema_version", "dtype"}
    if accuracy.keys() == modern_keys:
        if type(accuracy.get("schema_version")) is not int or accuracy["schema_version"] != 1:
            raise ValueError("invalid accuracy evidence schema")
        stored = {
            "dtype": accuracy.get("dtype"),
            "atol": accuracy.get("atol"),
            "rtol": accuracy.get("rtol"),
        }
        expected = {
            "dtype": tolerance["dtype"],
            "atol": tolerance["atol"],
            "rtol": tolerance["rtol"],
        }
    elif (
        methodology_schema_version == 1
        and legacy_v1_accuracy
        and accuracy.keys() == legacy_keys
    ):
        stored = {
            "atol": accuracy.get("atol"),
            "rtol": accuracy.get("rtol"),
        }
        expected = {
            "atol": tolerance["atol"],
            "rtol": tolerance["rtol"],
        }
    else:
        raise ValueError("invalid accuracy evidence schema")
    if not strict_json_equal(stored, expected):
        raise ValueError("accuracy tolerance differs from workload contract")


def _json_object(path: Path) -> dict[str, object]:
    loaded = strict_json_loads(path.read_bytes())
    if not isinstance(loaded, dict):
        raise ValueError("expected JSON object")
    return loaded


def _validate_input_plan(
    plan: dict[str, object], workload: str, config: dict[str, object]
) -> list[int]:
    expected_keys = {
        "schema_version",
        "dynamic_strategy",
        "bucket_axis",
        "bucket_boundaries",
        "bucket_overflow_rule",
        "original_shapes",
        "effective_shapes",
        "compile_sweep_case_indices",
        "steady_state_case_index",
    }
    if plan.keys() != expected_keys or type(plan.get("schema_version")) is not int:
        raise ValueError("invalid input plan schema")
    if plan["schema_version"] != 1:
        raise ValueError("invalid input plan schema")

    try:
        expected_original = list(FROZEN_V1_ORIGINAL_SHAPES[workload])
    except KeyError as error:
        raise ValueError("unknown workload input shape") from error
    original = _shapes(plan.get("original_shapes"))
    effective = _shapes(plan.get("effective_shapes"))
    if original != expected_original or len(effective) != len(original):
        raise ValueError("input plan shapes do not match the frozen workload")

    indices = _case_indices(plan.get("compile_sweep_case_indices"))
    steady_state_case_index = plan.get("steady_state_case_index")
    if (
        indices != list(range(len(original)))
        or type(steady_state_case_index) is not int
        or steady_state_case_index != 0
    ):
        raise ValueError("input plan does not cover every case in order")

    dynamic = config.get("dynamic")
    if dynamic is not None and type(dynamic) is not bool:
        raise ValueError("invalid compile dynamic flag")
    strategy = plan.get("dynamic_strategy")
    if not isinstance(strategy, str):
        raise ValueError("invalid dynamic strategy")
    if strategy in {"static", "dynamic", "auto"}:
        expected_dynamic = {
            "static": False,
            "dynamic": True,
            "auto": None,
        }[strategy]
        if dynamic is not expected_dynamic:
            raise ValueError("identity plan disagrees with compile dynamic flag")
        if (
            plan.get("bucket_axis") is not None
            or plan.get("bucket_boundaries") is not None
            or plan.get("bucket_overflow_rule") is not None
            or effective != original
        ):
            raise ValueError("identity plan contains bucket effects")
        return indices

    if strategy != "bucketed" or workload != "dynamic_batch_text" or dynamic is not False:
        raise ValueError("invalid bucketed input plan")
    if (
        type(plan.get("bucket_axis")) is not int
        or plan.get("bucket_axis") != 0
        or not _same_frozen_value(plan.get("bucket_boundaries"), [2, 4])
        or plan.get("bucket_overflow_rule") != "reject"
    ):
        raise ValueError("invalid frozen bucket declaration")
    for source_shape, effective_shape in zip(original, effective, strict=True):
        batch = source_shape[0]
        if batch > 4:
            raise ValueError("bucket overflow")
        expected_batch = 2 if batch <= 2 else 4
        if effective_shape != (expected_batch, *source_shape[1:]):
            raise ValueError("invalid bucket shape transformation")
    return indices


def _validate_catalog_config(
    config: dict[str, object], workload: str, input_plan: dict[str, object]
) -> None:
    base_keys = {"backend", "mode", "dynamic", "fullgraph", "options", "disable"}
    if config.keys() != base_keys | {"catalog"}:
        raise ValueError("invalid catalog config keys")
    metadata = config.get("catalog")
    expected_effective = validate_catalog_metadata(metadata, workload)
    effective = metadata["effective"]
    assert isinstance(effective, dict)
    if not _same_frozen_value(effective, expected_effective):
        raise ValueError("effective catalog settings disagree with requested settings")
    for key in base_keys:
        if not _same_frozen_value(config.get(key), effective.get(key)):
            raise ValueError("catalog effective settings disagree with actual config")
    if input_plan.get("dynamic_strategy") != effective["input_plan_strategy"]:
        raise ValueError("catalog effective settings disagree with input plan")


def validate_catalog_metadata(
    metadata: object,
    workload: str,
) -> dict[str, object]:
    """Validate stored v1 catalog metadata without consulting the live catalog."""
    base_keys = {"backend", "mode", "dynamic", "fullgraph", "options", "disable"}
    expected_metadata_keys = {
        "schema_version",
        "entry_id",
        "role",
        "workload_name",
        "requested",
        "effective",
    }
    if not isinstance(metadata, dict) or metadata.keys() != expected_metadata_keys:
        raise ValueError("invalid catalog metadata")
    if type(metadata.get("schema_version")) is not int or metadata["schema_version"] != 1:
        raise ValueError("invalid catalog schema")
    entry_id = metadata.get("entry_id")
    if (
        not isinstance(entry_id, str)
        or not entry_id
        or entry_id != entry_id.strip()
        or "\n" in entry_id
        or "\r" in entry_id
    ):
        raise ValueError("invalid catalog entry ID")
    role = metadata.get("role")
    if role not in {"candidate", "clean-control", "planted"}:
        raise ValueError("invalid catalog role")
    if metadata.get("workload_name") != workload:
        raise ValueError("catalog workload mismatch")
    requested = metadata.get("requested")
    effective = metadata.get("effective")
    requested_keys = {
        "mode",
        "dynamic",
        "fullgraph",
        "options",
        "disable",
        "bucket_policy",
    }
    effective_keys = base_keys | {"input_plan_strategy"}
    if not isinstance(requested, dict) or requested.keys() != requested_keys:
        raise ValueError("invalid requested catalog settings")
    if not isinstance(effective, dict) or effective.keys() != effective_keys:
        raise ValueError("invalid effective catalog settings")

    requested_mode = requested.get("mode")
    mode_mapping = {
        "default": None,
        "reduce-overhead": "reduce-overhead",
        "max-autotune-equivalent": "max-autotune-no-cudagraphs",
    }
    if requested_mode not in mode_mapping:
        raise ValueError("invalid requested mode")
    requested_dynamic = requested.get("dynamic")
    dynamic_mapping = {"false": False, "true": True, "bucketed": False}
    strategy_mapping = {"false": "static", "true": "dynamic", "bucketed": "bucketed"}
    if requested_dynamic not in dynamic_mapping:
        raise ValueError("invalid requested dynamic strategy")
    requested_fullgraph = requested.get("fullgraph")
    fullgraph_mapping = {"false": False, "true": True}
    if requested_fullgraph not in fullgraph_mapping:
        raise ValueError("invalid requested fullgraph setting")
    requested_options = requested.get("options")
    if requested_options is not None and (
        not isinstance(requested_options, dict)
        or not requested_options
        or any(
            not isinstance(name, str)
            or not name
            or type(value) not in {str, int, bool}
            for name, value in requested_options.items()
        )
        or requested_mode != "default"
    ):
        raise ValueError("invalid requested backend options")
    requested_disable = requested.get("disable")
    if type(requested_disable) is not bool or (requested_disable and role != "planted"):
        raise ValueError("invalid requested disable setting")
    bucket_policy = requested.get("bucket_policy")
    if requested_dynamic == "bucketed":
        if workload != "dynamic_batch_text" or not _same_frozen_value(
            bucket_policy,
            {
                "axis": 0,
                "boundaries": [2, 4],
                "overflow_rule": "reject",
            },
        ):
            raise ValueError("invalid bucket policy")
    elif bucket_policy is not None:
        raise ValueError("unexpected bucket policy")

    expected_effective = {
        "backend": "inductor",
        "mode": mode_mapping[requested_mode],
        "dynamic": dynamic_mapping[requested_dynamic],
        "fullgraph": fullgraph_mapping[requested_fullgraph],
        "options": requested_options,
        "disable": requested_disable,
        "input_plan_strategy": strategy_mapping[requested_dynamic],
    }
    if not _same_frozen_value(effective, expected_effective):
        raise ValueError("effective catalog settings disagree with requested settings")
    return expected_effective


def _validate_timing_evidence(
    timings: dict[str, object],
    repeats: int,
    sweep_indices: list[int],
    *,
    compiler_disabled: bool,
    schema_version: int,
) -> None:
    seconds_by_series = {
        name: _timings(timings.get(f"{name}_seconds"))
        for name in ("eager", "compiled", "aa_baseline", "aa_candidate")
    }
    if any(len(values) != repeats for values in seconds_by_series.values()):
        raise _EvidenceError("methodology.repeats")

    timestamps_by_series = {
        name: _timestamps(timings.get(f"{name}_timestamps_utc"))
        for name in ("eager", "compiled", "aa_baseline", "aa_candidate")
    }
    timestamp_finding = (
        "methodology.timestamps"
        if schema_version == 1
        else "methodology.v2.timestamps"
    )
    if any(len(values) != repeats for values in timestamps_by_series.values()):
        raise _EvidenceError(timestamp_finding)
    if timestamps_by_series["compiled"] != timestamps_by_series["aa_baseline"]:
        raise _EvidenceError(timestamp_finding)

    expected_non_primary = sweep_indices[1:]
    try:
        actual_non_primary = _case_indices(
            timings.get("non_primary_compile_sweep_case_indices"), allow_empty=True
        )
        sweep_seconds = _timings(
            timings.get("non_primary_compile_sweep_seconds"), allow_empty=True
        )
        sweep_timestamps = _timestamps(
            timings.get("non_primary_compile_sweep_timestamps_utc"), allow_empty=True
        )
    except ValueError as error:
        raise _EvidenceError("methodology.compile_sweep") from error
    if (
        actual_non_primary != expected_non_primary
        or len(sweep_seconds) != len(expected_non_primary)
        or len(sweep_timestamps) != len(expected_non_primary)
    ):
        raise _EvidenceError("methodology.compile_sweep")

    try:
        previous = _utc_timestamp(timings.get("cold_compile_timestamp_utc"))
    except ValueError as error:
        raise _EvidenceError(timestamp_finding) from error
    warm_seconds = timings.get("warm_cache_compile_seconds")
    warm_timestamp = timings.get("warm_cache_compile_timestamp_utc")
    if compiler_disabled:
        if warm_seconds is not None or warm_timestamp is not None:
            raise _EvidenceError("methodology.cache_evidence")
    else:
        try:
            _nonnegative_number(warm_seconds)
            parsed_warm_timestamp = _utc_timestamp(warm_timestamp)
        except ValueError as error:
            raise _EvidenceError("methodology.cache_evidence") from error
        if parsed_warm_timestamp <= previous:
            raise _EvidenceError("methodology.cache_evidence")
        previous = parsed_warm_timestamp
    for timestamp in sweep_timestamps:
        if timestamp <= previous:
            raise _EvidenceError("methodology.compile_sweep")
        previous = timestamp

    eager_timestamps = timestamps_by_series["eager"]
    baseline_timestamps = timestamps_by_series["aa_baseline"]
    candidate_timestamps = timestamps_by_series["aa_candidate"]
    for iteration in range(repeats):
        if iteration % 2 == 0:
            ordered = (
                eager_timestamps[iteration],
                baseline_timestamps[iteration],
                candidate_timestamps[iteration],
            )
        elif schema_version == 1:
            ordered = (
                candidate_timestamps[iteration],
                baseline_timestamps[iteration],
                eager_timestamps[iteration],
            )
        else:
            ordered = (
                eager_timestamps[iteration],
                candidate_timestamps[iteration],
                baseline_timestamps[iteration],
            )
        if ordered[0] <= previous or not ordered[0] < ordered[1] < ordered[2]:
            raise _EvidenceError(timestamp_finding)
        previous = ordered[2]


def _validate_accuracy(accuracy: dict[str, object], case_count: int) -> bool:
    aggregate = accuracy.get("within_tolerance")
    if type(aggregate) is not bool:
        raise ValueError("invalid accuracy aggregate")
    _nonnegative_number(accuracy.get("atol"))
    _nonnegative_number(accuracy.get("rtol"))
    records = accuracy.get("cases")
    if not isinstance(records, list) or len(records) != case_count:
        raise ValueError("accuracy does not cover every case")

    statuses: list[bool] = []
    max_errors: list[float] = []
    first_failed_record: dict[str, object] | None = None
    expected_keys = {
        "case_index",
        "within_tolerance",
        "max_absolute_error",
        "mismatch",
    }
    for case_index, record in enumerate(records):
        if not isinstance(record, dict) or record.keys() != expected_keys:
            raise ValueError("invalid accuracy record")
        record_case_index = record.get("case_index")
        if type(record_case_index) is not int or record_case_index != case_index:
            raise ValueError("accuracy records are out of order")
        status = record.get("within_tolerance")
        if type(status) is not bool:
            raise ValueError("invalid accuracy status")
        max_error = record.get("max_absolute_error")
        if max_error is None:
            if status:
                raise ValueError("passing accuracy record lacks measured error")
        else:
            if type(max_error) is not float:
                raise ValueError("invalid accuracy error")
            max_errors.append(_nonnegative_number(max_error))
        mismatch = record.get("mismatch")
        if status and mismatch is not None:
            raise ValueError("passing accuracy record contains a mismatch")
        if not status and not isinstance(mismatch, dict):
            raise ValueError("failed accuracy record lacks mismatch evidence")
        if isinstance(mismatch, dict):
            _validate_accuracy_mismatch(mismatch)
        if not status and first_failed_record is None:
            first_failed_record = record
        statuses.append(status)
    if aggregate is not all(statuses):
        raise ValueError("accuracy aggregate disagrees with case records")
    expected_case_index = (
        0 if first_failed_record is None else first_failed_record["case_index"]
    )
    if type(accuracy.get("case_index")) is not int or not strict_json_equal(
        accuracy["case_index"], expected_case_index
    ):
        raise ValueError("accuracy aggregate case index is not bound")
    expected_max_error = max(max_errors) if max_errors else None
    if not strict_json_equal(accuracy.get("max_absolute_error"), expected_max_error):
        raise ValueError("accuracy aggregate error is not bound")
    expected_mismatch = (
        None if first_failed_record is None else first_failed_record["mismatch"]
    )
    if not strict_json_equal(accuracy.get("mismatch"), expected_mismatch):
        raise ValueError("accuracy aggregate mismatch is not bound")
    return aggregate


def _validate_accuracy_mismatch(mismatch: dict[str, object]) -> None:
    kind = mismatch.get("kind")
    if not isinstance(kind, str):
        raise ValueError("invalid accuracy mismatch evidence")
    expected_keys = {
        "value": {"kind"},
        "nonfinite": {"kind"},
        "shape": {"kind", "eager_shape", "compiled_shape"},
        "dtype": {"kind", "eager_dtype", "compiled_dtype"},
        "type": {"kind", "eager_type", "compiled_type"},
    }.get(kind)
    if expected_keys is None or mismatch.keys() != expected_keys:
        raise ValueError("invalid accuracy mismatch evidence")
    if kind == "shape":
        for key in ("eager_shape", "compiled_shape"):
            shape = mismatch[key]
            if not isinstance(shape, list) or any(
                type(dimension) is not int or dimension < 0 for dimension in shape
            ):
                raise ValueError("invalid accuracy mismatch shape")
    elif kind in {"dtype", "type"}:
        for key, value in mismatch.items():
            if key != "kind" and (not isinstance(value, str) or not value):
                raise ValueError("invalid accuracy mismatch label")


def _validate_cache_evidence(
    dynamo: dict[str, object],
    *,
    compiler_disabled: bool,
) -> None:
    if dynamo.keys() != {
        "schema_version",
        "graph_breaks",
        "recompiles",
        "compilation_metrics",
        "log_records",
        "cache_evidence",
        "path_normalization",
    }:
        raise ValueError("invalid Dynamo report schema")
    if type(dynamo.get("schema_version")) is not int or dynamo["schema_version"] != 1:
        raise ValueError("invalid Dynamo report schema")
    if not strict_json_equal(
        dynamo.get("path_normalization"),
        v1_path_normalization_json(),
    ):
        raise ValueError("invalid path-normalization declaration")
    for key in ("graph_breaks", "recompiles", "compilation_metrics", "log_records"):
        if not isinstance(dynamo.get(key), list):
            raise ValueError("invalid Dynamo report series")
    cache = dynamo.get("cache_evidence")
    if not isinstance(cache, dict) or cache.keys() != {
        "applicable",
        "reason",
        "cold_compilation_metrics",
        "warm_cache_compilation_metrics",
    }:
        raise ValueError("invalid cache evidence schema")
    compilation_metrics = dynamo["compilation_metrics"]
    assert isinstance(compilation_metrics, list)
    if compiler_disabled:
        expected = {
            "applicable": False,
            "reason": "compiler_disabled",
            "cold_compilation_metrics": [],
            "warm_cache_compilation_metrics": [],
        }
        if not strict_json_equal(cache, expected) or compilation_metrics:
            raise ValueError("disabled compiler cache evidence is not canonical")
        return
    if cache.get("applicable") is not True or cache.get("reason") is not None:
        raise ValueError("enabled compiler lacks applicable cache evidence")
    cold = cache.get("cold_compilation_metrics")
    warm = cache.get("warm_cache_compilation_metrics")
    if not isinstance(cold, list) or not cold or not isinstance(warm, list) or not warm:
        raise ValueError("enabled compiler cache metrics are incomplete")
    if len(warm) > len(compilation_metrics) or not strict_json_equal(
        compilation_metrics[: len(warm)],
        warm,
    ):
        raise ValueError("warm cache metrics are not an operational prefix")
    cold_counts = _cache_metric_counts(cold)
    warm_counts = _cache_metric_counts(warm)
    if (
        cold_counts["fx_local_miss"] < 1
        or cold_counts["aot_local_miss"] < 1
        or cold_counts["fx_local_hit"] != 0
        or cold_counts["aot_local_hit"] != 0
        or cold_counts["remote"] != 0
    ):
        raise ValueError("cold cache evidence does not prove local misses")
    if (
        warm_counts["fx_local_hit"] < 1
        or warm_counts["aot_local_hit"] < 1
        or warm_counts["remote"] != 0
    ):
        raise ValueError("warm cache evidence does not prove local hits")


def _cache_metric_counts(metrics: list[object]) -> dict[str, int]:
    keys = {
        "fx_local_hit": "inductor_fx_local_cache_hit_count",
        "fx_local_miss": "inductor_fx_local_cache_miss_count",
        "fx_remote_hit": "inductor_fx_remote_cache_hit_count",
        "fx_remote_miss": "inductor_fx_remote_cache_miss_count",
        "aot_local_hit": "aotautograd_local_cache_hit_count",
        "aot_local_miss": "aotautograd_local_cache_miss_count",
        "aot_remote_hit": "aotautograd_remote_cache_hit_count",
        "aot_remote_miss": "aotautograd_remote_cache_miss_count",
    }
    totals = {name: 0 for name in keys}
    for metric in metrics:
        if not isinstance(metric, dict):
            raise ValueError("cache compilation metric must be an object")
        for name, key in keys.items():
            count = metric.get(key)
            if type(count) is not int or count < 0:
                raise ValueError("cache compilation metric counter is invalid")
            totals[name] += count
        _validate_metric_cache_config(metric)
    totals["remote"] = sum(
        totals[name]
        for name in (
            "fx_remote_hit",
            "fx_remote_miss",
            "aot_remote_hit",
            "aot_remote_miss",
        )
    )
    return totals


def _validate_metric_cache_config(metric: dict[str, object]) -> None:
    compiler = _embedded_config(metric.get("compiler_config"))
    inductor = _embedded_config(metric.get("inductor_config"))
    functorch = _embedded_config(metric.get("functorch_config"))
    if compiler.get("force_disable_caches") is not False:
        raise ValueError("compiler cache config does not preserve caches")
    inductor_expected = dict(V1_INDUCTOR_CACHE_PATCH)
    inductor_expected.pop("force_disable_caches")
    _require_config_subset(inductor, inductor_expected)
    _require_config_subset(functorch, dict(V1_FUNCTORCH_CACHE_PATCH))


def _embedded_config(value: object) -> dict[str, object]:
    if not isinstance(value, str):
        raise ValueError("compilation metric lacks captured config")
    loaded = strict_json_loads(value)
    if not isinstance(loaded, dict):
        raise ValueError("captured compiler config must be an object")
    return loaded


def _require_config_subset(
    actual: dict[str, object],
    expected: dict[str, object],
) -> None:
    for key, expected_value in expected.items():
        if key not in actual or not strict_json_equal(actual[key], expected_value):
            raise ValueError("captured compiler cache config is not pinned")


def _shapes(value: object) -> list[tuple[int, ...]]:
    if not isinstance(value, list) or not value:
        raise ValueError("missing input shapes")
    shapes: list[tuple[int, ...]] = []
    for shape in value:
        if (
            not isinstance(shape, list)
            or not shape
            or any(type(dimension) is not int or dimension <= 0 for dimension in shape)
        ):
            raise ValueError("invalid input shape")
        shapes.append(tuple(shape))
    return shapes


def _case_indices(value: object, *, allow_empty: bool = False) -> list[int]:
    if not isinstance(value, list) or (not value and not allow_empty):
        raise ValueError("missing case indices")
    if any(type(index) is not int or index < 0 for index in value):
        raise ValueError("invalid case index")
    return value


def _timestamps(value: object, *, allow_empty: bool = False) -> list[datetime]:
    if not isinstance(value, list) or (not value and not allow_empty):
        raise ValueError("missing timestamps")
    return [_utc_timestamp(timestamp) for timestamp in value]


def _utc_timestamp(value: object) -> datetime:
    if not isinstance(value, str) or not value.endswith("Z"):
        raise ValueError("timestamp must be UTC ISO-8601")
    try:
        parsed = datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError as error:
        raise ValueError("timestamp must be UTC ISO-8601") from error
    if parsed.tzinfo is None or parsed.utcoffset() != UTC.utcoffset(parsed):
        raise ValueError("timestamp must be UTC ISO-8601")
    return parsed


def _hard_evidence(timings: dict[str, object], dynamo: dict[str, object]) -> dict[str, object]:
    eager_seconds = _timings(timings.get("eager_seconds"))
    compiled_seconds = _timings(timings.get("compiled_seconds"))
    cold_compile_seconds = _nonnegative_number(timings.get("cold_compile_seconds"))
    recompiles = _records(dynamo.get("recompiles"), "trigger")
    return {
        "eager_seconds": eager_seconds,
        "compiled_seconds": compiled_seconds,
        "cold_compile_seconds": cold_compile_seconds,
        "recompiles": recompiles,
    }


class _EvidenceError(ValueError):
    def __init__(self, finding: str) -> None:
        super().__init__(finding)
        self.finding = finding


def _validate_aa_evidence(
    timings: dict[str, object],
    stored_floor: float,
    *,
    samples: int,
    seed: int,
    confidence: float,
    schema_version: int,
) -> None:
    compiled = _timings(timings.get("compiled_seconds"))
    baseline = _timings(timings.get("aa_baseline_seconds"))
    candidate = _timings(timings.get("aa_candidate_seconds"))
    stored_effects = _finite_numbers(timings.get("aa_signed_paired_effects"))
    stored_bootstrap = _timings(timings.get("aa_bootstrap_absolute_medians"))
    if compiled != baseline:
        raise _EvidenceError("methodology.aa_baseline")
    if len(baseline) != len(candidate) or len(baseline) != len(stored_effects):
        raise _EvidenceError("methodology.aa_effects")

    if schema_version == 1:
        recomputed_effects = [
            _signed_paired_effect(left, right)
            for left, right in zip(baseline, candidate, strict=True)
        ]
    else:
        recomputed_effects = [
            effect
            for even_index in range(0, len(baseline), 2)
            for effect in (
                _signed_paired_effect(
                    baseline[even_index], candidate[even_index + 1]
                ),
                _signed_paired_effect(
                    baseline[even_index + 1], candidate[even_index]
                ),
            )
        ]
    if stored_effects != recomputed_effects:
        raise _EvidenceError("methodology.aa_effects")

    recomputed_bootstrap = _bootstrap_absolute_medians(
        recomputed_effects, samples=samples, seed=seed
    )
    if stored_bootstrap != recomputed_bootstrap:
        raise _EvidenceError("methodology.aa_bootstrap")
    if stored_floor != _quantile(recomputed_bootstrap, confidence):
        raise _EvidenceError("methodology.noise_floor")


def _signed_paired_effect(baseline: float, candidate: float) -> float:
    return _finite_derived(stable_signed_paired_effect(baseline, candidate))


def _bootstrap_absolute_medians(
    effects: list[float], *, samples: int, seed: int
) -> list[float]:
    random_source = random.Random(seed)
    return [
        _finite_derived(abs(_sample_median(effects, random_source)))
        for _ in range(samples)
    ]


def _graph_breaks(dynamo: dict[str, object]) -> list[dict[str, object]]:
    return _records(dynamo.get("graph_breaks"), "reason")


def _timings(value: object, *, allow_empty: bool = False) -> list[float]:
    if not isinstance(value, list) or (not value and not allow_empty):
        raise ValueError("missing timings")
    return [_nonnegative_number(item) for item in value]


def _finite_numbers(value: object) -> list[float]:
    if not isinstance(value, list) or not value:
        raise ValueError("missing numeric series")
    numbers: list[float] = []
    for item in value:
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            raise ValueError("expected numeric value")
        numbers.append(_finite_float(item))
    return numbers


def _records(value: object, required_key: str) -> list[dict[str, object]]:
    if not isinstance(value, list):
        raise ValueError("invalid dynamo report")
    records: list[dict[str, object]] = []
    for item in value:
        is_reasoned_record = (
            isinstance(item, dict)
            and isinstance(item.get(required_key), str)
            and bool(item[required_key])
        )
        if not is_reasoned_record:
            raise ValueError("unreasoned dynamo event")
        records.append(item)
    return records


def _positive_number(value: object) -> float:
    parsed = _nonnegative_number(value)
    if parsed <= 0:
        raise ValueError("expected positive number")
    return parsed


def _nonnegative_number(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("expected numeric value")
    parsed = _finite_float(value)
    if parsed < 0:
        raise ValueError("expected finite nonnegative value")
    return parsed


def _finite_float(value: int | float) -> float:
    try:
        parsed = float(value)
    except OverflowError as error:
        raise ValueError("expected finite numeric value") from error
    if not math.isfinite(parsed):
        raise ValueError("expected finite numeric value")
    return parsed


def _nonnegative_integer(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError("expected nonnegative integer")
    return value


def _positive_integer(value: object) -> int:
    parsed = _nonnegative_integer(value)
    if parsed == 0:
        raise ValueError("expected positive integer")
    return parsed


def _iqr(values: list[float]) -> float:
    return _finite_derived(_quantile(values, 0.75) - _quantile(values, 0.25))


def _quantile(values: list[float], percentile: float) -> float:
    if not values or not math.isfinite(percentile) or not 0 <= percentile <= 1:
        raise ValueError("derived quantile inputs must be finite")
    if any(not math.isfinite(value) for value in values):
        raise ValueError("derived quantile inputs must be finite")
    ordered = sorted(values)
    position = (len(ordered) - 1) * percentile
    lower_index = math.floor(position)
    upper_index = math.ceil(position)
    if lower_index == upper_index:
        return _finite_derived(ordered[lower_index])
    fraction = position - lower_index
    return _finite_derived(
        ordered[lower_index]
        + fraction * (ordered[upper_index] - ordered[lower_index])
    )


def _bootstrap_lower_bound(
    eager_seconds: list[float],
    compiled_seconds: list[float],
    samples: int,
    confidence: float,
    seed: int,
) -> float | None:
    random_source = random.Random(seed)
    ratios: list[float] = []
    for _ in range(samples):
        eager_median = _sample_median(eager_seconds, random_source)
        compiled_median = _sample_median(compiled_seconds, random_source)
        if compiled_median == 0:
            return None
        ratios.append(_finite_derived(eager_median / compiled_median))
    return _quantile(ratios, 1.0 - confidence)


def _sample_median(values: list[float], random_source: random.Random) -> float:
    return _quantile([values[random_source.randrange(len(values))] for _ in values], 0.5)


def _finite_derived(value: float) -> float:
    if not math.isfinite(value):
        raise ValueError("derived statistics must be finite")
    return value

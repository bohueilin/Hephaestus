"""Torch-free deterministic mathematics for the A/A steady-state meta-test."""

from __future__ import annotations

import hashlib
import json
import math
import random
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import Final

import yaml

from hephaestus.bundle import canonical_json_bytes, strict_json_loads, verify_manifest
from hephaestus.evidence_contract import (
    V1_AA_ACTION_BY_WORKLOAD,
    stable_signed_paired_effect,
    strict_json_equal,
    v1_run_settings_json,
    v2_run_settings_json,
)
from hephaestus.gate import FROZEN_V1_CRITERIA, evaluate_bundle
from hephaestus.provenance import ProvenanceError, validate_provenance_chain
from hephaestus.scope import is_scope_json

_REPEATS = 31
_BOOTSTRAP_SAMPLES = 2000
_BOOTSTRAP_SEED = 0
_CONFIDENCE = 0.95
_WORKLOAD_NAMES: Final = {
    "mlp_stack",
    "transformer_block",
    "dynamic_batch_text",
    "graph_break_bait",
}
AA_CATALOG_IDS: Final = MappingProxyType(
    {
        "mlp_stack": "candidate-mlp-default",
        "transformer_block": "candidate-transformer-default",
        "dynamic_batch_text": "candidate-dynamic-true",
        "graph_break_bait": "candidate-graph-break-visible",
    }
)

_AA_METHODOLOGY: Final[dict[str, object]] = {
    "schema_version": 1,
    "run_design": "two_independent_complete_normal_runs",
    "independent_runs": 2,
    "repeats": 31,
    "pairing": "declared_iteration_ordinal",
    "effect_formula": "(A-B)/((A+B)/2)",
    "estimator": "p95_absolute_bootstrap_median",
    "quantile_method": "linear_interpolation",
    "bootstrap_samples": 2000,
    "bootstrap_seed": 0,
    "confidence": 0.95,
    "directional_bound_estimator": "normal_gate_bootstrap_speedup_lower_bound",
}
_AA_METHODOLOGY_V2: Final[dict[str, object]] = {
    **_AA_METHODOLOGY,
    "schema_version": 2,
    "repeats": 64,
}


@dataclass(frozen=True, slots=True)
class AANullStatistics:
    """All deterministic derivations from two independent compiled timing series."""

    signed_effects: tuple[float, ...]
    bootstrap_absolute_medians: tuple[float, ...]
    p95_noise_floor: float
    speedup_lower_bound_a_over_b: float | None
    speedup_lower_bound_b_over_a: float | None


@dataclass(frozen=True, slots=True)
class AATestResult:
    """Immutable CLI-facing result of pure stored A/A evidence evaluation."""

    parent_path: Path
    verdict: str
    driving_finding: str
    child_relative_paths: tuple[str, ...]
    statistics: AANullStatistics | None
    mismatches: tuple[str, ...] = ()


class _AADerivationError(ValueError):
    """A finite raw input set could not produce finite schema-1 statistics."""


def compute_null_statistics(
    series_a: tuple[float, ...] | list[float],
    series_b: tuple[float, ...] | list[float],
    *,
    schema_version: int = 1,
) -> AANullStatistics:
    """Recompute the frozen ordinal-paired null statistics from raw timings."""
    left = _timing_series(series_a)
    right = _timing_series(series_b)
    repeats = _repeats_for_version(schema_version)
    if len(left) != repeats or len(right) != repeats:
        raise ValueError(
            f"A/A schema-{schema_version} timing series must each contain exactly {repeats} values"
        )
    effects = tuple(
        _signed_paired_effect(a_value, b_value)
        for a_value, b_value in zip(left, right, strict=True)
    )
    bootstrap = tuple(_bootstrap_absolute_medians(effects))
    statistics = AANullStatistics(
        signed_effects=effects,
        bootstrap_absolute_medians=bootstrap,
        p95_noise_floor=_quantile(bootstrap, _CONFIDENCE),
        speedup_lower_bound_a_over_b=_bootstrap_lower_bound(left, right),
        speedup_lower_bound_b_over_a=_bootstrap_lower_bound(right, left),
    )
    _require_finite_statistics(statistics)
    return statistics


def validate_stored_null_statistics(
    series_a: tuple[float, ...] | list[float],
    series_b: tuple[float, ...] | list[float],
    stored: AANullStatistics,
    *,
    schema_version: int = 1,
) -> AANullStatistics:
    """Reject any stored derivation that differs from raw-series recomputation."""
    if not isinstance(stored, AANullStatistics):
        raise ValueError("aa.statistics")
    recomputed = compute_null_statistics(
        series_a,
        series_b,
        schema_version=schema_version,
    )
    if stored.signed_effects != recomputed.signed_effects:
        raise ValueError("aa.effects")
    if stored.bootstrap_absolute_medians != recomputed.bootstrap_absolute_medians:
        raise ValueError("aa.bootstrap")
    if stored.p95_noise_floor != recomputed.p95_noise_floor:
        raise ValueError("methodology.noise_floor")
    if (
        stored.speedup_lower_bound_a_over_b
        != recomputed.speedup_lower_bound_a_over_b
        or stored.speedup_lower_bound_b_over_a
        != recomputed.speedup_lower_bound_b_over_a
    ):
        raise ValueError("aa.directional_bounds")
    return recomputed


def evaluate_aa_statistics(
    statistics: AANullStatistics,
    *,
    minimum_speedup: float,
) -> tuple[str, str]:
    """Apply the frozen A/A decision ordering to already validated statistics."""
    if not isinstance(statistics, AANullStatistics):
        raise ValueError("statistics must be AANullStatistics")
    _require_finite_statistics(statistics)
    if (
        isinstance(minimum_speedup, bool)
        or not isinstance(minimum_speedup, int | float)
        or not math.isfinite(minimum_speedup)
        or minimum_speedup <= 1
    ):
        raise ValueError("minimum_speedup must be finite and greater than one")
    configured_effect = Decimal(str(minimum_speedup)) - Decimal("1")
    if configured_effect <= Decimal(str(statistics.p95_noise_floor)):
        return "FAIL", "methodology.noise_floor"
    directional_bounds = (
        statistics.speedup_lower_bound_a_over_b,
        statistics.speedup_lower_bound_b_over_a,
    )
    if any(
        bound is not None and bound >= minimum_speedup
        for bound in directional_bounds
    ):
        return "FAIL", "aa.false_positive"
    return "PASS", "all_criteria_passed"


def aa_methodology_json(*, schema_version: int = 1) -> dict[str, object]:
    """Return a fresh copy of the selected frozen A/A methodology."""
    _repeats_for_version(schema_version)
    return dict(_AA_METHODOLOGY if schema_version == 1 else _AA_METHODOLOGY_V2)


def aa_statistics_json(statistics: AANullStatistics) -> dict[str, object]:
    """Serialize all raw-derived null statistics without dropping precision."""
    return {
        "signed_paired_effects": list(statistics.signed_effects),
        "bootstrap_absolute_medians": list(
            statistics.bootstrap_absolute_medians
        ),
        "p95_noise_floor": statistics.p95_noise_floor,
        "speedup_lower_bound_a_over_b": statistics.speedup_lower_bound_a_over_b,
        "speedup_lower_bound_b_over_a": statistics.speedup_lower_bound_b_over_a,
    }


def aa_invalid_distribution_json(*, schema_version: int = 1) -> dict[str, object]:
    """Return the selected canonical distribution for invalid child evidence."""
    _repeats_for_version(schema_version)
    return {
        "schema_version": schema_version,
        "compiled_seconds_a": [],
        "compiled_seconds_b": [],
        "signed_paired_effects": [],
        "bootstrap_absolute_medians": [],
        "p95_noise_floor": None,
        "speedup_lower_bound_a_over_b": None,
        "speedup_lower_bound_b_over_a": None,
    }


def aa_verdict_json(result: AATestResult) -> dict[str, object]:
    """Return the canonical parent verdict payload for finalization and CLI use."""
    finding_status = "PASS" if result.verdict == "PASS" else "FAIL"
    return {
        "schema_version": 1,
        "verdict": result.verdict,
        "driving_finding": result.driving_finding,
        "findings": [
            {
                "id": result.driving_finding,
                "status": finding_status,
                "mismatches": list(result.mismatches),
            }
        ],
        "statistics": (
            None
            if result.statistics is None
            else aa_statistics_json(result.statistics)
        ),
    }


def evaluate_aa_bundle(parent: Path) -> AATestResult:
    """Purely re-evaluate a complete A/A tree from manifested stored evidence."""
    return _evaluate_aa_bundle(parent, allow_provisional=False)


def _evaluate_provisional_aa_bundle(parent: Path) -> AATestResult:
    """Evaluate verdict-less A/A topology only for trusted parent finalization."""
    return _evaluate_aa_bundle(parent, allow_provisional=True)


def _evaluate_aa_bundle(parent: Path, *, allow_provisional: bool) -> AATestResult:
    """Shared evaluator with provisional authority kept off the public surface."""
    parent = Path(parent)
    integrity = verify_manifest(parent)
    if not integrity.valid:
        return _invalid_result(
            parent,
            "evidence.integrity",
            integrity.mismatches,
        )
    try:
        result = _evaluate_aa_semantics(parent, allow_provisional=allow_provisional)
    except _AAEvidenceError as error:
        result = _invalid_result(
            parent,
            error.finding,
            error.mismatches,
            child_paths=error.child_paths,
        )

    verdict_path = parent / "verdict.json"
    if verdict_path.exists():
        try:
            stored_verdict = verdict_path.read_bytes()
        except OSError:
            return _invalid_result(
                parent,
                "parent.verdict",
                ("verdict:unreadable",),
                child_paths=result.child_relative_paths,
            )
        if stored_verdict != canonical_json_bytes(aa_verdict_json(result)):
            return _invalid_result(
                parent,
                "parent.verdict",
                ("verdict:semantic_mismatch", *result.mismatches),
                child_paths=result.child_relative_paths,
            )
    elif not allow_provisional:
        return _invalid_result(
            parent,
            "parent.topology",
            ("parent:missing:verdict.json",),
            child_paths=result.child_relative_paths,
        )
    return result


class _AAEvidenceError(ValueError):
    def __init__(
        self,
        finding: str,
        *mismatches: str,
        child_paths: tuple[str, ...] = (),
    ) -> None:
        super().__init__(finding)
        self.finding = finding
        self.mismatches = mismatches or (finding,)
        self.child_paths = child_paths


def _evaluate_aa_semantics(parent: Path, *, allow_provisional: bool) -> AATestResult:
    expected_top = {
        "aa_test.json",
        "aa_methodology.json",
        "aa_distribution.json",
        "gate_criteria.yaml",
        "scope.json",
        "runs",
        "manifest.json",
    }
    if not allow_provisional or (parent / "verdict.json").exists():
        expected_top.add("verdict.json")
    try:
        actual_top = {path.name for path in parent.iterdir()}
    except OSError as error:
        raise _AAEvidenceError("parent.topology", "parent:missing") from error
    topology = [
        *(f"parent:missing:{name}" for name in sorted(expected_top - actual_top)),
        *(f"parent:unexpected:{name}" for name in sorted(actual_top - expected_top)),
    ]
    if topology:
        child_authority_omissions = {
            "parent:missing:aa_test.json",
            "parent:missing:runs",
        }
        if child_authority_omissions.intersection(topology):
            _require_invalid_distribution(
                parent,
                (),
                schema_version=_placeholder_schema_version(parent),
            )
        raise _AAEvidenceError("parent.topology", *topology)

    if not is_scope_json(_json_object(parent / "scope.json")):
        raise _AAEvidenceError("scope.boundary", "scope:invalid")
    methodology = _json_object(parent / "aa_methodology.json")
    if strict_json_equal(methodology, _AA_METHODOLOGY):
        schema_version = 1
    elif strict_json_equal(methodology, _AA_METHODOLOGY_V2):
        schema_version = 2
    else:
        raise _AAEvidenceError("methodology.valid", "aa_methodology:invalid")
    criteria_bytes = _read_bytes(parent / "gate_criteria.yaml", "criteria:invalid")
    try:
        criteria = yaml.safe_load(criteria_bytes)
    except yaml.YAMLError as error:
        raise _AAEvidenceError("methodology.criteria", "criteria:invalid") from error
    if not strict_json_equal(criteria, FROZEN_V1_CRITERIA):
        raise _AAEvidenceError("methodology.criteria", "criteria:not_frozen_v1")

    child_paths: tuple[str, ...] = ()
    try:
        aa_test = _json_object(parent / "aa_test.json")
        workload, catalog_id, child_records = _parse_aa_test(
            aa_test,
            schema_version=schema_version,
        )
        child_paths = tuple(
            record["bundle_relative_path"] for record in child_records
        )
        contract = V1_AA_ACTION_BY_WORKLOAD[workload]
        if catalog_id != contract.catalog_id:
            raise _AAEvidenceError(
                "aa.catalog", "catalog_id:invalid", child_paths=child_paths
            )
        runs_root = parent / "runs"
        try:
            actual_children = {path.name for path in runs_root.iterdir()}
        except OSError as error:
            raise _AAEvidenceError(
                "parent.topology", "runs:missing", child_paths=child_paths
            ) from error
        expected_children = {
            relative.split("/", maxsplit=1)[1] for relative in child_paths
        }
        child_topology = [
            *(
                f"runs:missing:{name}"
                for name in sorted(expected_children - actual_children)
            ),
            *(
                f"runs:unexpected:{name}"
                for name in sorted(actual_children - expected_children)
            ),
        ]
        if child_topology:
            raise _AAEvidenceError(
                "parent.topology", *child_topology, child_paths=child_paths
            )
        children = _validated_aa_children(
            parent,
            child_records,
            child_paths,
            criteria_bytes=criteria_bytes,
            workload=workload,
            schema_version=schema_version,
        )
    except _AAEvidenceError as error:
        if error.finding != "methodology.version":
            _require_invalid_distribution(
                parent,
                error.child_paths or child_paths,
                schema_version=schema_version,
            )
        raise

    series = tuple(
        _compiled_series(_json_object(child / "timings.json")) for child in children
    )
    try:
        compute_null_statistics(
            series[0],
            series[1],
            schema_version=schema_version,
        )
    except (ArithmeticError, ValueError) as error:
        _require_invalid_distribution(
            parent,
            child_paths,
            schema_version=schema_version,
        )
        raise _AAEvidenceError(
            "aa.derivation",
            "distribution:derivation",
            child_paths=child_paths,
        ) from error
    distribution = _json_object(parent / "aa_distribution.json")
    if strict_json_equal(
        distribution,
        aa_invalid_distribution_json(schema_version=schema_version),
    ):
        raise _AAEvidenceError(
            "aa.derivation",
            "distribution:unexpected_invalid_placeholder",
            child_paths=child_paths,
        )
    stored_series, stored_statistics = _parse_distribution(
        distribution,
        schema_version=schema_version,
    )
    if stored_series != series:
        raise _AAEvidenceError(
            "aa.raw_series", "distribution:raw_series", child_paths=child_paths
        )
    try:
        statistics = validate_stored_null_statistics(
            series[0],
            series[1],
            stored_statistics,
            schema_version=schema_version,
        )
    except _AADerivationError as error:
        raise _AAEvidenceError(
            "aa.derivation", "distribution:derivation", child_paths=child_paths
        ) from error
    except ValueError as error:
        raise _AAEvidenceError(
            str(error), "distribution:derivation", child_paths=child_paths
        ) from error
    verdict, finding = evaluate_aa_statistics(
        statistics,
        minimum_speedup=float(criteria["minimum_speedup"]),
    )
    return AATestResult(
        parent,
        verdict,
        finding,
        child_paths,
        statistics,
    )


def _validated_aa_children(
    parent: Path,
    child_records: tuple[dict[str, str], dict[str, str]],
    child_paths: tuple[str, ...],
    *,
    criteria_bytes: bytes,
    workload: str,
    schema_version: int,
) -> tuple[Path, Path]:
    identity_names = (
        "config.json",
        "env.json",
        "manifest.json",
        "input_plan.json",
        "workload.digest",
        "gate_criteria.yaml",
    )
    child_identity: list[dict[str, bytes]] = []
    computed_digests: list[dict[str, str]] = []
    children: list[Path] = []
    for record in child_records:
        relative = record["bundle_relative_path"]
        child = parent / relative
        children.append(child)
        child_integrity = verify_manifest(child)
        if not child_integrity.valid:
            raise _AAEvidenceError(
                "child.integrity",
                *(f"{relative}:{item}" for item in child_integrity.mismatches),
                child_paths=child_paths,
            )
        identity = {
            name: _read_bytes(child / name, f"{relative}:{name}:invalid")
            for name in identity_names
        }
        child_identity.append(identity)
        computed_digests.append(
            {
                "config_sha256": _sha256(identity["config.json"]),
                "env_sha256": _sha256(identity["env.json"]),
                "input_plan_sha256": _sha256(identity["input_plan.json"]),
                "workload_sha256": _sha256(identity["workload.digest"]),
                "criteria_sha256": _sha256(identity["gate_criteria.yaml"]),
                "bundle_manifest_sha256": _sha256(identity["manifest.json"]),
            }
        )

    if child_identity[0]["env.json"] != child_identity[1]["env.json"]:
        raise _AAEvidenceError(
            "aa.environment", "children:environment_mismatch", child_paths=child_paths
        )
    comparable_names = tuple(
        name for name in identity_names if name not in {"env.json", "manifest.json"}
    )
    if any(
        child_identity[0][name] != child_identity[1][name]
        for name in comparable_names
    ) or any(
        identity["gate_criteria.yaml"] != criteria_bytes for identity in child_identity
    ):
        raise _AAEvidenceError(
            "aa.child_equality", "children:identity_mismatch", child_paths=child_paths
        )
    try:
        validate_provenance_chain(tuple(children))
    except ProvenanceError as error:
        raise _AAEvidenceError(
            "aa.provenance", str(error), child_paths=child_paths
        ) from error

    for record, expected_digests in zip(
        child_records, computed_digests, strict=True
    ):
        if any(record[key] != digest for key, digest in expected_digests.items()):
            raise _AAEvidenceError(
                "aa.child_digest",
                f"{record['bundle_relative_path']}:digest",
                child_paths=child_paths,
            )

    try:
        child_config = strict_json_loads(child_identity[0]["config.json"])
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        raise _AAEvidenceError(
            "aa.catalog", "child:config", child_paths=child_paths
        ) from error
    if not isinstance(child_config, dict):
        raise _AAEvidenceError("aa.catalog", "child:config", child_paths=child_paths)
    contract = V1_AA_ACTION_BY_WORKLOAD[workload]
    if not strict_json_equal(child_config.get("catalog"), contract.metadata_json()):
        raise _AAEvidenceError(
            "aa.catalog", "child:catalog_id", child_paths=child_paths
        )

    for child, relative in zip(children, child_paths, strict=True):
        methodology = _json_object(child / "methodology.json")
        child_version = (
            1
            if "schema_version" not in methodology
            else methodology.get("schema_version")
        )
        if type(child_version) is not int or child_version != schema_version:
            raise _AAEvidenceError(
                "methodology.version",
                f"{relative}:version",
                child_paths=child_paths,
            )
        expected_settings = (
            v1_run_settings_json()
            if schema_version == 1
            else v2_run_settings_json()
        )
        frozen_child_settings = {
            key: methodology.get(key) for key in expected_settings
        }
        if not strict_json_equal(frozen_child_settings, expected_settings):
            raise _AAEvidenceError(
                "methodology.settings",
                f"{relative}:settings",
                child_paths=child_paths,
            )
        stored_verdict = _read_bytes(child / "verdict.json", f"{relative}:verdict")
        regated = evaluate_bundle(child)
        if stored_verdict != canonical_json_bytes(regated):
            raise _AAEvidenceError(
                "child.gate_verdict",
                f"{relative}:verdict",
                child_paths=child_paths,
            )
        if regated.get("verdict") == "INVALID_EVIDENCE":
            raise _AAEvidenceError(
                "child.invalid_evidence",
                f"{relative}:{regated.get('driving_finding')}",
                child_paths=child_paths,
            )
    return children[0], children[1]


def _require_invalid_distribution(
    parent: Path,
    child_paths: tuple[str, ...],
    *,
    schema_version: int = 1,
) -> None:
    try:
        distribution = _json_object(parent / "aa_distribution.json")
    except _AAEvidenceError as error:
        raise _AAEvidenceError(
            "aa.schema",
            "aa_distribution:invalid",
            child_paths=child_paths,
        ) from error
    if not strict_json_equal(
        distribution,
        aa_invalid_distribution_json(schema_version=schema_version),
    ):
        raise _AAEvidenceError(
            "aa.schema",
            "aa_distribution:invalid",
            child_paths=child_paths,
        )


def _parse_aa_test(
    value: dict[str, object],
    *,
    schema_version: int = 1,
) -> tuple[str, str, tuple[dict[str, str], dict[str, str]]]:
    if value.keys() != {"schema_version", "workload_name", "catalog_id", "children"}:
        raise _AAEvidenceError("aa.schema", "aa_test:keys")
    if (
        type(value.get("schema_version")) is not int
        or value["schema_version"] != schema_version
    ):
        raise _AAEvidenceError("aa.schema", "aa_test:schema")
    workload = value.get("workload_name")
    catalog_id = value.get("catalog_id")
    children = value.get("children")
    if (
        workload not in _WORKLOAD_NAMES
        or not isinstance(catalog_id, str)
        or not catalog_id
        or catalog_id != catalog_id.strip()
    ):
        raise _AAEvidenceError("aa.schema", "aa_test:identity")
    if not isinstance(children, list) or len(children) != 2:
        raise _AAEvidenceError("aa.schema", "aa_test:children")
    parsed: list[dict[str, str]] = []
    expected_keys = {
        "ordinal",
        "bundle_relative_path",
        "config_sha256",
        "env_sha256",
        "input_plan_sha256",
        "workload_sha256",
        "criteria_sha256",
        "bundle_manifest_sha256",
    }
    for index, child in enumerate(children):
        if not isinstance(child, dict) or child.keys() != expected_keys:
            raise _AAEvidenceError("aa.schema", "aa_test:child")
        if child.get("ordinal") != ("A" if index == 0 else "B"):
            raise _AAEvidenceError("aa.schema", "aa_test:ordinal")
        relative = child.get("bundle_relative_path")
        if not _safe_child_path(relative):
            raise _AAEvidenceError("aa.child_path", "aa_test:path")
        if any(not _is_sha256(child.get(key)) for key in expected_keys if key.endswith("sha256")):
            raise _AAEvidenceError("aa.child_digest", "aa_test:digest")
        parsed.append({key: str(child[key]) for key in expected_keys})
    paths = tuple(item["bundle_relative_path"] for item in parsed)
    if len(set(paths)) != 2:
        raise _AAEvidenceError("aa.child_path", "aa_test:duplicate_path")
    return workload, catalog_id, (parsed[0], parsed[1])


def _parse_distribution(
    value: dict[str, object],
    *,
    schema_version: int = 1,
) -> tuple[tuple[tuple[float, ...], tuple[float, ...]], AANullStatistics]:
    expected_keys = {
        "schema_version",
        "compiled_seconds_a",
        "compiled_seconds_b",
        "signed_paired_effects",
        "bootstrap_absolute_medians",
        "p95_noise_floor",
        "speedup_lower_bound_a_over_b",
        "speedup_lower_bound_b_over_a",
    }
    if (
        value.keys() != expected_keys
        or type(value.get("schema_version")) is not int
        or value["schema_version"] != schema_version
    ):
        raise _AAEvidenceError("aa.schema", "aa_distribution:invalid")
    try:
        series_a = _numbers(value["compiled_seconds_a"], nonnegative=True)
        series_b = _numbers(value["compiled_seconds_b"], nonnegative=True)
        statistics = AANullStatistics(
            signed_effects=_numbers(value["signed_paired_effects"], nonnegative=False),
            bootstrap_absolute_medians=_numbers(
                value["bootstrap_absolute_medians"], nonnegative=True
            ),
            p95_noise_floor=_number(value["p95_noise_floor"], nonnegative=True),
            speedup_lower_bound_a_over_b=_optional_number(
                value["speedup_lower_bound_a_over_b"]
            ),
            speedup_lower_bound_b_over_a=_optional_number(
                value["speedup_lower_bound_b_over_a"]
            ),
        )
    except (KeyError, ValueError) as error:
        raise _AAEvidenceError("aa.schema", "aa_distribution:invalid") from error
    return (series_a, series_b), statistics


def _compiled_series(timings: dict[str, object]) -> tuple[float, ...]:
    try:
        return _numbers(timings["compiled_seconds"], nonnegative=True)
    except (KeyError, ValueError) as error:
        raise _AAEvidenceError("aa.raw_series", "child:compiled_seconds") from error


def _numbers(value: object, *, nonnegative: bool) -> tuple[float, ...]:
    if not isinstance(value, list | tuple):
        raise ValueError("expected numeric series")
    return tuple(_number(item, nonnegative=nonnegative) for item in value)


def _number(value: object, *, nonnegative: bool) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError("expected number")
    try:
        parsed = float(value)
    except OverflowError as error:
        raise ValueError("expected finite number") from error
    if not math.isfinite(parsed) or (nonnegative and parsed < 0):
        raise ValueError("expected finite number")
    return parsed


def _optional_number(value: object) -> float | None:
    if value is None:
        return None
    return _number(value, nonnegative=True)


def _json_object(path: Path) -> dict[str, object]:
    try:
        value = strict_json_loads(path.read_bytes())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        raise _AAEvidenceError("aa.schema", f"{path.name}:invalid") from error
    if not isinstance(value, dict):
        raise _AAEvidenceError("aa.schema", f"{path.name}:invalid")
    return value


def _read_bytes(path: Path, mismatch: str) -> bytes:
    try:
        return path.read_bytes()
    except OSError as error:
        raise _AAEvidenceError("aa.schema", mismatch) from error


def _safe_child_path(value: object) -> bool:
    if not isinstance(value, str) or "\\" in value:
        return False
    path = PurePosixPath(value)
    return (
        not path.is_absolute()
        and len(path.parts) == 2
        and path.parts[0] == "runs"
        and path.parts[1] not in {"", ".", ".."}
        and path.as_posix() == value
    )


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _is_sha256(value: object) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(
        character in "0123456789abcdef" for character in value
    )


def _schema_one(value: object) -> bool:
    return (
        isinstance(value, dict)
        and type(value.get("schema_version")) is int
        and value["schema_version"] == 1
    )


def _repeats_for_version(schema_version: int) -> int:
    if type(schema_version) is not int or schema_version not in {1, 2}:
        raise ValueError("A/A schema_version must be exactly integer 1 or 2")
    return _REPEATS if schema_version == 1 else 64


def _placeholder_schema_version(parent: Path) -> int:
    try:
        methodology = _json_object(parent / "aa_methodology.json")
    except _AAEvidenceError:
        return 1
    return 2 if strict_json_equal(methodology, _AA_METHODOLOGY_V2) else 1


def _invalid_result(
    parent: Path,
    finding: str,
    mismatches: tuple[str, ...],
    *,
    child_paths: tuple[str, ...] = (),
) -> AATestResult:
    return AATestResult(
        parent,
        "INVALID_EVIDENCE",
        finding,
        child_paths,
        None,
        tuple(dict.fromkeys(mismatches)),
    )


def _timing_series(value: object) -> tuple[float, ...]:
    if not isinstance(value, tuple | list):
        raise ValueError("A/A timing series must be a sequence")
    try:
        return tuple(_number(item, nonnegative=True) for item in value)
    except ValueError as error:
        raise ValueError("A/A timings must be finite and nonnegative") from error


def _signed_paired_effect(left: float, right: float) -> float:
    try:
        return _finite_derived(stable_signed_paired_effect(left, right))
    except ValueError as error:
        raise _AADerivationError("A/A paired effect must be finite") from error


def _bootstrap_absolute_medians(effects: tuple[float, ...]) -> list[float]:
    random_source = random.Random(_BOOTSTRAP_SEED)
    return [
        _finite_derived(abs(_sample_median(effects, random_source)))
        for _ in range(_BOOTSTRAP_SAMPLES)
    ]


def _bootstrap_lower_bound(
    numerator: tuple[float, ...],
    denominator: tuple[float, ...],
) -> float | None:
    random_source = random.Random(_BOOTSTRAP_SEED)
    ratios: list[float] = []
    for _ in range(_BOOTSTRAP_SAMPLES):
        numerator_median = _sample_median(numerator, random_source)
        denominator_median = _sample_median(denominator, random_source)
        if denominator_median == 0:
            return None
        ratios.append(_finite_derived(numerator_median / denominator_median))
    return _quantile(ratios, 1.0 - _CONFIDENCE)


def _sample_median(
    values: tuple[float, ...], random_source: random.Random
) -> float:
    return _quantile(
        [values[random_source.randrange(len(values))] for _ in values],
        0.5,
    )


def _quantile(values: tuple[float, ...] | list[float], percentile: float) -> float:
    if not values or not math.isfinite(percentile) or not 0 <= percentile <= 1:
        raise _AADerivationError("A/A quantile inputs must be finite")
    if any(not math.isfinite(value) for value in values):
        raise _AADerivationError("A/A quantile inputs must be finite")
    ordered = sorted(values)
    position = (len(ordered) - 1) * percentile
    lower_index = math.floor(position)
    upper_index = math.ceil(position)
    if lower_index == upper_index:
        return _finite_derived(ordered[lower_index])
    fraction = position - lower_index
    lower = ordered[lower_index]
    upper = ordered[upper_index]
    interpolated = lower + fraction * (upper - lower)
    return _finite_derived(interpolated)


def _require_finite_statistics(statistics: AANullStatistics) -> None:
    values = (
        *statistics.signed_effects,
        *statistics.bootstrap_absolute_medians,
        statistics.p95_noise_floor,
        *(
            bound
            for bound in (
                statistics.speedup_lower_bound_a_over_b,
                statistics.speedup_lower_bound_b_over_a,
            )
            if bound is not None
        ),
    )
    if any(not math.isfinite(value) for value in values):
        raise _AADerivationError("A/A derived statistics must be finite")


def _finite_derived(value: float) -> float:
    if not math.isfinite(value):
        raise _AADerivationError("A/A derived statistics must be finite")
    return value


__all__ = [
    "AA_CATALOG_IDS",
    "AANullStatistics",
    "AATestResult",
    "aa_invalid_distribution_json",
    "aa_methodology_json",
    "aa_statistics_json",
    "aa_verdict_json",
    "compute_null_statistics",
    "evaluate_aa_bundle",
    "evaluate_aa_statistics",
    "validate_stored_null_statistics",
]

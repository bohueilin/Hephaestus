"""Trusted execution and evidence finalization for independent-run A/A tests."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from hephaestus.aa import (
    AA_CATALOG_IDS,
    AATestResult,
    _evaluate_provisional_aa_bundle,
    aa_invalid_distribution_json,
    aa_methodology_json,
    aa_statistics_json,
    aa_verdict_json,
    compute_null_statistics,
    evaluate_aa_bundle,
)
from hephaestus.bundle import (
    canonical_json_bytes,
    strict_json_loads,
    verify_manifest,
    write_json,
    write_manifest,
)
from hephaestus.catalog import Proposal, WorkloadName, get_catalog_entry
from hephaestus.catalog_runtime import CatalogRuntime
from hephaestus.durability import prepare_host_state_output_root
from hephaestus.evidence_contract import (
    V1_AA_ACTION_BY_WORKLOAD,
    strict_json_equal,
    v2_run_settings_json,
)
from hephaestus.gate import evaluate_bundle
from hephaestus.measure import RunResult, RunSettings, run_to_bundle
from hephaestus.provenance import ProvenanceError, validate_provenance_chain
from hephaestus.scope import scope_json


class TrustedAAOrchestrator:
    """Run exactly two reviewed normal actions and seal their null evidence tree."""

    def __init__(
        self,
        output_root: Path,
        criteria_path: Path,
        *,
        capability_snapshot: Mapping[str, object] | None = None,
        runner: Callable[..., RunResult] = run_to_bundle,
        _worker_relative_root: bool = False,
        _host_state_sink: object | None = None,
    ) -> None:
        self._output_root = (
            Path(output_root)
            if _worker_relative_root
            else Path(output_root).resolve(strict=False)
        )
        self._criteria_path = Path(criteria_path).resolve(strict=False)
        self._capability_snapshot = capability_snapshot
        self._runner = runner
        self._worker_relative_root = _worker_relative_root
        self._host_state_sink = _host_state_sink

    def run(self, workload_name: str | WorkloadName) -> AATestResult:
        """Execute two independent complete bundles with exact frozen settings."""
        workload = _workload_name(workload_name)
        if self._host_state_sink is None:
            with prepare_host_state_output_root(self._output_root) as sink:
                return self._run(workload, sink)
        return self._run(workload, self._host_state_sink)

    def _run(self, workload: WorkloadName, host_state_sink: object) -> AATestResult:
        parent = self._output_root / _parent_name(workload)
        (parent / "runs").mkdir(parents=True, exist_ok=False)
        runtime = CatalogRuntime(
            parent / "runs",
            self._criteria_path,
            RunSettings(schema_version=2, repeats=64),
            _host_state_sink=host_state_sink,  # type: ignore[arg-type]
            capability_snapshot=self._capability_snapshot,
            runner=self._runner,
            _worker_relative_root=self._worker_relative_root,
        )
        entry = get_catalog_entry(AA_CATALOG_IDS[workload.value])
        proposal = Proposal(entry.catalog_id, entry.workload, entry.rationale)
        first = runtime.run(proposal)
        second = runtime.run(proposal)
        child_paths = (first.bundle_relative_path, second.bundle_relative_path)
        criteria_bytes = self._criteria_path.read_bytes()
        preflight = _preflight_children(
            parent,
            child_paths,
            workload_name=workload.value,
            criteria_bytes=criteria_bytes,
        )

        if not preflight.valid:
            distribution = aa_invalid_distribution_json(schema_version=2)
        else:
            assert preflight.compiled_series is not None
            series = preflight.compiled_series
            try:
                statistics = compute_null_statistics(
                    series[0],
                    series[1],
                    schema_version=2,
                )
            except (ArithmeticError, ValueError):
                distribution = aa_invalid_distribution_json(schema_version=2)
            else:
                distribution = {
                    "schema_version": 2,
                    "compiled_seconds_a": list(series[0]),
                    "compiled_seconds_b": list(series[1]),
                    **aa_statistics_json(statistics),
                }
        (parent / "gate_criteria.yaml").write_bytes(criteria_bytes)
        write_json(parent / "scope.json", scope_json())
        write_json(
            parent / "aa_methodology.json",
            aa_methodology_json(schema_version=2),
        )
        write_json(parent / "aa_distribution.json", distribution)
        write_json(
            parent / "aa_test.json",
            {
                "schema_version": 2,
                "workload_name": workload.value,
                "catalog_id": entry.catalog_id,
                "children": [
                    _child_record(ordinal, relative, identity)
                    for ordinal, relative, identity in zip(
                        ("A", "B"),
                        child_paths,
                        preflight.identities,
                        strict=True,
                    )
                ],
            },
        )
        return _finalize_aa_parent(parent)


def run_aa_test(
    workload_name: WorkloadName,
    output_root: Path,
    criteria_path: Path,
    *,
    _worker_relative_root: bool = False,
    _host_state_sink: object | None = None,
) -> AATestResult:
    """Execute the public A/A workflow with no caller-controlled methodology."""
    if not isinstance(workload_name, WorkloadName):
        raise ValueError("workload_name must be a WorkloadName")
    return TrustedAAOrchestrator(
        output_root,
        criteria_path,
        _worker_relative_root=_worker_relative_root,
        _host_state_sink=_host_state_sink,
    ).run(workload_name)


def _finalize_aa_parent(parent: Path) -> AATestResult:
    write_manifest(parent)
    provisional = _evaluate_provisional_aa_bundle(parent)
    write_json(parent / "verdict.json", aa_verdict_json(provisional))
    write_manifest(parent)
    integrity = verify_manifest(parent)
    if not integrity.valid:
        raise RuntimeError(f"finalized A/A tree failed integrity: {integrity.mismatches}")
    final = evaluate_aa_bundle(parent)
    stored = (parent / "verdict.json").read_bytes()
    if stored != canonical_json_bytes(aa_verdict_json(final)):
        raise RuntimeError("stored A/A verdict differs from finalized offline evaluation")
    return final


def _child_record(
    ordinal: str,
    relative: str,
    identity: dict[str, bytes],
) -> dict[str, object]:
    return {
        "ordinal": ordinal,
        "bundle_relative_path": relative,
        "config_sha256": _sha256(identity["config.json"]),
        "env_sha256": _sha256(identity["env.json"]),
        "input_plan_sha256": _sha256(identity["input_plan.json"]),
        "workload_sha256": _sha256(identity["workload.digest"]),
        "criteria_sha256": _sha256(identity["gate_criteria.yaml"]),
        "bundle_manifest_sha256": _sha256(identity["manifest.json"]),
    }


@dataclass(frozen=True, slots=True)
class _AAPreflight:
    valid: bool
    identities: tuple[dict[str, bytes], dict[str, bytes]]
    compiled_series: tuple[tuple[float, ...], tuple[float, ...]] | None


def _preflight_children(
    parent: Path,
    child_paths: tuple[str, str],
    *,
    workload_name: str,
    criteria_bytes: bytes,
) -> _AAPreflight:
    """Snapshot and validate all child authority before consuming any timings."""
    valid = True
    identities: list[dict[str, bytes]] = []
    compiled_series: list[tuple[float, ...]] = []
    contract = V1_AA_ACTION_BY_WORKLOAD[workload_name]
    try:
        validate_provenance_chain(tuple(parent / relative for relative in child_paths))
    except ProvenanceError:
        valid = False
    for relative in child_paths:
        child = parent / relative
        integrity = verify_manifest(child)
        if not integrity.valid:
            valid = False
        identity, complete = _snapshot_child_identity(child)
        identities.append(identity)
        if not complete:
            valid = False
        try:
            stored_verdict = (child / "verdict.json").read_bytes()
        except OSError:
            stored_verdict = b""
            valid = False
        try:
            regated = evaluate_bundle(child)
        except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError):
            regated = {"verdict": "INVALID_EVIDENCE"}
            valid = False
        if (
            stored_verdict != canonical_json_bytes(regated)
            or regated.get("verdict") == "INVALID_EVIDENCE"
        ):
            valid = False
        try:
            config = strict_json_loads(identity["config.json"])
        except (UnicodeDecodeError, json.JSONDecodeError, ValueError):
            config = None
        if (
            not isinstance(config, dict)
            or not strict_json_equal(config.get("catalog"), contract.metadata_json())
        ):
            valid = False
        try:
            methodology = strict_json_loads((child / "methodology.json").read_bytes())
        except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError):
            methodology = None
        if not isinstance(methodology, dict):
            valid = False
        else:
            expected_settings = v2_run_settings_json()
            settings = {key: methodology.get(key) for key in expected_settings}
            if not strict_json_equal(settings, expected_settings):
                valid = False
        try:
            compiled_series.append(
                _compiled_seconds(child / "timings.json", schema_version=2)
            )
        except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError):
            compiled_series.append(())
            valid = False
    comparable_names = (
        "config.json",
        "input_plan.json",
        "workload.digest",
        "gate_criteria.yaml",
    )
    if (
        any(identities[0][name] != identities[1][name] for name in comparable_names)
        or identities[0]["env.json"] != identities[1]["env.json"]
        or any(identity["gate_criteria.yaml"] != criteria_bytes for identity in identities)
    ):
        valid = False
    series = (
        (compiled_series[0], compiled_series[1])
        if valid
        else None
    )
    return _AAPreflight(valid, (identities[0], identities[1]), series)


def _snapshot_child_identity(child: Path) -> tuple[dict[str, bytes], bool]:
    identity: dict[str, bytes] = {}
    complete = True
    for name in (
        "config.json",
        "env.json",
        "manifest.json",
        "input_plan.json",
        "workload.digest",
        "gate_criteria.yaml",
    ):
        try:
            identity[name] = (child / name).read_bytes()
        except OSError:
            identity[name] = b""
            complete = False
    return identity, complete


def _compiled_seconds(path: Path, *, schema_version: int) -> tuple[float, ...]:
    loaded = strict_json_loads(path.read_bytes())
    if not isinstance(loaded, dict) or not isinstance(loaded.get("compiled_seconds"), list):
        raise ValueError("child timings lack compiled_seconds")
    values = loaded["compiled_seconds"]
    expected_repeats = 31 if schema_version == 1 else 64
    if len(values) != expected_repeats:
        raise ValueError("child timings contain an invalid compiled series")
    parsed: list[float] = []
    for value in values:
        if isinstance(value, bool) or not isinstance(value, int | float):
            raise ValueError("child timings contain an invalid compiled series")
        try:
            number = float(value)
        except OverflowError as error:
            raise ValueError(
                "child timings contain an invalid compiled series"
            ) from error
        if not math.isfinite(number) or number < 0:
            raise ValueError("child timings contain an invalid compiled series")
        parsed.append(number)
    return tuple(parsed)


def _workload_name(value: str | WorkloadName) -> WorkloadName:
    if isinstance(value, WorkloadName):
        return value
    try:
        return WorkloadName(value)
    except (TypeError, ValueError) as error:
        raise ValueError("unknown workload") from error


def _parent_name(workload: WorkloadName) -> str:
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S.%fZ")
    return f"aa-test-{timestamp}-{workload.value}"


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


__all__ = ["TrustedAAOrchestrator", "run_aa_test"]

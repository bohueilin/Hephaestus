"""Trusted scripted-search execution and recursively manifested parent evidence."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from hephaestus.agent import AgentObservation, next_proposal, scripted_policy
from hephaestus.bundle import (
    IntegrityResult,
    canonical_json_bytes,
    strict_json_loads,
    verify_manifest,
    write_json,
    write_manifest,
)
from hephaestus.catalog import Proposal, ReadOnlyRunResult, WorkloadName, catalog_json
from hephaestus.catalog_runtime import CatalogRuntime, ExecutionReceipt
from hephaestus.durability import prepare_host_state_output_root
from hephaestus.evidence_contract import (
    V1_SEARCH_CANDIDATES_BY_WORKLOAD,
    strict_json_equal,
    v2_run_settings_json,
)
from hephaestus.gate import evaluate_bundle, validate_catalog_metadata
from hephaestus.measure import RunResult, RunSettings, run_to_bundle
from hephaestus.provenance import ProvenanceError, validate_provenance_chain
from hephaestus.scope import is_scope_json, scope_json

_WORKLOAD_NAMES = {
    "mlp_stack",
    "transformer_block",
    "dynamic_batch_text",
    "graph_break_bait",
}
_VERDICTS = {"PROVEN", "CONDITIONAL", "NOT_PROVEN", "INVALID_EVIDENCE"}
_CATALOG_METADATA_KEYS = {
    "schema_version",
    "entry_id",
    "role",
    "workload_name",
    "requested",
    "effective",
}


@dataclass(frozen=True, slots=True)
class SearchStep:
    """One trusted proposal/result receipt retained outside the agent boundary."""

    proposal: Proposal
    result: ReadOnlyRunResult


@dataclass(frozen=True, slots=True)
class SearchTranscript:
    """The trusted ordered trace, including child evidence paths."""

    workload_name: WorkloadName
    steps: tuple[SearchStep, ...]

    @property
    def final_result(self) -> ReadOnlyRunResult | None:
        if self.steps and self.steps[-1].result.verdict == "PROVEN":
            return self.steps[-1].result
        return None


@dataclass(frozen=True, slots=True)
class SearchRunResult:
    """Trusted parent path plus the immutable optimizer-visible transcript."""

    parent_path: Path
    transcript: SearchTranscript

    @property
    def final_result(self) -> ReadOnlyRunResult | None:
        return self.transcript.final_result


@dataclass(frozen=True, slots=True)
class _StoredCatalogEntry:
    catalog_id: str
    role: str
    workload_name: str
    rationale: str
    metadata: dict[str, object]


@dataclass(frozen=True, slots=True)
class _StoredCatalog:
    entries: tuple[_StoredCatalogEntry, ...]
    by_id: dict[str, _StoredCatalogEntry]

    def candidates(self, workload_name: str) -> tuple[_StoredCatalogEntry, ...]:
        return tuple(
            entry
            for entry in self.entries
            if entry.role == "candidate" and entry.workload_name == workload_name
        )


@dataclass(frozen=True, slots=True)
class _StoredResult:
    bundle_relative_path: str
    verdict: str
    driving_finding: str


@dataclass(frozen=True, slots=True)
class _StoredStep:
    entry: _StoredCatalogEntry
    result: _StoredResult


@dataclass(frozen=True, slots=True)
class _StoredTranscript:
    workload_name: str
    steps: tuple[_StoredStep, ...]


class _StoredEvidenceError(ValueError):
    def __init__(self, mismatch: str) -> None:
        super().__init__(mismatch)
        self.mismatch = mismatch


def _require_frozen_run_settings(settings: object) -> None:
    if not isinstance(settings, RunSettings):
        raise ValueError("scripted search settings are frozen at methodology schema v2")
    actual = {
        "schema_version": settings.schema_version,
        "warmup_runs": settings.warmup_runs,
        "repeats": settings.repeats,
        "bootstrap_samples": settings.bootstrap_samples,
        "inter_run_spacing_seconds": settings.inter_run_spacing_seconds,
    }
    if not strict_json_equal(actual, v2_run_settings_json()):
        raise ValueError("scripted search settings are frozen at methodology schema v2")


class TrustedSearchOrchestrator:
    """Own policy transition, execution, receipts, and final evidence."""

    def __init__(
        self,
        output_root: Path,
        criteria_path: Path,
        settings: RunSettings,
        *,
        capability_snapshot: Mapping[str, object] | None = None,
        runner: Callable[..., RunResult] = run_to_bundle,
        _host_state_sink: object | None = None,
    ) -> None:
        _require_frozen_run_settings(settings)
        self._output_root = Path(output_root).resolve(strict=False)
        self._criteria_path = Path(criteria_path).resolve(strict=False)
        self._settings = settings
        self._capability_snapshot = capability_snapshot
        self._runner = runner
        self._host_state_sink = _host_state_sink

    def optimize(self, workload_name: WorkloadName) -> SearchRunResult:
        """Run the fixed optimizer and seal its authentic child receipts as evidence."""
        if not isinstance(workload_name, WorkloadName):
            raise ValueError("workload_name must be a WorkloadName")
        if self._host_state_sink is None:
            with prepare_host_state_output_root(self._output_root) as sink:
                return self._optimize(workload_name, sink)
        return self._optimize(workload_name, self._host_state_sink)

    def _optimize(
        self,
        workload_name: WorkloadName,
        host_state_sink: object,
    ) -> SearchRunResult:
        parent = self._output_root / _parent_name(workload_name)
        (parent / "runs").mkdir(parents=True, exist_ok=False)
        runtime = CatalogRuntime(
            parent / "runs",
            self._criteria_path,
            self._settings,
            _host_state_sink=host_state_sink,  # type: ignore[arg-type]
            capability_snapshot=self._capability_snapshot,
            runner=self._runner,
        )
        policy = scripted_policy(workload_name)
        observations: list[AgentObservation] = []
        steps: list[SearchStep] = []
        while (proposal := next_proposal(policy, tuple(observations))) is not None:
            result = runtime.run(proposal)
            steps.append(SearchStep(proposal, result))
            observations.append(AgentObservation(result.verdict, result.driving_finding))
        transcript = SearchTranscript(workload_name, tuple(steps))
        finalize_search_tree(parent, transcript, runtime.receipts)
        return SearchRunResult(parent, transcript)


def run_scripted_search(
    workload_name: WorkloadName,
    output_root: Path,
    criteria_path: Path,
    settings: RunSettings | None = None,
) -> SearchRunResult:
    """Execute one real trusted scripted search with frozen settings by default."""
    selected_settings = (
        RunSettings(schema_version=2, repeats=64) if settings is None else settings
    )
    _require_frozen_run_settings(selected_settings)
    return TrustedSearchOrchestrator(
        output_root,
        criteria_path,
        selected_settings,
    ).optimize(workload_name)


def finalize_search_tree(
    parent: Path,
    transcript: SearchTranscript,
    receipts: Sequence[ExecutionReceipt],
) -> None:
    """Write parent evidence only when transcript steps equal private receipts exactly."""
    authentic_steps = tuple(SearchStep(receipt.proposal, receipt.result) for receipt in receipts)
    if authentic_steps != transcript.steps:
        raise ValueError("search transcript differs from trusted runtime receipts")
    capability_bytes = _capability_snapshot_bytes()
    (parent / "torch_capabilities.json").write_bytes(capability_bytes)
    capability_digest = hashlib.sha256(capability_bytes).hexdigest()
    write_json(parent / "catalog.json", catalog_json(capability_digest))
    write_json(parent / "scope.json", scope_json())
    transcript_payload = _transcript_json(transcript)
    write_json(parent / "transcript.json", transcript_payload)
    semantic = _verify_semantics(
        parent,
        manifest_expected=False,
        expected_transcript=transcript_payload,
    )
    if not semantic.valid:
        raise ValueError(f"search evidence is invalid before manifest: {semantic.mismatches}")
    write_manifest(parent)
    finalized = verify_search_tree(parent)
    if not finalized.valid:
        raise ValueError(f"finalized search evidence is invalid: {finalized.mismatches}")


def verify_search_tree(parent: Path) -> IntegrityResult:
    """Verify hashes plus stored catalog, transcript, child, and receipt semantics."""
    integrity = verify_manifest(parent)
    if not integrity.valid:
        return integrity
    return _verify_semantics(parent, manifest_expected=True, expected_transcript=None)


def _verify_semantics(
    parent: Path,
    *,
    manifest_expected: bool,
    expected_transcript: dict[str, object] | None,
) -> IntegrityResult:
    mismatches = _topology_mismatches(parent, manifest_expected=manifest_expected)
    try:
        if not is_scope_json(_json_object(parent / "scope.json")):
            mismatches.append("scope:invalid")
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError):
        mismatches.append("scope:invalid")
    try:
        stored_catalog = _read_stored_catalog(parent)
    except _StoredEvidenceError as error:
        mismatches.append(error.mismatch)
        return IntegrityResult(False, tuple(dict.fromkeys(mismatches)))
    try:
        transcript_payload = _json_object(parent / "transcript.json")
        if expected_transcript is not None and transcript_payload != expected_transcript:
            raise _StoredEvidenceError("transcript:invalid")
        transcript = _parse_stored_transcript(transcript_payload, stored_catalog)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        mismatch = (
            error.mismatch
            if isinstance(error, _StoredEvidenceError)
            else "transcript:invalid"
        )
        mismatches.append(mismatch)
        return IntegrityResult(False, tuple(dict.fromkeys(mismatches)))

    expected_children = {
        step.result.bundle_relative_path.split("/", maxsplit=1)[1]
        for step in transcript.steps
    }
    runs_root = parent / "runs"
    try:
        actual_children = {path.name for path in runs_root.iterdir()}
    except OSError:
        actual_children = set()
        mismatches.append("runs:missing")
    mismatches.extend(
        f"runs:missing:{name}" for name in sorted(expected_children - actual_children)
    )
    mismatches.extend(
        f"runs:unexpected:{name}" for name in sorted(actual_children - expected_children)
    )

    ordered_children = tuple(
        parent / step.result.bundle_relative_path for step in transcript.steps
    )
    try:
        validate_provenance_chain(ordered_children)
    except ProvenanceError as error:
        mismatches.append(str(error))

    for step in transcript.steps:
        child = parent / step.result.bundle_relative_path
        if not child.is_dir() or child.is_symlink():
            mismatches.append(f"child:invalid:{step.result.bundle_relative_path}")
            continue
        child_integrity = verify_manifest(child)
        mismatches.extend(f"child:{item}" for item in child_integrity.mismatches)
        try:
            verdict_bytes = (child / "verdict.json").read_bytes()
            verdict = strict_json_loads(verdict_bytes)
        except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError):
            mismatches.append("child:verdict:invalid")
            continue
        if not isinstance(verdict, dict):
            mismatches.append("child:verdict:invalid")
            continue
        if verdict.get("verdict") != step.result.verdict:
            mismatches.append("receipt:verdict")
        if verdict.get("driving_finding") != step.result.driving_finding:
            mismatches.append("receipt:driving_finding")
        try:
            config = _json_object(child / "config.json")
        except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError):
            config = {}
        if not strict_json_equal(config.get("catalog"), step.entry.metadata):
            mismatches.append("receipt:catalog_metadata")
        try:
            methodology = _json_object(child / "methodology.json")
            frozen_settings = v2_run_settings_json()
            stored_settings = {
                key: methodology.get(key) for key in frozen_settings
            }
        except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError):
            stored_settings = {}
            frozen_settings = v2_run_settings_json()
        if not strict_json_equal(stored_settings, frozen_settings):
            mismatches.append("child:methodology.settings")
        regated = evaluate_bundle(child)
        if verdict_bytes != canonical_json_bytes(regated):
            mismatches.append("child:gate_verdict")
    return IntegrityResult(not mismatches, tuple(dict.fromkeys(mismatches)))


def _topology_mismatches(parent: Path, *, manifest_expected: bool) -> list[str]:
    expected_top = {
        "catalog.json",
        "torch_capabilities.json",
        "scope.json",
        "transcript.json",
        "runs",
    }
    if manifest_expected:
        expected_top.add("manifest.json")
    try:
        actual_top = {path.name for path in parent.iterdir()}
    except OSError:
        return ["parent:missing"]
    mismatches = [
        f"parent:missing:{name}" for name in sorted(expected_top - actual_top)
    ]
    mismatches.extend(
        f"parent:unexpected:{name}" for name in sorted(actual_top - expected_top)
    )
    return mismatches


def _read_stored_catalog(parent: Path) -> _StoredCatalog:
    try:
        capability_bytes = (parent / "torch_capabilities.json").read_bytes()
        capability_digest = hashlib.sha256(capability_bytes).hexdigest()
        capabilities = strict_json_loads(capability_bytes)
        payload = _json_object(parent / "catalog.json")
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        raise _StoredEvidenceError("catalog:invalid") from error
    if payload.keys() != {"schema_version", "entries", "torch_capabilities_sha256"}:
        raise _StoredEvidenceError("catalog:invalid")
    if type(payload.get("schema_version")) is not int or payload["schema_version"] != 1:
        raise _StoredEvidenceError("catalog:invalid")
    if payload.get("torch_capabilities_sha256") != capability_digest:
        raise _StoredEvidenceError("catalog:capability_digest")
    try:
        backends, modes, options = _stored_capability_names(capabilities)
        entries = _stored_catalog_entries(
            payload.get("entries"),
            backends=backends,
            modes=modes,
            options=options,
        )
    except ValueError as error:
        raise _StoredEvidenceError("catalog:invalid") from error
    try:
        _validate_v1_search_candidates(entries)
    except ValueError as error:
        raise _StoredEvidenceError("catalog:action") from error
    by_id = {entry.catalog_id: entry for entry in entries}
    if len(by_id) != len(entries):
        raise _StoredEvidenceError("catalog:invalid")
    if any(
        not any(entry.workload_name == workload for entry in entries)
        for workload in _WORKLOAD_NAMES
    ):
        raise _StoredEvidenceError("catalog:invalid")
    return _StoredCatalog(entries, by_id)


def _validate_v1_search_candidates(
    entries: tuple[_StoredCatalogEntry, ...],
) -> None:
    for workload, contracts in V1_SEARCH_CANDIDATES_BY_WORKLOAD.items():
        candidates = tuple(
            entry
            for entry in entries
            if entry.role == "candidate" and entry.workload_name == workload
        )
        if len(candidates) != len(contracts):
            raise ValueError("invalid candidate count")
        for entry, contract in zip(candidates, contracts, strict=True):
            if entry.rationale != contract.rationale or not strict_json_equal(
                entry.metadata,
                contract.action.metadata_json(),
            ):
                raise ValueError("invalid frozen candidate")


def _stored_capability_names(
    value: object,
) -> tuple[set[str], set[str], set[str]]:
    expected_keys = {
        "schema_version",
        "torch_version",
        "compiler_backends",
        "inductor_modes",
        "inductor_options",
    }
    if not isinstance(value, dict) or value.keys() != expected_keys:
        raise ValueError("invalid capability snapshot")
    if type(value.get("schema_version")) is not int or value["schema_version"] != 1:
        raise ValueError("invalid capability snapshot")
    if not isinstance(value.get("torch_version"), str) or not value["torch_version"]:
        raise ValueError("invalid capability snapshot")
    return (
        _stored_string_set(value.get("compiler_backends")),
        _stored_string_set(value.get("inductor_modes")),
        _stored_string_set(value.get("inductor_options")),
    )


def _stored_string_set(value: object) -> set[str]:
    if (
        not isinstance(value, list)
        or not value
        or any(not isinstance(item, str) or not item for item in value)
        or len(value) != len(set(value))
    ):
        raise ValueError("invalid capability names")
    return set(value)


def _stored_catalog_entries(
    value: object,
    *,
    backends: set[str],
    modes: set[str],
    options: set[str],
) -> tuple[_StoredCatalogEntry, ...]:
    if not isinstance(value, list) or not value:
        raise ValueError("invalid catalog entries")
    entries: list[_StoredCatalogEntry] = []
    expected_keys = _CATALOG_METADATA_KEYS | {
        "rationale",
        "expected_verdict",
        "expected_driving_finding",
    }
    for raw_entry in value:
        if not isinstance(raw_entry, dict) or raw_entry.keys() != expected_keys:
            raise ValueError("invalid catalog entry")
        workload = raw_entry.get("workload_name")
        if workload not in _WORKLOAD_NAMES:
            raise ValueError("invalid catalog workload")
        metadata = {key: raw_entry[key] for key in _CATALOG_METADATA_KEYS}
        effective = validate_catalog_metadata(metadata, workload)
        _validate_stored_capability_references(
            metadata,
            effective,
            backends=backends,
            modes=modes,
            options=options,
        )
        rationale = raw_entry.get("rationale")
        if (
            not isinstance(rationale, str)
            or not rationale
            or rationale != rationale.strip()
            or "\n" in rationale
            or "\r" in rationale
        ):
            raise ValueError("invalid catalog rationale")
        role = raw_entry.get("role")
        expected_verdict = raw_entry.get("expected_verdict")
        expected_finding = raw_entry.get("expected_driving_finding")
        if role == "candidate":
            if expected_verdict is not None or expected_finding is not None:
                raise ValueError("candidate has a demo expectation")
        elif not (
            isinstance(expected_verdict, str)
            and expected_verdict
            and isinstance(expected_finding, str)
            and expected_finding
        ):
            raise ValueError("demo entry lacks an expectation")
        catalog_id = raw_entry.get("entry_id")
        assert isinstance(catalog_id, str)
        assert isinstance(role, str)
        entries.append(
            _StoredCatalogEntry(
                catalog_id,
                role,
                workload,
                rationale,
                metadata,
            )
        )
    return tuple(entries)


def _validate_stored_capability_references(
    metadata: dict[str, object],
    effective: dict[str, object],
    *,
    backends: set[str],
    modes: set[str],
    options: set[str],
) -> None:
    backend = effective["backend"]
    if backend not in backends:
        raise ValueError("catalog backend is unavailable")
    requested = metadata["requested"]
    assert isinstance(requested, dict)
    requested_mode = requested["mode"]
    effective_mode = effective["mode"]
    referenced_mode = "default" if requested_mode == "default" else effective_mode
    if referenced_mode not in modes:
        raise ValueError("catalog mode is unavailable")
    requested_options = requested["options"]
    if isinstance(requested_options, dict) and not set(requested_options) <= options:
        raise ValueError("catalog option is unavailable")


def _parse_stored_transcript(
    payload: dict[str, object],
    catalog: _StoredCatalog,
) -> _StoredTranscript:
    if payload.keys() != {"schema_version", "workload_name", "steps"}:
        raise _StoredEvidenceError("transcript:invalid")
    if type(payload.get("schema_version")) is not int or payload["schema_version"] != 1:
        raise _StoredEvidenceError("transcript:invalid")
    workload = payload.get("workload_name")
    if workload not in _WORKLOAD_NAMES:
        raise _StoredEvidenceError("transcript:invalid")
    raw_steps = payload.get("steps")
    if not isinstance(raw_steps, list) or not raw_steps:
        raise _StoredEvidenceError("transcript:invalid")
    candidates = catalog.candidates(workload)
    if not candidates or len(raw_steps) > len(candidates):
        raise _StoredEvidenceError("transcript:catalog_order")
    steps: list[_StoredStep] = []
    paths: list[str] = []
    for index, raw_step in enumerate(raw_steps):
        step = _parse_stored_step(raw_step, workload, catalog)
        if step.entry.role != "candidate":
            raise _StoredEvidenceError("transcript:catalog_role")
        if step.entry is not candidates[index]:
            raise _StoredEvidenceError("transcript:catalog_order")
        steps.append(step)
        paths.append(step.result.bundle_relative_path)
        if step.result.verdict == "PROVEN" and index != len(raw_steps) - 1:
            raise _StoredEvidenceError("transcript:continued_after_proven")
    if len(paths) != len(set(paths)):
        raise _StoredEvidenceError("transcript:duplicate_bundle_path")
    if steps[-1].result.verdict != "PROVEN" and len(steps) != len(candidates):
        raise _StoredEvidenceError("transcript:stopped_early")
    return _StoredTranscript(workload, tuple(steps))


def _parse_stored_step(
    value: object,
    workload: str,
    catalog: _StoredCatalog,
) -> _StoredStep:
    if not isinstance(value, dict) or value.keys() != {"proposal", "result"}:
        raise _StoredEvidenceError("transcript:invalid")
    proposal = value.get("proposal")
    result = value.get("result")
    if not isinstance(proposal, dict) or proposal.keys() != {
        "catalog_id",
        "workload_name",
        "rationale",
    }:
        raise _StoredEvidenceError("transcript:invalid")
    if not isinstance(result, dict) or result.keys() != {
        "bundle_relative_path",
        "verdict",
        "driving_finding",
    }:
        raise _StoredEvidenceError("transcript:invalid")
    catalog_id = proposal.get("catalog_id")
    if not isinstance(catalog_id, str) or catalog_id not in catalog.by_id:
        raise _StoredEvidenceError("transcript:catalog_id")
    entry = catalog.by_id[catalog_id]
    if proposal.get("workload_name") != workload or entry.workload_name != workload:
        raise _StoredEvidenceError("transcript:catalog_workload")
    if proposal.get("rationale") != entry.rationale:
        raise _StoredEvidenceError("transcript:catalog_rationale")
    try:
        read_only = ReadOnlyRunResult(
            result.get("bundle_relative_path"),  # type: ignore[arg-type]
            result.get("verdict"),  # type: ignore[arg-type]
            result.get("driving_finding"),  # type: ignore[arg-type]
        )
    except (TypeError, ValueError) as error:
        raise _StoredEvidenceError("transcript:invalid") from error
    if read_only.verdict not in _VERDICTS:
        raise _StoredEvidenceError("transcript:invalid")
    return _StoredStep(
        entry,
        _StoredResult(
            read_only.bundle_relative_path,
            read_only.verdict,
            read_only.driving_finding,
        ),
    )


def _transcript_json(transcript: SearchTranscript) -> dict[str, object]:
    return {
        "schema_version": 1,
        "workload_name": transcript.workload_name.value,
        "steps": [
            {
                "proposal": {
                    "catalog_id": step.proposal.catalog_id,
                    "workload_name": step.proposal.workload_name.value,
                    "rationale": step.proposal.rationale,
                },
                "result": {
                    "bundle_relative_path": step.result.bundle_relative_path,
                    "verdict": step.result.verdict,
                    "driving_finding": step.result.driving_finding,
                },
            }
            for step in transcript.steps
        ],
    }


def _json_object(path: Path) -> dict[str, object]:
    loaded = strict_json_loads(path.read_bytes())
    if not isinstance(loaded, dict):
        raise ValueError("expected JSON object")
    return loaded


def _parent_name(workload: WorkloadName) -> str:
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S.%fZ")
    return f"agent-search-{timestamp}-{workload.value}"


def _capability_snapshot_bytes() -> bytes:
    return Path(__file__).with_name("torch_capabilities.json").read_bytes()


__all__ = [
    "SearchRunResult",
    "SearchStep",
    "SearchTranscript",
    "TrustedSearchOrchestrator",
    "finalize_search_tree",
    "run_scripted_search",
    "verify_search_tree",
]

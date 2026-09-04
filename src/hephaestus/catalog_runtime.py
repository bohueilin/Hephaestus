"""Trusted resolution and harness execution outside the optimizer authority boundary."""

from __future__ import annotations

import hashlib
import secrets
import sys
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from hephaestus.bundle import strict_json_loads
from hephaestus.catalog import (
    CatalogRole,
    Proposal,
    ReadOnlyRunResult,
    catalog_metadata,
    get_catalog_entry,
    resolve_action,
    validate_catalog_capabilities,
)
from hephaestus.host_state import HostStateCapture, capture_operation
from hephaestus.input_plan import build_bucketed_input_plan, build_identity_input_plan
from hephaestus.measure import RunResult, RunSettings, run_to_bundle
from hephaestus.provenance import ProvenancePredecessor, RunProvenance
from hephaestus.torchbind import CompileRequest
from hephaestus.workloads import get_workload


class _Runner(Protocol):
    def __call__(
        self,
        workload_name: str,
        request: CompileRequest,
        output_root: Path,
        criteria_path: Path,
        settings: RunSettings,
        *,
        input_plan: object = None,
        catalog_metadata: Mapping[str, object] | None = None,
        run_provenance: RunProvenance | None = None,
    ) -> RunResult: ...


class _HostStateSink(Protocol):
    def append(self, capture: HostStateCapture, bundle_path: Path) -> None: ...


@dataclass(frozen=True, slots=True)
class ExecutionReceipt:
    """A trusted proposal/result pair retained privately by search orchestration."""

    proposal: Proposal
    result: ReadOnlyRunResult


class CatalogRuntime:
    """Re-resolve proposals, execute real harness actions, and project narrow receipts."""

    def __init__(
        self,
        output_root: Path,
        criteria_path: Path,
        settings: RunSettings,
        *,
        _host_state_sink: _HostStateSink,
        capability_snapshot: Mapping[str, object] | None = None,
        runner: _Runner | Callable[..., RunResult] = run_to_bundle,
        _worker_relative_root: bool = False,
    ) -> None:
        runs_root = Path(output_root)
        if runs_root.name != "runs":
            raise ValueError("catalog runtime output root must be a directory named 'runs'")
        self._runs_root = runs_root if _worker_relative_root else runs_root.absolute()
        self._parent_root = self._runs_root.parent
        self._criteria_path = Path(criteria_path).resolve(strict=False)
        self._settings = settings
        self._runner = runner
        self._host_state_sink = _host_state_sink
        self._worker_relative_root = _worker_relative_root
        snapshot = (
            _load_capability_snapshot()
            if capability_snapshot is None
            else capability_snapshot
        )
        validate_catalog_capabilities(snapshot)
        self._receipts: list[ExecutionReceipt] = []
        self._orchestration_id = secrets.token_hex(32)

    @property
    def receipts(self) -> tuple[ExecutionReceipt, ...]:
        """Return an immutable trusted view for parent evidence finalization."""
        return tuple(self._receipts)

    @property
    def parent_path(self) -> Path:
        """Expose the trusted parent root only to trusted orchestration."""
        return self._parent_root

    def run(self, proposal: Proposal) -> ReadOnlyRunResult:
        """Execute only ordinary optimizer candidates through trusted orchestration."""
        return self._execute(proposal, allowed_roles=(CatalogRole.CANDIDATE,))

    def run_demo(self, proposal: Proposal) -> ReadOnlyRunResult:
        """Execute only planted/control entries through an explicit trusted demo path."""
        return self._execute(
            proposal,
            allowed_roles=(CatalogRole.PLANTED, CatalogRole.CLEAN_CONTROL),
        )

    def _execute(
        self,
        proposal: Proposal,
        *,
        allowed_roles: tuple[CatalogRole, ...],
    ) -> ReadOnlyRunResult:
        if not isinstance(proposal, Proposal):
            raise ValueError("proposal must be a Proposal")
        canonical_proposal = Proposal(
            proposal.catalog_id,
            proposal.workload_name,
            proposal.rationale,
        )
        entry = get_catalog_entry(canonical_proposal.catalog_id)
        if entry.role not in allowed_roles:
            if allowed_roles == (CatalogRole.CANDIDATE,):
                raise ValueError("optimizer execution admits only candidate catalog entries")
            raise ValueError("trusted demo path admits only planted or clean-control entries")
        action = resolve_action(entry)
        issued_provenance = self._next_provenance()
        request = CompileRequest(
            backend=action.backend,
            mode=action.mode,
            dynamic=action.dynamic,
            fullgraph=action.fullgraph,
            options=action.options,
            disable=action.disable,
        )
        workload = get_workload(entry.workload.value)
        if action.input_plan_strategy == "bucketed":
            plan = build_bucketed_input_plan(
                workload,
                boundaries=action.bucket_boundaries or (),
            )
        else:
            plan = build_identity_input_plan(workload, action.input_plan_strategy)

        run_result, capture = capture_operation(
            lambda: self._runner(
                entry.workload.value,
                request,
                self._runs_root,
                self._criteria_path,
                self._settings,
                input_plan=plan,
                catalog_metadata=catalog_metadata(entry),
                run_provenance=issued_provenance,
            ),
            issued_provenance,
        )
        result = self._project_result(run_result)
        self._receipts.append(ExecutionReceipt(canonical_proposal, result))
        try:
            self._host_state_sink.append(capture, run_result.bundle_path)
        except Exception as error:
            print(f"host-state ledger error: {error}", file=sys.stderr)
        return result

    def _next_provenance(self) -> RunProvenance:
        sequence_index = len(self._receipts)
        predecessor: ProvenancePredecessor | None = None
        if self._receipts:
            prior = self._receipts[-1].result
            prior_bundle = self._parent_root / prior.bundle_relative_path
            try:
                manifest_bytes = (prior_bundle / "manifest.json").read_bytes()
                prior_provenance = strict_json_loads(
                    (prior_bundle / "run_provenance.json").read_bytes()
                )
            except (OSError, UnicodeDecodeError, ValueError) as error:
                raise RuntimeError("prior run lacks finalized provenance evidence") from error
            if not isinstance(prior_provenance, dict) or not isinstance(
                prior_provenance.get("run_id"), str
            ):
                raise RuntimeError("prior run lacks finalized provenance evidence")
            predecessor = ProvenancePredecessor(
                prior_provenance["run_id"],
                hashlib.sha256(manifest_bytes).hexdigest(),
            )
        return RunProvenance(
            orchestration_id=self._orchestration_id,
            run_id=secrets.token_hex(32),
            sequence_index=sequence_index,
            predecessor=predecessor,
        )

    def bundle_path_for(self, result: ReadOnlyRunResult) -> Path:
        """Resolve an authentic receipt for trusted demo/search orchestration only."""
        if not isinstance(result, ReadOnlyRunResult) or not any(
            receipt.result == result for receipt in self._receipts
        ):
            raise ValueError("result is not an authentic runtime receipt")
        return self._parent_root / result.bundle_relative_path

    def _project_result(self, result: RunResult) -> ReadOnlyRunResult:
        if not isinstance(result, RunResult) or not isinstance(result.bundle_path, Path):
            raise ValueError("runner returned an invalid run result")
        bundle_path = result.bundle_path
        expected_absolute = not self._worker_relative_root
        if bundle_path.is_absolute() is not expected_absolute or "\\" in bundle_path.name:
            raise ValueError("runner bundle path is not canonical")
        try:
            relative = bundle_path.relative_to(self._parent_root)
        except ValueError as error:
            raise ValueError("runner bundle path escaped the trusted parent") from error
        if (
            len(relative.parts) != 2
            or relative.parts[0] != "runs"
            or relative.parts[1] in {"", ".", ".."}
            or bundle_path.parent != self._runs_root
            or not bundle_path.is_dir()
            or bundle_path.is_symlink()
        ):
            raise ValueError("runner bundle path is not a direct child of runs")
        relative_path = relative.as_posix()
        if any(
            receipt.result.bundle_relative_path == relative_path
            for receipt in self._receipts
        ):
            raise ValueError("runner returned a duplicate child bundle path")
        driving_finding = result.summary.get("driving_finding")
        summary_verdict = result.summary.get("verdict")
        if not isinstance(driving_finding, str) or not driving_finding:
            raise ValueError("runner summary lacks a scalar driving finding")
        if summary_verdict != result.verdict:
            raise ValueError("runner summary verdict disagrees with run result")
        return ReadOnlyRunResult(relative_path, result.verdict, driving_finding)


def _load_capability_snapshot() -> Mapping[str, object]:
    path = Path(__file__).with_name("torch_capabilities.json")
    loaded = strict_json_loads(path.read_bytes())
    if not isinstance(loaded, dict):
        raise ValueError("invalid packaged capability snapshot")
    return loaded


__all__ = ["CatalogRuntime", "ExecutionReceipt"]

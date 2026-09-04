"""Trusted single-run orchestration behind the closed catalog boundary."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path

from hephaestus.catalog import CatalogRole, Proposal, WorkloadName, get_catalog_entry
from hephaestus.catalog_runtime import CatalogRuntime
from hephaestus.durability import prepare_host_state_output_root
from hephaestus.measure import RunResult, RunSettings, run_to_bundle


@dataclass(frozen=True, slots=True)
class SingleRunResult:
    """Immutable CLI-facing projection of one authentic catalog execution."""

    bundle_path: Path
    verdict: str
    driving_finding: str


def run_catalog_action(
    workload_name: WorkloadName,
    catalog_id: str,
    output_root: Path,
    criteria_path: Path,
    *,
    capability_snapshot: Mapping[str, object] | None = None,
    runner: Callable[..., RunResult] = run_to_bundle,
) -> SingleRunResult:
    """Resolve one canonical entry, execute it, and expose only its real receipt."""
    if not isinstance(workload_name, WorkloadName):
        raise ValueError("workload_name must be a WorkloadName")
    entry = get_catalog_entry(catalog_id)
    if entry.workload is not workload_name:
        raise ValueError("catalog entry does not match the requested workload")

    root = Path(output_root).resolve(strict=False)
    with prepare_host_state_output_root(root) as host_state_sink:
        runtime = CatalogRuntime(
            root / "runs",
            criteria_path,
            RunSettings(schema_version=2, repeats=64),
            _host_state_sink=host_state_sink,
            capability_snapshot=capability_snapshot,
            runner=runner,
        )
        proposal = Proposal(entry.catalog_id, entry.workload, entry.rationale)
        if entry.role is CatalogRole.CANDIDATE:
            receipt = runtime.run(proposal)
        else:
            receipt = runtime.run_demo(proposal)
        return SingleRunResult(
            runtime.bundle_path_for(receipt),
            receipt.verdict,
            receipt.driving_finding,
        )


__all__ = ["SingleRunResult", "run_catalog_action"]

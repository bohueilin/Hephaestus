"""Torch-free trusted workflow for read-only offline normal-bundle gating."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from hephaestus.gate import evaluate_bundle


@dataclass(frozen=True, slots=True)
class OfflineGateResult:
    """Complete deterministic verdict plus its unchanged stored evidence path."""

    bundle_path: Path
    verdict: str
    driving_finding: str
    complete_verdict: dict[str, object]


def run_offline_gate(bundle_path: Path) -> OfflineGateResult:
    """Evaluate stored evidence without creating, mutating, or rerunning anything."""
    path = Path(bundle_path)
    verdict = evaluate_bundle(path)
    verdict_name = verdict.get("verdict")
    driving_finding = verdict.get("driving_finding")
    if not isinstance(verdict_name, str) or not isinstance(driving_finding, str):
        raise RuntimeError("offline gate returned a malformed verdict")
    return OfflineGateResult(path, verdict_name, driving_finding, verdict)


__all__ = ["OfflineGateResult", "run_offline_gate"]

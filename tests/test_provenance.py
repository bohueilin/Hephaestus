from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from types import MappingProxyType

import pytest

import hephaestus.measure as measure_module
from hephaestus.bundle import write_json, write_manifest
from hephaestus.catalog import Proposal, WorkloadName, get_catalog_entry
from hephaestus.catalog_runtime import CatalogRuntime
from hephaestus.measure import RunResult, RunSettings, run_to_bundle

ROOT = Path(__file__).parents[1]
CRITERIA = ROOT / "gates" / "default.yaml"
CAPABILITIES = json.loads((ROOT / "src/hephaestus/torch_capabilities.json").read_bytes())


class _HostSink:
    def append(self, capture: object, bundle_path: Path) -> None:
        del capture, bundle_path


class _ProvenanceCaptureRunner:
    def __init__(self) -> None:
        self.provenance: list[object] = []

    def __call__(
        self,
        workload_name: str,
        request: object,
        output_root: Path,
        criteria_path: Path,
        settings: RunSettings,
        *,
        input_plan: object = None,
        catalog_metadata: Mapping[str, object] | None = None,
        run_provenance: object = None,
    ) -> RunResult:
        del workload_name, request, criteria_path, settings, input_plan, catalog_metadata
        self.provenance.append(run_provenance)
        child = output_root / f"child-{len(self.provenance)}"
        child.mkdir(parents=True)
        if run_provenance is not None:
            payload = run_provenance.as_json()  # type: ignore[attr-defined]
            write_json(child / "run_provenance.json", payload)
        write_manifest(child)
        summary = MappingProxyType(
            {"verdict": "NOT_PROVEN", "driving_finding": "perf.speedup_proven"}
        )
        return RunResult(child, "NOT_PROVEN", summary)


def test_catalog_runtime_issues_distinct_chained_unsigned_run_provenance(
    tmp_path: Path,
) -> None:
    """Every trusted run must carry an exact session/order chain into its child bundle."""
    runner = _ProvenanceCaptureRunner()
    runtime = CatalogRuntime(
        tmp_path / "runs",
        CRITERIA,
        RunSettings(),
        _host_state_sink=_HostSink(),
        capability_snapshot=CAPABILITIES,
        runner=runner,
    )
    entry = get_catalog_entry("candidate-mlp-default")
    proposal = Proposal(entry.catalog_id, WorkloadName.MLP_STACK, entry.rationale)

    first = runtime.run(proposal)
    second = runtime.run(proposal)

    assert first.bundle_relative_path == "runs/child-1"
    assert second.bundle_relative_path == "runs/child-2"
    assert len(runner.provenance) == 2
    first_provenance, second_provenance = runner.provenance
    first_json = first_provenance.as_json()  # type: ignore[attr-defined]
    second_json = second_provenance.as_json()  # type: ignore[attr-defined]
    assert set(first_json) == {
        "schema_version",
        "orchestration_id",
        "run_id",
        "sequence_index",
        "predecessor",
    }
    assert first_json["schema_version"] == 1
    assert first_json["sequence_index"] == 0
    assert first_json["predecessor"] is None
    assert second_json["orchestration_id"] == first_json["orchestration_id"]
    assert second_json["run_id"] != first_json["run_id"]
    assert second_json["sequence_index"] == 1
    assert second_json["predecessor"] == {
        "run_id": first_json["run_id"],
        "manifest_sha256": hashlib.sha256(
            (tmp_path / "runs/child-1/manifest.json").read_bytes()
        ).hexdigest(),
    }
    for identifier in (
        first_json["orchestration_id"],
        first_json["run_id"],
        second_json["run_id"],
    ):
        assert isinstance(identifier, str)
        assert len(identifier) == 64
        assert all(character in "0123456789abcdef" for character in identifier)


def test_low_level_bundle_producer_rejects_missing_trusted_provenance_before_measurement(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A catalog-shaped caller cannot bypass CatalogRuntime and mint gate-eligible evidence."""
    measured = False

    def forbidden_measure(*args: object, **kwargs: object) -> object:
        nonlocal measured
        measured = True
        raise AssertionError("measurement must not start without trusted provenance")

    monkeypatch.setattr(measure_module, "measure", forbidden_measure)
    entry = get_catalog_entry("candidate-mlp-default")

    with pytest.raises(ValueError, match="provenance"):
        run_to_bundle(
            "mlp_stack",
            object(),  # type: ignore[arg-type]
            tmp_path,
            CRITERIA,
            RunSettings(),
            catalog_metadata={"entry_id": entry.catalog_id},
        )

    assert measured is False
    assert list(tmp_path.iterdir()) == []

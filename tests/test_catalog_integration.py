from __future__ import annotations

import json
import platform
from pathlib import Path

import pytest

from hephaestus.catalog import (
    CatalogRole,
    Proposal,
    WorkloadName,
    get_catalog_entry,
)
from hephaestus.catalog_runtime import CatalogRuntime
from hephaestus.durability import prepare_host_state_output_root
from hephaestus.measure import RunSettings
from hephaestus.search import TrustedSearchOrchestrator, verify_search_tree

pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(
        platform.system() != "Darwin" or platform.machine() != "arm64",
        reason="Apple-arm64 calibrated catalog acceptance",
    ),
]


@pytest.fixture
def _demo_host_state_sink(tmp_path: Path):
    with prepare_host_state_output_root(tmp_path / "demo") as sink:
        yield sink


def test_real_catalog_runs_exact_three_plants_and_clean_control_with_frozen_settings(
    tmp_path: Path,
    _demo_host_state_sink: object,
) -> None:
    """The closed trusted path must retain all four calibrated outcomes and distinct evidence."""
    settings = RunSettings(schema_version=2, repeats=64)
    assert settings.schema_version == 2
    assert settings.repeats == 64
    runtime = CatalogRuntime(
        tmp_path / "demo" / "runs",
        _criteria_path(),
        settings,
        _host_state_sink=_demo_host_state_sink,  # type: ignore[arg-type]
    )
    expected = (
        ("planted-eager-fallback", "NOT_PROVEN", "perf.speedup_proven"),
        ("planted-static-shape-storm", "NOT_PROVEN", "graph.recompile_bound"),
        ("planted-graph-break-exposure", "CONDITIONAL", "graph.no_breaks"),
        ("clean-control-mlp", "PROVEN", "all_criteria_passed"),
    )

    observed: list[tuple[str, str, str]] = []
    bundles: dict[str, Path] = {}
    for catalog_id, _, _ in expected:
        entry = get_catalog_entry(catalog_id)
        assert entry.role in {CatalogRole.PLANTED, CatalogRole.CLEAN_CONTROL}
        result = runtime.run_demo(Proposal(catalog_id, entry.workload, entry.rationale))
        observed.append((catalog_id, result.verdict, result.driving_finding))
        bundles[catalog_id] = runtime.bundle_path_for(result)

    assert observed == list(expected)
    fallback_config = _json(bundles["planted-eager-fallback"], "config.json")
    assert fallback_config["backend"] == "inductor"
    assert fallback_config["disable"] is True
    assert fallback_config["catalog"]["role"] == "planted"  # type: ignore[index]
    storm = _json(bundles["planted-static-shape-storm"], "dynamo_report.json")
    assert len(storm["recompiles"]) == 3
    assert all(record["trigger"] for record in storm["recompiles"])  # type: ignore[index]
    graph_break = _json(bundles["planted-graph-break-exposure"], "dynamo_report.json")
    assert len(graph_break["graph_breaks"]) == 1
    assert graph_break["graph_breaks"][0]["reason"]  # type: ignore[index]
    clean = _json(bundles["clean-control-mlp"], "dynamo_report.json")
    assert clean["graph_breaks"] == []
    assert clean["recompiles"] == []


def test_real_scripted_search_preserves_refusal_then_finds_proven_candidate(
    tmp_path: Path,
) -> None:
    """The authentic dynamic search must keep a plausible refusal before accepting PROVEN."""
    search = TrustedSearchOrchestrator(
        tmp_path,
        _criteria_path(),
        RunSettings(schema_version=2, repeats=64),
    ).optimize(WorkloadName.DYNAMIC_BATCH_TEXT)

    assert [step.proposal.catalog_id for step in search.transcript.steps] == [
        "candidate-dynamic-static",
        "candidate-dynamic-true",
    ]
    assert [
        (step.result.verdict, step.result.driving_finding)
        for step in search.transcript.steps
    ] == [
        ("NOT_PROVEN", "graph.recompile_bound"),
        ("PROVEN", "all_criteria_passed"),
    ]
    assert search.final_result == search.transcript.steps[-1].result
    assert verify_search_tree(search.parent_path).valid is True
    first_bundle = search.parent_path / search.transcript.steps[0].result.bundle_relative_path
    first_dynamo = _json(first_bundle, "dynamo_report.json")
    assert len(first_dynamo["recompiles"]) == 3
    assert all(record["trigger"] for record in first_dynamo["recompiles"])  # type: ignore[index]


def _criteria_path() -> Path:
    return Path(__file__).parents[1] / "gates" / "default.yaml"


def _json(bundle: Path, filename: str) -> dict[str, object]:
    loaded = json.loads((bundle / filename).read_bytes())
    assert isinstance(loaded, dict)
    return loaded

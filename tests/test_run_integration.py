from __future__ import annotations

import getpass
import hashlib
import inspect
import json
import platform
import socket
import sys
import tempfile
from datetime import datetime
from pathlib import Path

import pytest
import torch

from hephaestus.bundle import canonical_json_bytes, verify_manifest
from hephaestus.catalog import CatalogRole, Proposal, get_catalog_entry
from hephaestus.catalog_runtime import CatalogRuntime
from hephaestus.durability import prepare_host_state_output_root
from hephaestus.gate import evaluate_bundle
from hephaestus.measure import RunSettings, measure
from hephaestus.torchbind import CompileRequest
from hephaestus.workloads.dynamic_batch_text import SPEC as DYNAMIC_BATCH_TEXT
from hephaestus.workloads.mlp_stack import SPEC as MLP_STACK

pytestmark = pytest.mark.slow

INDUCTOR_REQUEST = CompileRequest(
    backend="inductor",
    mode="default",
    dynamic=False,
    fullgraph=False,
    options=None,
    disable=False,
)
FULLGRAPH_REQUEST = CompileRequest(
    backend="inductor",
    mode=None,
    dynamic=False,
    fullgraph=True,
    options=None,
    disable=False,
)
EAGER_AUTO_DYNAMIC_REQUEST = CompileRequest(
    backend="eager",
    mode=None,
    dynamic=None,
    fullgraph=False,
    options=None,
    disable=False,
)
EXPECTED_FILES = {
    "env.json",
    "workload.digest",
    "config.json",
    "input_plan.json",
    "timings.json",
    "dynamo_report.json",
    "accuracy.json",
    "methodology.json",
    "gate_criteria.yaml",
    "run_provenance.json",
    "verdict.json",
    "manifest.json",
}


def test_real_compile_run_finalizes_an_offline_regatable_bundle(tmp_path: Path) -> None:
    """A live run must preserve enough immutable raw evidence to reproduce its verdict."""
    criteria = Path(__file__).parents[1] / "gates" / "default.yaml"
    settings = RunSettings(schema_version=2, repeats=64)
    result, bundle = _run_catalog_bundle(
        tmp_path,
        "candidate-mlp-default",
        settings=settings,
    )

    assert {path.name for path in bundle.iterdir()} == EXPECTED_FILES
    assert (bundle / "gate_criteria.yaml").read_bytes() == criteria.read_bytes()
    source_path = Path(inspect.getsourcefile(MLP_STACK.make_module) or "")
    assert json.loads((bundle / "workload.digest").read_bytes()) == {
        "schema_version": 1,
        "name": "mlp_stack",
        "sha256": hashlib.sha256(source_path.read_bytes()).hexdigest(),
        "accuracy_tolerance": {
            "dtype": "torch.float32",
            "atol": 1e-5,
            "rtol": 1e-5,
        },
    }

    timings = json.loads((bundle / "timings.json").read_bytes())
    methodology = json.loads((bundle / "methodology.json").read_bytes())
    input_plan = json.loads((bundle / "input_plan.json").read_bytes())
    assert methodology["bootstrap_samples"] == 2000
    assert input_plan == {
        "schema_version": 1,
        "dynamic_strategy": "static",
        "bucket_axis": None,
        "bucket_boundaries": None,
        "bucket_overflow_rule": None,
        "original_shapes": [[48, 32]],
        "effective_shapes": [[48, 32]],
        "compile_sweep_case_indices": [0],
        "steady_state_case_index": 0,
    }
    for series in ("eager", "compiled", "aa_baseline", "aa_candidate"):
        seconds = timings[f"{series}_seconds"]
        timestamps = timings[f"{series}_timestamps_utc"]
        assert len(seconds) == len(timestamps) == settings.repeats
        assert all(value >= 0 for value in seconds)
        assert all(_is_utc_iso8601(value) for value in timestamps)
    assert timings["cold_compile_seconds"] >= 0
    assert _is_utc_iso8601(timings["cold_compile_timestamp_utc"])
    assert timings["non_primary_compile_sweep_case_indices"] == []
    assert timings["non_primary_compile_sweep_seconds"] == []
    assert timings["non_primary_compile_sweep_timestamps_utc"] == []

    accuracy = json.loads((bundle / "accuracy.json").read_bytes())
    assert [record["case_index"] for record in accuracy["cases"]] == [0]

    dynamo = json.loads((bundle / "dynamo_report.json").read_bytes())
    assert all(record["reason"] for record in dynamo["graph_breaks"])
    assert all(record["trigger"] for record in dynamo["recompiles"])
    assert dynamo["compilation_metrics"]

    assert verify_manifest(bundle).valid is True
    manifest = json.loads((bundle / "manifest.json").read_bytes())
    assert "input_plan.json" in manifest["files"]
    all_bundle_bytes = b"".join(
        path.read_bytes() for path in sorted(bundle.iterdir()) if path.is_file()
    )
    private_values = {
        getpass.getuser(),
        socket.gethostname(),
        str(Path.home()),
        str(Path.cwd()),
        sys.prefix,
        sys.base_prefix,
        str(Path(torch.__file__).resolve().parent),
        tempfile.gettempdir(),
    }
    assert all(
        value.encode() not in all_bundle_bytes for value in private_values if value
    )
    stored_verdict = json.loads((bundle / "verdict.json").read_bytes())
    first_offline = evaluate_bundle(bundle)
    second_offline = evaluate_bundle(bundle)
    assert stored_verdict == first_offline
    assert canonical_json_bytes(first_offline) == canonical_json_bytes(second_offline)
    assert (bundle / "verdict.json").read_bytes() == canonical_json_bytes(first_offline)

    assert result.verdict == stored_verdict["verdict"]
    assert not hasattr(result, "timings")
    assert not hasattr(result, "summary")


def test_bucketed_plan_is_manifested_and_semantically_offline_regatable(
    tmp_path: Path,
) -> None:
    """Genuine bucket tensors and their exact shape plan must survive finalization."""
    result, bundle = _run_catalog_bundle(
        tmp_path,
        "candidate-dynamic-bucketed",
    )

    stored_plan = _bundle_json(bundle, "input_plan.json")
    assert stored_plan["original_shapes"] == [
        [2, 24, 32],
        [1, 48, 32],
        [3, 16, 32],
        [4, 12, 32],
    ]
    assert stored_plan["effective_shapes"] == [
        [2, 24, 32],
        [2, 48, 32],
        [4, 16, 32],
        [4, 12, 32],
    ]
    assert stored_plan["compile_sweep_case_indices"] == [0, 1, 2, 3]
    assert verify_manifest(bundle).valid is True
    first = evaluate_bundle(bundle)
    second = evaluate_bundle(bundle)
    assert canonical_json_bytes(first) == canonical_json_bytes(second)
    assert first["driving_finding"] not in {
        "methodology.input_plan",
        "methodology.compile_sweep",
        "methodology.repeats",
        "methodology.timestamps",
        "accuracy.tolerance",
    }


def test_auto_dynamic_measurement_is_truthful_without_minting_a_bundle() -> None:
    """A non-catalog probe may measure auto dynamic but cannot mint trusted evidence."""
    evidence = measure(
        MLP_STACK,
        EAGER_AUTO_DYNAMIC_REQUEST,
        RunSettings(warmup_runs=0, repeats=2),
    )

    input_plan = evidence.input_plan
    assert input_plan["dynamic_strategy"] == "auto"
    assert input_plan["bucket_axis"] is None
    assert input_plan["original_shapes"] == input_plan["effective_shapes"]


@pytest.mark.skipif(
    platform.system() != "Darwin" or platform.machine() != "arm64",
    reason="Apple-arm64 calibrated compiler behavior",
)
def test_static_dynamic_text_sweep_records_three_recompiles_without_warmups() -> None:
    """Compiler-case coverage, not discarded warmups, must expose all three recompiles."""
    evidence = measure(
        DYNAMIC_BATCH_TEXT,
        FULLGRAPH_REQUEST,
        RunSettings(warmup_runs=0),
    )

    assert evidence.timings["non_primary_compile_sweep_case_indices"] == (1, 2, 3)
    assert len(evidence.dynamo_report["recompiles"]) == 3
    assert all(record["trigger"] for record in evidence.dynamo_report["recompiles"])
    assert len(evidence.accuracy["cases"]) == 4
    assert evidence.accuracy["within_tolerance"] is True


@pytest.mark.skipif(
    platform.system() != "Darwin" or platform.machine() != "arm64",
    reason="Apple-arm64 timing regression",
)
def test_default_mlp_aa_floor_clears_frozen_ten_percent_effect() -> None:
    """Default local paired measurement must remain usable on its target Apple CPU."""
    evidence = measure(
        MLP_STACK,
        INDUCTOR_REQUEST,
        RunSettings(schema_version=2, repeats=64),
    )

    assert evidence.methodology["aa_noise_floor"] < 0.10


@pytest.mark.skipif(
    platform.system() != "Darwin" or platform.machine() != "arm64",
    reason="Apple-arm64 calibrated acceptance",
)
def test_calibrated_clean_mlp_bundle_is_proven(tmp_path: Path) -> None:
    """The clean control must clear every frozen hard and soft criterion."""
    result, bundle = _run_catalog_bundle(tmp_path, "clean-control-mlp")
    verdict = _bundle_json(bundle, "verdict.json")
    methodology = _bundle_json(bundle, "methodology.json")
    accuracy = _bundle_json(bundle, "accuracy.json")
    dynamo = _bundle_json(bundle, "dynamo_report.json")
    timings = _bundle_json(bundle, "timings.json")

    assert result.verdict == verdict["verdict"] == "PROVEN"
    assert verdict["driving_finding"] == "all_criteria_passed"
    assert methodology["aa_noise_floor"] < 0.10
    assert verdict["measurements"]["speedup_lower_confidence_bound"] >= 1.10
    assert accuracy["within_tolerance"] is True
    assert dynamo["graph_breaks"] == []
    assert dynamo["recompiles"] == []
    assert timings["cold_compile_seconds"] <= 30.0


@pytest.mark.skipif(
    platform.system() != "Darwin" or platform.machine() != "arm64",
    reason="Apple-arm64 calibrated acceptance",
)
def test_calibrated_static_dynamic_text_fails_only_recompile_bound(
    tmp_path: Path,
) -> None:
    """All four static shapes must expose exactly three authentic recompiles."""
    result, bundle = _run_catalog_bundle(tmp_path, "planted-static-shape-storm")
    verdict = _bundle_json(bundle, "verdict.json")
    methodology = _bundle_json(bundle, "methodology.json")
    accuracy = _bundle_json(bundle, "accuracy.json")
    dynamo = _bundle_json(bundle, "dynamo_report.json")

    assert result.verdict == verdict["verdict"] == "NOT_PROVEN"
    assert verdict["driving_finding"] == "graph.recompile_bound"
    assert methodology["aa_noise_floor"] < 0.10
    assert accuracy["within_tolerance"] is True
    assert dynamo["graph_breaks"] == []
    assert len(dynamo["recompiles"]) == 3
    assert all(record["trigger"] for record in dynamo["recompiles"])
    assert _finding_status(verdict, "perf.speedup_proven") == "PASS"
    assert _finding_status(verdict, "perf.compile_budget") == "PASS"


@pytest.mark.skipif(
    platform.system() != "Darwin" or platform.machine() != "arm64",
    reason="Apple-arm64 calibrated acceptance",
)
def test_calibrated_graph_bait_is_conditional_on_one_reasoned_break(
    tmp_path: Path,
) -> None:
    """The non-fullgraph bait must expose one Tensor.item break and no hard failure."""
    result, bundle = _run_catalog_bundle(tmp_path, "planted-graph-break-exposure")
    verdict = _bundle_json(bundle, "verdict.json")
    methodology = _bundle_json(bundle, "methodology.json")
    accuracy = _bundle_json(bundle, "accuracy.json")
    dynamo = _bundle_json(bundle, "dynamo_report.json")

    assert result.verdict == verdict["verdict"] == "CONDITIONAL"
    assert verdict["driving_finding"] == "graph.no_breaks"
    assert methodology["aa_noise_floor"] < 0.10
    assert accuracy["within_tolerance"] is True
    assert len(dynamo["graph_breaks"]) == 1
    assert "Tensor.item" in dynamo["graph_breaks"][0]["reason"]
    assert dynamo["recompiles"] == []
    assert _finding_status(verdict, "perf.speedup_proven") == "PASS"
    assert _finding_status(verdict, "perf.compile_budget") == "PASS"


@pytest.mark.skipif(
    platform.system() != "Darwin" or platform.machine() != "arm64",
    reason="Apple-arm64 calibrated acceptance",
)
def test_calibrated_transformer_is_valid_but_speed_not_proven(tmp_path: Path) -> None:
    """The stable transformer evidence must honestly preserve its local speed miss."""
    result, bundle = _run_catalog_bundle(tmp_path, "candidate-transformer-default")
    verdict = _bundle_json(bundle, "verdict.json")
    methodology = _bundle_json(bundle, "methodology.json")
    accuracy = _bundle_json(bundle, "accuracy.json")
    dynamo = _bundle_json(bundle, "dynamo_report.json")

    assert result.verdict == verdict["verdict"] == "NOT_PROVEN"
    assert verdict["driving_finding"] == "perf.speedup_proven"
    assert methodology["valid"] is True
    assert methodology["aa_noise_floor"] < 0.10
    assert accuracy["within_tolerance"] is True
    assert dynamo["graph_breaks"] == []
    assert dynamo["recompiles"] == []
    assert _finding_status(verdict, "perf.speedup_proven") == "FAIL"
    assert _finding_status(verdict, "perf.compile_budget") == "PASS"


def _is_utc_iso8601(value: object) -> bool:
    if not isinstance(value, str) or not value.endswith("Z"):
        return False
    return datetime.fromisoformat(value.replace("Z", "+00:00")).utcoffset().total_seconds() == 0


def _run_catalog_bundle(
    output_root: Path,
    catalog_id: str,
    *,
    settings: RunSettings | None = None,
):
    criteria = Path(__file__).parents[1] / "gates" / "default.yaml"
    settings = (
        RunSettings(schema_version=2, repeats=64) if settings is None else settings
    )
    with prepare_host_state_output_root(output_root) as sink:
        runtime = CatalogRuntime(
            output_root / "runs",
            criteria,
            settings,
            _host_state_sink=sink,
        )
        entry = get_catalog_entry(catalog_id)
        proposal = Proposal(catalog_id, entry.workload, entry.rationale)
        result = (
            runtime.run(proposal)
            if entry.role is CatalogRole.CANDIDATE
            else runtime.run_demo(proposal)
        )
        bundle = runtime.bundle_path_for(result)
    assert (bundle / "gate_criteria.yaml").read_bytes() == criteria.read_bytes()
    return result, bundle


def _bundle_json(bundle_path: Path, filename: str) -> dict[str, object]:
    value = json.loads((bundle_path / filename).read_bytes())
    assert isinstance(value, dict)
    return value


def _finding_status(verdict: dict[str, object], identifier: str) -> object:
    findings = verdict["findings"]
    assert isinstance(findings, list)
    return next(
        finding["status"]
        for finding in findings
        if isinstance(finding, dict) and finding.get("id") == identifier
    )

from __future__ import annotations

import json
import shutil
from pathlib import Path
from types import MappingProxyType

import pytest

from hephaestus.aa import _evaluate_provisional_aa_bundle
from hephaestus.bundle import canonical_json_bytes, write_json, write_manifest
from hephaestus.gate import evaluate_bundle
from hephaestus.measure import RunResult, RunSettings
from tests.evidence_helpers import write_normal_child
from tests.methodology_v2_helpers import convert_normal_bundle_to_v2
from tests.test_aa_runtime import (
    _AARunner,
    _orchestrator,
    _refresh_aa_child_authority,
)


class _V2RequiredAARunner(_AARunner):
    def __init__(self) -> None:
        super().__init__()
        self.run_provenance: list[object] = []

    def __call__(self, *args: object, **kwargs: object) -> RunResult:
        settings = args[4]
        assert getattr(settings, "schema_version", None) == 2, (
            "TrustedAAOrchestrator must explicitly select methodology schema v2"
        )
        assert getattr(settings, "repeats", None) == 64
        self.run_provenance.append(kwargs["run_provenance"])
        result = super().__call__(*args, **kwargs)  # type: ignore[arg-type]
        convert_normal_bundle_to_v2(result.bundle_path)
        verdict = evaluate_bundle(result.bundle_path)
        return RunResult(
            result.bundle_path,
            str(verdict["verdict"]),
            MappingProxyType(verdict),
        )


def _v2_parent(tmp_path: Path) -> tuple[Path, _V2RequiredAARunner]:
    runner = _V2RequiredAARunner()
    result = _orchestrator(tmp_path, runner).run("mlp_stack")
    return result.parent_path, runner


def _tree_versions(parent: Path) -> tuple[list[object], list[object]]:
    parent_versions = [
        json.loads((parent / filename).read_bytes()).get("schema_version")
        for filename in (
            "aa_methodology.json",
            "aa_test.json",
            "aa_distribution.json",
        )
    ]
    aa_test = json.loads((parent / "aa_test.json").read_bytes())
    child_versions = [
        json.loads((parent / record["bundle_relative_path"] / "methodology.json").read_bytes()).get(
            "schema_version"
        )
        for record in aa_test["children"]
    ]
    return parent_versions, child_versions


def _require_complete_v2_tree(parent: Path) -> None:
    parent_versions, child_versions = _tree_versions(parent)
    assert parent_versions == [2, 2, 2]
    assert child_versions == [2, 2]


def test_schema_v2_aa_tree_declares_one_version_at_every_methodology_surface(
    tmp_path: Path,
) -> None:
    """The parent meta-contract cannot claim v2 while a parent file or child stays v1."""
    parent, runner = _v2_parent(tmp_path)

    _require_complete_v2_tree(parent)
    assert len(runner.calls) == 2
    assert RunSettings().schema_version == 1
    assert all(call["settings"].schema_version == 2 for call in runner.calls)  # type: ignore[union-attr]
    assert all(call["settings"].repeats == 64 for call in runner.calls)  # type: ignore[union-attr]
    methodology = json.loads((parent / "aa_methodology.json").read_bytes())
    distribution = json.loads((parent / "aa_distribution.json").read_bytes())
    assert methodology["repeats"] == 64
    assert len(distribution["compiled_seconds_a"]) == 64
    assert len(distribution["compiled_seconds_b"]) == 64
    aa_test = json.loads((parent / "aa_test.json").read_bytes())
    for child_record in aa_test["children"]:
        timings = json.loads(
            (parent / child_record["bundle_relative_path"] / "timings.json").read_bytes()
        )
        assert len(timings["compiled_seconds"]) == 64
        assert len(timings["aa_signed_paired_effects"]) == 64


@pytest.mark.parametrize(
    "target",
    (
        "aa_methodology.json",
        "aa_test.json",
        "aa_distribution.json",
    ),
)
def test_schema_v2_aa_tree_fails_closed_on_any_version_mixture(
    tmp_path: Path,
    target: str,
) -> None:
    """Changing any one of the three parent version surfaces invalidates the v2 tree."""
    parent, _ = _v2_parent(tmp_path)
    _require_complete_v2_tree(parent)
    path = parent / target
    payload = json.loads(path.read_bytes())
    payload["schema_version"] = 1
    write_json(path, payload)
    (parent / "verdict.json").unlink()
    write_manifest(parent)

    checked = _evaluate_provisional_aa_bundle(parent)

    assert checked.verdict == "INVALID_EVIDENCE"


@pytest.mark.parametrize("child_index", (0, 1))
def test_schema_v2_aa_parent_rejects_independently_valid_v1_child(
    tmp_path: Path,
    child_index: int,
) -> None:
    """A coherent v1 child cannot enter an otherwise valid v2 A/A tree."""
    parent, runner = _v2_parent(tmp_path)
    _require_complete_v2_tree(parent)
    aa_test_path = parent / "aa_test.json"
    aa_test = json.loads(aa_test_path.read_bytes())
    record = aa_test["children"][child_index]
    child = parent / record["bundle_relative_path"]
    call = runner.calls[child_index]
    shutil.rmtree(child)
    child.mkdir()
    write_normal_child(
        child,
        workload_name=str(call["workload_name"]),
        request=call["request"],
        plan=call["input_plan"],
        metadata=call["catalog_metadata"],  # type: ignore[arg-type]
        driving_finding="perf.speedup_proven",
        settings=RunSettings(),
        run_provenance=runner.run_provenance[child_index],  # type: ignore[arg-type]
    )
    child_verdict = evaluate_bundle(child)
    assert child_verdict["verdict"] != "INVALID_EVIDENCE"
    assert (child / "verdict.json").read_bytes() == canonical_json_bytes(child_verdict)
    _refresh_aa_child_authority(parent, aa_test=aa_test)
    (parent / "verdict.json").unlink()
    write_manifest(parent)
    parent_versions, child_versions = _tree_versions(parent)
    assert parent_versions == [2, 2, 2]
    assert child_versions == ([None, 2] if child_index == 0 else [2, None])
    for child_record in aa_test["children"]:
        child_path = parent / child_record["bundle_relative_path"]
        regated = evaluate_bundle(child_path)
        assert regated["verdict"] != "INVALID_EVIDENCE"
        assert (child_path / "verdict.json").read_bytes() == canonical_json_bytes(regated)

    checked = _evaluate_provisional_aa_bundle(parent)

    assert checked.verdict == "INVALID_EVIDENCE"
    assert checked.driving_finding == "methodology.version"

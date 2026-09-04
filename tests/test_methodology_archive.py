from __future__ import annotations

import hashlib
import json
import shutil
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

import pytest

import hephaestus.durability as durability_module
import hephaestus.gate as gate_module
from hephaestus.bundle import canonical_json_bytes, write_json, write_manifest
from hephaestus.gate import _evaluate_provisional_bundle, evaluate_bundle
from hephaestus.host_state import capture_operation
from hephaestus.provenance import parse_run_provenance
from tests.methodology_v2_helpers import write_v2_normal_bundle
from tests.test_gate import _write_bundle

ARCHIVED_V1_BUNDLES = (
    Path(__file__).parent / "data" / "archived_v1_bundle",
    Path(__file__).parent / "data" / "archived_v1_graph_break_bundle",
)
ARCHIVED_V1_GRAPH_BREAK_BUNDLE = ARCHIVED_V1_BUNDLES[1]
ARCHIVED_V1_MANIFEST_SHA256 = frozenset(
    {
        "8210d74300e4e8b7a2a2035a0a8af504b5c97dc2e03c18e9ccdd92b5ddc239d7",
        "c235ccb572c858bb8037f9474f0eaa4fb19e95a9dc0a0738835cc1c42897e641",
        "d15aae605a0037bdca707830da3477c50407c855f7b42a7f7164f1bc7fd6117d",
        "b04f7026eaa57656e2db5e586c73cb1b54efcb34bec7a726c8016fb37c5a6811",
    }
)


def test_legacy_v1_accuracy_compatibility_is_scoped_to_exact_archived_manifests() -> None:
    """Adding or removing an archive identity must be an explicit compatibility decision."""
    assert getattr(gate_module, "_LEGACY_V1_ACCURACY_MANIFEST_SHA256", None) == (
        ARCHIVED_V1_MANIFEST_SHA256
    )
    assert {
        hashlib.sha256((bundle / "manifest.json").read_bytes()).hexdigest()
        for bundle in ARCHIVED_V1_BUNDLES
    } <= ARCHIVED_V1_MANIFEST_SHA256


@pytest.mark.parametrize(
    "archived_bundle",
    ARCHIVED_V1_BUNDLES,
    ids=("clean-control", "conditional-graph-break"),
)
def test_archived_v1_verdict_bytes_equal_fresh_offline_evaluation(
    archived_bundle: Path,
    tmp_path: Path,
) -> None:
    """A v2 gate change must preserve the real archived schema-v1 verdict bytes."""
    copied_bundle = tmp_path / archived_bundle.name
    shutil.copytree(archived_bundle, copied_bundle)

    assert (
        canonical_json_bytes(evaluate_bundle(copied_bundle))
        == (archived_bundle / "verdict.json").read_bytes()
    )


def test_archived_v1_modern_accuracy_schema_remains_accepted(tmp_path: Path) -> None:
    """Adding the two current metadata fields must not retire valid schema-v1 evidence."""
    copied_bundle = tmp_path / "modern-v1"
    shutil.copytree(ARCHIVED_V1_GRAPH_BREAK_BUNDLE, copied_bundle)
    accuracy_path = copied_bundle / "accuracy.json"
    accuracy = json.loads(accuracy_path.read_bytes())
    accuracy.update(schema_version=1, dtype="torch.float32")
    write_json(accuracy_path, accuracy)
    write_manifest(copied_bundle)

    assert (
        canonical_json_bytes(evaluate_bundle(copied_bundle))
        == (copied_bundle / "verdict.json").read_bytes()
    )


def test_current_v1_bundle_rejects_omitted_accuracy_metadata(tmp_path: Path) -> None:
    """Absence-means-v1 does not make newly produced legacy-shaped accuracy admissible."""
    bundle = tmp_path / "current-v1"
    _write_bundle(bundle)
    accuracy_path = bundle / "accuracy.json"
    accuracy = json.loads(accuracy_path.read_bytes())
    del accuracy["schema_version"]
    del accuracy["dtype"]
    write_json(accuracy_path, accuracy)
    (bundle / "verdict.json").unlink()
    write_manifest(bundle)

    verdict = _evaluate_provisional_bundle(bundle)

    assert verdict["verdict"] == "INVALID_EVIDENCE"
    assert verdict["driving_finding"] == "accuracy.contract"


def test_transplanted_legacy_accuracy_with_fresh_manifest_is_rejected(
    tmp_path: Path,
) -> None:
    """Historical bytes cannot grant compatibility to a non-archived schema-v1 bundle."""
    bundle = tmp_path / "transplant"
    _write_bundle(bundle)
    (bundle / "accuracy.json").write_bytes(
        (ARCHIVED_V1_GRAPH_BREAK_BUNDLE / "accuracy.json").read_bytes()
    )
    (bundle / "verdict.json").unlink()
    write_manifest(bundle)

    verdict = _evaluate_provisional_bundle(bundle)

    assert verdict["verdict"] == "INVALID_EVIDENCE"
    assert verdict["driving_finding"] == "accuracy.contract"


def test_archived_legacy_accuracy_mutation_loses_integrity_and_eligibility(
    tmp_path: Path,
) -> None:
    """Changing archived bytes cannot retain the immutable-manifest compatibility grant."""
    bundle = tmp_path / "mutated-archive"
    shutil.copytree(ARCHIVED_V1_GRAPH_BREAK_BUNDLE, bundle)
    accuracy_path = bundle / "accuracy.json"
    accuracy = json.loads(accuracy_path.read_bytes())
    accuracy["atol"] = 1
    write_json(accuracy_path, accuracy)

    integrity_verdict = evaluate_bundle(bundle)

    assert integrity_verdict["verdict"] == "INVALID_EVIDENCE"
    assert integrity_verdict["driving_finding"] == "evidence.integrity"

    (bundle / "verdict.json").unlink()
    write_manifest(bundle)
    rehashed_verdict = _evaluate_provisional_bundle(bundle)

    assert rehashed_verdict["verdict"] == "INVALID_EVIDENCE"
    assert rehashed_verdict["driving_finding"] == "accuracy.contract"


def _add_partial_schema_version(accuracy: dict[str, object]) -> None:
    accuracy["schema_version"] = 1


def _add_partial_dtype(accuracy: dict[str, object]) -> None:
    accuracy["dtype"] = "torch.float32"


def _add_extra_legacy_key(accuracy: dict[str, object]) -> None:
    accuracy["extra"] = None


def _change_legacy_tolerance(accuracy: dict[str, object]) -> None:
    accuracy["atol"] = 1


def _use_modern_schema_type_alias(accuracy: dict[str, object]) -> None:
    accuracy.update(schema_version=True, dtype="torch.float32")


@pytest.mark.parametrize(
    "mutation",
    (
        _add_partial_schema_version,
        _add_partial_dtype,
        _add_extra_legacy_key,
        _change_legacy_tolerance,
        _use_modern_schema_type_alias,
    ),
)
def test_archived_v1_accuracy_compatibility_rejects_mixed_or_mutated_shapes(
    tmp_path: Path,
    mutation: Callable[[dict[str, object]], None],
) -> None:
    """Only the exact historical key set with frozen tolerances earns compatibility."""
    copied_bundle = tmp_path / mutation.__name__
    shutil.copytree(ARCHIVED_V1_GRAPH_BREAK_BUNDLE, copied_bundle)
    accuracy_path = copied_bundle / "accuracy.json"
    accuracy = json.loads(accuracy_path.read_bytes())
    mutation(accuracy)
    write_json(accuracy_path, accuracy)
    (copied_bundle / "verdict.json").unlink()
    write_manifest(copied_bundle)

    verdict = _evaluate_provisional_bundle(copied_bundle)

    assert verdict["verdict"] == "INVALID_EVIDENCE"
    assert verdict["driving_finding"] == "accuracy.contract"


def test_archived_v1_accuracy_compatibility_keeps_privacy_fail_closed(
    tmp_path: Path,
) -> None:
    """Legacy shape admission cannot bypass the public-evidence scan."""
    copied_bundle = tmp_path / "private-v1"
    shutil.copytree(ARCHIVED_V1_GRAPH_BREAK_BUNDLE, copied_bundle)
    accuracy_path = copied_bundle / "accuracy.json"
    accuracy = json.loads(accuracy_path.read_bytes())
    accuracy["mismatch"] = "<PRIVATE_ROOT>"
    write_json(accuracy_path, accuracy)
    (copied_bundle / "verdict.json").unlink()
    write_manifest(copied_bundle)

    verdict = _evaluate_provisional_bundle(copied_bundle)

    assert verdict["verdict"] == "INVALID_EVIDENCE"
    assert verdict["driving_finding"] == "evidence.privacy"


@pytest.mark.parametrize(
    "source_bundle",
    (*ARCHIVED_V1_BUNDLES, None),
    ids=("archived-v1-clean", "archived-v1-conditional", "synthetic-v2"),
)
def test_sibling_host_ledger_preserves_manifest_and_verdict_bytes(
    tmp_path: Path,
    source_bundle: Path | None,
) -> None:
    """Outer operational evidence cannot enter or alter either frozen bundle schema."""
    root = tmp_path / ("synthetic-v2" if source_bundle is None else source_bundle.name)
    child = root / "runs" / "child"
    if source_bundle is not None:
        shutil.copytree(source_bundle, child)
    else:
        child.parent.mkdir(parents=True)
        write_v2_normal_bundle(child)
    provenance = parse_run_provenance(
        json.loads((child / "run_provenance.json").read_bytes())
    )
    before_manifest = (child / "manifest.json").read_bytes()
    before_verdict = (child / "verdict.json").read_bytes()
    snapshot = {
        "load_average": {"value": [0.1, 0.2, 0.3], "unavailable_reason": None},
    }
    _, capture = capture_operation(
        lambda: None,
        provenance,
        sampler=lambda: snapshot,
        monotonic=iter((1.0, 2.0)).__next__,
        utc_now=iter(
            (
                datetime(2026, 8, 31, 10, 0, 0, tzinfo=UTC),
                datetime(2026, 8, 31, 10, 0, 1, tzinfo=UTC),
            )
        ).__next__,
    )
    factory = getattr(
        durability_module,
        "prepare_host_state_output_root",
        None,
    )
    assert callable(factory), "missing standalone host-state writer"

    with factory(root) as sink:
        sink.append(capture, child)

    assert (child / "manifest.json").read_bytes() == before_manifest
    assert (child / "verdict.json").read_bytes() == before_verdict
    assert canonical_json_bytes(evaluate_bundle(child)) == before_verdict
    assert (root / "host_state.jsonl").is_file()
    assert not tuple(child.rglob("host_state.jsonl"))

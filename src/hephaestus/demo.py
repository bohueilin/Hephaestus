"""Trusted planted-regression demo execution and offline evidence verification."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath

from hephaestus.bundle import (
    canonical_json_bytes,
    strict_json_loads,
    verify_manifest,
    write_json,
    write_manifest,
)
from hephaestus.catalog import (
    CATALOG,
    CatalogRole,
    Proposal,
    catalog_json,
)
from hephaestus.catalog_runtime import CatalogRuntime
from hephaestus.durability import prepare_host_state_output_root
from hephaestus.evidence_contract import (
    V1_DEMO_ACTIONS,
    action_references_capabilities,
    strict_json_equal,
    validate_v1_capability_snapshot,
)
from hephaestus.gate import evaluate_bundle
from hephaestus.measure import RunResult, RunSettings, run_to_bundle
from hephaestus.provenance import ProvenanceError, validate_provenance_chain
from hephaestus.scope import is_scope_json, scope_json

_CATALOG_METADATA_KEYS = {
    "schema_version",
    "entry_id",
    "role",
    "workload_name",
    "requested",
    "effective",
}
@dataclass(frozen=True, slots=True)
class DemoRow:
    """One expected-versus-actual planted-demo outcome."""

    catalog_id: str
    expected_verdict: str
    expected_driving_finding: str
    actual_verdict: str
    actual_driving_finding: str
    passed: bool
    bundle_relative_path: str


@dataclass(frozen=True, slots=True)
class DemoResult:
    """Immutable result of executing or offline-verifying the four-row demo."""

    parent_path: Path
    rows: tuple[DemoRow, ...]
    passed: bool
    mismatches: tuple[str, ...] = ()


class TrustedDemoOrchestrator:
    """Execute all three plants then the clean control through trusted demo authority."""

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

    def run(self) -> DemoResult:
        """Preserve real outcomes without tuning when a planted expectation misses."""
        if self._host_state_sink is None:
            with prepare_host_state_output_root(self._output_root) as sink:
                return self._run(sink)
        return self._run(self._host_state_sink)

    def _run(self, host_state_sink: object) -> DemoResult:
        parent = self._output_root / _parent_name()
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
        entries = _live_demo_entries()
        rows: list[DemoRow] = []
        for entry in entries:
            result = runtime.run_demo(
                Proposal(entry.catalog_id, entry.workload, entry.rationale)
            )
            assert entry.expected_verdict is not None
            assert entry.expected_driving_finding is not None
            rows.append(
                DemoRow(
                    entry.catalog_id,
                    entry.expected_verdict,
                    entry.expected_driving_finding,
                    result.verdict,
                    result.driving_finding,
                    (
                        result.verdict == entry.expected_verdict
                        and result.driving_finding
                        == entry.expected_driving_finding
                    ),
                    result.bundle_relative_path,
                )
            )

        capability_bytes = self._capability_bytes()
        (parent / "torch_capabilities.json").write_bytes(capability_bytes)
        capability_digest = hashlib.sha256(capability_bytes).hexdigest()
        write_json(parent / "catalog.json", catalog_json(capability_digest))
        write_json(parent / "scope.json", scope_json())
        write_json(parent / "demo.json", _demo_json(tuple(rows)))
        write_manifest(parent)
        result = verify_demo_tree(parent)
        if not verify_manifest(parent).valid:
            raise RuntimeError("finalized demo tree failed recursive integrity")
        return result

    def _capability_bytes(self) -> bytes:
        if self._capability_snapshot is not None:
            return canonical_json_bytes(dict(self._capability_snapshot))
        return Path(__file__).with_name("torch_capabilities.json").read_bytes()


def run_planted_demo(
    output_root: Path,
    criteria_path: Path,
    *,
    _worker_relative_root: bool = False,
    _host_state_sink: object | None = None,
) -> DemoResult:
    """Execute the public planted-regression workflow with frozen settings."""
    return TrustedDemoOrchestrator(
        output_root,
        criteria_path,
        _worker_relative_root=_worker_relative_root,
        _host_state_sink=_host_state_sink,
    ).run()


def verify_demo_tree(parent: Path) -> DemoResult:
    """Verify recursive hashes, stored expectations, child gates, and all four rows."""
    parent = Path(parent)
    integrity = verify_manifest(parent)
    if not integrity.valid:
        return DemoResult(parent, (), False, integrity.mismatches)
    mismatches: list[str] = []
    expected_top = {
        "catalog.json",
        "torch_capabilities.json",
        "scope.json",
        "demo.json",
        "runs",
        "manifest.json",
    }
    try:
        actual_top = {path.name for path in parent.iterdir()}
    except OSError:
        return DemoResult(parent, (), False, ("parent:missing",))
    mismatches.extend(
        f"parent:missing:{name}" for name in sorted(expected_top - actual_top)
    )
    mismatches.extend(
        f"parent:unexpected:{name}" for name in sorted(actual_top - expected_top)
    )

    try:
        scope = _json_object(parent / "scope.json")
        if not is_scope_json(scope):
            mismatches.append("scope:invalid")
        catalog = _json_object(parent / "catalog.json")
        capability_bytes = (parent / "torch_capabilities.json").read_bytes()
        demo = _json_object(parent / "demo.json")
        stored_entries = _stored_demo_entries(catalog, capability_bytes)
        rows = _parse_rows(demo)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError):
        return DemoResult(
            parent,
            (),
            False,
            tuple(dict.fromkeys([*mismatches, "schema:invalid"])),
        )
    for entry, contract in zip(stored_entries, V1_DEMO_ACTIONS, strict=True):
        if (
            entry.get("entry_id") != contract.catalog_id
            or entry.get("role") != contract.role
            or entry.get("workload_name") != contract.workload_name
            or entry.get("expected_verdict") != contract.expected_verdict
            or entry.get("expected_driving_finding")
            != contract.expected_driving_finding
        ):
            mismatches.append("catalog:expectation")
        metadata = {key: entry[key] for key in _CATALOG_METADATA_KEYS}
        if not strict_json_equal(metadata, contract.metadata_json()):
            mismatches.append("catalog:action")
    expected_ids = [entry["entry_id"] for entry in stored_entries]
    if [row.catalog_id for row in rows] != expected_ids:
        mismatches.append("row:catalog_order")

    expected_children = {
        row.bundle_relative_path.split("/", maxsplit=1)[1] for row in rows
    }
    try:
        actual_children = {path.name for path in (parent / "runs").iterdir()}
    except OSError:
        actual_children = set()
        mismatches.append("runs:missing")
    mismatches.extend(
        f"runs:missing:{name}" for name in sorted(expected_children - actual_children)
    )
    mismatches.extend(
        f"runs:unexpected:{name}" for name in sorted(actual_children - expected_children)
    )

    try:
        validate_provenance_chain(
            tuple(parent / row.bundle_relative_path for row in rows)
        )
    except ProvenanceError as error:
        mismatches.append(str(error))

    if len({row.bundle_relative_path for row in rows}) != len(rows):
        mismatches.append("row:duplicate_bundle_path")
    for row, entry in zip(rows, stored_entries, strict=False):
        _verify_row(parent, row, entry, mismatches)

    unique_mismatches = tuple(dict.fromkeys(mismatches))
    return DemoResult(
        parent,
        rows,
        not unique_mismatches and all(row.passed for row in rows),
        unique_mismatches,
    )


def _verify_row(
    parent: Path,
    row: DemoRow,
    entry: dict[str, object],
    mismatches: list[str],
) -> None:
    if row.catalog_id != entry.get("entry_id"):
        mismatches.append("row:catalog_id")
    if row.expected_verdict != entry.get("expected_verdict"):
        mismatches.append("row:expected_verdict")
    if row.expected_driving_finding != entry.get("expected_driving_finding"):
        mismatches.append("row:expected_driving_finding")
    child = parent / row.bundle_relative_path
    if not child.is_dir() or child.is_symlink():
        mismatches.append("row:child_path")
        return
    child_integrity = verify_manifest(child)
    mismatches.extend(f"child:{item}" for item in child_integrity.mismatches)
    try:
        stored_verdict_bytes = (child / "verdict.json").read_bytes()
        stored_verdict = strict_json_loads(stored_verdict_bytes)
        config = _json_object(child / "config.json")
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError):
        mismatches.append("child:invalid")
        return
    if not isinstance(stored_verdict, dict):
        mismatches.append("child:verdict")
        return
    actual_verdict = stored_verdict.get("verdict")
    actual_finding = stored_verdict.get("driving_finding")
    if row.actual_verdict != actual_verdict:
        mismatches.append("row:actual_verdict")
    if row.actual_driving_finding != actual_finding:
        mismatches.append("row:actual_driving_finding")
    expected_pass = (
        row.expected_verdict == actual_verdict
        and row.expected_driving_finding == actual_finding
        and actual_verdict != "INVALID_EVIDENCE"
    )
    if row.passed is not expected_pass:
        mismatches.append("row:pass")
    metadata = {key: entry[key] for key in _CATALOG_METADATA_KEYS}
    if config.get("catalog") != metadata:
        mismatches.append("row:catalog_metadata")
    regated = evaluate_bundle(child)
    if stored_verdict_bytes != canonical_json_bytes(regated):
        mismatches.append("child:gate_verdict")
    if regated.get("verdict") == "INVALID_EVIDENCE":
        mismatches.append("child:invalid_evidence")


def _stored_demo_entries(
    catalog: dict[str, object], capability_bytes: bytes
) -> tuple[dict[str, object], ...]:
    if catalog.keys() != {"schema_version", "entries", "torch_capabilities_sha256"}:
        raise ValueError("invalid catalog")
    if (
        type(catalog.get("schema_version")) is not int
        or catalog["schema_version"] != 1
        or catalog.get("torch_capabilities_sha256")
        != hashlib.sha256(capability_bytes).hexdigest()
    ):
        raise ValueError("invalid catalog binding")
    capabilities = validate_v1_capability_snapshot(
        strict_json_loads(capability_bytes)
    )
    entries = catalog.get("entries")
    if not isinstance(entries, list):
        raise ValueError("invalid entries")
    plants = [
        entry
        for entry in entries
        if isinstance(entry, dict) and entry.get("role") == "planted"
    ]
    controls = [
        entry
        for entry in entries
        if isinstance(entry, dict) and entry.get("role") == "clean-control"
    ]
    selected = plants + controls
    if len(plants) != 3 or len(controls) != 1:
        raise ValueError("invalid demo roles")
    expected_entry_keys = _CATALOG_METADATA_KEYS | {
        "rationale",
        "expected_verdict",
        "expected_driving_finding",
    }
    for entry, contract in zip(selected, V1_DEMO_ACTIONS, strict=True):
        if entry.keys() != expected_entry_keys:
            raise ValueError("invalid catalog metadata")
        rationale = entry.get("rationale")
        if (
            not isinstance(rationale, str)
            or not rationale
            or rationale != rationale.strip()
            or "\n" in rationale
            or "\r" in rationale
        ):
            raise ValueError("invalid catalog rationale")
        if not isinstance(entry.get("expected_verdict"), str) or not isinstance(
            entry.get("expected_driving_finding"), str
        ):
            raise ValueError("invalid catalog expectation")
        if not action_references_capabilities(contract, capabilities):
            raise ValueError("unavailable demo action")
    return tuple(selected)


def _parse_rows(demo: dict[str, object]) -> tuple[DemoRow, ...]:
    if (
        demo.keys() != {"schema_version", "rows"}
        or type(demo.get("schema_version")) is not int
        or demo["schema_version"] != 1
    ):
        raise ValueError("invalid demo schema")
    raw_rows = demo.get("rows")
    if not isinstance(raw_rows, list) or len(raw_rows) != 4:
        raise ValueError("invalid demo rows")
    expected_keys = {
        "catalog_id",
        "expected_verdict",
        "expected_driving_finding",
        "actual_verdict",
        "actual_driving_finding",
        "passed",
        "bundle_relative_path",
    }
    rows: list[DemoRow] = []
    for value in raw_rows:
        if not isinstance(value, dict) or value.keys() != expected_keys:
            raise ValueError("invalid demo row")
        strings = (
            value["catalog_id"],
            value["expected_verdict"],
            value["expected_driving_finding"],
            value["actual_verdict"],
            value["actual_driving_finding"],
        )
        if any(not isinstance(item, str) or not item for item in strings):
            raise ValueError("invalid demo row values")
        if type(value["passed"]) is not bool or not _safe_child_path(
            value["bundle_relative_path"]
        ):
            raise ValueError("invalid demo row values")
        rows.append(
            DemoRow(
                strings[0],
                strings[1],
                strings[2],
                strings[3],
                strings[4],
                value["passed"],
                value["bundle_relative_path"],
            )
        )
    return tuple(rows)


def _demo_json(rows: tuple[DemoRow, ...]) -> dict[str, object]:
    return {
        "schema_version": 1,
        "rows": [
            {
                "catalog_id": row.catalog_id,
                "expected_verdict": row.expected_verdict,
                "expected_driving_finding": row.expected_driving_finding,
                "actual_verdict": row.actual_verdict,
                "actual_driving_finding": row.actual_driving_finding,
                "passed": row.passed,
                "bundle_relative_path": row.bundle_relative_path,
            }
            for row in rows
        ],
    }


def _live_demo_entries() -> tuple[object, ...]:
    plants = tuple(entry for entry in CATALOG if entry.role is CatalogRole.PLANTED)
    controls = tuple(
        entry for entry in CATALOG if entry.role is CatalogRole.CLEAN_CONTROL
    )
    if len(plants) != 3 or len(controls) != 1:
        raise RuntimeError("v1 demo requires exactly three plants and one clean control")
    return (*plants, *controls)


def _safe_child_path(value: object) -> bool:
    if not isinstance(value, str) or "\\" in value:
        return False
    path = PurePosixPath(value)
    return (
        not path.is_absolute()
        and path.as_posix() == value
        and len(path.parts) == 2
        and path.parts[0] == "runs"
        and path.parts[1] not in {"", ".", ".."}
    )


def _json_object(path: Path) -> dict[str, object]:
    loaded = strict_json_loads(path.read_bytes())
    if not isinstance(loaded, dict):
        raise ValueError("expected JSON object")
    return loaded


def _parent_name() -> str:
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S.%fZ")
    return f"planted-demo-{timestamp}"


__all__ = [
    "DemoResult",
    "DemoRow",
    "TrustedDemoOrchestrator",
    "run_planted_demo",
    "verify_demo_tree",
]

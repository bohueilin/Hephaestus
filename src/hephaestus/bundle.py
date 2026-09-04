"""Canonical evidence-bundle serialization and payload-integrity checks."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Final

MANIFEST_NAME: Final = "manifest.json"
MANIFEST_SCHEMA_VERSION: Final = 1


@dataclass(frozen=True)
class IntegrityResult:
    """The result of checking a bundle's unsigned payload manifest."""

    valid: bool
    mismatches: tuple[str, ...]


def canonical_json_bytes(value: object) -> bytes:
    """Serialize JSON into deterministic, compact UTF-8-compatible ASCII bytes."""
    encoded = json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )
    return encoded.encode("utf-8")


def strict_json_loads(value: str | bytes | bytearray) -> object:
    """Parse only standards-compliant JSON, rejecting NaN and Infinity tokens."""
    return json.loads(
        value,
        parse_constant=_reject_nonfinite_constant,
        parse_float=_finite_json_float,
    )


def write_json(path: Path, value: object) -> None:
    """Write a value using the bundle's canonical JSON encoding."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_json_bytes(value))


def write_manifest(bundle_dir: Path) -> dict[str, object]:
    """Write a recursive SHA-256 manifest for every evidence file except itself."""
    files = _payload_hashes(bundle_dir)
    manifest: dict[str, object] = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "files": dict(sorted(files.items())),
    }
    write_json(bundle_dir / MANIFEST_NAME, manifest)
    return manifest


def verify_manifest(bundle_dir: Path) -> IntegrityResult:
    """Verify that stored payload files exactly match the unsigned manifest mapping."""
    manifest_path = bundle_dir / MANIFEST_NAME
    if not manifest_path.is_file() or manifest_path.is_symlink():
        return IntegrityResult(valid=False, mismatches=(f"missing:{MANIFEST_NAME}",))

    try:
        manifest = strict_json_loads(manifest_path.read_bytes())
        expected = _manifest_files(manifest)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError):
        return IntegrityResult(valid=False, mismatches=(f"changed:{MANIFEST_NAME}",))

    try:
        actual = _payload_hashes(bundle_dir)
    except ValueError as error:
        return IntegrityResult(valid=False, mismatches=(str(error),))

    mismatches = [f"missing:{path}" for path in sorted(set(expected) - set(actual))]
    mismatches.extend(
        f"changed:{path}"
        for path in sorted(set(expected) & set(actual))
        if expected[path] != actual[path]
    )
    mismatches.extend(f"unexpected:{path}" for path in sorted(set(actual) - set(expected)))
    return IntegrityResult(valid=not mismatches, mismatches=tuple(mismatches))


def finalize_bundle(
    bundle_dir: Path,
    evaluator: Callable[[Path], dict[str, object]],
) -> dict[str, object]:
    """Seal raw evidence, store its verdict, then prove the finalized bundle re-gates."""
    write_manifest(bundle_dir)
    provisional_verdict = evaluator(bundle_dir)
    write_json(bundle_dir / "verdict.json", provisional_verdict)
    write_manifest(bundle_dir)

    integrity = verify_manifest(bundle_dir)
    if not integrity.valid:
        raise RuntimeError(
            f"finalized bundle failed integrity verification: {integrity.mismatches}"
        )
    offline_verdict = evaluator(bundle_dir)
    stored_bytes = (bundle_dir / "verdict.json").read_bytes()
    if stored_bytes != canonical_json_bytes(offline_verdict):
        raise RuntimeError("stored verdict differs from fresh finalized offline evaluation")
    return offline_verdict


def _manifest_files(manifest: object) -> dict[str, str]:
    if (
        not isinstance(manifest, dict)
        or manifest.keys() != {"schema_version", "files"}
        or type(manifest.get("schema_version")) is not int
        or manifest["schema_version"] != MANIFEST_SCHEMA_VERSION
    ):
        raise ValueError("invalid manifest schema")
    files = manifest.get("files")
    if not isinstance(files, dict):
        raise ValueError("invalid manifest files")

    parsed: dict[str, str] = {}
    for path, digest in files.items():
        if not isinstance(path, str) or not isinstance(digest, str):
            raise ValueError("invalid manifest entry")
        if not _is_relative_payload_path(path) or not _is_sha256(digest):
            raise ValueError("invalid manifest entry")
        parsed[path] = digest
    return parsed


def _payload_hashes(bundle_dir: Path) -> dict[str, str]:
    if not bundle_dir.is_dir() or bundle_dir.is_symlink():
        raise ValueError(f"invalid bundle directory:{bundle_dir}")

    files: dict[str, str] = {}
    for path in sorted(bundle_dir.rglob("*"), key=lambda candidate: candidate.as_posix()):
        relative = path.relative_to(bundle_dir).as_posix()
        if path.is_symlink():
            raise ValueError(f"symlink:{relative}")
        if path.is_dir():
            continue
        if not path.is_file():
            raise ValueError(f"unsupported:{relative}")
        if relative != MANIFEST_NAME:
            try:
                payload = path.read_bytes()
            except OSError as error:
                raise ValueError(f"unreadable:{relative}") from error
            files[relative] = hashlib.sha256(payload).hexdigest()
    return files


def _is_relative_payload_path(path: str) -> bool:
    candidate = Path(path)
    return (
        candidate.as_posix() == path
        and not candidate.is_absolute()
        and path != MANIFEST_NAME
        and ".." not in candidate.parts
        and path != "."
    )


def _is_sha256(digest: str) -> bool:
    return len(digest) == 64 and all(character in "0123456789abcdef" for character in digest)


def _reject_nonfinite_constant(value: str) -> object:
    raise ValueError(f"nonfinite JSON constant: {value}")


def _finite_json_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ValueError(f"nonfinite JSON number: {value}")
    return parsed

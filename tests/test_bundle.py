import hashlib
import json
from pathlib import Path

import pytest

from hephaestus.bundle import (
    canonical_json_bytes,
    strict_json_loads,
    verify_manifest,
    write_json,
    write_manifest,
)


def test_canonical_json_uses_sorted_compact_utf8_bytes() -> None:
    """A key-order change must not change the persisted JSON bytes."""
    assert canonical_json_bytes({"z": "café", "a": [2, 1]}) == b'{"a":[2,1],"z":"caf\\u00e9"}'


@pytest.mark.parametrize("value", (float("nan"), float("inf"), float("-inf")))
def test_canonical_json_rejects_nonfinite_numbers(value: float) -> None:
    """Evidence bytes must never contain Python's non-standard NaN/Infinity tokens."""
    with pytest.raises(ValueError, match="Out of range float values"):
        canonical_json_bytes({"measurement": value})


@pytest.mark.parametrize("token", (b"NaN", b"Infinity", b"-Infinity"))
def test_strict_json_loader_rejects_nonstandard_nonfinite_constants(
    token: bytes,
) -> None:
    """Every evidence/config consumer shares the same strict JSON ingress boundary."""
    with pytest.raises(ValueError, match="nonfinite JSON constant"):
        strict_json_loads(b'{"measurement":' + token + b"}")


@pytest.mark.parametrize("payload", [b'{"value":1e400}', b'{"value":-1e400}'])
def test_strict_json_loader_rejects_finite_syntax_that_overflows_float(
    payload: bytes,
) -> None:
    """A standards-shaped exponent must not enter evidence as an infinite Python float."""
    with pytest.raises(ValueError, match="finite"):
        strict_json_loads(payload)


def test_manifest_hashes_every_non_manifest_file_recursively(tmp_path: Path) -> None:
    """A nested evidence payload must be represented by its POSIX relative path and SHA-256."""
    write_json(tmp_path / "env.json", {"os": "test"})
    nested = tmp_path / "raw" / "timings.bin"
    nested.parent.mkdir()
    nested.write_bytes(b"cold\x00warm")

    manifest = write_manifest(tmp_path)

    assert manifest == {
        "schema_version": 1,
        "files": {
            "env.json": "926f616aef61cbc05498abbb584d6ce2b440e95dc9094b5130e568c6d26ad664",
            "raw/timings.bin": "e5293ff8acec8f419d53aa7bcc6790606614c6b0fde10faeef5272330c47a1ea",
        },
    }
    assert hashlib.sha256(b'{"os":"test"}').hexdigest() == manifest["files"]["env.json"]
    assert verify_manifest(tmp_path).valid is True


def test_manifest_reports_an_altered_payload(tmp_path: Path) -> None:
    """Changing bytes after finalization must invalidate the payload rather than its summary."""
    payload = tmp_path / "evidence.json"
    payload.write_bytes(b"before")
    write_manifest(tmp_path)
    payload.write_bytes(b"after")

    result = verify_manifest(tmp_path)

    assert result.valid is False
    assert result.mismatches == ("changed:evidence.json",)


def test_manifest_reports_a_missing_payload(tmp_path: Path) -> None:
    """Removing a committed evidence file must leave a named integrity mismatch."""
    payload = tmp_path / "evidence.json"
    payload.write_bytes(b"present")
    write_manifest(tmp_path)
    payload.unlink()

    result = verify_manifest(tmp_path)

    assert result.valid is False
    assert result.mismatches == ("missing:evidence.json",)


def test_manifest_reports_an_unexpected_payload(tmp_path: Path) -> None:
    """Files added after finalization must not silently escape the integrity boundary."""
    (tmp_path / "evidence.json").write_bytes(b"present")
    write_manifest(tmp_path)
    (tmp_path / "extra.txt").write_bytes(b"not committed")

    result = verify_manifest(tmp_path)

    assert result.valid is False
    assert result.mismatches == ("unexpected:extra.txt",)


def test_manifest_rejects_a_symlink(tmp_path: Path) -> None:
    """A linked file must not be accepted as bundle evidence."""
    target = tmp_path / "payload.txt"
    target.write_text("payload")
    (tmp_path / "linked.txt").symlink_to(target)

    try:
        write_manifest(tmp_path)
    except ValueError as error:
        assert str(error) == "symlink:linked.txt"
    else:
        raise AssertionError("write_manifest accepted a symlink")


@pytest.mark.parametrize("recursive", (False, True), ids=("normal", "recursive"))
@pytest.mark.parametrize(
    "schema_mutation",
    ("boolean-version", "float-version", "unexpected-key"),
)
def test_manifest_rejects_non_exact_schema_for_normal_and_recursive_parents(
    tmp_path: Path,
    recursive: bool,
    schema_mutation: str,
) -> None:
    """Manifest schema 1 requires exact keys and an exact integer version."""
    if recursive:
        child = tmp_path / "runs" / "child"
        child.mkdir(parents=True)
        (child / "evidence.json").write_bytes(b"child")
        write_manifest(child)
    else:
        (tmp_path / "evidence.json").write_bytes(b"normal")
    manifest_path = tmp_path / "manifest.json"
    write_manifest(tmp_path)
    manifest = json.loads(manifest_path.read_bytes())
    if schema_mutation == "boolean-version":
        manifest["schema_version"] = True
    elif schema_mutation == "float-version":
        manifest["schema_version"] = 1.0
    else:
        manifest["unexpected"] = True
    write_json(manifest_path, manifest)

    result = verify_manifest(tmp_path)

    assert result.valid is False
    assert result.mismatches == ("changed:manifest.json",)


def test_manifest_reports_an_unreadable_payload(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A payload read failure must become named invalid evidence instead of escaping."""
    payload = tmp_path / "evidence.json"
    payload.write_bytes(b"present")
    write_manifest(tmp_path)
    original_read_bytes = Path.read_bytes

    def fail_payload_read(path: Path) -> bytes:
        if path == payload:
            raise OSError("simulated payload read failure")
        return original_read_bytes(path)

    monkeypatch.setattr(Path, "read_bytes", fail_payload_read)

    result = verify_manifest(tmp_path)

    assert result.valid is False
    assert result.mismatches == ("unreadable:evidence.json",)

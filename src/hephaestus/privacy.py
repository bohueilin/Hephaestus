"""Torch-free semantic path normalization and public-evidence validation."""

from __future__ import annotations

import getpass
import re
import socket
from collections.abc import Mapping
from pathlib import Path
from typing import Final

_TOKENS: Final = {
    "<INDUCTOR_CACHE>",
    "<HEPHAESTUS_PACKAGE>",
    "<TORCH_PACKAGE>",
    "<PYTHON_ENV>",
    "<PYTHON_RUNTIME>",
    "<PROJECT_ROOT>",
    "<HOME>",
    "<TEMP_ROOT>",
}
_TOKEN_PATTERN = re.compile(r"<[A-Z][A-Z_]*>")
_PRIVATE_PATH_PATTERNS: Final = (
    re.compile(r"(?:^|[\s\"'=:(\[,])/(?!/)[^\s\"']+"),
    re.compile(r"(?:^|[\s\"'=:(\[,])(?:[A-Za-z]:[\\/]|\\\\)[^\s\"']+"),
)


def normalize_public_evidence(
    value: object,
    roots: Mapping[str, Path],
) -> object:
    """Replace private roots recursively, including mapping keys, and reject residue."""
    replacements = _replacements(roots)
    normalized = _normalize(value, replacements)
    validate_public_evidence(normalized)
    return normalized


def normalize_dynamo_report(
    report: object,
    roots: Mapping[str, Path],
) -> dict[str, object]:
    """Normalize report paths while preserving upstream reason and trigger bytes exactly."""
    if not isinstance(report, dict):
        raise ValueError("Dynamo report must be an object")
    reasons = _verbatim_values(report.get("graph_breaks"), "reason")
    triggers = _verbatim_values(report.get("recompiles"), "trigger")
    replacements = _replacements(roots)
    for value in (*reasons, *triggers):
        if _contains_private_root(value, replacements):
            raise ValueError("verbatim compiler evidence contains a private root")
        validate_public_evidence(value)
    normalized = normalize_public_evidence(report, roots)
    if not isinstance(normalized, dict):
        raise ValueError("Dynamo report must remain an object")
    normalized_breaks = normalized.get("graph_breaks")
    normalized_recompiles = normalized.get("recompiles")
    assert isinstance(normalized_breaks, list)
    assert isinstance(normalized_recompiles, list)
    for record, reason in zip(normalized_breaks, reasons, strict=True):
        assert isinstance(record, dict)
        record["reason"] = reason
    for record, trigger in zip(normalized_recompiles, triggers, strict=True):
        assert isinstance(record, dict)
        record["trigger"] = trigger
    validate_public_evidence(normalized)
    return normalized


def validate_public_evidence(value: object) -> None:
    """Reject unknown tokens and residual private identity/path fragments recursively."""
    for text in _strings(value):
        unknown_tokens = set(_TOKEN_PATTERN.findall(text)) - _TOKENS
        if unknown_tokens:
            raise ValueError("public evidence contains an unknown semantic token")
        if any(pattern.search(text) for pattern in _PRIVATE_PATH_PATTERNS):
            raise ValueError("public evidence contains a private absolute path")
        home = str(Path.home())
        if home and home in text:
            raise ValueError("public evidence contains a private home path")
        username = getpass.getuser()
        if len(username) >= 4 and username.casefold() in text.casefold():
            raise ValueError("public evidence contains a private username")
        hostname = socket.gethostname()
        if len(hostname) >= 4 and hostname.casefold() in text.casefold():
            raise ValueError("public evidence contains a private hostname")


def _replacements(roots: Mapping[str, Path]) -> tuple[tuple[str, str], ...]:
    items: list[tuple[str, str]] = []
    rendered_roots: set[str] = set()
    for token, root in roots.items():
        if token not in _TOKENS or not isinstance(root, Path):
            raise ValueError("invalid semantic-root mapping")
        rendered = str(root)
        if rendered in {"", ".", "/"}:
            raise ValueError("semantic roots must be specific absolute paths")
        if rendered in rendered_roots:
            raise ValueError("duplicate rendered semantic root")
        rendered_roots.add(rendered)
        items.append((rendered, token))
    return tuple(sorted(items, key=lambda item: len(item[0]), reverse=True))


def _normalize(value: object, replacements: tuple[tuple[str, str], ...]) -> object:
    if isinstance(value, str):
        return _replace(value, replacements)
    if value is None or type(value) in {bool, int, float}:
        return value
    if isinstance(value, list | tuple):
        return [_normalize(item, replacements) for item in value]
    if isinstance(value, Mapping):
        normalized: dict[str, object] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise ValueError("public evidence mapping keys must be strings")
            normalized_key = _replace(key, replacements)
            if normalized_key in normalized:
                raise ValueError("semantic path normalization caused a key collision")
            normalized[normalized_key] = _normalize(item, replacements)
        return normalized
    raise ValueError("public evidence must contain only JSON values")


def _replace(value: str, replacements: tuple[tuple[str, str], ...]) -> str:
    result = value
    for root, token in replacements:
        result = result.replace(root, token)
    return result


def _strings(value: object) -> tuple[str, ...]:
    if isinstance(value, str):
        return (value,)
    if isinstance(value, Mapping):
        return tuple(
            text
            for key, item in value.items()
            for text in (*_strings(key), *_strings(item))
        )
    if isinstance(value, list | tuple):
        return tuple(text for item in value for text in _strings(item))
    return ()


def _verbatim_values(records: object, key: str) -> tuple[str, ...]:
    if not isinstance(records, list):
        raise ValueError("Dynamo event records must be a list")
    values: list[str] = []
    for record in records:
        if not isinstance(record, dict) or not isinstance(record.get(key), str):
            raise ValueError("Dynamo event lacks verbatim evidence")
        values.append(record[key])
    return tuple(values)


def _contains_private_root(
    value: str,
    replacements: tuple[tuple[str, str], ...],
) -> bool:
    return any(root in value for root, _ in replacements)


__all__ = [
    "normalize_dynamo_report",
    "normalize_public_evidence",
    "validate_public_evidence",
]

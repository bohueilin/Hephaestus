"""Lifetime-safe resolution of the canonical packaged v0.1 gate criteria."""

from __future__ import annotations

import hashlib
from collections.abc import Iterator
from contextlib import contextmanager
from importlib.resources import as_file, files
from pathlib import Path
from typing import Final

_V1_GATE_SHA256: Final = "e1566b35b5727cd2dcb8507800f45ee5bcb2c0ed7a96a511224d9b1bc962bd56"


@contextmanager
def packaged_criteria_path() -> Iterator[Path]:
    """Keep the canonical resource materialized for the full trusted workflow."""
    resource = files("hephaestus").joinpath("gates", "default.yaml")
    try:
        with as_file(resource) as path:
            candidate = Path(path)
            if candidate.is_file():
                _require_v1_bytes(candidate)
                yield candidate
                return
    except FileNotFoundError:
        pass

    # Editable source checkouts do not materialize Hatch force-includes. The fallback
    # is admitted only at the fixed repository location with the reviewed exact bytes.
    checkout = Path(__file__).resolve().parents[2] / "gates" / "default.yaml"
    if not (checkout.is_file() and (checkout.parents[1] / "pyproject.toml").is_file()):
        raise RuntimeError("canonical packaged gate criteria are unavailable")
    _require_v1_bytes(checkout)
    yield checkout


def _require_v1_bytes(path: Path) -> None:
    try:
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError as error:
        raise RuntimeError("canonical gate criteria are unreadable") from error
    if digest != _V1_GATE_SHA256:
        raise RuntimeError("canonical gate criteria bytes differ from frozen v0.1")


__all__ = ["packaged_criteria_path"]

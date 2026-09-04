"""Exec-created owner for one durable A/A or planted-demo attempt."""

from __future__ import annotations

import json
import os
import stat
import sys
from contextlib import ExitStack, suppress
from pathlib import Path

_BOOTSTRAP_EXIT_CODE = 1


def main(argv: list[str] | None = None) -> int:
    """Preflight, pin, and execute one scoped attempt in this disposable process."""
    arguments = sys.argv[1:] if argv is None else argv
    if len(arguments) != 2:
        print(
            "durable worker bootstrap failed: expected payload and control descriptor",
            file=sys.stderr,
        )
        return _BOOTSTRAP_EXIT_CODE
    try:
        control_fd = _control_descriptor(arguments[1])
    except (OSError, ValueError) as error:
        print(f"durable worker bootstrap failed: invalid control channel: {error}", file=sys.stderr)
        return _BOOTSTRAP_EXIT_CODE
    attempt_root = None
    science_result: int | None = None
    try:
        try:
            source_argv = json.loads(arguments[0])
        except json.JSONDecodeError as error:
            print(
                f"durable worker bootstrap failed: invalid argv payload: {error}",
                file=sys.stderr,
            )
            return _BOOTSTRAP_EXIT_CODE
        if not isinstance(source_argv, list) or not all(
            isinstance(item, str) for item in source_argv
        ):
            print("durable worker bootstrap failed: argv payload must be strings", file=sys.stderr)
            return _BOOTSTRAP_EXIT_CODE

        from hephaestus.cli import (
            _WORKER_SCIENCE_START_MARKER,
            _parser,
            _prepare_science,
            _run_after_parse,
        )
        from hephaestus.durability import DurabilityError, prepare_attempt_output_root

        parser = _parser()
        parsed = parser.parse_args(source_argv)
        if parsed.command not in {"aa-test", "demo-planted-regressions"}:
            print("durable worker bootstrap failed: unsupported command", file=sys.stderr)
            return _BOOTSTRAP_EXIT_CODE
        attempt_root = prepare_attempt_output_root(
            parsed.output_root,
            allow_volatile_output=parsed.allow_volatile_output,
        )
        attempt_root.__enter__()
        attempt_root.enter_worker_directory()
        parsed.output_root = Path(".")
        parsed.attempt_root = attempt_root
        parsed._worker_relative_root = True
        preparation = ExitStack()
        try:
            preparation.enter_context(_prepare_science(parser, parsed))
        except Exception as error:
            with suppress(Exception):
                preparation.close()
            print(
                f"durable worker bootstrap failed: science preparation unavailable: {error}",
                file=sys.stderr,
            )
            return _BOOTSTRAP_EXIT_CODE
        with preparation:
            try:
                _emit_science_start(control_fd, _WORKER_SCIENCE_START_MARKER)
            except OSError as error:
                print(
                    f"durable worker bootstrap failed: control marker unavailable: {error}",
                    file=sys.stderr,
                )
                return _BOOTSTRAP_EXIT_CODE
            control_fd = None
            science_result = _run_after_parse(parser, parsed)
    except DurabilityError as error:
        parser.error(str(error))
    finally:
        for stream in (sys.stdout, sys.stderr):
            with suppress(OSError, ValueError):
                stream.flush()
        if attempt_root is not None:
            host_ledger_fd = getattr(attempt_root, "_host_ledger_fd", None)
            failures = attempt_root.close()
            if host_ledger_fd in failures:
                print(
                    "durable worker cleanup error: host-state ledger descriptor close failed",
                    file=sys.stderr,
                )
                failures = tuple(
                    descriptor for descriptor in failures if descriptor != host_ledger_fd
                )
            if failures:
                print("durable worker cleanup error: descriptor close failed", file=sys.stderr)
                if science_result == 0:
                    science_result = _BOOTSTRAP_EXIT_CODE
                with suppress(OSError, ValueError):
                    sys.stderr.flush()
        if control_fd is not None:
            with suppress(OSError):
                os.close(control_fd)
    if science_result is None:
        return _BOOTSTRAP_EXIT_CODE
    return science_result


def _control_descriptor(raw: str) -> int:
    try:
        descriptor = int(raw)
    except ValueError as error:
        raise ValueError("control descriptor must be an integer") from error
    if descriptor <= 2 or str(descriptor) != raw:
        raise ValueError("control descriptor is outside the inherited range")
    if not stat.S_ISFIFO(os.fstat(descriptor).st_mode):
        raise ValueError("control descriptor is not a pipe")
    return descriptor


def _emit_science_start(descriptor: int, marker: bytes) -> None:
    offset = 0
    write_error: OSError | None = None
    try:
        while offset < len(marker):
            written = os.write(descriptor, marker[offset:])
            if written <= 0:
                raise OSError("control marker write made no progress")
            if written > len(marker) - offset:
                raise OSError("control marker write returned an invalid count")
            offset += written
    except OSError as error:
        write_error = error
    with suppress(OSError):
        os.close(descriptor)
    if write_error is not None:
        raise write_error


if __name__ == "__main__":
    raise SystemExit(main())

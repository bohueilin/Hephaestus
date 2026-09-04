"""Pinned durable CLI-attempt roots and append-only operational receipts."""

from __future__ import annotations

import errno
import hashlib
import json
import math
import os
import re
import stat
import tempfile
from collections.abc import Callable
from contextlib import suppress
from datetime import UTC, datetime
from fcntl import F_GETPATH, LOCK_EX, LOCK_UN, fcntl, flock
from pathlib import Path
from threading import Lock

from hephaestus.bundle import canonical_json_bytes, strict_json_loads
from hephaestus.host_state import (
    HostStateCapture,
    finalize_host_state_record,
)
from hephaestus.provenance import RunProvenance, parse_run_provenance

_LEDGER_READ_CHUNK_BYTES = 8_192
_MAX_LEDGER_RECORD_BYTES = 65_536


class DurabilityError(ValueError):
    """A durable-attempt filesystem boundary could not be established."""


class VolatileOutputRootError(DurabilityError):
    """The requested evidence root resolves into the system temporary directory."""


class AttemptOutputRoot:
    """One opened output-root identity shared by science and its root-level ledger."""

    def __init__(
        self,
        output_root: Path,
        *,
        allow_volatile_output: bool,
        _after_open: Callable[[], None] | None = None,
    ) -> None:
        self._requested_root = Path(output_root)
        self._allow_volatile_output = allow_volatile_output
        self._after_open = _after_open
        self._root_fd: int | None = None
        self._ledger_fd: int | None = None
        self._host_ledger_fd: int | None = None
        self._workflow_root: Path | None = None
        self._stable_root: Path | None = None
        self._host_lifecycle_lock = Lock()

    def __enter__(self) -> AttemptOutputRoot:
        requested = self._requested_root.resolve(strict=False)
        temporary = Path(tempfile.gettempdir()).resolve(strict=False)
        _reject_volatile(requested, temporary, self._allow_volatile_output)
        try:
            requested.mkdir(parents=True, exist_ok=True)
            root_fd = os.open(requested, _directory_open_flags())
        except OSError as error:
            raise DurabilityError("output root must be a real directory") from error
        self._root_fd = root_fd
        try:
            stable_root = _path_for_open_directory(root_fd)
            _reject_volatile(stable_root, temporary, self._allow_volatile_output)
            self._stable_root = stable_root
            self._ledger_fd = _open_verified_ledger(root_fd)
            _validate_existing_ledger(self._ledger_fd, stable_root, root_fd)
            self._host_ledger_fd = _open_verified_host_ledger(root_fd)
        except BaseException as error:
            close_failures = self.close()
            if close_failures:
                raise DurabilityError(
                    "attempt output root cleanup descriptor close failed"
                ) from error
            if isinstance(error, DurabilityError):
                raise
            if not isinstance(error, Exception):
                raise
            raise DurabilityError("attempts ledger is not a safe regular file") from error
        try:
            if self._after_open is not None:
                self._after_open()
            stable_root = _path_for_open_directory(root_fd)
            _reject_volatile(stable_root, temporary, self._allow_volatile_output)
            if os.stat(stable_root) != os.fstat(root_fd):
                raise DurabilityError("output root identity changed during preflight")
        except BaseException as error:
            close_failures = self.close()
            if close_failures:
                raise DurabilityError(
                    "attempt output root cleanup descriptor close failed"
                ) from error
            if isinstance(error, DurabilityError):
                raise
            if not isinstance(error, OSError | ValueError):
                raise
            raise DurabilityError("output root identity changed during preflight") from error
        self._workflow_root = stable_root
        self._stable_root = stable_root
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def enter_worker_directory(self) -> None:
        """Make this disposable worker process operate from the pinned root inode."""
        root_fd = self._root_fd
        if root_fd is None:
            raise RuntimeError("attempt output root is not open")
        try:
            os.fchdir(root_fd)
        except OSError as error:
            raise DurabilityError("could not enter pinned output root") from error

    @property
    def workflow_root(self) -> Path:
        """A descriptor-backed absolute path for all attempt-science filesystem use."""
        if self._workflow_root is None:
            raise RuntimeError("attempt output root is not open")
        return self._workflow_root

    def append_attempt(
        self,
        *,
        workload: str,
        verdict: str,
        clean_control_floor: float | None,
        parent_path: Path,
    ) -> None:
        """Append one canonical record through the preflighted ledger descriptor."""
        ledger_fd = self._ledger_fd
        if ledger_fd is None:
            raise RuntimeError("attempt ledger is not open")
        floor = _finite_floor(clean_control_floor)
        canonical_workload = _nonempty_string(workload, "workload")
        canonical_verdict = _nonempty_string(verdict, "verdict")
        parent = self._absolute_parent(parent_path)
        record = {
            "timestamp_utc": _timestamp_utc(),
            "workload": canonical_workload,
            "verdict": canonical_verdict,
            "clean_control_floor": floor,
            "parent_path": str(parent),
        }
        encoded = (
            json.dumps(record, ensure_ascii=True, separators=(",", ":"), sort_keys=True)
            + "\n"
        ).encode("utf-8")
        if len(encoded) - 1 > _MAX_LEDGER_RECORD_BYTES:
            raise DurabilityError("attempts ledger record exceeds the byte limit")
        locked = False
        offset: int | None = None
        try:
            flock(ledger_fd, LOCK_EX)
            locked = True
            offset = os.lseek(ledger_fd, 0, os.SEEK_END)
            try:
                written = os.write(ledger_fd, encoded)
            except OSError as error:
                _rollback_ledger_append(ledger_fd, offset)
                raise DurabilityError("attempts ledger append failed") from error
            if written != len(encoded):
                _rollback_ledger_append(ledger_fd, offset)
                raise DurabilityError("attempts ledger append was incomplete")
        except DurabilityError:
            raise
        except OSError as error:
            raise DurabilityError("attempts ledger append failed") from error
        finally:
            if locked:
                with suppress(OSError):
                    flock(ledger_fd, LOCK_UN)

    def append(self, capture: HostStateCapture, bundle_path: Path) -> None:
        """Append one host-state row through this root's distinct pinned ledger."""
        with self._host_lifecycle_lock:
            _append_host_state(
                self._required_root_fd(),
                self._required_host_ledger_fd(),
                self._required_stable_root(),
                capture,
                bundle_path,
            )

    def close(self) -> tuple[int, ...]:
        """Close every descriptor owned by this attempt root exactly once."""
        with self._host_lifecycle_lock:
            failures: list[int] = []
            for attribute in ("_host_ledger_fd", "_ledger_fd", "_root_fd"):
                descriptor = getattr(self, attribute)
                if descriptor is not None:
                    try:
                        if not _close_descriptor(descriptor):
                            failures.append(descriptor)
                    finally:
                        setattr(self, attribute, None)
            return tuple(failures)

    def _required_root_fd(self) -> int:
        if self._root_fd is None:
            raise RuntimeError("attempt output root is not open")
        return self._root_fd

    def _required_host_ledger_fd(self) -> int:
        if self._host_ledger_fd is None:
            raise RuntimeError("host-state ledger is not open")
        return self._host_ledger_fd

    def _required_stable_root(self) -> Path:
        if self._stable_root is None:
            raise RuntimeError("attempt output root is not open")
        return self._stable_root

    def _absolute_parent(self, parent_path: Path) -> Path:
        parent = Path(parent_path)
        if parent.is_absolute():
            stable_root = self._stable_root
            if stable_root is None:
                raise RuntimeError("attempt output root is not open")
            try:
                relative = parent.relative_to(stable_root)
            except ValueError as error:
                raise DurabilityError(
                    "completed attempt parent must be below the pinned output root"
                ) from error
        else:
            relative = parent
        if not relative.parts or any(part in {"", ".", ".."} for part in relative.parts):
            raise DurabilityError("completed attempt parent must be below the pinned output root")
        root_fd = self._root_fd
        if root_fd is None:
            raise RuntimeError("attempt output root is not open")
        return _validated_parent_path(
            root_fd,
            relative,
            error_message="completed attempt parent must be a real directory",
        )


class HostStateOutputRoot:
    """Descriptor-pinned outer root owning only the host-state ledger capability."""

    def __init__(self, output_root: Path) -> None:
        self._requested_root = Path(output_root)
        self._root_fd: int | None = None
        self._host_ledger_fd: int | None = None
        self._stable_root: Path | None = None
        self._host_lifecycle_lock = Lock()

    def __enter__(self) -> HostStateOutputRoot:
        requested = self._requested_root.resolve(strict=False)
        try:
            requested.mkdir(parents=True, exist_ok=True)
            root_fd = os.open(requested, _directory_open_flags())
        except OSError as error:
            raise DurabilityError("host-state output root must be a real directory") from error
        self._root_fd = root_fd
        try:
            stable_root = _path_for_open_directory(root_fd)
            self._stable_root = stable_root
            self._host_ledger_fd = _open_verified_host_ledger(root_fd)
            if os.stat(stable_root) != os.fstat(root_fd):
                raise DurabilityError("host-state output root identity changed during preflight")
        except BaseException as error:
            close_failures = self.close()
            if close_failures:
                raise DurabilityError(
                    "host-state output root cleanup descriptor close failed"
                ) from error
            if isinstance(error, DurabilityError):
                raise
            if not isinstance(error, Exception):
                raise
            raise DurabilityError("host-state ledger is not a safe regular file") from error
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def append(self, capture: HostStateCapture, bundle_path: Path) -> None:
        with self._host_lifecycle_lock:
            root_fd = self._root_fd
            ledger_fd = self._host_ledger_fd
            stable_root = self._stable_root
            if root_fd is None or ledger_fd is None or stable_root is None:
                raise RuntimeError("host-state output root is not open")
            _append_host_state(
                root_fd,
                ledger_fd,
                stable_root,
                capture,
                bundle_path,
            )

    def close(self) -> tuple[int, ...]:
        with self._host_lifecycle_lock:
            failures: list[int] = []
            for attribute in ("_host_ledger_fd", "_root_fd"):
                descriptor = getattr(self, attribute)
                if descriptor is not None:
                    try:
                        if not _close_descriptor(descriptor):
                            failures.append(descriptor)
                    finally:
                        setattr(self, attribute, None)
            return tuple(failures)


def prepare_attempt_output_root(
    output_root: Path,
    *,
    allow_volatile_output: bool,
    _after_open: Callable[[], None] | None = None,
) -> AttemptOutputRoot:
    """Preflight and pin an output root before any scoped workflow begins science."""
    return AttemptOutputRoot(
        output_root,
        allow_volatile_output=allow_volatile_output,
        _after_open=_after_open,
    )


def prepare_host_state_output_root(output_root: Path) -> HostStateOutputRoot:
    """Pin an outer root and open only its host-state ledger."""
    return HostStateOutputRoot(output_root)


def resolve_durable_output_root(
    output_root: Path, *, allow_volatile_output: bool
) -> Path:
    """Compatibility check for callers that need only the preflight path decision."""
    resolved = Path(output_root).resolve(strict=False)
    temporary = Path(tempfile.gettempdir()).resolve(strict=False)
    _reject_volatile(resolved, temporary, allow_volatile_output)
    return resolved


def append_attempt_record(
    output_root: Path,
    *,
    workload: str,
    verdict: str,
    clean_control_floor: float | None,
    parent_path: Path,
) -> None:
    """Compatibility wrapper for direct callers; scoped CLI attempts retain one handle."""
    with prepare_attempt_output_root(output_root, allow_volatile_output=True) as root:
        root.append_attempt(
            workload=workload,
            verdict=verdict,
            clean_control_floor=clean_control_floor,
            parent_path=parent_path,
        )


def _directory_open_flags() -> int:
    required = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    nofollow = getattr(os, "O_NOFOLLOW", None)
    if nofollow is None:
        raise DurabilityError("platform lacks no-follow directory opens")
    return required | nofollow


def _path_for_open_directory(descriptor: int) -> Path:
    """Materialize Darwin's path for an already-open directory identity."""
    raw = fcntl(descriptor, F_GETPATH, b"\0" * 512)
    if not isinstance(raw, bytes):
        raise DurabilityError("could not materialize pinned output-root path")
    encoded = raw.split(b"\0", 1)[0]
    if not encoded:
        raise DurabilityError("could not materialize pinned output-root path")
    return Path(os.fsdecode(encoded))


def _open_verified_ledger(root_fd: int) -> int:
    nofollow = getattr(os, "O_NOFOLLOW", None)
    if nofollow is None:
        raise DurabilityError("platform lacks no-follow file opens")
    flags = os.O_RDWR | os.O_APPEND | os.O_CREAT | os.O_NONBLOCK | nofollow
    try:
        ledger_fd = os.open("attempts.jsonl", flags, 0o600, dir_fd=root_fd)
    except OSError as error:
        raise DurabilityError("attempts ledger is not a safe regular file") from error
    try:
        metadata = os.fstat(ledger_fd)
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
            raise DurabilityError("attempts ledger must be an unlinked regular file")
        return ledger_fd
    except BaseException as error:
        if not _close_descriptor(ledger_fd):
            raise DurabilityError(
                "attempts ledger setup cleanup descriptor close failed"
            ) from error
        raise


def _open_verified_host_ledger(root_fd: int) -> int:
    nofollow = getattr(os, "O_NOFOLLOW", None)
    if nofollow is None:
        raise DurabilityError("platform lacks no-follow file opens")
    flags = os.O_RDWR | os.O_APPEND | os.O_CREAT | os.O_NONBLOCK | nofollow
    try:
        ledger_fd = os.open("host_state.jsonl", flags, 0o600, dir_fd=root_fd)
    except OSError as error:
        raise DurabilityError("host-state ledger is not a safe regular file") from error
    try:
        metadata = os.fstat(ledger_fd)
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
            raise DurabilityError("host-state ledger must be a one-link regular file")
        return ledger_fd
    except BaseException as error:
        if not _close_descriptor(ledger_fd):
            raise DurabilityError(
                "host-state ledger setup cleanup descriptor close failed"
            ) from error
        raise


def _append_host_state(
    root_fd: int,
    ledger_fd: int,
    stable_root: Path,
    capture: HostStateCapture,
    bundle_path: Path,
) -> None:
    relative = _relative_bundle_path(stable_root, bundle_path)
    record = finalize_host_state_record(capture, relative.as_posix())
    encoded = canonical_json_bytes(record) + b"\n"
    if len(encoded) - 1 > _MAX_LEDGER_RECORD_BYTES:
        raise DurabilityError("host-state ledger record exceeds the byte limit")

    locked = False
    try:
        flock(ledger_fd, LOCK_EX)
        locked = True
        _recheck_host_ledger_identity(ledger_fd, root_fd)
        stored = _read_manifested_provenance(root_fd, relative)
        if stored != capture.run_provenance:
            raise DurabilityError("host-state capture provenance does not match bundle")
        _recheck_host_ledger_identity(ledger_fd, root_fd)
        written = os.write(ledger_fd, encoded)
        if written != len(encoded):
            raise DurabilityError("host-state ledger append was incomplete")
        _recheck_host_ledger_identity(ledger_fd, root_fd)
    except DurabilityError:
        raise
    except OSError as error:
        raise DurabilityError("host-state ledger append failed") from error
    finally:
        if locked:
            with suppress(OSError):
                flock(ledger_fd, LOCK_UN)


def _relative_bundle_path(stable_root: Path, bundle_path: Path) -> Path:
    candidate = Path(bundle_path)
    if candidate.is_absolute():
        try:
            relative = candidate.relative_to(stable_root)
        except ValueError as error:
            raise DurabilityError(
                "host-state bundle path must be below the pinned output root"
            ) from error
    else:
        relative = candidate
    if (
        not relative.parts
        or relative.as_posix() != str(relative)
        or any(part in {"", ".", ".."} for part in relative.parts)
    ):
        raise DurabilityError("host-state bundle path is not canonical")
    return relative


def _recheck_host_ledger_identity(descriptor: int, root_fd: int) -> None:
    try:
        opened = os.fstat(descriptor)
        named = os.stat(
            "host_state.jsonl",
            dir_fd=root_fd,
            follow_symlinks=False,
        )
    except OSError as error:
        raise DurabilityError("host-state ledger identity is unavailable") from error
    if (
        not stat.S_ISREG(opened.st_mode)
        or opened.st_nlink != 1
        or not stat.S_ISREG(named.st_mode)
        or named.st_nlink != 1
        or (opened.st_dev, opened.st_ino) != (named.st_dev, named.st_ino)
    ):
        raise DurabilityError("host-state ledger identity or link count changed")


def _read_manifested_provenance(root_fd: int, relative: Path) -> RunProvenance:
    directory_chain = _open_bundle_directory(root_fd, relative)
    bundle_fd = directory_chain[-1]
    provenance_fd: int | None = None
    manifest_fd: int | None = None
    try:
        _recheck_bundle_directory_chain(root_fd, relative, directory_chain)
        provenance_fd = _open_regular_file_at(
            bundle_fd,
            "run_provenance.json",
            error_message="host-state bundle provenance is unreadable",
        )
        provenance_bytes = _read_pinned_regular_file(
            provenance_fd,
            error_message="host-state bundle provenance is unreadable",
        )
        manifest_fd = _open_regular_file_at(
            bundle_fd,
            "manifest.json",
            error_message="host-state bundle manifest is unreadable",
        )
        manifest_bytes = _read_pinned_regular_file(
            manifest_fd,
            error_message="host-state bundle manifest is unreadable",
        )
        _recheck_bundle_directory_chain(root_fd, relative, directory_chain)
        _recheck_named_regular_file(
            bundle_fd,
            "run_provenance.json",
            provenance_fd,
            error_message="host-state bundle provenance identity changed",
        )
        _recheck_named_regular_file(
            bundle_fd,
            "manifest.json",
            manifest_fd,
            error_message="host-state bundle manifest identity changed",
        )
        if (
            _read_pinned_regular_file(
                provenance_fd,
                error_message="host-state bundle provenance changed during validation",
            )
            != provenance_bytes
            or _read_pinned_regular_file(
                manifest_fd,
                error_message="host-state bundle manifest changed during validation",
            )
            != manifest_bytes
        ):
            raise DurabilityError(
                "host-state bundle provenance or manifest changed during validation"
            )
        _recheck_named_regular_file(
            bundle_fd,
            "run_provenance.json",
            provenance_fd,
            error_message="host-state bundle provenance identity changed",
        )
        _recheck_named_regular_file(
            bundle_fd,
            "manifest.json",
            manifest_fd,
            error_message="host-state bundle manifest identity changed",
        )
        _recheck_bundle_directory_chain(root_fd, relative, directory_chain)
        try:
            provenance_value = strict_json_loads(provenance_bytes)
            provenance = parse_run_provenance(provenance_value)
            manifest = strict_json_loads(manifest_bytes)
        except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
            raise DurabilityError("host-state bundle provenance is invalid") from error
        if (
            type(manifest) is not dict
            or manifest.keys() != {"schema_version", "files"}
            or type(manifest.get("schema_version")) is not int
            or manifest["schema_version"] != 1
            or type(manifest.get("files")) is not dict
            or manifest["files"].get("run_provenance.json")
            != hashlib.sha256(provenance_bytes).hexdigest()
        ):
            raise DurabilityError("host-state bundle provenance is not manifested")
        _recheck_bundle_directory_chain(root_fd, relative, directory_chain)
        return provenance
    finally:
        for descriptor in (
            manifest_fd,
            provenance_fd,
            *reversed(directory_chain),
        ):
            if descriptor is not None:
                _close_descriptor(descriptor)


def _open_bundle_directory(root_fd: int, relative: Path) -> tuple[int, ...]:
    descriptors: list[int] = []
    try:
        descriptors.append(os.dup(root_fd))
        if not stat.S_ISDIR(os.fstat(descriptors[0]).st_mode):
            raise DurabilityError("host-state bundle path must be a real directory")
        for part in relative.parts:
            descriptor = os.open(
                part,
                _directory_open_flags(),
                dir_fd=descriptors[-1],
            )
            descriptors.append(descriptor)
            if not stat.S_ISDIR(os.fstat(descriptor).st_mode):
                raise DurabilityError(
                    "host-state bundle path must be a real directory"
                )
        return tuple(descriptors)
    except OSError as error:
        close_failures = tuple(
            descriptor
            for descriptor in reversed(descriptors)
            if not _close_descriptor(descriptor)
        )
        if close_failures:
            raise DurabilityError("host-state bundle descriptor close failed") from error
        raise DurabilityError("host-state bundle path must be a real directory") from error
    except BaseException as error:
        close_failures = tuple(
            descriptor
            for descriptor in reversed(descriptors)
            if not _close_descriptor(descriptor)
        )
        if close_failures:
            raise DurabilityError(
                "host-state bundle descriptor close failed"
            ) from error
        raise


def _recheck_bundle_directory_chain(
    root_fd: int,
    relative: Path,
    descriptors: tuple[int, ...],
) -> None:
    if len(descriptors) != len(relative.parts) + 1:
        raise DurabilityError("host-state bundle directory chain is invalid")
    try:
        pinned_root = os.fstat(root_fd)
        opened_root = os.fstat(descriptors[0])
        if (
            not stat.S_ISDIR(pinned_root.st_mode)
            or not stat.S_ISDIR(opened_root.st_mode)
            or (pinned_root.st_dev, pinned_root.st_ino)
            != (opened_root.st_dev, opened_root.st_ino)
        ):
            raise DurabilityError("host-state bundle directory identity changed")
        for part, parent_fd, opened_fd in zip(
            relative.parts,
            descriptors[:-1],
            descriptors[1:],
            strict=True,
        ):
            opened = os.fstat(opened_fd)
            named = os.stat(part, dir_fd=parent_fd, follow_symlinks=False)
            if (
                not stat.S_ISDIR(opened.st_mode)
                or not stat.S_ISDIR(named.st_mode)
                or (opened.st_dev, opened.st_ino) != (named.st_dev, named.st_ino)
            ):
                raise DurabilityError("host-state bundle directory identity changed")
    except OSError as error:
        raise DurabilityError("host-state bundle directory identity changed") from error


def _open_regular_file_at(
    directory_fd: int,
    name: str,
    *,
    error_message: str,
) -> int:
    nofollow = getattr(os, "O_NOFOLLOW", None)
    if nofollow is None:
        raise DurabilityError("platform lacks no-follow file opens")
    descriptor: int | None = None
    try:
        descriptor = os.open(
            name,
            os.O_RDONLY | os.O_NONBLOCK | nofollow,
            dir_fd=directory_fd,
        )
        metadata = os.fstat(descriptor)
        if (
            not stat.S_ISREG(metadata.st_mode)
            or metadata.st_nlink != 1
            or metadata.st_size > _MAX_LEDGER_RECORD_BYTES
        ):
            raise DurabilityError(error_message)
        return descriptor
    except OSError as error:
        if descriptor is not None:
            owned_descriptor = descriptor
            descriptor = None
        else:
            owned_descriptor = None
        if owned_descriptor is not None and not _close_descriptor(owned_descriptor):
            raise DurabilityError(
                "host-state bundle descriptor close failed"
            ) from error
        raise DurabilityError(error_message) from error
    except BaseException as error:
        if descriptor is not None:
            owned_descriptor = descriptor
            descriptor = None
        else:
            owned_descriptor = None
        if owned_descriptor is not None and not _close_descriptor(owned_descriptor):
            raise DurabilityError(
                "host-state bundle descriptor close failed"
            ) from error
        raise


def _read_pinned_regular_file(
    descriptor: int,
    *,
    error_message: str,
) -> bytes:
    try:
        before = os.fstat(descriptor)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_nlink != 1
            or before.st_size > _MAX_LEDGER_RECORD_BYTES
        ):
            raise DurabilityError(error_message)
        os.lseek(descriptor, 0, os.SEEK_SET)
        chunks: list[bytes] = []
        remaining = before.st_size + 1
        while remaining > 0:
            chunk = os.read(descriptor, min(_LEDGER_READ_CHUNK_BYTES, remaining))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        payload = b"".join(chunks)
        after = os.fstat(descriptor)
        if (
            len(payload) != before.st_size
            or not stat.S_ISREG(after.st_mode)
            or after.st_nlink != 1
            or (before.st_dev, before.st_ino, before.st_size)
            != (after.st_dev, after.st_ino, after.st_size)
        ):
            raise DurabilityError(error_message)
        return payload
    except OSError as error:
        raise DurabilityError(error_message) from error


def _recheck_named_regular_file(
    directory_fd: int,
    name: str,
    descriptor: int,
    *,
    error_message: str,
) -> None:
    try:
        opened = os.fstat(descriptor)
        named = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
    except OSError as error:
        raise DurabilityError(error_message) from error
    if (
        not stat.S_ISREG(opened.st_mode)
        or opened.st_nlink != 1
        or not stat.S_ISREG(named.st_mode)
        or named.st_nlink != 1
        or (opened.st_dev, opened.st_ino) != (named.st_dev, named.st_ino)
    ):
        raise DurabilityError(error_message)


def _close_descriptor(descriptor: int) -> bool:
    """Close once; an error leaves numeric-descriptor ownership ambiguous."""
    try:
        os.close(descriptor)
        return True
    except OSError:
        try:
            os.fstat(descriptor)
        except OSError as probe_error:
            return probe_error.errno == errno.EBADF
        return False


def _validate_existing_ledger(
    descriptor: int,
    stable_root: Path,
    root_fd: int,
) -> None:
    """Fail closed unless each prior byte is one canonical operational receipt."""
    pending = bytearray()
    try:
        os.lseek(descriptor, 0, os.SEEK_SET)
        while True:
            remaining = _MAX_LEDGER_RECORD_BYTES + 1 - len(pending)
            if remaining <= 0:
                raise DurabilityError("attempts ledger record exceeds the byte limit")
            chunk = os.read(descriptor, min(_LEDGER_READ_CHUNK_BYTES, remaining))
            if not chunk:
                break
            pending.extend(chunk)
            while (newline := pending.find(b"\n")) >= 0:
                if newline > _MAX_LEDGER_RECORD_BYTES:
                    raise DurabilityError("attempts ledger record exceeds the byte limit")
                line = bytes(pending[:newline])
                del pending[: newline + 1]
                _validate_existing_record(line, stable_root, root_fd)
            if len(pending) > _MAX_LEDGER_RECORD_BYTES:
                raise DurabilityError("attempts ledger record exceeds the byte limit")
    except OSError as error:
        raise DurabilityError("attempts ledger existing records are unreadable") from error
    finally:
        with suppress(OSError):
            os.lseek(descriptor, 0, os.SEEK_END)
    if pending:
        raise DurabilityError("attempts ledger existing records are not canonical JSONL")


def _validate_existing_record(line: bytes, stable_root: Path, root_fd: int) -> None:
    try:
        record = json.loads(line)
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        raise DurabilityError(
            "attempts ledger existing records are not canonical JSONL"
        ) from error
    if not isinstance(record, dict) or _canonical_record_bytes(record) != line:
        raise DurabilityError("attempts ledger existing records are not canonical JSONL")
    _validate_record_schema(record, stable_root, root_fd)


def _canonical_record_bytes(record: dict[str, object]) -> bytes:
    return json.dumps(
        record, ensure_ascii=True, separators=(",", ":"), sort_keys=True
    ).encode("utf-8")


def _validate_record_schema(
    record: dict[str, object],
    stable_root: Path,
    root_fd: int,
) -> None:
    if set(record) != {
        "clean_control_floor",
        "parent_path",
        "timestamp_utc",
        "verdict",
        "workload",
    }:
        raise DurabilityError("attempts ledger existing records have an invalid schema")
    _finite_floor(record["clean_control_floor"])
    _nonempty_string(record["workload"], "workload")
    _nonempty_string(record["verdict"], "verdict")
    timestamp = record["timestamp_utc"]
    if not isinstance(timestamp, str) or not re.fullmatch(
        r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{6}Z", timestamp
    ):
        raise DurabilityError("attempts ledger existing records have an invalid timestamp")
    try:
        parsed = datetime.fromisoformat(timestamp.removesuffix("Z") + "+00:00")
    except ValueError as error:
        raise DurabilityError(
            "attempts ledger existing records have an invalid timestamp"
        ) from error
    if parsed.tzinfo is not UTC:
        raise DurabilityError("attempts ledger existing records have an invalid timestamp")
    raw_parent = record["parent_path"]
    if not isinstance(raw_parent, str):
        raise DurabilityError("attempts ledger existing records have an invalid parent path")
    parent = Path(raw_parent)
    if not parent.is_absolute() or str(parent) != raw_parent:
        raise DurabilityError("attempts ledger existing records have an invalid parent path")
    try:
        relative = parent.relative_to(stable_root)
    except ValueError as error:
        raise DurabilityError(
            "attempts ledger existing records have an invalid parent path"
        ) from error
    if not relative.parts or any(part in {"", ".", ".."} for part in relative.parts):
        raise DurabilityError("attempts ledger existing records have an invalid parent path")
    try:
        _validated_parent_path(
            root_fd,
            relative,
            error_message="attempts ledger existing records have an invalid parent path",
        )
    except DurabilityError as error:
        cause = error.__cause__
        if not isinstance(cause, OSError) or cause.errno != errno.ENOENT:
            raise


def _validated_parent_path(
    root_fd: int,
    relative: Path,
    *,
    error_message: str,
) -> Path:
    """Walk one parent beneath the pinned root without following any component link."""
    parent_fd = os.dup(root_fd)
    try:
        for part in relative.parts:
            next_fd = os.open(part, _directory_open_flags(), dir_fd=parent_fd)
            _close_descriptor(parent_fd)
            parent_fd = next_fd
        metadata = os.fstat(parent_fd)
        if not stat.S_ISDIR(metadata.st_mode):
            raise DurabilityError(error_message)
        return _path_for_open_directory(parent_fd)
    except OSError as error:
        raise DurabilityError(error_message) from error
    finally:
        _close_descriptor(parent_fd)


def _rollback_ledger_append(descriptor: int, offset: int) -> None:
    """Restore the locked ledger to its prior EOF after an incomplete append."""
    try:
        os.ftruncate(descriptor, offset)
    except OSError as error:
        raise DurabilityError("attempts ledger append rollback failed") from error


def _reject_volatile(root: Path, temporary: Path, allow_volatile_output: bool) -> None:
    if not allow_volatile_output and (root == temporary or temporary in root.parents):
        raise VolatileOutputRootError(
            "--output-root resolves under the volatile system temporary directory; "
            "pass --allow-volatile-output to override"
        )


def _finite_floor(value: float | None) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise DurabilityError("clean-control floor must be a finite number or null")
    try:
        floor = float(value)
    except (OverflowError, ValueError) as error:
        raise DurabilityError("clean-control floor must be a finite number or null") from error
    if not math.isfinite(floor):
        raise DurabilityError("clean-control floor must be a finite number or null")
    return floor


def _nonempty_string(value: str, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise DurabilityError(f"{name} must be a nonempty string")
    return value


def _timestamp_utc() -> str:
    return datetime.now(UTC).isoformat(timespec="microseconds").replace("+00:00", "Z")


__all__ = [
    "AttemptOutputRoot",
    "DurabilityError",
    "HostStateOutputRoot",
    "VolatileOutputRootError",
    "append_attempt_record",
    "prepare_attempt_output_root",
    "prepare_host_state_output_root",
    "resolve_durable_output_root",
]

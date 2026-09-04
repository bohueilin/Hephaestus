from __future__ import annotations

import errno
import json
import os
import stat
import threading
from contextlib import suppress
from datetime import UTC, datetime
from pathlib import Path
from queue import Queue

import pytest

import hephaestus.durability as durability
from hephaestus.bundle import write_json, write_manifest
from hephaestus.host_state import capture_operation
from hephaestus.provenance import RunProvenance


class _HelperValidationAbort(BaseException):
    pass


def _snapshot() -> dict[str, object]:
    return {
        "load_average": {"value": [0.1, 0.2, 0.3], "unavailable_reason": None},
    }


def _provenance(run_id: str = "1" * 64) -> RunProvenance:
    return RunProvenance(
        orchestration_id="2" * 64,
        run_id=run_id,
        sequence_index=0,
        predecessor=None,
    )


def _capture(provenance: RunProvenance):
    _, capture = capture_operation(
        lambda: None,
        provenance,
        sampler=lambda: _snapshot(),
        monotonic=iter((10.0, 11.25)).__next__,
        utc_now=iter(
            (
                datetime(2026, 8, 31, 10, 0, 0, tzinfo=UTC),
                datetime(2026, 8, 31, 10, 0, 2, tzinfo=UTC),
            )
        ).__next__,
    )
    return capture


def _bundle(root: Path, provenance: RunProvenance, *, name: str = "child") -> Path:
    child = root / "runs" / name
    return _bundle_at(child, provenance)


def _bundle_at(child: Path, provenance: RunProvenance) -> Path:
    child.mkdir(parents=True)
    write_json(child / "run_provenance.json", provenance.as_json())
    write_manifest(child)
    return child


def _standalone(root: Path):
    factory = getattr(durability, "prepare_host_state_output_root", None)
    assert callable(factory), "missing standalone host-state output-root writer"
    return factory(root)


@pytest.mark.parametrize("kind", ("symlink", "hardlink", "fifo", "directory"))
def test_host_ledger_rejects_unsafe_existing_targets_without_mutation(
    tmp_path: Path,
    kind: str,
) -> None:
    """Following or blocking on a non-single-link regular target would corrupt other state."""
    root = tmp_path / "root"
    root.mkdir()
    target = root / "host_state.jsonl"
    protected = tmp_path / "protected"
    protected.write_bytes(b"protected")
    if kind == "symlink":
        target.symlink_to(protected)
    elif kind == "hardlink":
        os.link(protected, target)
    elif kind == "fifo":
        os.mkfifo(target)
    else:
        target.mkdir()

    with pytest.raises(ValueError, match="host.state|ledger"), _standalone(root):
        pass

    assert protected.read_bytes() == b"protected"


def test_host_ledger_append_is_one_locked_write_and_standalone_creates_no_attempts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A standalone host writer emits one complete row and grants no attempts-ledger authority."""
    provenance = _provenance()
    root = tmp_path / "root"
    child = _bundle(root, provenance)
    writes: list[bytes] = []
    real_write = durability.os.write

    def recording_write(descriptor: int, payload: bytes) -> int:
        writes.append(payload)
        return real_write(descriptor, payload)

    monkeypatch.setattr(durability.os, "write", recording_write)
    with _standalone(root) as sink:
        sink.append(_capture(provenance), child)

    assert len(writes) == 1
    assert writes[0].endswith(b"\n")
    assert (root / "host_state.jsonl").read_bytes() == writes[0]
    assert not (root / "attempts.jsonl").exists()




@pytest.mark.parametrize("mutation", ("replacement", "new-hardlink"))
def test_host_ledger_rechecks_open_descriptor_identity_and_link_count_under_lock(
    tmp_path: Path,
    mutation: str,
) -> None:
    """A post-open path replacement or added link cannot redirect or alias the append."""
    provenance = _provenance()
    root = tmp_path / "root"
    child = _bundle(root, provenance)
    ledger = root / "host_state.jsonl"
    with _standalone(root) as sink:
        if mutation == "replacement":
            moved = root / "opened-ledger"
            ledger.rename(moved)
            ledger.write_bytes(b"replacement")
            protected_paths = (moved, ledger)
        else:
            alias = tmp_path / "host-ledger-alias"
            os.link(ledger, alias)
            protected_paths = (ledger, alias)
        before = tuple(path.read_bytes() for path in protected_paths)

        with pytest.raises(ValueError, match="identity|link|ledger"):
            sink.append(_capture(provenance), child)

        assert tuple(path.read_bytes() for path in protected_paths) == before


def test_host_ledger_requires_manifested_provenance_to_match_capture(
    tmp_path: Path,
) -> None:
    """A row cannot bind a capture to a child whose manifested run identity differs."""
    captured = _provenance("1" * 64)
    stored = _provenance("3" * 64)
    root = tmp_path / "root"
    child = _bundle(root, stored)
    before = {
        path.relative_to(child).as_posix(): path.read_bytes()
        for path in child.rglob("*")
        if path.is_file()
    }

    with _standalone(root) as sink, pytest.raises(ValueError, match="provenance"):
        sink.append(_capture(captured), child)

    after = {
        path.relative_to(child).as_posix(): path.read_bytes()
        for path in child.rglob("*")
        if path.is_file()
    }
    assert after == before
    assert (root / "host_state.jsonl").read_bytes() == b""


@pytest.mark.parametrize("owner_kind", ("standalone", "attempt"))
def test_close_waits_for_inflight_append_before_releasing_owned_descriptors(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    owner_kind: str,
) -> None:
    """Concurrent close cannot release and reuse an fd still needed by append rollback."""
    provenance = _provenance()
    root = tmp_path / "root"
    child = _bundle(root, provenance)
    foreign_path = tmp_path / "foreign"
    foreign_payload = b"foreign-file-must-not-be-truncated"
    foreign_path.write_bytes(foreign_payload)
    foreign_source_fd = os.open(foreign_path, os.O_RDWR)
    if owner_kind == "standalone":
        sink = _standalone(root)
    else:
        sink = durability.prepare_attempt_output_root(
            root,
            allow_volatile_output=True,
        )
    sink.__enter__()
    ledger_fd = sink._host_ledger_fd
    root_fd = sink._root_fd
    assert isinstance(ledger_fd, int)
    assert isinstance(root_fd, int)
    assert ledger_fd != foreign_source_fd
    write_paused = threading.Event()
    release_write = threading.Event()
    close_started = threading.Event()
    close_finished = threading.Event()
    append_outcomes: Queue[BaseException | None] = Queue()
    close_outcomes: Queue[tuple[int, ...] | BaseException] = Queue()
    real_write = durability.os.write
    reused_ledger_fd = False

    def pause_after_write(descriptor: int, payload: bytes) -> int:
        written = real_write(descriptor, payload)
        write_paused.set()
        release_write.wait(2.0)
        return written

    def append() -> None:
        try:
            sink.append(_capture(provenance), child)
        except BaseException as error:
            append_outcomes.put(error)
        else:
            append_outcomes.put(None)

    def close() -> None:
        close_started.set()
        try:
            close_outcomes.put(sink.close())
        except BaseException as error:
            close_outcomes.put(error)
        finally:
            close_finished.set()

    append_thread = threading.Thread(target=append, name="host-append")
    close_thread = threading.Thread(target=close, name="host-close")
    monkeypatch.setattr(durability.os, "write", pause_after_write)
    try:
        append_thread.start()
        assert write_paused.wait(1.0)
        close_thread.start()
        assert close_started.wait(1.0)
        closed_while_write_paused = close_finished.wait(0.25)
        if closed_while_write_paused:
            os.dup2(foreign_source_fd, ledger_fd)
            reused_ledger_fd = True
        release_write.set()
        append_thread.join(2.0)
        close_thread.join(2.0)
        assert not append_thread.is_alive() and not close_thread.is_alive()
    finally:
        release_write.set()
        append_thread.join(2.0)
        close_thread.join(2.0)
        if reused_ledger_fd:
            with suppress(OSError):
                os.close(ledger_fd)
        sink.close()
        os.close(foreign_source_fd)

    append_outcome = append_outcomes.get_nowait()
    close_outcome = close_outcomes.get_nowait()
    assert (
        closed_while_write_paused,
        append_outcome,
        close_outcome,
        foreign_path.read_bytes(),
    ) == (False, None, (), foreign_payload)
    rows = (root / "host_state.jsonl").read_bytes().splitlines()
    assert len(rows) == 1
    assert json.loads(rows[0])["run_provenance"]["run_id"] == provenance.run_id




@pytest.mark.parametrize("mutation", ("replacement", "in-place"))
def test_manifested_provenance_read_rejects_split_view_pair(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
) -> None:
    """Cached p1 bytes cannot be validated against a manifest after the name becomes p2."""
    captured = _provenance("1" * 64)
    swapped = _provenance("3" * 64)
    root = tmp_path / "root"
    child = _bundle(root, captured)
    provenance_path = child / "run_provenance.json"
    displaced = child / "displaced-run-provenance.json"
    original_size = provenance_path.stat().st_size
    real_open = durability.os.open
    mutated = False

    def open_after_swap(
        path: str | bytes | os.PathLike[str] | os.PathLike[bytes],
        flags: int,
        mode: int = 0o777,
        *,
        dir_fd: int | None = None,
    ) -> int:
        nonlocal mutated
        if path == "manifest.json" and not mutated:
            mutated = True
            if mutation == "replacement":
                provenance_path.rename(displaced)
            write_json(provenance_path, swapped.as_json())
            assert provenance_path.stat().st_size == original_size
        return real_open(path, flags, mode, dir_fd=dir_fd)

    with _standalone(root) as sink:
        monkeypatch.setattr(durability.os, "open", open_after_swap)
        with pytest.raises(ValueError, match="provenance|manifest|changed|identity"):
            sink.append(_capture(captured), child)

    assert mutated
    assert (root / "host_state.jsonl").read_bytes() == b""














def test_ambiguous_close_never_retries_a_reused_numeric_descriptor(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An errored close that freed its fd must not close a foreign replacement on retry."""
    owned = os.open(tmp_path / "owned", os.O_RDWR | os.O_CREAT, 0o600)
    replacement_path = tmp_path / "replacement"
    replacement_path.write_bytes(b"replacement")
    replacement: list[int] = []
    calls = 0
    real_close = durability.os.close
    real_open = durability.os.open

    def ambiguous_close(descriptor: int) -> None:
        nonlocal calls
        calls += 1
        if calls == 1:
            real_close(descriptor)
            reused = real_open(replacement_path, os.O_RDONLY)
            assert reused == descriptor
            replacement.append(reused)
            raise OSError("ambiguous close result")
        real_close(descriptor)

    monkeypatch.setattr(durability.os, "close", ambiguous_close)
    try:
        assert durability._close_descriptor(owned) is False
        assert calls == 1
        assert os.fstat(replacement[0]).st_size == len(b"replacement")
    finally:
        for descriptor in replacement:
            with suppress(OSError):
                real_close(descriptor)


def test_close_eio_plus_fstat_eio_is_not_misreported_as_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Only EBADF proves closure; unrelated probe failure leaves ownership ambiguous."""
    owned = os.open(tmp_path / "owned", os.O_RDWR | os.O_CREAT, 0o600)
    real_close = durability.os.close
    real_fstat = durability.os.fstat
    close_calls = 0

    def fail_close(descriptor: int) -> None:
        nonlocal close_calls
        if descriptor == owned:
            close_calls += 1
            raise OSError(errno.EIO, "injected close failure")
        real_close(descriptor)

    def fail_probe(descriptor: int) -> os.stat_result:
        if descriptor == owned:
            raise OSError(errno.EIO, "injected fstat failure")
        return real_fstat(descriptor)

    monkeypatch.setattr(durability.os, "close", fail_close)
    monkeypatch.setattr(durability.os, "fstat", fail_probe)
    try:
        assert durability._close_descriptor(owned) is False
        assert close_calls == 1
        assert stat.S_ISREG(real_fstat(owned).st_mode)
    finally:
        with suppress(OSError):
            real_close(owned)


def test_host_ledger_setup_surfaces_unassigned_descriptor_cleanup_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Failure before host-fd assignment cannot discard its failed local cleanup."""
    sink = _standalone(tmp_path / "root")
    opened_host_fds: list[int] = []
    close_calls: list[int] = []
    metadata_failure = OSError(errno.EIO, "injected host metadata failure")
    real_open = durability.os.open
    real_close = durability.os.close
    real_fstat = durability.os.fstat
    real_close_descriptor = durability._close_descriptor

    def record_host_open(
        path: str | bytes | os.PathLike[str] | os.PathLike[bytes],
        flags: int,
        mode: int = 0o777,
        *,
        dir_fd: int | None = None,
    ) -> int:
        descriptor = real_open(path, flags, mode, dir_fd=dir_fd)
        if path == "host_state.jsonl":
            opened_host_fds.append(descriptor)
        return descriptor

    def fail_host_metadata(descriptor: int) -> os.stat_result:
        if opened_host_fds and descriptor == opened_host_fds[0]:
            raise metadata_failure
        return real_fstat(descriptor)

    def fail_host_cleanup(descriptor: int) -> bool:
        close_calls.append(descriptor)
        if opened_host_fds and descriptor == opened_host_fds[0]:
            return False
        return real_close_descriptor(descriptor)

    monkeypatch.setattr(durability.os, "open", record_host_open)
    monkeypatch.setattr(durability.os, "fstat", fail_host_metadata)
    monkeypatch.setattr(durability, "_close_descriptor", fail_host_cleanup)
    try:
        with pytest.raises(
            durability.DurabilityError,
            match="cleanup|close|descriptor",
        ) as caught:
            sink.__enter__()

        assert caught.value.__cause__ is metadata_failure
        assert len(opened_host_fds) == 1
        assert close_calls[0] == opened_host_fds[0]
        assert len(close_calls) == 2
        assert len(set(close_calls)) == 2
        assert sink._host_ledger_fd is None
        assert sink._root_fd is None
        calls_before_reclose = tuple(close_calls)
        assert sink.close() == ()
        assert tuple(close_calls) == calls_before_reclose
        assert stat.S_ISREG(real_fstat(opened_host_fds[0]).st_mode)
    finally:
        for descriptor in opened_host_fds:
            with suppress(OSError):
                real_close(descriptor)


def test_attempt_ledger_setup_surfaces_unassigned_descriptor_cleanup_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A failed local attempts-fd close cannot be hidden before attribute assignment."""
    sink = durability.prepare_attempt_output_root(
        tmp_path / "root",
        allow_volatile_output=True,
    )
    opened_attempt_fds: list[int] = []
    close_calls: list[int] = []
    metadata_failure = OSError(errno.EIO, "injected attempts metadata failure")
    real_open = durability.os.open
    real_close = durability.os.close
    real_fstat = durability.os.fstat
    real_close_descriptor = durability._close_descriptor

    def record_attempt_open(
        path: str | bytes | os.PathLike[str] | os.PathLike[bytes],
        flags: int,
        mode: int = 0o777,
        *,
        dir_fd: int | None = None,
    ) -> int:
        descriptor = real_open(path, flags, mode, dir_fd=dir_fd)
        if path == "attempts.jsonl":
            opened_attempt_fds.append(descriptor)
        return descriptor

    def fail_attempt_metadata(descriptor: int) -> os.stat_result:
        if opened_attempt_fds and descriptor == opened_attempt_fds[0]:
            raise metadata_failure
        return real_fstat(descriptor)

    def fail_attempt_cleanup(descriptor: int) -> bool:
        close_calls.append(descriptor)
        if opened_attempt_fds and descriptor == opened_attempt_fds[0]:
            return False
        return real_close_descriptor(descriptor)

    monkeypatch.setattr(durability.os, "open", record_attempt_open)
    monkeypatch.setattr(durability.os, "fstat", fail_attempt_metadata)
    monkeypatch.setattr(durability, "_close_descriptor", fail_attempt_cleanup)
    try:
        with pytest.raises(
            durability.DurabilityError,
            match="cleanup|close|descriptor",
        ) as caught:
            sink.__enter__()

        assert caught.value.__cause__ is metadata_failure
        assert len(opened_attempt_fds) == 1
        assert close_calls[0] == opened_attempt_fds[0]
        assert len(close_calls) == 2
        assert len(set(close_calls)) == 2
        assert sink._host_ledger_fd is None
        assert sink._ledger_fd is None
        assert sink._root_fd is None
        calls_before_reclose = tuple(close_calls)
        assert sink.close() == ()
        assert tuple(close_calls) == calls_before_reclose
        assert stat.S_ISREG(real_fstat(opened_attempt_fds[0]).st_mode)
    finally:
        for descriptor in opened_attempt_fds:
            with suppress(OSError):
                real_close(descriptor)


@pytest.mark.parametrize("ledger_name", ("attempts.jsonl", "host_state.jsonl"))
@pytest.mark.parametrize(
    "cleanup_failure",
    (False, True),
    ids=("cleanup-succeeds", "cleanup-fails"),
)
def test_ledger_open_helpers_clean_local_fd_on_baseexception_before_transfer(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    ledger_name: str,
    cleanup_failure: bool,
) -> None:
    """A helper-owned ledger fd is cleaned before control flow can leave the helper."""
    root = tmp_path / "root"
    root.mkdir()
    ledger = root / ledger_name
    prior_bytes = b"preexisting-ledger-bytes-must-survive\n"
    ledger.write_bytes(prior_bytes)
    if ledger_name == "attempts.jsonl":
        sink = durability.prepare_attempt_output_root(
            root,
            allow_volatile_output=True,
        )
        attributes = ("_host_ledger_fd", "_ledger_fd", "_root_fd")
    else:
        sink = _standalone(root)
        attributes = ("_host_ledger_fd", "_root_fd")

    sentinel = _HelperValidationAbort(f"{ledger_name} validation abort")
    opened_local_fds: list[int] = []
    owner_fds: list[int] = []
    close_calls: list[int] = []
    failed_local_fds: list[int] = []
    injected = False
    real_open = durability.os.open
    real_close = durability.os.close
    real_fstat = durability.os.fstat
    real_close_descriptor = durability._close_descriptor

    def record_ledger_open(
        path: str | bytes | os.PathLike[str] | os.PathLike[bytes],
        flags: int,
        mode: int = 0o777,
        *,
        dir_fd: int | None = None,
    ) -> int:
        descriptor = real_open(path, flags, mode, dir_fd=dir_fd)
        if path == ledger_name:
            opened_local_fds.append(descriptor)
        return descriptor

    def abort_local_validation(descriptor: int) -> os.stat_result:
        nonlocal injected
        if opened_local_fds and descriptor == opened_local_fds[0] and not injected:
            injected = True
            values = [getattr(sink, attribute) for attribute in attributes]
            owner_fds.extend(value for value in values if isinstance(value, int))
            raise sentinel
        return real_fstat(descriptor)

    def record_cleanup(descriptor: int) -> bool:
        close_calls.append(descriptor)
        if (
            cleanup_failure
            and opened_local_fds
            and descriptor == opened_local_fds[0]
            and not failed_local_fds
        ):
            failed_local_fds.append(descriptor)
            return False
        return real_close_descriptor(descriptor)

    monkeypatch.setattr(durability.os, "open", record_ledger_open)
    monkeypatch.setattr(durability.os, "fstat", abort_local_validation)
    monkeypatch.setattr(durability, "_close_descriptor", record_cleanup)
    expected_error = (
        durability.DurabilityError if cleanup_failure else _HelperValidationAbort
    )
    try:
        with pytest.raises(expected_error) as caught:
            sink.__enter__()

        if cleanup_failure:
            assert caught.value.__cause__ is sentinel
        else:
            assert caught.value is sentinel
        assert injected
        assert len(opened_local_fds) == 1
        assert owner_fds and len(owner_fds) == len(set(owner_fds))
        assert close_calls == [opened_local_fds[0], *owner_fds]
        assert all(getattr(sink, attribute) is None for attribute in attributes)
        calls_before_reclose = tuple(close_calls)
        assert sink.close() == ()
        assert tuple(close_calls) == calls_before_reclose
        assert ledger.read_bytes() == prior_bytes
        if cleanup_failure:
            assert failed_local_fds == opened_local_fds
            assert stat.S_ISREG(real_fstat(opened_local_fds[0]).st_mode)
        else:
            with pytest.raises(OSError):
                real_fstat(opened_local_fds[0])
        for descriptor in owner_fds:
            with pytest.raises(OSError):
                real_fstat(descriptor)
    finally:
        sink.close()
        for descriptor in (*opened_local_fds, *owner_fds):
            with suppress(OSError):
                real_close(descriptor)


@pytest.mark.parametrize(
    "validation_stage",
    ("root-dup", "intermediate", "final"),
)
@pytest.mark.parametrize(
    "cleanup_failure",
    (False, True),
    ids=("cleanup-succeeds", "cleanup-fails"),
)
def test_bundle_directory_helper_cleans_every_local_fd_on_baseexception(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    validation_stage: str,
    cleanup_failure: bool,
) -> None:
    """Every partially opened directory chain is closed before ownership can escape."""
    prior = _provenance("4" * 64)
    captured = _provenance("1" * 64)
    root = tmp_path / "root"
    prior_child = _bundle(root, prior, name="prior")
    child = _bundle_at(root / "parent" / "runs" / "child", captured)
    ledger = root / "host_state.jsonl"
    with _standalone(root) as seed_sink:
        seed_sink.append(_capture(prior), prior_child)
    prior_ledger_bytes = ledger.read_bytes()
    assert prior_ledger_bytes

    sink = _standalone(root)
    sink.__enter__()
    owner_fds = (sink._host_ledger_fd, sink._root_fd)
    assert all(isinstance(descriptor, int) for descriptor in owner_fds)
    sentinel = _HelperValidationAbort(f"{validation_stage} directory abort")
    helper_fds: list[int] = []
    validation_fd: list[int] = []
    helper_close_calls: list[int] = []
    failed_helper_fds: list[int] = []
    target_armed = False
    injected = False
    target_relative = Path("parent/runs/child")
    real_close = durability.os.close
    real_close_descriptor = durability._close_descriptor
    real_dup = durability.os.dup
    real_fstat = durability.os.fstat
    real_open = durability.os.open
    real_read = durability._read_manifested_provenance

    def arm_only_target(root_fd: int, relative: Path) -> RunProvenance:
        nonlocal target_armed
        if relative != target_relative:
            return real_read(root_fd, relative)
        assert not target_armed
        target_armed = True
        try:
            return real_read(root_fd, relative)
        finally:
            target_armed = False

    def record_root_dup(descriptor: int) -> int:
        duplicated = real_dup(descriptor)
        if target_armed:
            helper_fds.append(duplicated)
            if validation_stage == "root-dup":
                validation_fd.append(duplicated)
        return duplicated

    def record_component_open(
        path: str | bytes | os.PathLike[str] | os.PathLike[bytes],
        flags: int,
        mode: int = 0o777,
        *,
        dir_fd: int | None = None,
    ) -> int:
        descriptor = real_open(path, flags, mode, dir_fd=dir_fd)
        if target_armed and path in {"parent", "runs", "child"}:
            helper_fds.append(descriptor)
            if (
                validation_stage == "intermediate"
                and path == "runs"
                or validation_stage == "final"
                and path == "child"
            ):
                validation_fd.append(descriptor)
        return descriptor

    def abort_component_validation(descriptor: int) -> os.stat_result:
        nonlocal injected
        if validation_fd and descriptor == validation_fd[0] and not injected:
            injected = True
            raise sentinel
        return real_fstat(descriptor)

    def record_helper_cleanup(descriptor: int) -> bool:
        if descriptor not in helper_fds:
            return real_close_descriptor(descriptor)
        helper_close_calls.append(descriptor)
        if (
            cleanup_failure
            and descriptor == helper_fds[-1]
            and not failed_helper_fds
        ):
            failed_helper_fds.append(descriptor)
            return False
        return real_close_descriptor(descriptor)

    monkeypatch.setattr(
        durability,
        "_read_manifested_provenance",
        arm_only_target,
    )
    monkeypatch.setattr(durability.os, "dup", record_root_dup)
    monkeypatch.setattr(durability.os, "open", record_component_open)
    monkeypatch.setattr(durability.os, "fstat", abort_component_validation)
    monkeypatch.setattr(durability, "_close_descriptor", record_helper_cleanup)
    expected_error = (
        durability.DurabilityError if cleanup_failure else _HelperValidationAbort
    )
    try:
        try:
            with pytest.raises(expected_error) as caught:
                sink.append(_capture(captured), child)
        finally:
            assert sink.close() == ()

        if cleanup_failure:
            assert caught.value.__cause__ is sentinel
        else:
            assert caught.value is sentinel
        assert injected
        assert len(validation_fd) == 1
        assert len(helper_fds) == {
            "root-dup": 1,
            "intermediate": 3,
            "final": 4,
        }[validation_stage]
        assert len(helper_fds) == len(set(helper_fds))
        assert helper_close_calls == list(reversed(helper_fds))
        assert all(
            getattr(sink, attribute) is None
            for attribute in ("_host_ledger_fd", "_root_fd")
        )
        calls_before_reclose = tuple(helper_close_calls)
        assert sink.close() == ()
        assert tuple(helper_close_calls) == calls_before_reclose
        assert ledger.read_bytes() == prior_ledger_bytes
        for descriptor in helper_fds:
            if descriptor in failed_helper_fds:
                assert stat.S_ISDIR(real_fstat(descriptor).st_mode)
            else:
                with pytest.raises(OSError):
                    real_fstat(descriptor)
        for descriptor in owner_fds:
            assert descriptor is not None
            with pytest.raises(OSError):
                real_fstat(descriptor)
    finally:
        sink.close()
        for descriptor in (*helper_fds, *owner_fds):
            if descriptor is not None:
                with suppress(OSError):
                    real_close(descriptor)


@pytest.mark.parametrize("file_name", ("run_provenance.json", "manifest.json"))
@pytest.mark.parametrize(
    "cleanup_failure",
    (False, True),
    ids=("cleanup-succeeds", "cleanup-fails"),
)
def test_regular_file_open_helper_cleans_local_fd_on_baseexception(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    file_name: str,
    cleanup_failure: bool,
) -> None:
    """A provenance or manifest fd is cleaned before its helper transfers ownership."""
    prior = _provenance("4" * 64)
    captured = _provenance("1" * 64)
    root = tmp_path / "root"
    prior_child = _bundle(root, prior, name="prior")
    child = _bundle_at(root / "parent" / "runs" / "child", captured)
    bundle_bytes = {
        path.name: path.read_bytes() for path in child.iterdir() if path.is_file()
    }
    ledger = root / "host_state.jsonl"
    with _standalone(root) as seed_sink:
        seed_sink.append(_capture(prior), prior_child)
    prior_ledger_bytes = ledger.read_bytes()
    assert prior_ledger_bytes

    sink = _standalone(root)
    sink.__enter__()
    owner_fds = (sink._host_ledger_fd, sink._root_fd)
    assert all(isinstance(descriptor, int) for descriptor in owner_fds)
    sentinel = _HelperValidationAbort(f"{file_name} validation abort")
    read_fds: list[int] = []
    local_file_fds: list[int] = []
    read_close_calls: list[int] = []
    failed_local_fds: list[int] = []
    target_armed = False
    injected = False
    target_relative = Path("parent/runs/child")
    real_close = durability.os.close
    real_close_descriptor = durability._close_descriptor
    real_dup = durability.os.dup
    real_fstat = durability.os.fstat
    real_open = durability.os.open
    real_read = durability._read_manifested_provenance

    def arm_only_target(root_fd: int, relative: Path) -> RunProvenance:
        nonlocal target_armed
        if relative != target_relative:
            return real_read(root_fd, relative)
        assert not target_armed
        target_armed = True
        try:
            return real_read(root_fd, relative)
        finally:
            target_armed = False

    def record_root_dup(descriptor: int) -> int:
        duplicated = real_dup(descriptor)
        if target_armed:
            read_fds.append(duplicated)
        return duplicated

    def record_read_open(
        path: str | bytes | os.PathLike[str] | os.PathLike[bytes],
        flags: int,
        mode: int = 0o777,
        *,
        dir_fd: int | None = None,
    ) -> int:
        descriptor = real_open(path, flags, mode, dir_fd=dir_fd)
        if target_armed and path in {
            "parent",
            "runs",
            "child",
            "run_provenance.json",
            "manifest.json",
        }:
            read_fds.append(descriptor)
            if path == file_name:
                local_file_fds.append(descriptor)
        return descriptor

    def abort_file_validation(descriptor: int) -> os.stat_result:
        nonlocal injected
        if local_file_fds and descriptor == local_file_fds[0] and not injected:
            injected = True
            raise sentinel
        return real_fstat(descriptor)

    def record_read_cleanup(descriptor: int) -> bool:
        if descriptor not in read_fds:
            return real_close_descriptor(descriptor)
        read_close_calls.append(descriptor)
        if (
            cleanup_failure
            and local_file_fds
            and descriptor == local_file_fds[0]
            and not failed_local_fds
        ):
            failed_local_fds.append(descriptor)
            return False
        return real_close_descriptor(descriptor)

    monkeypatch.setattr(
        durability,
        "_read_manifested_provenance",
        arm_only_target,
    )
    monkeypatch.setattr(durability.os, "dup", record_root_dup)
    monkeypatch.setattr(durability.os, "open", record_read_open)
    monkeypatch.setattr(durability.os, "fstat", abort_file_validation)
    monkeypatch.setattr(durability, "_close_descriptor", record_read_cleanup)
    expected_error = (
        durability.DurabilityError if cleanup_failure else _HelperValidationAbort
    )
    try:
        try:
            with pytest.raises(expected_error) as caught:
                sink.append(_capture(captured), child)
        finally:
            assert sink.close() == ()

        if cleanup_failure:
            assert caught.value.__cause__ is sentinel
        else:
            assert caught.value is sentinel
        assert injected
        assert len(local_file_fds) == 1
        assert len(read_fds) == {
            "run_provenance.json": 5,
            "manifest.json": 6,
        }[file_name]
        assert len(read_fds) == len(set(read_fds))
        assert read_close_calls == list(reversed(read_fds))
        assert all(
            getattr(sink, attribute) is None
            for attribute in ("_host_ledger_fd", "_root_fd")
        )
        calls_before_reclose = tuple(read_close_calls)
        assert sink.close() == ()
        assert tuple(read_close_calls) == calls_before_reclose
        assert ledger.read_bytes() == prior_ledger_bytes
        assert {
            path.name: path.read_bytes() for path in child.iterdir() if path.is_file()
        } == bundle_bytes
        for descriptor in read_fds:
            if descriptor in failed_local_fds:
                assert stat.S_ISREG(real_fstat(descriptor).st_mode)
            else:
                with pytest.raises(OSError):
                    real_fstat(descriptor)
        for descriptor in owner_fds:
            assert descriptor is not None
            with pytest.raises(OSError):
                real_fstat(descriptor)
    finally:
        sink.close()
        for descriptor in (*read_fds, *owner_fds):
            if descriptor is not None:
                with suppress(OSError):
                    real_close(descriptor)


def test_host_only_root_ignores_malformed_attempts_ledger(
    tmp_path: Path,
) -> None:
    """Host-only authority neither validates nor rewrites the separate attempts ledger."""
    provenance = _provenance()
    root = tmp_path / "root"
    child = _bundle(root, provenance)
    malformed = b'{"workload":"mlp_stack"}\n'
    (root / "attempts.jsonl").write_bytes(malformed)

    with _standalone(root) as sink:
        sink.append(_capture(provenance), child)

    assert (root / "attempts.jsonl").read_bytes() == malformed
    assert len((root / "host_state.jsonl").read_bytes().splitlines()) == 1

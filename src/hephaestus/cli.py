"""Thin argparse parsing and rendering for the five v0.1 command surfaces."""

from __future__ import annotations

import argparse
import codecs
import errno
import fcntl
import io
import json
import os
import selectors
import subprocess
import sys
import threading
import time
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager, suppress
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

from hephaestus.catalog import CATALOG, WorkloadName, get_catalog_entry
from hephaestus.scope import EVIDENCE_BOUNDARY

_NORMAL_VERDICTS = {"PROVEN", "CONDITIONAL", "NOT_PROVEN"}
_WORKER_STREAM_CHUNK_BYTES = 65_536
_WORKER_CONTROL_POLL_SECONDS = 0.1
_WORKER_CLEANUP_TIMEOUT_SECONDS = 1.0
_DESCRIPTOR_IDENTITY_ATTEMPTS = 3
_WORKER_SCIENCE_START_MARKER = b"hephaestus-durable-worker:science-start:v1\n"
_PREPARED_SCIENCE_ATTRIBUTE = "_hephaestus_prepared_science"


@dataclass(frozen=True, slots=True)
class _PreparedScience:
    criteria_path: Path
    target: Callable[..., Any]


@dataclass(frozen=True, slots=True)
class _DescriptorIdentity:
    device: int
    inode: int
    mode: int
    device_type: int
    access_mode: int


class _DescriptorState(Enum):
    CLOSED = "closed"
    SAME = "same"
    REUSED = "reused"
    UNVERIFIABLE = "unverifiable"


class _OwnedControlWriter:
    """Retain an uncertain close until identity-checked final cleanup."""

    def __init__(self, descriptor: int) -> None:
        self._descriptor: int | None = descriptor
        self._identity = _descriptor_identity(descriptor)

    @property
    def descriptor(self) -> int:
        if self._descriptor is None:
            raise RuntimeError("control writer is no longer owned")
        return self._descriptor

    @property
    def is_owned(self) -> bool:
        return self._descriptor is not None

    def close_parent_copy(self) -> None:
        descriptor = self.descriptor
        if _close_worker_descriptor(descriptor):
            self._descriptor = None
            return
        state, _ = self._resolved_state(descriptor)
        if state in {_DescriptorState.CLOSED, _DescriptorState.REUSED}:
            self._descriptor = None

    def finalize(self) -> Exception | None:
        descriptor = self._descriptor
        if descriptor is None:
            return None
        state, identity_error = self._resolved_state(descriptor)
        if state in {_DescriptorState.CLOSED, _DescriptorState.REUSED}:
            self._descriptor = None
            return None
        if state is _DescriptorState.UNVERIFIABLE:
            return identity_error
        # The exact descriptor was identity-checked immediately before this fallback,
        # so a descriptor reused after an uncertain close is never blindly retried.
        cleanup_error = _force_close_worker_descriptor(descriptor)
        if cleanup_error is not None:
            return cleanup_error
        state, identity_error = self._resolved_state(descriptor)
        if state is _DescriptorState.SAME:
            return OSError("owned control writer remained open after final cleanup")
        if state is _DescriptorState.UNVERIFIABLE:
            return identity_error
        self._descriptor = None
        return None

    def _resolved_state(
        self,
        descriptor: int,
    ) -> tuple[_DescriptorState, OSError | None]:
        identity_error: OSError | None = None
        for _ in range(_DESCRIPTOR_IDENTITY_ATTEMPTS):
            state, identity_error = self._ownership_state(descriptor)
            if state is not _DescriptorState.UNVERIFIABLE:
                return state, None
        return _DescriptorState.UNVERIFIABLE, identity_error

    def _ownership_state(
        self,
        descriptor: int,
    ) -> tuple[_DescriptorState, OSError | None]:
        try:
            current = _descriptor_identity(descriptor)
        except OSError as error:
            if error.errno == errno.EBADF:
                return _DescriptorState.CLOSED, None
            return _DescriptorState.UNVERIFIABLE, error
        if current == self._identity:
            return _DescriptorState.SAME, None
        return _DescriptorState.REUSED, None


_RETAINED_CONTROL_WRITERS: list[_OwnedControlWriter] = []
_RETAINED_CONTROL_WRITERS_LOCK = threading.Lock()


class _WorkerControlChannel:
    """Own the control pipe and parent-memory completion state."""

    def __init__(self) -> None:
        retained_error = _retry_retained_control_writers()
        if retained_error is not None:
            raise OSError(
                "retained control writer cleanup remains unverifiable"
            ) from retained_error
        control_read_fd, control_write_fd = os.pipe()
        try:
            writer = _OwnedControlWriter(control_write_fd)
            completion = threading.Event()
        except Exception:
            _force_close_worker_descriptor(control_read_fd)
            _force_close_worker_descriptor(control_write_fd)
            raise
        self._control_read_fd: int | None = control_read_fd
        self._writer: _OwnedControlWriter | None = writer
        self._completion = completion
        self._reader_thread: threading.Thread | None = None
        self._reader_started = False

    @property
    def child_descriptor(self) -> int:
        writer = self._writer
        if writer is None:
            raise RuntimeError("control writer ownership was already transferred")
        return writer.descriptor

    def close_parent_writer(self) -> None:
        writer = self._writer
        if writer is None:
            raise RuntimeError("control writer ownership was already transferred")
        writer.close_parent_copy()

    def start_reader(
        self,
        payloads: list[bytes],
        failures: list[Exception],
    ) -> threading.Thread:
        descriptor = self._control_read_fd
        if descriptor is None or self._reader_thread is not None:
            raise RuntimeError("control reader ownership was already transferred")
        thread = threading.Thread(
            target=_drain_worker_control,
            args=(descriptor, self._completion, payloads, failures),
            daemon=True,
        )
        self._reader_thread = thread
        try:
            thread.start()
        except Exception:
            if thread.ident is None:
                self._reader_thread = None
            else:
                self._reader_started = True
                self._control_read_fd = None
            raise
        self._reader_started = True
        self._control_read_fd = None
        return thread

    def signal_process_complete(self) -> None:
        self._completion.set()

    def finalize_writer(self) -> Exception | None:
        writer = self._writer
        self._writer = None
        if writer is None:
            return None
        cleanup_error = writer.finalize()
        if writer.is_owned:
            _retain_control_writer(writer)
        return cleanup_error

    def join_reader(self, timeout: float | None = None) -> Exception | None:
        thread = self._reader_thread
        if thread is None or not self._reader_started:
            return None
        try:
            thread.join(timeout=timeout)
        except Exception as error:
            return error
        if thread.is_alive():
            return TimeoutError("control reader thread remained alive after cleanup")
        self._reader_thread = None
        return None

    def close_unstarted(self, *, join_timeout: float | None = None) -> Exception | None:
        self.signal_process_complete()
        cleanup_error = self.join_reader(timeout=join_timeout)
        descriptor = self._control_read_fd
        self._control_read_fd = None
        if descriptor is not None:
            descriptor_error = _force_close_worker_descriptor(descriptor)
            if cleanup_error is None:
                cleanup_error = descriptor_error
        writer_error = self.finalize_writer()
        if cleanup_error is None:
            cleanup_error = writer_error
        return cleanup_error


def main(argv: list[str] | None = None) -> int:
    """Parse one command, invoke one trusted workflow, render, and choose an exit code."""
    source_argv = sys.argv[1:] if argv is None else argv
    parser = _parser()
    arguments = parser.parse_args(source_argv)
    if arguments.command in {"demo-planted-regressions", "aa-test"}:
        return _run_durable_attempt(source_argv)
    return _run_after_parse(parser, arguments)


def _run_durable_attempt(argv: list[str]) -> int:
    """Exec a descriptor-owning worker, then forward its exact scientific streams."""
    try:
        control = _WorkerControlChannel()
    except OSError as error:
        print(f"durable worker spawn failure: {error}", file=sys.stderr)
        return 1
    command = [
        sys.executable,
        "-m",
        "hephaestus.durable_worker",
        json.dumps(argv),
        str(control.child_descriptor),
    ]
    try:
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            close_fds=True,
            pass_fds=(control.child_descriptor,),
        )
    except OSError as error:
        cleanup_error = control.close_unstarted()
        detail = (
            f"; control cleanup failure: {cleanup_error}"
            if cleanup_error is not None
            else ""
        )
        print(f"durable worker spawn failure: {error}{detail}", file=sys.stderr)
        return 1
    control.close_parent_writer()
    if process.stdout is None or process.stderr is None:
        cleanup_error = _rollback_durable_worker(
            process,
            control,
            (process.stdout, process.stderr),
            (),
        )
        detail = (
            f"; cleanup failure: {cleanup_error}"
            if cleanup_error is not None
            else ""
        )
        print(
            f"durable worker bootstrap failure: missing worker pipes{detail}",
            file=sys.stderr,
        )
        return 1
    forwarding_failures: list[Exception] = []
    failure_lock = threading.Lock()
    control_payloads: list[bytes] = []
    control_failures: list[Exception] = []
    stream_arguments = (
        (process.stdout, sys.stdout, forwarding_failures, failure_lock),
        (process.stderr, sys.stderr, forwarding_failures, failure_lock),
    )
    stream_threads: list[threading.Thread] = []
    try:
        for arguments in stream_arguments:
            stream_threads.append(
                threading.Thread(
                    target=_forward_worker_stream,
                    args=arguments,
                    daemon=True,
                )
            )
        control.start_reader(control_payloads, control_failures)
        for thread in stream_threads:
            thread.start()
    except Exception as error:
        cleanup_error = _rollback_durable_worker(
            process,
            control,
            (process.stdout, process.stderr),
            stream_threads,
        )
        detail = (
            f"; cleanup failure: {cleanup_error}"
            if cleanup_error is not None
            else ""
        )
        _report_worker_failure(
            f"durable worker bootstrap failure: worker thread startup failed: "
            f"{error}{detail}"
        )
        return 1
    try:
        returncode = process.wait()
    except Exception as error:
        cleanup_error = _rollback_durable_worker(
            process,
            control,
            (process.stdout, process.stderr),
            stream_threads,
        )
        detail = (
            f"; cleanup failure: {cleanup_error}"
            if cleanup_error is not None
            else ""
        )
        _report_worker_failure(
            f"durable worker bootstrap failure: worker wait failed: "
            f"{error}{detail}"
        )
        return 1
    control.signal_process_complete()
    for thread in stream_threads:
        thread.join()
    reader_error = control.join_reader()
    if reader_error is not None:
        control_failures.append(reader_error)
    cleanup_error = control.finalize_writer()
    if cleanup_error is not None:
        control_failures.append(cleanup_error)
    if forwarding_failures:
        _report_worker_failure(
            f"durable worker stream failure: {forwarding_failures[0]}"
        )
        return 1
    if control_failures:
        _report_worker_failure(
            f"durable worker bootstrap failure: control channel failure: "
            f"{control_failures[0]}"
        )
        return 1
    control_payload = control_payloads[0] if control_payloads else b""
    if control_payload != _WORKER_SCIENCE_START_MARKER:
        if control_payload:
            _report_worker_failure(
                "durable worker bootstrap failure: invalid control marker"
            )
        elif returncode < 0:
            _report_worker_failure(
                "durable worker bootstrap signal failure: "
                f"terminated by signal {-returncode} before science"
            )
        elif returncode == 2:
            return returncode
        else:
            _report_worker_failure(
                "durable worker bootstrap failure: "
                f"worker exited with status {returncode} before science"
            )
        return 1
    if returncode < 0:
        _report_worker_failure(
            "durable worker science signal failure: "
            f"terminated by signal {-returncode} after science start"
        )
        return 1
    return returncode


def _forward_worker_stream(
    source: Any,
    target: Any,
    failures: list[Exception],
    failure_lock: threading.Lock,
) -> None:
    """Decode and forward one pipe incrementally while retaining only one bounded chunk."""
    decoder = codecs.getincrementaldecoder("utf-8")(errors="backslashreplace")
    forwarding_error: Exception | None = None
    try:
        destination, binary = _worker_stream_destination(target)
    except Exception as error:
        forwarding_error = error
        destination, binary = target, False
    try:
        while chunk := source.read(_WORKER_STREAM_CHUNK_BYTES):
            if forwarding_error is not None:
                continue
            try:
                rendered = decoder.decode(chunk, final=False)
                if rendered:
                    _write_worker_stream(destination, rendered, binary=binary)
            except Exception as error:
                forwarding_error = error
        if forwarding_error is None:
            rendered = decoder.decode(b"", final=True)
            if rendered:
                _write_worker_stream(destination, rendered, binary=binary)
    except Exception as error:
        if forwarding_error is None:
            forwarding_error = error
    finally:
        try:
            target.flush()
        except Exception as error:
            if forwarding_error is None:
                forwarding_error = error
        try:
            source.close()
        except Exception as error:
            if forwarding_error is None:
                forwarding_error = error
    if forwarding_error is not None:
        with failure_lock:
            failures.append(forwarding_error)


def _worker_stream_destination(target: Any) -> tuple[Any, bool]:
    binary = getattr(target, "buffer", None)
    if binary is not None:
        return binary, True
    return target, isinstance(target, (io.BufferedIOBase, io.RawIOBase, io.BytesIO))


def _write_worker_stream(destination: Any, rendered: str, *, binary: bool) -> None:
    value: str | bytes = rendered.encode("utf-8") if binary else rendered
    offset = 0
    while offset < len(value):
        boundary = min(offset + _WORKER_STREAM_CHUNK_BYTES, len(value))
        piece = value[offset:boundary]
        written = destination.write(piece)
        if (
            not isinstance(written, int)
            or isinstance(written, bool)
            or written <= 0
        ):
            raise OSError("worker stream write made no progress")
        if written > len(piece):
            raise OSError("worker stream write returned an invalid count")
        offset += written


def _drain_worker_control(
    descriptor: int,
    completion: threading.Event,
    payloads: list[bytes],
    failures: list[Exception],
) -> None:
    """Drain concurrently, then stop on process completion without requiring EOF."""
    captured = bytearray()
    capture_limit = len(_WORKER_SCIENCE_START_MARKER) + 1
    control_error: Exception | None = None
    selector: selectors.BaseSelector | None = None
    try:
        os.set_blocking(descriptor, False)
        selector = selectors.DefaultSelector()
        selector.register(descriptor, selectors.EVENT_READ, "control")
        control_open = True
        while not completion.is_set():
            if not control_open:
                completion.wait()
                continue
            events = selector.select(_WORKER_CONTROL_POLL_SECONDS)
            if events:
                control_open = _drain_available_control(
                    descriptor,
                    captured,
                    capture_limit,
                )
                if not control_open:
                    selector.unregister(descriptor)
        if control_open:
            _drain_available_control(descriptor, captured, capture_limit)
    except Exception as error:
        control_error = error
    finally:
        if selector is not None:
            with suppress(Exception):
                selector.close()
        try:
            os.close(descriptor)
        except Exception as error:
            if control_error is None:
                control_error = error
    payloads.append(bytes(captured))
    if control_error is not None:
        failures.append(control_error)


def _drain_available_control(
    descriptor: int,
    captured: bytearray,
    capture_limit: int,
) -> bool:
    while True:
        try:
            chunk = os.read(descriptor, _WORKER_STREAM_CHUNK_BYTES)
        except BlockingIOError:
            return True
        if not chunk:
            return False
        remaining = capture_limit - len(captured)
        if remaining > 0:
            captured.extend(chunk[:remaining])


def _descriptor_identity(descriptor: int) -> _DescriptorIdentity:
    status = os.fstat(descriptor)
    return _DescriptorIdentity(
        status.st_dev,
        status.st_ino,
        status.st_mode,
        status.st_rdev,
        fcntl.fcntl(descriptor, fcntl.F_GETFL) & os.O_ACCMODE,
    )


def _close_worker_descriptor(descriptor: int) -> bool:
    try:
        os.close(descriptor)
    except OSError:
        return False
    return True


def _force_close_worker_descriptor(descriptor: int) -> Exception | None:
    try:
        os.closerange(descriptor, descriptor + 1)
    except Exception as error:
        return error
    return None


def _rollback_durable_worker(
    process: Any,
    control: _WorkerControlChannel,
    sources: tuple[Any | None, Any | None],
    stream_threads: Sequence[threading.Thread],
) -> Exception | None:
    cleanup_failures: list[Exception] = []
    deadline = time.monotonic() + _WORKER_CLEANUP_TIMEOUT_SECONDS
    try:
        process.kill()
    except ProcessLookupError:
        pass
    except Exception as error:
        cleanup_failures.append(error)
    try:
        process.wait(timeout=max(0.0, deadline - time.monotonic()))
    except Exception as error:
        cleanup_failures.append(error)
    control.signal_process_complete()

    source_failures: list[Exception] = []
    source_close_threads: list[threading.Thread] = []
    for source in sources:
        if source is None:
            continue
        try:
            thread = threading.Thread(
                target=_close_worker_source,
                args=(source, source_failures),
                daemon=True,
            )
        except Exception as error:
            cleanup_failures.append(error)
            continue
        source_close_threads.append(thread)
        try:
            thread.start()
        except Exception as error:
            cleanup_failures.append(error)

    control_error = control.close_unstarted(
        join_timeout=max(0.0, deadline - time.monotonic())
    )
    if control_error is not None:
        cleanup_failures.append(control_error)
    for index, thread in enumerate(stream_threads, start=1):
        thread_error = _join_cleanup_thread(
            thread,
            f"worker stream thread {index}",
            deadline,
        )
        if thread_error is not None:
            cleanup_failures.append(thread_error)
    for index, thread in enumerate(source_close_threads, start=1):
        thread_error = _join_cleanup_thread(
            thread,
            f"worker stream closer {index}",
            deadline,
        )
        if thread_error is not None:
            cleanup_failures.append(thread_error)
    cleanup_failures.extend(source_failures)
    if not cleanup_failures:
        return None
    if len(cleanup_failures) == 1:
        return cleanup_failures[0]
    return OSError("; ".join(str(error) for error in cleanup_failures))


def _close_worker_source(source: Any, failures: list[Exception]) -> None:
    try:
        source.close()
    except Exception as error:
        failures.append(error)


def _join_cleanup_thread(
    thread: threading.Thread,
    label: str,
    deadline: float,
) -> Exception | None:
    if thread.ident is None:
        return None
    try:
        thread.join(timeout=max(0.0, deadline - time.monotonic()))
    except Exception as error:
        return error
    if thread.is_alive():
        return TimeoutError(f"{label} remained alive after cleanup")
    return None


def _retain_control_writer(writer: _OwnedControlWriter) -> None:
    with _RETAINED_CONTROL_WRITERS_LOCK:
        _RETAINED_CONTROL_WRITERS.append(writer)


def _retry_retained_control_writers() -> Exception | None:
    cleanup_error: Exception | None = None
    with _RETAINED_CONTROL_WRITERS_LOCK:
        retained = tuple(_RETAINED_CONTROL_WRITERS)
        _RETAINED_CONTROL_WRITERS.clear()
        for writer in retained:
            writer_error = writer.finalize()
            if writer.is_owned:
                _RETAINED_CONTROL_WRITERS.append(writer)
                if cleanup_error is None:
                    cleanup_error = writer_error or OSError(
                        "retained control writer remains open"
                    )
    return cleanup_error


def _report_worker_failure(message: str) -> None:
    with suppress(Exception):
        print(message, file=sys.stderr)


def _run_after_parse(parser: argparse.ArgumentParser, arguments: argparse.Namespace) -> int:
    """Validate non-durable grammar and execute the established criteria boundary."""
    try:
        prepared = getattr(arguments, _PREPARED_SCIENCE_ATTRIBUTE, None)
        if prepared is not None:
            if type(prepared) is not _PreparedScience:
                raise RuntimeError("invalid prepared science authority")
            return _dispatch(arguments, prepared.criteria_path, prepared.target)
        with _prepare_science(parser, arguments):
            prepared = getattr(arguments, _PREPARED_SCIENCE_ATTRIBUTE)
            return _dispatch(arguments, prepared.criteria_path, prepared.target)
    except Exception as error:
        _print_json(
            {
                "boundary": EVIDENCE_BOUNDARY,
                "verdict": "INVALID_EVIDENCE",
                "driving_finding": "runtime.failure",
                "evidence_path": None,
                "error": str(error),
            }
        )
        return 1


@contextmanager
def _prepare_science(
    parser: argparse.ArgumentParser,
    arguments: argparse.Namespace,
) -> Iterator[None]:
    """Resolve all criteria and target imports before authorizing science."""
    if hasattr(arguments, _PREPARED_SCIENCE_ATTRIBUTE):
        raise RuntimeError("science is already prepared")
    if arguments.command == "run":
        entry = get_catalog_entry(arguments.config)
        workload = WorkloadName(arguments.workload)
        if entry.workload is not workload:
            parser.error("--config does not belong to the requested workload")

    from hephaestus.criteria import packaged_criteria_path

    with packaged_criteria_path() as criteria_path:
        prepared = _PreparedScience(criteria_path, _load_science_target(arguments.command))
        setattr(arguments, _PREPARED_SCIENCE_ATTRIBUTE, prepared)
        try:
            yield
        finally:
            delattr(arguments, _PREPARED_SCIENCE_ATTRIBUTE)


def _load_science_target(command: str) -> Callable[..., Any]:
    if command == "run":
        from hephaestus.workflows import run_catalog_action

        return run_catalog_action
    if command == "gate":
        from hephaestus.offline_gate import run_offline_gate

        return run_offline_gate
    if command == "agent":
        from hephaestus.search import run_scripted_search

        return run_scripted_search
    if command == "demo-planted-regressions":
        from hephaestus.demo import run_planted_demo

        return run_planted_demo
    if command == "aa-test":
        from hephaestus.aa_runtime import run_aa_test

        return run_aa_test
    raise RuntimeError("unhandled command")


def _dispatch(
    arguments: argparse.Namespace,
    criteria_path: Path,
    target: Callable[..., Any],
) -> int:
    if arguments.command == "run":
        result = target(
            WorkloadName(arguments.workload),
            arguments.config,
            arguments.output_root,
            criteria_path,
        )
        _print_json(_result_payload(result.bundle_path, result.verdict, result.driving_finding))
        return 0 if result.verdict in _NORMAL_VERDICTS else 1

    if arguments.command == "gate":
        result = target(arguments.bundle)
        payload = _result_payload(
            result.bundle_path,
            result.verdict,
            result.driving_finding,
        )
        payload["complete_verdict"] = result.complete_verdict
        _print_json(payload)
        return 0 if result.verdict in _NORMAL_VERDICTS else 1

    if arguments.command == "agent":
        search = target(
            WorkloadName(arguments.workload),
            arguments.output_root,
            criteria_path,
        )
        steps = search.transcript.steps
        if not steps:
            raise RuntimeError("trusted search returned no receipts")
        last = steps[-1].result
        _print_json(
            {
                **_result_payload(
                    search.parent_path,
                    last.verdict,
                    last.driving_finding,
                ),
                "receipts": [
                    {
                        "proposal": {
                            "catalog_id": step.proposal.catalog_id,
                            "workload_name": step.proposal.workload_name.value,
                            "rationale": step.proposal.rationale,
                        },
                        "result": {
                            "bundle_relative_path": step.result.bundle_relative_path,
                            "verdict": step.result.verdict,
                            "driving_finding": step.result.driving_finding,
                        },
                    }
                    for step in steps
                ],
            }
        )
        return 1 if any(step.result.verdict == "INVALID_EVIDENCE" for step in steps) else 0

    if arguments.command == "demo-planted-regressions":
        result = target(
            arguments.output_root,
            criteria_path,
            _worker_relative_root=getattr(arguments, "_worker_relative_root", False),
            _host_state_sink=arguments.attempt_root,
        )
        _append_demo_attempt(arguments.attempt_root, result)
        _print_demo(result)
        return 0 if result.passed else 1

    if arguments.command == "aa-test":
        result = target(
            WorkloadName(arguments.workload),
            arguments.output_root,
            criteria_path,
            _worker_relative_root=getattr(arguments, "_worker_relative_root", False),
            _host_state_sink=arguments.attempt_root,
        )
        _append_aa_attempt(arguments.attempt_root, arguments.workload, result)
        payload = _result_payload(
            result.parent_path.resolve(),
            result.verdict,
            result.driving_finding,
        )
        payload["statistics"] = _statistics_payload(result.statistics)
        _print_json(payload)
        return 0 if result.verdict == "PASS" else 1

    raise RuntimeError("unhandled command")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="hephaestus")
    commands = parser.add_subparsers(dest="command", required=True)
    workloads = tuple(workload.value for workload in WorkloadName)
    catalog_ids = tuple(entry.catalog_id for entry in CATALOG)

    run = commands.add_parser("run", help="execute one closed-catalog run")
    run.add_argument("workload", choices=workloads)
    run.add_argument("--config", required=True, choices=catalog_ids)
    run.add_argument("--output-root", type=Path, default=Path("artifacts"))

    gate = commands.add_parser("gate", help="re-gate one stored bundle offline")
    gate.add_argument("bundle", type=Path)

    agent = commands.add_parser("agent", help="run the scripted optimizer")
    agent_commands = agent.add_subparsers(dest="agent_command", required=True)
    optimize = agent_commands.add_parser("optimize", help="optimize one workload")
    optimize.add_argument("workload", choices=workloads)
    optimize.add_argument("--output-root", type=Path, default=Path("artifacts"))

    demo = commands.add_parser(
        "demo-planted-regressions",
        help="execute all planted defects and the clean control",
    )
    demo.add_argument("--output-root", type=Path, default=Path("artifacts"))
    demo.add_argument("--allow-volatile-output", action="store_true")

    aa_test = commands.add_parser("aa-test", help="execute the independent-run A/A meta-test")
    aa_test.add_argument("workload", choices=workloads)
    aa_test.add_argument("--output-root", type=Path, default=Path("artifacts"))
    aa_test.add_argument("--allow-volatile-output", action="store_true")
    return parser


def _result_payload(path: Path, verdict: str, driving_finding: str) -> dict[str, object]:
    return {
        "boundary": EVIDENCE_BOUNDARY,
        "verdict": verdict,
        "driving_finding": driving_finding,
        "evidence_path": str(path),
    }


def _statistics_payload(statistics: object) -> dict[str, object] | None:
    if statistics is None:
        return None
    return {
        "signed_paired_effects": list(statistics.signed_effects),  # type: ignore[attr-defined]
        "bootstrap_absolute_medians": list(  # type: ignore[attr-defined]
            statistics.bootstrap_absolute_medians  # type: ignore[attr-defined]
        ),
        "p95_noise_floor": statistics.p95_noise_floor,  # type: ignore[attr-defined]
        "speedup_lower_bound_a_over_b": statistics.speedup_lower_bound_a_over_b,  # type: ignore[attr-defined]
        "speedup_lower_bound_b_over_a": statistics.speedup_lower_bound_b_over_a,  # type: ignore[attr-defined]
    }


def _print_demo(result: Any) -> None:
    print(EVIDENCE_BOUNDARY)
    print(
        "catalog_id | expected_verdict | expected_finding | actual_verdict | "
        "actual_finding | pass | bundle"
    )
    for row in result.rows:
        passed = "yes" if row.passed else "no"
        print(
            f"{row.catalog_id} | {row.expected_verdict} | "
            f"{row.expected_driving_finding} | {row.actual_verdict} | "
            f"{row.actual_driving_finding} | {passed} | {row.bundle_relative_path}"
        )
    print(f"evidence_path: {result.parent_path.resolve()}")


def _append_aa_attempt(attempt_root: Any, workload: str, result: Any) -> None:
    statistics = result.statistics
    floor = None if statistics is None else statistics.p95_noise_floor
    _append_preserving_science(
        attempt_root,
        workload=workload,
        verdict=result.verdict,
        clean_control_floor=floor,
        parent_path=result.parent_path,
    )


def _append_demo_attempt(attempt_root: Any, result: Any) -> None:
    try:
        floor = _demo_clean_control_floor(result)
        _append_preserving_science(
            attempt_root,
            workload="demo-planted-regressions",
            verdict="PASS" if result.passed else "FAIL",
            clean_control_floor=floor,
            parent_path=result.parent_path,
        )
    except Exception as error:
        print(f"attempt ledger error: {error}", file=sys.stderr)


def _append_preserving_science(
    attempt_root: Any,
    *,
    workload: str,
    verdict: str,
    clean_control_floor: float | None,
    parent_path: Path,
) -> None:
    """Do not replace a completed scientific receipt if the filesystem later fails."""
    try:
        attempt_root.append_attempt(
            workload=workload,
            verdict=verdict,
            clean_control_floor=clean_control_floor,
            parent_path=parent_path,
        )
    except Exception as error:
        print(f"attempt ledger error: {error}", file=sys.stderr)


def _demo_clean_control_floor(result: Any) -> float | None:
    """Read the clean-control methodology floor without touching the manifested demo tree."""
    clean_row = next(
        (row for row in result.rows if row.catalog_id.startswith("clean-control-")),
        None,
    )
    if clean_row is None:
        return None
    payload = json.loads(
        (result.parent_path / clean_row.bundle_relative_path / "methodology.json").read_bytes()
    )
    if not isinstance(payload, dict):
        raise ValueError("clean-control methodology must be a JSON object")
    value = payload.get("aa_noise_floor")
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError("clean-control methodology lacks a numeric floor")
    return float(value)


def _print_json(payload: dict[str, object]) -> None:
    print(json.dumps(payload, ensure_ascii=True, separators=(",", ":"), sort_keys=True))


if __name__ == "__main__":
    sys.exit(main())

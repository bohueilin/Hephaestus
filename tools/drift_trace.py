"""Produce the ratified drift trace for 173c48b62d44f40fd10ff33ed863ed917177bd34."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from collections.abc import Callable, Iterable, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

RATIFIED_COMMIT = "173c48b62d44f40fd10ff33ed863ed917177bd34"
SETUP_DEVIATIONS = [
    "no non-primary compile sweep",
    "no accuracy comparison",
    "no bundle",
]
_BLOCK_FIELDS = (
    "eager_seconds",
    "eager_timestamps_utc",
    "baseline_seconds",
    "baseline_timestamps_utc",
    "candidate_seconds",
    "candidate_timestamps_utc",
)


def build_header(
    *,
    git_head: str,
    git_dirty: bool,
    workload: str,
    request: Any,
    config_digest_reference: str,
    start_utc: str,
    declared_duration_seconds: int,
    torch_num_threads: int,
    cpu_count: int | None,
    baseline_load_samples: Sequence[Sequence[float]],
    checklist: dict[str, bool],
    host_state: dict[str, object],
    mode: str | None = None,
    stop_reason: str | None = None,
) -> dict[str, object]:
    """Build one JSON-safe header without importing the compiler runtime."""
    samples = [list(sample) for sample in baseline_load_samples]
    header: dict[str, object] = {
        "row": "header",
        "git_head": git_head,
        "git_dirty": git_dirty,
        "ratified_commit": RATIFIED_COMMIT,
        "workload": workload,
        "request": {
            "backend": request.backend,
            "mode": request.mode,
            "dynamic": request.dynamic,
            "fullgraph": request.fullgraph,
            "options": request.options,
            "disable": request.disable,
        },
        "config_digest_reference": config_digest_reference,
        "start_utc": start_utc,
        "declared_duration_seconds": declared_duration_seconds,
        "torch_num_threads": torch_num_threads,
        "os_cpu_count": cpu_count,
        "baseline_load_samples": samples,
        "baseline_load": sum(sample[0] for sample in samples) / len(samples) if samples else None,
        "checklist": dict(checklist),
        "host_state_start": host_state,
    }
    if mode is not None:
        header["mode"] = mode
    if stop_reason is not None:
        header["stop_reason"] = stop_reason
    return header


def measure_block(
    *,
    measure_interleaved: Callable[..., tuple[tuple[object, ...], ...]],
    eager: object,
    compiled: object,
    case: object,
    clock: object,
) -> dict[str, object]:
    """Measure one fixed block and attach the six verbatim series names."""
    measured = measure_interleaved(
        eager=eager,
        compiled=compiled,
        case=case,
        repeats=64,
        spacing_seconds=0.0,
        clock=clock,
        schema_version=2,
    )
    if len(measured) != len(_BLOCK_FIELDS):
        raise ValueError("_measure_interleaved returned an unexpected shape")
    return {"row": "block"} | {
        name: list(values) for name, values in zip(_BLOCK_FIELDS, measured, strict=True)
    }


def build_trailer(
    *,
    end_utc: str,
    block_count: int,
    host_state: dict[str, object],
    recompile_reasons: Sequence[object],
    stop_reason: str,
    load_series: Sequence[dict[str, object]],
) -> dict[str, object]:
    """Build the terminal row retained for complete and partial traces."""
    reasons = list(recompile_reasons)
    return {
        "row": "trailer",
        "end_utc": end_utc,
        "achieved_block_count": block_count,
        "host_state_end": host_state,
        "recompile_reasons": reasons,
        "recompiled": bool(reasons),
        "setup_deviations": list(SETUP_DEVIATIONS),
        "stop_reason": stop_reason,
        "load_series": list(load_series),
    }


def write_jsonl(path: Path, rows: Iterable[dict[str, object]]) -> None:
    """Write accumulated rows once, without durability or ledger machinery."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(
                json.dumps(row, allow_nan=False, separators=(",", ":"), sort_keys=True) + "\n"
            )


def collect_blocks(
    *,
    measure_interleaved: Callable[..., tuple[tuple[object, ...], ...]],
    eager: object,
    compiled: object,
    case: object,
    clock: Any,
    getloadavg: Callable[[], tuple[float, float, float]],
    duration_seconds: float,
    block_rows: list[dict[str, object]],
    load_series: list[dict[str, object]],
) -> None:
    """Collect timed blocks in caller-owned memory until the fresh UTC duration guard."""
    load_series.append({"t": 0.0, "load_average": list(getloadavg())})
    last_load_monotonic = clock.monotonic()
    first_baseline_timestamp: datetime | None = None

    try:
        while True:
            if first_baseline_timestamp is not None:
                elapsed = (clock.timestamp_utc() - first_baseline_timestamp).total_seconds()
                if elapsed >= duration_seconds:
                    break

            row = measure_block(
                measure_interleaved=measure_interleaved,
                eager=eager,
                compiled=compiled,
                case=case,
                clock=clock,
            )
            block_rows.append(row)
            if first_baseline_timestamp is None:
                timestamps = row["baseline_timestamps_utc"]
                if not isinstance(timestamps, list) or not timestamps:
                    raise ValueError("block has no baseline timestamp")
                first_baseline_timestamp = _parse_utc(str(timestamps[0]))

            monotonic_now = clock.monotonic()
            if monotonic_now - last_load_monotonic >= 1.0:
                elapsed = (clock.timestamp_utc() - first_baseline_timestamp).total_seconds()
                load_series.append({"t": elapsed, "load_average": list(getloadavg())})
                last_load_monotonic = monotonic_now
    finally:
        if first_baseline_timestamp is None:
            elapsed = 0.0
        else:
            elapsed = (clock.timestamp_utc() - first_baseline_timestamp).total_seconds()
        load_series.append({"t": max(0.0, elapsed), "load_average": list(getloadavg())})


def capture_baseline(
    *,
    getloadavg: Callable[[], tuple[float, float, float]],
    sleep: Callable[[float], None],
    samples: list[list[float]],
) -> None:
    """Capture exactly sixty one-second load-average samples."""
    for _ in range(60):
        samples.append(list(getloadavg()))
        sleep(1.0)


def produce_trace(
    *,
    path: Path,
    start: datetime,
    workload_name: str,
    baseline_only: bool,
    checklist: dict[str, bool],
    git_head: str,
    git_dirty: bool,
    torch_num_threads: int,
    cpu_count: int | None,
    compile_request: Callable[..., Any],
    get_workload: Callable[[str], Any],
    measure_interleaved: Callable[..., tuple[tuple[object, ...], ...]],
    clock: Any,
    inductor_cache_scope: Callable[[], Any],
    reset_compiler_state: Callable[[], None],
    clear_compiler_memory_state: Callable[[], None],
    compile_module: Callable[[object, object], Any],
    capture_compiler_evidence: Callable[[], Any],
    read_compiler_evidence: Callable[[], dict[str, list[object]]],
    inference_mode: Callable[[], Any],
    sample_host_state: Callable[[], dict[str, object]],
    getloadavg: Callable[[], tuple[float, float, float]],
    utc_now: Callable[[], datetime],
    write_rows: Callable[[Path, Iterable[dict[str, object]]], None],
) -> None:
    """Run the exact producer sequence with injected runtime boundaries."""
    request_literals = {
        "graph_break_bait": (False, "9a7509d09749"),
        "mlp_stack": (True, "0bb4c54b98c6"),
    }
    fullgraph, digest_reference = request_literals[workload_name]
    request = compile_request(
        backend="inductor",
        mode=None,
        dynamic=False,
        fullgraph=fullgraph,
        options=None,
        disable=False,
    )
    baseline_samples: list[list[float]] = []
    header: dict[str, object] | None = None
    pending_error: BaseException | None = None
    try:
        capture_baseline(getloadavg=getloadavg, sleep=clock.sleep, samples=baseline_samples)
        header = build_header(
            git_head=git_head,
            git_dirty=git_dirty,
            workload=workload_name,
            request=request,
            config_digest_reference=digest_reference,
            start_utc=_utc_text(start),
            declared_duration_seconds=0 if baseline_only else 600,
            torch_num_threads=torch_num_threads,
            cpu_count=cpu_count,
            baseline_load_samples=baseline_samples,
            checklist=checklist,
            host_state=sample_host_state(),
            mode="baseline_only" if baseline_only else None,
        )
        if not baseline_only:
            workload = get_workload(workload_name)
            primary_case = workload.input_cases()[0]
            block_rows: list[dict[str, object]] = []
            load_series: list[dict[str, object]] = []
            evidence: dict[str, list[object]]

            with inductor_cache_scope():
                reset_compiler_state()
                module = workload.make_module()
                with inference_mode():
                    module(*primary_case)
                    block_loop_started = False
                    second_capture_error: BaseException | None = None
                    try:
                        with capture_compiler_evidence():
                            cold = compile_module(module, request)
                            cold(*primary_case)
                        clear_compiler_memory_state()
                        with capture_compiler_evidence():
                            compiled = compile_module(module, request)
                            compiled(*primary_case)
                            for _ in range(5):
                                module(*primary_case)
                                compiled(*primary_case)
                            block_loop_started = True
                            collect_blocks(
                                measure_interleaved=measure_interleaved,
                                eager=module,
                                compiled=compiled,
                                case=primary_case,
                                clock=clock,
                                getloadavg=getloadavg,
                                duration_seconds=600.0,
                                block_rows=block_rows,
                                load_series=load_series,
                            )
                    except (KeyboardInterrupt, Exception) as error:
                        second_capture_error = error
                    finally:
                        evidence = read_compiler_evidence()
                    if second_capture_error is not None:
                        if not block_loop_started:
                            raise second_capture_error
                        pending_error = second_capture_error
    except KeyboardInterrupt:
        if header is None:
            header = build_header(
                git_head=git_head,
                git_dirty=git_dirty,
                workload=workload_name,
                request=request,
                config_digest_reference=digest_reference,
                start_utc=_utc_text(start),
                declared_duration_seconds=0 if baseline_only else 600,
                torch_num_threads=torch_num_threads,
                cpu_count=cpu_count,
                baseline_load_samples=baseline_samples,
                checklist=checklist,
                host_state=sample_host_state(),
                mode="baseline_only" if baseline_only else None,
                stop_reason="KeyboardInterrupt",
            )
        else:
            header["stop_reason"] = "KeyboardInterrupt"
        write_rows(path, [header])
        raise
    if baseline_only:
        write_rows(path, [header])
        return
    if pending_error is None:
        stop_reason = "completed"
    elif isinstance(pending_error, KeyboardInterrupt):
        stop_reason = "KeyboardInterrupt"
    else:
        stop_reason = f"{type(pending_error).__name__}: {pending_error}"
    trailer = build_trailer(
        end_utc=_utc_text(utc_now()),
        block_count=len(block_rows),
        host_state=sample_host_state(),
        recompile_reasons=evidence["recompile_reasons"],
        stop_reason=stop_reason,
        load_series=load_series,
    )
    write_rows(path, [header, *block_rows, trailer])
    if pending_error is not None:
        raise pending_error


def _parse_utc(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.utcoffset() != UTC.utcoffset(parsed):
        raise ValueError("timestamp must be UTC")
    return parsed.astimezone(UTC)


def _utc_now() -> datetime:
    return datetime.now(UTC)


def _utc_text(value: datetime) -> str:
    return value.astimezone(UTC).isoformat(timespec="microseconds").replace("+00:00", "Z")


def _trace_path(workload: str, start: datetime) -> Path:
    timestamp = start.astimezone(UTC).strftime("%Y%m%dT%H%M%S.%fZ")
    return Path("artifacts/methodology-v2/drift-trace") / f"{workload}-{timestamp}.jsonl"


def _git_state() -> tuple[str, bool]:
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    dirty = bool(
        subprocess.run(
            ["git", "status", "--porcelain"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    )
    return head, dirty


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("workload", choices=("graph_break_bait", "mlp_stack"))
    parser.add_argument("--baseline-only", action="store_true")
    parser.add_argument("--checklist-ac-power", action="store_true")
    parser.add_argument("--checklist-lid-open", action="store_true")
    parser.add_argument("--checklist-display-unlocked", action="store_true")
    parser.add_argument("--checklist-no-other-user-applications", action="store_true")
    parser.add_argument("--checklist-no-other-codex-agent-process", action="store_true")
    parser.add_argument("--checklist-no-test-suite-or-browser", action="store_true")
    parser.add_argument("--checklist-clean-git-status", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Capture a baseline-only header or one compile-once drift trace."""
    args = _parser().parse_args(argv)

    # These private measurement imports are authorized for this analysis-grade script.
    import torch

    from hephaestus.host_state import sample_host_state
    from hephaestus.measure import _measure_interleaved, _SystemClock
    from hephaestus.torchbind import (
        CompileRequest,
        capture_compiler_evidence,
        clear_compiler_memory_state,
        compile_module,
        inductor_cache_scope,
        read_compiler_evidence,
        reset_compiler_state,
    )
    from hephaestus.workloads import get_workload

    start = _utc_now()
    path = _trace_path(args.workload, start)
    git_head, git_dirty = _git_state()
    checklist = {
        "ac_power": args.checklist_ac_power,
        "lid_open": args.checklist_lid_open,
        "display_unlocked": args.checklist_display_unlocked,
        "no_other_user_applications": args.checklist_no_other_user_applications,
        "no_other_codex_agent_process": args.checklist_no_other_codex_agent_process,
        "no_test_suite_or_browser": args.checklist_no_test_suite_or_browser,
        "clean_git_status": args.checklist_clean_git_status,
    }
    produce_trace(
        path=path,
        start=start,
        workload_name=args.workload,
        baseline_only=args.baseline_only,
        checklist=checklist,
        git_head=git_head,
        git_dirty=git_dirty,
        torch_num_threads=torch.get_num_threads(),
        cpu_count=os.cpu_count(),
        compile_request=CompileRequest,
        get_workload=get_workload,
        measure_interleaved=_measure_interleaved,
        clock=_SystemClock(),
        inductor_cache_scope=inductor_cache_scope,
        reset_compiler_state=reset_compiler_state,
        clear_compiler_memory_state=clear_compiler_memory_state,
        compile_module=compile_module,
        capture_compiler_evidence=capture_compiler_evidence,
        read_compiler_evidence=read_compiler_evidence,
        inference_mode=torch.inference_mode,
        sample_host_state=sample_host_state,
        getloadavg=os.getloadavg,
        utc_now=_utc_now,
        write_rows=write_jsonl,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

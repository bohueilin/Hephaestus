"""Torch-free producer row-contract tests for the drift trace."""

from __future__ import annotations

import builtins
import importlib
import json
import sys
from contextlib import contextmanager
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest

_TORCH_BEFORE_IMPORT = sys.modules.get("torch")

drift_trace = importlib.import_module("tools.drift_trace")

assert sys.modules.get("torch") is _TORCH_BEFORE_IMPORT


def _request() -> SimpleNamespace:
    return SimpleNamespace(
        backend="inductor",
        mode=None,
        dynamic=False,
        fullgraph=True,
        options=None,
        disable=False,
    )


def _header(*, mode: str | None = None) -> dict[str, object]:
    return drift_trace.build_header(
        git_head="b" * 40,
        git_dirty=False,
        workload="mlp_stack",
        request=_request(),
        config_digest_reference="0bb4c54b98c6",
        start_utc="2026-09-01T12:00:00.000000Z",
        declared_duration_seconds=0 if mode else 600,
        torch_num_threads=14,
        cpu_count=14,
        baseline_load_samples=[[1.0, 2.0, 3.0]] * 60,
        checklist={
            "ac_power": True,
            "lid_open": True,
            "display_unlocked": True,
            "no_other_user_applications": True,
            "no_other_codex_agent_process": True,
            "no_test_suite_or_browser": True,
            "clean_git_status": True,
        },
        host_state={"load_average": {"value": [1.0, 2.0, 3.0]}},
        mode=mode,
    )


def test_rows_round_trip_with_exact_prescribed_fields(tmp_path: Path) -> None:
    calls: list[dict[str, object]] = []

    def fake_measure_interleaved(**kwargs: object) -> tuple[tuple[object, ...], ...]:
        calls.append(kwargs)
        return (
            (0.10, 0.11),
            ("2026-09-01T12:01:00.100000Z", "2026-09-01T12:01:00.200000Z"),
            (0.01, 0.02),
            ("2026-09-01T12:01:00.110000Z", "2026-09-01T12:01:00.210000Z"),
            (0.03, 0.04),
            ("2026-09-01T12:01:00.120000Z", "2026-09-01T12:01:00.220000Z"),
        )

    host_samples = iter(
        [
            {"load_average": {"value": [1.0, 2.0, 3.0]}},
            {"load_average": {"value": [1.5, 2.5, 3.5]}},
        ]
    )
    start_host = next(host_samples)
    block = drift_trace.measure_block(
        measure_interleaved=fake_measure_interleaved,
        eager="eager",
        compiled="compiled",
        case=("input",),
        clock="clock",
    )
    trailer = drift_trace.build_trailer(
        end_utc="2026-09-01T12:11:00.000000Z",
        block_count=1,
        host_state=next(host_samples),
        recompile_reasons=[],
        stop_reason="completed",
        load_series=[{"t": 0.0, "load_average": [1.0, 2.0, 3.0]}],
    )
    path = tmp_path / "trace.jsonl"
    drift_trace.write_jsonl(path, [_header() | {"host_state_start": start_host}, block, trailer])
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]

    assert set(rows[0]) == {
        "row",
        "git_head",
        "git_dirty",
        "ratified_commit",
        "workload",
        "request",
        "config_digest_reference",
        "start_utc",
        "declared_duration_seconds",
        "torch_num_threads",
        "os_cpu_count",
        "baseline_load_samples",
        "baseline_load",
        "checklist",
        "host_state_start",
    }
    assert rows[0]["ratified_commit"] == "173c48b62d44f40fd10ff33ed863ed917177bd34"
    assert rows[0]["request"] == {
        "backend": "inductor",
        "mode": None,
        "dynamic": False,
        "fullgraph": True,
        "options": None,
        "disable": False,
    }
    assert set(rows[1]) == {
        "row",
        "eager_seconds",
        "eager_timestamps_utc",
        "baseline_seconds",
        "baseline_timestamps_utc",
        "candidate_seconds",
        "candidate_timestamps_utc",
    }
    assert rows[1]["baseline_seconds"] == [0.01, 0.02]
    assert set(rows[2]) == {
        "row",
        "end_utc",
        "achieved_block_count",
        "host_state_end",
        "recompile_reasons",
        "recompiled",
        "setup_deviations",
        "stop_reason",
        "load_series",
    }
    assert calls == [
        {
            "eager": "eager",
            "compiled": "compiled",
            "case": ("input",),
            "repeats": 64,
            "spacing_seconds": 0.0,
            "clock": "clock",
            "schema_version": 2,
        }
    ]


def test_baseline_only_writes_one_explicit_header(tmp_path: Path) -> None:
    path = tmp_path / "baseline.jsonl"
    drift_trace.write_jsonl(path, [_header(mode="baseline_only")])

    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]

    assert len(rows) == 1
    assert rows[0]["mode"] == "baseline_only"
    assert rows[0]["request"]["mode"] is None
    assert rows[0]["declared_duration_seconds"] == 0


def _measurement(timestamp: str = "2026-09-01T12:00:00.000000Z") -> tuple[tuple[object, ...], ...]:
    return (
        (0.1,) * 64,
        (timestamp,) * 64,
        (0.01,) * 64,
        (timestamp,) * 64,
        (0.02,) * 64,
        (timestamp,) * 64,
    )


class _FakeClock:
    def __init__(
        self,
        *,
        timestamps: list[datetime],
        monotonics: list[float] | None = None,
    ) -> None:
        self.timestamps = iter(timestamps)
        self.monotonics = iter(monotonics or [0.0, 0.1])
        self.sleep_calls: list[float] = []

    def monotonic(self) -> float:
        return next(self.monotonics)

    def timestamp_utc(self) -> datetime:
        return next(self.timestamps)

    def sleep(self, seconds: float) -> None:
        self.sleep_calls.append(seconds)


class _ProducerHarness:
    def __init__(
        self,
        *,
        measurements: list[tuple[tuple[object, ...], ...] | BaseException] | None = None,
        compile_error_call: int | None = None,
    ) -> None:
        start = datetime(2026, 9, 1, 12, tzinfo=UTC)
        self.clock = _FakeClock(
            timestamps=[start + timedelta(seconds=601), start + timedelta(seconds=602)],
        )
        self.events: list[str] = []
        self.writes: list[list[dict[str, object]]] = []
        self.request_calls: list[dict[str, object]] = []
        self.measurements = iter(measurements or [_measurement()])
        self.compile_error_call = compile_error_call
        self.compile_calls = 0
        self.load_calls = 0
        self.host_calls = 0

    def request_factory(self, **kwargs: object) -> SimpleNamespace:
        self.events.append("request")
        self.request_calls.append(kwargs)
        return SimpleNamespace(**kwargs)

    def getloadavg(self) -> tuple[float, float, float]:
        self.load_calls += 1
        value = float(self.load_calls)
        return value, value + 1.0, value + 2.0

    def sample_host_state(self) -> dict[str, object]:
        self.host_calls += 1
        self.events.append(f"host:{self.host_calls}")
        return {"load_average": {"value": [1.0, 2.0, 3.0]}}

    def get_workload(self, name: str) -> SimpleNamespace:
        self.events.append(f"get_workload:{name}")

        def make_module() -> object:
            self.events.append("make_module")

            def module(*_case: object) -> None:
                self.events.append("eager_call")

            return module

        return SimpleNamespace(input_cases=lambda: (("input",),), make_module=make_module)

    @contextmanager
    def _context(self, name: str) -> object:
        self.events.append(f"{name}_enter")
        try:
            yield
        except BaseException:
            self.events.append(f"{name}_exit_exception")
            raise
        else:
            self.events.append(f"{name}_exit")

    def cache_scope(self) -> object:
        return self._context("cache")

    def inference_mode(self) -> object:
        return self._context("inference")

    def capture_evidence(self) -> object:
        capture_number = 1 + sum(
            event.endswith("_enter") for event in self.events if "capture" in event
        )
        return self._context(f"capture{capture_number}")

    def reset(self) -> None:
        self.events.append("reset")

    def clear(self) -> None:
        self.events.append("clear")

    def compile(self, _module: object, _request: object) -> object:
        self.compile_calls += 1
        role = "cold" if self.compile_calls == 1 else "warm"
        self.events.append(f"compile:{role}")
        if self.compile_calls == self.compile_error_call:
            raise RuntimeError(f"{role} compile failed")

        def compiled(*_case: object) -> None:
            self.events.append(f"{role}_call")

        return compiled

    def measure(self, **_kwargs: object) -> tuple[tuple[object, ...], ...]:
        self.events.append("measure")
        outcome = next(self.measurements)
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome

    def read_evidence(self) -> dict[str, list[object]]:
        self.events.append("read_evidence")
        return {"recompile_reasons": ["guard changed"]}

    def write_rows(self, _path: Path, rows: object) -> None:
        self.events.append("write")
        self.writes.append(list(rows))

    def kwargs(self, path: Path, *, workload_name: str, baseline_only: bool) -> dict[str, object]:
        return {
            "path": path,
            "start": datetime(2026, 9, 1, 12, tzinfo=UTC),
            "workload_name": workload_name,
            "baseline_only": baseline_only,
            "checklist": {"ac_power": True},
            "git_head": "b" * 40,
            "git_dirty": False,
            "torch_num_threads": 14,
            "cpu_count": 14,
            "compile_request": self.request_factory,
            "get_workload": self.get_workload,
            "measure_interleaved": self.measure,
            "clock": self.clock,
            "inductor_cache_scope": self.cache_scope,
            "reset_compiler_state": self.reset,
            "clear_compiler_memory_state": self.clear,
            "compile_module": self.compile,
            "capture_compiler_evidence": self.capture_evidence,
            "read_compiler_evidence": self.read_evidence,
            "inference_mode": self.inference_mode,
            "sample_host_state": self.sample_host_state,
            "getloadavg": self.getloadavg,
            "utc_now": lambda: datetime(2026, 9, 1, 12, 10, 1, tzinfo=UTC),
            "write_rows": self.write_rows,
        }


def test_collect_blocks_checks_fresh_elapsed_before_every_later_block() -> None:
    start = datetime(2026, 9, 1, 12, tzinfo=UTC)
    clock = _FakeClock(
        timestamps=[
            start + timedelta(seconds=599.9),
            start + timedelta(seconds=600.1),
            start + timedelta(seconds=600.2),
        ],
        monotonics=[0.0, 1.1],
    )
    measure_calls = 0
    rows: list[dict[str, object]] = []
    load_series: list[dict[str, object]] = []

    def measure(**_kwargs: object) -> tuple[tuple[object, ...], ...]:
        nonlocal measure_calls
        measure_calls += 1
        if measure_calls > 1:
            raise AssertionError("duration guard permitted an extra block")
        return _measurement()

    drift_trace.collect_blocks(
        measure_interleaved=measure,
        eager="eager",
        compiled="compiled",
        case=(),
        clock=clock,
        getloadavg=lambda: (1.0, 2.0, 3.0),
        duration_seconds=600.0,
        block_rows=rows,
        load_series=load_series,
    )

    assert measure_calls == 1
    assert len(rows) == 1
    assert load_series[1]["t"] == pytest.approx(599.9)
    assert load_series[-1]["t"] == pytest.approx(600.2)


@pytest.mark.parametrize(
    ("workload_name", "fullgraph", "digest"),
    [
        ("graph_break_bait", False, "9a7509d09749"),
        ("mlp_stack", True, "0bb4c54b98c6"),
    ],
)
def test_producer_baseline_only_writes_one_literal_header(
    tmp_path: Path, workload_name: str, fullgraph: bool, digest: str
) -> None:
    harness = _ProducerHarness()
    drift_trace.produce_trace(
        **harness.kwargs(
            tmp_path / "baseline.jsonl", workload_name=workload_name, baseline_only=True
        )
    )

    assert harness.request_calls == [
        {
            "backend": "inductor",
            "mode": None,
            "dynamic": False,
            "fullgraph": fullgraph,
            "options": None,
            "disable": False,
        }
    ]
    assert len(harness.writes) == 1
    assert len(harness.writes[0]) == 1
    header = harness.writes[0][0]
    assert header["mode"] == "baseline_only"
    assert header["config_digest_reference"] == digest
    assert header["declared_duration_seconds"] == 0
    assert len(header["baseline_load_samples"]) == 60
    assert header["baseline_load"] == pytest.approx(30.5)
    assert harness.clock.sleep_calls == [1.0] * 60
    assert not any(event.startswith(("cache", "compile", "measure")) for event in harness.events)


def test_producer_runs_exact_compile_once_sequence_and_writes_once(tmp_path: Path) -> None:
    harness = _ProducerHarness()
    drift_trace.produce_trace(
        **harness.kwargs(tmp_path / "trace.jsonl", workload_name="mlp_stack", baseline_only=False)
    )

    expected = [
        "request",
        "host:1",
        "get_workload:mlp_stack",
        "cache_enter",
        "reset",
        "make_module",
        "inference_enter",
        "eager_call",
        "capture1_enter",
        "compile:cold",
        "cold_call",
        "capture1_exit",
        "clear",
        "capture2_enter",
        "compile:warm",
        "warm_call",
        *[event for _ in range(5) for event in ("eager_call", "warm_call")],
        "measure",
        "capture2_exit",
        "read_evidence",
        "inference_exit",
        "cache_exit",
        "host:2",
        "write",
    ]
    assert harness.events == expected
    assert harness.compile_calls == 2
    assert len(harness.writes) == 1
    assert [row["row"] for row in harness.writes[0]] == ["header", "block", "trailer"]
    assert harness.writes[0][-1]["setup_deviations"] == [
        "no non-primary compile sweep",
        "no accuracy comparison",
        "no bundle",
    ]
    assert harness.writes[0][-1]["recompiled"] is True


@pytest.mark.parametrize(
    ("error", "stop_reason"),
    [
        (RuntimeError("measurement failed"), "RuntimeError: measurement failed"),
        (KeyboardInterrupt(), "KeyboardInterrupt"),
    ],
)
def test_producer_writes_partial_trailer_after_capture_exit_then_reraises(
    tmp_path: Path, error: BaseException, stop_reason: str
) -> None:
    harness = _ProducerHarness(measurements=[_measurement(), error])
    harness.clock = _FakeClock(
        timestamps=[
            datetime(2026, 9, 1, 12, 0, 1, tzinfo=UTC),
            datetime(2026, 9, 1, 12, 0, 2, tzinfo=UTC),
        ],
    )

    with pytest.raises(type(error), match=str(error) or None):
        drift_trace.produce_trace(
            **harness.kwargs(
                tmp_path / "partial.jsonl", workload_name="mlp_stack", baseline_only=False
            )
        )

    assert harness.events.index("capture2_exit_exception") < harness.events.index("read_evidence")
    assert harness.events.index("read_evidence") < harness.events.index("write")
    assert len(harness.writes) == 1
    assert [row["row"] for row in harness.writes[0]] == ["header", "block", "trailer"]
    assert harness.writes[0][-1]["stop_reason"] == stop_reason
    assert harness.writes[0][-1]["achieved_block_count"] == 1


@pytest.mark.parametrize("compile_error_call", [1, 2])
def test_compile_failure_propagates_before_any_write(
    tmp_path: Path, compile_error_call: int
) -> None:
    harness = _ProducerHarness(compile_error_call=compile_error_call)
    path = tmp_path / "compile-failed.jsonl"

    with pytest.raises(RuntimeError, match="compile failed"):
        drift_trace.produce_trace(
            **harness.kwargs(path, workload_name="mlp_stack", baseline_only=False)
        )

    assert harness.writes == []
    assert not path.exists()
    if compile_error_call == 2:
        assert harness.events.index("capture2_exit_exception") < harness.events.index(
            "read_evidence"
        )


@pytest.mark.parametrize("failed_import", ["torch", "hephaestus.torchbind"])
def test_runtime_import_failure_precedes_any_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failed_import: str
) -> None:
    real_import = builtins.__import__

    def fake_import(
        name: str,
        globals_: object = None,
        locals_: object = None,
        fromlist: object = (),
        level: int = 0,
    ) -> object:
        if name == failed_import:
            raise ImportError(f"blocked {name}")
        if name == "torch":
            return SimpleNamespace()
        if name == "hephaestus.host_state":
            return SimpleNamespace(sample_host_state=lambda: {})
        if name == "hephaestus.measure":
            return SimpleNamespace(_measure_interleaved=lambda **_kwargs: (), _SystemClock=object)
        return real_import(name, globals_, locals_, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    monkeypatch.chdir(tmp_path)

    with pytest.raises(ImportError, match=f"blocked {failed_import}"):
        drift_trace.main(["mlp_stack", "--baseline-only"])

    assert not (tmp_path / "artifacts").exists()


@pytest.mark.parametrize("location", ["baseline_empty", "baseline_partial", "compile", "output"])
def test_producer_writes_one_aborted_baseline_header_on_keyboard_interrupt(
    tmp_path: Path, location: str
) -> None:
    harness = _ProducerHarness()
    kwargs = harness.kwargs(
        tmp_path / "aborted.jsonl", workload_name="mlp_stack", baseline_only=False
    )

    if location == "baseline_empty":

        def interrupting_loadavg() -> tuple[float, float, float]:
            raise KeyboardInterrupt

        kwargs["getloadavg"] = interrupting_loadavg
        expected_samples = []
        expected_load = None
    elif location == "baseline_partial":
        load_calls = 0

        def interrupting_loadavg() -> tuple[float, float, float]:
            nonlocal load_calls
            load_calls += 1
            if load_calls == 3:
                raise KeyboardInterrupt
            return harness.getloadavg()

        kwargs["getloadavg"] = interrupting_loadavg
        expected_samples = [[1.0, 2.0, 3.0], [2.0, 3.0, 4.0]]
        expected_load = pytest.approx(1.5)
    elif location == "compile":

        def interrupting_compile(_module: object, _request: object) -> object:
            raise KeyboardInterrupt

        kwargs["compile_module"] = interrupting_compile
        expected_samples = [[float(i), float(i + 1), float(i + 2)] for i in range(1, 61)]
        expected_load = pytest.approx(30.5)
    else:

        def interrupting_write(path: Path, rows: object) -> None:
            harness.write_rows(path, rows)
            raise KeyboardInterrupt

        kwargs["write_rows"] = interrupting_write
        expected_samples = [[float(i), float(i + 1), float(i + 2)] for i in range(1, 61)]
        expected_load = pytest.approx(30.5)

    with pytest.raises(KeyboardInterrupt):
        drift_trace.produce_trace(**kwargs)

    assert len(harness.writes) == 1
    if location == "output":
        assert [row["row"] for row in harness.writes[0]] == ["header", "block", "trailer"]
    else:
        assert len(harness.writes[0]) == 1
        header = harness.writes[0][0]
        assert header["row"] == "header"
        assert header["baseline_load_samples"] == expected_samples
        assert header["baseline_load"] == expected_load
        assert header["stop_reason"] == "KeyboardInterrupt"

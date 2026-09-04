"""Deterministic methodology around real PyTorch compiler execution."""

from __future__ import annotations

import hashlib
import inspect
import math
import random
import sys
import tempfile
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from types import MappingProxyType
from typing import Any, Protocol

import torch

from hephaestus.bundle import canonical_json_bytes, finalize_bundle, write_json
from hephaestus.evidence_contract import (
    V1_ACCURACY_TOLERANCE,
    V1_WORKLOAD_SHA256,
    stable_signed_paired_effect,
    v1_compile_cache_json,
    v1_path_normalization_json,
)
from hephaestus.gate import _evaluate_provisional_bundle
from hephaestus.input_plan import (
    InputPlan,
    build_identity_input_plan,
    canonicalize_input_plan,
    input_plan_json,
)
from hephaestus.privacy import normalize_dynamo_report, normalize_public_evidence
from hephaestus.provenance import RunProvenance
from hephaestus.torchbind import (
    CompileRequest,
    capture_compiler_evidence,
    clear_compiler_memory_state,
    compile_module,
    environment_snapshot,
    inductor_cache_scope,
    read_compiler_evidence,
    reset_compiler_state,
    snapshot_compiler_evidence,
)
from hephaestus.workloads import get_workload
from hephaestus.workloads.base import InputCase, WorkloadSpec


@dataclass(frozen=True, slots=True)
class RunSettings:
    """Declared settings for one measurement run."""

    warmup_runs: int = 5
    repeats: int = 31
    bootstrap_samples: int = 2000
    inter_run_spacing_seconds: float = 0.0
    schema_version: int = 1

    def __post_init__(self) -> None:
        if type(self.schema_version) is not int or self.schema_version not in {1, 2}:
            raise ValueError("schema_version must be exactly integer 1 or 2")
        if not isinstance(self.warmup_runs, int) or isinstance(self.warmup_runs, bool):
            raise ValueError("warmup_runs must be a nonnegative integer")
        if self.warmup_runs < 0:
            raise ValueError("warmup_runs must be a nonnegative integer")
        if not isinstance(self.repeats, int) or isinstance(self.repeats, bool):
            raise ValueError("repeats must be a positive integer")
        if self.repeats <= 0:
            raise ValueError("repeats must be a positive integer")
        if self.schema_version == 2 and self.repeats % 2 != 0:
            raise ValueError("schema_version 2 repeats must be even")
        if not isinstance(self.bootstrap_samples, int) or isinstance(
            self.bootstrap_samples, bool
        ):
            raise ValueError("bootstrap_samples must be a positive integer")
        if self.bootstrap_samples <= 0:
            raise ValueError("bootstrap_samples must be a positive integer")
        if self.bootstrap_samples != 2000:
            raise ValueError("bootstrap_samples must be exactly 2000 for the frozen gate")
        spacing = self.inter_run_spacing_seconds
        if isinstance(spacing, bool) or not math.isfinite(spacing) or spacing < 0:
            raise ValueError("inter_run_spacing_seconds must be finite and nonnegative")


@dataclass(frozen=True, slots=True)
class RawRunEvidence:
    """Raw and directly derived evidence from one isolated compiler run."""

    timings: dict[str, object]
    accuracy: dict[str, object]
    methodology: dict[str, object]
    dynamo_report: dict[str, object]
    input_plan: dict[str, object]


@dataclass(frozen=True, slots=True)
class RunResult:
    """Narrow, read-only result boundary exposed to callers and agents."""

    bundle_path: Path
    verdict: str
    summary: Mapping[str, object]


class _Clock(Protocol):
    def monotonic(self) -> float: ...

    def timestamp_utc(self) -> datetime: ...

    def sleep(self, seconds: float) -> None: ...


@dataclass(frozen=True, slots=True)
class _SystemClock:
    monotonic: Callable[[], float] = time.perf_counter
    timestamp_utc: Callable[[], datetime] = lambda: datetime.now(UTC)
    sleep: Callable[[float], None] = time.sleep


def measure(
    workload: WorkloadSpec,
    request: CompileRequest,
    settings: RunSettings,
    *,
    input_plan: InputPlan | None = None,
    _clock: _Clock | None = None,
) -> RawRunEvidence:
    """Measure eager and compiled execution while preserving raw per-iteration evidence."""
    clock = _SystemClock() if _clock is None else _clock
    request_strategy = _identity_strategy(request.dynamic)
    plan = build_identity_input_plan(workload, request_strategy)
    if input_plan is not None:
        plan = canonicalize_input_plan(workload, input_plan)
    strategy = plan.evidence.get("dynamic_strategy")
    expected_dynamic = {
        "static": False,
        "dynamic": True,
        "auto": None,
        "bucketed": False,
    }.get(strategy)
    if strategy not in {"static", "dynamic", "auto", "bucketed"} or (
        request.dynamic is not expected_dynamic
    ):
        raise ValueError("input plan dynamic strategy disagrees with compile request")
    effective_workload = plan.workload
    cases = plan.cases

    with inductor_cache_scope() as cache_scope:
        reset_compiler_state()
        module = effective_workload.make_module()
        primary_case = cases[0]
        with torch.inference_mode():
            eager_outputs = tuple(module(*case) for case in cases)

            with capture_compiler_evidence():
                cold_started = clock.monotonic()
                cold_compiled = compile_module(module, request)
                cold_first_output = cold_compiled(*primary_case)
                cold_compile_seconds = _duration(cold_started, clock.monotonic())
                cold_compile_timestamp = _timestamp(clock)
            cold_captured = read_compiler_evidence()

            cache_applicable = request.backend == "inductor" and not request.disable
            if not cache_applicable:
                warm_cache_compile_seconds = None
                warm_cache_compile_timestamp = None
                warm_cache_metrics: list[object] = []
                with capture_compiler_evidence():
                    operational = _measure_operational_phase(
                        module,
                        cold_compiled,
                        cold_first_output,
                        cases,
                        settings,
                        clock,
                    )
            else:
                clear_compiler_memory_state()
                with capture_compiler_evidence():
                    warm_started = clock.monotonic()
                    warm_compiled = compile_module(module, request)
                    warm_first_output = warm_compiled(*primary_case)
                    warm_cache_compile_seconds = _duration(
                        warm_started,
                        clock.monotonic(),
                    )
                    warm_cache_compile_timestamp = _timestamp(clock)
                    warm_cache_metrics = snapshot_compiler_evidence()[
                        "compilation_metrics"
                    ]
                    operational = _measure_operational_phase(
                        module,
                        warm_compiled,
                        warm_first_output,
                        cases,
                        settings,
                        clock,
                    )
            captured = read_compiler_evidence()
        normalization_roots = _normalization_roots(cache_scope.cache_root)

    (
        compiled_outputs,
        sweep_case_indices,
        sweep_seconds,
        sweep_timestamps,
        eager_seconds,
        eager_timestamps,
        baseline_seconds,
        baseline_timestamps,
        candidate_seconds,
        candidate_timestamps,
    ) = operational

    if settings.schema_version == 1:
        aa_signed_paired_effects = tuple(
            _signed_paired_effect(baseline, candidate)
            for baseline, candidate in zip(baseline_seconds, candidate_seconds, strict=True)
        )
    else:
        aa_signed_paired_effects = tuple(
            effect
            for even_index in range(0, settings.repeats, 2)
            for effect in (
                _signed_paired_effect(
                    baseline_seconds[even_index], candidate_seconds[even_index + 1]
                ),
                _signed_paired_effect(
                    baseline_seconds[even_index + 1], candidate_seconds[even_index]
                ),
            )
        )
    aa_bootstrap_absolute_medians = _bootstrap_absolute_medians(
        aa_signed_paired_effects,
        settings.bootstrap_samples,
        seed=0,
    )
    accuracy = _all_case_accuracy(
        eager_outputs,
        tuple(compiled_outputs),
        dtype=str(effective_workload.dtype),
        atol=effective_workload.atol,
        rtol=effective_workload.rtol,
    )
    captured = read_compiler_evidence()
    timings: dict[str, object] = {
        "eager_seconds": eager_seconds,
        "eager_timestamps_utc": eager_timestamps,
        "cold_compile_seconds": cold_compile_seconds,
        "cold_compile_timestamp_utc": cold_compile_timestamp,
        "warm_cache_compile_seconds": warm_cache_compile_seconds,
        "warm_cache_compile_timestamp_utc": warm_cache_compile_timestamp,
        "non_primary_compile_sweep_case_indices": tuple(sweep_case_indices),
        "non_primary_compile_sweep_seconds": tuple(sweep_seconds),
        "non_primary_compile_sweep_timestamps_utc": tuple(sweep_timestamps),
        "compiled_seconds": baseline_seconds,
        "compiled_timestamps_utc": baseline_timestamps,
        "aa_baseline_seconds": baseline_seconds,
        "aa_baseline_timestamps_utc": baseline_timestamps,
        "aa_candidate_seconds": candidate_seconds,
        "aa_candidate_timestamps_utc": candidate_timestamps,
        "aa_signed_paired_effects": aa_signed_paired_effects,
        "aa_bootstrap_absolute_medians": aa_bootstrap_absolute_medians,
        "summary": {
            "eager_median_seconds": _quantile(eager_seconds, 0.5),
            "eager_iqr_seconds": _iqr(eager_seconds),
            "compiled_median_seconds": _quantile(baseline_seconds, 0.5),
            "compiled_iqr_seconds": _iqr(baseline_seconds),
        },
    }
    methodology: dict[str, object] = {
        "valid": True,
        "warmup_runs": settings.warmup_runs,
        "repeats": settings.repeats,
        "bootstrap_samples": settings.bootstrap_samples,
        "bootstrap_seed": 0,
        "bootstrap_confidence": 0.95,
        "inter_run_spacing_seconds": settings.inter_run_spacing_seconds,
        "aa_noise_floor": _quantile(aa_bootstrap_absolute_medians, 0.95),
        "aa_effect_formula": "(A-B)/((A+B)/2)",
        "aa_estimator": "p95_absolute_bootstrap_median",
        "aa_pairing": (
            "within_iteration"
            if settings.schema_version == 1
            else "position_matched_cross_parity"
        ),
        "measurement_schedule": (
            "alternate_eager-A-B__B-A-eager"
            if settings.schema_version == 1
            else "alternate_eager-A-B__eager-B-A"
        ),
        "quantile_method": "linear_interpolation",
        "compiler_state_reset": True,
        "compile_cache": v1_compile_cache_json(),
    }
    if settings.schema_version == 2:
        methodology["schema_version"] = 2
    cache_evidence: dict[str, object]
    if not cache_applicable:
        cache_evidence = {
            "applicable": False,
            "reason": (
                "compiler_disabled" if request.disable else "non_inductor_backend"
            ),
            "cold_compilation_metrics": [],
            "warm_cache_compilation_metrics": [],
        }
    else:
        cache_evidence = {
            "applicable": True,
            "reason": None,
            "cold_compilation_metrics": cold_captured["compilation_metrics"],
            "warm_cache_compilation_metrics": warm_cache_metrics,
        }
    raw_dynamo_report: dict[str, object] = {
        "schema_version": 1,
        "graph_breaks": captured["graph_breaks"],
        "recompiles": [{"trigger": trigger} for trigger in captured["recompile_reasons"]],
        "compilation_metrics": captured["compilation_metrics"],
        "log_records": captured["log_records"],
        "cache_evidence": cache_evidence,
        "path_normalization": v1_path_normalization_json(),
    }
    dynamo_report = normalize_dynamo_report(raw_dynamo_report, normalization_roots)
    return RawRunEvidence(
        timings=timings,
        accuracy=accuracy,
        methodology=methodology,
        dynamo_report=dynamo_report,
        input_plan=input_plan_json(plan),
    )


def _measure_operational_phase(
    module: Callable[..., Any],
    compiled: Callable[..., Any],
    first_output: Any,
    cases: tuple[InputCase, ...],
    settings: RunSettings,
    clock: _Clock,
) -> tuple[
    list[Any],
    list[int],
    list[float],
    list[str],
    tuple[float, ...],
    tuple[str, ...],
    tuple[float, ...],
    tuple[str, ...],
    tuple[float, ...],
    tuple[str, ...],
]:
    compiled_outputs = [first_output]
    sweep_case_indices: list[int] = []
    sweep_seconds: list[float] = []
    sweep_timestamps: list[str] = []
    for case_index, case in enumerate(cases[1:], start=1):
        sweep_started = clock.monotonic()
        compiled_outputs.append(compiled(*case))
        sweep_seconds.append(_duration(sweep_started, clock.monotonic()))
        sweep_timestamps.append(_timestamp(clock))
        sweep_case_indices.append(case_index)
    for warmup_index in range(settings.warmup_runs):
        warmup_case = cases[warmup_index % len(cases)]
        module(*warmup_case)
        compiled(*warmup_case)
    measured = _measure_interleaved(
        module,
        compiled,
        cases[0],
        settings.repeats,
        settings.inter_run_spacing_seconds,
        clock,
        schema_version=settings.schema_version,
    )
    return (
        compiled_outputs,
        sweep_case_indices,
        sweep_seconds,
        sweep_timestamps,
        *measured,
    )


def _normalization_roots(cache_root: Path) -> dict[str, Path]:
    torch_package = Path(torch.__file__).resolve().parent
    return {
        "<INDUCTOR_CACHE>": cache_root.resolve(),
        "<HEPHAESTUS_PACKAGE>": Path(__file__).resolve().parent,
        "<TORCH_PACKAGE>": torch_package,
        "<PYTHON_ENV>": Path(sys.prefix).resolve(),
        "<PYTHON_RUNTIME>": Path(sys.base_prefix).resolve(),
        "<PROJECT_ROOT>": Path.cwd().resolve(),
        "<HOME>": Path.home().resolve(),
        "<TEMP_ROOT>": Path(tempfile.gettempdir()).resolve(),
    }


def _static_normalization_roots() -> dict[str, Path]:
    roots = _normalization_roots(Path(tempfile.gettempdir()) / "hephaestus-inactive-cache")
    roots.pop("<INDUCTOR_CACHE>")
    return roots


def _identity_strategy(dynamic: bool | None) -> str:
    if dynamic is True:
        return "dynamic"
    if dynamic is False:
        return "static"
    if dynamic is None:
        return "auto"
    raise ValueError("compile dynamic must be true, false, or null")


def run_to_bundle(
    workload_name: str,
    request: CompileRequest,
    output_root: Path,
    criteria_path: Path,
    settings: RunSettings,
    *,
    input_plan: InputPlan | None = None,
    catalog_metadata: Mapping[str, object] | None = None,
    run_provenance: RunProvenance | None = None,
) -> RunResult:
    """Run one workload and finalize a self-contained, offline-regatable bundle."""
    if not isinstance(run_provenance, RunProvenance):
        raise ValueError("trusted run provenance is required before measurement")
    if catalog_metadata is None:
        raise ValueError("trusted catalog metadata is required before measurement")
    workload = get_workload(workload_name)
    criteria_bytes = criteria_path.read_bytes()
    config = _config_evidence(request, catalog_metadata)
    normalize_public_evidence(config, _static_normalization_roots())
    evidence = measure(workload, request, settings, input_plan=input_plan)
    bundle_name = _bundle_name(workload_name, config)
    bundle_path = output_root / bundle_name
    bundle_path.mkdir(parents=True, exist_ok=False)

    write_json(bundle_path / "env.json", environment_snapshot())
    write_json(bundle_path / "workload.digest", _workload_digest(workload))
    write_json(bundle_path / "config.json", config)
    write_json(bundle_path / "run_provenance.json", run_provenance.as_json())
    write_json(bundle_path / "input_plan.json", evidence.input_plan)
    write_json(bundle_path / "timings.json", evidence.timings)
    write_json(bundle_path / "dynamo_report.json", evidence.dynamo_report)
    write_json(bundle_path / "accuracy.json", evidence.accuracy)
    write_json(bundle_path / "methodology.json", evidence.methodology)
    (bundle_path / "gate_criteria.yaml").write_bytes(criteria_bytes)

    verdict = finalize_bundle(bundle_path, _evaluate_provisional_bundle)
    verdict_name = verdict.get("verdict")
    if not isinstance(verdict_name, str):
        raise RuntimeError("gate returned a verdict without a verdict name")
    return RunResult(
        bundle_path=bundle_path,
        verdict=verdict_name,
        summary=_read_only(verdict),
    )


def _bundle_name(workload_name: str, config: dict[str, object]) -> str:
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S.%fZ")
    config_digest = hashlib.sha256(canonical_json_bytes(config)).hexdigest()[:12]
    return f"{timestamp}-{workload_name}-{config_digest}"


def _config_evidence(
    request: CompileRequest,
    catalog_metadata: Mapping[str, object] | None = None,
) -> dict[str, object]:
    config: dict[str, object] = {
        "backend": request.backend,
        "mode": request.mode,
        "dynamic": request.dynamic,
        "fullgraph": request.fullgraph,
        "options": None if request.options is None else dict(request.options),
        "disable": request.disable,
    }
    if catalog_metadata is not None:
        copied = _json_value_copy(catalog_metadata)
        if not isinstance(copied, dict):
            raise ValueError("catalog metadata must be a string-keyed mapping")
        config["catalog"] = copied
    return config


def _json_value_copy(value: object) -> object:
    if value is None or type(value) in {str, int, float, bool}:
        return value
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise ValueError("catalog metadata keys must be strings")
        return {key: _json_value_copy(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_json_value_copy(item) for item in value]
    raise ValueError("catalog metadata must contain only JSON values")


def _workload_digest(workload: WorkloadSpec) -> dict[str, object]:
    source_name = inspect.getsourcefile(workload.make_module)
    if source_name is None:
        raise RuntimeError(f"cannot resolve source for workload {workload.name!r}")
    source_bytes = Path(source_name).read_bytes()
    digest = hashlib.sha256(source_bytes).hexdigest()
    expected = V1_WORKLOAD_SHA256.get(workload.name)
    if digest != expected:
        raise RuntimeError(f"workload source bytes differ from frozen v0.1: {workload.name}")
    return {
        "schema_version": 1,
        "name": workload.name,
        "sha256": digest,
        "accuracy_tolerance": dict(V1_ACCURACY_TOLERANCE),
    }


def _read_only(value: object) -> Any:
    if isinstance(value, dict):
        return MappingProxyType({key: _read_only(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(_read_only(item) for item in value)
    return value


def _measure_interleaved(
    eager: Callable[..., Any],
    compiled: Callable[..., Any],
    case: InputCase,
    repeats: int,
    spacing_seconds: float,
    clock: _Clock,
    *,
    schema_version: int,
) -> tuple[
    tuple[float, ...],
    tuple[str, ...],
    tuple[float, ...],
    tuple[str, ...],
    tuple[float, ...],
    tuple[str, ...],
]:
    eager_seconds: list[float] = []
    eager_timestamps: list[str] = []
    baseline_seconds: list[float] = []
    baseline_timestamps: list[str] = []
    candidate_seconds: list[float] = []
    candidate_timestamps: list[str] = []

    def record(callable_: Callable[..., Any], series: str) -> None:
        started = clock.monotonic()
        callable_(*case)
        elapsed = _duration(started, clock.monotonic())
        timestamp = _timestamp(clock)
        if series == "eager":
            eager_seconds.append(elapsed)
            eager_timestamps.append(timestamp)
        elif series == "baseline":
            baseline_seconds.append(elapsed)
            baseline_timestamps.append(timestamp)
        else:
            candidate_seconds.append(elapsed)
            candidate_timestamps.append(timestamp)

    for iteration in range(repeats):
        if iteration % 2 == 0:
            record(eager, "eager")
            record(compiled, "baseline")
            record(compiled, "candidate")
        else:
            if schema_version == 2:
                record(eager, "eager")
                record(compiled, "candidate")
                record(compiled, "baseline")
            else:
                record(compiled, "candidate")
                record(compiled, "baseline")
                record(eager, "eager")
        if spacing_seconds > 0:
            clock.sleep(spacing_seconds)

    return (
        tuple(eager_seconds),
        tuple(eager_timestamps),
        tuple(baseline_seconds),
        tuple(baseline_timestamps),
        tuple(candidate_seconds),
        tuple(candidate_timestamps),
    )


def _duration(started: float, ended: float) -> float:
    elapsed = float(ended - started)
    if not math.isfinite(elapsed) or elapsed < 0:
        raise ValueError("clock produced a non-finite or negative duration")
    return elapsed


def _timestamp(clock: _Clock) -> str:
    value = clock.timestamp_utc()
    if value.tzinfo is None or value.utcoffset() != UTC.utcoffset(value):
        raise ValueError("timestamp source must return a UTC-aware datetime")
    return value.astimezone(UTC).isoformat().replace("+00:00", "Z")


def _all_case_accuracy(
    eager_outputs: tuple[Any, ...],
    compiled_outputs: tuple[Any, ...],
    *,
    dtype: str,
    atol: float,
    rtol: float,
) -> dict[str, object]:
    records = tuple(
        _accuracy_record(
            eager_output,
            compiled_output,
            case_index=case_index,
            atol=atol,
            rtol=rtol,
        )
        for case_index, (eager_output, compiled_output) in enumerate(
            zip(eager_outputs, compiled_outputs, strict=True)
        )
    )
    within_tolerance = all(record["within_tolerance"] is True for record in records)
    finite_errors = [
        error
        for record in records
        if isinstance(error := record["max_absolute_error"], float)
    ]
    first_failed_record = next(
        (record for record in records if record["within_tolerance"] is False),
        None,
    )
    return {
        "schema_version": 1,
        "dtype": dtype,
        "within_tolerance": within_tolerance,
        "case_index": (
            0 if first_failed_record is None else first_failed_record["case_index"]
        ),
        "atol": atol,
        "rtol": rtol,
        "max_absolute_error": max(finite_errors) if finite_errors else None,
        "mismatch": (
            None if first_failed_record is None else first_failed_record["mismatch"]
        ),
        "cases": records,
    }


def _accuracy_record(
    eager_output: Any,
    compiled_output: Any,
    *,
    case_index: int,
    atol: float,
    rtol: float,
) -> dict[str, object]:
    try:
        torch.testing.assert_close(compiled_output, eager_output, atol=atol, rtol=rtol)
        within_tolerance = True
    except AssertionError:
        within_tolerance = False

    max_absolute_error = _max_absolute_error(eager_output, compiled_output)
    return {
        "within_tolerance": within_tolerance,
        "case_index": case_index,
        "max_absolute_error": max_absolute_error,
        "mismatch": None
        if within_tolerance
        else _mismatch_details(eager_output, compiled_output),
    }


def _max_absolute_error(eager_output: Any, compiled_output: Any) -> float | None:
    if not isinstance(eager_output, torch.Tensor) or not isinstance(compiled_output, torch.Tensor):
        return None
    if eager_output.shape != compiled_output.shape:
        return None
    if eager_output.numel() == 0:
        return 0.0
    error = float((compiled_output - eager_output).abs().max().item())
    if not math.isfinite(error):
        return None
    return error


def _mismatch_details(eager_output: Any, compiled_output: Any) -> dict[str, object]:
    if not isinstance(eager_output, torch.Tensor) or not isinstance(compiled_output, torch.Tensor):
        return {
            "kind": "type",
            "eager_type": type(eager_output).__name__,
            "compiled_type": type(compiled_output).__name__,
        }
    if eager_output.shape != compiled_output.shape:
        return {
            "kind": "shape",
            "eager_shape": list(eager_output.shape),
            "compiled_shape": list(compiled_output.shape),
        }
    if eager_output.dtype != compiled_output.dtype:
        return {
            "kind": "dtype",
            "eager_dtype": str(eager_output.dtype),
            "compiled_dtype": str(compiled_output.dtype),
        }
    if not torch.isfinite(eager_output).all() or not torch.isfinite(compiled_output).all():
        return {"kind": "nonfinite"}
    return {"kind": "value"}


def _signed_paired_effect(baseline: float, candidate: float) -> float:
    return stable_signed_paired_effect(baseline, candidate)


def _bootstrap_absolute_medians(
    effects: tuple[float, ...], samples: int, *, seed: int
) -> tuple[float, ...]:
    random_source = random.Random(seed)
    return tuple(
        abs(
            _quantile(
                tuple(effects[random_source.randrange(len(effects))] for _ in effects),
                0.5,
            )
        )
        for _ in range(samples)
    )


def _iqr(values: tuple[float, ...]) -> float:
    return _quantile(values, 0.75) - _quantile(values, 0.25)


def _quantile(values: tuple[float, ...], percentile: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * percentile
    lower_index = math.floor(position)
    upper_index = math.ceil(position)
    if lower_index == upper_index:
        return ordered[lower_index]
    fraction = position - lower_index
    return ordered[lower_index] + fraction * (ordered[upper_index] - ordered[lower_index])


__all__ = ["RawRunEvidence", "RunResult", "RunSettings", "measure", "run_to_bundle"]

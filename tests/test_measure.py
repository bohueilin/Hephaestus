from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime, timedelta
from types import MappingProxyType

import pytest
import torch
from torch import nn

import hephaestus.measure as measure_module
from hephaestus.input_plan import build_bucketed_input_plan, build_identity_input_plan
from hephaestus.measure import RunSettings, measure
from hephaestus.torchbind import CompileRequest
from hephaestus.workloads.base import WorkloadSpec
from hephaestus.workloads.dynamic_batch_text import SPEC as DYNAMIC_BATCH_TEXT

EAGER_REQUEST = CompileRequest(
    backend="eager",
    mode=None,
    dynamic=False,
    fullgraph=False,
    options=None,
    disable=False,
)
AUTO_DYNAMIC_REQUEST = replace(EAGER_REQUEST, dynamic=None)


class _DeterministicClock:
    def __init__(self, durations: list[float]) -> None:
        self._durations = iter(durations)
        self._elapsed = 100.0
        self._pending_duration: float | None = None
        self._timestamp = datetime(2026, 8, 29, 12, 0, tzinfo=UTC)
        self.sleeps: list[float] = []

    def monotonic(self) -> float:
        if self._pending_duration is None:
            self._pending_duration = next(self._durations)
            return self._elapsed
        self._elapsed += self._pending_duration
        self._pending_duration = None
        return self._elapsed

    def timestamp_utc(self) -> datetime:
        current = self._timestamp
        self._timestamp += timedelta(seconds=1)
        return current

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)


class _Identity(nn.Module):
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return inputs * 2


class _CompileDrift(nn.Module):
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        output = inputs * 2
        if torch.compiler.is_compiling():
            output = output + 1
        return output


class _CompileNan(nn.Module):
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        output = inputs * 2
        if torch.compiler.is_compiling():
            output = torch.full_like(output, float("nan"))
        return output


class _CompileShapeMismatch(nn.Module):
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        output = inputs * 2
        if torch.compiler.is_compiling():
            output = torch.cat((output, output[:1]))
        return output


class _EvenBatchOnly(nn.Module):
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if inputs.shape[0] not in (2, 4):
            raise AssertionError("measurement did not consume bucketed effective inputs")
        return inputs * 2


class _CompileThirdCaseDrift(nn.Module):
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        output = inputs * 2
        if torch.compiler.is_compiling() and inputs.shape[0] == 3:
            output = output + 1
        return output


def _workload(module_type: type[nn.Module] = _Identity) -> WorkloadSpec:
    first = torch.tensor([1.0, 2.0], dtype=torch.float32)
    second = torch.tensor([7.0, 11.0], dtype=torch.float32)
    return WorkloadSpec(
        name="deterministic_test",
        seed=7,
        dtype=torch.float32,
        atol=1e-6,
        rtol=1e-6,
        compile_budget_seconds=1.0,
        max_recompiles=0,
        make_module=module_type,
        input_cases=lambda: ((first,), (second,)),
    )


def _four_case_workload(module_type: type[nn.Module] = _Identity) -> WorkloadSpec:
    cases = tuple(
        (torch.arange(size, dtype=torch.float32).reshape(size, 1),)
        for size in (1, 2, 3, 4)
    )
    return WorkloadSpec(
        name="four_case_test",
        seed=17,
        dtype=torch.float32,
        atol=1e-6,
        rtol=1e-6,
        compile_budget_seconds=1.0,
        max_recompiles=3,
        make_module=module_type,
        input_cases=lambda: cases,
    )


def _fresh_case_workload() -> WorkloadSpec:
    def input_cases() -> tuple[tuple[torch.Tensor, ...], ...]:
        return (
            (torch.tensor([1.0, 2.0], dtype=torch.float32),),
            (torch.tensor([7.0, 11.0], dtype=torch.float32),),
        )

    return WorkloadSpec(
        name="fresh_case_test",
        seed=23,
        dtype=torch.float32,
        atol=1e-6,
        rtol=1e-6,
        compile_budget_seconds=1.0,
        max_recompiles=0,
        make_module=_Identity,
        input_cases=input_cases,
    )


def test_run_settings_publish_frozen_defaults() -> None:
    """Changing a methodology default must be an explicit contract change."""
    assert RunSettings() == RunSettings(
        warmup_runs=5,
        repeats=31,
        bootstrap_samples=2000,
        inter_run_spacing_seconds=0.0,
    )


@pytest.mark.parametrize(
    ("field", "value"),
    [("warmup_runs", 1.5), ("repeats", 1.5), ("bootstrap_samples", 1.5)],
)
def test_run_settings_reject_noninteger_run_counts(field: str, value: float) -> None:
    """Fractional counts must fail at the settings boundary rather than later in range()."""
    with pytest.raises(ValueError, match="integer"):
        RunSettings(**{field: value})  # type: ignore[arg-type]


def test_run_settings_reject_bootstrap_count_different_from_frozen_gate() -> None:
    """A run cannot declare a sample count different from the gate computation."""
    with pytest.raises(ValueError, match="exactly 2000"):
        RunSettings(bootstrap_samples=7)


def test_measure_interleaves_local_pairs_and_preserves_bootstrap_methodology() -> None:
    """Call order must alternate while raw local pairs and bootstrap evidence stay complete."""
    clock = _DeterministicClock(
        [
            10.0,
            0.5,
            1.0,
            2.0,
            3.0,
            4.0,
            5.0,
            6.0,
            7.0,
            8.0,
            9.0,
        ]
    )
    settings = RunSettings(
        warmup_runs=2,
        repeats=3,
        bootstrap_samples=2000,
        inter_run_spacing_seconds=0.0,
    )

    evidence = measure(_workload(), EAGER_REQUEST, settings, _clock=clock)

    assert {
        key: value
        for key, value in evidence.timings.items()
        if key != "aa_bootstrap_absolute_medians"
    } == {
        "eager_seconds": (1.0, 6.0, 7.0),
        "eager_timestamps_utc": (
            "2026-08-29T12:00:02Z",
            "2026-08-29T12:00:07Z",
            "2026-08-29T12:00:08Z",
        ),
        "cold_compile_seconds": 10.0,
        "cold_compile_timestamp_utc": "2026-08-29T12:00:00Z",
        "warm_cache_compile_seconds": None,
        "warm_cache_compile_timestamp_utc": None,
        "non_primary_compile_sweep_case_indices": (1,),
        "non_primary_compile_sweep_seconds": (0.5,),
        "non_primary_compile_sweep_timestamps_utc": ("2026-08-29T12:00:01Z",),
        "compiled_seconds": (2.0, 5.0, 8.0),
        "compiled_timestamps_utc": (
            "2026-08-29T12:00:03Z",
            "2026-08-29T12:00:06Z",
            "2026-08-29T12:00:09Z",
        ),
        "aa_baseline_seconds": (2.0, 5.0, 8.0),
        "aa_baseline_timestamps_utc": (
            "2026-08-29T12:00:03Z",
            "2026-08-29T12:00:06Z",
            "2026-08-29T12:00:09Z",
        ),
        "aa_candidate_seconds": (3.0, 4.0, 9.0),
        "aa_candidate_timestamps_utc": (
            "2026-08-29T12:00:04Z",
            "2026-08-29T12:00:05Z",
            "2026-08-29T12:00:10Z",
        ),
        "aa_signed_paired_effects": (-0.4, 2.0 / 9.0, -2.0 / 17.0),
        "summary": {
            "eager_median_seconds": 6.0,
            "eager_iqr_seconds": 3.0,
            "compiled_median_seconds": 5.0,
            "compiled_iqr_seconds": 3.0,
        },
    }
    bootstrap = evidence.timings["aa_bootstrap_absolute_medians"]
    assert isinstance(bootstrap, tuple)
    assert len(bootstrap) == 2000
    assert bootstrap[:12] == (
        2.0 / 9.0,
        2.0 / 9.0,
        2.0 / 9.0,
        2.0 / 17.0,
        2.0 / 17.0,
        0.4,
        2.0 / 17.0,
        2.0 / 17.0,
        0.4,
        2.0 / 9.0,
        2.0 / 17.0,
        2.0 / 9.0,
    )
    assert evidence.methodology["aa_noise_floor"] == 0.4
    assert evidence.methodology["warmup_runs"] == 2
    assert evidence.methodology["repeats"] == 3
    assert evidence.methodology["bootstrap_samples"] == 2000
    assert evidence.methodology["bootstrap_seed"] == 0
    assert evidence.methodology["aa_effect_formula"] == "(A-B)/((A+B)/2)"
    assert evidence.methodology["aa_estimator"] == "p95_absolute_bootstrap_median"
    assert evidence.methodology["aa_pairing"] == "within_iteration"
    assert evidence.methodology["measurement_schedule"] == "alternate_eager-A-B__B-A-eager"
    assert evidence.methodology["valid"] is True
    assert clock.sleeps == []


def test_measure_paired_effect_preserves_large_finite_inputs() -> None:
    """The producer must store the same stable effect that offline gates recompute."""
    assert measure_module._signed_paired_effect(1.79e308, 1.5e308) == pytest.approx(
        0.1762917933130699
    )


@pytest.mark.parametrize(
    ("baseline", "candidate", "expected"),
    [(5e-324, 0.0, 2.0), (0.0, 5e-324, -2.0), (0.0, 0.0, 0.0)],
)
def test_measure_paired_effect_preserves_subnormal_and_zero_inputs(
    baseline: float,
    candidate: float,
    expected: float,
) -> None:
    """Producer arithmetic must match the offline stable schema-1 formula."""
    assert measure_module._signed_paired_effect(baseline, candidate) == expected


def test_measure_sweeps_all_four_cases_when_warmups_are_zero() -> None:
    """Removing warmups must not suppress non-primary compiler executions or accuracy."""
    workload = _four_case_workload()
    plan = build_identity_input_plan(workload, "static")
    clock = _DeterministicClock([0.25, 0.1, 0.2, 0.3, 1.0, 1.0, 1.0])

    evidence = measure(
        workload,
        EAGER_REQUEST,
        RunSettings(warmup_runs=0, repeats=1),
        input_plan=plan,
        _clock=clock,
    )

    assert evidence.timings["non_primary_compile_sweep_case_indices"] == (1, 2, 3)
    assert evidence.timings["non_primary_compile_sweep_seconds"] == pytest.approx(
        (0.1, 0.2, 0.3)
    )
    assert evidence.timings["non_primary_compile_sweep_timestamps_utc"] == (
        "2026-08-29T12:00:01Z",
        "2026-08-29T12:00:02Z",
        "2026-08-29T12:00:03Z",
    )
    assert [record["case_index"] for record in evidence.accuracy["cases"]] == [0, 1, 2, 3]
    assert all(record["within_tolerance"] for record in evidence.accuracy["cases"])
    assert evidence.accuracy["within_tolerance"] is True


def test_measure_uses_bucketed_effective_cases_for_both_eager_and_compiled_paths() -> None:
    """Passing a plan must replace the tensors consumed on both accuracy paths."""
    workload = replace(DYNAMIC_BATCH_TEXT, make_module=_EvenBatchOnly)
    plan = build_bucketed_input_plan(workload)
    clock = _DeterministicClock([0.25, 0.1, 0.2, 0.3, 1.0, 1.0, 1.0])

    evidence = measure(
        workload,
        EAGER_REQUEST,
        RunSettings(warmup_runs=0, repeats=1),
        input_plan=plan,
        _clock=clock,
    )

    assert evidence.accuracy["within_tolerance"] is True
    assert [record["case_index"] for record in evidence.accuracy["cases"]] == [0, 1, 2, 3]


def test_non_primary_compiler_drift_fails_aggregate_accuracy() -> None:
    """A compiler error outside case zero must invalidate the complete accuracy result."""
    workload = _four_case_workload(_CompileThirdCaseDrift)
    clock = _DeterministicClock([0.25, 0.1, 0.2, 0.3, 1.0, 1.0, 1.0])

    evidence = measure(
        workload,
        EAGER_REQUEST,
        RunSettings(warmup_runs=0, repeats=1),
        _clock=clock,
    )

    assert evidence.accuracy["within_tolerance"] is False
    assert [record["within_tolerance"] for record in evidence.accuracy["cases"]] == [
        True,
        True,
        False,
        True,
    ]
    assert evidence.accuracy["cases"][2]["max_absolute_error"] == 1.0


def test_measure_rejects_plan_strategy_that_disagrees_with_compile_request() -> None:
    """A trusted plan cannot make config.dynamic semantically false after measurement."""
    workload = _workload()
    dynamic_plan = build_identity_input_plan(workload, "dynamic")
    clock = _DeterministicClock([0.1] * 5)

    with pytest.raises(ValueError, match="dynamic strategy"):
        measure(
            workload,
            EAGER_REQUEST,
            RunSettings(warmup_runs=0, repeats=1),
            input_plan=dynamic_plan,
            _clock=clock,
        )


def test_measure_rejects_forged_effective_plan_before_compiler_reset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A caller cannot substitute module, tolerance, tensors, or serialized shapes."""
    workload = _fresh_case_workload()
    plan = build_identity_input_plan(workload, "static")
    forged_cases = (
        (torch.tensor([101.0, 202.0, 303.0], dtype=torch.float32),),
        (torch.tensor([707.0, 1111.0, 1212.0], dtype=torch.float32),),
    )
    forged_workload = replace(
        plan.workload,
        atol=999.0,
        make_module=_CompileDrift,
        input_cases=lambda: forged_cases,
    )
    forged = replace(plan, workload=forged_workload, cases=forged_cases)

    def fail_if_reset() -> None:
        pytest.fail("forged plan reached compiler reset")

    monkeypatch.setattr(measure_module, "reset_compiler_state", fail_if_reset)

    with pytest.raises(ValueError, match="canonical"):
        measure(
            workload,
            EAGER_REQUEST,
            RunSettings(warmup_runs=0, repeats=1),
            input_plan=forged,
            _clock=_DeterministicClock([0.1] * 5),
        )


def test_measure_rejects_factory_plan_mutated_after_construction_before_reset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """In-place tensor mutation cannot change the canonical cases a run consumes."""
    workload = _fresh_case_workload()
    plan = build_identity_input_plan(workload, "static")
    plan.cases[0][0].add_(1000.0)

    def fail_if_reset() -> None:
        pytest.fail("mutated plan reached compiler reset")

    monkeypatch.setattr(measure_module, "reset_compiler_state", fail_if_reset)

    with pytest.raises(ValueError, match="canonical"):
        measure(
            workload,
            EAGER_REQUEST,
            RunSettings(warmup_runs=0, repeats=1),
            input_plan=plan,
            _clock=_DeterministicClock([0.1] * 5),
        )


def test_measure_rejects_plan_evidence_not_derived_from_canonical_cases(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Serialized plan shapes cannot be supplied independently of consumed tensors."""
    workload = _fresh_case_workload()
    plan = build_identity_input_plan(workload, "static")
    forged_evidence = dict(plan.evidence)
    forged_evidence["effective_shapes"] = ((999,), (999,))
    forged = replace(plan, evidence=MappingProxyType(forged_evidence))

    def fail_if_reset() -> None:
        pytest.fail("forged evidence reached compiler reset")

    monkeypatch.setattr(measure_module, "reset_compiler_state", fail_if_reset)

    with pytest.raises(ValueError, match="canonical"):
        measure(
            workload,
            EAGER_REQUEST,
            RunSettings(warmup_runs=0, repeats=1),
            input_plan=forged,
            _clock=_DeterministicClock([0.1] * 5),
        )


def test_measure_accepts_torch_auto_dynamic_request_as_truthful_identity_plan() -> None:
    """CompileRequest.dynamic=None must retain Torch auto mode without claiming static."""
    evidence = measure(
        _workload(),
        AUTO_DYNAMIC_REQUEST,
        RunSettings(warmup_runs=0, repeats=1),
        _clock=_DeterministicClock([0.1] * 5),
    )

    assert evidence.input_plan["dynamic_strategy"] == "auto"
    assert evidence.input_plan["bucket_axis"] is None
    assert evidence.input_plan["original_shapes"] == evidence.input_plan["effective_shapes"]
    assert evidence.accuracy["within_tolerance"] is True


def test_accuracy_compares_eager_and_compiled_on_the_same_pinned_first_case() -> None:
    """Using a different pinned case on either side must fail this distinct-input fixture."""
    clock = _DeterministicClock([0.1] * 5)
    settings = RunSettings(
        warmup_runs=0,
        repeats=1,
        bootstrap_samples=2000,
        inter_run_spacing_seconds=0.0,
    )

    evidence = measure(_workload(), EAGER_REQUEST, settings, _clock=clock)

    assert evidence.accuracy["within_tolerance"] is True
    assert evidence.accuracy["case_index"] == 0
    assert evidence.accuracy["atol"] == 1e-6
    assert evidence.accuracy["rtol"] == 1e-6


def test_out_of_tolerance_compiled_output_marks_accuracy_false() -> None:
    """A compiler-only output drift must invalidate accuracy evidence."""
    clock = _DeterministicClock([0.1] * 5)
    settings = RunSettings(
        warmup_runs=0,
        repeats=1,
        bootstrap_samples=2000,
        inter_run_spacing_seconds=0.0,
    )

    evidence = measure(_workload(_CompileDrift), EAGER_REQUEST, settings, _clock=clock)

    assert evidence.accuracy["within_tolerance"] is False
    assert evidence.accuracy["max_absolute_error"] == 1.0


def test_nonfinite_compiled_output_records_failed_accuracy_instead_of_aborting() -> None:
    """NaN output must remain gateable invalid evidence rather than preventing a bundle."""
    clock = _DeterministicClock([0.1] * 5)
    settings = RunSettings(
        warmup_runs=0,
        repeats=1,
        bootstrap_samples=2000,
        inter_run_spacing_seconds=0.0,
    )

    evidence = measure(_workload(_CompileNan), EAGER_REQUEST, settings, _clock=clock)

    assert evidence.accuracy["within_tolerance"] is False
    assert evidence.accuracy["max_absolute_error"] is None


def test_shape_mismatch_records_failed_accuracy_instead_of_aborting() -> None:
    """Different compiled shape must become deterministic invalid accuracy evidence."""
    clock = _DeterministicClock([0.1] * 5)
    settings = RunSettings(
        warmup_runs=0,
        repeats=1,
        bootstrap_samples=2000,
        inter_run_spacing_seconds=0.0,
    )

    evidence = measure(_workload(_CompileShapeMismatch), EAGER_REQUEST, settings, _clock=clock)

    assert evidence.accuracy["within_tolerance"] is False
    assert evidence.accuracy["max_absolute_error"] is None
    assert evidence.accuracy["mismatch"] == {
        "kind": "shape",
        "eager_shape": [2],
        "compiled_shape": [3],
    }

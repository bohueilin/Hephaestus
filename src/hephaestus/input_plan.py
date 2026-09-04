"""Trusted, immutable input plans for compiler measurement runs."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
from types import MappingProxyType
from typing import Final, Literal

import torch

from hephaestus.workloads.base import InputCase, WorkloadSpec

DynamicStrategy = Literal["static", "dynamic", "auto", "bucketed"]

_BUCKETED_WORKLOAD: Final = "dynamic_batch_text"
_BUCKET_AXIS: Final = 0
_BUCKET_BOUNDARIES: Final = (2, 4)
_BUCKET_OVERFLOW_RULE: Final = "reject"


@dataclass(frozen=True, slots=True)
class InputPlan:
    """Effective workload inputs plus immutable, serializable plan evidence."""

    source_workload: WorkloadSpec
    workload: WorkloadSpec
    cases: tuple[InputCase, ...]
    evidence: Mapping[str, object]


@dataclass(frozen=True, slots=True)
class _CaseSupplier:
    cases: tuple[InputCase, ...]

    def __call__(self) -> tuple[InputCase, ...]:
        return self.cases


def build_identity_input_plan(workload: WorkloadSpec, strategy: str) -> InputPlan:
    """Build a static, dynamic, or Torch-auto plan without changing pinned inputs."""
    if strategy not in {"static", "dynamic", "auto"}:
        raise ValueError("identity strategy must be 'static', 'dynamic', or 'auto'")
    cases = _clone_cases(_validated_cases(workload))
    return _plan(workload, cases, strategy=strategy)


def build_bucketed_input_plan(
    workload: WorkloadSpec,
    *,
    boundaries: tuple[int, ...] = _BUCKET_BOUNDARIES,
) -> InputPlan:
    """Pad dynamic-text batches to the frozen inclusive v1 bucket boundaries."""
    if workload.name != _BUCKETED_WORKLOAD:
        raise ValueError("bucketed input plans are legal only for dynamic_batch_text")
    parsed_boundaries = _validated_boundaries(boundaries)
    original_cases = _clone_cases(_validated_cases(workload))
    maximum_batch = max(case[0].shape[_BUCKET_AXIS] for case in original_cases)
    if parsed_boundaries[-1] < maximum_batch:
        raise ValueError("bucket boundaries must cover every batch")
    if parsed_boundaries != _BUCKET_BOUNDARIES:
        raise ValueError("v1 bucket boundaries must be exactly (2, 4)")

    effective_cases: list[InputCase] = []
    for case in original_cases:
        inputs = case[0]
        batch = inputs.shape[_BUCKET_AXIS]
        target = next(boundary for boundary in parsed_boundaries if batch <= boundary)
        if target == batch:
            effective = inputs
        else:
            padding = torch.zeros(
                (target - batch, *inputs.shape[1:]),
                dtype=inputs.dtype,
                device=inputs.device,
            )
            effective = torch.cat((inputs, padding), dim=_BUCKET_AXIS)
        effective_cases.append((effective,))

    return _plan(
        workload,
        tuple(effective_cases),
        strategy="bucketed",
        original_cases=original_cases,
        bucket_boundaries=parsed_boundaries,
    )


def input_plan_json(plan: InputPlan) -> dict[str, object]:
    """Return a mutable JSON-compatible copy without exposing mutable plan evidence."""
    return dict(plan.evidence)


def canonicalize_input_plan(workload: WorkloadSpec, plan: InputPlan) -> InputPlan:
    """Reject untrusted plan drift and return a fresh source-derived reconstruction."""
    if not isinstance(plan, InputPlan) or plan.source_workload is not workload:
        raise ValueError("input plan does not match the canonical source workload")
    if not isinstance(plan.evidence, Mapping):
        raise ValueError("input plan has no canonical evidence")
    strategy = plan.evidence.get("dynamic_strategy")
    if not isinstance(strategy, str):
        raise ValueError("input plan has no canonical dynamic strategy")
    if strategy == "bucketed":
        canonical = build_bucketed_input_plan(workload)
    elif strategy in {"static", "dynamic", "auto"}:
        canonical = build_identity_input_plan(workload, strategy)
    else:
        raise ValueError("input plan has no canonical dynamic strategy")
    if (
        not _same_effective_workload(plan, canonical)
        or not _same_cases(plan.cases, canonical.cases)
        or not _same_value(plan.evidence, canonical.evidence)
    ):
        raise ValueError("input plan differs from its canonical reconstruction")
    return canonical


def _plan(
    source_workload: WorkloadSpec,
    effective_cases: tuple[InputCase, ...],
    *,
    strategy: str,
    original_cases: tuple[InputCase, ...] | None = None,
    bucket_boundaries: tuple[int, ...] | None = None,
) -> InputPlan:
    original = effective_cases if original_cases is None else original_cases
    effective_workload = replace(
        source_workload,
        input_cases=_CaseSupplier(effective_cases),
    )
    is_bucketed = strategy == "bucketed"
    evidence = MappingProxyType(
        {
            "schema_version": 1,
            "dynamic_strategy": strategy,
            "bucket_axis": _BUCKET_AXIS if is_bucketed else None,
            "bucket_boundaries": bucket_boundaries if is_bucketed else None,
            "bucket_overflow_rule": _BUCKET_OVERFLOW_RULE if is_bucketed else None,
            "original_shapes": tuple(_shape(case) for case in original),
            "effective_shapes": tuple(_shape(case) for case in effective_cases),
            "compile_sweep_case_indices": tuple(range(len(effective_cases))),
            "steady_state_case_index": 0,
        }
    )
    return InputPlan(
        source_workload=source_workload,
        workload=effective_workload,
        cases=effective_cases,
        evidence=evidence,
    )


def _validated_cases(workload: WorkloadSpec) -> tuple[InputCase, ...]:
    cases = workload.input_cases()
    if not isinstance(cases, tuple) or not cases:
        raise ValueError("workload must provide nonempty tuple input cases")
    for case in cases:
        if (
            not isinstance(case, tuple)
            or len(case) != 1
            or not isinstance(case[0], torch.Tensor)
            or case[0].ndim < 1
            or case[0].shape[0] <= 0
        ):
            raise ValueError("input plans require one nonempty tensor per case")
    return cases


def _clone_cases(cases: tuple[InputCase, ...]) -> tuple[InputCase, ...]:
    cloned_cases: list[InputCase] = []
    for case in cases:
        cloned_tensors: list[torch.Tensor] = []
        for tensor in case:
            cloned = tensor.detach().clone(memory_format=torch.preserve_format)
            cloned.requires_grad_(tensor.requires_grad)
            cloned_tensors.append(cloned)
        cloned_cases.append(tuple(cloned_tensors))
    return tuple(cloned_cases)


def _same_effective_workload(actual: InputPlan, expected: InputPlan) -> bool:
    actual_workload = actual.workload
    expected_workload = expected.workload
    if not isinstance(actual_workload, WorkloadSpec):
        return False
    scalar_fields = (
        "name",
        "seed",
        "dtype",
        "atol",
        "rtol",
        "compile_budget_seconds",
        "max_recompiles",
    )
    if any(
        type(getattr(actual_workload, field)) is not type(getattr(expected_workload, field))
        or getattr(actual_workload, field) != getattr(expected_workload, field)
        for field in scalar_fields
    ):
        return False
    supplier = actual_workload.input_cases
    return (
        actual_workload.make_module is expected_workload.make_module
        and isinstance(supplier, _CaseSupplier)
        and supplier.cases is actual.cases
    )


def _same_cases(
    actual_cases: tuple[InputCase, ...], expected_cases: tuple[InputCase, ...]
) -> bool:
    if not isinstance(actual_cases, tuple) or len(actual_cases) != len(expected_cases):
        return False
    for actual_case, expected_case in zip(actual_cases, expected_cases, strict=True):
        if not isinstance(actual_case, tuple) or len(actual_case) != len(expected_case):
            return False
        for actual, expected in zip(actual_case, expected_case, strict=True):
            if not isinstance(actual, torch.Tensor):
                return False
            if (
                actual.shape != expected.shape
                or actual.dtype != expected.dtype
                or actual.device != expected.device
                or actual.layout != expected.layout
                or actual.requires_grad != expected.requires_grad
                or actual.stride() != expected.stride()
                or not torch.equal(actual, expected)
            ):
                return False
    return True


def _same_value(actual: object, expected: object) -> bool:
    if type(actual) is not type(expected):
        return False
    if isinstance(expected, Mapping):
        return isinstance(actual, Mapping) and actual.keys() == expected.keys() and all(
            _same_value(actual[key], expected[key]) for key in expected
        )
    if isinstance(expected, tuple):
        return isinstance(actual, tuple) and len(actual) == len(expected) and all(
            _same_value(left, right)
            for left, right in zip(actual, expected, strict=True)
        )
    return bool(actual == expected)


def _validated_boundaries(boundaries: tuple[int, ...]) -> tuple[int, ...]:
    if not isinstance(boundaries, tuple) or not boundaries:
        raise ValueError("bucket boundaries must be a nonempty tuple")
    if any(
        isinstance(boundary, bool) or not isinstance(boundary, int) or boundary <= 0
        for boundary in boundaries
    ):
        raise ValueError("bucket boundaries must be positive integers")
    if any(left >= right for left, right in zip(boundaries, boundaries[1:], strict=False)):
        raise ValueError("bucket boundaries must be strictly increasing")
    return boundaries


def _shape(case: InputCase) -> tuple[int, ...]:
    return tuple(case[0].shape)


__all__ = [
    "DynamicStrategy",
    "InputPlan",
    "build_bucketed_input_plan",
    "build_identity_input_plan",
    "canonicalize_input_plan",
    "input_plan_json",
]

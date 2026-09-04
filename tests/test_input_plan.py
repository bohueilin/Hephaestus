from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from hephaestus.input_plan import (
    build_bucketed_input_plan,
    build_identity_input_plan,
)
from hephaestus.workloads.dynamic_batch_text import SPEC as DYNAMIC_BATCH_TEXT
from hephaestus.workloads.mlp_stack import SPEC as MLP_STACK


def test_bucketed_plan_pads_only_batch_rows_and_publishes_immutable_evidence() -> None:
    """Wrong-axis padding, value drift, or mutable evidence must break the plan contract."""
    original_cases = DYNAMIC_BATCH_TEXT.input_cases()

    plan = build_bucketed_input_plan(DYNAMIC_BATCH_TEXT)

    assert [list(case[0].shape) for case in original_cases] == [
        [2, 24, 32],
        [1, 48, 32],
        [3, 16, 32],
        [4, 12, 32],
    ]
    assert [list(case[0].shape) for case in plan.cases] == [
        [2, 24, 32],
        [2, 48, 32],
        [4, 16, 32],
        [4, 12, 32],
    ]
    for original_case, effective_case in zip(original_cases, plan.cases, strict=True):
        original = original_case[0]
        effective = effective_case[0]
        torch.testing.assert_close(effective[: original.shape[0]], original)
        if effective.shape[0] > original.shape[0]:
            assert torch.count_nonzero(effective[original.shape[0] :]).item() == 0
        assert effective.shape[1:] == original.shape[1:]
        assert effective.dtype == original.dtype
        assert effective.device == original.device

    assert plan.workload.input_cases() == plan.cases
    assert dict(plan.evidence) == {
        "schema_version": 1,
        "dynamic_strategy": "bucketed",
        "bucket_axis": 0,
        "bucket_boundaries": (2, 4),
        "bucket_overflow_rule": "reject",
        "original_shapes": (
            (2, 24, 32),
            (1, 48, 32),
            (3, 16, 32),
            (4, 12, 32),
        ),
        "effective_shapes": (
            (2, 24, 32),
            (2, 48, 32),
            (4, 16, 32),
            (4, 12, 32),
        ),
        "compile_sweep_case_indices": (0, 1, 2, 3),
        "steady_state_case_index": 0,
    }
    with pytest.raises(TypeError):
        plan.evidence["dynamic_strategy"] = "static"  # type: ignore[index]


@pytest.mark.parametrize("strategy", ["static", "dynamic", "auto"])
def test_identity_plan_preserves_cases_and_uses_null_bucket_fields(strategy: str) -> None:
    """An identity strategy must not silently alter tensors or claim bucket behavior."""
    original_cases = DYNAMIC_BATCH_TEXT.input_cases()

    plan = build_identity_input_plan(DYNAMIC_BATCH_TEXT, strategy)

    assert len(plan.cases) == len(original_cases)
    for original_case, effective_case in zip(original_cases, plan.cases, strict=True):
        torch.testing.assert_close(effective_case[0], original_case[0])
    assert plan.workload.input_cases() is plan.cases
    assert plan.evidence["dynamic_strategy"] == strategy
    assert plan.evidence["bucket_axis"] is None
    assert plan.evidence["bucket_boundaries"] is None
    assert plan.evidence["bucket_overflow_rule"] is None
    assert plan.evidence["original_shapes"] == plan.evidence["effective_shapes"]
    assert plan.evidence["compile_sweep_case_indices"] == (0, 1, 2, 3)


@pytest.mark.parametrize(
    ("boundaries", "message"),
    [
        ((), "nonempty"),
        ((False, 4), "positive integers"),
        ((0, 4), "positive integers"),
        ((2, 2), "strictly increasing"),
        ((4, 2), "strictly increasing"),
        ((2,), "cover every batch"),
    ],
)
def test_bucketed_plan_rejects_invalid_boundaries(
    boundaries: tuple[int, ...], message: str
) -> None:
    """Invalid bucket geometry must fail before it can become trusted run input."""
    with pytest.raises(ValueError, match=message):
        build_bucketed_input_plan(DYNAMIC_BATCH_TEXT, boundaries=boundaries)


def test_bucketed_plan_rejects_nonfrozen_boundaries_and_wrong_workload() -> None:
    """A valid-looking alternative or unrelated workload cannot impersonate v1 bucketing."""
    with pytest.raises(ValueError, match="exactly"):
        build_bucketed_input_plan(DYNAMIC_BATCH_TEXT, boundaries=(2, 5))
    with pytest.raises(ValueError, match="dynamic_batch_text"):
        build_bucketed_input_plan(MLP_STACK)


def test_bucketed_plan_rejects_uncovered_overflow_case() -> None:
    """An input above the last inclusive boundary must reject instead of truncating."""
    overflow = replace(
        DYNAMIC_BATCH_TEXT,
        input_cases=lambda: ((torch.zeros(5, 12, 32),),),
    )

    with pytest.raises(ValueError, match="cover every batch"):
        build_bucketed_input_plan(overflow)

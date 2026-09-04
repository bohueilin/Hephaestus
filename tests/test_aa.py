from __future__ import annotations

import math
from dataclasses import replace

import pytest

from hephaestus.aa import (
    AANullStatistics,
    compute_null_statistics,
    evaluate_aa_statistics,
    validate_stored_null_statistics,
)


def _statistics(
    *,
    floor: float,
    a_over_b: float = 1.0,
    b_over_a: float = 1.0,
) -> AANullStatistics:
    return AANullStatistics(
        signed_effects=(0.0,) * 31,
        bootstrap_absolute_medians=(floor,) * 2000,
        p95_noise_floor=floor,
        speedup_lower_bound_a_over_b=a_over_b,
        speedup_lower_bound_b_over_a=b_over_a,
    )


def test_null_statistics_use_exact_paired_math_and_seeded_bootstrap() -> None:
    """Changing the signed formula or bootstrap RNG/order must change this fixture."""
    series_a = tuple(float(index + 1) for index in range(31))
    series_b = tuple(float(index + 2) for index in range(31))

    first = compute_null_statistics(series_a, series_b)
    second = compute_null_statistics(series_a, series_b)

    assert first == second
    assert first.signed_effects[:5] == pytest.approx(
        (
            -0.6666666666666666,
            -0.4,
            -0.2857142857142857,
            -0.2222222222222222,
            -0.18181818181818182,
        )
    )
    assert first.bootstrap_absolute_medians[:5] == pytest.approx(
        (
            0.05714285714285714,
            0.05405405405405406,
            0.046511627906976744,
            0.05405405405405406,
            0.08695652173913043,
        )
    )
    assert len(first.bootstrap_absolute_medians) == 2000
    assert first.p95_noise_floor == pytest.approx(0.08)
    assert first.speedup_lower_bound_a_over_b == pytest.approx(0.6111111111111112)
    assert first.speedup_lower_bound_b_over_a == pytest.approx(0.7055882352941177)


def test_zero_denominator_effect_is_exactly_zero() -> None:
    """A zero/zero ordinal pair must not create a NaN that poisons stored evidence."""
    result = compute_null_statistics((0.0,) * 31, (0.0,) * 31)

    assert result.signed_effects == (0.0,) * 31
    assert result.bootstrap_absolute_medians == (0.0,) * 2000
    assert result.p95_noise_floor == 0.0
    assert result.speedup_lower_bound_a_over_b is None
    assert result.speedup_lower_bound_b_over_a is None


def test_large_finite_timings_preserve_effect_and_cannot_false_pass() -> None:
    """Finite inputs whose naive sum overflows must retain the declared paired effect."""
    high = 1.79e308
    low = 1.5e308
    series_a = tuple(high if index % 2 == 0 else low for index in range(31))
    series_b = tuple(low if index % 2 == 0 else high for index in range(31))

    result = compute_null_statistics(series_a, series_b)

    expected_magnitude = 0.1762917933130699
    assert result.signed_effects == pytest.approx(
        tuple(
            expected_magnitude if index % 2 == 0 else -expected_magnitude
            for index in range(31)
        )
    )
    assert all(math.isfinite(value) for value in result.signed_effects)
    assert all(math.isfinite(value) for value in result.bootstrap_absolute_medians)
    assert math.isfinite(result.p95_noise_floor)
    assert result.speedup_lower_bound_a_over_b is not None
    assert result.speedup_lower_bound_b_over_a is not None
    assert math.isfinite(result.speedup_lower_bound_a_over_b)
    assert math.isfinite(result.speedup_lower_bound_b_over_a)
    assert evaluate_aa_statistics(result, minimum_speedup=1.10) == (
        "FAIL",
        "methodology.noise_floor",
    )


def test_null_statistics_reject_nonfinite_derived_directional_ratio() -> None:
    """A finite-input directional ratio that overflows cannot enter evidence."""
    with pytest.raises(ValueError, match="finite"):
        compute_null_statistics((1.79e308,) * 31, (1e-308,) * 31)


@pytest.mark.parametrize(
    ("left", "right", "expected"),
    [
        (5e-324, 0.0, 2.0),
        (0.0, 5e-324, -2.0),
        (0.0, 0.0, 0.0),
    ],
)
def test_null_statistics_preserve_subnormal_and_zero_paired_effects(
    left: float,
    right: float,
    expected: float,
) -> None:
    """Underflow in the literal half-sum must not crash or erase a paired effect."""
    statistics = compute_null_statistics([left] * 31, [right] * 31)

    assert statistics.signed_effects == (expected,) * 31
    assert statistics.bootstrap_absolute_medians == (abs(expected),) * 2000
    assert statistics.p95_noise_floor == abs(expected)


def test_noise_floor_requires_strict_clearance_of_configured_effect() -> None:
    """Weakening strict greater-than to greater-than-or-equal must fail at 0.10."""
    equal = evaluate_aa_statistics(_statistics(floor=0.10), minimum_speedup=1.10)
    clears = evaluate_aa_statistics(_statistics(floor=0.0999), minimum_speedup=1.10)

    assert equal == ("FAIL", "methodology.noise_floor")
    assert clears == ("PASS", "all_criteria_passed")


@pytest.mark.parametrize(
    ("a_over_b", "b_over_a"),
    ((1.10, 1.0), (1.0, 1.10)),
)
def test_directional_false_positive_fails_in_either_direction(
    a_over_b: float,
    b_over_a: float,
) -> None:
    """Ignoring either directional lower bound would accept a systematic null effect."""
    result = evaluate_aa_statistics(
        _statistics(floor=0.01, a_over_b=a_over_b, b_over_a=b_over_a),
        minimum_speedup=1.10,
    )

    assert result == ("FAIL", "aa.false_positive")


@pytest.mark.parametrize(
    ("series_a", "series_b"),
    (
        ((), ()),
        ((1.0,) * 30, (1.0,) * 30),
        ((1.0,) * 31, (1.0,) * 30),
        ((float("nan"),) + (1.0,) * 30, (1.0,) * 31),
        ((float("inf"),) + (1.0,) * 30, (1.0,) * 31),
        ((-1.0,) + (1.0,) * 30, (1.0,) * 31),
    ),
)
def test_null_statistics_reject_invalid_nonfinite_or_wrong_length_series(
    series_a: tuple[float, ...],
    series_b: tuple[float, ...],
) -> None:
    """Malformed raw series must not be converted into a scientific null verdict."""
    with pytest.raises(ValueError):
        compute_null_statistics(series_a, series_b)


def test_stored_bootstrap_derivation_is_recomputed_before_use() -> None:
    """Rehashing a forged stored bootstrap distribution cannot alter the pure decision."""
    series_a = tuple(float(index + 1) for index in range(31))
    series_b = tuple(float(index + 2) for index in range(31))
    authentic = compute_null_statistics(series_a, series_b)
    forged = replace(
        authentic,
        bootstrap_absolute_medians=(0.0,) * 2000,
        p95_noise_floor=0.0,
    )

    with pytest.raises(ValueError, match="aa.bootstrap"):
        validate_stored_null_statistics(series_a, series_b, forged)

"""Torch-free fixed-statistics tests for drift-trace analysis."""

from __future__ import annotations

import json
import math
from pathlib import Path
from types import SimpleNamespace

import pytest

from tools import drift_trace_analysis

PERCENTILES = (0.5, 0.857, 0.95, 0.99, 0.99488)


def _series_block(value: float) -> dict[str, object]:
    return {
        "baseline_seconds": [value] * 64,
        "eager_seconds": [value + 1.0] * 64,
        "candidate_seconds": [value + 2.0] * 64,
    }


def test_series_summary_uses_schema_two_and_publishes_fixed_tail_statistics() -> None:
    floors = [0.01, 0.07, 0.10, 0.20]
    blocks: list[dict[str, object]] = []
    pairs: list[dict[str, object]] = []
    for index, floor in enumerate(floors):
        blocks.extend([_series_block(floor), _series_block(floor + 0.001)])
        pairs.append({"i": index * 2, "j": index * 2 + 1, "lag_seconds": 2.6})

    seen_schema_versions: list[int] = []

    def fake_compute(
        series_a: list[float], series_b: list[float], *, schema_version: int
    ) -> SimpleNamespace:
        seen_schema_versions.append(schema_version)
        floor = series_a[0]
        assert series_b[0] == pytest.approx(floor + 0.001)
        return SimpleNamespace(
            p95_noise_floor=floor,
            speedup_lower_bound_a_over_b=1.11 if floor >= 0.10 else 1.01,
            speedup_lower_bound_b_over_a=1.02,
        )

    summary = drift_trace_analysis.analyze_series(
        blocks,
        pairs,
        series_name="baseline_seconds",
        compute_statistics=fake_compute,
        quantile=drift_trace_analysis.aa_quantile,
        bootstrap_resamples=20,
    )

    assert seen_schema_versions == [2, 2, 2, 2]
    assert summary["sorted_pair_floors"] == floors
    assert summary["percentiles"]["50"]["value"] == pytest.approx(0.085)
    assert summary["percentiles"]["85.7"]["value"] == pytest.approx(0.1571)
    assert summary["percentiles"]["95"]["value"] == pytest.approx(0.185)
    assert summary["percentiles"]["99"]["value"] == pytest.approx(0.197)
    assert summary["percentiles"]["99.488"]["value"] == pytest.approx(0.198464)
    assert summary["at_or_above_0.10"] == {"count": 2, "fraction": 0.5}
    assert summary["at_or_above_0.07"] == {"count": 3, "fraction": 0.75}
    assert summary["false_positive_arm"] == {"count": 2, "fraction": 0.5}
    assert set(summary["pairs"][0]) == {
        "i",
        "j",
        "lag_seconds",
        "p95_noise_floor",
        "speedup_lower_bound_a_over_b",
        "speedup_lower_bound_b_over_a",
        "false_positive_arm_fires",
    }


def test_moving_bootstrap_is_deterministic_and_autocorrelation_has_fixed_shape() -> None:
    values = [float(index % 7) / 100.0 for index in range(80)]
    first = drift_trace_analysis.moving_block_bootstrap_interval(
        values,
        0.95,
        quantile=drift_trace_analysis.aa_quantile,
        block_length=64,
        resamples=100,
        seed=0,
    )
    second = drift_trace_analysis.moving_block_bootstrap_interval(
        values,
        0.95,
        quantile=drift_trace_analysis.aa_quantile,
        block_length=64,
        resamples=100,
        seed=0,
    )
    autocorrelations = drift_trace_analysis.autocorrelations(values, max_lag=64)
    hand_checked = drift_trace_analysis.autocorrelations([1.0, 2.0, 3.0, 4.0], max_lag=4)

    assert first == second
    assert len(first) == 2
    assert first[0] <= first[1]
    assert len(autocorrelations) == 64
    assert autocorrelations[0]["lag"] == 1
    assert autocorrelations[-1]["lag"] == 64
    assert hand_checked == [
        {"lag": 1, "value": 0.25},
        {"lag": 2, "value": -0.3},
        {"lag": 3, "value": -0.45},
        {"lag": 4, "value": None},
    ]
    assert 1.0 <= drift_trace_analysis.implied_effective_sample_size(values, max_lag=64) <= 80


def test_default_seeded_bootstrap_interval_and_hand_computed_ess() -> None:
    assert drift_trace_analysis.moving_block_bootstrap_interval(
        [float(index) for index in range(80)],
        0.5,
        quantile=drift_trace_analysis.aa_quantile,
    ) == [23.5, 39.5]
    assert (
        drift_trace_analysis.implied_effective_sample_size(
            [1.0, 1.0, 1.0, 2.0, 2.0, 2.0], max_lag=1
        )
        == 3.0
    )


def test_b_over_a_bound_alone_fires_false_positive_arm() -> None:
    blocks = [_series_block(0.01), _series_block(0.02)]

    def fake_compute(*_args: object, **_kwargs: object) -> SimpleNamespace:
        return SimpleNamespace(
            p95_noise_floor=0.03,
            speedup_lower_bound_a_over_b=1.01,
            speedup_lower_bound_b_over_a=1.10,
        )

    summary = drift_trace_analysis.analyze_series(
        blocks,
        [{"i": 0, "j": 1, "lag_seconds": 2.6}],
        series_name="baseline_seconds",
        compute_statistics=fake_compute,
        quantile=drift_trace_analysis.aa_quantile,
        bootstrap_resamples=2,
    )

    assert summary["pairs"][0]["false_positive_arm_fires"] is True
    assert summary["false_positive_arm"] == {"count": 1, "fraction": 1.0}


@pytest.mark.parametrize(
    ("baseline_load", "threads", "load_series", "expected"),
    [
        (
            1.0,
            4,
            [
                {"t": 0.0, "load_average": [1.5, 2.0, 3.0]},
                {"t": 120.2, "load_average": [5.0, 2.0, 3.0]},
                {"t": 121.3, "load_average": [5.9, 2.0, 3.0]},
            ],
            "quiet",
        ),
        (
            1.0,
            4,
            [
                {"t": 0.0, "load_average": [1.5, 2.0, 3.0]},
                {"t": 120.2, "load_average": [5.0, 2.0, 3.0]},
                {"t": 121.3, "load_average": [6.1, 2.0, 3.0]},
            ],
            "disturbed",
        ),
        (
            1.0,
            4,
            [
                {"t": 0.0, "load_average": [2.1, 2.0, 3.0]},
                {"t": 120.2, "load_average": [5.0, 2.0, 3.0]},
            ],
            "disturbed",
        ),
        (
            1.0,
            4,
            [
                {"t": 0.0, "load_average": [1.5, 2.0, 3.0]},
                {"t": 120.2, "load_average": [6.1, 2.0, 3.0]},
            ],
            "disturbed",
        ),
        (
            1.0,
            4,
            [
                {"t": 0.0, "load_average": [None, None, None]},
                {"t": 120.2, "load_average": [5.0, 2.0, 3.0]},
            ],
            "disturbed",
        ),
    ],
)
def test_host_classification_uses_only_recorded_load_numbers(
    baseline_load: float,
    threads: int,
    load_series: list[dict[str, object]],
    expected: str,
) -> None:
    assert (
        drift_trace_analysis.classify_host(
            baseline_load=baseline_load,
            torch_num_threads=threads,
            load_series=load_series,
        )
        == expected
    )


@pytest.mark.parametrize(
    "malformed",
    [
        None,
        [],
        {"load_average": [2.0, 2.0, 3.0]},
        {"t": None, "load_average": [2.0, 2.0, 3.0]},
        {"t": True, "load_average": [2.0, 2.0, 3.0]},
        {"t": math.inf, "load_average": [2.0, 2.0, 3.0]},
        {"t": math.nan, "load_average": [2.0, 2.0, 3.0]},
        {"t": "60", "load_average": [2.0, 2.0, 3.0]},
    ],
)
def test_malformed_load_timestamp_is_disturbed_not_ignored_or_raised(malformed: object) -> None:
    assert (
        drift_trace_analysis.classify_host(
            baseline_load=1.0,
            torch_num_threads=4,
            load_series=[
                {"t": 0.0, "load_average": [1.5, 2.0, 3.0]},
                malformed,
                {"t": 120.2, "load_average": [5.0, 2.0, 3.0]},
            ],
        )
        == "disturbed"
    )


def test_analyze_trace_routes_fixed_series_and_keeps_load_classification(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[tuple[float, float, int]] = []

    def fake_compute(
        series_a: list[float], series_b: list[float], *, schema_version: int
    ) -> SimpleNamespace:
        calls.append((series_a[0], series_b[0], schema_version))
        return SimpleNamespace(
            p95_noise_floor=series_a[0] / 100.0,
            speedup_lower_bound_a_over_b=1.01,
            speedup_lower_bound_b_over_a=1.02,
        )

    monkeypatch.setattr(drift_trace_analysis, "compute_null_statistics", fake_compute)
    rows = [
        {"row": "header", "baseline_load": 1.0, "torch_num_threads": 4},
        {
            "row": "block",
            "baseline_timestamps_utc": ["2026-09-01T12:00:00.000000Z"],
            "baseline_seconds": [1.0] * 64,
            "eager_seconds": [2.0] * 64,
            "candidate_seconds": [3.0] * 64,
        },
        {
            "row": "block",
            "baseline_timestamps_utc": ["2026-09-01T12:00:02.600000Z"],
            "baseline_seconds": [1.1] * 64,
            "eager_seconds": [2.1] * 64,
            "candidate_seconds": [3.1] * 64,
        },
        {
            "row": "trailer",
            "stop_reason": "RuntimeError: measurement failed",
            "load_series": [
                {"t": 0.0, "load_average": [1.5, 2.0, 3.0]},
                {"t": 120.2, "load_average": [5.0, 2.0, 3.0]},
                {"t": 121.3, "load_average": [5.9, 2.0, 3.0]},
            ],
        },
    ]
    path = tmp_path / "synthetic.jsonl"
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")

    result = drift_trace_analysis.analyze_trace(path)

    assert result["classification"] == "quiet"
    assert calls == [(1.0, 1.1, 2), (2.0, 2.1, 2)]
    assert result["compiled"]["sorted_pair_floors"] == [0.01]
    assert result["eager"]["sorted_pair_floors"] == [0.02]
    assert all(3.0 not in call and 3.1 not in call for call in calls)


def test_fixed_percentile_set_is_complete() -> None:
    assert drift_trace_analysis.PERCENTILES == PERCENTILES

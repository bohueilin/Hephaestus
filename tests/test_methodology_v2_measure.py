from __future__ import annotations

import pytest

from hephaestus.evidence_contract import v2_run_settings_json
from hephaestus.measure import RunSettings, measure
from tests.test_measure import (
    EAGER_REQUEST,
    _DeterministicClock,
    _workload,
)


def test_bare_run_settings_remain_schema_v1() -> None:
    """Adding v2 must not silently move callers off the absence-means-v1 contract."""
    assert RunSettings().schema_version == 1


def test_schema_v2_settings_pin_the_frozen_run_counts() -> None:
    """Trusted v2 callers must select the declared 64-repeat methodology explicitly."""
    assert RunSettings(schema_version=2, repeats=64) == RunSettings(
        schema_version=2,
        warmup_runs=5,
        repeats=64,
        bootstrap_samples=2000,
        inter_run_spacing_seconds=0.0,
    )
    assert v2_run_settings_json() == {
        "schema_version": 2,
        "warmup_runs": 5,
        "repeats": 64,
        "bootstrap_samples": 2000,
        "inter_run_spacing_seconds": 0.0,
    }


@pytest.mark.parametrize("version", (True, False, 0, 3, 1.0, 2.0, "2"))
def test_run_settings_accept_only_exact_integer_schema_versions(version: object) -> None:
    """Boolean, numeric-alias, string, and unknown versions cannot select a producer."""
    with pytest.raises(ValueError, match="schema_version"):
        RunSettings(schema_version=version)  # type: ignore[arg-type]


def test_v2_run_settings_reject_odd_repeats() -> None:
    """Cross-parity pairing cannot silently floor an unmatched final iteration."""
    with pytest.raises(ValueError, match="even"):
        RunSettings(schema_version=2, repeats=31)


def test_v2_measurement_emits_even_odd_slots_and_cross_parity_effects() -> None:
    """Moving eager or either role to a wrong slot must change the literal raw series."""
    durations = [10.0, 0.5]
    durations.extend(
        (
            1.0,
            10.0,
            20.0,
            2.0,
            30.0,
            40.0,
            3.0,
            50.0,
            60.0,
            4.0,
            70.0,
            80.0,
        )
    )
    evidence = measure(
        _workload(),
        EAGER_REQUEST,
        RunSettings(schema_version=2, warmup_runs=0, repeats=4),
        _clock=_DeterministicClock(durations),
    )

    assert evidence.timings["eager_seconds"] == (1.0, 2.0, 3.0, 4.0)
    assert evidence.timings["aa_baseline_seconds"] == (10.0, 40.0, 50.0, 80.0)
    assert evidence.timings["aa_candidate_seconds"] == (20.0, 30.0, 60.0, 70.0)
    assert evidence.timings["aa_signed_paired_effects"] == (
        -1.0,
        2.0 / 3.0,
        -1.0 / 3.0,
        2.0 / 7.0,
    )
    assert evidence.timings["aa_baseline_timestamps_utc"] == (
        "2026-08-29T12:00:03Z",
        "2026-08-29T12:00:07Z",
        "2026-08-29T12:00:09Z",
        "2026-08-29T12:00:13Z",
    )
    assert evidence.timings["aa_candidate_timestamps_utc"] == (
        "2026-08-29T12:00:04Z",
        "2026-08-29T12:00:06Z",
        "2026-08-29T12:00:10Z",
        "2026-08-29T12:00:12Z",
    )


def test_v2_measurement_emits_versioned_methodology_and_baseline_aliases() -> None:
    """Version selection must bind declarations and both compiled/A aliases together."""
    durations = [10.0, 0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    evidence = measure(
        _workload(),
        EAGER_REQUEST,
        RunSettings(schema_version=2, warmup_runs=0, repeats=2),
        _clock=_DeterministicClock(durations),
    )

    assert evidence.methodology["schema_version"] == 2
    assert evidence.methodology["repeats"] == 2
    assert evidence.methodology["bootstrap_samples"] == 2000
    assert evidence.methodology["bootstrap_seed"] == 0
    assert evidence.methodology["bootstrap_confidence"] == 0.95
    assert evidence.methodology["quantile_method"] == "linear_interpolation"
    assert evidence.methodology["aa_pairing"] == "position_matched_cross_parity"
    assert evidence.methodology["measurement_schedule"] == ("alternate_eager-A-B__eager-B-A")
    assert evidence.timings["compiled_seconds"] is evidence.timings["aa_baseline_seconds"]
    assert (
        evidence.timings["compiled_timestamps_utc"]
        is evidence.timings["aa_baseline_timestamps_utc"]
    )

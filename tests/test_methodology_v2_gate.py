from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

import pytest

from hephaestus.bundle import write_json
from hephaestus.gate import evaluate_bundle
from tests.methodology_v2_helpers import (
    finalize_normal_bundle,
    write_v2_normal_bundle,
)
from tests.test_gate import _write_bundle


def _rewrite(bundle: Path, filename: str, mutate: Callable[[dict[str, object]], None]) -> None:
    path = bundle / filename
    payload = json.loads(path.read_bytes())
    mutate(payload)
    write_json(path, payload)
    finalize_normal_bundle(bundle)


def test_gate_accepts_complete_schema_v2_bundle_and_cross_parity_pairing(
    tmp_path: Path,
) -> None:
    """A gate that dispatches v2 through v1 or pairs within iterations rejects this bundle."""
    bundle = tmp_path / "bundle"
    write_v2_normal_bundle(bundle)
    methodology = json.loads((bundle / "methodology.json").read_bytes())
    timings = json.loads((bundle / "timings.json").read_bytes())

    assert methodology["repeats"] == 64
    assert len(timings["compiled_seconds"]) == 64
    assert len(timings["aa_baseline_seconds"]) == 64
    assert len(timings["aa_candidate_seconds"]) == 64
    assert len(timings["aa_signed_paired_effects"]) == 64

    verdict = evaluate_bundle(bundle)

    assert verdict["verdict"] == "PROVEN"
    assert verdict["driving_finding"] == "all_criteria_passed"


def test_gate_rejects_legacy_accuracy_shape_under_schema_v2(tmp_path: Path) -> None:
    """The historical metadata omission is admitted only for absent-version schema v1."""
    bundle = tmp_path / "bundle"
    write_v2_normal_bundle(bundle)

    def remove_modern_metadata(value: dict[str, object]) -> None:
        del value["schema_version"]
        del value["dtype"]

    _rewrite(bundle, "accuracy.json", remove_modern_metadata)

    verdict = evaluate_bundle(bundle)

    assert verdict["verdict"] == "INVALID_EVIDENCE"
    assert verdict["driving_finding"] == "accuracy.contract"


@pytest.mark.parametrize(
    ("field", "wrong_value"),
    (
        ("measurement_schedule", "alternate_eager-A-B__B-A-eager"),
        ("aa_pairing", "within_iteration"),
    ),
)
def test_gate_rejects_wrong_schema_v2_methodology_literals(
    tmp_path: Path,
    field: str,
    wrong_value: str,
) -> None:
    """A v2 version tag cannot bypass the frozen schedule or pairing declaration."""
    bundle = tmp_path / "bundle"
    write_v2_normal_bundle(bundle)
    _rewrite(
        bundle,
        "methodology.json",
        lambda value: value.update({field: wrong_value}),
    )

    verdict = evaluate_bundle(bundle)

    assert verdict["verdict"] == "INVALID_EVIDENCE"
    assert verdict["driving_finding"] == "methodology.valid"


@pytest.mark.parametrize("version", (True, False, 1, 3, 2.0, "2"))
def test_gate_fails_closed_on_present_non_v2_methodology_version(
    tmp_path: Path,
    version: object,
) -> None:
    """Absence alone spells v1; every present value except exact integer two is invalid."""
    bundle = tmp_path / "bundle"
    _write_bundle(bundle)
    _rewrite(bundle, "methodology.json", lambda value: value.update(schema_version=version))

    verdict = evaluate_bundle(bundle)

    assert verdict["verdict"] == "INVALID_EVIDENCE"
    assert verdict["driving_finding"] == "methodology.valid"


def test_gate_rejects_odd_schema_v2_repeats_before_pair_derivation(tmp_path: Path) -> None:
    """An odd v2 count must fail as repeats, not be truncated or parsed as effects."""
    bundle = tmp_path / "bundle"
    write_v2_normal_bundle(bundle, repeats=31)
    _rewrite(
        bundle,
        "timings.json",
        lambda value: value.update(aa_signed_paired_effects=None),
    )

    verdict = evaluate_bundle(bundle)

    assert verdict["verdict"] == "INVALID_EVIDENCE"
    assert verdict["driving_finding"] == "methodology.repeats"


def test_gate_rejects_otherwise_coherent_schema_v2_32_repeat_bundle(
    tmp_path: Path,
) -> None:
    """Evenness is necessary for cross-parity pairing but does not relax the frozen count."""
    control = tmp_path / "control"
    write_v2_normal_bundle(control)
    assert evaluate_bundle(control)["verdict"] == "PROVEN"
    bundle = tmp_path / "bundle"
    write_v2_normal_bundle(bundle, repeats=32)

    verdict = evaluate_bundle(bundle)

    assert verdict["verdict"] == "INVALID_EVIDENCE"
    assert verdict["driving_finding"] == "methodology.repeats"


def test_gate_requires_64_cross_parity_effects_from_64_repeats(tmp_path: Path) -> None:
    """A v2 gate cannot accept one effect per pair or omit the final slot-3 effect."""
    bundle = tmp_path / "bundle"
    write_v2_normal_bundle(bundle)
    _rewrite(
        bundle,
        "timings.json",
        lambda value: value["aa_signed_paired_effects"].pop(),  # type: ignore[union-attr]
    )

    verdict = evaluate_bundle(bundle)

    assert verdict["verdict"] == "INVALID_EVIDENCE"
    assert verdict["driving_finding"] == "methodology.aa_effects"


def _swap_slot_effect_order(value: dict[str, object]) -> None:
    effects = value["aa_signed_paired_effects"]
    assert isinstance(effects, list)
    swapped = [item for pair in zip(effects[1::2], effects[0::2], strict=True) for item in pair]
    assert swapped != effects
    value["aa_signed_paired_effects"] = swapped


def _reverse_effect_role_direction(value: dict[str, object]) -> None:
    effects = value["aa_signed_paired_effects"]
    assert isinstance(effects, list)
    reversed_roles = [-effect for effect in effects]
    assert reversed_roles != effects
    value["aa_signed_paired_effects"] = reversed_roles


@pytest.mark.parametrize(
    "mutation",
    (_swap_slot_effect_order, _reverse_effect_role_direction),
)
def test_gate_rejects_wrong_cross_parity_effect_order_or_role_direction(
    tmp_path: Path,
    mutation: Callable[[dict[str, object]], None],
) -> None:
    """Stored effects must retain slot-2/slot-3 order and baseline-first direction."""
    bundle = tmp_path / "bundle"
    write_v2_normal_bundle(bundle)
    _rewrite(bundle, "timings.json", mutation)

    verdict = evaluate_bundle(bundle)

    assert verdict["verdict"] == "INVALID_EVIDENCE"
    assert verdict["driving_finding"] == "methodology.aa_effects"


def _collapse_even_slot(value: dict[str, object]) -> None:
    value["aa_candidate_timestamps_utc"][0] = value[  # type: ignore[index]
        "aa_baseline_timestamps_utc"
    ][0]


def _reverse_odd_slots(value: dict[str, object]) -> None:
    value["aa_candidate_timestamps_utc"][1] = value[  # type: ignore[index]
        "aa_baseline_timestamps_utc"
    ][1]


def _break_cross_iteration_chain(value: dict[str, object]) -> None:
    value["eager_timestamps_utc"][1] = value[  # type: ignore[index]
        "aa_candidate_timestamps_utc"
    ][0]


def _delete_timestamp(value: dict[str, object]) -> None:
    value["aa_candidate_timestamps_utc"].pop()  # type: ignore[union-attr]


@pytest.mark.parametrize(
    "mutation",
    (
        _collapse_even_slot,
        _reverse_odd_slots,
        _break_cross_iteration_chain,
        _delete_timestamp,
    ),
)
def test_gate_binds_schema_v2_timestamp_order_and_lengths(
    tmp_path: Path,
    mutation: Callable[[dict[str, object]], None],
) -> None:
    """Equal, reversed, cross-iteration, or missing v2 timestamps must fail closed."""
    bundle = tmp_path / "bundle"
    write_v2_normal_bundle(bundle)
    _rewrite(bundle, "timings.json", mutation)

    verdict = evaluate_bundle(bundle)

    assert verdict["verdict"] == "INVALID_EVIDENCE"
    assert verdict["driving_finding"] == "methodology.v2.timestamps"


@pytest.mark.parametrize(
    ("field", "source", "expected_finding"),
    (
        ("compiled_seconds", "aa_candidate_seconds", "methodology.aa_baseline"),
        (
            "compiled_timestamps_utc",
            "aa_candidate_timestamps_utc",
            "methodology.v2.timestamps",
        ),
    ),
)
def test_gate_binds_schema_v2_compiled_aliases_to_baseline_role(
    tmp_path: Path,
    field: str,
    source: str,
    expected_finding: str,
) -> None:
    """The speedup series cannot be replaced by candidate-role seconds or timestamps."""
    bundle = tmp_path / "bundle"
    write_v2_normal_bundle(bundle)
    _rewrite(bundle, "timings.json", lambda value: value.update({field: value[source]}))

    verdict = evaluate_bundle(bundle)

    assert verdict["verdict"] == "INVALID_EVIDENCE"
    assert verdict["driving_finding"] == expected_finding

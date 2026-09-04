from __future__ import annotations

import json
import math
import random
from datetime import UTC, datetime, timedelta
from pathlib import Path

from hephaestus.bundle import write_json, write_manifest
from hephaestus.gate import _evaluate_provisional_bundle
from tests.test_gate import _write_bundle

V2_REPEATS = 64


def write_v2_normal_bundle(bundle: Path, *, repeats: int = V2_REPEATS) -> None:
    """Create a complete normal bundle, then replace only its methodology-v2 evidence."""
    _write_bundle(bundle)
    convert_normal_bundle_to_v2(bundle, repeats=repeats)


def convert_normal_bundle_to_v2(bundle: Path, *, repeats: int = V2_REPEATS) -> None:
    """Upgrade a test-only normal bundle to the literal schema-v2 contract."""
    methodology_path = bundle / "methodology.json"
    methodology = json.loads(methodology_path.read_bytes())
    methodology.update(
        {
            "schema_version": 2,
            "repeats": repeats,
            "aa_pairing": "position_matched_cross_parity",
            "measurement_schedule": "alternate_eager-A-B__eager-B-A",
        }
    )

    eager_seconds = [1.0] * repeats
    baseline_seconds: list[float] = []
    candidate_seconds: list[float] = []
    for pair in range((repeats + 1) // 2):
        base = 0.8 + pair / 1000
        baseline_seconds.append(base + 0.001)
        candidate_seconds.append(base + 0.006)
        if len(baseline_seconds) < repeats:
            baseline_seconds.append(base + 0.004)
            candidate_seconds.append(base)

    effects = _cross_parity_effects(baseline_seconds, candidate_seconds)
    bootstrap = _bootstrap_absolute_medians(effects)

    eager_timestamps, baseline_timestamps, candidate_timestamps = _v2_timestamps(repeats)
    timings_path = bundle / "timings.json"
    timings = json.loads(timings_path.read_bytes())
    timings.update(
        {
            "eager_seconds": eager_seconds,
            "eager_timestamps_utc": eager_timestamps,
            "compiled_seconds": baseline_seconds,
            "compiled_timestamps_utc": baseline_timestamps,
            "aa_baseline_seconds": baseline_seconds,
            "aa_baseline_timestamps_utc": baseline_timestamps,
            "aa_candidate_seconds": candidate_seconds,
            "aa_candidate_timestamps_utc": candidate_timestamps,
            "aa_signed_paired_effects": effects,
            "aa_bootstrap_absolute_medians": bootstrap,
        }
    )
    methodology["aa_noise_floor"] = _quantile(bootstrap, 0.95)

    write_json(methodology_path, methodology)
    write_json(timings_path, timings)
    finalize_normal_bundle(bundle)


def finalize_normal_bundle(bundle: Path) -> dict[str, object]:
    """Re-seal a test bundle with the gate result for its current stored evidence."""
    verdict_path = bundle / "verdict.json"
    verdict_path.unlink(missing_ok=True)
    write_manifest(bundle)
    verdict = _evaluate_provisional_bundle(bundle)
    write_json(verdict_path, verdict)
    write_manifest(bundle)
    return verdict


def _v2_timestamps(repeats: int) -> tuple[list[str], list[str], list[str]]:
    origin = datetime(2026, 8, 29, 12, 0, tzinfo=UTC)

    def timestamp(offset: int) -> str:
        return (origin + timedelta(seconds=offset)).isoformat().replace("+00:00", "Z")

    eager: list[str] = []
    baseline: list[str] = []
    candidate: list[str] = []
    for iteration in range(repeats):
        offset = 2 + iteration * 3
        eager.append(timestamp(offset))
        if iteration % 2 == 0:
            baseline.append(timestamp(offset + 1))
            candidate.append(timestamp(offset + 2))
        else:
            candidate.append(timestamp(offset + 1))
            baseline.append(timestamp(offset + 2))
    return eager, baseline, candidate


def _cross_parity_effects(baseline: list[float], candidate: list[float]) -> list[float]:
    effects: list[float] = []
    for even in range(0, len(baseline) - 1, 2):
        odd = even + 1
        effects.append(_signed_effect(baseline[even], candidate[odd]))
        effects.append(_signed_effect(baseline[odd], candidate[even]))
    return effects


def _signed_effect(baseline: float, candidate: float) -> float:
    return (baseline - candidate) / ((baseline + candidate) / 2.0)


def _bootstrap_absolute_medians(effects: list[float]) -> list[float]:
    random_source = random.Random(0)
    return [
        abs(
            _quantile(
                [effects[random_source.randrange(len(effects))] for _ in effects],
                0.5,
            )
        )
        for _ in range(2000)
    ]


def _quantile(values: list[float], percentile: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * percentile
    lower_index = math.floor(position)
    upper_index = math.ceil(position)
    if lower_index == upper_index:
        return ordered[lower_index]
    fraction = position - lower_index
    return ordered[lower_index] + fraction * (ordered[upper_index] - ordered[lower_index])


__all__ = [
    "V2_REPEATS",
    "convert_normal_bundle_to_v2",
    "finalize_normal_bundle",
    "write_v2_normal_bundle",
]

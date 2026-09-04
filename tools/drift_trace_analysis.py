"""Apply the ratified single-trace drift analysis."""

from __future__ import annotations

import argparse
import json
import math
import random
from collections.abc import Callable, Sequence
from datetime import UTC, datetime
from pathlib import Path

# This private quantile import is authorized for this analysis-grade script.
from hephaestus.aa import _quantile as aa_quantile
from hephaestus.aa import compute_null_statistics

PERCENTILES = (0.5, 0.857, 0.95, 0.99, 0.99488)
_PERCENTILE_NAMES = ("50", "85.7", "95", "99", "99.488")


def _timestamp(block: dict[str, object]) -> datetime:
    timestamps = block.get("baseline_timestamps_utc")
    if not isinstance(timestamps, list) or not timestamps:
        raise ValueError("block has no baseline timestamp")
    parsed = datetime.fromisoformat(str(timestamps[0]).replace("Z", "+00:00"))
    if parsed.utcoffset() != UTC.utcoffset(parsed):
        raise ValueError("block baseline timestamp must be UTC")
    return parsed.astimezone(UTC)


def sort_blocks(blocks: Sequence[dict[str, object]]) -> list[dict[str, object]]:
    """Sort blocks by the first baseline timestamp."""
    return sorted(blocks, key=_timestamp)


def pair_blocks(
    blocks: Sequence[dict[str, object]],
) -> tuple[list[dict[str, int | float]], int]:
    """Greedily pair each block with the smallest eligible later index."""
    ordered = sort_blocks(blocks)
    times = [_timestamp(block) for block in ordered]
    paired: set[int] = set()
    pairs: list[dict[str, int | float]] = []
    for i in range(len(ordered)):
        if i in paired:
            continue
        for j in range(i + 1, len(ordered)):
            if j in paired:
                continue
            lag = (times[j] - times[i]).total_seconds()
            if lag > 2.9:
                break
            if 2.3 <= lag <= 2.9:
                pairs.append({"i": i, "j": j, "lag_seconds": lag})
                paired.update((i, j))
                break
    return pairs, len(ordered) - len(paired)


def count_mutually_non_overlapping_pairs(
    pairs: Sequence[dict[str, int | float]],
    blocks: Sequence[dict[str, object]],
) -> int:
    """Count a maximum set of pair spans that do not intersect."""
    ordered = sort_blocks(blocks)
    spans = sorted(
        (
            (_timestamp(ordered[int(pair["i"])]), _timestamp(ordered[int(pair["j"])]))
            for pair in pairs
        ),
        key=lambda span: span[1],
    )
    count = 0
    previous_end: datetime | None = None
    for start, end in spans:
        if previous_end is None or start > previous_end:
            count += 1
            previous_end = end
    return count


def moving_block_bootstrap_interval(
    values: Sequence[float],
    percentile: float,
    *,
    quantile: Callable[[list[float], float], float],
    block_length: int = 64,
    resamples: int = 2000,
    seed: int = 0,
) -> list[float] | None:
    """Return a deterministic 95% moving-block-bootstrap percentile interval."""
    if not values:
        return None
    if block_length <= 0 or resamples <= 0:
        raise ValueError("bootstrap dimensions must be positive")
    source = [float(value) for value in values]
    effective_length = min(block_length, len(source))
    max_start = len(source) - effective_length
    random_source = random.Random(seed)
    estimates: list[float] = []
    for _ in range(resamples):
        sample: list[float] = []
        while len(sample) < len(source):
            start = random_source.randrange(max_start + 1)
            sample.extend(source[start : start + effective_length])
        estimates.append(quantile(sample[: len(source)], percentile))
    return [quantile(estimates, 0.025), quantile(estimates, 0.975)]


def autocorrelations(
    values: Sequence[float], *, max_lag: int = 64
) -> list[dict[str, int | float | None]]:
    """Publish population-normalized autocorrelation through the fixed lag."""
    source = [float(value) for value in values]
    if not source:
        return [{"lag": lag, "value": None} for lag in range(1, max_lag + 1)]
    mean = sum(source) / len(source)
    denominator = sum((value - mean) ** 2 for value in source)
    result: list[dict[str, int | float | None]] = []
    for lag in range(1, max_lag + 1):
        if lag >= len(source) or denominator == 0.0:
            value = None
        else:
            numerator = sum(
                (source[index] - mean) * (source[index + lag] - mean)
                for index in range(len(source) - lag)
            )
            value = numerator / denominator
        result.append({"lag": lag, "value": value})
    return result


def implied_effective_sample_size(values: Sequence[float], *, max_lag: int = 64) -> float:
    """Derive the autocorrelation-adjusted effective pair count."""
    count = len(values)
    if count == 0:
        return 0.0
    correlations = autocorrelations(values, max_lag=max_lag)
    correlation_sum = sum(
        float(record["value"]) for record in correlations if record["value"] is not None
    )
    denominator = 1.0 + 2.0 * correlation_sum
    if denominator <= 0.0:
        return float(count)
    return min(float(count), max(1.0, count / denominator))


def _threshold_summary(values: Sequence[float], threshold: float) -> dict[str, int | float | None]:
    count = sum(value >= threshold for value in values)
    return {
        "count": count,
        "fraction": count / len(values) if values else None,
    }


def analyze_series(
    blocks: Sequence[dict[str, object]],
    pairs: Sequence[dict[str, int | float]],
    *,
    series_name: str,
    compute_statistics: Callable[..., object],
    quantile: Callable[[list[float], float], float],
    bootstrap_resamples: int = 2000,
) -> dict[str, object]:
    """Apply the fixed A/A statistic and tail analysis to one stored series."""
    pair_rows: list[dict[str, object]] = []
    floors_in_time_order: list[float] = []
    for pair in pairs:
        i = int(pair["i"])
        j = int(pair["j"])
        left = blocks[i][series_name]
        right = blocks[j][series_name]
        statistics = compute_statistics(left, right, schema_version=2)
        floor = float(statistics.p95_noise_floor)
        lower_a_over_b = statistics.speedup_lower_bound_a_over_b
        lower_b_over_a = statistics.speedup_lower_bound_b_over_a
        arm_fires = (lower_a_over_b is not None and float(lower_a_over_b) >= 1.10) or (
            lower_b_over_a is not None and float(lower_b_over_a) >= 1.10
        )
        pair_rows.append(
            {
                "i": i,
                "j": j,
                "lag_seconds": pair["lag_seconds"],
                "p95_noise_floor": floor,
                "speedup_lower_bound_a_over_b": lower_a_over_b,
                "speedup_lower_bound_b_over_a": lower_b_over_a,
                "false_positive_arm_fires": arm_fires,
            }
        )
        floors_in_time_order.append(floor)

    percentile_records: dict[str, object] = {}
    for name, percentile in zip(_PERCENTILE_NAMES, PERCENTILES, strict=True):
        value = quantile(floors_in_time_order, percentile) if floors_in_time_order else None
        interval = moving_block_bootstrap_interval(
            floors_in_time_order,
            percentile,
            quantile=quantile,
            block_length=64,
            resamples=bootstrap_resamples,
            seed=0,
        )
        percentile_records[name] = {
            "value": value,
            "bootstrap_95_interval": interval,
        }

    arm_count = sum(bool(row["false_positive_arm_fires"]) for row in pair_rows)
    return {
        "pairs": pair_rows,
        "sorted_pair_floors": sorted(floors_in_time_order),
        "percentiles": percentile_records,
        "at_or_above_0.10": _threshold_summary(floors_in_time_order, 0.10),
        "at_or_above_0.07": _threshold_summary(floors_in_time_order, 0.07),
        "false_positive_arm": {
            "count": arm_count,
            "fraction": arm_count / len(pair_rows) if pair_rows else None,
        },
        "autocorrelations": autocorrelations(floors_in_time_order, max_lag=64),
        "implied_effective_sample_size": implied_effective_sample_size(
            floors_in_time_order, max_lag=64
        ),
    }


def classify_host(
    *,
    baseline_load: object,
    torch_num_threads: object,
    load_series: object,
) -> str:
    """Classify the host using only the preregistered recorded load numbers."""
    if not _finite_number(baseline_load) or not _finite_number(torch_num_threads):
        return "disturbed"
    if not isinstance(load_series, list) or not load_series:
        return "disturbed"
    if any(
        type(record) is not dict or not _finite_number(record.get("t")) for record in load_series
    ):
        return "disturbed"
    ordered = sorted(load_series, key=lambda record: record["t"])
    first = ordered[0]
    if first.get("t") != 0.0:
        return "disturbed"
    initial_load = _one_minute_load(first)
    at_120 = next((record for record in ordered if _at_or_after_120(record)), None)
    load_120 = _one_minute_load(at_120) if at_120 is not None else None
    if initial_load is None or load_120 is None:
        return "disturbed"
    baseline = float(baseline_load)
    threads = float(torch_num_threads)
    if initial_load > baseline + 1.0 or load_120 > baseline + 1.0 + threads:
        return "disturbed"
    for record in ordered:
        if _at_or_after_120(record):
            value = _one_minute_load(record)
            if value is None or value > load_120 + 1.0:
                return "disturbed"
    return "quiet"


def _at_or_after_120(record: object) -> bool:
    return isinstance(record, dict) and _finite_number(record.get("t")) and record["t"] >= 120.0


def _one_minute_load(record: object) -> float | None:
    if not isinstance(record, dict):
        return None
    values = record.get("load_average")
    if not isinstance(values, list) or not values or not _finite_number(values[0]):
        return None
    return float(values[0])


def _finite_number(value: object) -> bool:
    return type(value) in {int, float} and math.isfinite(float(value))


def analyze_trace(path: Path) -> dict[str, object]:
    """Read and analyze exactly one drift-trace JSONL file."""
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    headers = [row for row in rows if row.get("row") == "header"]
    trailers = [row for row in rows if row.get("row") == "trailer"]
    if len(headers) != 1 or len(trailers) != 1:
        raise ValueError("trace must contain exactly one header and one trailer")
    ordered_blocks = sort_blocks([row for row in rows if row.get("row") == "block"])
    pairs, unpaired_count = pair_blocks(ordered_blocks)
    header = headers[0]
    trailer = trailers[0]
    classification = classify_host(
        baseline_load=header.get("baseline_load"),
        torch_num_threads=header.get("torch_num_threads"),
        load_series=trailer.get("load_series"),
    )
    return {
        "trace_path": str(path),
        "pair_count": len(pairs),
        "unpaired_count": unpaired_count,
        "mutually_non_overlapping_pair_count": count_mutually_non_overlapping_pairs(
            pairs, ordered_blocks
        ),
        "classification": classification,
        "compiled": analyze_series(
            ordered_blocks,
            pairs,
            series_name="baseline_seconds",
            compute_statistics=compute_null_statistics,
            quantile=aa_quantile,
        ),
        "eager": analyze_series(
            ordered_blocks,
            pairs,
            series_name="eager_seconds",
            compute_statistics=compute_null_statistics,
            quantile=aa_quantile,
        ),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace_path", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    output = args.trace_path.with_suffix(".analysis.json")
    result = analyze_trace(args.trace_path)
    with open(output, "w", encoding="utf-8") as stream:
        json.dump(result, stream, allow_nan=False, indent=2, sort_keys=True)
        stream.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

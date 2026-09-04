"""Torch-free fixed-pairing tests for drift-trace analysis."""

from __future__ import annotations

from tools import drift_trace_analysis


def _block(seconds: float) -> dict[str, object]:
    return {
        "row": "block",
        "baseline_timestamps_utc": [f"2026-09-01T12:00:{seconds:09.6f}Z"],
        "baseline_seconds": [0.01] * 64,
        "eager_seconds": [0.10] * 64,
        "candidate_seconds": [0.02] * 64,
    }


def test_pairing_includes_window_edges_and_uses_smallest_eligible_index() -> None:
    pairs, unpaired = drift_trace_analysis.pair_blocks(
        [_block(5.2), _block(2.9), _block(0.0), _block(2.3), _block(8.1)]
    )

    assert pairs == [
        {"i": 0, "j": 1, "lag_seconds": 2.3},
        {"i": 2, "j": 3, "lag_seconds": 2.3},
    ]
    assert unpaired == 1
    assert (
        drift_trace_analysis.count_mutually_non_overlapping_pairs(
            pairs,
            [_block(0.0), _block(2.3), _block(2.9), _block(5.2), _block(8.1)],
        )
        == 2
    )

    edge_pairs, edge_unpaired = drift_trace_analysis.pair_blocks([_block(0.0), _block(2.9)])
    assert edge_pairs == [{"i": 0, "j": 1, "lag_seconds": 2.9}]
    assert edge_unpaired == 0


def test_pairing_leaves_tail_and_counts_maximum_nonintersecting_spans() -> None:
    blocks = [_block(0.0), _block(0.1), _block(2.3), _block(2.4), _block(7.0)]
    pairs, unpaired = drift_trace_analysis.pair_blocks(blocks)

    assert pairs == [
        {"i": 0, "j": 2, "lag_seconds": 2.3},
        {"i": 1, "j": 3, "lag_seconds": 2.3},
    ]
    assert unpaired == 1
    assert drift_trace_analysis.count_mutually_non_overlapping_pairs(pairs, blocks) == 1

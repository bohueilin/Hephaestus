# Pre-registration 1 — Amendment 1 (ratified by owner ruling 6, 2026-09-02 UTC)

Pre-registration 1 (`docs/calibration/preregistration-1-cross-run-calibration.md`, commit
`173c48b`) forbids editing itself; this is a new document. It **changes no threshold and no
outcome**. It ratifies, after the fact and explicitly, the choices the implementer had to
make where the pre-registration was silent, so that every published confidence interval
rests on ratified procedure; corrects three statements the pre-registration made that the
traces show to be inexact; and fixes definitions and conditions before the §3 series runs.
Every item applies **prospectively and to the already-published analyses as-is**; nothing
is recomputed and Commits A–C are not amended.

## A1. Outcome of §6 on this host (binding, from the pre-registered rule)

On the sole quiet trace (`graph_break_bait-20260902T011653.483792Z`), the compiled
99.488th-percentile 95% interval is [0.13017056924464904, 0.2938272185033995].
§6.3 (a) FAILS (the upper end does not classify 0.16022381230908075 as an exceedance);
§6.3 (b) FAILS (the upper end is not strictly below 0.1377); §6.3 (c) passes. §6.1's pool
(every quiet trace from at least two calendar days) does not exist, so §6.2 cannot yet be
evaluated. **By §6.5, 0.10 remains the cross-run bar in both directions.** Further: any
pooled interval must contain 0.2616 to satisfy §6.2, so its upper end is ≥ 0.2616 > 0.1377
and fails §6.3 (b); if it does not contain 0.2616, §6.2 fails; and §6.1 forbids removing
this trace from the pool. **No second-day trace can yield an adoptable limit on this
host.** The "separate authorization" for a second day in §6.1 is closed for adoption.
(Owner error 16: this consequence was foreseeable and should have been stated in the
pre-registration.)

## A2. Bootstrap procedure, ratified as implemented (owner error 9)

The moving-block bootstrap of §4.3 is: `ceil(n/64)` blocks of 64 consecutive pairs in time
order, non-wrapping, block starts uniform on `[0, n−64]`, concatenated and truncated to
`n`; `random.Random(0)` re-seeded per percentile so all five percentiles share block draws;
one `randrange` per block; 2000 resamples; percentile interval at 0.025 / 0.975 via
`aa._quantile`. Sensitivity recorded: the graph compiled 99.488th lower end is 0.1302 under
this procedure, 0.1259 circular, 0.1276 with `floor(n/64)` blocks; the upper end is 0.2938
under all three; no §6 outcome depends on the choice.

## A3. Effective sample size and the operative sample size (owner error 10)

`ESS = n / (1 + 2·Σ_{k=1..64} r_k)` with the biased (population-normalised) sample
autocorrelation, the quotient clamped to `[1, n]`, and `ESS = n` whenever
`1 + 2·Σ r_k ≤ 0` (the implemented rule; that branch was not reached on either trace), is
ratified as the published "implied effective sample size". **It is not the operative sample
size for the tail.** The mutually non-overlapping pair count (201 graph, 132 mlp) bounds the
effective number of independent windows for the 99th and 99.488th percentiles and is the
figure any reader should use for those; the ESS (632, 1117) is reported beside it, never
instead of it.

## A4. "Mutually non-overlapping pairs", defined (owner error 11)

The size of a maximum subset of pairs whose spans do not intersect, where a pair's span
runs from block i's first `baseline_timestamps_utc` to block j's last
`baseline_timestamps_utc`, spans sorted by their end and taken greedily whenever the start
exceeds the last taken end (interval scheduling, which attains the maximum). (The published
201 / 132 used block j's *first* timestamp as the span end. Under this definition the counts
are unchanged — 201 graph, 132 mlp — because a block lasts about 76 ms while consecutive
pairs overlap by about 2.3 s, so lengthening every span by one block changes no greedy
choice. Published values stand; future analyses use this definition and say so.)

## A5. Realized lag and bootstrap-block span (owner error 12)

The §4.3 rule "smallest unpaired j > i with 2.3 s ≤ lag ≤ 2.9 s" always selects the first
block past 2.3 s. The traces therefore measured lags of **2.300–2.422 s** (mean 2.335 s
graph, 2.325 s mlp), not the 2.6 s the motivation names. Because the rule marks both i and
j paired, pairs come in runs of thirty consecutive pairs separated by thirty blocks that are
only ever the j of a pair; 64 consecutive pairs — one bootstrap block — therefore span about
12 s of wall clock (11.8–14.4 s on the graph trace, 12.2–14.7 s on mlp, first pair's start to
last pair's end), several times the lag; §4.3's "≈ 10 s" referred to the start-to-start
distance (9.4–12.3 s). The rule was applied faithfully; the design was the owner's. Every
citation of these traces states 2.30–2.42 s. The §4.1 lower-bound argument is unaffected.

## A6. Calendar day is UTC (owner error 13)

"Calendar day" in §5 (daily cap) and §6.1 means the UTC date in the trace file stem.
Archive directory names follow their own convention and carry no meaning under those
sections.

## A7. "This host" includes runtime configuration

For §3 (2b) — "SERIES_NOT_MET is final for this host" — and for §6, *host* means the
machine **and** the runtime configuration recorded in the evidence: `torch_num_threads` and
`os_cpu_count` as recorded in the trace and baseline-only headers, and the torch build,
Python, OS and chip recorded in the bundles' `env.json` (byte-identical across the archived
probes). A change to any of these does not create a new host that escapes a final result. A
series or calibration taken under a changed configuration is a new measurement requiring
its own pre-registration; it is reported beside this result and never in place of it, it
cannot alter the README's §6.2 status sentence or the 0.10 limit, and **no future pool on
this machine may exclude the trace named in A1.**

## A8. "Interrupted" and "aborted baseline"

The fix commit `a72fb17` removed the explicit *interrupted* label. It is derived: a trace
whose trailer — or, for a file that has no trailer, whose header — carries
`stop_reason != "completed"` is **interrupted**, is published, is counted beside disturbed
traces, and never enters a quiet pool. A header-only file that carries
`stop_reason` and no `mode` (A9) is an **aborted baseline**: it counts as a trace start under
the §5 daily cap, is published under the classification *interrupted* with its partial
samples, and is never analysed — `drift_trace_analysis.py` rejects it by construction. A
header-only file with `mode: "baseline_only"` is neither a trace nor a start, with or
without `stop_reason`; if it carries `stop_reason` it is an aborted baseline-only record,
published, never used as the series reference, and the series baseline is retaken by a
fresh `--baseline-only` run. No script change beyond A9.

## A9. Aborted baseline — the last permitted fix round (owner error 14)

§5 required that an abandoned baseline be written as an aborted header; the shipped script
writes nothing on an interrupt before the block loop. Authorized, as the **second and
last** fix round under §8, for `KeyboardInterrupt` only — a compile exception still creates
no file (§9) and the existing compile-failure test is unchanged: on an interrupt during
`capture_baseline` or the compile phase, write the header with whatever
`baseline_load_samples` exist (`baseline_load` is `null` when there are none) and
`stop_reason: "KeyboardInterrupt"` (the trailer's existing literal), then re-raise. **Permitted solution space, exhaustive:**
`capture_baseline` appends into a caller-owned list; one `try`/`except KeyboardInterrupt`
in `produce_trace`; one optional `stop_reason: str | None = None` argument on
`build_header`, written only when given; no new function, module, field or flag beyond
these; at most 25 net lines in `tools/drift_trace.py` plus one torch-free test in
`tests/tools/test_drift_trace_rows.py` asserting: exactly one write, exactly one `header`
row, the partial samples present, `baseline_load` null, and `stop_reason` present. The ledger records that no baseline was abandoned on
2026-09-02 (gaps between runs were 46 s and 27 s, too short to hide a complete 60 s
baseline).

## A10. The §3 series: conditions fixed before it starts (owner error 15)

- **A human operator starts the loop.** The executing agent cannot make "no other
  Codex/agent process" true; the operator can. Every checklist flag is passed to the
  `--baseline-only` run and is therefore written down before the loop; the implementer
  writes all seven flags on that line, and **before running the operator deletes any flag
  that is not true at that moment** — the header records exactly what was passed and the
  ledger copies it; nothing else in the script is edited. No agent, subagent, test suite or
  browser runs during the loop.
- **The host idles for ≥ 300 s *before* the `--baseline-only` run** (the operator's own
  launch activity must decay before the reference is taken; a settle *after* the baseline
  could only lower the attempts' load relative to it and loosen §5). The loop starts
  immediately after the baseline-only run exits, as §3 item 1 requires.
- Output root for the ten attempts: `artifacts/methodology-v2/series-1/` (plain path, via
  `aa-test --output-root`). The `--baseline-only` header is written where the unchanged
  script writes it — `artifacts/methodology-v2/drift-trace/graph_break_bait-<UTC>.jsonl` —
  and the script is given no output flag; the ledger cites that stem and a copy of that one
  file is placed in `series-1/` before the tree is archived.
- Ten attempts, one shell loop, `sleep 30` between, no human action between first and
  last. **The loop ignores every exit status**: `aa-test` exits 1 on every non-PASS attempt,
  and a loop that stops on the first FAIL voids the only authorized series.
- The ledger entry records the baseline-only header's `torch_num_threads` and
  `os_cpu_count` (A7).
- **Attempt classification is computed and published exactly as §5 prescribes, and its
  expected behaviour is stated now:** the graph trace's loop raised the 1-minute load from
  2.72 at t = 0 to 8.06 at t = 120 s against a 2.525 baseline (+5.5 with ten threads), so a
  10 s attempt every 40 s is expected to leave a residual of roughly +0.8 to +1.7 at run B's
  `after` sample and at later attempts' `before` sample. A *disturbed* label consistent with
  that residual is the series' own load; it is published as such beside the raw
  `before`/`after` values in the ten-row table and changes nothing in (2a)/(2b), which do
  not depend on classification.
- **The four README sentences are fixed now**, exactly one of which is copied verbatim
  after the series and none of which may be edited:
  - PASS: *"DESIGN.md §6.2 is met under pre-registration 1 §3: 10 of 10 attempts below 0.10
    and 0 false-positive fires (all ten floors, both bounds and the host-state
    classification per attempt listed below). Ten attempts with no exceedance bound the
    per-attempt exceedance rate only to below about 26% (95% upper bound); this licenses no
    cross-run speedup claim and no noise-floor claim beyond these ten attempts. The
    per-window base rate measured on this host is in Calibration."*
  - FAIL: *"DESIGN.md §6.2 is not met on this host: k of 10 attempts at or above 0.10 and m
    false-positive fires (all ten floors, both bounds and the host-state classification per
    attempt listed below). Final for this host under pre-registration 1 §3 (2b). This
    invalidates no published bundle and moves no limit."*
  - NOT EVALUABLE: *"DESIGN.md §6.2 is not demonstrated on this host: the series is not
    evaluable on 10 under pre-registration 1 §3 item 6 — n of 10 attempts carry no
    statistics; of the evaluable attempts, k at or above 0.10 and m false-positive fires
    (all floors, both bounds and the host-state classification per attempt listed below).
    This is not SERIES_MET; a further series requires a new pre-registration citing this
    outcome. This invalidates no published bundle and moves no limit."*
  - INTERRUPTED: *"DESIGN.md §6.2 is not demonstrated on this host: the pre-registration 1 §3
    series stopped after n of 10 attempts; of these, k at or above 0.10 and m false-positive
    fires (every floor and both bounds that exist listed below). Under §3 item 7 this is the authorized series, published as it stands; a further
    series requires a new pre-registration citing this outcome. This invalidates no
    published bundle and moves no limit."*
- **Substitution and precedence.** The italic letters k, m, n, r are the only
  substitutions permitted. Precedence: INTERRUPTED if fewer than ten attempts exist;
  otherwise FAIL if any evaluable attempt is at or above 0.10 or fires ((2b) fails the
  series regardless of item 6, and the non-evaluable count is stated in the table);
  otherwise NOT EVALUABLE if any attempt carries no statistics; otherwise PASS.
- The ten-row table has exactly these columns: attempt, parent directory (repo-relative),
  start UTC, §5 classification (quiet/disturbed) with the raw `before`/`after` loads, p95
  floor, lower bound A/B, lower bound B/A, arm fired (yes/no), verdict / driving finding.
  One line beneath it: "At or above 0.07 (reference line only): r of 10." A quiet-only
  exceedance count, if given, appears only beside the all-attempts count.

## A11. Probe 2, reframed in prose only (owner error 17)

Prior rulings described probe 2 as "a host disturbance", as if exogenous and rare. The
calibration shows that a host classified quiet by the load rule produces compiled-window
pair floors at or above 0.10 in 1.5% of pairs (59 of 3,875) and 21 arm fires on the compiled
series, against 0 and 0 on the eager series timed in the same blocks (eager 99.488th
percentile 0.041) — a compiled/eager asymmetry with no load-average signature — and probe
2's 0.1602 lies inside the published compiled tail (99.488th percentile 0.2616). Whether
probe 2 was one such draw is plausible and unproven. **Its permanent FAIL is unchanged.**
The honest statement is: *cross-run windows on this host straddle a compiled-path slowdown
at least about 1.5% of the time.*

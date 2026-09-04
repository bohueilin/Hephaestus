# Pre-registration 1 — cross-run calibration protocol and the replacement for "ten consecutive"

**Status: RATIFIED by owner ruling 5 (2026-09-01), after three-role adversarial review
(FIX_THEN_RATIFY → all findings applied).** *Ratified* means: the owner commits this file;
that commit hash is cited in every trace header, in the module docstring of
`tools/drift_trace.py`, and in the §3 ledger entry. **Any later edit to this file is a new
document requiring its own ratification and cannot govern a measurement already taken.**
Every number below is fixed now so that no later result can move it.

## 0. Why this document exists

Ruling 4 found that the ten-consecutive-A/A-run requirement cannot be evaluated from
existing evidence (one clean within-attempt observation; a disturbance denominator of
two) and that it demands roughly two orders of magnitude more tail resolution than
exists. It left one textual ambiguity and one measurement deliberately undone until they
could be settled at a moment when no probe result depends on the answer. That moment is
now: Task 5 is blocked, **no probe result exists that the answer could reclassify**, and
the next series is scheduled by this document only after the answer — and a series
verdict that does not depend on it (§3 item 2b) — are both fixed.

## 1. What is frozen and does not move

| Value | Status |
|---|---|
| 0.10 noise limit (derived: minimum_speedup − 1) — in-run AND cross-run | frozen |
| 1.10 minimum speedup | frozen |
| criteria file and its SHA-256 pin | frozen |
| bootstrap: seed 0, 2000 resamples, abs-median, p95, linear interpolation | frozen |
| methodology schema v2, repeats 64, cross-parity pairing | frozen |
| the in-run gate (`perf.speedup_proven` and its noise-floor check) | frozen |
| the cross-run `aa.false_positive` arm (directional bounds vs 1.10) | frozen |
| the cross-run `methodology.noise_floor` comparison against 0.10 | frozen; §6 is the only route to a different value and it changes nothing until all of §6 holds |
| probe 2 = permanent FAIL under the per-attempt rule, which is unchanged | frozen |
| the protected evidence archive (composite `bf32ad1d…`, 63 files) | frozen |

**Nothing in this document derives a new threshold.** §6 fixes, in advance, the only
route by which one could be earned and what happens with the result.

## 2. Resolution of the textual ambiguity (prospective only)

`DESIGN.md` §6.2 requires that `aa-test` "never flags — across 10 consecutive A/A runs —
while the noise floor it measures is published in every bundle." `DESIGN.md` §1 states the
intent: the gate must catch planted regressions "and A/A null runs (same config twice)
that it MUST NOT flag — the noise floor is measured, not assumed."

**Resolution.** "Flag" denotes the false-positive outcome: `aa.false_positive`, the arm
that fires when a directional bound reaches 1.10 on an identical proposal. That is the
outcome §1 forbids, and it is what `CODEX_PROMPT.md` §5 counts: "`aa-test`: 3 consecutive
runs, zero *false flags*, noise floor published." `DESIGN.md` §3.2 separately requires that
"the gate's speedup threshold must clear the measured noise floor — a criterion the A/A
test itself enforces"; that criterion is the `methodology.noise_floor` FAIL, it stays
enforced exactly as implemented (§3 item 4), and nothing here reclassifies it. What this
resolution fixes is only which outcome §6.2's count of "flags" counts.

**Why this is not a goalpost move, and what would make it one.** Ruling 4 refused this
reading because it arrived in direct response to probe 2 failing the *other* arm. It is
adopted here prospectively only: probe 2 remains a permanent FAIL under the per-attempt
rule, which is unchanged — only the scoring of a series changes (§3) — and its floor
0.16022381230908075 remains an exceedance under §3. If any future document retroactively
reclassifies probe 2 as a pass on the strength of this resolution, that document is wrong
and this one says so in advance. **Disclosed, because it matters:** on the frozen estimator
the false-positive arm is evaluated only after the 0.10 comparison and, on every archived
series, cannot fire before the floor has already reached 0.10 (a ≈9.5% between-run level
shift trips the floor; ≈14% is needed for a directional bound to reach 1.10, by which
point the floor is ≥ 0.144). A series criterion resting on that arm alone would therefore
be unfailable. That is why §3 item 2b exists.

## 3. What replaces "ten consecutive passes"

The old requirement was a single-observation trigger fired ten times. The replacement
publishes strictly more and, as a pass/fail requirement, is **at least as strong**:

1. Execute **N = 10** `aa-test` attempts on `graph_break_bait`, **back-to-back, started by
   one command as a single shell loop with a fixed 30-second sleep between attempts and no
   human action between the first and the last.** The loop's start and end UTC are recorded
   in the ledger entry. The §5 baseline is taken once, by `tools/drift_trace.py` in
   baseline-only mode, immediately before the loop. Every attempt is preserved, pass or
   fail. **Deleting, re-running, or omitting any attempt voids the series.**
2. Series verdict, pre-registered:
   - **(2a)** `aa.false_positive` fires on **zero** of the ten. *"Fires" is evaluated from
     the stored statistics, never from the driving finding:* an attempt fires if either
     `speedup_lower_bound_a_over_b` or `speedup_lower_bound_b_over_a` in its own
     `verdict.json` (`statistics`, value-identical to the same keys in
     `aa_distribution.json`) is ≥ 1.10.
     This is evaluated for every attempt that carries statistics, including attempts whose
     verdict is FAIL / `methodology.noise_floor`, because `aa.py` returns the noise-floor
     finding before it evaluates the bounds, so the driving finding cannot answer the
     question. This is the evaluation ruling 4 §4 applied to probe 2 (bounds 0.8087 and
     1.0370). Both bounds are published for every attempt.
   - **(2b)** **Zero** of the ten attempts has a floor **at or above 0.10** (the implemented
     comparison, `Decimal(limit) <= Decimal(str(floor))`). One such attempt does not void
     the record — it is published in full — but it **fails the series**. This is what makes
     the replacement at least as strong as the old rule: under the old rule one exceedance
     voided the series; here it fails it, visibly, with its number. **(2b) is scored at 0.10
     regardless of any §6 outcome; SERIES_NOT_MET is final for this host, and any README must
     then state that `DESIGN.md` §6.2 is not met. A quiet-only exceedance count may appear
     only beside the all-attempts count, never instead of it.**
   - **SERIES_MET** requires (2a) and (2b) both, on ten attempts that all carry statistics.
3. Reported: the full distribution of the ten cross-run p95 floors; the count at or above
   0.10; and — as a reference line only — the count at or above 0.07. Each attempt is
   published with its host-state classification (§5).
4. The per-attempt verdict is exactly as implemented: an attempt whose floor reaches 0.10
   is verdict FAIL / `methodology.noise_floor` in its own `verdict.json`. A FAIL is a FAIL.
5. **The 0.07 early-warning rule is retired as a trigger. This sentence supersedes ruling 4
   §5's "Both 0.07 and 0.10 stay untouched" as to 0.07 only, and ruling 2 §4.2's authorized
   re-derivation of 0.07 is subsumed by §6; the record must show both.** Why retiring it is
   not moving it: (a) it had no derivation and none is offered now — a moved rule is a new
   number, a retired rule is no number; (b) it changes no recorded classification — probe 1
   at 0.036571 was under it, probe 2 at 0.160224 fails the 0.10 comparison regardless; (c)
   it was a stop rule for a three-attempt preflight (addendum-1 §8), never a criterion of
   the ten-run mandate, and its early-warning function passes to the §4 trace and the §9
   STOP conditions, which are stricter than a single-observation trigger; (d) it is retired
   before its next opportunity to fire. It survives only as a reported reference line.
6. An attempt that is INVALID for a reason other than the noise floor (manifest,
   provenance, child gate) voids nothing but carries no statistics (`statistics: null`) and
   is therefore **not evaluable** under item 2 — neither a firing nor a non-firing. A series
   containing a non-evaluable attempt is published in full and reported as "not evaluable
   on 10", which is not SERIES_MET; the number of non-evaluable attempts is published.
7. **This document authorizes exactly one §3 series.** A voided or interrupted series is
   published as such with every attempt it contains and *is* this document's series; a
   further series requires a new pre-registration that cites this one's outcome and may not
   begin until that is ratified.

## 4. The drift trace — the measurement that answers the feasibility question

### 4.1 What it is and is not

An **analysis-grade** timing trace used to estimate the clean between-window differential
distribution at the 2.6-second lag. *Analysis-grade* means: a script with no schema file,
no validator, no manifest, no lock, no rollback, no versioned format, no CLI subcommand, no
class hierarchy and no new abstraction; plain functions, `argparse` with one positional
argument, stdlib `json`, and only the imports in §4.2. It is **not** a bundle, **not**
gated, produces **no verdict**, and carries **no manifest**. **No README, abstract, summary
table, badge or headline sentence may cite any number derived from a drift trace —
including any limit adopted under §6, any percentile, and any exceedance fraction — except
inside a section titled "Calibration (analysis-grade, not evidence)" whose first sentence
states that the trace is un-manifested, un-gated and produces no verdict, which reproduces
the exact trace and analysis commands, and which appears after the probe-2 record. A
drift-trace number may not appear in the same sentence, table row or figure as a speedup
claim or an A/A verdict.**

Why the lag is 2.6 s: between an attempt's two timed windows lie run B's cold compile
(archived 2.244 s and 2.268 s) and its warm-cache compile (0.053 s and 0.052 s); the
archived window-start lags are 2.605 s and 2.632 s. The lag is the compile time between
the two runs, not a host property, which is why it is fixed rather than measured. The
trace runs continuously with no compile between blocks, so it measures host differential
at that lag under steady load — a **lower bound** on what an attempt sees. The §3 series
measures the attempt. Both are reported; neither replaces the other.

### 4.2 Construction (prescribed — this is the permitted solution space)

- **Entry point:** `tools/drift_trace.py` (create `tools/`; **no** `tools/__init__.py`).
  Not a CLI subcommand, not a console entry, not imported by anything under
  `src/hephaestus/`. Guarded by `if __name__ == "__main__":`. One positional argument: the
  workload name; a `--baseline-only` flag (§5).
- **Permitted imports for `tools/drift_trace.py`, exhaustive:**
  `hephaestus.measure._measure_interleaved`, `hephaestus.measure._SystemClock`,
  `hephaestus.torchbind.inductor_cache_scope`, `hephaestus.torchbind.reset_compiler_state`,
  `hephaestus.torchbind.clear_compiler_memory_state`, `hephaestus.torchbind.compile_module`,
  `hephaestus.torchbind.CompileRequest`, `hephaestus.torchbind.capture_compiler_evidence`,
  `hephaestus.torchbind.read_compiler_evidence`, `hephaestus.workloads.get_workload`,
  `hephaestus.host_state.sample_host_state`, `torch`, and the standard library. Nothing else from the package — in
  particular not `durability`, `durable_worker`, `bundle`, `catalog`, `catalog_runtime`,
  `evidence_contract`, `input_plan` or `provenance`. Importing private names is authorized
  for this script only, and the script says so in a comment. **The trace script imports
  `torch` and every `hephaestus.*` name lazily inside `main()`; row construction and the
  duration/pairing helpers are pure functions taking their inputs as arguments, so
  `import tools.drift_trace` is torch-free** (which is what lets the §7 tests be torch-free
  and keeps a `torch_capabilities.json` mismatch from tripping a §9 STOP inside a unit test).
- **Workloads and requests (literal; the script derives nothing from the catalog and
  computes no digest):** `graph_break_bait` with
  `CompileRequest(backend="inductor", mode=None, dynamic=False, fullgraph=False, options=None, disable=False)`
  — the archived A/A attempts' config, digest reference `9a7509d09749`; then `mlp_stack`
  with `CompileRequest(backend="inductor", mode=None, dynamic=False, fullgraph=True, options=None, disable=False)`
  — the `aa-test` contract config `candidate-mlp-default`, digest reference `0bb4c54b98c6`
  (no archived A/A attempt exists for this workload). `mlp_stack` replaces addendum-1 §8's
  `transformer_block` because it is the headline v1→v2 repair workload and its compiled
  call is the shortest of the four (~0.22 ms), so relative timer quantization is largest —
  the harder case for the instrument, not the easier. The header records the six request
  fields verbatim and the digest as a copied literal string. Both workloads are required
  before §6 may be invoked. `primary_case = get_workload(name).input_cases()[0]` — both
  workloads have exactly one case.
- **Compile once — the exact step list (mirrors `measure()`; deviations named):** enter
  `inductor_cache_scope()`; `reset_compiler_state()`; `module = workload.make_module()`;
  then, **inside `with torch.inference_mode():` for everything that follows including the
  entire block loop**: one eager pass `module(*primary_case)`; inside
  `capture_compiler_evidence()`: `cold = compile_module(module, request)` and
  `cold(*primary_case)`; `clear_compiler_memory_state()`; inside a **second**
  `capture_compiler_evidence()` that stays open for the whole block loop (as `measure()`'s
  operational phase does): `compiled = compile_module(module, request)` and
  `compiled(*primary_case)` — **this warm-cache callable is the one `measure()` times and
  the one the trace times**; then five warm-up iterations of `module(*primary_case)`
  followed by `compiled(*primary_case)` (`warmup_runs = 5`). Then **do not recompile.**
  Deviations from `measure()`, written to the trailer as `setup_deviations`: no
  non-primary compile sweep, no accuracy comparison, no bundle. The trailer's `recompile_reasons` is read by
  calling `read_compiler_evidence()` **after the second `capture_compiler_evidence()`
  context has exited** (normally or by exception) — it returns only the most recently
  *completed* capture, so calling it inside the open context would return the cold-compile
  capture and make the check vacuous; the `finally:` that assembles the trailer therefore
  sits outside that `with`. That value goes into the trailer; **a non-empty list
  means the do-not-recompile invariant was violated: the trace is kept and reported as
  `recompiled: true`, and §6 may not be invoked on it.** Repeated `measure()` calls are
  forbidden for this purpose: each recompiles inside its own cache scope (archived cold
  compiles 2.2–5.9 s) and would turn minutes into hours while changing what is measured.
- **Loop:** call
  `_measure_interleaved(eager=module, compiled=compiled, case=primary_case, repeats=64, spacing_seconds=0.0, clock=clock, schema_version=2)`
  repeatedly. Each call is one **block**. The loop runs until **≥ 600 s have elapsed since
  the first block's first `baseline_timestamps_utc`**, checked before starting each block
  (the last block may end after 600 s); the 60 s baseline and the compile are not counted.
  `declared_duration_seconds: 600` in the header; the achieved block count in the trailer.
  Expect ≈ 6,600–7,400 blocks at the archived 81–88 ms per clean block (a disturbed block
  ran 105 ms).
- **Output:** one plain JSONL file per trace at
  `artifacts/methodology-v2/drift-trace/<workload>-<start UTC as YYYYMMDDTHHMMSS.ffffffZ>.jsonl`
  — a new directory; never inside any `aa-test-*` parent or run bundle; never named
  `attempts.jsonl` or `host_state.jsonl`. **"Durable output root" in this document means
  only the plain path `artifacts/methodology-v2/`, written with `open(path, "w")`; nothing
  under `hephaestus.durability` is used and no ledger is created or appended.** Rows are
  accumulated in memory and written once after the loop, so the trace performs **no disk
  I/O between blocks**. Rows: a header (`git_head`, `git_dirty`, `ratified_commit`,
  workload, the six request fields, `config_digest_reference`, start UTC,
  `declared_duration_seconds`, `torch.get_num_threads()`, `os.cpu_count()`,
  `baseline_load_samples` [60 triples], `baseline_load`, `checklist` [§5, seven booleans],
  `sample_host_state()` at start); one row per block with the six series
  `_measure_interleaved` returns, verbatim, under these fixed names (`eager_seconds`,
  `eager_timestamps_utc`, `baseline_seconds`, `baseline_timestamps_utc`,
  `candidate_seconds`, `candidate_timestamps_utc`); a trailer (end UTC, achieved block count,
  `sample_host_state()` at end, `recompile_reasons`, `setup_deviations`, `stop_reason`, and
  `load_series`). **Every row carries `row` ∈ {`header`, `block`, `trailer`}.** `load_series`
  is a field of the trailer: `os.getloadavg()` triples sampled **on the main thread between
  blocks whenever ≥ 1.0 s has elapsed since the previous sample (no thread)**, from
  immediately before the first block to immediately after the last; its t is seconds since
  the first block's first `baseline_timestamps_utc` — the same clock as the duration — so
  L(0) is the sample taken immediately before the first block.
  Expected size ≈ 70 MB per trace; no disk-space check is performed and none is to be
  added.
- **Nothing under `src/hephaestus/` is modified.** If the script cannot be written without
  a change to the package, STOP and report — do not make the change.

### 4.3 Analysis (fixed now, applied later)

- **Entry point:** `tools/drift_trace_analysis.py`, one positional argument (the trace
  path), output `<same stem>.analysis.json` beside the trace. **Permitted imports,
  exhaustive:** `hephaestus.aa.compute_null_statistics`, `hephaestus.aa._quantile` (a
  private import explicitly authorized for this script only, stated in a comment), and the
  standard library. It imports no `torch`.
- **Block A-role series:** each block's `baseline_seconds` (64 values) — exactly what
  `compiled_seconds` denotes in a bundle.
- **Pairing algorithm (fixed):** sort blocks by their first `baseline_timestamps_utc`
  entry, t_i. Walk i in ascending order; if block i is already paired, skip it; otherwise
  let j be the **smallest index > i** such that block j is unpaired and
  **2.3 s ≤ t_j − t_i ≤ 2.9 s (inclusive)**; if such a j exists, (i, j) is a pair and both
  are marked paired; if not, block i stays unpaired. No other rule (nearest-to-2.6 s,
  index-based, or non-overlapping-only) is used. Reported: the pair count, the unpaired
  count, **and the number of mutually non-overlapping pairs (pairs whose wall-clock spans
  do not intersect)** — that number, not the pair count, bounds the effective sample size
  of the tail estimate: roughly one independent window per 2.7 s of trace, ≈ 130–220 per
  10 minutes. Consecutive pairs are shifted by one block and share almost all of their
  wall clock; they are not independent draws.
- **Per-pair statistic:** `compute_null_statistics(block_i["baseline_seconds"], block_j["baseline_seconds"], schema_version=2)`
  — the same call `aa_runtime.py` makes for two children; its `p95_noise_floor` is the pair
  floor and its two lower bounds are published beside it. Reimplementing any part of it is
  forbidden. **Per-pair record (the only per-pair fields written):** `i`, `j`,
  `lag_seconds`, `p95_noise_floor`, `speedup_lower_bound_a_over_b`,
  `speedup_lower_bound_b_over_a`, `false_positive_arm_fires` (either bound ≥ 1.10). The
  per-pair effects and bootstrap medians are **not** written.
- **Reported, for the compiled series and identically for the eager series
  (`eager_seconds` in place of `baseline_seconds`; `candidate_seconds` stored, not
  analysed; no eager-versus-compiled cross statistic):** the full sorted list of pair
  floors; the 50th, 85.7th, 95th, 99th and **99.488th** percentiles, each computed by
  `aa._quantile` (sort; position (n−1)·p; linear interpolation) — 99.488 = 1 − 0.95^(1/10),
  the Šidák per-attempt exceedance probability for a 5% family-wise false-stop rate over
  ten attempts, recorded so the reader can see how far the clean tail actually reaches;
  the count and fraction of pairs **at or above** 0.10 and **at or above** 0.07; the count
  and fraction of pairs on which the false-positive arm fires — the empirical per-pair rate
  of the §3 (2a) outcome under steady load, expected to be zero, reported not explained;
  the classification (§5) computed from the recorded numbers.
- **Scope of the analysis script:** it analyses one trace. Attempt classification for the
  §3 series and pooling across traces are *not* this action's script; this document fixes
  only their method.
- **Uncertainty (fixed now):** the 95% confidence interval of each percentile is a
  moving-block bootstrap over pairs in time order with block length 64 pairs (under the
  pairing rule, 64 consecutive pairs span ≈ 10 s of wall clock, several times the lag),
  2000 resamples, seed 0, quantiles by `aa._quantile`. The lag-1 through
  lag-64 autocorrelations of the pair-floor series and the implied effective sample size are
  published beside it. Stated now so it cannot surprise: ten minutes holds ≈ 231 disjoint
  2.6-second windows, so the 99.488th percentile — a 1-in-195 event — is estimated from on
  the order of one to two hundred effectively independent windows, and its interval will be
  wide.

## 5. Quiet-host protocol and classification (declared before, never fitted after)

- **Operational checklist**, supplied by the operator as seven `--checklist-*` boolean
  flags (absent = false) and recorded verbatim in the header as seven booleans: AC power; lid open; display unlocked; no other user applications; no
  other Codex/agent process; no test suite or browser; `git status --porcelain` empty.
- **Baseline:** `tools/drift_trace.py` itself captures `os.getloadavg()` once per second
  for 60 s before compiling, writes all 60 triples to the header, and sets `baseline_load`
  to the mean of their 1-minute figures (the kernel refreshes it on a ~5 s tick, so about
  twelve distinct values are averaged). **Exactly one baseline per trace**; an abandoned
  baseline is still written as an aborted trace header and counts as a trace start under
  the §5 daily cap. The §3 series baseline is captured the same way by the same script in
  `--baseline-only` mode immediately before the loop: **in that mode the script writes a
  header-only file at the §4.2 path with `declared_duration_seconds: 0` and
  `mode: "baseline_only"`, then exits; it is not a trace start under the daily cap.** The script records; it never refuses
  to start on the strength of the baseline.
- **Classification rule (fixed here; computed by `tools/drift_trace_analysis.py` from the
  recorded numbers — a hand-assigned classification is invalid).** The trace is itself a
  zero-spacing busy loop, so its own steady-state contribution to the 1-minute figure is at
  least 1.0 and bounded by its thread count; the rule accounts for that. Let L(t) be the
  1-minute figure from `load_series`. A trace is **quiet** iff (i) L(0) ≤ `baseline_load`
  + 1.0; (ii) L(120 s) ≤ `baseline_load` + 1.0 + `torch.get_num_threads()`; and (iii) for
  every t ≥ 120 s, L(t) ≤ L(120 s) + 1.0. Otherwise **disturbed**. A §3 attempt is quiet
  iff the `before` load of run A and the `after` load of run B in `host_state.jsonl`
  (`["load_average"]["value"][0]`) are both ≤ `baseline_load` + 1.0 (an attempt lasts
  ~10 s; its own contribution is small and accepted). If any required value is `null`
  (unavailable), the classification is **disturbed**. On this 14-logical-core host (10
  performance + 4 efficiency), +1.0 corresponds to roughly one core continuously busy for
  the minute. **Classifying by the floor value is forbidden** — it is circular and is itself
  a goalpost move.
- **Known limitations, recorded:** load average cannot resolve a 2.6-second excursion; it
  detects a sustained disturbance, not a transient one, so a transient like probe 2's may
  be classified *quiet*. A disturbance already present at baseline time inflates
  `baseline_load` and is not detected. A disturbance already present at t = 120 s and
  sustained thereafter is absorbed into L(120 s) within rule (ii)'s headroom and is
  invisible to rule (iii); the published per-second series exists so a reader can see it.
  All three are accepted and stated; they are why the
  trace's non-overlapping-pair count and confidence interval are published.
- **Every trace ever started is published** with its header, its per-second load series
  and its classification — quiet, disturbed or interrupted — and none is discarded. The
  declared duration is the same for every trace. **At most two trace starts per workload
  per calendar day; a second disturbed trace on one day is a STOP for the owner, not a
  third start.**

## 6. Adoption rule and falsification check (what a future limit would need)

This section creates **no** limit. It fixes the only route by which one could be earned:

1. Any future cross-run limit is the **99.488th percentile of the pooled quiet null**:
   every quiet trace of both workloads on **every calendar day traced (at least two)**,
   pooled into one pair-floor sample in chronological order, computed by `aa._quantile`,
   with the §4.3 moving-block bootstrap applied to the concatenated series. The second
   day's traces and the pooled computation are a **separate authorization after Commit C**,
   using the same script and the same declared duration. **No trace, block, pair or time segment may be excluded**;
   the number of disturbed and interrupted traces is published beside it.
2. **Replication** means: each individual quiet trace's 99.488th percentile lies inside the
   pooled estimate's 95% confidence interval (§4.3). If any does not, the null is not
   stationary across days or workloads and 0.10 stays. All values are published.
3. **Falsification**, three parts, each applied to the **95% confidence interval** of the
   pooled percentile, never to its point estimate: (a) the interval's upper end must
   classify probe 2's 0.16022381230908075 as an exceedance; (b) the interval's upper end
   must classify every one of the ten contaminated pairs in ruling 2 §1's matrix as an
   exceedance — it must be **strictly below 0.1377**; (c) the interval's lower end must lie
   **above the largest archived clean pair, 0.0617**, or the derivation is broken. The
   admissible band is therefore (0.0617, 0.1377); 0.10 lies inside it. If any part fails,
   the derivation is wrong or the host is unfit, and 0.10 stays. These three archived
   numbers are fixed here and are never recomputed from later data.
4. Old and new limits are published side by side with the direction and size of the change
   stated plainly.
5. Until all four hold, **0.10 remains the cross-run bar**, and this document commits to
   publishing whatever the trace yields — including a tail that makes ten quiet attempts
   look unlikely.

## 7. Untouchable surface and permitted solution space

**Untouchable:** everything in §1; every file under `src/hephaestus/`; `gates/`;
`pyproject.toml`; `tests/test_methodology_archive.py`; the protected archive; every
existing bundle and ledger under `artifacts/`; `torch_capabilities.json`.

**Permitted, exhaustive:** `tools/drift_trace.py`; `tools/drift_trace_analysis.py`;
`tests/tools/__init__.py` (one-line docstring, matching `tests/__init__.py`, so the files
collect as `tests.tools.*` with the repository root on `sys.path` — **no** `conftest.py`,
**no** `pythonpath` setting, **no** `tools/__init__.py`); **exactly three** torch-free,
non-`slow` test files importing the scripts as `tools.drift_trace` /
`tools.drift_trace_analysis`: `tests/tools/test_drift_trace_rows.py` (header/block/trailer
rows round-trip through `json.loads` with the §4.2 field names, using a fake
`_measure_interleaved` and a fake `sample_host_state`),
`tests/tools/test_drift_trace_pairing.py` (the §4.3 pairing on synthetic timestamps:
window edges 2.3/2.9 inclusive, disjointness, unpaired tail, non-overlapping count),
`tests/tools/test_drift_trace_analysis.py` (percentiles, exceedance counts and the
classification rule on synthetic inputs; that `compute_null_statistics` is called with
`schema_version=2`). No test compiles anything, imports torch, parses the scripts' import
statements, inspects git, or asserts anything about `src/hephaestus/`. Trace outputs and analysis outputs under `artifacts/methodology-v2/`. **For Commit C
only:** `docs/calibration/<trace stem>.analysis.json` copies,
`docs/calibration/drift-trace-report.md`, and the out-of-tree archive directory
`drift-trace-<date>/`. One new ledger file
`.superpowers/sdd/2026-08-30-methodology-v2/drift-trace-report.md`, which is a
one-paragraph pointer to the `docs/calibration/` report and **contains no numbers**;
nothing appended to any existing file.

**Not this action:** the §3 series (a separate authorization that follows the trace); any
README; any change to `aa-test`.

## 8. Commits and review budget

- **Commit A:** this file, unchanged. Its hash is `ratified_commit`.
- **Commit B:** `tools/drift_trace.py`, `tools/drift_trace_analysis.py`, `tests/tools/` —
  one commit, reviewed **once** adversarially against §4 and §7 before any trace runs; at
  most **two** fix rounds; a third needed → STOP and report.
- **No trace may start** unless `git status --porcelain` is empty and HEAD is Commit B or a
  descendant containing it; the header records `git_head` and `git_dirty`.
- Then: baseline, trace `graph_break_bait`; baseline, trace `mlp_stack`; run the analysis on
  **every** trace file that exists.
- **Commit C:** the `.analysis.json` files copied to `docs/calibration/` plus one Markdown
  report `docs/calibration/drift-trace-report.md` reproducing the §4.3 numbers verbatim
  (no plots, no HTML, no manifest). The raw JSONL stays untracked under `artifacts/` and is
  copied to the out-of-tree evidence archive as a **new sibling directory**
  `drift-trace-<date>/`, never inside `task5-aa-stop-20260831`.
- Commits A–C are never amended, squashed or rebased once a trace has started.
- Then **STOP** for the owner. The §3 series is a separate authorization.

## 9. STOP conditions

- The script cannot be written without modifying anything under `src/hephaestus/`.
- `import torch`, `import hephaestus.torchbind` or `compile_module` raises (including the
  `torch_capabilities.json` mismatch `RuntimeError`): STOP before any file is created;
  report the exception text verbatim; do not refresh `torch_capabilities.json`.
- The trace process is interrupted (`KeyboardInterrupt`) or any exception escapes the
  block loop: the trailer is written with the matching `stop_reason`, the partial file is
  kept and reported as partial, and any fresh trace is a **new** file under the §5 daily
  cap. No trace file — partial or complete, quiet or disturbed — is ever deleted,
  overwritten or omitted.
- The trailer's `recompile_reasons` is non-empty: report; the trace may not feed §6.
- A trace is classified *disturbed* at start (§5 rule i): record it, do not discard it,
  and STOP (a rule-(i) disturbance is a STOP even though §5 would otherwise permit a second
  start that day).
- Either trace yields fewer than 1,000 block pairs at the declared lag, **or fewer than
  100 mutually non-overlapping pairs**.

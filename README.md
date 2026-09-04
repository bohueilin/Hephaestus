# Hephaestus

![Hephaestus forging a compiler graph through a verification gate into a sealed evidence artifact](Hephaestus_github.png)

## What this is

Hephaestus is an evidence-gated `torch.compile` optimization lab. An optimization policy proposes from a closed catalog, the environment executes the candidate, and an offline gate derives `PROVEN`, `NOT_PROVEN`, `CONDITIONAL`, or `INVALID_EVIDENCE` from stored evidence.

The gate is tested against planted regressions. The policy cannot execute code, change criteria, or decide a verdict. The verdict is a pure function of stored evidence bytes. Performance numbers are machine-local.

`Machine-local CPU evidence; laptop thermal state may affect timings.`

## The headline result

The same configuration digest produced an in-run A/A noise floor of `0.10393068646259071` under methodology v1 (the number that stopped the project) → `0.024382885822453626` under methodology v2. The v1 bundle is committed: `.venv/bin/hephaestus gate tests/data/archived_v1_bundle` returns `INVALID_EVIDENCE / methodology.noise_floor` with exit `1`. The exact v2 command is `.venv/bin/hephaestus gate artifacts/methodology-v2/task-4/planted-demo-20260831T020425.145134Z/runs/20260831T020439.343896Z-mlp_stack-f0946b9c24f2`, which returns `PROVEN`; that bundle is preserved in the local evidence archive and is not in the repository. Both floors are read from each bundle's `methodology.json` key `aa_noise_floor`; the gate prints the verdict, not the floor. The two `config.json` files are byte-identical (SHA-256 `f0946b9c24f20eee32535e9dad507e9aa88fddb91da41aabce9d438b6dee6496`).

A clean clone includes the v1 fixtures under `tests/data/`, calibration analyses under `docs/calibration/`, and the series-1 verdicts under `docs/series-1/`. Other cited artifacts remain on the measuring machine; their composite digests are recorded here.

The v1 parity split is reproduced from committed evidence with this stdlib-only command:

```text
python -c 'import json,statistics; x=json.load(open("tests/data/archived_v1_bundle/timings.json"))["aa_signed_paired_effects"]; print(statistics.median(x[::2])); print(statistics.median(x[1::2]))'
0.1137070355672043
-0.023922068034116763
```

> Determinism: the same config gated twice from stored evidence yields byte-identical verdicts; note (honestly) that raw TIMINGS are not byte-reproducible — the gate's determinism is over stored evidence, and fresh runs may differ within the noise floor.

## Quick start

Hephaestus requires Python 3.14. The resolved development environment is pinned in `requirements.lock`; this editable install uses the project metadata in `pyproject.toml`.

```text
uv venv --python 3.14
uv pip install -e '.[dev]'
.venv/bin/hephaestus gate tests/data/archived_v1_bundle
```

That final command performs no measurement. It re-gates a committed evidence bundle from stored bytes and should return `INVALID_EVIDENCE / methodology.noise_floor` with exit `1`.

## Commands

The first four examples were executed once against the code in this repository. Output is captured and trimmed only by deleting whole lines or replacing a run of lines with `…`.

```text
.venv/bin/hephaestus run mlp_stack --config candidate-mlp-default --output-root artifacts/readme-capture/
{"boundary":"Machine-local CPU evidence; laptop thermal state may affect timings.","driving_finding":"all_criteria_passed","evidence_path":"artifacts/readme-capture/runs/20260903T235457.173897Z-mlp_stack-0bb4c54b98c6 (path abbreviated)","verdict":"PROVEN"}
```

```text
.venv/bin/hephaestus gate tests/data/archived_v1_bundle
{"boundary":"Machine-local CPU evidence; laptop thermal state may affect timings.","complete_verdict":{"driving_finding":"methodology.noise_floor","findings":[{"id":"methodology.noise_floor","status":"FAIL"}],"verdict":"INVALID_EVIDENCE"},"driving_finding":"methodology.noise_floor","evidence_path":"tests/data/archived_v1_bundle","verdict":"INVALID_EVIDENCE"}
```

Exit: `1`.

```text
.venv/bin/hephaestus agent optimize mlp_stack --output-root artifacts/readme-capture/
{"boundary":"Machine-local CPU evidence; laptop thermal state may affect timings.","driving_finding":"all_criteria_passed","evidence_path":"artifacts/readme-capture/agent-search-20260903T235527.445440Z-mlp_stack (path abbreviated)","receipts":[{"proposal":{"catalog_id":"candidate-mlp-default","rationale":"Try the calibrated native default with a required full graph.","workload_name":"mlp_stack"},"result":{"bundle_relative_path":"runs/20260903T235532.949175Z-mlp_stack-0bb4c54b98c6","driving_finding":"all_criteria_passed","verdict":"PROVEN"}}],"verdict":"PROVEN"}
```

```text
.venv/bin/hephaestus demo-planted-regressions --output-root artifacts/readme-capture/
Machine-local CPU evidence; laptop thermal state may affect timings.
catalog_id | expected_verdict | expected_finding | actual_verdict | actual_finding | pass | bundle
planted-eager-fallback | NOT_PROVEN | perf.speedup_proven | NOT_PROVEN | perf.speedup_proven | yes | runs/20260903T235604.365398Z-mlp_stack-6bd2b64fdf89
planted-static-shape-storm | NOT_PROVEN | graph.recompile_bound | NOT_PROVEN | graph.recompile_bound | yes | runs/20260903T235613.413512Z-dynamic_batch_text-dff92fdba073
planted-graph-break-exposure | CONDITIONAL | graph.no_breaks | CONDITIONAL | graph.no_breaks | yes | runs/20260903T235615.785611Z-graph_break_bait-2e19426b4299
clean-control-mlp | PROVEN | all_criteria_passed | PROVEN | all_criteria_passed | yes | runs/20260903T235618.597572Z-mlp_stack-f0946b9c24f2
evidence_path: artifacts/readme-capture/planted-demo-20260903T235604.158530Z (path abbreviated)
```

```text
.venv/bin/hephaestus aa-test graph_break_bait
```

From the archived probe; not a fresh run:

```json
{"statistics":{"p95_noise_floor":0.036571047775528484,"speedup_lower_bound_a_over_b":0.9822859469861602,"speedup_lower_bound_b_over_a":0.9665664937167092},"verdict":"PASS","driving_finding":"all_criteria_passed"}
```

## Planted regressions

This is the preserved v2 recalibration table, not the fresh command output above. Every planted defect is caught by its named criterion, and the clean control passes.

| catalog ID | named criterion | verdict | passed | bundle |
| --- | --- | --- | --- | --- |
| `planted-eager-fallback` | `perf.speedup_proven` | `NOT_PROVEN` | `true` | `runs/20260831T020425.352068Z-mlp_stack-6bd2b64fdf89` |
| `planted-static-shape-storm` | `graph.recompile_bound` | `NOT_PROVEN` | `true` | `runs/20260831T020434.201129Z-dynamic_batch_text-dff92fdba073` |
| `planted-graph-break-exposure` | `graph.no_breaks` | `CONDITIONAL` | `true` | `runs/20260831T020436.547141Z-graph_break_bait-2e19426b4299` |
| `clean-control-mlp` | `all_criteria_passed` | `PROVEN` | `true` | `runs/20260831T020439.343896Z-mlp_stack-f0946b9c24f2` |

The complete planted-demo archive composite is `d65cfb377025125dec7c8dca50c80d9e807119fb9816094ff5c886f34d6bb2ad` in the local evidence archive.

## What the gate refuses, and what the agent cannot do

The agent can make an inert proposal from the closed catalog. It cannot execute outside that catalog, alter criteria, alter evidence, or decide a verdict. The environment executes and the gate decides.

The captured `mlp_stack` transcript records the first proposal and its result:

```json
{"schema_version":1,"steps":[{"proposal":{"catalog_id":"candidate-mlp-default","rationale":"Try the calibrated native default with a required full graph.","workload_name":"mlp_stack"},"result":{"bundle_relative_path":"runs/20260903T235532.949175Z-mlp_stack-0bb4c54b98c6","driving_finding":"all_criteria_passed","verdict":"PROVEN"}}],"workload_name":"mlp_stack"}
```

The separately captured `transformer_block` transcript records a proposal that the gate refused:

```json
{"schema_version":1,"steps":[{"proposal":{"catalog_id":"candidate-transformer-default","rationale":"Try the calibrated native default on the transformer block.","workload_name":"transformer_block"},"result":{"bundle_relative_path":"runs/20260903T235548.660295Z-transformer_block-671abd1c453e","driving_finding":"perf.speedup_proven","verdict":"NOT_PROVEN"}}],"workload_name":"transformer_block"}
```

## The record: what went wrong and what was done about it

The rulings below, beginning with owner ruling 1 (2026-08-30), are recorded in a project decision ledger that is not part of this repository. Every number in this section was independently verified before acceptance.

- The historical audit corrected the count to seven Important evidence defects, not four: environment/workload/config/topology binding; accuracy-tolerance binding; a filesystem cache hit mislabeled cold; compiler-path privacy leakage; cloned or unordered A/A provenance; porous optimizer execution authority; and a wheel missing packaged pinned criteria. In the first demo, an unpreserved and unverifiable `0.1154` observation preceded the preserved v1 stop at `0.10393068646259071`. Its `+0.1137 / -0.0239` parity split led to the versioned v2 repair, with `0.10`, `1.10`, criteria, and bootstrap frozen.
- owner ruling 1 (2026-08-30) was corrected by the errata: the mistaken schema-v2 `32` repeats/effects was corrected to the controlling `64` while v1 remained `31`; the claim that A-role adjacency stayed unchanged was false, so dependent numbers were re-derived.
- owner ruling 2 (2026-08-31) accepted Task 5's STOP: probe 1 was `0.036571` and PASS; probe 2 was `0.16022381230908075`, permanent `FAIL / methodology.noise_floor`; the probe archive composite is `bf32ad1d531295a3872fc2798bccef6f2b80376def2c564529e0a91f9837dd4c`; and there was no retry. It refused raising `0.10`, lowering `1.10`, raising repeats, trimming samples, or rerunning until pass. It also recorded three corrections owned in that ruling: a refuted monotone-drift proposal, an unjustified `0.07`, and unseen n-dependence of the prospective `32`→`64` change.
- The authorized host-ledger work reached its terminal fifth fix round without acceptance: a `BaseException` after write could leave partial `b'{"after":'` bytes, or an interruption during post-write provenance read could leave a complete unvalidated `975`-byte row. No sixth patch was authorized. owner ruling 3 (2026-09-01) later closed that defect by deleting the rollback protocol rather than by extending the patch sequence.
- Instrumentation then acquired four ways to gate or misreport science: append before receipt, run-B abort, close failure mapped to scientific failure, and historical O(N²)/missing-bundle denial. owner ruling 3 (2026-09-01) recorded owner errors 1–4: unspecified cross-file integrity; “gates nothing” not made a call-site invariant; a missing untouchable/solution boundary; and a partially wrong rollback/torn-tail hypothesis.
- owner ruling 4 (2026-09-01) accepted the subtractive cut and closed stage 6 by deletion: `5,240` insertions had produced zero rows; three subprocess probes were deleted, leaving only fork-free load-average sampling. It recorded owner errors 5–7: the wrong instrument was mandated as a precondition; torn-tail reasoning was self-contradictory; and authorizations were repeated without solution boundaries. It also decided that the cross-run floor is a validation instrument outside bundle evidence, refused to reclassify probe 2, and kept `0.10`.
- owner ruling 5 (2026-09-01) recorded owner error 8: the proposed series criterion was unfailable; its draft location and quiet-host rule were also wrong.
- owner ruling 6 (2026-09-02) accepted the calibration-analysis implementation while recording the missing aborted-baseline write, a lost interrupted label, dependency-injection scope creep, and a report that omitted the realized-lag and ESS/non-overlap caveat. Amendment 1 recorded owner errors 9–13: underspecified bootstrap; ESS confused with the operative tail sample; undefined non-overlap count; realized lag presented as the motivating lag; and UTC-day semantics omitted. Owner error 14 was that an abandoned-baseline `KeyboardInterrupt` wrote no header; a subsequent trace-script fix corrected it. Owner error 15 was that operator, checklist, settle, loop, classification, precedence, and result language were not frozen before the series; Amendment A10 fixed them before execution. Owner error 16 was that adoption impossibility from the sole quiet trace was foreseeable but unstated. Owner error 17 was that probe 2 was wrongly framed as an exogenous disturbance; a compiled-path-tail interpretation is plausible but unproven, and its permanent FAIL is unchanged.
- The README-capture controller first offline-gated a fresh local bundle instead of the required committed v1 fixture. It preserved that erroneous result, then gated the required fixture exactly once. The correction performed no measurement, changed no bundle, and was neither a retry nor an `aa-test`; the erroneous result is not presented as a required capture.

Calibration's disposition is stated here only in words: no limit was adopted, and `0.10` is unchanged in both directions.

## A/A meta-test status

**Pre-series status (historical):**
DESIGN.md §6.2's ten-consecutive-run requirement was NOT DEMONSTRATED. Its pre-registered
replacement had not yet run.

**Series 1 result:**
DESIGN.md §6.2 is not met on this host: 1 of 10 attempts at or above 0.10 and 1 false-positive fires (all ten floors, both bounds and the host-state classification per attempt listed below). Final for this host under pre-registration 1 §3 (2b). This invalidates no published bundle and moves no limit.

| attempt | parent directory (repo-relative) | start UTC | §5 classification (quiet/disturbed) with raw `before`/`after` loads | p95 floor | lower bound A/B | lower bound B/A | arm fired (yes/no) | verdict / driving finding |
| ---: | --- | --- | --- | ---: | ---: | ---: | --- | --- |
| 1 | `artifacts/methodology-v2/series-1/aa-test-20260904T012158.412717Z-graph_break_bait` | `2026-09-04T01:21:58.412717Z` | quiet (`before` 5.3994140625; `after` 5.287109375) | 0.29938933135166906 | 1.2868892731860855 | 0.7335772682461127 | yes | FAIL / `methodology.noise_floor` |
| 2 | `artifacts/methodology-v2/series-1/aa-test-20260904T012239.799190Z-graph_break_bait` | `2026-09-04T01:22:39.799190Z` | quiet (`before` 6.18359375; `after` 5.9208984375) | 0.04341021231215218 | 0.957721167740961 | 1.0047992378643977 | no | PASS / `all_criteria_passed` |
| 3 | `artifacts/methodology-v2/series-1/aa-test-20260904T012320.476847Z-graph_break_bait` | `2026-09-04T01:23:20.476847Z` | quiet (`before` 5.3662109375; `after` 5.0146484375) | 0.08728424616310919 | 0.9517558433765605 | 1.0081946498466354 | no | PASS / `all_criteria_passed` |
| 4 | `artifacts/methodology-v2/series-1/aa-test-20260904T012401.559145Z-graph_break_bait` | `2026-09-04T01:24:01.559145Z` | disturbed (`before` 6.3037109375; `after` 5.87548828125) | 0.03157474981427261 | 0.9822284515941258 | 0.9846816874731518 | no | PASS / `all_criteria_passed` |
| 5 | `artifacts/methodology-v2/series-1/aa-test-20260904T012444.163748Z-graph_break_bait` | `2026-09-04T01:24:44.163748Z` | disturbed (`before` 7.9814453125; `after` 8.5732421875) | — | — | — | not evaluable | INVALID_EVIDENCE / `child.invalid_evidence` (`statistics: null`) |
| 6 | `artifacts/methodology-v2/series-1/aa-test-20260904T012527.414444Z-graph_break_bait` | `2026-09-04T01:25:27.414444Z` | disturbed (`before` 6.5224609375; `after` 5.90673828125) | 0.017873636502292078 | 0.9746905719593802 | 0.9934477479148106 | no | PASS / `all_criteria_passed` |
| 7 | `artifacts/methodology-v2/series-1/aa-test-20260904T012607.576461Z-graph_break_bait` | `2026-09-04T01:26:07.576461Z` | quiet (`before` 5.64501953125; `after` 5.458984375) | 0.06686831662095535 | 0.950941381905603 | 1.012101002201962 | no | PASS / `all_criteria_passed` |
| 8 | `artifacts/methodology-v2/series-1/aa-test-20260904T012648.037288Z-graph_break_bait` | `2026-09-04T01:26:48.037288Z` | quiet (`before` 5.2314453125; `after` 5.212890625) | 0.024613088558489503 | 0.9794411313587884 | 0.9972563897444874 | no | PASS / `all_criteria_passed` |
| 9 | `artifacts/methodology-v2/series-1/aa-test-20260904T012728.117300Z-graph_break_bait` | `2026-09-04T01:27:28.117300Z` | quiet (`before` 5.1962890625; `after` 5.02001953125) | 0.03174286054270926 | 0.9703935894865381 | 0.9867915085137707 | no | PASS / `all_criteria_passed` |
| 10 | `artifacts/methodology-v2/series-1/aa-test-20260904T012808.241204Z-graph_break_bait` | `2026-09-04T01:28:08.241204Z` | disturbed (`before` 8.1640625; `after` 7.83056640625) | 0.03345593637607722 | 0.9792967836379174 | 0.9776504745881198 | no | PASS / `all_criteria_passed` |

At or above 0.07 (reference line only): 2 of 10.

> **Disclosure recorded after the run.** The committed baseline header above asserts all
> seven operational checklist flags as true. That is not what happened: the operator
> supplied no post-run honesty line, and the project ledger records that the operator
> interacted with the executing agent's session after launch, so
> `no_other_codex_agent_process` and `no_other_user_applications` were not true for the
> entire loop and at least one keystroke or window switch occurred. The baseline 1-minute
> load was 5.29 on a 14-logical-core host, so the table's "quiet" labels are relative to a
> loaded baseline (threshold 6.29). Under pre-registration 1 §3 the series score depends on
> neither the checklist nor the §5 label, the series is published exactly as it stands,
> and it was not re-run. Attempt 5 is non-evaluable because one of its two runs was
> invalidated by its own in-run gate (`INVALID_EVIDENCE / methodology.noise_floor`, in-run
> floor 0.1159 on a run whose compiled median was about twice its sibling's); that child's
> evidence is in the local series-1 archive and is not committed.

## Calibration (analysis-grade, not evidence)

The drift traces cited in this section are un-manifested, un-gated and produce no
verdict; the exact trace and analysis commands, the raw-trace stems and the per-pair
analysis files are reproduced in `docs/calibration/drift-trace-report.md` under
pre-registration 1 (`docs/calibration/preregistration-1-cross-run-calibration.md`),
ratified before measurement and amended by
`docs/calibration/preregistration-1-amendment-1.md`). No published claim in this
repository crosses runs: every speedup claim is formed inside a single run from one
bundle's timings, and the cross-run A/A floor is a validation instrument outside the
evidence contract, so a cross-run failure invalidates no published bundle and a cross-run
pass proves nothing about one. What that instrument's own null looks like on this host
was measured once, on 2026-09-02 UTC, by a ten-minute steady loop of `graph_break_bait`
with no compile between timing blocks on a host classified quiet by the pre-registered
load-average rule (a rule that detects sustained external load, not transient episodes;
five of the seven operational checklist flags were unasserted): identical compiled timing
windows paired 2.30–2.42 s apart gave a 99.488th-percentile pair floor of 0.2616 — the
same estimator an attempt applies to its two windows — with a 95%
moving-block-bootstrap interval of 0.1302 to 0.2938 (3,875 pairs, of which 201 are
mutually non-overlapping and are the operative sample size for that tail), 59 of 3,875
pairs (1.5%) at or above the frozen 0.10 limit, and 21 pairs (0.54%) on which the
false-positive arm fired on unchanged code, every one of them also at or above 0.10; the
eager series measured in the same blocks shows no such tail (its 99.488th percentile was
0.041 and no eager pair reached 0.10); these are lower bounds on what an attempt sees,
because an attempt also carries a compile between its two windows. A second trace,
`mlp_stack`, was classified disturbed by the load rule, is published in full, and enters
no pool. Under the pre-registered falsification test the `graph_break_bait` interval
fails parts (a) and (b), so no cross-run limit was adopted, the 0.10 limit is unchanged in
both directions, and no later pooling can change that while this trace is in the pool,
which the pre-registration forbids removing. Probe 2's permanent FAIL /
`methodology.noise_floor` is recorded above and is unchanged.

### Commands recorded

```text
.venv/bin/python tools/drift_trace.py graph_break_bait --checklist-ac-power --checklist-clean-git-status
.venv/bin/python tools/drift_trace_analysis.py artifacts/methodology-v2/drift-trace/graph_break_bait-20260902T011653.483792Z.jsonl
.venv/bin/python tools/drift_trace.py mlp_stack --checklist-ac-power --checklist-clean-git-status
.venv/bin/python tools/drift_trace_analysis.py artifacts/methodology-v2/drift-trace/mlp_stack-20260902T012826.821445Z.jsonl
```

### Supplemental graph baseline-only record

The following extra header-only record is not a trace start and was not analysed:

```text
.venv/bin/python tools/drift_trace.py graph_break_bait --baseline-only --checklist-ac-power --checklist-clean-git-status
```

Its raw file is `artifacts/methodology-v2/drift-trace/graph_break_bait-20260902T011507.539792Z.jsonl`.

### `graph_break_bait-20260902T011653.483792Z`

Classification: `quiet`. Pair count: `3875`; unpaired count: `23`; mutually non-overlapping pair count: `201`.

Compiled:

| Percentile | Value | 95% bootstrap interval |
| --- | --- | --- |
| 50 | `0.025497146109041434` | [`0.024869269418517716`, `0.026027557260249197`] |
| 85.7 | `0.03920488549909389` | [`0.03699336919376753`, `0.04183964415165236`] |
| 95 | `0.059678188887078105` | [`0.05485885290220482`, `0.06579531321267296`] |
| 99 | `0.13049660465605678` | [`0.09926455084913179`, `0.26654739466480615`] |
| 99.488 | `0.26156911343467654` | [`0.13017056924464904`, `0.2938272185033995`] |

At or above `0.10`: `59` / `0.015225806451612903`. At or above `0.07`: `137` / `0.03535483870967742`. False-positive arm: `21` / `0.005419354838709678`. Implied effective sample size: `631.647710634709`.

Eager:

| Percentile | Value | 95% bootstrap interval |
| --- | --- | --- |
| 50 | `0.013460346682218164` | [`0.013177987600151875`, `0.013762901136749708`] |
| 85.7 | `0.019276124822467957` | [`0.018814303884361748`, `0.019978688899908663`] |
| 95 | `0.025240099809168498` | [`0.02419735252625687`, `0.02623740111733781`] |
| 99 | `0.036617939366194875` | [`0.03399640937754036`, `0.04008943533282007`] |
| 99.488 | `0.04143705831601004` | [`0.0384941520210148`, `0.04531351946421281`] |

At or above `0.10`: `0` / `0.0`. At or above `0.07`: `1` / `0.00025806451612903227`. False-positive arm: `0` / `0.0`. Implied effective sample size: `801.8304683674374`.

This trace completed with `7773` blocks and zero recompiles; it is quiet. It clears the `1000`-pair and `100`-non-overlap STOP minima.

### `mlp_stack-20260902T012826.821445Z`

Classification: `disturbed`. Pair count: `3613`; unpaired count: `4`; mutually non-overlapping pair count: `132`.

Compiled:

| Percentile | Value | 95% bootstrap interval |
| --- | --- | --- |
| 50 | `0.014762574640532078` | [`0.014318163779072583`, `0.015209749797810826`] |
| 85.7 | `0.024319792677733446` | [`0.02332135460343297`, `0.025884041937757344`] |
| 95 | `0.03619607413216279` | [`0.03293247814948669`, `0.039608622483073154`] |
| 99 | `0.0606532968660696` | [`0.056674199018758996`, `0.06652130368439006`] |
| 99.488 | `0.06962558805068138` | [`0.06587476721323662`, `0.07362356099817896`] |

At or above `0.10`: `0` / `0.0`. At or above `0.07`: `18` / `0.00498200941046222`. False-positive arm: `0` / `0.0`. Implied effective sample size: `1117.0356931237357`.

Eager:

| Percentile | Value | 95% bootstrap interval |
| --- | --- | --- |
| 50 | `0.019916301121000807` | [`0.019516818739410566`, `0.02032252893318699`] |
| 85.7 | `0.02897672754372298` | [`0.027977305639488758`, `0.029952339885386647`] |
| 95 | `0.0374906009814349` | [`0.035746390062953264`, `0.03994479416295561`] |
| 99 | `0.06245294498280109` | [`0.05429417589955664`, `0.0718167562094078`] |
| 99.488 | `0.07482960403956186` | [`0.06437889063626298`, `0.08413796422793303`] |

At or above `0.10`: `4` / `0.0011071132023249377`. At or above `0.07`: `25` / `0.006919457514530861`. False-positive arm: `0` / `0.0`. Implied effective sample size: `978.5726503988186`.

This trace completed with `7230` blocks and zero recompiles; it is disturbed by rule (iii). It clears the `1000`-pair and `100`-non-overlap STOP minima.

## What this repository does not claim

It does not claim run-to-run timing reproducibility, a cross-run speedup, anything beyond this machine, or anything beyond CPU.

## Evidence map and reproduction commands

Repository-relative paths under `tests/data/`, `docs/calibration/`, `docs/series-1/`, and `src/` are available in a clean clone. Paths under `artifacts/` and commands using archive-root environment variables require the original measuring machine. Those local paths are included to make the evidence boundary explicit, not to imply that uncommitted bytes ship with the repository.

Frozen protocols, tooling, tests, and recorded headers preserve original development commit identifiers; the pre-registration also preserves one non-public decision-ledger path. Those references intentionally do not resolve in this fresh public history, which therefore cannot independently establish the original commit chronology.

The following are read-only commands. The operator supplies the three role-only archive roots as environment variables; their values are intentionally not published.

For series 1, the operator additionally supplies the role-only series archive root.

```text
: "${PROBE_ARCHIVE:?set to the root of the local probe archive}"
: "${TASK4_DEMO_ARCHIVE:?set to the root of the local complete planted-demo archive}"
: "${CALIBRATION_ARCHIVE:?set to the root of the local calibration archive}"
: "${SERIES1_ARCHIVE:?set to the root of the local series-1 archive}"

python -c 'import json; print(json.load(open("tests/data/archived_v1_bundle/methodology.json"))["aa_noise_floor"])'
python -c 'import json; print(json.load(open("artifacts/methodology-v2/task-4/planted-demo-20260831T020425.145134Z/runs/20260831T020439.343896Z-mlp_stack-f0946b9c24f2/methodology.json"))["aa_noise_floor"])'
.venv/bin/hephaestus gate tests/data/archived_v1_bundle
.venv/bin/hephaestus gate artifacts/methodology-v2/task-4/planted-demo-20260831T020425.145134Z/runs/20260831T020439.343896Z-mlp_stack-f0946b9c24f2
cmp -s tests/data/archived_v1_bundle/config.json artifacts/methodology-v2/task-4/planted-demo-20260831T020425.145134Z/runs/20260831T020439.343896Z-mlp_stack-f0946b9c24f2/config.json && shasum -a 256 tests/data/archived_v1_bundle/config.json artifacts/methodology-v2/task-4/planted-demo-20260831T020425.145134Z/runs/20260831T020439.343896Z-mlp_stack-f0946b9c24f2/config.json
python -m json.tool artifacts/readme-capture/runs/20260903T235457.173897Z-mlp_stack-0bb4c54b98c6/verdict.json
python -m json.tool artifacts/readme-capture/agent-search-20260903T235527.445440Z-mlp_stack/transcript.json
python -m json.tool artifacts/readme-capture/agent-search-20260903T235547.853996Z-transformer_block/transcript.json
python -m json.tool artifacts/readme-capture/planted-demo-20260903T235604.158530Z/demo.json
python -m json.tool "$PROBE_ARCHIVE/aa-test-20260831T024848.562522Z-graph_break_bait/verdict.json"
python -m json.tool "$TASK4_DEMO_ARCHIVE/demo.json"
python -c 'import hashlib,os,pathlib; r=pathlib.Path(os.environ["PROBE_ARCHIVE"]); s="".join(f"{p.relative_to(r).as_posix()}\t{hashlib.sha256(p.read_bytes()).hexdigest()}\n" for p in sorted((p for p in r.rglob("*") if p.is_file() and p.name != "COMPOSITE.txt"), key=lambda p:p.relative_to(r).as_posix())); print(hashlib.sha256(s.encode()).hexdigest())'
python -c 'import hashlib,os,pathlib; r=pathlib.Path(os.environ["TASK4_DEMO_ARCHIVE"]); s="".join(f"{p.relative_to(r).as_posix()}\t{hashlib.sha256(p.read_bytes()).hexdigest()}\n" for p in sorted((p for p in r.rglob("*") if p.is_file()), key=lambda p:p.relative_to(r).as_posix())); print(hashlib.sha256(s.encode()).hexdigest())'
python -c 'import hashlib,os,pathlib; r=pathlib.Path(os.environ["CALIBRATION_ARCHIVE"]); s="".join(f"{p.relative_to(r).as_posix()}\t{hashlib.sha256(p.read_bytes()).hexdigest()}\n" for p in sorted((p for p in r.rglob("*") if p.is_file()), key=lambda p:p.relative_to(r).as_posix())); print(hashlib.sha256(s.encode()).hexdigest())'
sed -n '1,76p' src/hephaestus/agent.py
sed -n '167,199p' src/hephaestus/catalog.py
sed -n '146,200p' src/hephaestus/search.py
sed -n '149,168p' src/hephaestus/catalog_runtime.py
sed -n '1,196p' docs/calibration/preregistration-1-amendment-1.md
sed -n '/^## A\/A meta-test status$/,/^## Calibration (analysis-grade, not evidence)$/p' README.md
sed -n '1,220p' docs/calibration/preregistration-1-cross-run-calibration.md
sed -n '1,94p' docs/calibration/drift-trace-report.md
python -m json.tool docs/calibration/graph_break_bait-20260902T011653.483792Z.analysis.json
python -m json.tool docs/calibration/mlp_stack-20260902T012826.821445Z.analysis.json
python -m json.tool docs/series-1/graph_break_bait-20260904T012056.749366Z.jsonl
for verdict in docs/series-1/aa-test-*/verdict.json; do python -m json.tool "$verdict"; done
python -m json.tool --json-lines "$SERIES1_ARCHIVE/host_state.jsonl"
python -c 'import hashlib,os,pathlib; r=pathlib.Path(os.environ["SERIES1_ARCHIVE"]); s="".join(f"{p.relative_to(r).as_posix()}\t{hashlib.sha256(p.read_bytes()).hexdigest()}\n" for p in sorted((p for p in r.rglob("*") if p.is_file()), key=lambda p:p.relative_to(r).as_posix())); print(hashlib.sha256(s.encode()).hexdigest())'
```

| claim | command | evidence path it reads | in repository? |
| --- | --- | --- | --- |
| evidence-gated workflow and machine-local boundary | `sed -n '1,120p' src/hephaestus/scope.py` | `src/hephaestus/scope.py` | yes |
| v1 in-run floor | exact first `python -c` above | `tests/data/archived_v1_bundle/methodology.json` | yes |
| v1 offline gate verdict | exact v1 `hephaestus gate` above | `tests/data/archived_v1_bundle/` | yes |
| v1 parity split | stdlib parity command in section 2 | `tests/data/archived_v1_bundle/timings.json` | yes |
| byte-for-byte same configuration digest | exact `cmp` and `shasum` command above | `tests/data/archived_v1_bundle/config.json`; `artifacts/methodology-v2/task-4/planted-demo-20260831T020425.145134Z/runs/20260831T020439.343896Z-mlp_stack-f0946b9c24f2/config.json` | v1 yes; v2 no |
| v2 in-run floor | exact second `python -c` above | `artifacts/methodology-v2/task-4/planted-demo-20260831T020425.145134Z/runs/20260831T020439.343896Z-mlp_stack-f0946b9c24f2/methodology.json` | no |
| v2 gate verdict | exact v2 `hephaestus gate` above | `artifacts/methodology-v2/task-4/planted-demo-20260831T020425.145134Z/runs/20260831T020439.343896Z-mlp_stack-f0946b9c24f2/` | no |
| stored-evidence determinism contract | `.venv/bin/hephaestus gate tests/data/archived_v1_bundle` twice; compare emitted JSON bytes | `tests/data/archived_v1_bundle/` | yes |
| fresh measured-run capture | exact `json.tool` command above | `artifacts/readme-capture/runs/20260903T235457.173897Z-mlp_stack-0bb4c54b98c6/verdict.json` | no |
| captured accepted agent proposal | exact `json.tool` command for the MLP transcript above | `artifacts/readme-capture/agent-search-20260903T235527.445440Z-mlp_stack/transcript.json` | no |
| captured refused agent proposal | exact `json.tool` command for the transformer transcript above | `artifacts/readme-capture/agent-search-20260903T235547.853996Z-transformer_block/transcript.json` | no |
| closed-catalog, inert-policy, tool-layer authority boundary | four exact `sed` commands for `agent.py`, `catalog.py`, `search.py`, and `catalog_runtime.py` above | `src/hephaestus/agent.py`; `src/hephaestus/catalog.py`; `src/hephaestus/search.py`; `src/hephaestus/catalog_runtime.py` | yes |
| fresh planted-demo capture | exact `json.tool` command for the fresh demo above | `artifacts/readme-capture/planted-demo-20260903T235604.158530Z/demo.json` | no |
| archived probe-1 A/A payload | exact `json.tool` command using `$PROBE_ARCHIVE` above | probe archive, composite `bf32ad1d531295a3872fc2798bccef6f2b80376def2c564529e0a91f9837dd4c` | no |
| probe archive composite integrity | exact composite-hash command using `PROBE_ARCHIVE` above | probe archive, composite `bf32ad1d531295a3872fc2798bccef6f2b80376def2c564529e0a91f9837dd4c` | no |
| preserved planted-regression table | exact `json.tool` command using `$TASK4_DEMO_ARCHIVE` above | complete planted-demo archive, composite `d65cfb377025125dec7c8dca50c80d9e807119fb9816094ff5c886f34d6bb2ad` | no |
| complete planted-demo archive composite integrity | exact composite-hash command using `TASK4_DEMO_ARCHIVE` above | complete planted-demo archive, composite `d65cfb377025125dec7c8dca50c80d9e807119fb9816094ff5c886f34d6bb2ad` | no |
| historical chronology, terminal round-five failure, and capture-target correction | summarized in this README; the operator decision ledger is intentionally not published | operator-only decision ledger | no |
| A/A meta-test status and series marker | exact `sed` commands for this README and the pre-registration above | `README.md`; `docs/calibration/preregistration-1-cross-run-calibration.md` | yes |
| calibration trace procedure | exact read-only report and analysis-file commands above | `docs/calibration/drift-trace-report.md`; `docs/calibration/graph_break_bait-20260902T011653.483792Z.analysis.json`; `docs/calibration/mlp_stack-20260902T012826.821445Z.analysis.json` | yes |
| supplemental baseline-only record | exact read-only report command above | `docs/calibration/drift-trace-report.md` | yes |
| `graph_break_bait` calibration trace | exact read-only report and graph analysis-file commands above | `docs/calibration/drift-trace-report.md`; `docs/calibration/graph_break_bait-20260902T011653.483792Z.analysis.json` | yes |
| `mlp_stack` calibration trace | exact read-only report and MLP analysis-file commands above | `docs/calibration/drift-trace-report.md`; `docs/calibration/mlp_stack-20260902T012826.821445Z.analysis.json` | yes |
| calibration archive composite integrity | exact composite-hash command using `CALIBRATION_ARCHIVE` above | local calibration archive composite `1f3be6c831be3bb289f4a70792f7d90074d08c3c37f021b6dc515d47bf4ab173` | no |
| series-1 baseline, checklist and host limits | exact `json.tool` command for the committed baseline header above | `docs/series-1/graph_break_bait-20260904T012056.749366Z.jsonl` | yes |
| series-1 parent verdicts, floors and directional bounds | exact `for verdict` command above | `docs/series-1/aa-test-*/verdict.json` | yes |
| series-1 raw before/after loads and host-state classifications | exact `json.tool --json-lines` command using `SERIES1_ARCHIVE` above | series-1 archive, composite `22059f4ae33b6b6fc9c2dee12543bf1e3d7fa0cc66338437516a7ab785d85ba2` | no |
| series-1 archive composite integrity | exact composite-hash command using `SERIES1_ARCHIVE` above | local series-1 archive composite `22059f4ae33b6b6fc9c2dee12543bf1e3d7fa0cc66338437516a7ab785d85ba2` | no |
| stated non-claims | review this README against the cited evidence | this README | yes |

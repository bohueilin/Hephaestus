# Cross-run calibration publication

## Permanent probe-2 record

Probe 2 remains permanent `FAIL / methodology.noise_floor`: floor `0.16022381230908075`, A/B lower bound `0.8087130460496199`, and B/A lower bound `1.037008611201484`; it was not rerun or reclassified.

## Calibration (analysis-grade, not evidence)

These traces are un-manifested, un-gated, and produce no verdict.

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

### Checklist and disposition

The stored checklist is: AC `true`; lid `false`; display `false`; no-other-user-applications `false`; no-other-Codex/agent `false`; no-test-suite-or-browser `false`; clean-git-status `true`.

The MLP disturbed trace is published but cannot enter a future quiet §6 pool. The graph trace is the only quiet trace from this authorization; §6 still requires every quiet trace from at least two calendar days and a separate authorization, so no new limit is adopted. `0.10` remains unchanged, the §3 ten-attempt series was not run, Task 5 remains blocked, no re-probe or push occurred, and no §9 STOP condition fired.

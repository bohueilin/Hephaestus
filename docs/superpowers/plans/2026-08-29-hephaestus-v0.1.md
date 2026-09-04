# Hephaestus v0.1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the five-command Hephaestus v0.1 CPU lab with real `torch.compile` measurement, deterministic evidence gates, planted regressions, A/A checks, a closed scripted-agent seam, and hash-manifested offline-regatable bundles.

**Architecture:** Workloads and compile configurations are typed inputs to one measurement harness. The harness owns all execution and uses version-pinned bindings in `torchbind.py` to collect real Dynamo evidence; it writes canonical evidence bundles that a side-effect-free gate evaluates from stored raw data. CLI orchestration, planted regressions, and the scripted agent call the same harness/results boundary and never bypass the gate.

**Tech Stack:** CPython 3.14.7, PyTorch 2.13.0 on Apple Silicon CPU, PyYAML, pytest, and Ruff.

**Spec:** `DESIGN.md` (normative), with `CODEX_PROMPT.md` as the frozen implementation contract.

## Global Constraints

- CPU is the contract; make no GPU, CUDA, Triton, kernel, or machine-general performance claim.
- Runtime makes no network calls and never fakes a timing, graph break, recompile, trigger, or reason.
- Pin all PyTorch/Dynamo bindings in `src/hephaestus/torchbind.py`; if any required behavior fails its real-build capability probe, STOP and report.
- Each production behavior starts with a failing test that is observed failing for the intended reason, then receives the minimum implementation needed to pass.
- Gate criteria exist before formal runs; accuracy or methodology violations produce `INVALID_EVIDENCE` without meaningful performance findings.
- Every bundle contains canonical raw evidence and a recursive SHA-256 manifest; the manifest covers every non-manifest file and rejects missing, changed, and unexpected files.
- The manifest cannot hash itself; this unavoidable self-reference ruling must be disclosed in `HANDBACK.md`.
- A live run and `hephaestus gate <bundle>` call the same pure evaluation function; verdict JSON contains no evaluation timestamp and is byte-deterministic over stored evidence.
- The agent may select only catalogued configurations and consumes results only through a read-only results API.
- Do not push, publish, rent hardware, patch upstream, add an LLM, or read material from other local projects.
- No committed file may contain a username, home-directory path, hostname, job framing, or another project's private material.

---

### Task 1: Project scaffold, locked environment, and real-build STOP probe

**Files:**
- Create: `.gitignore`
- Create: `.python-version`
- Create: `pyproject.toml`
- Create: `requirements.lock`
- Create: `src/hephaestus/__init__.py`
- Create: `tests/test_package.py`
- Commit unchanged: `DESIGN.md`
- Commit unchanged: `CODEX_PROMPT.md`
- Commit: this implementation plan

**Interfaces:**
- Produces: importable `hephaestus` package with `__version__ == "0.1.0"`.
- Produces: `.venv/bin/python` at CPython 3.14.7 with exactly resolved dependencies, including `torch==2.13.0`.
- Produces: verified access to real Inductor CPU compilation and the exact Dynamo evidence symbols used by Task 3.

- [ ] **Step 1: Initialize the repository on an implementation branch**

Run `git init -b codex/hephaestus-v0.1`. Do not create a separate worktree because the user fixed this new project's workspace to the repository root and no repository exists yet.

- [ ] **Step 2: Write the first failing package test**

```python
def test_package_exposes_version() -> None:
    import hephaestus

    assert hephaestus.__version__ == "0.1.0"
```

Run `python3 -m pytest tests/test_package.py -q` (or the temporary bootstrap pytest runner if system pytest is unavailable). Expected: failure because the package does not exist.

- [ ] **Step 3: Add the minimal package and project configuration**

Use a `src/` layout, an entry point `hephaestus = "hephaestus.cli:main"`, `requires-python = ">=3.14,<3.15"`, runtime dependencies `torch==2.13.0` and `PyYAML>=6,<7`, and dev dependencies for pytest and Ruff. Ignore `.venv/`, caches, build outputs, `artifacts/`, and `.superpowers/`.

```python
"""Hephaestus: an evidence-gated torch.compile optimization lab."""

__version__ = "0.1.0"
```

- [ ] **Step 4: Create and lock the dedicated environment**

Run `uv python install 3.14.7`, `uv venv --python 3.14.7 .venv`, install the editable project with dev dependencies, and generate `requirements.lock` with every transitive version pinned. Confirm `.venv/bin/python --version` is exactly 3.14.7 and `.venv/bin/python -c 'import torch; print(torch.__version__)'` is exactly 2.13.0.

- [ ] **Step 5: Run the real-build capability probe**

In `.venv`, compile and execute a small CPU tensor function with backend `inductor`. Verify callable access to `torch._dynamo.reset`, `torch._dynamo.utils.clear_compilation_metrics`, `torch._dynamo.utils.get_compilation_metrics`, `torch._dynamo.utils.graph_break_reasons`, `torch._logging.set_logs`, `torch.compiler.list_backends`, `torch._inductor.list_options`, and `torch._inductor.list_mode_options`. Exercise one `.item()` graph break and one shape-guard recompile; verify a non-empty verbatim reason/trigger is observable for each. If any check fails, STOP and report rather than continuing.

- [ ] **Step 6: Verify GREEN and commit the scaffold**

Run `.venv/bin/pytest tests/test_package.py -q` and `.venv/bin/ruff check .`. Expected: both exit 0. Commit all scaffold files, both supplied documents unchanged, the lock, and the plan as `Scaffold Hephaestus project`.

---

### Task 2: Canonical bundles and pure deterministic gate

**Files:**
- Create: `gates/default.yaml`
- Create: `src/hephaestus/bundle.py`
- Create: `src/hephaestus/gate.py`
- Create: `tests/test_bundle.py`
- Create: `tests/test_gate.py`

**Interfaces:**
- Produces: `canonical_json_bytes(value: object) -> bytes`.
- Produces: `write_json(path: Path, value: object) -> None` and `write_manifest(bundle_dir: Path) -> dict[str, object]`.
- Produces: `verify_manifest(bundle_dir: Path) -> IntegrityResult`, where `IntegrityResult.valid` and `.mismatches` name changed, missing, or unexpected relative paths.
- Produces: `evaluate_bundle(bundle_dir: Path) -> dict[str, object]`, a deterministic pure verdict over stored evidence.
- Consumes: bundle-local `gate_criteria.yaml`, copied from `gates/default.yaml` before the run; this makes later offline re-gating independent of mutable repository state.

- [ ] **Step 1: Write failing manifest behavior tests**

Cover canonical sorted JSON, recursive hashing of every non-manifest file, rejection of an altered file, rejection of a missing file, and rejection of an unexpected file. Hand-derive expected SHA-256 strings from fixed byte fixtures instead of calling production helpers in expectations.

- [ ] **Step 2: Run manifest tests RED**

Run `.venv/bin/pytest tests/test_bundle.py -q`. Expected: import failure for `hephaestus.bundle`.

- [ ] **Step 3: Implement canonical bundle integrity**

`manifest.json` uses schema version 1 and a sorted `files` mapping of POSIX relative path to lowercase SHA-256. Exclude only `manifest.json`; reject symlinks and any non-manifest regular file absent from the mapping. Do not add signing or claim authenticity.

- [ ] **Step 4: Run manifest tests GREEN**

Run `.venv/bin/pytest tests/test_bundle.py -q`. Expected: pass.

- [ ] **Step 5: Write failing gate behavior tests**

Use literal stored timing arrays and criteria. Cover:

```text
accuracy mismatch -> INVALID_EVIDENCE driven by accuracy.tolerance, with no perf findings
methodology violation -> INVALID_EVIDENCE driven by methodology.valid
manifest mismatch -> INVALID_EVIDENCE driven by evidence.integrity and named path
speedup lower bound below 1.10 -> NOT_PROVEN driven by perf.speedup_proven
cold compile over workload budget -> NOT_PROVEN driven by perf.compile_budget
recompile count over bound -> NOT_PROVEN driven by graph.recompile_bound
enumerated graph breaks -> CONDITIONAL driven by graph.no_breaks
all hard pass and zero breaks -> PROVEN
same stored evidence evaluated twice -> byte-identical canonical verdict
configured 10% effect not above stored A/A floor -> INVALID_EVIDENCE driven by methodology.noise_floor
```

- [ ] **Step 6: Run gate tests RED**

Run `.venv/bin/pytest tests/test_gate.py -q`. Expected: import failure or missing evaluator behavior.

- [ ] **Step 7: Implement the pure gate**

The criteria file declares `minimum_speedup: 1.10`, deterministic bootstrap settings (`samples: 2000`, `confidence: 0.95`, `seed: 0`), compile budgets in seconds (`mlp_stack: 30.0`, `transformer_block: 60.0`, `dynamic_batch_text: 90.0`, `graph_break_bait: 30.0`), maximum recompiles (`mlp_stack: 0`, `transformer_block: 0`, `dynamic_batch_text: 2`, `graph_break_bait: 0`), and graph-break policy `conditional_if_any` with `reasons_required: true`. Recompute medians, IQRs, and the lower confidence bound from raw stored timings; never trust a stored summary. Evaluate integrity and methodology first, then accuracy, hard findings, and soft findings. Emit ordered findings and one `driving_finding`; omit performance findings when evidence is invalid.

- [ ] **Step 8: Run all Task 2 tests and commit**

Run `.venv/bin/pytest tests/test_bundle.py tests/test_gate.py -q` and `.venv/bin/ruff check .`. Commit as `Add deterministic evidence gate`.

---

### Task 3: Four workloads and pinned PyTorch/Dynamo bindings

**Files:**
- Create: `src/hephaestus/workloads/__init__.py`
- Create: `src/hephaestus/workloads/base.py`
- Create: `src/hephaestus/workloads/mlp_stack.py`
- Create: `src/hephaestus/workloads/transformer_block.py`
- Create: `src/hephaestus/workloads/dynamic_batch_text.py`
- Create: `src/hephaestus/workloads/graph_break_bait.py`
- Create: `src/hephaestus/torchbind.py`
- Create: `src/hephaestus/torch_capabilities.json`
- Create: `tests/test_workloads.py`
- Create: `tests/test_torchbind.py`

**Interfaces:**
- Produces: immutable `WorkloadSpec(name, seed, dtype, atol, rtol, compile_budget_seconds, max_recompiles, make_module, input_cases)` and `get_workload(name: str) -> WorkloadSpec`.
- Produces: `CompileRequest(backend, mode, dynamic, fullgraph, options, disable)` consumed by `compile_module(module, request)`.
- Produces: `reset_compiler_state()`, `capture_compiler_evidence()` context manager, and `read_compiler_evidence() -> dict` containing raw graph-break reasons/stacks, compilation metrics, recompile reasons, and verbose log records.
- Produces: `environment_snapshot() -> dict` without hostname, username, or home path.

- [ ] **Step 1: Write failing workload contract tests**

Assert the registry has exactly the four specified names; repeated construction under each pinned seed yields equal eager outputs; each workload returns CPU tensors; dynamic text cases include multiple batch/sequence shapes; graph-break bait contains a real data-dependent `.item()` branch; and dtype-aware tolerances accept eager self-comparison.

- [ ] **Step 2: Run workload tests RED, implement minimal workloads, and run GREEN**

Run `.venv/bin/pytest tests/test_workloads.py -q` before and after implementation. Use modest fixed CPU shapes, `eval()` modules, and `torch.inference_mode()`; no downloads or external model packages.

- [ ] **Step 3: Write failing binding tests against real PyTorch**

Mark them `slow`. Verify reset makes evidence per-run, `.item()` yields at least one stored non-empty reason, varying shapes under `dynamic=False` yields at least one stored non-empty recompile trigger, environment fields include Torch/Python/OS/chip but exclude identifying fields, and capabilities match the committed PyTorch 2.13.0 discovery snapshot.

- [ ] **Step 4: Run binding tests RED**

Run `.venv/bin/pytest tests/test_torchbind.py -q`. Expected: missing binding implementation.

- [ ] **Step 5: Implement and comment every binding**

Use the real 2.13.0 APIs proven in Task 1: `torch._dynamo.reset()`, explicit compilation-metric clearing, `graph_break_reasons`, compilation metrics with `recompile_reason`, and temporary verbose Dynamo logging. Preserve upstream reason/trigger strings exactly and serialize stacks separately. Discover backends and Inductor options, pin the snapshot, and fail fast on a Torch version or capability mismatch.

- [ ] **Step 6: Run Task 3 tests and commit**

Run `.venv/bin/pytest tests/test_workloads.py tests/test_torchbind.py -q` and `.venv/bin/ruff check .`. Commit as `Bind real compiler evidence`.

---

### Task 3b amendment: Calibrated Apple-CPU workload evidence

The original structurally correct micro-workloads could not satisfy the frozen clean-control
contract on this CPU: the clean three-linear MLP produced a speedup lower bound of only
0.299–0.343. Read-only Apple Silicon arm64 calibration with Torch 2.13.0 therefore pinned
the following conventional workload sizes before catalog or formal evidence work:

- `mlp_stack`: 16 residual FFN blocks, each
  `Linear(32,64) -> GELU -> Linear(64,32) -> residual add -> LayerNorm(32)`, with one
  `(48,32)` float32 input.
- `transformer_block`: the unchanged `TransformerEncoderLayer` with one
  `(16,128,32)` float32 input.
- `dynamic_batch_text`: 16 residual text FFN blocks with ordered cases
  `(2,24,32)`, `(1,48,32)`, `(3,16,32)`, and `(4,12,32)`.
- `graph_break_bait`: `Linear(128,128)`, the retained data-dependent `Tensor.item`
  condition, 12 branch-local refinements, and one positive `(128,128)` float32 case.

The clean, transformer, and static-dynamic requests use Inductor with `mode=None`,
`dynamic=False`, `fullgraph=True`, no options, and compilation enabled. The graph-bait
request differs only by `fullgraph=False`. Cache-disabled calibration established this
machine-local GO evidence without changing the frozen criteria:

| Workload | Runs | Cold compile (s) | Speedup lower bound | A/A floor | Compiler outcome |
|---|---:|---:|---:|---:|---|
| MLP | 7 | 4.375–4.745 | 3.655–3.924 | 0.0573–0.0769 | accurate; 0 breaks; 0 recompiles; `PROVEN` |
| Transformer | 10 | 0.549–0.600 | 0.885–0.937 | 0.0176–0.0667 | accurate; stable; `NOT_PROVEN / perf.speedup_proven` |
| Dynamic text | 7 | 4.451–5.192 | 3.393–3.523 | 0.0560–0.0914 | accurate; 0 breaks; exactly 3 recompiles |
| Graph bait | 3 | 3.545–3.673 | 1.333–1.410 | 0.0211–0.0508 | accurate; exactly 1 reasoned break; 0 recompiles |

The 1.10 speed threshold, 2,000 bootstrap samples, compile budgets, accuracy tolerances,
recompile bounds, and graph-break policy remain frozen. If the named outcomes fail after
the source amendment, stop instead of tuning criteria or sample counts.

---

### Task 4: Measurement engine and finalized evidence bundles

**Files:**
- Create: `src/hephaestus/measure.py`
- Create: `tests/test_measure.py`
- Create: `tests/test_run_integration.py`
- Modify: `src/hephaestus/bundle.py`
- Modify: `src/hephaestus/gate.py`
- Modify: `tests/test_gate.py`

**Interfaces:**
- Produces: `RunSettings(warmup_runs=5, repeats=31, bootstrap_samples=2000, inter_run_spacing_seconds=0.0)` with frozen bootstrap seed 0.
- Produces: `measure(workload: WorkloadSpec, request: CompileRequest, settings: RunSettings) -> RawRunEvidence`.
- Produces: `run_to_bundle(workload_name: str, request: CompileRequest, output_root: Path, criteria_path: Path, settings: RunSettings) -> RunResult`.
- `RunResult` exposes only bundle path, verdict, and read-only summary; it does not expose mutable gate or timing internals to the agent.

- [ ] **Step 1: Write failing deterministic methodology tests**

With an injected monotonic clock and timestamp source, assert warmups are absent from raw steady-state arrays, eager/compiled iteration times and UTC timestamps have equal declared lengths, cold first-call compile time is separate, median/IQR are not bare means, pinned inputs feed both eager and compiled accuracy comparison, and an out-of-tolerance output marks accuracy false. Compute the untimed eager reference, then the cold compiled call, and warm both paths. Measure eager, compiled A, and compiled B in one loop on the same pinned input case, alternating `eager,A,B` and `B,A,eager` by iteration. Preserve the two compiled series as `aa_baseline_seconds` and `aa_candidate_seconds`, bind `compiled_seconds` exactly to compiled A, and never call `sleep(0)`.

- [ ] **Step 2: Run unit tests RED, implement minimal measurement, and run GREEN**

Run `.venv/bin/pytest tests/test_measure.py -q` before and after implementation. Preserve seconds as finite non-negative floats and ISO-8601 UTC timestamps for every iteration.

- [ ] **Step 3: Write failing real-compile integration tests**

Assert a tiny `mlp_stack` run creates `env.json`, `workload.digest`, `config.json`, `timings.json`, `dynamo_report.json`, `accuracy.json`, `methodology.json`, `gate_criteria.yaml`, `verdict.json`, and `manifest.json`; `workload.digest` contains both the workload name and the SHA-256 of its source file; every timing series has per-iteration UTC timestamps; manifest verification passes; offline evaluation equals stored verdict; and two offline evaluations serialize byte-identically. Add gate regressions proving it independently recomputes signed paired effects `(A-B)/((A+B)/2)`, 2,000 seed-0 bootstrap resamples of the paired-effect median, their absolute values, and the linearly interpolated p95 noise floor, and invalidates any stored derivation or methodology metadata that disagrees. Prove a single paired outlier remains below the 10% floor while a systematic effect exceeds it, and run a real default Apple-arm64 MLP regression.

- [ ] **Step 4: Run integration RED, implement finalization flow, and run GREEN**

The flow is: reset compiler state; compute the untimed eager reference; time the cold compiled call; discard five warmups of both paths; collect 31 interleaved eager/A/B repetitions with zero spacing and alternating order; preserve raw timings/timestamps and real Dynamo evidence; compute and store raw signed A/A effects, the complete deterministic bootstrap distribution, and its p95 floor; write raw evidence and copied criteria; write a provisional manifest; call the gate (which independently recomputes and checks every A/A derivation); write canonical `verdict.json`; rewrite the manifest to include every evidence file except itself; verify the finalized bundle; call the same gate again and require equality with the stored verdict. Run `.venv/bin/pytest tests/test_run_integration.py -q` before and after.

- [ ] **Step 5: Run Task 4 tests and commit**

Run `.venv/bin/pytest tests/test_measure.py tests/test_run_integration.py -q` and `.venv/bin/ruff check .`. Commit as `Measure and preserve compiler runs`.

---

### Task 5: Closed catalog, planted defects, and scripted agent seam

**Files:**
- Create: `src/hephaestus/catalog.py`
- Create: `src/hephaestus/agent.py`
- Create: `tests/test_catalog.py`
- Create: `tests/test_agent.py`
- Create: `tests/test_planted_regressions.py`

**Interfaces:**
- Produces typed enums for mode (`default`, `reduce-overhead`, backend max-autotune equivalent), dynamic strategy (`false`, `true`, `bucketed`), and `fullgraph`; rejects unknown values/options.
- Produces: immutable named configurations including clean candidates and exactly three distinct planted defects.
- Produces: `ResultsAPI.run(proposal) -> ReadOnlyRunResult`; the agent receives no harness, gate, criteria, filesystem, or mutable evidence object.
- Produces: `ScriptedOptimizer.optimize(workload_name) -> AgentTranscript`, with ordered proposal, one-line rationale, bundle-relative result path, verdict, and driving finding per step.

- [ ] **Step 1: Write failing catalog authority tests**

Assert unknown modes/options are rejected; bucket boundaries are explicit and increasing; all installed-option references exist in the pinned capability snapshot; and the three planted configurations are distinct: eager fallback, static-shape recompile storm, and graph-break exposure.

- [ ] **Step 2: Run catalog tests RED, implement the closed schema, and run GREEN**

Run `.venv/bin/pytest tests/test_catalog.py -q` before and after implementation.

- [ ] **Step 3: Write failing agent-boundary tests**

Use a fake `ResultsAPI` that returns immutable summaries. Assert staged catalog order, early stopping only after a PROVEN result, one rationale per proposal, rejected out-of-catalog proposals, and transcript serialization without raw timing mutation or gate access.

- [ ] **Step 4: Run agent tests RED, implement the scripted agent, and run GREEN**

Run `.venv/bin/pytest tests/test_agent.py -q` before and after implementation.

- [ ] **Step 5: Write and run real planted-regression tests**

Run each real planted config through the harness with test-sized repeats. Require eager fallback to be caught by `perf.speedup_proven`, static dynamic-shape storm by `graph.recompile_bound`, graph-break exposure by `graph.no_breaks`, and the clean control to be `PROVEN`. If fewer than three defects are genuinely observable on installed Torch, STOP and report.

- [ ] **Step 6: Run Task 5 tests and commit**

Run `.venv/bin/pytest tests/test_catalog.py tests/test_agent.py tests/test_planted_regressions.py -q` and `.venv/bin/ruff check .`. Commit as `Add gated optimization catalog`.

---

### Task 6: Five CLI commands

**Files:**
- Create: `src/hephaestus/cli.py`
- Create: `tests/test_cli.py`
- Modify: `pyproject.toml` only if the Task 1 entry point needs correction

**Interfaces:**
- Produces these exact command surfaces:
  - `hephaestus run <workload> --config <compile-config>`
  - `hephaestus gate <bundle>`
  - `hephaestus agent optimize <workload>`
  - `hephaestus demo-planted-regressions`
  - `hephaestus aa-test <workload>`
- All command output starts with or visibly includes the machine-local CPU boundary and exits non-zero on invalid evidence, failed demo criteria, or a self-failing A/A methodology check.

- [ ] **Step 1: Write failing subprocess-level CLI tests**

Assert help exposes exactly the five surfaces; invalid workload/config names exit 2; `gate` names an integrity mismatch and exits non-zero; successful command JSON/table output names the verdict and driving finding; and every evidence-producing surface prints the machine-local boundary.

- [ ] **Step 2: Run CLI tests RED**

Run `.venv/bin/pytest tests/test_cli.py -q`. Expected: missing CLI module or command behavior.

- [ ] **Step 3: Implement thin CLI orchestration**

Use `argparse`; keep gate logic, measurement, agent search, demo evaluation, and A/A math out of `cli.py`. Default artifacts to `artifacts/` with collision-resistant UTC-plus-digest directory names. `aa-test` measures the same named config twice, stores both child bundle paths plus the raw null-effect distribution and noise floor in an A/A evidence directory, manifests it, and fails if `minimum_speedup - 1.0` is not strictly greater than the observed floor or if it falsely reports a meaningful difference.

- [ ] **Step 4: Run CLI tests GREEN and commit**

Run `.venv/bin/pytest tests/test_cli.py -q`, then the full suite and Ruff. Commit as `Expose Hephaestus CLI workflows`.

---

### Task 7: Formal runs, README evidence, and hand-back

**Files:**
- Create: `README.md`
- Create: `HANDBACK.md`
- Modify implementation/tests only through a new failing regression test if a formal run exposes a bug

**Interfaces:**
- Produces owner-reviewable real outputs and an honest machine-local boundary.
- Produces a clean repository with every commit independently test-green.

- [ ] **Step 1: Freeze formal criteria before formal evidence**

Record the committed criteria digest. Any earlier workload-size engineering runs are pilots, not evidence. Do not loosen criteria based on formal results; a formal failure remains a failure unless a test proves an implementation bug.

- [ ] **Step 2: Execute all five CLI surfaces**

Run one normal `run`, offline `gate` twice on the same bundle, `agent optimize`, `demo-planted-regressions`, and `aa-test`. Confirm offline verdict bytes match exactly while documenting that fresh raw timings do not.

- [ ] **Step 3: Execute the required A/A series**

Run `aa-test` at least three consecutive times as required by `CODEX_PROMPT.md`; continue to ten consecutive passes to satisfy `DESIGN.md` success criterion 2. Publish every observed noise floor and retain bundle paths.

- [ ] **Step 4: Capture the real planted-regression table**

Require three rows with distinct expected/actual driving criteria and one passing clean-control row. Do not invent or normalize output.

- [ ] **Step 5: Write README from captured output**

Every factual behavior or measurement claim cites its reproducing command immediately. Include install, methodology, architecture boundary, exact five-command usage, real trimmed output, evidence schema, tamper demo, determinism statement, and all §4 exclusions. Strip usernames and absolute home paths from examples.

- [ ] **Step 6: Write HANDBACK**

List work per commit, RED→GREEN commands/results, real numbers with the machine-local boundary, STOP checks/rulings and their costs, all out-of-scope items not done, manifest self-reference limitation, private/beta Torch binding risk, and surprises.

- [ ] **Step 7: Fresh final verification**

Run `.venv/bin/pytest -q`, `.venv/bin/ruff check .`, all five CLI commands, manifest tamper rejection, byte-identical double re-gating, `git diff --check`, and a secret/personal-path scan. Verify every commit with a clean checkout or scripted per-commit test replay if feasible. Commit docs as `Document reproducible local evidence`.

- [ ] **Step 8: Review and hand back without push**

Run a whole-branch code/spec review, address validated findings through regression tests, re-run the full final gate, and leave `git status --short` empty. Do not push or merge.

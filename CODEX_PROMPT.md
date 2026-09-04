# Implementation prompt — Hephaestus v0.1

You are building a NEW project from scratch. The normative design is
`DESIGN.md` in the repository root — READ IT FIRST AND IN FULL.
Follow this prompt exactly; where it says STOP, stop and report rather than improvising.
This is a green-field repo: you have freedom in API binding and internal structure, but
the BEHAVIORS, boundaries, and gates below are frozen.

## 1. Mission

Deliver Hephaestus v0.1 per DESIGN.md §3: the five CLI commands, four workloads, the
declarative gate, planted regressions + A/A null tests, the scripted agent behind the
closed catalog, and hash-manifested evidence bundles — with a full test suite and a
README whose every claim cites the command that reproduces it.

Out of scope (DESIGN.md §4 is normative): no GPU/CUDA/Triton work or claims; no LLM
agent; no network calls at runtime; no upstream patches; no benchmark claims beyond
this machine. The README and every evidence surface carry the machine-local boundary
explicitly.

## 2. Workspace and environment

- Project root: the repository root (this file and DESIGN.md
  already live here). Run `git init` if no repo exists; first commit is the scaffold.
  The repo may become PUBLIC: no personal information, no private-context framing, and
  no references to other local projects' private material. DESIGN.md remains the
  normative specification.
- Python: create a dedicated venv INSIDE the project (`.venv/`, gitignored) with a
  current stable CPython and the latest stable `torch` (CPU build is fine). Pin the
  exact resolved versions into `requirements.lock` (committed). Never touch any conda
  env belonging to other projects.
- Everything runs on Apple Silicon CPU (MPS only if it works without special-casing;
  CPU is the contract). If `torch.compile` is unavailable or broken in the installed
  build, STOP and report — do not fake timings.
- Commit style: small, reviewable commits with imperative subjects; TDD discipline —
  behavior tests RED before implementation wherever testable.

## 3. Frozen behaviors (bind APIs yourself, at build time)

The design pins BEHAVIORS, not API names. Discover and pin the concrete torch/Dynamo
APIs in one place (`src/hephaestus/torchbind.py`) with a comment per binding stating
what behavior it provides; if a needed behavior has no working API in the installed
torch (e.g. graph-break introspection), STOP and report the gap rather than shimming a
fake.

1. **Measurement** (DESIGN §3.2): warmup discarded; N steady-state repeats; median +
   IQR; cold compile time measured separately; per-iteration raw timings preserved;
   environment snapshot (torch version/build, Python, OS, chip) in every bundle;
   wall-clock timestamps per iteration (thermal honesty).
2. **Dynamo accounting**: per-run graph-break count WITH reasons and recompile count
   WITH triggers, recorded verbatim into `dynamo_report.json`. Reset compiler state
   between runs so counts are per-run, not cumulative.
3. **Accuracy contract**: each workload declares dtype-aware tolerances; compiled vs
   eager outputs compared on pinned inputs; violation ⇒ the bundle is
   INVALID_EVIDENCE (verdict names it; no perf findings are reported as if meaningful).
4. **The gate is a pure function of stored evidence**: `hephaestus gate <bundle>`
   recomputes the verdict offline from raw files; a live run calls exactly the same
   function on the bundle it just wrote. Criteria come from `gates/default.yaml`
   (DESIGN §3.3 table is the starting content); the verdict names its driving finding.
5. **A/A noise floor** (DESIGN §3.2/§6.2): `aa-test` runs one config twice, derives the
   null-effect distribution, stores it, and FAILS ITSELF if the gate's configured
   speedup threshold does not clear the measured floor — the methodology polices its
   own gate.
6. **Planted regressions** (DESIGN §1.1/§6.1): at least three deliberately bad configs
   (e.g. force-eager-fallback disguised as compiled; a pathological recompile-storm
   dynamic config; a graph-break-maximizing flag set — choose what the installed torch
   actually allows, and STOP if fewer than three genuinely distinct planted defects are
   expressible). `demo-planted-regressions` runs each plus a clean control; each defect
   must be caught by its own named criterion; the control must pass. The demo shows that
   the gate is able to fail.
7. **Agent seam** (DESIGN §3.4): closed catalog as a typed enum/config schema; the
   scripted agent iterates the catalog (simple strategy is fine — e.g. staged sweep
   with early stopping), records proposal + one-line rationale per step, and NEVER
   reads or writes gate code, criteria, or measured data except through the harness's
   read-only results API. The search transcript is written into the bundle directory
   tree as first-class evidence.
8. **Evidence bundles** (DESIGN §3.5): one directory per run; manifest with SHA-256 of
   every file; `gate` verifies manifest integrity before ruling; a tampered file ⇒
   INVALID_EVIDENCE with the mismatch named.

## 4. Suggested structure (adjust freely; the tests and behaviors may not change)

```
src/hephaestus/
  workloads/{mlp_stack,transformer_block,dynamic_batch_text,graph_break_bait}.py
  measure.py        # methodology engine
  torchbind.py      # all torch/Dynamo API bindings, one place
  gate.py           # pure verdict function + YAML criteria loader
  bundle.py         # evidence writing + manifest + integrity check
  catalog.py        # closed action catalog (+ planted regressions)
  agent.py          # scripted agent
  cli.py            # the five commands (argparse or typer)
gates/default.yaml
tests/              # unit + integration; a `slow` marker for real-compile tests
README.md
```

## 5. Gates before you stop

- Full test suite green; `ruff` (or the linter you configure) clean.
- All five CLI commands run end to end on this machine; capture their real output into
  the README's examples (real output, trimmed, never invented).
- `demo-planted-regressions`: every planted defect caught by its named criterion +
  clean control passes — paste the real table into the README.
- `aa-test`: 3 consecutive runs, zero false flags, noise floor published.
- Determinism: the same config gated twice from stored evidence yields byte-identical
  verdicts; note (honestly) that raw TIMINGS are not byte-reproducible — the gate's
  determinism is over stored evidence, and fresh runs may differ within the noise floor.
- `git status` clean; every commit's tests pass at that commit.

## 6. Hand-back

Write `HANDBACK.md` (untracked or committed, your call): what landed per commit,
RED→GREEN evidence, the real measured numbers with the machine-local boundary restated,
every STOP you hit or ruling you made (with cost), what you did NOT do (each §1 item),
and surprises. Do not push; hand back for owner review.

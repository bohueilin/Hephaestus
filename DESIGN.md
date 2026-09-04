# Hephaestus — an evidence-gated lab for agent-participated `torch.compile` optimization

*Simulation of a product discipline, not a benchmark suite. Local numbers prove the
workflow, not GPU performance.*

## 1. Thesis

Performance claims in the PyTorch compiler ecosystem die from the same disease safety
claims die from: the demo that made the claim cannot be re-verified, the measurement
methodology is invisible, and nobody wrote down what "proven" was supposed to mean
before the number appeared. Hephaestus applies a simple evidence discipline — **an agent
proposes; the environment executes; a deterministic gate decides; the trace proves** —
to `torch.compile` optimization work:

> An optimization agent proposes a compile configuration. The harness compiles and
> measures under a pinned environment with declared methodology. A gate — whose criteria
> were written down *before* the run — decides PROVEN, CONDITIONAL, NOT_PROVEN, or
> INVALID_EVIDENCE. The evidence bundle preserves everything needed to re-check the
> verdict without re-running.

Three domain-independent ideas define the lab:

1. **The gate must be shown to catch planted regressions.** A performance gate that has
   never failed is indistinguishable from one that cannot fail. Hephaestus ships
   deliberately deoptimized configurations ("planted regressions") that the gate MUST
   flag, and A/A null runs (same config twice) that it MUST NOT flag — the noise floor
   is measured, not assumed.
2. **The agent's authority boundary is enforced in the tool layer.** The agent chooses
   from a closed action catalog and never touches the measurement or the gate. A
   proposal is recorded beside the verdict, never in place of it.
3. **Trust is decomposed.** Speedup, compile-time budget, graph-break count, recompile
   count, and numerical-accuracy drift are reported as separate findings — the aggregate
   verdict states which finding drove it. A single "3.2× faster" number is the failure
   mode, not the product.

## 2. Honest scope

This lab runs on Apple Silicon: `torch.compile` with the Inductor CPU backend (and MPS
where it works). That is deliberate and must be stated everywhere numbers appear:

- Local measurements prove the **workflow, methodology, and gate discipline** — they say
  nothing about NVIDIA GPU performance, Triton kernels, or CUDA graphs.
- The artifacts that generalize are the **methodology, the gate criteria, the agent
  seam, and the evidence format** — the things that would survive a port to an A100
  unchanged while every number changed.
- A cloud-GPU appendix (one rented GPU day, same harness, CUDA backend) is a later
  milestone, not v0.1.

## 3. What v0.1 does, end to end

```
hephaestus run <workload> --config <compile-config>   # one measured, gated run
hephaestus gate <bundle>                              # re-verify a stored bundle offline
hephaestus agent optimize <workload>                  # scripted agent walks the catalog
hephaestus demo-planted-regressions                   # the gate catches each planted defect
hephaestus aa-test <workload>                         # null-effect run; gate must stay quiet
```

### 3.1 Workload registry (pinned, tiny, meaningful)

Four workloads, each a single file with pinned seeds and pinned input shapes, chosen to
exercise the compiler behaviors the ecosystem actually fights about:

| Workload | Why it earns its place |
|---|---|
| `mlp_stack` | Clean fusion case — the "compile should just win" baseline |
| `transformer_block` | Attention + LayerNorm + residuals; the realistic unit of LLM work |
| `dynamic_batch_text` | Variable batch/sequence lengths — exercises dynamic shapes vs. recompiles, the #1 customer-issue category |
| `graph_break_bait` | Contains a data-dependent branch and a `.item()` call — exercises graph breaks and their cost |

Each workload declares its own numerical-tolerance contract (dtype-aware) for
eager-vs-compiled output comparison.

### 3.2 Measurement methodology (declared, versioned, boring on purpose)

- Warmup runs discarded; steady-state timing over N repeats; **median and IQR** reported,
  never bare means.
- **Compile time measured separately** from steady-state (cold compile, and warm-cache
  where applicable), because compile overhead is a first-class customer issue, not noise.
- **Graph breaks and recompiles counted** via Dynamo's own introspection (the
  implementation binds the exact APIs at build time — behaviors are pinned here, not API
  names): the run
  records break reasons and recompile triggers verbatim into the bundle.
- **A/A noise floor**: the same config measured twice; the observed delta distribution
  defines the minimum effect the gate may treat as real. The gate's speedup threshold
  must clear the measured noise floor — a criterion the A/A test itself enforces.
- Environment pinned into every bundle: torch version + build, Python, OS, chip, thermal
  caveat note (laptop thermals are real; runs record wall-clock spacing).
- Accuracy: compiled output vs eager output under the workload's tolerance contract;
  a tolerance failure is INVALID_EVIDENCE for the whole bundle, not a footnote.

### 3.3 The gate (declarative, written before the run)

`gates/default.yaml` declares the criteria:

| Finding | Kind | Default criterion |
|---|---|---|
| `perf.speedup_proven` | hard | median speedup CI lower bound ≥ 1.10× AND clears the measured A/A noise floor |
| `perf.compile_budget` | hard | cold compile time ≤ declared budget for the workload |
| `graph.no_breaks` | soft | zero graph breaks, or every break enumerated with reason |
| `graph.recompile_bound` | hard | recompile count ≤ declared bound across the dynamic-shape sweep |
| `accuracy.tolerance` | hard (gating validity, not verdict) | outputs within the workload's dtype-aware tolerance |

Verdicts: **PROVEN** (all hard pass, no soft fail), **CONDITIONAL** (hard pass, soft
findings open), **NOT_PROVEN** (any hard perf finding fails), **INVALID_EVIDENCE**
(accuracy or methodology violation — the number is not wrong, it is *meaningless*).
The verdict names the driving finding, always.

### 3.4 The agent seam

A closed action catalog (v0.1 is scripted; an LLM agent is a later milestone that
changes nothing about the boundary):

- `mode`: default / reduce-overhead / max-autotune-equivalent for the backend
- `dynamic`: False / True / bucketed (with declared bucket boundaries)
- `fullgraph`: False / True
- backend options exposed by the installed torch, discovered and pinned at build time
- workload-level: batch-size bucketing strategy for `dynamic_batch_text`

The agent proposes a config + a one-line rationale; the harness runs it; the gate rules.
The agent's search transcript (what it tried, what the gate said) is itself evidence —
the "agent-native optimization workflow" artifact. Planted-regression configs live in
the same catalog shape so the demo and the agent share one vocabulary.

### 3.5 Evidence bundle

One directory per run: `env.json`, `workload.digest`, `config.json`, `timings.json`
(raw per-iteration), `dynamo_report.json` (breaks/recompiles/reasons), `verdict.json`
(findings + driving criterion), and a manifest with SHA-256 of every file. `hephaestus
gate <bundle>` recomputes the verdict from stored raw data offline — the gate is a pure
function of evidence, never of a live run.

## 4. What v0.1 does NOT do (stated so nobody discovers it later)

- No GPU, CUDA, Triton, or kernel-level claims of any kind.
- No LLM in the loop (the scripted agent proves the seam; the LLM is milestone 3).
- No megakernel/inter-kernel claims — out of local hardware's reach; the *gate criteria
  design* for such features is a paper exercise in the strategy doc, not code.
- No upstream patches; this is a lab, not a contribution pipeline (reading upstream is
  milestone 2's exercise).
- Numbers are machine-local. The README carries this boundary explicitly.

## 5. Roadmap (Now / Next / Later — itself written as the product artifact)

- **Now (v0.1):** the five commands above; four workloads; the gate;
  planted regressions + A/A; scripted agent; evidence bundles; full test suite.
- **Next (v0.2):** the dynamic-shape lab — sweep sequence/batch distributions, chart the
  bucketing-vs-`dynamic=True` tradeoff (recompiles vs. per-shape performance vs. compile
  budget) and encode the tradeoff frontier as gate criteria; ecosystem-signals digest —
  read three real upstream issues/RFCs (vLLM, SGLang, TorchTitan) and translate each
  into a gate criterion or workload the lab could host, as a written strategy exercise.
- **Later (v0.3+):** LLM agent behind the same seam with budgets and an approval record
  for catalog changes; one rented GPU day porting the harness to CUDA (the port diff IS
  the deliverable — what changed, what didn't); a profile-reading appendix (torch
  profiler traces attached to bundles).

## 6. Success criteria for the lab itself

1. `demo-planted-regressions` catches every planted defect by its own named criterion,
   with a clean-config control (the gate can pass).
2. `aa-test` never flags — across 10 consecutive A/A runs — while the noise floor it
   measures is published in every bundle.
3. A bundle produced on one day re-gates identically a week later (`hephaestus gate` on
   stored evidence, no re-run).
4. The scripted agent finds a config the naive default beats-or-ties on at least one
   workload AND is refused by the gate on at least one plausible-looking config — both
   transcripts preserved.
5. Every README claim cites a command that reproduces it.

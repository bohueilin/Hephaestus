# Hephaestus v0.1 trust boundary

Hephaestus records schema-1 run provenance to bind the order, distinct identity, and
predecessor manifest of runs produced by one trusted orchestration. These random IDs
and manifest hashes are unsigned tool-layer provenance. They are useful for detecting
accidental aliasing, cloning, or reordering inside the supported workflow, but they are
not cryptographic identity, authentication, attestation, or protection from a local
attacker who can rewrite evidence and recompute manifests.

The v0.1 scripted agent is an inert, deterministic proposal policy. It receives only a
verdict and driving finding; trusted orchestration alone owns the compiler runtime,
filesystem paths, evidence, verification, and execution receipts. This authority split
is an application-layer control, not an operating-system security boundary.

A future hostile or LLM-driven policy must run behind an OS-enforced process or
container sandbox with explicit filesystem, process, syscall, resource, and network
limits. Its outputs must still be treated only as proposals and independently resolved,
executed, verified, and gated by the trusted environment.

"""Pinned workload registry."""

from hephaestus.workloads.base import WorkloadSpec
from hephaestus.workloads.dynamic_batch_text import SPEC as DYNAMIC_BATCH_TEXT
from hephaestus.workloads.graph_break_bait import SPEC as GRAPH_BREAK_BAIT
from hephaestus.workloads.mlp_stack import SPEC as MLP_STACK
from hephaestus.workloads.transformer_block import SPEC as TRANSFORMER_BLOCK

_WORKLOADS = {
    spec.name: spec
    for spec in (DYNAMIC_BATCH_TEXT, GRAPH_BREAK_BAIT, MLP_STACK, TRANSFORMER_BLOCK)
}
WORKLOAD_NAMES = tuple(sorted(_WORKLOADS))


def get_workload(name: str) -> WorkloadSpec:
    """Return a pinned workload by name."""

    try:
        return _WORKLOADS[name]
    except KeyError as error:
        available = ", ".join(WORKLOAD_NAMES)
        raise ValueError(f"unknown workload {name!r}; choose from: {available}") from error


__all__ = ["WORKLOAD_NAMES", "WorkloadSpec", "get_workload"]

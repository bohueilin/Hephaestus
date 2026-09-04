"""Workload with an intentional data-dependent scalar branch."""

import torch
from torch import nn

from hephaestus.workloads.base import WorkloadSpec

SEED = 4404
DTYPE = torch.float32


class GraphBreakBait(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.projection = nn.Linear(128, 128)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        y = self.projection(inputs)
        if inputs.mean().item() > 0:
            for _ in range(12):
                y = torch.tanh(y * 1.01 + 0.1)
        else:
            for _ in range(12):
                y = torch.sin(y * 1.01 - 0.1)
        return y


def make_module() -> nn.Module:
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(SEED)
        module = GraphBreakBait()
    return module.to(device="cpu", dtype=DTYPE).eval()


def input_cases() -> tuple[tuple[torch.Tensor, ...], ...]:
    positive = torch.full((128, 128), 0.5, dtype=DTYPE)
    return ((positive,),)


SPEC = WorkloadSpec(
    name="graph_break_bait",
    seed=SEED,
    dtype=DTYPE,
    atol=1e-5,
    rtol=1e-5,
    compile_budget_seconds=30.0,
    max_recompiles=0,
    make_module=make_module,
    input_cases=input_cases,
)

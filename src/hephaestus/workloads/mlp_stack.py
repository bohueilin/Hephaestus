"""Residual feed-forward stack used as the clean fusion baseline."""

import torch
from torch import nn

from hephaestus.workloads.base import WorkloadSpec

SEED = 1101
DTYPE = torch.float32


class ResidualFFNBlock(nn.Module):
    """Conventional residual feed-forward block with post-residual normalization."""

    def __init__(self) -> None:
        super().__init__()
        self.up = nn.Linear(32, 64)
        self.activation = nn.GELU()
        self.down = nn.Linear(64, 32)
        self.norm = nn.LayerNorm(32)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        hidden = self.down(self.activation(self.up(inputs)))
        return self.norm(inputs + hidden)


class ResidualFFNStack(nn.Module):
    """Sixteen identical-shape residual blocks."""

    def __init__(self) -> None:
        super().__init__()
        self.blocks = nn.Sequential(*(ResidualFFNBlock() for _ in range(16)))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.blocks(inputs)


def make_module() -> nn.Module:
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(SEED)
        module = ResidualFFNStack()
    return module.to(device="cpu", dtype=DTYPE).eval()


def input_cases() -> tuple[tuple[torch.Tensor, ...], ...]:
    generator = torch.Generator(device="cpu").manual_seed(SEED + 1)
    return ((torch.randn(48, 32, generator=generator, dtype=DTYPE),),)


SPEC = WorkloadSpec(
    name="mlp_stack",
    seed=SEED,
    dtype=DTYPE,
    atol=1e-5,
    rtol=1e-5,
    compile_budget_seconds=30.0,
    max_recompiles=0,
    make_module=make_module,
    input_cases=input_cases,
)

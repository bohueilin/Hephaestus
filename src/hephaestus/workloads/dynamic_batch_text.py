"""Text-shaped block with pinned variable batch and sequence dimensions."""

import torch
from torch import nn

from hephaestus.workloads.base import WorkloadSpec

SEED = 3303
DTYPE = torch.float32


class ResidualTextFFNBlock(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.up = nn.Linear(32, 64)
        self.activation = nn.GELU()
        self.down = nn.Linear(64, 32)
        self.norm = nn.LayerNorm(32)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        hidden = self.down(self.activation(self.up(tokens)))
        return self.norm(tokens + hidden)


class DynamicTextStack(nn.Module):
    """Sixteen residual FFN blocks accepting rank-three text tensors."""

    def __init__(self) -> None:
        super().__init__()
        self.blocks = nn.Sequential(*(ResidualTextFFNBlock() for _ in range(16)))

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.blocks(tokens)


def make_module() -> nn.Module:
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(SEED)
        module = DynamicTextStack()
    return module.to(device="cpu", dtype=DTYPE).eval()


def input_cases() -> tuple[tuple[torch.Tensor, ...], ...]:
    generator = torch.Generator(device="cpu").manual_seed(SEED + 1)
    shapes = ((2, 24, 32), (1, 48, 32), (3, 16, 32), (4, 12, 32))
    return tuple(
        (torch.randn(*shape, generator=generator, dtype=DTYPE),) for shape in shapes
    )


SPEC = WorkloadSpec(
    name="dynamic_batch_text",
    seed=SEED,
    dtype=DTYPE,
    atol=1e-5,
    rtol=1e-5,
    compile_budget_seconds=90.0,
    max_recompiles=2,
    make_module=make_module,
    input_cases=input_cases,
)

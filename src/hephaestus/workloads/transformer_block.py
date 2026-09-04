"""Small attention block representing a realistic transformer unit."""

import torch
from torch import nn

from hephaestus.workloads.base import WorkloadSpec

SEED = 2202
DTYPE = torch.float32


def make_module() -> nn.Module:
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(SEED)
        module = nn.TransformerEncoderLayer(
            d_model=32,
            nhead=4,
            dim_feedforward=64,
            dropout=0.0,
            batch_first=True,
            activation="gelu",
        )
    return module.to(device="cpu", dtype=DTYPE).eval()


def input_cases() -> tuple[tuple[torch.Tensor, ...], ...]:
    generator = torch.Generator(device="cpu").manual_seed(SEED + 1)
    return ((torch.randn(16, 128, 32, generator=generator, dtype=DTYPE),),)


SPEC = WorkloadSpec(
    name="transformer_block",
    seed=SEED,
    dtype=DTYPE,
    atol=1e-5,
    rtol=1e-5,
    compile_budget_seconds=60.0,
    max_recompiles=0,
    make_module=make_module,
    input_cases=input_cases,
)

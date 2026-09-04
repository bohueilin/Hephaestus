"""Shared workload contract."""

from collections.abc import Callable
from dataclasses import dataclass

import torch
from torch import nn

InputCase = tuple[torch.Tensor, ...]


@dataclass(frozen=True, slots=True)
class WorkloadSpec:
    """Immutable description of one reproducible CPU workload."""

    name: str
    seed: int
    dtype: torch.dtype
    atol: float
    rtol: float
    compile_budget_seconds: float
    max_recompiles: int
    make_module: Callable[[], nn.Module]
    input_cases: Callable[[], tuple[InputCase, ...]]

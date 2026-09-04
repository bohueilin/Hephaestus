"""Immutable Torch-free schema-1 action and strict-value evidence contracts."""

from __future__ import annotations

import math
from dataclasses import dataclass
from types import MappingProxyType
from typing import Final

Scalar = str | int | bool

V1_WORKLOAD_SHA256: Final = MappingProxyType(
    {
        "mlp_stack": "eeb39c122ab98b19f2248e8413322164c67a6c14bef6786a8e364a9d55b2c139",
        "transformer_block": "0649cc484d4fb4ba644cb51763cd969cf80b5cef9c21a7be94ae82d32b3a6260",
        "dynamic_batch_text": "8acdb31ea02ae7a86750665e550af0942bd5d1e8e658436fbefed4a16aeed472",
        "graph_break_bait": "0968a4452eb4d2ec0a3e1ccf9b7862a4411451336a43bcfbf1b7546ba80ebab0",
    }
)

V1_ACCURACY_TOLERANCE: Final = MappingProxyType(
    {"dtype": "torch.float32", "atol": 1e-5, "rtol": 1e-5}
)

V1_INDUCTOR_CACHE_PATCH: Final = MappingProxyType(
    {
        "fx_graph_cache": True,
        "fx_graph_remote_cache": False,
        "autotune_remote_cache": False,
        "bundled_autotune_remote_cache": False,
        "remote_gemm_autotune_cache": False,
        "force_disable_caches": False,
        "cpp_cache_precompile_headers": False,
    }
)

V1_FUNCTORCH_CACHE_PATCH: Final = MappingProxyType(
    {
        "enable_autograd_cache": True,
        "enable_remote_autograd_cache": False,
    }
)


@dataclass(frozen=True, slots=True)
class RequestedAction:
    mode: str
    dynamic: str
    fullgraph: str
    options: tuple[tuple[str, Scalar], ...] | None
    disable: bool
    bucket_policy: tuple[int, tuple[int, ...], str] | None = None

    def as_json(self) -> dict[str, object]:
        bucket = self.bucket_policy
        return {
            "mode": self.mode,
            "dynamic": self.dynamic,
            "fullgraph": self.fullgraph,
            "options": None if self.options is None else dict(self.options),
            "disable": self.disable,
            "bucket_policy": (
                None
                if bucket is None
                else {
                    "axis": bucket[0],
                    "boundaries": list(bucket[1]),
                    "overflow_rule": bucket[2],
                }
            ),
        }


@dataclass(frozen=True, slots=True)
class EffectiveAction:
    backend: str
    mode: str | None
    dynamic: bool
    fullgraph: bool
    options: tuple[tuple[str, Scalar], ...] | None
    disable: bool
    input_plan_strategy: str

    def as_json(self) -> dict[str, object]:
        return {
            "backend": self.backend,
            "mode": self.mode,
            "dynamic": self.dynamic,
            "fullgraph": self.fullgraph,
            "options": None if self.options is None else dict(self.options),
            "disable": self.disable,
            "input_plan_strategy": self.input_plan_strategy,
        }


@dataclass(frozen=True, slots=True)
class V1ActionContract:
    catalog_id: str
    role: str
    workload_name: str
    requested: RequestedAction
    effective: EffectiveAction
    expected_verdict: str | None = None
    expected_driving_finding: str | None = None

    def metadata_json(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "entry_id": self.catalog_id,
            "role": self.role,
            "workload_name": self.workload_name,
            "requested": self.requested.as_json(),
            "effective": self.effective.as_json(),
        }


@dataclass(frozen=True, slots=True)
class V1SearchCandidateContract:
    action: V1ActionContract
    rationale: str

    def catalog_entry_json(self) -> dict[str, object]:
        return {
            **self.action.metadata_json(),
            "rationale": self.rationale,
            "expected_verdict": None,
            "expected_driving_finding": None,
        }


_STATIC_FULLGRAPH_REQUESTED = RequestedAction(
    "default", "false", "true", None, False
)
_STATIC_FULLGRAPH_EFFECTIVE = EffectiveAction(
    "inductor", None, False, True, None, False, "static"
)
_GRAPH_BREAK_REQUESTED = RequestedAction(
    "default", "false", "false", None, False
)
_GRAPH_BREAK_EFFECTIVE = EffectiveAction(
    "inductor", None, False, False, None, False, "static"
)

V1_DEMO_ACTIONS: Final = (
    V1ActionContract(
        "planted-eager-fallback",
        "planted",
        "mlp_stack",
        RequestedAction("default", "false", "true", None, True),
        EffectiveAction("inductor", None, False, True, None, True, "static"),
        "NOT_PROVEN",
        "perf.speedup_proven",
    ),
    V1ActionContract(
        "planted-static-shape-storm",
        "planted",
        "dynamic_batch_text",
        _STATIC_FULLGRAPH_REQUESTED,
        _STATIC_FULLGRAPH_EFFECTIVE,
        "NOT_PROVEN",
        "graph.recompile_bound",
    ),
    V1ActionContract(
        "planted-graph-break-exposure",
        "planted",
        "graph_break_bait",
        _GRAPH_BREAK_REQUESTED,
        _GRAPH_BREAK_EFFECTIVE,
        "CONDITIONAL",
        "graph.no_breaks",
    ),
    V1ActionContract(
        "clean-control-mlp",
        "clean-control",
        "mlp_stack",
        _STATIC_FULLGRAPH_REQUESTED,
        _STATIC_FULLGRAPH_EFFECTIVE,
        "PROVEN",
        "all_criteria_passed",
    ),
)
V1_DEMO_ACTION_BY_ID: Final = MappingProxyType(
    {contract.catalog_id: contract for contract in V1_DEMO_ACTIONS}
)

V1_AA_ACTION_BY_WORKLOAD: Final = MappingProxyType(
    {
        "mlp_stack": V1ActionContract(
            "candidate-mlp-default",
            "candidate",
            "mlp_stack",
            _STATIC_FULLGRAPH_REQUESTED,
            _STATIC_FULLGRAPH_EFFECTIVE,
        ),
        "transformer_block": V1ActionContract(
            "candidate-transformer-default",
            "candidate",
            "transformer_block",
            _STATIC_FULLGRAPH_REQUESTED,
            _STATIC_FULLGRAPH_EFFECTIVE,
        ),
        "dynamic_batch_text": V1ActionContract(
            "candidate-dynamic-true",
            "candidate",
            "dynamic_batch_text",
            RequestedAction("default", "true", "true", None, False),
            EffectiveAction("inductor", None, True, True, None, False, "dynamic"),
        ),
        "graph_break_bait": V1ActionContract(
            "candidate-graph-break-visible",
            "candidate",
            "graph_break_bait",
            _GRAPH_BREAK_REQUESTED,
            _GRAPH_BREAK_EFFECTIVE,
        ),
    }
)


V1_SEARCH_CANDIDATES_BY_WORKLOAD: Final = MappingProxyType(
    {
        "mlp_stack": (
            V1SearchCandidateContract(
                V1ActionContract(
                    "candidate-mlp-default",
                    "candidate",
                    "mlp_stack",
                    _STATIC_FULLGRAPH_REQUESTED,
                    _STATIC_FULLGRAPH_EFFECTIVE,
                ),
                "Try the calibrated native default with a required full graph.",
            ),
            V1SearchCandidateContract(
                V1ActionContract(
                    "candidate-mlp-reduce-overhead",
                    "candidate",
                    "mlp_stack",
                    RequestedAction(
                        "reduce-overhead", "false", "true", None, False
                    ),
                    EffectiveAction(
                        "inductor",
                        "reduce-overhead",
                        False,
                        True,
                        None,
                        False,
                        "static",
                    ),
                ),
                "Try Torch reduce-overhead without making a performance promise.",
            ),
            V1SearchCandidateContract(
                V1ActionContract(
                    "candidate-mlp-max-autotune-equivalent",
                    "candidate",
                    "mlp_stack",
                    RequestedAction(
                        "max-autotune-equivalent", "false", "true", None, False
                    ),
                    EffectiveAction(
                        "inductor",
                        "max-autotune-no-cudagraphs",
                        False,
                        True,
                        None,
                        False,
                        "static",
                    ),
                ),
                "Try the installed CPU-compatible max-autotune equivalent.",
            ),
            V1SearchCandidateContract(
                V1ActionContract(
                    "candidate-mlp-epilogue-fusion-off",
                    "candidate",
                    "mlp_stack",
                    RequestedAction(
                        "default",
                        "false",
                        "true",
                        (("epilogue_fusion", False),),
                        False,
                    ),
                    EffectiveAction(
                        "inductor",
                        None,
                        False,
                        True,
                        (("epilogue_fusion", False),),
                        False,
                        "static",
                    ),
                ),
                "Try the explicitly admitted epilogue-fusion option setting.",
            ),
        ),
        "transformer_block": (
            V1SearchCandidateContract(
                V1ActionContract(
                    "candidate-transformer-default",
                    "candidate",
                    "transformer_block",
                    _STATIC_FULLGRAPH_REQUESTED,
                    _STATIC_FULLGRAPH_EFFECTIVE,
                ),
                "Try the calibrated native default on the transformer block.",
            ),
        ),
        "dynamic_batch_text": (
            V1SearchCandidateContract(
                V1ActionContract(
                    "candidate-dynamic-static",
                    "candidate",
                    "dynamic_batch_text",
                    _STATIC_FULLGRAPH_REQUESTED,
                    _STATIC_FULLGRAPH_EFFECTIVE,
                ),
                "Try static specialization across the pinned shape sweep.",
            ),
            V1SearchCandidateContract(
                V1ActionContract(
                    "candidate-dynamic-true",
                    "candidate",
                    "dynamic_batch_text",
                    RequestedAction("default", "true", "true", None, False),
                    EffectiveAction(
                        "inductor", None, True, True, None, False, "dynamic"
                    ),
                ),
                "Try Torch dynamic shapes across the pinned shape sweep.",
            ),
            V1SearchCandidateContract(
                V1ActionContract(
                    "candidate-dynamic-bucketed",
                    "candidate",
                    "dynamic_batch_text",
                    RequestedAction(
                        "default",
                        "bucketed",
                        "true",
                        None,
                        False,
                        (0, (2, 4), "reject"),
                    ),
                    EffectiveAction(
                        "inductor", None, False, True, None, False, "bucketed"
                    ),
                ),
                "Try the frozen batch buckets while preserving sequence variation.",
            ),
        ),
        "graph_break_bait": (
            V1SearchCandidateContract(
                V1ActionContract(
                    "candidate-graph-break-visible",
                    "candidate",
                    "graph_break_bait",
                    _GRAPH_BREAK_REQUESTED,
                    _GRAPH_BREAK_EFFECTIVE,
                ),
                "Expose and enumerate the workload's data-dependent graph break.",
            ),
        ),
    }
)


def v1_run_settings_json() -> dict[str, object]:
    """Return the exact schema-1 normal-run settings used by trusted workflows."""
    return {
        "warmup_runs": 5,
        "repeats": 31,
        "bootstrap_samples": 2000,
        "inter_run_spacing_seconds": 0.0,
    }


def v2_run_settings_json() -> dict[str, object]:
    """Return the exact schema-2 normal-run settings used by trusted workflows."""
    return {
        "schema_version": 2,
        "warmup_runs": 5,
        "repeats": 64,
        "bootstrap_samples": 2000,
        "inter_run_spacing_seconds": 0.0,
    }


def v1_compile_cache_json() -> dict[str, object]:
    """Return the exact schema-1 local-filesystem compiler-cache methodology."""
    return {
        "schema_version": 1,
        "policy": "fresh-per-run-local-filesystem-v1",
        "binding": "torch._inductor.utils.fresh_cache",
        "inductor_memory_cleared_between_phases": True,
        "local_fx_graph_cache": True,
        "local_aotautograd_cache": True,
        "remote_caches": False,
        "cpp_precompiled_headers": False,
        "warm_probe": "same-filesystem-cache-after-dynamo-and-inductor-memory-reset",
    }


def v1_path_normalization_json() -> dict[str, object]:
    """Return the exact schema-1 path-normalization declaration."""
    return {
        "schema_version": 1,
        "policy": "semantic-root-tokens-v1",
        "verbatim_fields": ["graph_breaks[].reason", "recompiles[].trigger"],
    }


def stable_signed_paired_effect(left: float, right: float) -> float:
    """Evaluate the declared paired-effect formula without overflow or half-sum underflow."""
    if any(
        isinstance(value, bool)
        or not isinstance(value, int | float)
        or not math.isfinite(value)
        or value < 0
        for value in (left, right)
    ):
        raise ValueError("paired-effect inputs must be finite and nonnegative")
    left_value = float(left)
    right_value = float(right)
    total = left_value + right_value
    if total == 0:
        return 0.0
    half_total = total / 2.0
    if math.isfinite(total) and half_total != 0:
        effect = (left_value - right_value) / half_total
    else:
        scale = max(left_value, right_value)
        if scale == 0:
            return 0.0
        normalized_left = left_value / scale
        normalized_right = right_value / scale
        denominator = (normalized_left + normalized_right) / 2.0
        if denominator == 0:
            raise ValueError("paired-effect denominator is zero")
        effect = (normalized_left - normalized_right) / denominator
    if not math.isfinite(effect):
        raise ValueError("paired effect must be finite")
    return effect


def strict_json_equal(actual: object, expected: object) -> bool:
    """Compare JSON-like values recursively with exact container and scalar types."""
    if type(actual) is not type(expected):
        return False
    if isinstance(expected, dict):
        assert isinstance(actual, dict)
        return actual.keys() == expected.keys() and all(
            strict_json_equal(actual[key], expected[key]) for key in expected
        )
    if isinstance(expected, list):
        assert isinstance(actual, list)
        return len(actual) == len(expected) and all(
            strict_json_equal(left, right)
            for left, right in zip(actual, expected, strict=True)
        )
    return actual == expected


def validate_v1_capability_snapshot(
    value: object,
) -> tuple[set[str], set[str], set[str]]:
    """Parse the exact schema-1 capability surface used to validate stored actions."""
    expected_keys = {
        "schema_version",
        "torch_version",
        "compiler_backends",
        "inductor_modes",
        "inductor_options",
    }
    if not isinstance(value, dict) or value.keys() != expected_keys:
        raise ValueError("invalid capability snapshot")
    if type(value.get("schema_version")) is not int or value["schema_version"] != 1:
        raise ValueError("invalid capability snapshot")
    if not isinstance(value.get("torch_version"), str) or not value["torch_version"]:
        raise ValueError("invalid capability snapshot")
    return (
        _string_set(value.get("compiler_backends")),
        _string_set(value.get("inductor_modes")),
        _string_set(value.get("inductor_options")),
    )


def action_references_capabilities(
    contract: V1ActionContract,
    capabilities: tuple[set[str], set[str], set[str]],
) -> bool:
    """Check the complete frozen action against its manifested capability names."""
    backends, modes, options = capabilities
    requested_options = contract.requested.options
    referenced_mode = (
        "default"
        if contract.requested.mode == "default"
        else contract.effective.mode
    )
    return (
        contract.effective.backend in backends
        and referenced_mode in modes
        and (
            requested_options is None
            or {name for name, _ in requested_options} <= options
        )
    )


def _string_set(value: object) -> set[str]:
    if (
        not isinstance(value, list)
        or not value
        or any(not isinstance(item, str) or not item for item in value)
        or len(value) != len(set(value))
    ):
        raise ValueError("invalid capability names")
    return set(value)


__all__ = [
    "V1_ACCURACY_TOLERANCE",
    "V1_AA_ACTION_BY_WORKLOAD",
    "V1_DEMO_ACTIONS",
    "V1_DEMO_ACTION_BY_ID",
    "V1_SEARCH_CANDIDATES_BY_WORKLOAD",
    "V1_FUNCTORCH_CACHE_PATCH",
    "V1_INDUCTOR_CACHE_PATCH",
    "V1_WORKLOAD_SHA256",
    "V1ActionContract",
    "V1SearchCandidateContract",
    "action_references_capabilities",
    "stable_signed_paired_effect",
    "strict_json_equal",
    "validate_v1_capability_snapshot",
    "v1_compile_cache_json",
    "v1_path_normalization_json",
    "v1_run_settings_json",
    "v2_run_settings_json",
]

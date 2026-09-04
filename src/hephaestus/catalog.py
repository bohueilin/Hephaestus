"""Pure, immutable authority schema for compiler configuration proposals."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from pathlib import PurePosixPath
from types import MappingProxyType


class CompileMode(StrEnum):
    DEFAULT = "default"
    REDUCE_OVERHEAD = "reduce-overhead"
    MAX_AUTOTUNE_EQUIVALENT = "max-autotune-equivalent"


class DynamicStrategy(StrEnum):
    FALSE = "false"
    TRUE = "true"
    BUCKETED = "bucketed"


class FullGraph(StrEnum):
    FALSE = "false"
    TRUE = "true"


class CatalogRole(StrEnum):
    CANDIDATE = "candidate"
    CLEAN_CONTROL = "clean-control"
    PLANTED = "planted"


class WorkloadName(StrEnum):
    MLP_STACK = "mlp_stack"
    TRANSFORMER_BLOCK = "transformer_block"
    DYNAMIC_BATCH_TEXT = "dynamic_batch_text"
    GRAPH_BREAK_BAIT = "graph_break_bait"


ScalarOption = str | int | bool


@dataclass(frozen=True, slots=True)
class OptionSetting:
    """One explicitly admitted scalar Inductor option."""

    name: str
    value: ScalarOption

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name or self.name != self.name.strip():
            raise ValueError("option name must be a nonempty normalized string")
        if type(self.value) not in {str, int, bool}:
            raise ValueError("option value must be a scalar string, integer, or boolean")


@dataclass(frozen=True, slots=True)
class BatchBucketPolicy:
    """The only v1 batch-bucketing policy admitted by the catalog."""

    boundaries: tuple[int, ...]
    axis: int = 0
    overflow_rule: str = "reject"

    def __post_init__(self) -> None:
        if (
            not isinstance(self.boundaries, tuple)
            or any(type(boundary) is not int for boundary in self.boundaries)
            or self.boundaries != (2, 4)
        ):
            raise ValueError("v1 bucket boundaries must be exactly (2, 4)")
        if type(self.axis) is not int or self.axis != 0 or self.overflow_rule != "reject":
            raise ValueError("invalid frozen bucket policy")


@dataclass(frozen=True, slots=True)
class CompileConfig:
    """A typed requested configuration before trusted resolution."""

    mode: CompileMode
    dynamic: DynamicStrategy
    fullgraph: FullGraph
    options: tuple[OptionSetting, ...] = ()
    disable: bool = False
    bucket_policy: BatchBucketPolicy | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.mode, CompileMode):
            raise ValueError("mode must be a CompileMode")
        if not isinstance(self.dynamic, DynamicStrategy):
            raise ValueError("dynamic must be a DynamicStrategy")
        if not isinstance(self.fullgraph, FullGraph):
            raise ValueError("fullgraph must be a FullGraph")
        if not isinstance(self.options, tuple) or any(
            not isinstance(option, OptionSetting) for option in self.options
        ):
            raise ValueError("options must be an immutable tuple of OptionSetting values")
        if type(self.disable) is not bool:
            raise ValueError("disable must be a boolean")
        if self.bucket_policy is not None and not isinstance(
            self.bucket_policy, BatchBucketPolicy
        ):
            raise ValueError("bucket_policy must be a BatchBucketPolicy")
        names = tuple(option.name for option in self.options)
        if len(names) != len(set(names)):
            raise ValueError("duplicate option names are not allowed")
        if self.options and self.mode is not CompileMode.DEFAULT:
            raise ValueError("backend options may be combined only with requested default mode")
        if self.dynamic is DynamicStrategy.BUCKETED and self.bucket_policy is None:
            raise ValueError("bucketed dynamic strategy requires a bucket policy")
        if self.dynamic is not DynamicStrategy.BUCKETED and self.bucket_policy is not None:
            raise ValueError("bucket policy requires the bucketed dynamic strategy")


@dataclass(frozen=True, slots=True)
class CatalogEntry:
    """One named executable action in the closed catalog."""

    catalog_id: str
    workload: WorkloadName
    config: CompileConfig
    role: CatalogRole
    rationale: str
    expected_verdict: str | None = None
    expected_driving_finding: str | None = None

    def __post_init__(self) -> None:
        if (
            not isinstance(self.catalog_id, str)
            or not self.catalog_id
            or self.catalog_id != self.catalog_id.strip()
        ):
            raise ValueError("catalog ID must be a nonempty normalized string")
        if not isinstance(self.workload, WorkloadName):
            raise ValueError("catalog workload must be a WorkloadName")
        if not isinstance(self.config, CompileConfig):
            raise ValueError("catalog config must be a CompileConfig")
        if not isinstance(self.role, CatalogRole):
            raise ValueError("catalog role must be a CatalogRole")
        _validate_rationale(self.rationale)
        if (
            self.config.dynamic is DynamicStrategy.BUCKETED
            and self.workload is not WorkloadName.DYNAMIC_BATCH_TEXT
        ):
            raise ValueError("bucketing is legal only for dynamic_batch_text")
        is_eager_plant = (
            self.catalog_id == "planted-eager-fallback"
            and self.role is CatalogRole.PLANTED
            and self.workload is WorkloadName.MLP_STACK
        )
        if self.config.disable and not is_eager_plant:
            raise ValueError("disable=True is reserved for the named planted eager fallback")
        has_expectation = (
            self.expected_verdict is not None
            and self.expected_driving_finding is not None
        )
        if self.role is CatalogRole.CANDIDATE and (
            self.expected_verdict is not None or self.expected_driving_finding is not None
        ):
            raise ValueError("optimizer candidates cannot declare expected verdicts")
        if self.role is not CatalogRole.CANDIDATE and not has_expectation:
            raise ValueError("demo entries must declare their expected verdict and finding")


@dataclass(frozen=True, slots=True)
class Proposal:
    """The complete authority an optimizer may submit."""

    catalog_id: str
    workload_name: WorkloadName
    rationale: str

    def __post_init__(self) -> None:
        if not isinstance(self.workload_name, WorkloadName):
            raise ValueError("proposal workload must be a WorkloadName")
        _validate_rationale(self.rationale)
        entry = get_catalog_entry(self.catalog_id)
        if entry.workload is not self.workload_name:
            raise ValueError("proposal workload does not match its catalog entry")


@dataclass(frozen=True, slots=True)
class ReadOnlyRunResult:
    """The only run information visible through the optimizer protocol."""

    bundle_relative_path: str
    verdict: str
    driving_finding: str

    def __post_init__(self) -> None:
        if not _is_run_relative_path(self.bundle_relative_path):
            raise ValueError("bundle relative path must be a canonical path below runs/")
        if not isinstance(self.verdict, str) or not self.verdict:
            raise ValueError("verdict must be a nonempty string")
        if not isinstance(self.driving_finding, str) or not self.driving_finding:
            raise ValueError("driving finding must be a nonempty string")


@dataclass(frozen=True, slots=True)
class ResolvedAction:
    """Pure scalar execution settings derived from one named entry."""

    backend: str
    mode: str | None
    dynamic: bool
    fullgraph: bool
    options: Mapping[str, ScalarOption] | None
    disable: bool
    input_plan_strategy: str
    bucket_boundaries: tuple[int, ...] | None


def _validate_rationale(rationale: object) -> None:
    if not isinstance(rationale, str) or not rationale.strip():
        raise ValueError("rationale must be nonempty")
    if rationale != rationale.strip() or "\n" in rationale or "\r" in rationale:
        raise ValueError("rationale must be one-line normalized text")


_STATIC_FULLGRAPH = CompileConfig(
    mode=CompileMode.DEFAULT,
    dynamic=DynamicStrategy.FALSE,
    fullgraph=FullGraph.TRUE,
)
_DYNAMIC_FULLGRAPH = CompileConfig(
    mode=CompileMode.DEFAULT,
    dynamic=DynamicStrategy.TRUE,
    fullgraph=FullGraph.TRUE,
)
_BUCKETED_FULLGRAPH = CompileConfig(
    mode=CompileMode.DEFAULT,
    dynamic=DynamicStrategy.BUCKETED,
    fullgraph=FullGraph.TRUE,
    bucket_policy=BatchBucketPolicy((2, 4)),
)


CATALOG: tuple[CatalogEntry, ...] = (
    CatalogEntry(
        "candidate-mlp-default",
        WorkloadName.MLP_STACK,
        _STATIC_FULLGRAPH,
        CatalogRole.CANDIDATE,
        "Try the calibrated native default with a required full graph.",
    ),
    CatalogEntry(
        "candidate-mlp-reduce-overhead",
        WorkloadName.MLP_STACK,
        CompileConfig(
            CompileMode.REDUCE_OVERHEAD,
            DynamicStrategy.FALSE,
            FullGraph.TRUE,
        ),
        CatalogRole.CANDIDATE,
        "Try Torch reduce-overhead without making a performance promise.",
    ),
    CatalogEntry(
        "candidate-mlp-max-autotune-equivalent",
        WorkloadName.MLP_STACK,
        CompileConfig(
            CompileMode.MAX_AUTOTUNE_EQUIVALENT,
            DynamicStrategy.FALSE,
            FullGraph.TRUE,
        ),
        CatalogRole.CANDIDATE,
        "Try the installed CPU-compatible max-autotune equivalent.",
    ),
    CatalogEntry(
        "candidate-mlp-epilogue-fusion-off",
        WorkloadName.MLP_STACK,
        CompileConfig(
            CompileMode.DEFAULT,
            DynamicStrategy.FALSE,
            FullGraph.TRUE,
            options=(OptionSetting("epilogue_fusion", False),),
        ),
        CatalogRole.CANDIDATE,
        "Try the explicitly admitted epilogue-fusion option setting.",
    ),
    CatalogEntry(
        "candidate-transformer-default",
        WorkloadName.TRANSFORMER_BLOCK,
        _STATIC_FULLGRAPH,
        CatalogRole.CANDIDATE,
        "Try the calibrated native default on the transformer block.",
    ),
    CatalogEntry(
        "candidate-dynamic-static",
        WorkloadName.DYNAMIC_BATCH_TEXT,
        _STATIC_FULLGRAPH,
        CatalogRole.CANDIDATE,
        "Try static specialization across the pinned shape sweep.",
    ),
    CatalogEntry(
        "candidate-dynamic-true",
        WorkloadName.DYNAMIC_BATCH_TEXT,
        _DYNAMIC_FULLGRAPH,
        CatalogRole.CANDIDATE,
        "Try Torch dynamic shapes across the pinned shape sweep.",
    ),
    CatalogEntry(
        "candidate-dynamic-bucketed",
        WorkloadName.DYNAMIC_BATCH_TEXT,
        _BUCKETED_FULLGRAPH,
        CatalogRole.CANDIDATE,
        "Try the frozen batch buckets while preserving sequence variation.",
    ),
    CatalogEntry(
        "candidate-graph-break-visible",
        WorkloadName.GRAPH_BREAK_BAIT,
        CompileConfig(
            CompileMode.DEFAULT,
            DynamicStrategy.FALSE,
            FullGraph.FALSE,
        ),
        CatalogRole.CANDIDATE,
        "Expose and enumerate the workload's data-dependent graph break.",
    ),
    CatalogEntry(
        "clean-control-mlp",
        WorkloadName.MLP_STACK,
        _STATIC_FULLGRAPH,
        CatalogRole.CLEAN_CONTROL,
        "Run the calibrated passing MLP control.",
        "PROVEN",
        "all_criteria_passed",
    ),
    CatalogEntry(
        "planted-eager-fallback",
        WorkloadName.MLP_STACK,
        CompileConfig(
            CompileMode.DEFAULT,
            DynamicStrategy.FALSE,
            FullGraph.TRUE,
            disable=True,
        ),
        CatalogRole.PLANTED,
        "Plant a compiler-disabled eager fallback.",
        "NOT_PROVEN",
        "perf.speedup_proven",
    ),
    CatalogEntry(
        "planted-static-shape-storm",
        WorkloadName.DYNAMIC_BATCH_TEXT,
        _STATIC_FULLGRAPH,
        CatalogRole.PLANTED,
        "Plant static specialization across four incompatible shapes.",
        "NOT_PROVEN",
        "graph.recompile_bound",
    ),
    CatalogEntry(
        "planted-graph-break-exposure",
        WorkloadName.GRAPH_BREAK_BAIT,
        CompileConfig(
            CompileMode.DEFAULT,
            DynamicStrategy.FALSE,
            FullGraph.FALSE,
        ),
        CatalogRole.PLANTED,
        "Plant a visible data-dependent graph break.",
        "CONDITIONAL",
        "graph.no_breaks",
    ),
)

CATALOG_BY_ID: Mapping[str, CatalogEntry] = MappingProxyType(
    {entry.catalog_id: entry for entry in CATALOG}
)
if len(CATALOG_BY_ID) != len(CATALOG):
    raise RuntimeError("catalog IDs must be unique")


def get_catalog_entry(catalog_id: str) -> CatalogEntry:
    """Resolve only an exact named action from the closed catalog."""
    if not isinstance(catalog_id, str):
        raise ValueError("unknown catalog ID")
    try:
        return CATALOG_BY_ID[catalog_id]
    except KeyError as error:
        raise ValueError(f"unknown catalog ID {catalog_id!r}") from error


def catalog_entries_for_optimizer(workload: WorkloadName) -> tuple[CatalogEntry, ...]:
    """Return fixed-order candidate actions, excluding all demo-only authority."""
    if not isinstance(workload, WorkloadName):
        raise ValueError("workload must be a WorkloadName")
    return tuple(
        entry
        for entry in CATALOG
        if entry.workload is workload and entry.role is CatalogRole.CANDIDATE
    )


def resolve_action(entry: CatalogEntry) -> ResolvedAction:
    """Map a reviewed entry to exact backend arguments without importing Torch."""
    if not isinstance(entry, CatalogEntry) or CATALOG_BY_ID.get(entry.catalog_id) is not entry:
        raise ValueError("entry is not the canonical catalog object")
    mode = {
        CompileMode.DEFAULT: None,
        CompileMode.REDUCE_OVERHEAD: "reduce-overhead",
        CompileMode.MAX_AUTOTUNE_EQUIVALENT: "max-autotune-no-cudagraphs",
    }[entry.config.mode]
    dynamic = entry.config.dynamic is DynamicStrategy.TRUE
    input_plan_strategy = {
        DynamicStrategy.FALSE: "static",
        DynamicStrategy.TRUE: "dynamic",
        DynamicStrategy.BUCKETED: "bucketed",
    }[entry.config.dynamic]
    options = (
        None
        if not entry.config.options
        else MappingProxyType({option.name: option.value for option in entry.config.options})
    )
    bucket_boundaries = (
        None
        if entry.config.bucket_policy is None
        else entry.config.bucket_policy.boundaries
    )
    return ResolvedAction(
        backend="inductor",
        mode=mode,
        dynamic=dynamic,
        fullgraph=entry.config.fullgraph is FullGraph.TRUE,
        options=options,
        disable=entry.config.disable,
        input_plan_strategy=input_plan_strategy,
        bucket_boundaries=bucket_boundaries,
    )


def validate_catalog_capabilities(snapshot: Mapping[str, object]) -> None:
    """Prove every reviewed backend, mode, and option exists in the pinned snapshot."""
    if not isinstance(snapshot, Mapping) or type(snapshot.get("schema_version")) is not int:
        raise ValueError("invalid capability snapshot")
    if snapshot.get("schema_version") != 1:
        raise ValueError("invalid capability snapshot")
    backends = _string_set(snapshot.get("compiler_backends"), "backend")
    modes = _string_set(snapshot.get("inductor_modes"), "mode")
    options = _string_set(snapshot.get("inductor_options"), "option")
    if "inductor" not in backends:
        raise ValueError("catalog backend is absent from capability snapshot")
    required_modes = {"default", "reduce-overhead", "max-autotune-no-cudagraphs"}
    if not required_modes <= modes:
        raise ValueError("catalog mode is absent from capability snapshot")
    required_options = {
        option.name for entry in CATALOG for option in entry.config.options
    }
    if not required_options <= options:
        raise ValueError("catalog option is absent from capability snapshot")


def catalog_metadata(entry: CatalogEntry) -> dict[str, object]:
    """Serialize requested and effective action settings for a child evidence bundle."""
    action = resolve_action(entry)
    requested_options = (
        None
        if not entry.config.options
        else {option.name: option.value for option in entry.config.options}
    )
    requested_bucket = (
        None
        if entry.config.bucket_policy is None
        else {
            "axis": entry.config.bucket_policy.axis,
            "boundaries": list(entry.config.bucket_policy.boundaries),
            "overflow_rule": entry.config.bucket_policy.overflow_rule,
        }
    )
    return {
        "schema_version": 1,
        "entry_id": entry.catalog_id,
        "role": entry.role.value,
        "workload_name": entry.workload.value,
        "requested": {
            "mode": entry.config.mode.value,
            "dynamic": entry.config.dynamic.value,
            "fullgraph": entry.config.fullgraph.value,
            "options": requested_options,
            "disable": entry.config.disable,
            "bucket_policy": requested_bucket,
        },
        "effective": {
            "backend": action.backend,
            "mode": action.mode,
            "dynamic": action.dynamic,
            "fullgraph": action.fullgraph,
            "options": None if action.options is None else dict(action.options),
            "disable": action.disable,
            "input_plan_strategy": action.input_plan_strategy,
        },
    }


def catalog_json(torch_capabilities_sha256: str | None = None) -> dict[str, object]:
    """Return the complete canonical catalog evidence payload."""
    result: dict[str, object] = {
        "schema_version": 1,
        "entries": [_catalog_entry_json(entry) for entry in CATALOG],
    }
    if torch_capabilities_sha256 is not None:
        if not _is_sha256(torch_capabilities_sha256):
            raise ValueError("torch capability digest must be lowercase SHA-256")
        result["torch_capabilities_sha256"] = torch_capabilities_sha256
    return result


def _catalog_entry_json(entry: CatalogEntry) -> dict[str, object]:
    evidence = catalog_metadata(entry)
    evidence.update(
        {
            "rationale": entry.rationale,
            "expected_verdict": entry.expected_verdict,
            "expected_driving_finding": entry.expected_driving_finding,
        }
    )
    return evidence


def _is_run_relative_path(path: object) -> bool:
    if not isinstance(path, str) or not path or "\\" in path:
        return False
    candidate = PurePosixPath(path)
    return (
        candidate.as_posix() == path
        and not candidate.is_absolute()
        and candidate.parts
        and candidate.parts[0] == "runs"
        and len(candidate.parts) > 1
        and "." not in candidate.parts
        and ".." not in candidate.parts
    )


def _string_set(value: object, kind: str) -> set[str]:
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        raise ValueError(f"invalid capability {kind} list")
    return set(value)


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


__all__ = [
    "CATALOG",
    "CATALOG_BY_ID",
    "BatchBucketPolicy",
    "CatalogEntry",
    "CatalogRole",
    "CompileConfig",
    "CompileMode",
    "DynamicStrategy",
    "FullGraph",
    "OptionSetting",
    "Proposal",
    "ReadOnlyRunResult",
    "ResolvedAction",
    "WorkloadName",
    "catalog_entries_for_optimizer",
    "catalog_json",
    "catalog_metadata",
    "get_catalog_entry",
    "resolve_action",
    "validate_catalog_capabilities",
]

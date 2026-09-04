"""Version-pinned bindings to the real PyTorch 2.13 compiler APIs."""

from __future__ import annotations

import dataclasses
import logging
import os
import platform
import tempfile
import threading
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn

from hephaestus.bundle import strict_json_loads
from hephaestus.scope import EVIDENCE_BOUNDARY

_CAPABILITIES_PATH = Path(__file__).with_name("torch_capabilities.json")
_EMPTY_EVIDENCE: dict[str, list[Any]] = {
    "graph_breaks": [],
    "compilation_metrics": [],
    "recompile_reasons": [],
    "log_records": [],
}
_LAST_EVIDENCE: dict[str, list[Any]] = deepcopy(_EMPTY_EVIDENCE)
_CACHE_SCOPE_LOCK = threading.RLock()
_ACTIVE_CACHE_SCOPE: InductorCacheScope | None = None


@dataclass(frozen=True, slots=True)
class InductorCacheScope:
    """Process-owned local filesystem cache boundary for one measured run."""

    owned_root: Path
    cache_root: Path


@dataclass(frozen=True, slots=True)
class CompileRequest:
    """Arguments admitted by the v0.1 compiler configuration boundary."""

    backend: str
    mode: str | None
    dynamic: bool | None
    fullgraph: bool
    options: Mapping[str, str | int | bool] | None
    disable: bool


def _discover_capabilities() -> dict[str, Any]:
    # Binding: these public discovery APIs enumerate names accepted by this exact build.
    return {
        "schema_version": 1,
        "torch_version": torch.__version__,
        "compiler_backends": sorted(torch.compiler.list_backends()),
        "inductor_modes": sorted(torch._inductor.list_mode_options()),
        # Names are availability metadata, not a claim that every option works on CPU.
        "inductor_options": sorted(torch._inductor.list_options()),
    }


def _verify_runtime_capabilities() -> None:
    pinned = strict_json_loads(_CAPABILITIES_PATH.read_text(encoding="utf-8"))
    discovered = _discover_capabilities()
    if discovered != pinned:
        raise RuntimeError(
            "PyTorch compiler capabilities differ from torch_capabilities.json; "
            "use the pinned Torch build or deliberately refresh and review the snapshot"
        )


def compile_module(module: nn.Module, request: CompileRequest) -> Any:
    """Compile a module through the pinned torch.compile argument surface."""

    _verify_runtime_capabilities()
    if request.backend == "inductor" and not request.disable and _ACTIVE_CACHE_SCOPE is None:
        raise RuntimeError("enabled Inductor compile requires an owned cache scope")
    # Binding: torch.compile is the real execution boundary; no eager substitute is used.
    return torch.compile(
        module,
        backend=request.backend,
        mode=request.mode,
        dynamic=request.dynamic,
        fullgraph=request.fullgraph,
        options=dict(request.options) if request.options is not None else None,
        disable=request.disable,
    )


@contextmanager
def inductor_cache_scope() -> Iterator[InductorCacheScope]:
    """Own an empty fresh-cache directory and disable every remote cache path."""
    from torch._functorch import config as functorch_config
    from torch._inductor import config as inductor_config
    from torch._inductor.utils import fresh_cache

    global _ACTIVE_CACHE_SCOPE

    inductor_patch = {
        "fx_graph_cache": True,
        "fx_graph_remote_cache": False,
        "autotune_remote_cache": False,
        "bundled_autotune_remote_cache": False,
        "remote_gemm_autotune_cache": False,
        "force_disable_caches": False,
        "cpp_cache_precompile_headers": False,
    }
    functorch_patch = {
        "enable_autograd_cache": True,
        "enable_remote_autograd_cache": False,
    }
    with _CACHE_SCOPE_LOCK:
        if _ACTIVE_CACHE_SCOPE is not None:
            raise RuntimeError("nested Inductor cache scopes are not admitted")
        with tempfile.TemporaryDirectory(prefix="hephaestus-cache-") as owned_name:
            owned_root = Path(owned_name)
            with (
                inductor_config.patch(inductor_patch),
                functorch_config.patch(functorch_patch),
                fresh_cache(dir=str(owned_root), delete=False),
            ):
                cache_root = Path(os.environ["TORCHINDUCTOR_CACHE_DIR"])
                scope = InductorCacheScope(owned_root, cache_root)
                _ACTIVE_CACHE_SCOPE = scope
                try:
                    yield scope
                finally:
                    _ACTIVE_CACHE_SCOPE = None


def active_inductor_cache_scope() -> InductorCacheScope | None:
    """Return the active trusted scope to the measurement layer."""
    return _ACTIVE_CACHE_SCOPE


def clear_compiler_memory_state() -> None:
    """Clear process memory while deliberately retaining the active filesystem cache."""
    global _LAST_EVIDENCE

    torch._dynamo.reset()
    torch._inductor.utils.clear_caches()
    torch._dynamo.utils.clear_compilation_metrics()
    torch._dynamo.utils.graph_break_reasons.clear()
    _LAST_EVIDENCE = deepcopy(_EMPTY_EVIDENCE)


def reset_compiler_state() -> None:
    """Clear Dynamo caches and all evidence accumulators for one isolated run."""

    global _LAST_EVIDENCE

    # Binding: Dynamo reset drops compiled-frame caches between measured runs.
    torch._dynamo.reset()
    torch._inductor.utils.clear_caches()
    # Binding: the 2.13 metrics utility otherwise retains compile records process-wide.
    torch._dynamo.utils.clear_compilation_metrics()
    # Binding: graph_break_reasons is the 2.13 store of GraphCompileReason records.
    torch._dynamo.utils.graph_break_reasons.clear()
    _LAST_EVIDENCE = deepcopy(_EMPTY_EVIDENCE)


class _EvidenceLogHandler(logging.Handler):
    def __init__(self) -> None:
        super().__init__()
        self.records: list[dict[str, Any]] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.records.append(
            {
                "name": record.name,
                "level": record.levelname,
                # getMessage retains the full formatted composite guard-failure payload.
                "message": record.getMessage(),
            }
        )


@dataclass(slots=True)
class _LoggerState:
    level: int
    handlers: tuple[logging.Handler, ...]
    filters: tuple[logging.Filter, ...]
    propagate: bool
    disabled: bool


def _torch_logger_names() -> set[str]:
    import torch._logging._internal as logging_internal

    # Binding: the 2.13 private registry enumerates every component, artifact, and child
    # logger whose handlers/levels torch._logging.set_logs may mutate during capture.
    registry = logging_internal.log_registry
    return {
        *registry.get_log_qnames(),
        *registry.get_artifact_log_qnames(),
        *registry.get_child_log_qnames(),
        "torch._dynamo",
    }


def _snapshot_loggers() -> dict[str, _LoggerState]:
    snapshot: dict[str, _LoggerState] = {}
    for name in _torch_logger_names():
        logger = logging.getLogger(name)
        snapshot[name] = _LoggerState(
            level=logger.level,
            handlers=tuple(logger.handlers),
            filters=tuple(logger.filters),
            propagate=logger.propagate,
            disabled=logger.disabled,
        )
    return snapshot


def _restore_loggers(snapshot: dict[str, _LoggerState]) -> None:
    for name, state in snapshot.items():
        logger = logging.getLogger(name)
        logger.handlers[:] = state.handlers
        logger.filters[:] = state.filters
        logger.setLevel(state.level)
        logger.propagate = state.propagate
        logger.disabled = state.disabled


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, str | int | float | bool):
        return value
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, set | frozenset):
        return sorted(_json_safe(item) for item in value)
    return str(value)


def _serialize_stack(stack: list[Any]) -> list[dict[str, Any]]:
    return [
        {
            "filename": frame.filename,
            "lineno": frame.lineno,
            "name": frame.name,
            "line": frame.line,
        }
        for frame in stack
    ]


def _collect_evidence(log_records: list[dict[str, Any]]) -> dict[str, list[Any]]:
    # Binding: each GraphCompileReason exposes the verbatim reason and separate user stack.
    graph_breaks = [
        {
            "reason": reason.reason,
            "user_stack": _serialize_stack(reason.user_stack),
        }
        for reason in torch._dynamo.utils.graph_break_reasons
    ]
    # Binding: get_compilation_metrics returns one 2.13 record per compile/recompile.
    metrics = [
        _json_safe(dataclasses.asdict(metric))
        for metric in torch._dynamo.utils.get_compilation_metrics()
    ]
    # Binding: recompile_reason carries the upstream guard trigger string verbatim.
    recompile_reasons = [
        metric.recompile_reason
        for metric in torch._dynamo.utils.get_compilation_metrics()
        if metric.recompile_reason
    ]
    return {
        "graph_breaks": graph_breaks,
        "compilation_metrics": metrics,
        "recompile_reasons": recompile_reasons,
        "log_records": deepcopy(log_records),
    }


@contextmanager
def capture_compiler_evidence() -> Iterator[None]:
    """Capture verbose Dynamo logs and materialize compiler evidence on exit."""

    import torch._logging._internal as logging_internal

    global _LAST_EVIDENCE

    logger_snapshot = _snapshot_loggers()
    # Binding: the 2.13 private log_state stores the explicit qualified logger levels and
    # enabled artifact names that set_logs replaces; both collections must be preserved.
    level_snapshot = dict(logging_internal.log_state.log_qname_to_level)
    artifact_snapshot = set(logging_internal.log_state.artifact_names)

    # Binding: these 2.13 switches emit graph breaks and full composite guard failures.
    torch._logging.set_logs(graph_breaks=True, recompiles_verbose=True)
    handler = _EvidenceLogHandler()
    dynamo_logger = logging.getLogger("torch._dynamo")
    dynamo_logger.addHandler(handler)
    try:
        yield
    finally:
        try:
            _LAST_EVIDENCE = _collect_evidence(handler.records)
        finally:
            dynamo_logger.removeHandler(handler)
            # Binding: restore those exact 2.13 log_state collections before restoring the
            # concrete logger handlers, filters, levels, propagation, and disabled flags.
            logging_internal.log_state.clear()
            logging_internal.log_state.log_qname_to_level.update(level_snapshot)
            logging_internal.log_state.artifact_names.update(artifact_snapshot)
            _restore_loggers(logger_snapshot)


def read_compiler_evidence() -> dict[str, list[Any]]:
    """Return a defensive copy of the most recently completed capture."""

    return deepcopy(_LAST_EVIDENCE)


def snapshot_compiler_evidence() -> dict[str, list[Any]]:
    """Snapshot current compiler counters inside an active capture phase."""
    return _collect_evidence([])


def environment_snapshot() -> dict[str, Any]:
    """Return reproducibility fields without user, host, or home-directory identity."""

    return {
        "schema_version": 1,
        "torch": {
            "version": torch.__version__,
            "git_version": torch.version.git_version,
            "debug": torch.version.debug,
        },
        "python": {
            "implementation": platform.python_implementation(),
            "version": platform.python_version(),
        },
        "os": {
            "system": platform.system(),
            "release": platform.release(),
        },
        "chip": platform.machine(),
        "boundary": EVIDENCE_BOUNDARY,
    }


_verify_runtime_capabilities()


__all__ = [
    "InductorCacheScope",
    "CompileRequest",
    "active_inductor_cache_scope",
    "capture_compiler_evidence",
    "clear_compiler_memory_state",
    "compile_module",
    "environment_snapshot",
    "inductor_cache_scope",
    "read_compiler_evidence",
    "reset_compiler_state",
    "snapshot_compiler_evidence",
]

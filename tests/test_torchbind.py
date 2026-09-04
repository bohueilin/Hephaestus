import getpass
import json
import logging
import platform
import socket
import warnings
from importlib.resources import files
from pathlib import Path

import pytest
import torch
from torch import nn

import hephaestus.torchbind as torchbind_module
from hephaestus.torchbind import (
    CompileRequest,
    capture_compiler_evidence,
    compile_module,
    environment_snapshot,
    inductor_cache_scope,
    read_compiler_evidence,
    reset_compiler_state,
)

pytestmark = pytest.mark.slow

EMPTY_EVIDENCE = {
    "graph_breaks": [],
    "compilation_metrics": [],
    "recompile_reasons": [],
    "log_records": [],
}
INDUCTOR_REQUEST = CompileRequest(
    backend="inductor",
    mode="default",
    dynamic=False,
    fullgraph=False,
    options=None,
    disable=False,
)


class ItemBranch(nn.Module):
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if inputs.sum().item() > 0:
            return torch.sin(inputs)
        return torch.cos(inputs)


class ShapeSensitive(nn.Module):
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return torch.sin(inputs) + 1


def test_reset_makes_compiler_evidence_per_run() -> None:
    with inductor_cache_scope():
        reset_compiler_state()
        with capture_compiler_evidence():
            compiled = compile_module(ItemBranch().eval(), INDUCTOR_REQUEST)
            compiled(torch.ones(2, 3))
        assert read_compiler_evidence()["graph_breaks"]

        reset_compiler_state()

        assert read_compiler_evidence() == EMPTY_EVIDENCE


def test_item_break_preserves_upstream_reason_and_stack() -> None:
    with inductor_cache_scope():
        reset_compiler_state()
        with capture_compiler_evidence():
            compiled = compile_module(ItemBranch().eval(), INDUCTOR_REQUEST)
            compiled(torch.ones(2, 3))
        evidence = read_compiler_evidence()

    reasons = [entry["reason"] for entry in evidence["graph_breaks"]]
    assert all(reasons)
    assert any("Tensor.item" in reason for reason in reasons)
    assert any(entry["user_stack"] for entry in evidence["graph_breaks"])
    assert evidence["compilation_metrics"]
    assert any(record["message"] for record in evidence["log_records"])


def test_static_shape_change_preserves_recompile_triggers() -> None:
    with inductor_cache_scope():
        reset_compiler_state()
        with capture_compiler_evidence():
            compiled = compile_module(ShapeSensitive().eval(), INDUCTOR_REQUEST)
            compiled(torch.ones(2, 3))
            compiled(torch.ones(4, 3))
        evidence = read_compiler_evidence()

    assert evidence["recompile_reasons"]
    assert all(reason for reason in evidence["recompile_reasons"])
    assert any("size mismatch" in reason for reason in evidence["recompile_reasons"])
    verbose_messages = [
        record["message"]
        for record in evidence["log_records"]
        if "recompiles_verbose" in record["name"]
    ]
    assert verbose_messages
    assert any("size mismatch" in message for message in verbose_messages)


def test_capture_restores_torch_logging_and_handlers() -> None:
    import torch._logging._internal as logging_internal

    dynamo_logger = logging.getLogger("torch._dynamo")
    handlers_before = tuple(dynamo_logger.handlers)
    levels_before = dict(logging_internal.log_state.log_qname_to_level)
    artifacts_before = set(logging_internal.log_state.artifact_names)

    with capture_compiler_evidence():
        pass

    assert tuple(dynamo_logger.handlers) == handlers_before
    assert logging_internal.log_state.log_qname_to_level == levels_before
    assert logging_internal.log_state.artifact_names == artifacts_before


def test_environment_snapshot_is_useful_but_machine_nonidentifying() -> None:
    snapshot = environment_snapshot()

    assert set(snapshot) >= {"torch", "python", "os", "chip"}
    assert snapshot["torch"]["version"] == torch.__version__
    assert snapshot["python"] == {
        "implementation": platform.python_implementation(),
        "version": platform.python_version(),
    }
    assert snapshot["os"]["system"] == platform.system()
    assert snapshot["chip"] == platform.machine()
    serialized = json.dumps(snapshot).casefold()
    identifying_values = {getpass.getuser(), socket.gethostname(), str(Path.home())}
    assert all(value.casefold() not in serialized for value in identifying_values if value)
    assert not ({"hostname", "username", "home"} & set(snapshot))


def test_warning_policy_filters_only_the_pinned_torch_warning() -> None:
    pinned_message = (
        "`torch.jit.script_method` is not supported in Python 3.14+ and may break. "
        "Please switch to `torch.compile` or `torch.export`."
    )
    with warnings.catch_warnings(record=True) as captured:
        warnings.warn_explicit(
            pinned_message,
            DeprecationWarning,
            filename="torch/jit/_script.py",
            lineno=359,
            module="torch.jit._script",
            registry={},
        )
        warnings.warn("unrelated deprecation", DeprecationWarning, stacklevel=1)

    assert [str(warning.message) for warning in captured] == ["unrelated deprecation"]


def test_committed_capabilities_match_the_runtime_discovery_snapshot() -> None:
    capability_path = files("hephaestus").joinpath("torch_capabilities.json")
    pinned = json.loads(capability_path.read_text(encoding="utf-8"))
    discovered = {
        "schema_version": 1,
        "torch_version": torch.__version__,
        "compiler_backends": sorted(torch.compiler.list_backends()),
        "inductor_modes": sorted(torch._inductor.list_mode_options()),
        "inductor_options": sorted(torch._inductor.list_options()),
    }

    assert pinned == discovered
    assert pinned["torch_version"] == "2.13.0"
    assert "inductor" in pinned["compiler_backends"]
    assert pinned["inductor_modes"] == [
        "default",
        "lite",
        "max-autotune",
        "max-autotune-no-cudagraphs",
        "reduce-overhead",
    ]
    assert len(pinned["inductor_options"]) == 622


def test_enabled_inductor_compile_requires_owned_cache_scope() -> None:
    """An enabled Inductor compile outside the authenticated fresh-cache scope must fail."""
    with pytest.raises(RuntimeError, match="cache scope"):
        compile_module(ShapeSensitive().eval(), INDUCTOR_REQUEST)


def test_owned_cache_scope_applies_exact_local_only_patches_and_restores() -> None:
    """The trusted scope must own an empty filesystem cache and patch every remote path off."""
    scope_factory = getattr(torchbind_module, "inductor_cache_scope", None)
    assert callable(scope_factory)

    from torch._functorch import config as functorch_config
    from torch._inductor import config as inductor_config

    before = {
        "fx_graph_cache": inductor_config.fx_graph_cache,
        "fx_graph_remote_cache": inductor_config.fx_graph_remote_cache,
        "autotune_remote_cache": inductor_config.autotune_remote_cache,
        "bundled_autotune_remote_cache": inductor_config.bundled_autotune_remote_cache,
        "remote_gemm_autotune_cache": inductor_config.remote_gemm_autotune_cache,
        "force_disable_caches": inductor_config.force_disable_caches,
        "cpp_cache_precompile_headers": inductor_config.cpp_cache_precompile_headers,
        "enable_autograd_cache": functorch_config.enable_autograd_cache,
        "enable_remote_autograd_cache": functorch_config.enable_remote_autograd_cache,
    }
    with scope_factory() as scope:
        assert scope.cache_root.is_dir()
        assert list(scope.cache_root.iterdir()) == []
        assert inductor_config.fx_graph_cache is True
        assert inductor_config.fx_graph_remote_cache is False
        assert inductor_config.autotune_remote_cache is False
        assert inductor_config.bundled_autotune_remote_cache is False
        assert inductor_config.remote_gemm_autotune_cache is False
        assert inductor_config.force_disable_caches is False
        assert inductor_config.cpp_cache_precompile_headers is False
        assert functorch_config.enable_autograd_cache is True
        assert functorch_config.enable_remote_autograd_cache is False
        owned_root = scope.owned_root
        assert scope.cache_root.parent == owned_root
    assert not owned_root.exists()
    assert {
        "fx_graph_cache": inductor_config.fx_graph_cache,
        "fx_graph_remote_cache": inductor_config.fx_graph_remote_cache,
        "autotune_remote_cache": inductor_config.autotune_remote_cache,
        "bundled_autotune_remote_cache": inductor_config.bundled_autotune_remote_cache,
        "remote_gemm_autotune_cache": inductor_config.remote_gemm_autotune_cache,
        "force_disable_caches": inductor_config.force_disable_caches,
        "cpp_cache_precompile_headers": inductor_config.cpp_cache_precompile_headers,
        "enable_autograd_cache": functorch_config.enable_autograd_cache,
        "enable_remote_autograd_cache": functorch_config.enable_remote_autograd_cache,
    } == before

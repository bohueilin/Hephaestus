from __future__ import annotations

import getpass
import importlib
import importlib.util
from pathlib import Path

import pytest


def _privacy_module():
    assert importlib.util.find_spec("hephaestus.privacy") is not None
    return importlib.import_module("hephaestus.privacy")


def test_recursive_normalization_replaces_longest_roots_in_keys_and_values() -> None:
    """Private roots nested in metric keys, stacks, logs, and config strings become tokens."""
    privacy = _privacy_module()
    roots = {
        "<INDUCTOR_CACHE>": Path("/private/cache"),
        "<HEPHAESTUS_PACKAGE>": Path("/private/project/src/hephaestus"),
        "<TORCH_PACKAGE>": Path("/private/env/site-packages/torch"),
        "<PYTHON_ENV>": Path("/private/env"),
        "<PYTHON_RUNTIME>": Path("/private/runtime"),
        "<PROJECT_ROOT>": Path("/private/project"),
        "<HOME>": Path("/Users/private"),
        "<TEMP_ROOT>": Path("/private/tmp"),
    }
    value = {
        "/private/project/src/hephaestus/metric.py": {
            "stack": "/private/env/site-packages/torch/_dynamo/x.py",
            "cache": "/private/cache/fx/entry",
            "runtime": "/private/runtime/bin/python",
            "home": "/Users/private/file",
            "temp": "/private/tmp/log",
        }
    }

    normalized = privacy.normalize_public_evidence(value, roots)

    assert normalized == {
        "<HEPHAESTUS_PACKAGE>/metric.py": {
            "stack": "<TORCH_PACKAGE>/_dynamo/x.py",
            "cache": "<INDUCTOR_CACHE>/fx/entry",
            "runtime": "<PYTHON_RUNTIME>/bin/python",
            "home": "<HOME>/file",
            "temp": "<TEMP_ROOT>/log",
        }
    }


def test_normalization_rejects_key_collisions_and_residual_user_paths() -> None:
    """Normalization must fail closed when distinct private inputs collapse or survive."""
    privacy = _privacy_module()
    roots = {"<HOME>": Path("/Users/private")}
    with pytest.raises(ValueError, match="collision"):
        privacy.normalize_public_evidence(
            {"/Users/private/key": 1, "<HOME>/key": 2},
            roots,
        )
    with pytest.raises(ValueError, match="private"):
        privacy.validate_public_evidence({"message": "/Users/someone/file.py"})


@pytest.mark.parametrize(
    "absolute_path",
    ("/opt/private-project/model.py", "/private/var/custom/build.log"),
)
def test_validation_rejects_generic_absolute_local_paths(absolute_path: str) -> None:
    """Absolute local paths outside known user-home prefixes cannot enter evidence."""
    with pytest.raises(ValueError, match="path"):
        _privacy_module().validate_public_evidence({"message": absolute_path})


def test_normalization_rejects_duplicate_rendered_semantic_roots() -> None:
    """One rendered private root cannot nondeterministically name two semantic tokens."""
    privacy = _privacy_module()

    with pytest.raises(ValueError, match="duplicate"):
        privacy.normalize_public_evidence(
            {"message": "/private/shared/file.py"},
            {
                "<HOME>": Path("/private/shared"),
                "<PROJECT_ROOT>": Path("/private/shared"),
            },
        )


def test_validation_rejects_the_residual_local_username() -> None:
    """A bare machine-local account name is private even outside a path."""
    username = getpass.getuser()
    assert len(username) >= 4

    with pytest.raises(ValueError, match="username"):
        _privacy_module().validate_public_evidence({"owner": username})


def test_verbatim_reason_and_trigger_are_never_rewritten_and_private_values_fail() -> None:
    """Scientific reason/trigger bytes stay upstream-authentic or invalidate the run."""
    privacy = _privacy_module()
    roots = {"<HOME>": Path("/Users/private")}
    report = {
        "graph_breaks": [{"reason": "Tensor.item", "user_stack": []}],
        "recompiles": [{"trigger": "size mismatch"}],
        "log_records": [{"message": "/Users/private/log.py"}],
    }

    normalized = privacy.normalize_dynamo_report(report, roots)

    assert normalized["graph_breaks"][0]["reason"] == "Tensor.item"
    assert normalized["recompiles"][0]["trigger"] == "size mismatch"
    assert normalized["log_records"][0]["message"] == "<HOME>/log.py"
    report["graph_breaks"][0]["reason"] = "/Users/private/secret.py"
    with pytest.raises(ValueError, match="verbatim"):
        privacy.normalize_dynamo_report(report, roots)

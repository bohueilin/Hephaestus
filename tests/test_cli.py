from __future__ import annotations

import json
import os
import re
import subprocess
import tempfile
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest

from hephaestus.bundle import write_json
from hephaestus.scope import EVIDENCE_BOUNDARY
from tests.test_gate import _write_bundle

ROOT = Path(__file__).parents[1]
PYTHON = ROOT / ".venv" / "bin" / "python"
HEPHAESTUS = ROOT / ".venv" / "bin" / "hephaestus"


def _command(*arguments: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [str(HEPHAESTUS), *arguments],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )


def _injected(
    arguments: list[str],
    module_name: str,
    source: str,
) -> subprocess.CompletedProcess[str]:
    code = f"""
import sys, types
from pathlib import Path
from types import SimpleNamespace as NS
module = types.ModuleType({module_name!r})
{source}
sys.modules[{module_name!r}] = module
"""
    if arguments[0] not in {"aa-test", "demo-planted-regressions"}:
        direct = code + f"""
from hephaestus.cli import main
raise SystemExit(main({arguments!r}))
"""
        return subprocess.run(
            [str(PYTHON), "-c", direct],
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=False,
        )
    with tempfile.TemporaryDirectory() as temporary:
        Path(temporary, "sitecustomize.py").write_text(code, encoding="utf-8")
        environment = os.environ.copy()
        existing = environment.get("PYTHONPATH")
        environment["PYTHONPATH"] = (
            temporary if not existing else f"{temporary}{os.pathsep}{existing}"
        )
        control_read_fd, control_write_fd = os.pipe()
        try:
            completed = subprocess.run(
                [
                    str(PYTHON),
                    "-m",
                    "hephaestus.durable_worker",
                    json.dumps(arguments),
                    str(control_write_fd),
                ],
                cwd=ROOT,
                env=environment,
                text=True,
                capture_output=True,
                check=False,
                pass_fds=(control_write_fd,),
            )
        finally:
            os.close(control_write_fd)
        try:
            while os.read(control_read_fd, 4096):
                pass
        finally:
            os.close(control_read_fd)
        return completed


def test_top_help_exposes_exactly_five_commands_without_importing_torch() -> None:
    """Adding a hidden authority surface or eager Torch import must fail top-level help."""
    code = """
import sys
from hephaestus.cli import main
try:
    main(['--help'])
except SystemExit as error:
    assert error.code == 0
else:
    raise AssertionError('argparse help did not exit')
assert 'torch' not in sys.modules
"""
    completed = subprocess.run(
        [str(PYTHON), "-c", code], cwd=ROOT, text=True, capture_output=True, check=False
    )

    assert completed.returncode == 0, completed.stderr
    command_group = re.search(r"\{([^}]+)\}", completed.stdout)
    assert command_group is not None
    assert command_group.group(1).split(",") == [
        "run",
        "gate",
        "agent",
        "demo-planted-regressions",
        "aa-test",
    ]


def test_nested_agent_help_exposes_only_optimize() -> None:
    """The agent namespace must not expose direct harness or catalog mutation commands."""
    completed = _command("agent", "--help")

    assert completed.returncode == 0
    command_group = re.search(r"\{([^}]+)\}", completed.stdout)
    assert command_group is not None
    assert command_group.group(1) == "optimize"


def test_cli_keeps_packaged_criteria_context_open_through_workflow(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A temporary importlib resource must outlive the complete trusted workflow."""
    import hephaestus.criteria as criteria_module
    import hephaestus.workflows as workflows_module
    from hephaestus.cli import main

    events: list[str] = []
    criteria_path = tmp_path / "materialized-gate.yaml"

    @contextmanager
    def materialized_criteria():
        events.append("enter")
        yield criteria_path
        events.append("exit")

    def run_catalog_action(*args: object, **kwargs: object) -> SimpleNamespace:
        del args, kwargs
        assert events == ["enter"]
        events.append("workflow")
        return SimpleNamespace(
            bundle_path=tmp_path / "bundle",
            verdict="PROVEN",
            driving_finding="all_criteria_passed",
        )

    monkeypatch.setattr(criteria_module, "packaged_criteria_path", materialized_criteria)
    monkeypatch.setattr(workflows_module, "run_catalog_action", run_catalog_action)

    assert main(["run", "mlp_stack", "--config", "candidate-mlp-default"]) == 0
    assert events == ["enter", "workflow", "exit"]
    capsys.readouterr()


@pytest.mark.parametrize(
    "arguments",
    (
        ["run", "unknown", "--config", "candidate-mlp-default"],
        ["run", "mlp_stack", "--config", "unknown"],
        ["run", "transformer_block", "--config", "candidate-mlp-default"],
    ),
)
def test_invalid_name_or_workload_config_pair_exits_two_without_artifacts(
    tmp_path: Path,
    arguments: list[str],
) -> None:
    """Parser validation must precede output-directory creation and harness imports."""
    output = tmp_path / "must-not-exist"

    completed = _command(*arguments, "--output-root", str(output))

    assert completed.returncode == 2
    assert not output.exists()


@pytest.mark.parametrize(
    ("verdict", "expected_exit"),
    (("PROVEN", 0), ("CONDITIONAL", 0), ("NOT_PROVEN", 0), ("INVALID_EVIDENCE", 1)),
)
def test_run_scientific_exit_mapping_and_canonical_json(
    verdict: str,
    expected_exit: int,
) -> None:
    """A negative scientific result is valid process evidence unless evidence is invalid."""
    finding = "evidence.integrity" if verdict == "INVALID_EVIDENCE" else "perf.speedup_proven"
    source = f"""
module.run_catalog_action = lambda *args, **kwargs: NS(
    bundle_path=Path('artifacts/runs/child'),
    verdict={verdict!r},
    driving_finding={finding!r},
)
"""

    completed = _injected(
        ["run", "mlp_stack", "--config", "candidate-mlp-default"],
        "hephaestus.workflows",
        source,
    )

    expected = {
        "boundary": EVIDENCE_BOUNDARY,
        "driving_finding": finding,
        "evidence_path": "artifacts/runs/child",
        "verdict": verdict,
    }
    assert completed.returncode == expected_exit, completed.stderr
    assert completed.stdout == json.dumps(expected, separators=(",", ":"), sort_keys=True) + "\n"


@pytest.mark.parametrize(
    ("verdict", "expected_exit"),
    (("NOT_PROVEN", 0), ("INVALID_EVIDENCE", 1)),
)
def test_agent_preserves_ordered_receipts_and_exit_mapping(
    verdict: str,
    expected_exit: int,
) -> None:
    """A valid exhausted search exits zero, while any invalid child exits one."""
    source = f"""
proposal = NS(
    catalog_id='candidate-mlp-default', workload_name=NS(value='mlp_stack'),
    rationale='Try it.',
)
receipt = NS(
    bundle_relative_path='runs/child', verdict={verdict!r},
    driving_finding='perf.speedup_proven',
)
step = NS(proposal=proposal, result=receipt)
transcript = NS(steps=(step,), final_result=None)
module.run_scripted_search = lambda *args, **kwargs: NS(
    parent_path=Path('artifacts/agent-search'), transcript=transcript,
)
"""

    completed = _injected(
        ["agent", "optimize", "mlp_stack"],
        "hephaestus.search",
        source,
    )
    payload = json.loads(completed.stdout)

    assert completed.returncode == expected_exit, completed.stderr
    assert payload["boundary"] == EVIDENCE_BOUNDARY
    assert payload["verdict"] == verdict
    assert payload["evidence_path"] == "artifacts/agent-search"
    assert payload["receipts"] == [
        {
            "proposal": {
                "catalog_id": "candidate-mlp-default",
                "workload_name": "mlp_stack",
                "rationale": "Try it.",
            },
            "result": {
                "bundle_relative_path": "runs/child",
                "verdict": verdict,
                "driving_finding": "perf.speedup_proven",
            },
        }
    ]
    assert completed.stdout == json.dumps(payload, separators=(",", ":"), sort_keys=True) + "\n"


def test_agent_exits_one_when_an_earlier_receipt_is_invalid_before_proven() -> None:
    """A later PROVEN result cannot erase invalid evidence already used by the search."""
    source = """
first_proposal = NS(
    catalog_id='candidate-dynamic-static',
    workload_name=NS(value='dynamic_batch_text'), rationale='First.',
)
second_proposal = NS(
    catalog_id='candidate-dynamic-true',
    workload_name=NS(value='dynamic_batch_text'), rationale='Second.',
)
first_result = NS(
    bundle_relative_path='runs/first', verdict='INVALID_EVIDENCE',
    driving_finding='evidence.integrity',
)
second_result = NS(
    bundle_relative_path='runs/second', verdict='PROVEN',
    driving_finding='all_criteria_passed',
)
steps = (
    NS(proposal=first_proposal, result=first_result),
    NS(proposal=second_proposal, result=second_result),
)
transcript = NS(steps=steps, final_result=second_result)
module.run_scripted_search = lambda *args, **kwargs: NS(
    parent_path=Path('artifacts/agent-search'), transcript=transcript,
)
"""

    completed = _injected(
        ["agent", "optimize", "dynamic_batch_text"],
        "hephaestus.search",
        source,
    )

    assert completed.returncode == 1
    payload = json.loads(completed.stdout)
    assert [receipt["result"]["verdict"] for receipt in payload["receipts"]] == [
        "INVALID_EVIDENCE",
        "PROVEN",
    ]
    assert payload["verdict"] == "PROVEN"


def test_trusted_workflow_exception_maps_to_runtime_failure_exit_one() -> None:
    """An execution exception must become bounded process failure with the scope visible."""
    source = """
def fail(*args, **kwargs):
    raise RuntimeError('compiler failed')
module.run_catalog_action = fail
"""

    completed = _injected(
        ["run", "mlp_stack", "--config", "candidate-mlp-default"],
        "hephaestus.workflows",
        source,
    )
    payload = json.loads(completed.stdout)

    assert completed.returncode == 1
    assert payload == {
        "boundary": EVIDENCE_BOUNDARY,
        "driving_finding": "runtime.failure",
        "evidence_path": None,
        "error": "compiler failed",
        "verdict": "INVALID_EVIDENCE",
    }


@pytest.mark.parametrize(
    ("verdict", "finding", "expected_exit"),
    (
        ("PASS", "all_criteria_passed", 0),
        ("FAIL", "methodology.noise_floor", 1),
        ("INVALID_EVIDENCE", "child.invalid_evidence", 1),
    ),
)
def test_aa_exit_mapping_and_json_fields(
    verdict: str,
    finding: str,
    expected_exit: int,
    tmp_path: Path,
) -> None:
    """Only a complete PASS is a successful A/A process outcome."""
    output_root = tmp_path / "output"
    parent = output_root / "aa-test"
    source = f"""
parent = Path({str(parent)!r})
parent.mkdir(parents=True)
statistics = NS(
    signed_effects=(0.0,), bootstrap_absolute_medians=(0.0,), p95_noise_floor=0.01,
    speedup_lower_bound_a_over_b=1.0, speedup_lower_bound_b_over_a=1.0,
)
module.run_aa_test = lambda *args, **kwargs: NS(
    parent_path=parent, verdict={verdict!r},
    driving_finding={finding!r}, statistics=statistics,
)
"""

    completed = _injected(
        [
            "aa-test", "mlp_stack", "--output-root", str(output_root),
            "--allow-volatile-output",
        ],
        "hephaestus.aa_runtime",
        source,
    )
    payload = json.loads(completed.stdout)

    assert completed.returncode == expected_exit, completed.stderr
    assert payload["boundary"] == EVIDENCE_BOUNDARY
    assert payload["verdict"] == verdict
    assert payload["driving_finding"] == finding
    assert payload["evidence_path"] == str(parent.resolve())
    assert payload["statistics"]["p95_noise_floor"] == 0.01


@pytest.mark.parametrize(("passed", "expected_exit"), ((True, 0), (False, 1)))
def test_demo_prints_boundary_and_deterministic_table(
    passed: bool,
    expected_exit: int,
    tmp_path: Path,
) -> None:
    """The four named expected/actual outcomes must be visible in stable row order."""
    output_root = tmp_path / "output"
    parent = output_root / "demo"
    source = f"""
parent = Path({str(parent)!r})
parent.mkdir(parents=True)
row = NS(
    catalog_id='planted-eager-fallback', expected_verdict='NOT_PROVEN',
    expected_driving_finding='perf.speedup_proven', actual_verdict='NOT_PROVEN',
    actual_driving_finding='perf.speedup_proven', passed={passed!r},
    bundle_relative_path='runs/child-1',
)
(parent / 'runs' / 'child-1').mkdir(parents=True)
(parent / 'runs' / 'child-1' / 'methodology.json').write_text(
    '{{"aa_noise_floor":0.0}}', encoding='utf-8'
)
module.run_planted_demo = lambda *args, **kwargs: NS(
    parent_path=parent, rows=(row,), passed={passed!r},
)
"""

    completed = _injected(
        [
            "demo-planted-regressions", "--output-root", str(output_root),
            "--allow-volatile-output",
        ],
        "hephaestus.demo",
        source,
    )

    assert completed.returncode == expected_exit, completed.stderr
    assert completed.stdout.splitlines() == [
        EVIDENCE_BOUNDARY,
        "catalog_id | expected_verdict | expected_finding | actual_verdict | "
        "actual_finding | pass | bundle",
        "planted-eager-fallback | NOT_PROVEN | perf.speedup_proven | "
        "NOT_PROVEN | perf.speedup_proven | "
        f"{'yes' if passed else 'no'} | runs/child-1",
        f"evidence_path: {parent.resolve()}",
    ]


def test_gate_is_lazy_read_only_byte_deterministic_and_includes_complete_verdict(
    tmp_path: Path,
) -> None:
    """Offline gating must perform zero writes and expose every named finding deterministically."""
    bundle = tmp_path / "bundle"
    _write_bundle(bundle)
    before = {
        path.relative_to(bundle).as_posix(): path.read_bytes()
        for path in bundle.rglob("*")
        if path.is_file()
    }
    code = f"""
import sys
from hephaestus.cli import main
result = main(['gate', {str(bundle)!r}])
assert 'torch' not in sys.modules
raise SystemExit(result)
"""

    first = subprocess.run(
        [str(PYTHON), "-c", code], cwd=ROOT, text=True, capture_output=True, check=False
    )
    second = subprocess.run(
        [str(PYTHON), "-c", code], cwd=ROOT, text=True, capture_output=True, check=False
    )
    after = {
        path.relative_to(bundle).as_posix(): path.read_bytes()
        for path in bundle.rglob("*")
        if path.is_file()
    }
    payload = json.loads(first.stdout)

    assert first.returncode == second.returncode == 0
    assert first.stdout == second.stdout
    assert before == after
    assert payload["boundary"] == EVIDENCE_BOUNDARY
    assert payload["evidence_path"] == str(bundle)
    assert payload["complete_verdict"]["verdict"] == payload["verdict"]
    assert payload["complete_verdict"]["findings"]


@pytest.mark.parametrize("tamper", ("changed", "missing", "unexpected", "symlink"))
def test_gate_integrity_mismatch_is_visible_and_exits_one(
    tmp_path: Path,
    tamper: str,
) -> None:
    """Changed, missing, unexpected, and symlink payloads must remain named in CLI output."""
    bundle = tmp_path / "bundle"
    _write_bundle(bundle)
    expected_fragment = {
        "changed": "changed:timings.json",
        "missing": "missing:config.json",
        "unexpected": "unexpected:extra.json",
        "symlink": "symlink:linked.json",
    }[tamper]
    if tamper == "changed":
        write_json(bundle / "timings.json", {"forged": True})
    elif tamper == "missing":
        (bundle / "config.json").unlink()
    elif tamper == "unexpected":
        write_json(bundle / "extra.json", {"forged": True})
    else:
        (bundle / "linked.json").symlink_to(bundle / "config.json")

    completed = _command("gate", str(bundle))

    assert completed.returncode == 1
    payload = json.loads(completed.stdout)
    assert payload["verdict"] == "INVALID_EVIDENCE"
    assert expected_fragment in payload["complete_verdict"]["findings"][0]["mismatches"]


@pytest.mark.slow
def test_real_subprocess_run_smoke_uses_compiler_and_writes_bundle(tmp_path: Path) -> None:
    """The installed entry point must execute one authentic calibrated compiler run."""
    completed = _command(
        "run",
        "mlp_stack",
        "--config",
        "candidate-mlp-default",
        "--output-root",
        str(tmp_path),
    )

    assert completed.returncode == 0, completed.stderr
    payload = json.loads(completed.stdout)
    assert payload["boundary"] == EVIDENCE_BOUNDARY
    assert payload["verdict"] in {"PROVEN", "CONDITIONAL", "NOT_PROVEN"}
    assert Path(payload["evidence_path"]).is_dir()
    assert Path(payload["evidence_path"], "manifest.json").is_file()

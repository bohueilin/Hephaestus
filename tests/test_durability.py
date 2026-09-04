from __future__ import annotations

import builtins
import json
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import uuid
import warnings
from collections.abc import Callable
from contextlib import nullcontext, redirect_stderr, redirect_stdout, suppress
from datetime import UTC, datetime
from io import BytesIO, StringIO
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

from hephaestus.durability import append_attempt_record, prepare_attempt_output_root

ROOT = Path(__file__).parents[1]
PYTHON = ROOT / ".venv" / "bin" / "python"
HEPHAESTUS = ROOT / ".venv" / "bin" / "hephaestus"
SCIENCE_START_MARKER = b"hephaestus-durable-worker:science-start:v1\n"


class _FakeWorkerProcess:
    def __init__(self, stdout: bytes, stderr: bytes, returncode: int) -> None:
        self.stdout = BytesIO(stdout)
        self.stderr = BytesIO(stderr)
        self._returncode = returncode

    def wait(self) -> int:
        return self._returncode


class _ChunkedBinaryStream:
    def __init__(self, *chunks: bytes) -> None:
        self._chunks = iter(chunks)

    def read(self, size: int = -1) -> bytes:
        del size
        return next(self._chunks, b"")

    def close(self) -> None:
        return None


class _CountingTextSink:
    def __init__(self, expected_character: str) -> None:
        self.expected_character = expected_character
        self.total = 0
        self.max_write = 0
        self.flushes = 0

    def write(self, value: str) -> int:
        assert value and set(value) == {self.expected_character}
        self.total += len(value)
        self.max_write = max(self.max_write, len(value))
        return len(value)

    def flush(self) -> None:
        self.flushes += 1


class _ShortTextSink:
    def __init__(self, maximum_write: int = 2) -> None:
        self.maximum_write = maximum_write
        self.value = ""
        self.write_calls = 0
        self.flushes = 0

    def write(self, value: str) -> int:
        accepted = value[: self.maximum_write]
        self.value += accepted
        self.write_calls += 1
        return len(accepted)

    def flush(self) -> None:
        self.flushes += 1


class _ShortBinarySink(BytesIO):
    def __init__(self, maximum_write: int = 2) -> None:
        super().__init__()
        self.maximum_write = maximum_write
        self.write_calls = 0
        self.flushes = 0

    def write(self, value: bytes) -> int:
        assert isinstance(value, bytes)
        self.write_calls += 1
        return super().write(value[: self.maximum_write])

    def flush(self) -> None:
        self.flushes += 1
        super().flush()


def _fake_worker_start(
    process: _FakeWorkerProcess,
    *,
    control: bytes = SCIENCE_START_MARKER,
    passed_descriptors: list[int] | None = None,
) -> Callable[..., _FakeWorkerProcess]:
    def started(*args: object, **kwargs: object) -> _FakeWorkerProcess:
        del args
        descriptors = kwargs.get("pass_fds", ())
        if descriptors:
            descriptor = descriptors[0]
            if passed_descriptors is not None:
                passed_descriptors.append(descriptor)
            if control:
                os.write(descriptor, control)
        return process

    return started


def _run_direct_worker(
    arguments: list[str],
    *,
    environment: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
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


def _command(*arguments: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [str(HEPHAESTUS), *arguments],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )


def _injected(
    arguments: list[str], module_name: str, source: str
) -> subprocess.CompletedProcess[str]:
    code = f"""
import sys, types
from pathlib import Path
from types import SimpleNamespace as NS
module = types.ModuleType({module_name!r})
{source}
sys.modules[{module_name!r}] = module
"""
    with tempfile.TemporaryDirectory() as temporary:
        Path(temporary, "sitecustomize.py").write_text(code, encoding="utf-8")
        environment = os.environ.copy()
        existing = environment.get("PYTHONPATH")
        environment["PYTHONPATH"] = (
            temporary if not existing else f"{temporary}{os.pathsep}{existing}"
        )
        return _run_direct_worker(arguments, environment=environment)


def _sitecustomized(
    arguments: list[str], source: str
) -> subprocess.CompletedProcess[str]:
    with tempfile.TemporaryDirectory() as temporary:
        Path(temporary, "sitecustomize.py").write_text(source, encoding="utf-8")
        environment = os.environ.copy()
        existing = environment.get("PYTHONPATH")
        environment["PYTHONPATH"] = (
            temporary if not existing else f"{temporary}{os.pathsep}{existing}"
        )
        return subprocess.run(
            [str(HEPHAESTUS), *arguments],
            cwd=ROOT,
            env=environment,
            text=True,
            capture_output=True,
            check=False,
        )


@pytest.mark.parametrize(
    "kind", ("exact", "descendant", "normalized")
)
@pytest.mark.parametrize(
    "command", (("aa-test", "mlp_stack"), ("demo-planted-regressions",))
)
def test_volatile_roots_are_refused_before_creating_output(
    kind: str, command: tuple[str, ...]
) -> None:
    """Removing either CLI safety gate would allow a volatile attempt to start."""
    temporary = Path(tempfile.gettempdir())
    leaf = f"hephaestus-durability-{uuid.uuid4().hex}"
    output_root = {
        "exact": temporary,
        "descendant": temporary / leaf,
        "normalized": temporary / "." / leaf / ".." / leaf,
    }[kind]
    resolved = output_root.resolve(strict=False)

    completed = _command(*command, "--output-root", str(output_root))

    assert completed.returncode == 2
    assert "--allow-volatile-output" in completed.stderr
    assert not resolved.exists() or resolved == temporary.resolve()
    if resolved != temporary.resolve():
        assert not (resolved / "attempts.jsonl").exists()


def test_symlinked_temp_aa_root_is_refused_before_creating_target(tmp_path: Path) -> None:
    """Checking the raw spelling instead of the resolved path would bypass the safety gate."""
    link = tmp_path / "temp-link"
    link.symlink_to(Path(tempfile.gettempdir()), target_is_directory=True)
    target = Path(tempfile.gettempdir()) / f"hephaestus-durability-{uuid.uuid4().hex}"

    completed = _command("aa-test", "mlp_stack", "--output-root", str(link / target.name))

    assert completed.returncode == 2
    assert "--allow-volatile-output" in completed.stderr
    assert not target.exists()
    assert not (target / "attempts.jsonl").exists()


@pytest.mark.parametrize(
    ("verdict", "expected_exit"), (("PASS", 0), ("FAIL", 1)))
def test_aa_completed_attempt_prints_absolute_path_and_appends_canonical_ledger(
    tmp_path: Path, verdict: str, expected_exit: int
) -> None:
    """Dropping finalized-attempt recording would lose a passing or failed A/A evidence root."""
    output_root = tmp_path / "volatile-root"
    parent = output_root / f"aa-{verdict.lower()}"
    source = f"""
parent = Path({str(parent)!r})
parent.mkdir(parents=True, exist_ok=True)
statistics = NS(
    signed_effects=(0.0,), bootstrap_absolute_medians=(0.0,), p95_noise_floor=0.125,
    speedup_lower_bound_a_over_b=1.0, speedup_lower_bound_b_over_a=1.0,
)
module.run_aa_test = lambda *args, **kwargs: NS(
    parent_path=parent, verdict={verdict!r}, driving_finding='methodology.noise_floor',
    statistics=statistics,
)
"""

    completed = _injected(
        [
            "aa-test",
            "mlp_stack",
            "--output-root",
            str(output_root),
            "--allow-volatile-output",
        ],
        "hephaestus.aa_runtime",
        source,
    )

    assert completed.returncode == expected_exit, completed.stderr
    payload = json.loads(completed.stdout)
    assert payload["evidence_path"] == str(parent.resolve())
    records = (output_root / "attempts.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(records) == 1
    record = json.loads(records[0])
    assert list(record) == [
        "clean_control_floor",
        "parent_path",
        "timestamp_utc",
        "verdict",
        "workload",
    ]
    assert record["workload"] == "mlp_stack"
    assert record["verdict"] == verdict
    assert record["clean_control_floor"] == 0.125
    assert record["parent_path"] == str(parent.resolve())
    assert record["timestamp_utc"].endswith("Z")


def test_explicit_override_allows_demo_and_appends_without_rewriting_prior_records(
    tmp_path: Path,
) -> None:
    """Replacing rather than appending the root ledger would erase an earlier demo attempt."""
    output_root = tmp_path / "volatile-root"
    parent = output_root / "demo-parent"
    source = f"""
parent = Path({str(parent)!r})
parent.mkdir(parents=True, exist_ok=True)
row = NS(
    catalog_id='clean-control-mlp', expected_verdict='PROVEN',
    expected_driving_finding='all_criteria_passed', actual_verdict='PROVEN',
    actual_driving_finding='all_criteria_passed', passed=True,
    bundle_relative_path='runs/clean',
)
(parent / 'runs' / 'clean').mkdir(parents=True, exist_ok=True)
(parent / 'runs' / 'clean' / 'methodology.json').write_text(
    '{{"aa_noise_floor":0.25}}', encoding='utf-8'
)
module.run_planted_demo = lambda *args, **kwargs: NS(
    parent_path=parent, rows=(row,), passed=False,
)
"""
    arguments = [
        "demo-planted-regressions",
        "--output-root",
        str(output_root),
        "--allow-volatile-output",
    ]

    first = _injected(arguments, "hephaestus.demo", source)
    before = (output_root / "attempts.jsonl").read_bytes()
    second = _injected(arguments, "hephaestus.demo", source)
    records = (output_root / "attempts.jsonl").read_text(encoding="utf-8").splitlines()

    assert first.returncode == second.returncode == 1
    assert f"evidence_path: {parent.resolve()}" in second.stdout
    assert (output_root / "attempts.jsonl").read_bytes().startswith(before)
    assert len(records) == 2
    assert all(json.loads(line)["clean_control_floor"] == 0.25 for line in records)
    assert all(json.loads(line)["verdict"] == "FAIL" for line in records)


@pytest.mark.parametrize("floor", (float("nan"), float("inf"), float("-inf")))
def test_attempt_ledger_refuses_nonfinite_clean_control_floors(
    tmp_path: Path, floor: float
) -> None:
    """Serializing a nonfinite floor would make the canonical JSONL evidence invalid."""
    parent = tmp_path / "parent"
    parent.mkdir()

    with pytest.raises(ValueError, match="finite"):
        append_attempt_record(
            tmp_path / "root",
            workload="mlp_stack",
            verdict="PASS",
            clean_control_floor=floor,
            parent_path=parent,
        )

    assert (tmp_path / "root" / "attempts.jsonl").read_bytes() == b""


@pytest.mark.parametrize("kind", ("symlink", "hardlink"))
def test_attempt_ledger_cannot_mutate_a_finalized_parent_via_link(
    tmp_path: Path, kind: str
) -> None:
    """Following a ledger link would let an append corrupt finalized evidence."""
    root = tmp_path / "root"
    parent = root / "finalized-parent"
    parent.mkdir(parents=True)
    manifest = parent / "manifest.json"
    manifest.write_bytes(b'{"files":{}}\n')
    before = {
        path.relative_to(parent).as_posix(): path.read_bytes()
        for path in parent.rglob("*")
        if path.is_file()
    }
    ledger = root / "attempts.jsonl"
    if kind == "symlink":
        ledger.symlink_to(manifest)
    else:
        os.link(manifest, ledger)

    with pytest.raises(ValueError, match="ledger"):
        append_attempt_record(
            root,
            workload="mlp_stack",
            verdict="PASS",
            clean_control_floor=0.125,
            parent_path=parent,
        )

    after = {
        path.relative_to(parent).as_posix(): path.read_bytes()
        for path in parent.rglob("*")
        if path.is_file()
    }
    assert after == before


def test_pinned_output_root_keeps_science_and_ledger_on_opened_directory(
    tmp_path: Path,
) -> None:
    """The disposable worker stays on its open root after the public path is swapped."""
    root = tmp_path / "output"
    root.mkdir()
    moved = tmp_path / "moved"
    replacement = tmp_path / "replacement"
    replacement.mkdir()
    code = f"""
from pathlib import Path
from hephaestus.durability import prepare_attempt_output_root
root = Path({str(root)!r})
moved = Path({str(moved)!r})
replacement = Path({str(replacement)!r})
with prepare_attempt_output_root(root, allow_volatile_output=True) as attempt_root:
    root.rename(moved)
    root.symlink_to(replacement, target_is_directory=True)
    attempt_root.enter_worker_directory()
    parent = Path('finalized-parent')
    parent.mkdir()
    attempt_root.append_attempt(
        workload='mlp_stack', verdict='PASS', clean_control_floor=0.125, parent_path=parent,
    )
"""
    completed = subprocess.run(
        [str(PYTHON), "-c", code], cwd=ROOT, text=True, capture_output=True, check=False
    )

    assert completed.returncode == 0, completed.stderr

    assert (moved / "finalized-parent").is_dir()
    assert (moved / "attempts.jsonl").is_file()
    assert not (replacement / "finalized-parent").exists()
    assert not (replacement / "attempts.jsonl").exists()


@pytest.mark.parametrize(
    (
        "command",
        "orchestrator_module",
        "orchestrator_name",
        "helper_module",
        "runner_name",
        "parent_glob",
    ),
    (
        (
            ("aa-test", "mlp_stack"),
            "hephaestus.aa_runtime",
            "TrustedAAOrchestrator",
            "tests.test_aa_runtime",
            "_AARunner",
            "aa-test-*",
        ),
        (
            ("demo-planted-regressions",),
            "hephaestus.demo",
            "TrustedDemoOrchestrator",
            "tests.test_workflows",
            "_EvidenceRunner",
            "planted-demo-*",
        ),
    ),
)
def test_worker_orchestrators_keep_science_and_ledger_on_the_pinned_inode(
    tmp_path: Path,
    command: tuple[str, ...],
    orchestrator_module: str,
    orchestrator_name: str,
    helper_module: str,
    runner_name: str,
    parent_glob: str,
) -> None:
    """Path normalization at either orchestration layer must not discard the cwd capability."""
    root = tmp_path / "output"
    first_moved = tmp_path / "first-moved"
    pinned = tmp_path / "pinned"
    first_replacement = tmp_path / "first-replacement"
    second_replacement = tmp_path / "second-replacement"
    root.mkdir()
    first_replacement.mkdir()
    second_replacement.mkdir()
    source = f"""
import importlib
import sys
from pathlib import Path

sys.path[:0] = [{str(ROOT / "src")!r}, {str(ROOT)!r}]
root = Path({str(root)!r})
first_moved = Path({str(first_moved)!r})
pinned = Path({str(pinned)!r})
first_replacement = Path({str(first_replacement)!r})
second_replacement = Path({str(second_replacement)!r})
orchestrator_module = importlib.import_module({orchestrator_module!r})
helper_module = importlib.import_module({helper_module!r})
orchestrator = getattr(orchestrator_module, {orchestrator_name!r})
inner_runner = getattr(helper_module, {runner_name!r})()

class SwapAfterCatalogConstruction:
    def __init__(self):
        self.swapped = False

    def __call__(self, *args, **kwargs):
        if not self.swapped:
            first_moved.rename(pinned)
            first_moved.symlink_to(second_replacement, target_is_directory=True)
            self.swapped = True
        return inner_runner(*args, **kwargs)

runner = SwapAfterCatalogConstruction()
original_init = orchestrator.__init__

def init_then_replace_root(self, *args, **kwargs):
    kwargs['capability_snapshot'] = helper_module.CAPABILITIES
    kwargs['runner'] = runner
    original_init(self, *args, **kwargs)
    root.rename(first_moved)
    root.symlink_to(first_replacement, target_is_directory=True)

orchestrator.__init__ = init_then_replace_root
"""

    completed = _sitecustomized(
        [
            *command,
            "--output-root",
            str(root),
            "--allow-volatile-output",
        ],
        source,
    )

    assert completed.returncode == 0, completed.stderr
    assert not tuple(first_replacement.iterdir())
    assert not tuple(second_replacement.iterdir())
    parents = tuple(pinned.glob(parent_glob))
    assert len(parents) == 1, {
        "stdout": completed.stdout,
        "stderr": completed.stderr,
        "tmp_entries": tuple(path.name for path in tmp_path.iterdir()),
        "pinned_entries": (
            tuple(path.name for path in pinned.iterdir()) if pinned.is_dir() else None
        ),
        "first_moved_entries": (
            tuple(path.name for path in first_moved.iterdir())
            if first_moved.is_dir()
            else None
        ),
    }
    assert (parents[0] / "manifest.json").is_file()
    record = json.loads((pinned / "attempts.jsonl").read_bytes())
    assert record["parent_path"] == str(parents[0])


@pytest.mark.parametrize("failed_index", (0, 1, 2))
def test_closing_attempt_root_reports_one_failure_without_skipping_later_descriptors(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failed_index: int,
) -> None:
    """A close error after science must not replace the verdict or abandon another descriptor."""
    import hephaestus.durability as durability

    root = tmp_path / "root"
    root.mkdir()
    attempt_root = prepare_attempt_output_root(root, allow_volatile_output=True)
    attempt_root.__enter__()
    descriptors = (
        attempt_root._host_ledger_fd,
        attempt_root._ledger_fd,
        attempt_root._root_fd,
    )
    assert all(descriptor is not None for descriptor in descriptors)
    failed_descriptor = descriptors[failed_index]
    calls: list[int] = []
    real_close_descriptor = durability._close_descriptor

    def one_failure(descriptor: int) -> bool:
        calls.append(descriptor)
        if descriptor == failed_descriptor:
            return False
        return real_close_descriptor(descriptor)

    monkeypatch.setattr(durability, "_close_descriptor", one_failure)
    failures = attempt_root.close()

    assert attempt_root._host_ledger_fd is None
    assert attempt_root._ledger_fd is None
    assert attempt_root._root_fd is None
    assert calls == list(descriptors)
    assert failures == (failed_descriptor,)
    assert failed_descriptor is not None
    os.close(failed_descriptor)


def test_after_open_failure_closes_all_three_attempt_root_descriptors(
    tmp_path: Path,
) -> None:
    """A preflight hook exception cannot leak the host, attempts, or root descriptor."""
    opened: list[int] = []
    failure = RuntimeError("after-open failure")
    attempt_root = None

    def fail_after_open() -> None:
        assert attempt_root is not None
        opened.extend(
            (
                attempt_root._host_ledger_fd,
                attempt_root._ledger_fd,
                attempt_root._root_fd,
            )
        )
        raise failure

    attempt_root = prepare_attempt_output_root(
        tmp_path / "root",
        allow_volatile_output=True,
        _after_open=fail_after_open,
    )
    try:
        with pytest.raises(RuntimeError) as caught:
            attempt_root.__enter__()

        assert caught.value is failure
        assert attempt_root._host_ledger_fd is None
        assert attempt_root._ledger_fd is None
        assert attempt_root._root_fd is None
        assert len(opened) == 3 and all(isinstance(item, int) for item in opened)
        for descriptor in opened:
            with pytest.raises(OSError):
                os.fstat(descriptor)
    finally:
        for descriptor in opened:
            with suppress(OSError):
                os.close(descriptor)


def test_invalid_ledger_is_rejected_before_the_injected_aa_runner(tmp_path: Path) -> None:
    """A bad ledger must be an argument boundary failure, not a post-science runtime error."""
    output_root = tmp_path / "output"
    output_root.mkdir()
    protected = tmp_path / "protected.txt"
    protected.write_text("unchanged", encoding="utf-8")
    (output_root / "attempts.jsonl").symlink_to(protected)
    source = (
        "module.run_aa_test = lambda *args, **kwargs: "
        "(_ for _ in ()).throw(AssertionError('runner invoked'))"
    )

    completed = _injected(
        [
            "aa-test",
            "mlp_stack",
            "--output-root",
            str(output_root),
            "--allow-volatile-output",
        ],
        "hephaestus.aa_runtime",
        source,
    )

    assert completed.returncode == 2
    assert "ledger" in completed.stderr
    assert "runner invoked" not in completed.stdout
    assert protected.read_text(encoding="utf-8") == "unchanged"


def test_malformed_ledger_is_rejected_before_the_injected_aa_runner(tmp_path: Path) -> None:
    """Canonical prior receipts are a pre-science boundary, not an append-time detail."""
    output_root = tmp_path / "output"
    output_root.mkdir()
    (output_root / "attempts.jsonl").write_bytes(b'{"workload":"mlp_stack"}\n')
    source = (
        "module.run_aa_test = lambda *args, **kwargs: "
        "(_ for _ in ()).throw(AssertionError('runner invoked'))"
    )

    completed = _injected(
        [
            "aa-test",
            "mlp_stack",
            "--output-root",
            str(output_root),
            "--allow-volatile-output",
        ],
        "hephaestus.aa_runtime",
        source,
    )

    assert completed.returncode == 2
    assert "existing records" in completed.stderr
    assert "runner invoked" not in completed.stdout


def test_persistent_worker_close_errors_preserve_completed_science_output(
    tmp_path: Path,
) -> None:
    """The disposable worker must retain a completed verdict even if both closes fail."""
    output_root = tmp_path / "output"
    parent = output_root / "parent"
    code = f"""
import sys, types
from pathlib import Path
from types import SimpleNamespace as NS
import tempfile
import hephaestus.durability as durability
tempfile.gettempdir()
durability._close_descriptor = lambda descriptor: False
module = types.ModuleType('hephaestus.aa_runtime')
parent = Path({str(parent)!r})
parent.mkdir(parents=True)
statistics = NS(
    signed_effects=(0.0,), bootstrap_absolute_medians=(0.0,), p95_noise_floor=0.125,
    speedup_lower_bound_a_over_b=1.0, speedup_lower_bound_b_over_a=1.0,
)
module.run_aa_test = lambda *args, **kwargs: NS(
    parent_path=parent, verdict='FAIL', driving_finding='methodology.noise_floor',
    statistics=statistics,
)
sys.modules['hephaestus.aa_runtime'] = module
"""
    with tempfile.TemporaryDirectory() as temporary:
        Path(temporary, "sitecustomize.py").write_text(code, encoding="utf-8")
        environment = os.environ.copy()
        existing = environment.get("PYTHONPATH")
        environment["PYTHONPATH"] = (
            temporary if not existing else f"{temporary}{os.pathsep}{existing}"
        )
        completed = _run_direct_worker(
            [
                "aa-test",
                "mlp_stack",
                "--output-root",
                str(output_root),
                "--allow-volatile-output",
            ],
            environment=environment,
        )

    assert completed.returncode == 1
    assert completed.stdout, completed.stderr
    payload = json.loads(completed.stdout)
    assert payload["verdict"] == "FAIL"
    assert payload["evidence_path"] == str(parent.resolve())
    assert "cleanup error" in completed.stderr


@pytest.mark.parametrize(
    "arguments",
    (
        ("run", "mlp_stack", "--config", "candidate-mlp-default"),
        ("agent", "optimize", "mlp_stack"),
        ("gate", "missing-bundle"),
    ),
)
def test_volatile_override_flag_is_exclusive_to_durable_attempt_commands(
    arguments: tuple[str, ...],
) -> None:
    """Accepting the override elsewhere would silently broaden the five-command authority."""
    completed = _command(*arguments, "--allow-volatile-output")

    assert completed.returncode == 2
    assert "unrecognized arguments: --allow-volatile-output" in completed.stderr


def test_real_ledger_append_leaves_finalized_parent_bytes_unchanged_and_uses_strict_utc(
    tmp_path: Path,
) -> None:
    """A root-level receipt must neither alter a finalized tree nor emit a local timestamp."""
    root = tmp_path / "root"
    parent = root / "finalized-parent"
    parent.mkdir(parents=True)
    (parent / "manifest.json").write_bytes(b'{"files":{}}\n')
    before = {
        path.relative_to(parent).as_posix(): path.read_bytes()
        for path in parent.rglob("*")
        if path.is_file()
    }

    append_attempt_record(
        root,
        workload="mlp_stack",
        verdict="PASS",
        clean_control_floor=0.125,
        parent_path=parent,
    )

    after = {
        path.relative_to(parent).as_posix(): path.read_bytes()
        for path in parent.rglob("*")
        if path.is_file()
    }
    record = json.loads((root / "attempts.jsonl").read_text(encoding="utf-8"))
    timestamp = datetime.fromisoformat(record["timestamp_utc"].replace("Z", "+00:00"))
    assert after == before
    assert record["timestamp_utc"].endswith("Z")
    assert timestamp.tzinfo == UTC


def test_attempt_ledger_rejects_missing_and_escaping_parent_paths(tmp_path: Path) -> None:
    """Recording a nonexistent or external parent would create a false durable receipt."""
    root = tmp_path / "root"
    root.mkdir()
    external = tmp_path / "external"
    external.mkdir()

    for parent in (root / "missing", external):
        with pytest.raises(ValueError, match="parent"):
            append_attempt_record(
                root,
                workload="mlp_stack",
                verdict="PASS",
                clean_control_floor=0.125,
                parent_path=parent,
            )

    assert (root / "attempts.jsonl").read_bytes() == b""


def test_attempt_ledger_rejects_an_external_parent_hidden_by_an_intermediate_symlink(
    tmp_path: Path,
) -> None:
    """Lexical containment alone would record a parent reached through an external link."""
    root = tmp_path / "root"
    root.mkdir()
    external = tmp_path / "external"
    parent = external / "parent"
    parent.mkdir(parents=True)
    marker = parent / "manifest.json"
    marker.write_bytes(b'{"files":{}}\n')
    before = marker.read_bytes()
    (root / "linked").symlink_to(external, target_is_directory=True)

    with pytest.raises(ValueError, match="parent"):
        append_attempt_record(
            root,
            workload="mlp_stack",
            verdict="PASS",
            clean_control_floor=0.125,
            parent_path=root / "linked" / "parent",
        )

    assert marker.read_bytes() == before
    assert (root / "attempts.jsonl").read_bytes() == b""


@pytest.mark.parametrize("with_reader", (False, True))
def test_fifo_ledger_is_refused_without_blocking_science(
    tmp_path: Path, with_reader: bool
) -> None:
    """Opening a FIFO as an ordinary append target can block before the runner starts."""
    root = tmp_path / "root"
    parent = root / "parent"
    parent.mkdir(parents=True)
    fifo = root / "attempts.jsonl"
    os.mkfifo(fifo)
    reader: int | None = None
    if with_reader:
        reader = os.open(fifo, os.O_RDONLY | os.O_NONBLOCK)
    try:
        with pytest.raises(ValueError, match="ledger"):
            append_attempt_record(
                root,
                workload="mlp_stack",
                verdict="PASS",
                clean_control_floor=0.125,
                parent_path=parent,
            )
    finally:
        if reader is not None:
            os.close(reader)


def test_real_ledger_append_uses_one_complete_write_operation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Splitting the append would allow concurrent records to interleave on the ledger."""
    import hephaestus.durability as durability

    root = tmp_path / "root"
    parent = root / "parent"
    parent.mkdir(parents=True)
    writes: list[bytes] = []
    real_write = durability.os.write

    def recording_write(descriptor: int, payload: bytes) -> int:
        writes.append(payload)
        return real_write(descriptor, payload)

    monkeypatch.setattr(durability.os, "write", recording_write)
    append_attempt_record(
        root,
        workload="mlp_stack",
        verdict="PASS",
        clean_control_floor=0.125,
        parent_path=parent,
    )

    assert len(writes) == 1
    assert writes[0].endswith(b"\n")
    assert (root / "attempts.jsonl").read_bytes() == writes[0]


def test_short_ledger_write_rolls_back_the_partial_json_line(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A short append must not leave a fragment that makes subsequent evidence ambiguous."""
    import hephaestus.durability as durability

    root = tmp_path / "root"
    parent = root / "parent"
    parent.mkdir(parents=True)
    real_write = durability.os.write

    def short_write(descriptor: int, payload: bytes) -> int:
        real_write(descriptor, payload[:7])
        return 7

    monkeypatch.setattr(durability.os, "write", short_write)
    with pytest.raises(ValueError, match="incomplete"):
        append_attempt_record(
            root,
            workload="mlp_stack",
            verdict="PASS",
            clean_control_floor=0.125,
            parent_path=parent,
        )

    assert (root / "attempts.jsonl").read_bytes() == b""


def test_existing_malformed_ledger_is_refused_without_extension(tmp_path: Path) -> None:
    """Extending arbitrary JSONL would turn an unauditable prior ledger into evidence."""
    root = tmp_path / "root"
    parent = root / "parent"
    parent.mkdir(parents=True)
    ledger = root / "attempts.jsonl"
    ledger.write_bytes(b'{"workload":"mlp_stack"}\n')

    with pytest.raises(ValueError, match="existing records"):
        append_attempt_record(
            root,
            workload="mlp_stack",
            verdict="PASS",
            clean_control_floor=0.125,
            parent_path=parent,
        )

    assert ledger.read_bytes() == b'{"workload":"mlp_stack"}\n'


def test_pruned_historical_parent_does_not_block_a_new_valid_attempt(tmp_path: Path) -> None:
    """A removed old evidence directory is not proof that its receipt was malformed."""
    root = tmp_path / "root"
    historical_parent = root / "historical-parent"
    new_parent = root / "new-parent"
    historical_parent.mkdir(parents=True)

    append_attempt_record(
        root,
        workload="mlp_stack",
        verdict="PASS",
        clean_control_floor=0.125,
        parent_path=historical_parent,
    )
    ledger = root / "attempts.jsonl"
    historical_row = ledger.read_bytes()
    shutil.rmtree(historical_parent)
    new_parent.mkdir()

    append_attempt_record(
        root,
        workload="mlp_stack",
        verdict="PASS",
        clean_control_floor=0.125,
        parent_path=new_parent,
    )

    assert ledger.read_bytes().startswith(historical_row)
    assert len(ledger.read_bytes().splitlines()) == 2


def test_existing_canonical_record_parent_cannot_escape_through_a_symlink(
    tmp_path: Path,
) -> None:
    """A canonical historical spelling is not proof that its parent stayed under the root."""
    root = tmp_path / "root"
    external_parent = tmp_path / "external" / "parent"
    root.mkdir()
    external_parent.mkdir(parents=True)
    (root / "linked").symlink_to(external_parent.parent, target_is_directory=True)
    record = {
        "timestamp_utc": "2026-08-30T12:00:00.000000Z",
        "workload": "mlp_stack",
        "verdict": "PASS",
        "clean_control_floor": 0.125,
        "parent_path": str(root / "linked" / "parent"),
    }
    ledger = root / "attempts.jsonl"
    ledger.write_bytes(
        (
            json.dumps(record, ensure_ascii=True, separators=(",", ":"), sort_keys=True)
            + "\n"
        ).encode()
    )
    before = ledger.read_bytes()
    source = (
        "module.run_aa_test = lambda *args, **kwargs: "
        "(_ for _ in ()).throw(AssertionError('runner invoked'))"
    )

    completed = _injected(
        [
            "aa-test",
            "mlp_stack",
            "--output-root",
            str(root),
            "--allow-volatile-output",
        ],
        "hephaestus.aa_runtime",
        source,
    )

    assert completed.returncode == 2
    assert "parent path" in completed.stderr
    assert "runner invoked" not in completed.stdout
    assert ledger.read_bytes() == before


def test_existing_ledger_validates_multiple_records_without_a_total_size_cap(
    tmp_path: Path,
) -> None:
    """The safety bound applies to each receipt, not to append-only history as a whole."""
    root = tmp_path / "root"
    parents = (root / "first", root / "second", root / "new")
    for parent in parents:
        parent.mkdir(parents=True)

    def record(parent: Path, workload: str, timestamp: str) -> bytes:
        value = {
            "timestamp_utc": timestamp,
            "workload": workload,
            "verdict": "PASS",
            "clean_control_floor": 0.125,
            "parent_path": str(parent),
        }
        return (
            json.dumps(value, ensure_ascii=True, separators=(",", ":"), sort_keys=True) + "\n"
        ).encode()

    original = (
        record(parents[0], "a" * 40_000, "2026-08-30T12:00:00.000000Z")
        + record(parents[1], "b" * 40_000, "2026-08-30T12:01:00.000000Z")
    )
    ledger = root / "attempts.jsonl"
    ledger.write_bytes(original)

    append_attempt_record(
        root,
        workload="mlp_stack",
        verdict="PASS",
        clean_control_floor=0.125,
        parent_path=parents[2],
    )

    updated = ledger.read_bytes()
    assert updated.startswith(original)
    assert len(updated.splitlines()) == 3


def test_existing_ledger_rejects_one_oversized_record_without_extension(
    tmp_path: Path,
) -> None:
    """One adversarial JSON line must not force an unbounded allocation before science."""
    root = tmp_path / "root"
    parent = root / "parent"
    parent.mkdir(parents=True)
    record = {
        "timestamp_utc": "2026-08-30T12:00:00.000000Z",
        "workload": "x" * 70_000,
        "verdict": "PASS",
        "clean_control_floor": 0.125,
        "parent_path": str(parent),
    }
    ledger = root / "attempts.jsonl"
    ledger.write_bytes(
        (
            json.dumps(record, ensure_ascii=True, separators=(",", ":"), sort_keys=True)
            + "\n"
        ).encode()
    )
    before = ledger.read_bytes()

    with pytest.raises(ValueError, match="record exceeds"):
        append_attempt_record(
            root,
            workload="mlp_stack",
            verdict="PASS",
            clean_control_floor=0.125,
            parent_path=parent,
        )

    assert ledger.read_bytes() == before


def test_attempt_ledger_refuses_to_create_an_oversized_record(tmp_path: Path) -> None:
    """The writer must not create a receipt that the next preflight is required to reject."""
    root = tmp_path / "root"
    parent = root / "parent"
    parent.mkdir(parents=True)

    with pytest.raises(ValueError, match="record exceeds"):
        append_attempt_record(
            root,
            workload="x" * 70_000,
            verdict="PASS",
            clean_control_floor=0.125,
            parent_path=parent,
        )

    assert (root / "attempts.jsonl").read_bytes() == b""


def test_huge_numeric_floor_is_a_named_durability_error_without_a_partial_record(
    tmp_path: Path,
) -> None:
    """Converting an enormous integer to float must not escape as OverflowError."""
    root = tmp_path / "root"
    parent = root / "parent"
    parent.mkdir(parents=True)

    with pytest.raises(ValueError, match="finite") as raised:
        append_attempt_record(
            root,
            workload="mlp_stack",
            verdict="PASS",
            clean_control_floor=10**4_000,
            parent_path=parent,
        )

    assert raised.type.__name__ == "DurabilityError"
    assert (root / "attempts.jsonl").read_bytes() == b""


def test_existing_huge_numeric_floor_exits_two_before_science_without_traceback(
    tmp_path: Path,
) -> None:
    """A canonical huge integer in history is a boundary error, not a worker crash."""
    root = tmp_path / "root"
    parent = root / "parent"
    root.mkdir()
    parent.mkdir()
    raw_record = (
        '{"clean_control_floor":'
        + "9" * 4_000
        + f',"parent_path":{json.dumps(str(parent))},'
        '"timestamp_utc":"2026-08-30T12:00:00.000000Z",'
        '"verdict":"PASS","workload":"mlp_stack"}\n'
    ).encode()
    ledger = root / "attempts.jsonl"
    ledger.write_bytes(raw_record)
    source = (
        "module.run_aa_test = lambda *args, **kwargs: "
        "(_ for _ in ()).throw(AssertionError('runner invoked'))"
    )

    completed = _injected(
        [
            "aa-test",
            "mlp_stack",
            "--output-root",
            str(root),
            "--allow-volatile-output",
        ],
        "hephaestus.aa_runtime",
        source,
    )

    assert completed.returncode == 2
    assert "finite" in completed.stderr
    assert "Traceback" not in completed.stderr
    assert "runner invoked" not in completed.stdout
    assert ledger.read_bytes() == raw_record


def test_exec_worker_parent_forwards_output_without_raw_fork(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The parent must stream worker output without executing Python after fork."""
    from hephaestus import cli, durable_worker

    calls: list[tuple[object, ...]] = []

    def started(*args: object, **kwargs: object) -> _FakeWorkerProcess:
        calls.append((args, kwargs))
        os.write(kwargs["pass_fds"][0], SCIENCE_START_MARKER)  # type: ignore[index]
        return _FakeWorkerProcess(b'{"verdict":"FAIL"}\n', b"x\n", 1)

    monkeypatch.setattr(cli.subprocess, "Popen", started)
    monkeypatch.setattr(cli.subprocess, "run", lambda *args, **kwargs: pytest.fail("buffered run"))
    monkeypatch.setattr(os, "fork", lambda: pytest.fail("raw fork"), raising=False)
    stdout = StringIO()
    stderr = StringIO()
    with (
        warnings.catch_warnings(record=True) as caught,
        redirect_stdout(stdout),
        redirect_stderr(stderr),
    ):
        result = cli.main(["aa-test", "mlp_stack", "--output-root", "evidence"])

    assert result == 1
    assert stdout.getvalue() == '{"verdict":"FAIL"}\n'
    assert stderr.getvalue() == "x\n"
    assert not caught
    assert calls and "hephaestus.durable_worker" in calls[0][0][0]
    assert calls[0][1]["stdout"] is subprocess.PIPE
    assert calls[0][1]["stderr"] is subprocess.PIPE
    assert calls[0][1]["close_fds"] is True
    assert len(calls[0][1]["pass_fds"]) == 1
    assert "capture_output" not in calls[0][1]
    assert durable_worker is not None


def test_exec_worker_streams_simultaneous_large_output_with_bounded_chunks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Draining either pipe after the other would deadlock or buffer an arbitrary transcript."""
    from hephaestus import cli

    real_popen = subprocess.Popen
    processes: list[subprocess.Popen[bytes]] = []
    child_source = """
import os
import threading

def emit(descriptor, value):
    payload = value * 8192
    for _ in range(64):
        view = memoryview(payload)
        while view:
            written = os.write(descriptor, view)
            view = view[written:]

threads = (
    threading.Thread(target=emit, args=(1, b'o')),
    threading.Thread(target=emit, args=(2, b'e')),
)
for thread in threads:
    thread.start()
for thread in threads:
    thread.join()
"""

    def started(*args: object, **kwargs: object) -> subprocess.Popen[bytes]:
        del args
        os.write(kwargs["pass_fds"][0], SCIENCE_START_MARKER)  # type: ignore[index]
        process = real_popen([str(PYTHON), "-c", child_source], **kwargs)  # type: ignore[arg-type]
        processes.append(process)
        return process

    monkeypatch.setattr(cli.subprocess, "Popen", started)
    monkeypatch.setattr(cli.subprocess, "run", lambda *args, **kwargs: pytest.fail("buffered run"))
    stdout = _CountingTextSink("o")
    stderr = _CountingTextSink("e")
    result: list[int] = []
    with redirect_stdout(stdout), redirect_stderr(stderr):  # type: ignore[arg-type]
        thread = threading.Thread(
            target=lambda: result.append(
                cli.main(["aa-test", "mlp_stack", "--output-root", "evidence"])
            )
        )
        thread.start()
        thread.join(timeout=5)
        if thread.is_alive() and processes:
            processes[0].kill()
            thread.join(timeout=5)

    assert not thread.is_alive()
    assert result == [0]
    assert stdout.total == stderr.total == 64 * 8192
    assert stdout.max_write <= 65_536
    assert stderr.max_write <= 65_536
    assert stdout.flushes and stderr.flushes


def test_exec_worker_preserves_split_utf8_and_escapes_invalid_bytes_in_redirects(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Chunk boundaries and invalid bytes must not erase a scientific exit or evidence path."""
    from hephaestus import cli

    process = _FakeWorkerProcess(b"", b"", 1)
    process.stdout = _ChunkedBinaryStream(
        b'{"evidence_path":"/pinned/',
        b"\xe2",
        b"\x82\xac\xff",
        b'","verdict":"FAIL"}\n',
    )
    process.stderr = _ChunkedBinaryStream(b"science warning\n")
    monkeypatch.setattr(cli.subprocess, "Popen", _fake_worker_start(process))
    monkeypatch.setattr(cli.subprocess, "run", lambda *args, **kwargs: pytest.fail("buffered run"))
    stdout = StringIO()
    stderr = StringIO()

    with redirect_stdout(stdout), redirect_stderr(stderr):
        result = cli.main(["aa-test", "mlp_stack", "--output-root", "evidence"])

    assert result == 1
    assert stdout.getvalue() == (
        '{"evidence_path":"/pinned/€\\xff","verdict":"FAIL"}\n'
    )
    assert stderr.getvalue() == "science warning\n"


def test_worker_stream_completes_repeated_short_text_writes() -> None:
    """Treating one positive short count as completion silently drops evidence."""
    from hephaestus.cli import _forward_worker_stream

    target = _ShortTextSink(maximum_write=2)
    failures: list[Exception] = []

    _forward_worker_stream(
        _ChunkedBinaryStream(b"complete", b" evidence"),
        target,
        failures,
        threading.Lock(),
    )

    assert target.value == "complete evidence"
    assert target.write_calls > 2
    assert target.flushes == 1
    assert failures == []


def test_worker_stream_completes_repeated_short_binary_writes() -> None:
    """A byte destination must receive the complete deterministic UTF-8 rendering."""
    from hephaestus.cli import _forward_worker_stream

    target = _ShortBinarySink(maximum_write=2)
    failures: list[Exception] = []

    _forward_worker_stream(
        _ChunkedBinaryStream("路径".encode(), b"\xff"),
        target,
        failures,
        threading.Lock(),
    )

    assert target.getvalue() == "路径\\xff".encode()
    assert target.write_calls > 2
    assert target.flushes == 1
    assert failures == []


@pytest.mark.parametrize("write_result", (0, None))
def test_worker_stream_no_progress_is_failure_even_when_worker_succeeds(
    monkeypatch: pytest.MonkeyPatch,
    write_result: int | None,
) -> None:
    """Zero or None cannot be treated as acceptance of a successful transcript."""
    from hephaestus import cli

    class NoProgressSink:
        def __init__(self) -> None:
            self.flushes = 0

        def write(self, value: str) -> int | None:
            assert value == "evidence\n"
            return write_result

        def flush(self) -> None:
            self.flushes += 1

    target = NoProgressSink()
    monkeypatch.setattr(
        cli.subprocess,
        "Popen",
        _fake_worker_start(_FakeWorkerProcess(b"evidence\n", b"", 0)),
    )
    stderr = StringIO()

    with redirect_stdout(target), redirect_stderr(stderr):  # type: ignore[arg-type]
        result = cli.main(["aa-test", "mlp_stack", "--output-root", "evidence"])

    assert result == 1
    assert target.flushes == 1
    assert "stream failure" in stderr.getvalue()
    assert "no progress" in stderr.getvalue()


def test_worker_stream_flushes_accepted_evidence_after_later_write_failure() -> None:
    """A later destination error must not strand evidence accepted by an earlier write."""
    from hephaestus.cli import _forward_worker_stream

    class BufferedThenFailingSink:
        def __init__(self) -> None:
            self.pending = ""
            self.visible = ""
            self.writes = 0
            self.flushes = 0

        def write(self, value: str) -> int:
            self.writes += 1
            if self.writes == 2:
                raise OSError("write unavailable")
            self.pending += value
            return len(value)

        def flush(self) -> None:
            self.visible += self.pending
            self.pending = ""
            self.flushes += 1

    target = BufferedThenFailingSink()
    failures: list[Exception] = []

    _forward_worker_stream(
        _ChunkedBinaryStream(b"accepted", b" rejected"),
        target,
        failures,
        threading.Lock(),
    )

    assert target.visible == "accepted"
    assert target.flushes == 1
    assert len(failures) == 1
    assert str(failures[0]) == "write unavailable"


def test_worker_stream_flushes_accepted_evidence_after_later_read_failure() -> None:
    """A later pipe read error must not strand evidence already accepted by the destination."""
    from hephaestus.cli import _forward_worker_stream

    class ReadThenFail:
        def __init__(self) -> None:
            self.reads = 0
            self.closed = False

        def read(self, size: int = -1) -> bytes:
            del size
            self.reads += 1
            if self.reads == 1:
                return b"accepted"
            raise OSError("read unavailable")

        def close(self) -> None:
            self.closed = True

    class BufferedSink:
        def __init__(self) -> None:
            self.pending = ""
            self.visible = ""
            self.flushes = 0

        def write(self, value: str) -> int:
            self.pending += value
            return len(value)

        def flush(self) -> None:
            self.visible += self.pending
            self.pending = ""
            self.flushes += 1

    source = ReadThenFail()
    target = BufferedSink()
    failures: list[Exception] = []

    _forward_worker_stream(source, target, failures, threading.Lock())

    assert source.closed
    assert target.visible == "accepted"
    assert target.flushes == 1
    assert len(failures) == 1
    assert str(failures[0]) == "read unavailable"


def test_worker_stream_destination_error_is_flushed_and_reported() -> None:
    """Destination discovery itself cannot escape the pump and preserve worker success."""
    from hephaestus.cli import _forward_worker_stream

    class BrokenDestination:
        def __init__(self) -> None:
            self.flushes = 0

        @property
        def buffer(self) -> object:
            raise OSError("buffer unavailable")

        def flush(self) -> None:
            self.flushes += 1

    target = BrokenDestination()
    failures: list[Exception] = []

    _forward_worker_stream(
        _ChunkedBinaryStream(b"evidence"),
        target,
        failures,
        threading.Lock(),
    )

    assert target.flushes == 1
    assert len(failures) == 1
    assert str(failures[0]) == "buffer unavailable"


def test_destination_discovery_failure_still_drains_worker_before_overriding_success(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Skipping the pipe drain can stall the worker or turn success into SIGPIPE."""
    from hephaestus import cli

    class DrainRequiredStream:
        def __init__(self) -> None:
            self._chunks = iter((b"preserved evidence\n", b""))
            self.drained = threading.Event()
            self.closed = False

        def read(self, size: int = -1) -> bytes:
            del size
            chunk = next(self._chunks)
            if not chunk:
                self.drained.set()
            return chunk

        def close(self) -> None:
            self.closed = True

    class BrokenDestination:
        @property
        def buffer(self) -> object:
            raise OSError("buffer unavailable")

        def flush(self) -> None:
            return None

    class SuccessfulProcess:
        def __init__(self) -> None:
            self.stdout = DrainRequiredStream()
            self.stderr = BytesIO()

        def wait(self) -> int:
            return 0

    process = SuccessfulProcess()
    monkeypatch.setattr(cli.subprocess, "Popen", _fake_worker_start(process))
    stderr = StringIO()

    with redirect_stdout(BrokenDestination()), redirect_stderr(stderr):  # type: ignore[arg-type]
        result = cli.main(["aa-test", "mlp_stack", "--output-root", "evidence"])

    assert process.stdout.drained.is_set()
    assert process.stdout.closed
    assert result == 1
    assert "stream failure" in stderr.getvalue()
    assert "buffer unavailable" in stderr.getvalue()


def test_exec_worker_preserves_utf8_on_os_style_streams(
    monkeypatch: pytest.MonkeyPatch,
    capfd: pytest.CaptureFixture[str],
) -> None:
    """A normal encoded stdout/stderr pair must receive valid UTF-8 without transformation."""
    from hephaestus import cli

    monkeypatch.setattr(
        cli.subprocess,
        "Popen",
        _fake_worker_start(
            _FakeWorkerProcess("路径\n".encode(), "警告\n".encode(), 0)
        ),
    )
    monkeypatch.setattr(cli.subprocess, "run", lambda *args, **kwargs: pytest.fail("buffered run"))

    assert cli.main(["demo-planted-regressions", "--output-root", "evidence"]) == 0

    captured = capfd.readouterr()
    assert captured.out == "路径\n"
    assert captured.err == "警告\n"


def test_science_exit_seventy_after_control_marker_is_not_bootstrap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A science SystemExit(70) must not collide with a reserved bootstrap number."""
    from hephaestus import cli

    monkeypatch.setattr(
        cli.subprocess,
        "Popen",
        _fake_worker_start(_FakeWorkerProcess(b"science\n", b"", 70)),
    )
    stdout = StringIO()
    stderr = StringIO()

    with redirect_stdout(stdout), redirect_stderr(stderr):
        result = cli.main(["aa-test", "mlp_stack", "--output-root", "evidence"])

    assert result == 70
    assert stdout.getvalue() == "science\n"
    assert "bootstrap" not in stderr.getvalue()


def test_exit_one_without_control_marker_is_bootstrap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A pre-marker setup/import exit one must not be labeled as scientific failure."""
    from hephaestus import cli

    monkeypatch.setattr(
        cli.subprocess,
        "Popen",
        _fake_worker_start(
            _FakeWorkerProcess(b"", b"setup failed\n", 1),
            control=b"",
        ),
    )
    stderr = StringIO()

    with redirect_stderr(stderr):
        result = cli.main(["aa-test", "mlp_stack", "--output-root", "evidence"])

    assert result == 1
    assert "setup failed\n" in stderr.getvalue()
    assert "bootstrap failure" in stderr.getvalue()
    assert "before science" in stderr.getvalue()


@pytest.mark.parametrize(
    "control",
    (
        b"malformed-science-start\n",
        SCIENCE_START_MARKER + SCIENCE_START_MARKER,
    ),
    ids=("malformed", "duplicate"),
)
def test_malformed_or_duplicate_control_marker_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
    control: bytes,
) -> None:
    """Only one exact authenticated marker can authorize science-stage classification."""
    from hephaestus import cli

    monkeypatch.setattr(
        cli.subprocess,
        "Popen",
        _fake_worker_start(_FakeWorkerProcess(b"science\n", b"", 0), control=control),
    )
    stderr = StringIO()

    with redirect_stderr(stderr):
        result = cli.main(["aa-test", "mlp_stack", "--output-root", "evidence"])

    assert result == 1
    assert "control marker" in stderr.getvalue()
    assert "bootstrap failure" in stderr.getvalue()


@pytest.mark.parametrize(
    ("control", "expected_stage"),
    ((b"", "before science"), (SCIENCE_START_MARKER, "after science")),
    ids=("before-marker", "after-marker"),
)
def test_signal_classification_uses_control_marker_stage(
    monkeypatch: pytest.MonkeyPatch,
    control: bytes,
    expected_stage: str,
) -> None:
    """The same signal is bootstrap or science according to the authenticated stage."""
    from hephaestus import cli

    monkeypatch.setattr(
        cli.subprocess,
        "Popen",
        _fake_worker_start(
            _FakeWorkerProcess(b"", b"", -15),
            control=control,
        ),
    )
    stderr = StringIO()

    with redirect_stderr(stderr):
        result = cli.main(["aa-test", "mlp_stack", "--output-root", "evidence"])

    assert result == 1
    assert "signal 15" in stderr.getvalue()
    assert expected_stage in stderr.getvalue()


def test_stdout_and_stderr_cannot_inject_the_science_start_marker(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Marker-looking scientific streams cannot authenticate an unmarked worker."""
    from hephaestus import cli

    marker_text = SCIENCE_START_MARKER.decode()
    monkeypatch.setattr(
        cli.subprocess,
        "Popen",
        _fake_worker_start(
            _FakeWorkerProcess(SCIENCE_START_MARKER, SCIENCE_START_MARKER, 0),
            control=b"",
        ),
    )
    stdout = StringIO()
    stderr = StringIO()

    with redirect_stdout(stdout), redirect_stderr(stderr):
        result = cli.main(["aa-test", "mlp_stack", "--output-root", "evidence"])

    assert result == 1
    assert stdout.getvalue() == marker_text
    assert stderr.getvalue().startswith(marker_text)
    assert "bootstrap failure" in stderr.getvalue()


def test_parent_closes_both_control_pipe_ends(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The parent must retain neither end of the per-worker authentication channel."""
    from hephaestus import cli

    real_pipe = os.pipe
    created: list[tuple[int, int]] = []

    def tracked_pipe() -> tuple[int, int]:
        descriptors = real_pipe()
        created.append(descriptors)
        return descriptors

    monkeypatch.setattr(os, "pipe", tracked_pipe)
    monkeypatch.setattr(
        cli.subprocess,
        "Popen",
        _fake_worker_start(_FakeWorkerProcess(b"", b"", 0)),
    )

    assert cli.main(["aa-test", "mlp_stack", "--output-root", "evidence"]) == 0
    assert len(created) == 1
    for descriptor in created[0]:
        with pytest.raises(OSError):
            os.fstat(descriptor)


def test_worker_emits_exact_marker_after_setup_and_closes_channel_before_science(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Science starts only after root/ledger setup and cannot retain the control capability."""
    from argparse import Namespace

    from hephaestus import cli, durability, durable_worker

    events: list[str] = []
    read_fd, write_fd = os.pipe()

    class Parser:
        def parse_args(self, arguments: list[str]) -> Namespace:
            del arguments
            events.append("parse")
            return Namespace(
                command="aa-test",
                output_root=Path("evidence"),
                allow_volatile_output=True,
            )

    class AttemptRoot:
        def __enter__(self) -> AttemptRoot:
            events.append("root-enter")
            return self

        def enter_worker_directory(self) -> None:
            events.append("root-directory")

        def close(self) -> tuple[int, ...]:
            events.append("root-close")
            return ()

    def run_after_parse(parser: object, parsed: object) -> int:
        del parser, parsed
        events.append("science")
        assert os.read(read_fd, 4096) == SCIENCE_START_MARKER
        with pytest.raises(OSError):
            os.fstat(write_fd)
        return 0

    monkeypatch.setattr(cli, "_parser", lambda: Parser())
    monkeypatch.setattr(cli, "_run_after_parse", run_after_parse)
    monkeypatch.setattr(
        durability,
        "prepare_attempt_output_root",
        lambda *args, **kwargs: AttemptRoot(),
    )

    try:
        result = durable_worker.main(
            [json.dumps(["aa-test", "mlp_stack"]), str(write_fd)]
        )
    finally:
        with suppress(OSError):
            os.close(write_fd)
        os.close(read_fd)

    assert result == 0
    assert events == ["parse", "root-enter", "root-directory", "science", "root-close"]


@pytest.mark.parametrize(
    ("science_result", "host_ledger_close_failure", "expected_result", "expected_error"),
    (
        (0, True, 0, "host-state ledger descriptor close failed"),
        (0, False, 1, "descriptor close failed"),
        (7, True, 7, "host-state ledger descriptor close failed"),
        (7, False, 7, "descriptor close failed"),
    ),
    ids=(
        "successful-science-host-ledger-diagnostic-only",
        "successful-science-non-host-cleanup-failure",
        "scientific-failure-host-ledger-diagnostic-only",
        "scientific-failure-non-host-cleanup-failure-precedence",
    ),
)
def test_worker_close_failure_changes_only_a_successful_science_exit(
    monkeypatch: pytest.MonkeyPatch,
    science_result: int,
    host_ledger_close_failure: bool,
    expected_result: int,
    expected_error: str,
) -> None:
    """Cleanup failure is operational failure without erasing science or its marker."""
    from argparse import Namespace

    from hephaestus import cli, durability, durable_worker

    events: list[str] = []

    class Parser:
        def parse_args(self, arguments: list[str]) -> Namespace:
            del arguments
            return Namespace(
                command="aa-test",
                output_root=Path("evidence"),
                allow_volatile_output=True,
            )

    class AttemptRoot:
        _host_ledger_fd = 97

        def __enter__(self) -> AttemptRoot:
            return self

        def enter_worker_directory(self) -> None:
            return None

        def close(self) -> tuple[int, ...]:
            events.append("close")
            return (97,) if host_ledger_close_failure else (98,)

    def run_after_parse(parser: object, parsed: object) -> int:
        del parser, parsed
        events.append("science")
        print("completed-science-output")
        return science_result

    monkeypatch.setattr(cli, "_parser", lambda: Parser())
    monkeypatch.setattr(cli, "_run_after_parse", run_after_parse)
    monkeypatch.setattr(
        durability,
        "prepare_attempt_output_root",
        lambda *args, **kwargs: AttemptRoot(),
    )
    stdout = StringIO()
    stderr = StringIO()
    control_read_fd, control_write_fd = os.pipe()

    try:
        with redirect_stdout(stdout), redirect_stderr(stderr):
            result = durable_worker.main(
                [json.dumps(["aa-test", "mlp_stack"]), str(control_write_fd)]
            )
        marker = os.read(control_read_fd, len(SCIENCE_START_MARKER))
    finally:
        with suppress(OSError):
            os.close(control_write_fd)
        os.close(control_read_fd)

    assert result == expected_result
    assert marker == SCIENCE_START_MARKER
    assert stdout.getvalue() == "completed-science-output\n"
    assert expected_error in stderr.getvalue()
    assert events == ["science", "close"]


@pytest.mark.parametrize(
    ("arguments", "failed_bootstrap"),
    (
        (["aa-test", "mlp_stack"], "criteria"),
        (["aa-test", "mlp_stack"], "hephaestus.aa_runtime"),
        (["demo-planted-regressions"], "hephaestus.demo"),
    ),
    ids=("criteria-validation", "aa-import", "demo-import"),
)
def test_worker_authenticates_science_only_after_target_bootstrap(
    monkeypatch: pytest.MonkeyPatch,
    arguments: list[str],
    failed_bootstrap: str,
) -> None:
    """Criteria validation and target imports belong to bootstrap, before the marker."""
    from hephaestus import criteria, durability, durable_worker

    class AttemptRoot:
        def __enter__(self) -> AttemptRoot:
            return self

        def enter_worker_directory(self) -> None:
            return None

        def close(self) -> tuple[()]:
            return ()

    monkeypatch.setattr(
        durability,
        "prepare_attempt_output_root",
        lambda *args, **kwargs: AttemptRoot(),
    )
    if failed_bootstrap == "criteria":
        monkeypatch.setattr(
            criteria,
            "packaged_criteria_path",
            lambda: (_ for _ in ()).throw(RuntimeError("criteria unavailable")),
        )
    else:
        monkeypatch.setattr(
            criteria,
            "packaged_criteria_path",
            lambda: nullcontext(ROOT / "gates" / "default.yaml"),
        )
        real_import = builtins.__import__

        def fail_target_import(
            name: str,
            globals: object = None,
            locals: object = None,
            fromlist: tuple[str, ...] = (),
            level: int = 0,
        ) -> object:
            if name == failed_bootstrap:
                raise ImportError(f"{failed_bootstrap} unavailable")
            return real_import(name, globals, locals, fromlist, level)  # type: ignore[arg-type]

        monkeypatch.setattr(builtins, "__import__", fail_target_import)

    control_read_fd, control_write_fd = os.pipe()
    stdout = StringIO()
    stderr = StringIO()
    try:
        with redirect_stdout(stdout), redirect_stderr(stderr):
            result = durable_worker.main(
                [json.dumps(arguments), str(control_write_fd)]
            )
        assert os.read(control_read_fd, len(SCIENCE_START_MARKER) + 1) == b""
    finally:
        with suppress(OSError):
            os.close(control_write_fd)
        os.close(control_read_fd)

    assert result == 1
    assert stdout.getvalue() == ""
    assert "durable worker bootstrap failed" in stderr.getvalue()
    assert "unavailable" in stderr.getvalue()


def test_complete_marker_is_irrevocable_when_control_close_reports_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A close error after the complete marker cannot prevent prepared science."""
    from hephaestus import criteria, durability, durable_worker

    science_calls: list[str] = []
    attempts: list[dict[str, object]] = []

    class AttemptRoot:
        def __enter__(self) -> AttemptRoot:
            return self

        def enter_worker_directory(self) -> None:
            return None

        def append_attempt(self, **record: object) -> None:
            attempts.append(record)

        def close(self) -> tuple[()]:
            return ()

    aa_runtime = ModuleType("hephaestus.aa_runtime")

    def run_aa_test(*args: object, **kwargs: object) -> SimpleNamespace:
        del args, kwargs
        science_calls.append("science")
        return SimpleNamespace(
            parent_path=Path("aa-parent"),
            verdict="PASS",
            driving_finding="all_criteria_passed",
            statistics=None,
        )

    aa_runtime.run_aa_test = run_aa_test  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "hephaestus.aa_runtime", aa_runtime)
    monkeypatch.setattr(
        durability,
        "prepare_attempt_output_root",
        lambda *args, **kwargs: AttemptRoot(),
    )
    monkeypatch.setattr(
        criteria,
        "packaged_criteria_path",
        lambda: nullcontext(ROOT / "gates" / "default.yaml"),
    )

    control_read_fd, control_write_fd = os.pipe()
    real_close = os.close

    def close_with_reported_failure(descriptor: int) -> None:
        if descriptor == control_write_fd:
            raise OSError("close outcome unavailable")
        real_close(descriptor)

    monkeypatch.setattr(durable_worker.os, "close", close_with_reported_failure)
    stdout = StringIO()
    stderr = StringIO()
    try:
        with redirect_stdout(stdout), redirect_stderr(stderr):
            result = durable_worker.main(
                [json.dumps(["aa-test", "mlp_stack"]), str(control_write_fd)]
            )
        marker = os.read(control_read_fd, len(SCIENCE_START_MARKER))
    finally:
        with suppress(OSError):
            real_close(control_write_fd)
        real_close(control_read_fd)

    assert marker == SCIENCE_START_MARKER
    assert result == 0
    assert science_calls == ["science"]
    assert len(attempts) == 1
    assert "control marker unavailable" not in stderr.getvalue()


def test_durable_worker_threads_its_preopened_root_as_the_host_state_sink(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The worker must reuse one pinned root capability instead of reopening by path."""
    from hephaestus import criteria, durability, durable_worker

    received_sinks: list[object] = []

    class AttemptRoot:
        def __enter__(self) -> AttemptRoot:
            return self

        def enter_worker_directory(self) -> None:
            return None

        def append_attempt(self, **record: object) -> None:
            del record

        def close(self) -> tuple[()]:
            return ()

    attempt_root = AttemptRoot()
    aa_runtime = ModuleType("hephaestus.aa_runtime")

    def run_aa_test(
        *args: object,
        _host_state_sink: object,
        **kwargs: object,
    ) -> SimpleNamespace:
        del args, kwargs
        received_sinks.append(_host_state_sink)
        return SimpleNamespace(
            parent_path=Path("aa-parent"),
            verdict="PASS",
            driving_finding="all_criteria_passed",
            statistics=None,
        )

    aa_runtime.run_aa_test = run_aa_test  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "hephaestus.aa_runtime", aa_runtime)
    monkeypatch.setattr(
        durability,
        "prepare_attempt_output_root",
        lambda *args, **kwargs: attempt_root,
    )
    monkeypatch.setattr(
        criteria,
        "packaged_criteria_path",
        lambda: nullcontext(ROOT / "gates" / "default.yaml"),
    )

    control_read_fd, control_write_fd = os.pipe()
    try:
        result = durable_worker.main(
            [json.dumps(["aa-test", "mlp_stack"]), str(control_write_fd)]
        )
        assert os.read(control_read_fd, len(SCIENCE_START_MARKER)) == SCIENCE_START_MARKER
    finally:
        with suppress(OSError):
            os.close(control_write_fd)
        os.close(control_read_fd)

    assert result == 0
    assert received_sinks == [attempt_root]


def test_durable_worker_threads_preopened_root_to_demo_target(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Removing only demo sink wiring must fail before any durable demo can run."""
    from hephaestus import criteria, durability, durable_worker

    received_sinks: list[object] = []

    class AttemptRoot:
        def __enter__(self) -> AttemptRoot:
            return self

        def enter_worker_directory(self) -> None:
            return None

        def append_attempt(self, **record: object) -> None:
            del record

        def close(self) -> tuple[()]:
            return ()

    attempt_root = AttemptRoot()
    demo = ModuleType("hephaestus.demo")

    def run_planted_demo(
        *args: object,
        _host_state_sink: object,
        **kwargs: object,
    ) -> SimpleNamespace:
        del args, kwargs
        received_sinks.append(_host_state_sink)
        return SimpleNamespace(
            parent_path=Path("demo-parent"),
            rows=(),
            passed=True,
            mismatches=(),
        )

    demo.run_planted_demo = run_planted_demo  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "hephaestus.demo", demo)
    monkeypatch.setattr(
        durability,
        "prepare_attempt_output_root",
        lambda *args, **kwargs: attempt_root,
    )
    monkeypatch.setattr(
        criteria,
        "packaged_criteria_path",
        lambda: nullcontext(ROOT / "gates" / "default.yaml"),
    )

    control_read_fd, control_write_fd = os.pipe()
    try:
        result = durable_worker.main(
            [json.dumps(["demo-planted-regressions"]), str(control_write_fd)]
        )
        assert os.read(control_read_fd, len(SCIENCE_START_MARKER)) == SCIENCE_START_MARKER
    finally:
        with suppress(OSError):
            os.close(control_write_fd)
        os.close(control_read_fd)

    assert result == 0
    assert received_sinks == [attempt_root]


@pytest.mark.parametrize(
    ("control", "worker_returncode", "expected_result"),
    (
        (SCIENCE_START_MARKER, 0, 0),
        (b"", 1, 1),
    ),
    ids=("science-marker", "empty-bootstrap"),
)
def test_parent_control_classification_is_bounded_when_write_close_fails(
    monkeypatch: pytest.MonkeyPatch,
    control: bytes,
    worker_returncode: int,
    expected_result: int,
) -> None:
    """A retained parent write end must not make marker parsing wait for EOF."""
    from hephaestus import cli

    real_pipe = os.pipe
    real_close = os.close
    real_set_blocking = os.set_blocking
    created: list[tuple[int, int]] = []
    set_blocking_calls: list[tuple[int, bool]] = []

    def tracked_pipe() -> tuple[int, int]:
        descriptors = real_pipe()
        real_set_blocking(descriptors[0], False)
        created.append(descriptors)
        return descriptors

    def tracked_set_blocking(descriptor: int, blocking: bool) -> None:
        set_blocking_calls.append((descriptor, blocking))
        real_set_blocking(descriptor, blocking)

    def close_with_write_failure(descriptor: int) -> None:
        if created and descriptor == created[0][1]:
            return
        real_close(descriptor)

    monkeypatch.setattr(cli.os, "pipe", tracked_pipe)
    monkeypatch.setattr(cli.os, "set_blocking", tracked_set_blocking)
    monkeypatch.setattr(cli, "_close_worker_descriptor", close_with_write_failure)
    monkeypatch.setattr(
        cli.subprocess,
        "Popen",
        _fake_worker_start(
            _FakeWorkerProcess(b"", b"", worker_returncode),
            control=control,
        ),
    )
    stderr = StringIO()
    try:
        with redirect_stderr(stderr):
            result = cli.main(
                ["aa-test", "mlp_stack", "--output-root", "evidence"]
            )
    finally:
        if created:
            with suppress(OSError):
                real_close(created[0][1])

    assert set_blocking_calls == [(created[0][0], False)]
    assert result == expected_result
    if control:
        assert "bootstrap failure" not in stderr.getvalue()
    else:
        assert "before science" in stderr.getvalue()


def test_large_malformed_control_is_drained_concurrently_without_deadlock(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Waiting for worker exit before draining lets a large control write fill the pipe."""
    from hephaestus import cli

    real_popen = subprocess.Popen
    processes: list[subprocess.Popen[bytes]] = []

    def started(*args: object, **kwargs: object) -> subprocess.Popen[bytes]:
        del args
        descriptor = kwargs["pass_fds"][0]  # type: ignore[index]
        child_source = f"""
import os
import signal

signal.alarm(2)
remaining = memoryview(b'x' * (1024 * 1024))
while remaining:
    written = os.write({descriptor}, remaining)
    remaining = remaining[written:]
"""
        process = real_popen([str(PYTHON), "-c", child_source], **kwargs)  # type: ignore[arg-type]
        processes.append(process)
        return process

    monkeypatch.setattr(cli.subprocess, "Popen", started)
    stdout = StringIO()
    stderr = StringIO()

    with redirect_stdout(stdout), redirect_stderr(stderr):
        result = cli.main(["aa-test", "mlp_stack", "--output-root", "evidence"])

    assert len(processes) == 1
    assert processes[0].returncode == 0
    assert result == 1
    assert stdout.getvalue() == ""
    assert "invalid control marker" in stderr.getvalue()
    assert len(stderr.getvalue()) < 512


def test_repeated_parent_close_failures_do_not_leak_owned_control_writers() -> None:
    """A non-closing close error must be cleaned by production, not the test process."""
    source = f"""
import io
import json
import os

from hephaestus import cli

marker = {SCIENCE_START_MARKER!r}
real_close = os.close
owned_identities = {{}}
attempted = []

def identity(descriptor):
    status = os.fstat(descriptor)
    return (status.st_dev, status.st_ino, status.st_mode)

class Process:
    def __init__(self):
        self.stdout = io.BytesIO()
        self.stderr = io.BytesIO()

    def wait(self):
        return 0

def started(*args, **kwargs):
    del args
    descriptor = kwargs['pass_fds'][0]
    token = identity(descriptor)
    owned_identities[descriptor] = token
    attempted.append((descriptor, token))
    os.write(descriptor, marker)
    return Process()

def close_without_closing_owned_writer(descriptor):
    token = owned_identities.get(descriptor)
    if token is not None:
        try:
            current = identity(descriptor)
        except OSError:
            current = None
        if current == token:
            raise OSError('injected close failure without close')
    real_close(descriptor)

cli.subprocess.Popen = started
cli.os.close = close_without_closing_owned_writer
results = [
    cli.main(['aa-test', 'mlp_stack', '--output-root', 'evidence'])
    for _ in range(8)
]
leaked = []
for descriptor, token in attempted:
    try:
        current = identity(descriptor)
    except OSError:
        continue
    if current == token:
        leaked.append(descriptor)
print(json.dumps({{'results': results, 'leaked': leaked}}, sort_keys=True))
raise SystemExit(0 if results == [0] * 8 and not leaked else 1)
"""

    completed = subprocess.run(
        [str(PYTHON), "-c", source],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
        timeout=10,
    )

    assert completed.returncode == 0, completed.stderr
    payload = json.loads(completed.stdout)
    assert payload == {"leaked": [], "results": [0] * 8}


def test_control_reader_terminates_from_memory_completion_without_pipe_eof() -> None:
    """The in-memory completion event ends draining while the writer remains open."""
    from hephaestus import cli

    control_read_fd, control_write_fd = os.pipe()
    completion = threading.Event()
    payloads: list[bytes] = []
    failures: list[Exception] = []
    thread = threading.Thread(
        target=cli._drain_worker_control,
        args=(control_read_fd, completion, payloads, failures),
    )
    thread.start()
    try:
        os.write(control_write_fd, SCIENCE_START_MARKER)
        completion.set()
        thread.join(timeout=1)
        stranded = thread.is_alive()
    finally:
        os.close(control_write_fd)
        thread.join(timeout=1)

    assert stranded is False
    assert payloads == [SCIENCE_START_MARKER]
    assert failures == []


@pytest.mark.parametrize("failure_stage", ("construct", "start"))
def test_control_reader_startup_failure_rolls_back_owned_descriptors(
    failure_stage: str,
) -> None:
    """A control Thread constructor/start failure cannot orphan fds or its worker."""
    source = f"""
import io
import json
import os
import signal
import threading
from contextlib import redirect_stderr

from hephaestus import cli

failure_stage = {failure_stage!r}
marker = {SCIENCE_START_MARKER!r}
real_pipe = os.pipe
real_thread = threading.Thread
created = []
processes = []

def identity(descriptor):
    status = os.fstat(descriptor)
    return (status.st_dev, status.st_ino, status.st_mode, status.st_rdev)

def tracked_pipe():
    descriptors = real_pipe()
    created.append((descriptors[0], identity(descriptors[0])))
    return descriptors

def is_control_target(target):
    return getattr(target, '__name__', '') == '_drain_worker_control'

class ControlledThread(real_thread):
    def __init__(self, *args, **kwargs):
        self.control_target = is_control_target(kwargs.get('target'))
        if failure_stage == 'construct' and self.control_target:
            raise RuntimeError('injected control thread construction failure')
        super().__init__(*args, **kwargs)

    def start(self):
        if failure_stage == 'start' and self.control_target:
            raise RuntimeError('injected control thread start failure')
        return super().start()

class Process:
    def __init__(self):
        self.stdout = io.BytesIO()
        self.stderr = io.BytesIO()
        self.returncode = None
        self.stopped = False
        self.waited = False

    def poll(self):
        return self.returncode

    def terminate(self):
        self.stopped = True
        self.returncode = -15

    def kill(self):
        self.stopped = True
        self.returncode = -9

    def wait(self, timeout=None):
        del timeout
        self.waited = True
        return 0 if self.returncode is None else self.returncode

def started(*args, **kwargs):
    del args
    os.write(kwargs['pass_fds'][0], marker)
    process = Process()
    processes.append(process)
    return process

os.pipe = tracked_pipe
threading.Thread = ControlledThread
cli.subprocess.Popen = started
results = []
errors = io.StringIO()
signal.alarm(3)
with redirect_stderr(errors):
    for _ in range(8):
        try:
            results.append(
                cli.main(['aa-test', 'mlp_stack', '--output-root', 'evidence'])
            )
        except Exception as error:
            results.append(type(error).__name__)
signal.alarm(0)
leaked = []
for descriptor, token in created:
    try:
        current = identity(descriptor)
    except OSError:
        continue
    if current == token:
        leaked.append(descriptor)
cleaned = [
    process.stopped
    and process.waited
    and process.stdout.closed
    and process.stderr.closed
    for process in processes
]
payload = {{'cleaned': cleaned, 'leaked': leaked, 'results': results}}
print(json.dumps(payload, sort_keys=True))
expected = {{'cleaned': [True] * 8, 'leaked': [], 'results': [1] * 8}}
raise SystemExit(0 if payload == expected else 1)
"""

    completed = subprocess.run(
        [str(PYTHON), "-c", source],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
        timeout=6,
    )

    assert completed.returncode == 0, (
        f"stdout={completed.stdout!r} stderr={completed.stderr!r}"
    )
    assert json.loads(completed.stdout) == {
        "cleaned": [True] * 8,
        "leaked": [],
        "results": [1] * 8,
    }


def test_transient_writer_identity_failure_is_retried_without_leak() -> None:
    """An unverifiable retained writer stays owned until identity can be resolved."""
    source = f"""
import errno
import io
import json
import os

from hephaestus import cli

marker = {SCIENCE_START_MARKER!r}
real_close = os.close
real_descriptor_identity = cli._descriptor_identity
owned_identities = {{}}
attempted = []
state = {{'fail_next': False, 'injected': 0}}

def identity(descriptor):
    status = os.fstat(descriptor)
    return (status.st_dev, status.st_ino, status.st_mode, status.st_rdev)

class Process:
    def __init__(self):
        self.stdout = io.BytesIO()
        self.stderr = io.BytesIO()

    def wait(self):
        state['fail_next'] = True
        return 0

def started(*args, **kwargs):
    del args
    descriptor = kwargs['pass_fds'][0]
    token = identity(descriptor)
    owned_identities[descriptor] = token
    attempted.append((descriptor, token))
    os.write(descriptor, marker)
    return Process()

def close_without_closing_owned_writer(descriptor):
    token = owned_identities.get(descriptor)
    if token is not None:
        try:
            current = identity(descriptor)
        except OSError:
            current = None
        if current == token:
            raise OSError('injected close failure without close')
    real_close(descriptor)

def transient_descriptor_identity(descriptor):
    if state['fail_next']:
        state['fail_next'] = False
        state['injected'] += 1
        raise OSError(errno.EIO, 'injected transient identity failure')
    return real_descriptor_identity(descriptor)

cli.subprocess.Popen = started
cli.os.close = close_without_closing_owned_writer
cli._descriptor_identity = transient_descriptor_identity
results = [
    cli.main(['aa-test', 'mlp_stack', '--output-root', 'evidence'])
    for _ in range(8)
]
leaked = []
for descriptor, token in attempted:
    try:
        current = identity(descriptor)
    except OSError:
        continue
    if current == token:
        leaked.append(descriptor)
payload = {{'injected': state['injected'], 'leaked': leaked, 'results': results}}
print(json.dumps(payload, sort_keys=True))
expected = {{'injected': 8, 'leaked': [], 'results': [0] * 8}}
raise SystemExit(0 if payload == expected else 1)
"""

    completed = subprocess.run(
        [str(PYTHON), "-c", source],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
        timeout=10,
    )

    assert completed.returncode == 0, completed.stderr
    assert json.loads(completed.stdout) == {
        "injected": 8,
        "leaked": [],
        "results": [0] * 8,
    }


def test_persistently_unverifiable_writer_is_retained_for_later_cleanup() -> None:
    """An exhausted identity retry budget cannot drop the still-uncertain owner."""
    source = f"""
import errno
import io
import json
import os

from hephaestus import cli

marker = {SCIENCE_START_MARKER!r}
real_close = os.close
real_descriptor_identity = cli._descriptor_identity
owned_identities = {{}}
attempted = []
state = {{'arm': True, 'remaining': 0}}

def identity(descriptor):
    status = os.fstat(descriptor)
    return (status.st_dev, status.st_ino, status.st_mode, status.st_rdev)

class Process:
    def __init__(self):
        self.stdout = io.BytesIO()
        self.stderr = io.BytesIO()

    def wait(self):
        if state['arm']:
            state['arm'] = False
            state['remaining'] = 6
        return 0

def started(*args, **kwargs):
    del args
    descriptor = kwargs['pass_fds'][0]
    token = identity(descriptor)
    owned_identities[descriptor] = token
    attempted.append((descriptor, token))
    os.write(descriptor, marker)
    return Process()

def close_without_closing_owned_writer(descriptor):
    token = owned_identities.get(descriptor)
    if token is not None:
        try:
            current = identity(descriptor)
        except OSError:
            current = None
        if current == token:
            raise OSError('injected close failure without close')
    real_close(descriptor)

def persistently_unverifiable_identity(descriptor):
    if state['remaining']:
        state['remaining'] -= 1
        raise OSError(errno.EIO, 'injected persistent identity failure')
    return real_descriptor_identity(descriptor)

cli.subprocess.Popen = started
cli.os.close = close_without_closing_owned_writer
cli._descriptor_identity = persistently_unverifiable_identity
results = [
    cli.main(['aa-test', 'mlp_stack', '--output-root', 'evidence'])
    for _ in range(3)
]
leaked = []
for descriptor, token in attempted:
    try:
        current = identity(descriptor)
    except OSError:
        continue
    if current == token:
        leaked.append(descriptor)
payload = {{
    'leaked': leaked,
    'remaining': state['remaining'],
    'results': results,
    'started': len(attempted),
}}
print(json.dumps(payload, sort_keys=True))
expected = {{'leaked': [], 'remaining': 0, 'results': [1, 1, 0], 'started': 2}}
raise SystemExit(0 if payload == expected else 1)
"""

    completed = subprocess.run(
        [str(PYTHON), "-c", source],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
        timeout=10,
    )

    assert completed.returncode == 0, completed.stderr
    assert json.loads(completed.stdout) == {
        "leaked": [],
        "remaining": 0,
        "results": [1, 1, 0],
        "started": 2,
    }


def test_worker_wait_failure_signals_and_cleans_all_channel_owners() -> None:
    """A wait exception must enter the same bounded worker/channel rollback path."""
    source = f"""
import io
import json
import os
import signal

from hephaestus import cli

marker = {SCIENCE_START_MARKER!r}
real_pipe = os.pipe
created = []
processes = []

def identity(descriptor):
    status = os.fstat(descriptor)
    return (status.st_dev, status.st_ino, status.st_mode, status.st_rdev)

def tracked_pipe():
    descriptors = real_pipe()
    created.append((descriptors[0], identity(descriptors[0])))
    return descriptors

class Process:
    def __init__(self):
        self.stdout = io.BytesIO()
        self.stderr = io.BytesIO()
        self.returncode = None
        self.stopped = False
        self.wait_calls = 0

    def kill(self):
        self.stopped = True
        self.returncode = -9

    def wait(self, timeout=None):
        del timeout
        self.wait_calls += 1
        if self.wait_calls == 1:
            raise RuntimeError('injected worker wait failure')
        return self.returncode

def started(*args, **kwargs):
    del args
    os.write(kwargs['pass_fds'][0], marker)
    process = Process()
    processes.append(process)
    return process

os.pipe = tracked_pipe
cli.subprocess.Popen = started
signal.alarm(2)
result = cli.main(['aa-test', 'mlp_stack', '--output-root', 'evidence'])
signal.alarm(0)
leaked = []
for descriptor, token in created:
    try:
        current = identity(descriptor)
    except OSError:
        continue
    if current == token:
        leaked.append(descriptor)
process = processes[0]
payload = {{
    'cleaned': process.stopped
    and process.wait_calls == 2
    and process.stdout.closed
    and process.stderr.closed,
    'leaked': leaked,
    'result': result,
}}
print(json.dumps(payload, sort_keys=True))
expected = {{'cleaned': True, 'leaked': [], 'result': 1}}
raise SystemExit(0 if payload == expected else 1)
"""

    completed = subprocess.run(
        [str(PYTHON), "-c", source],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
        timeout=5,
    )

    assert completed.returncode == 0, (
        f"stdout={completed.stdout!r} stderr={completed.stderr!r}"
    )
    assert json.loads(completed.stdout) == {
        "cleaned": True,
        "leaked": [],
        "result": 1,
    }


def test_control_start_error_after_thread_started_preserves_single_owner() -> None:
    """A post-start exception cannot let the reader later close a reused descriptor."""
    source = f"""
import _thread
import io
import json
import os
import signal
import threading
import time

from hephaestus import cli

marker = {SCIENCE_START_MARKER!r}
real_pipe = os.pipe
real_thread = threading.Thread
entered = threading.Event()
release = threading.Event()
done = threading.Event()
created = []

def tracked_pipe():
    descriptors = real_pipe()
    created.append(descriptors)
    return descriptors

def controlled_drain(descriptor, completion, payloads, failures):
    del completion, payloads
    entered.set()
    release.wait()
    try:
        os.close(descriptor)
    except Exception as error:
        failures.append(error)
    finally:
        done.set()

class ControlledThread(real_thread):
    def __init__(self, *args, **kwargs):
        self.control_target = kwargs.get('target') is controlled_drain
        super().__init__(*args, **kwargs)

    def start(self):
        if self.control_target:
            super().start()
            if not entered.wait(1):
                raise RuntimeError('control reader failed to enter')
            raise RuntimeError('injected post-start failure')
        return super().start()

class Process:
    def __init__(self):
        self.stdout = io.BytesIO()
        self.stderr = io.BytesIO()
        self.returncode = None

    def kill(self):
        self.returncode = -9

    def wait(self, timeout=None):
        del timeout
        return 0 if self.returncode is None else self.returncode

def started(*args, **kwargs):
    del args
    os.write(kwargs['pass_fds'][0], marker)
    return Process()

def delayed_release():
    time.sleep(0.5)
    release.set()

os.pipe = tracked_pipe
threading.Thread = ControlledThread
cli._drain_worker_control = controlled_drain
cli.subprocess.Popen = started
_thread.start_new_thread(delayed_release, ())
signal.alarm(3)
result = cli.main(['aa-test', 'mlp_stack', '--output-root', 'evidence'])
sentinel_read, sentinel_write = real_pipe()
reused = bool(created) and sentinel_read == created[0][0]
finished = done.wait(1)
try:
    os.fstat(sentinel_read)
except OSError:
    sentinel_open = False
else:
    sentinel_open = True
signal.alarm(0)
payload = {{
    'finished': finished,
    'result': result,
    'reused': reused,
    'sentinel_open': sentinel_open,
}}
print(json.dumps(payload, sort_keys=True))
expected = {{
    'finished': True,
    'result': 1,
    'reused': True,
    'sentinel_open': True,
}}
raise SystemExit(0 if payload == expected else 1)
"""

    completed = subprocess.run(
        [str(PYTHON), "-c", source],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
        timeout=5,
    )

    assert completed.returncode == 0, (
        f"stdout={completed.stdout!r} stderr={completed.stderr!r}"
    )
    assert json.loads(completed.stdout) == {
        "finished": True,
        "result": 1,
        "reused": True,
        "sentinel_open": True,
    }


def test_stream_start_error_after_thread_started_waits_for_owned_pump() -> None:
    """A post-start stream exception cannot let rollback return with a live pump."""
    source = f"""
import _thread
import io
import json
import os
import signal
import threading
import time

from hephaestus import cli

marker = {SCIENCE_START_MARKER!r}
real_thread = threading.Thread
entered = threading.Event()
release = threading.Event()
done = threading.Event()
pumps = []

def controlled_pump(source, target, failures, failure_lock):
    del source, target, failures, failure_lock
    entered.set()
    release.wait()
    done.set()

class ControlledThread(real_thread):
    def __init__(self, *args, **kwargs):
        self.stream_target = kwargs.get('target') is controlled_pump
        super().__init__(*args, **kwargs)
        if self.stream_target:
            pumps.append(self)

    def start(self):
        if self.stream_target and self is pumps[0]:
            super().start()
            if not entered.wait(1):
                raise RuntimeError('worker stream failed to enter')
            raise RuntimeError('injected post-start stream failure')
        return super().start()

class Process:
    def __init__(self):
        self.stdout = io.BytesIO()
        self.stderr = io.BytesIO()
        self.returncode = None

    def kill(self):
        self.returncode = -9

    def wait(self, timeout=None):
        del timeout
        return 0 if self.returncode is None else self.returncode

def started(*args, **kwargs):
    del args
    os.write(kwargs['pass_fds'][0], marker)
    return Process()

def delayed_release():
    time.sleep(0.25)
    release.set()

threading.Thread = ControlledThread
cli._forward_worker_stream = controlled_pump
cli.subprocess.Popen = started
_thread.start_new_thread(delayed_release, ())
signal.alarm(3)
result = cli.main(['aa-test', 'mlp_stack', '--output-root', 'evidence'])
done_at_return = done.is_set()
alive_at_return = pumps[0].is_alive()
signal.alarm(0)
payload = {{
    'alive_at_return': alive_at_return,
    'done_at_return': done_at_return,
    'result': result,
}}
print(json.dumps(payload, sort_keys=True))
expected = {{'alive_at_return': False, 'done_at_return': True, 'result': 1}}
raise SystemExit(0 if payload == expected else 1)
"""

    completed = subprocess.run(
        [str(PYTHON), "-c", source],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
        timeout=5,
    )

    assert completed.returncode == 0, (
        f"stdout={completed.stdout!r} stderr={completed.stderr!r}"
    )
    assert json.loads(completed.stdout) == {
        "alive_at_return": False,
        "done_at_return": True,
        "result": 1,
    }


def test_failed_worker_termination_cannot_make_rollback_join_unbounded() -> None:
    """Failed kill/wait and uncooperative pumps must still return with cleanup failure."""
    source = f"""
import json
import os
import signal
import subprocess
import threading
from contextlib import redirect_stderr
from io import StringIO

from hephaestus import cli

marker = {SCIENCE_START_MARKER!r}
real_thread = threading.Thread
pumps = []
processes = []

class UncooperativeSource:
    def __init__(self):
        self.closed = False
        self.entered = threading.Event()
        self.block = threading.Event()

    def read(self, size):
        del size
        self.entered.set()
        self.block.wait()
        return b''

    def close(self):
        self.closed = True

class ControlledThread(real_thread):
    def __init__(self, *args, **kwargs):
        self.stream_target = kwargs.get('target') is cli._forward_worker_stream
        super().__init__(*args, **kwargs)
        if self.stream_target:
            pumps.append(self)

class Process:
    def __init__(self):
        self.stdout = UncooperativeSource()
        self.stderr = UncooperativeSource()
        self.wait_calls = 0

    def kill(self):
        raise OSError('injected kill failure')

    def wait(self, timeout=None):
        self.wait_calls += 1
        if timeout is None:
            raise RuntimeError('injected initial wait failure')
        raise subprocess.TimeoutExpired('worker', timeout)

def started(*args, **kwargs):
    del args
    os.write(kwargs['pass_fds'][0], marker)
    process = Process()
    processes.append(process)
    return process

threading.Thread = ControlledThread
cli.subprocess.Popen = started
cli._WORKER_CLEANUP_TIMEOUT_SECONDS = 0.05
cli._WORKER_CONTROL_POLL_SECONDS = 0.01
errors = StringIO()
signal.alarm(2)
with redirect_stderr(errors):
    result = cli.main(['aa-test', 'mlp_stack', '--output-root', 'evidence'])
process = processes[0]
payload = {{
    'cleanup_reported': 'cleanup failure:' in errors.getvalue(),
    'pumps_live': sum(thread.is_alive() for thread in pumps),
    'result': result,
    'sources_closed': process.stdout.closed and process.stderr.closed,
}}
print(json.dumps(payload, sort_keys=True))
expected = {{
    'cleanup_reported': True,
    'pumps_live': 2,
    'result': 1,
    'sources_closed': True,
}}
raise SystemExit(0 if payload == expected else 1)
"""

    completed = subprocess.run(
        [str(PYTHON), "-c", source],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
        timeout=4,
    )

    assert completed.returncode == 0, (
        f"stdout={completed.stdout!r} stderr={completed.stderr!r}"
    )
    assert json.loads(completed.stdout) == {
        "cleanup_reported": True,
        "pumps_live": 2,
        "result": 1,
        "sources_closed": True,
    }


def test_missing_worker_pipe_uses_unified_process_and_channel_rollback() -> None:
    """A spawned worker with one missing pipe must be killed, waited, and fully closed."""
    source = f"""
import io
import json
import os

from hephaestus import cli

marker = {SCIENCE_START_MARKER!r}
real_pipe = os.pipe
created = []
processes = []

def identity(descriptor):
    status = os.fstat(descriptor)
    return (status.st_dev, status.st_ino, status.st_mode, status.st_rdev)

def tracked_pipe():
    descriptors = real_pipe()
    created.extend((descriptor, identity(descriptor)) for descriptor in descriptors)
    return descriptors

class Process:
    def __init__(self):
        self.stdout = None
        self.stderr = io.BytesIO()
        self.killed = False
        self.waited = False

    def kill(self):
        self.killed = True

    def wait(self, timeout=None):
        del timeout
        self.waited = True
        return -9

def started(*args, **kwargs):
    del args
    os.write(kwargs['pass_fds'][0], marker)
    process = Process()
    processes.append(process)
    return process

os.pipe = tracked_pipe
cli.subprocess.Popen = started
result = cli.main(['aa-test', 'mlp_stack', '--output-root', 'evidence'])
leaked = []
for descriptor, token in created:
    try:
        current = identity(descriptor)
    except OSError:
        continue
    if current == token:
        leaked.append(descriptor)
process = processes[0]
payload = {{
    'killed': process.killed,
    'leaked': leaked,
    'result': result,
    'stream_closed': process.stderr.closed,
    'waited': process.waited,
}}
print(json.dumps(payload, sort_keys=True))
expected = {{
    'killed': True,
    'leaked': [],
    'result': 1,
    'stream_closed': True,
    'waited': True,
}}
raise SystemExit(0 if payload == expected else 1)
"""

    completed = subprocess.run(
        [str(PYTHON), "-c", source],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
        timeout=5,
    )

    assert completed.returncode == 0, completed.stderr
    assert json.loads(completed.stdout) == {
        "killed": True,
        "leaked": [],
        "result": 1,
        "stream_closed": True,
        "waited": True,
    }


def test_exec_worker_parent_maps_signal_and_bootstrap_failures(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Signal, spawn, and bootstrap failures are named apart from scientific exit one."""
    from hephaestus import cli

    monkeypatch.setattr(
        cli.subprocess,
        "Popen",
        _fake_worker_start(_FakeWorkerProcess(b"science\n", b"", -15)),
    )
    signal_stdout = StringIO()
    signal_stderr = StringIO()
    with redirect_stdout(signal_stdout), redirect_stderr(signal_stderr):
        assert cli.main(["aa-test", "mlp_stack", "--output-root", "evidence"]) == 1
    assert signal_stdout.getvalue() == "science\n"
    assert "signal failure" in signal_stderr.getvalue()
    assert "signal 15" in signal_stderr.getvalue()

    def failed_spawn(*args: object, **kwargs: object) -> _FakeWorkerProcess:
        raise OSError("worker unavailable")

    monkeypatch.setattr(cli.subprocess, "Popen", failed_spawn)
    spawn_stderr = StringIO()
    with redirect_stderr(spawn_stderr):
        assert cli.main(["aa-test", "mlp_stack", "--output-root", "evidence"]) == 1
    assert "spawn failure" in spawn_stderr.getvalue()

    monkeypatch.setattr(
        cli.subprocess,
        "Popen",
        _fake_worker_start(
            _FakeWorkerProcess(
                b"", b"durable worker bootstrap failed: invalid payload\n", 70
            ),
            control=b"",
        ),
    )
    bootstrap_stderr = StringIO()
    with redirect_stderr(bootstrap_stderr):
        assert cli.main(["aa-test", "mlp_stack", "--output-root", "evidence"]) == 1
    assert "bootstrap" in bootstrap_stderr.getvalue()
    assert "spawn failure" not in bootstrap_stderr.getvalue()


def test_multithreaded_parent_uses_exec_worker_without_fork_warning(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A threaded embedding must not run arbitrary Python in a forked child."""
    from hephaestus import cli

    monkeypatch.setattr(
        cli.subprocess,
        "Popen",
        _fake_worker_start(_FakeWorkerProcess(b"ok\n", b"", 0)),
    )
    monkeypatch.setattr(cli.subprocess, "run", lambda *args, **kwargs: pytest.fail("buffered run"))
    result: list[int] = []
    with warnings.catch_warnings(record=True) as caught:
        thread = threading.Thread(
            target=lambda: result.append(
                cli.main(["demo-planted-regressions", "--output-root", "evidence"])
            )
        )
        thread.start()
        thread.join(timeout=2)

    assert not thread.is_alive()
    assert result == [0]
    assert not caught


def test_worker_flushes_rendered_evidence_before_descriptor_cleanup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Buffered scientific evidence must be flushed before cleanup can report or fail."""
    from argparse import Namespace

    from hephaestus import cli, durability, durable_worker

    events: list[str] = []

    class RecordingStream(StringIO):
        def __init__(self, name: str) -> None:
            super().__init__()
            self.name = name

        def flush(self) -> None:
            events.append(f"flush:{self.name}")
            super().flush()

    class Parser:
        def parse_args(self, arguments: list[str]) -> Namespace:
            del arguments
            return Namespace(
                command="aa-test",
                output_root=Path("evidence"),
                allow_volatile_output=True,
            )

    class AttemptRoot:
        def __enter__(self) -> AttemptRoot:
            return self

        def enter_worker_directory(self) -> None:
            return None

        def close(self) -> tuple[int, ...]:
            events.append("close")
            return ()

    monkeypatch.setattr(cli, "_parser", lambda: Parser())
    monkeypatch.setattr(cli, "_run_after_parse", lambda parser, parsed: print("science") or 1)
    monkeypatch.setattr(
        durability,
        "prepare_attempt_output_root",
        lambda *args, **kwargs: AttemptRoot(),
    )
    stdout = RecordingStream("stdout")
    stderr = RecordingStream("stderr")
    control_read_fd, control_write_fd = os.pipe()

    try:
        with redirect_stdout(stdout), redirect_stderr(stderr):
            assert durable_worker.main(
                [json.dumps(["aa-test", "mlp_stack"]), str(control_write_fd)]
            ) == 1
        assert os.read(control_read_fd, 4096) == SCIENCE_START_MARKER
    finally:
        with suppress(OSError):
            os.close(control_write_fd)
        os.close(control_read_fd)

    assert stdout.getvalue() == "science\n"
    assert events.index("flush:stdout") < events.index("close")
    assert events.index("flush:stderr") < events.index("close")


def test_direct_orchestrators_keep_absolute_output_root_behavior(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The worker-only relative root must not weaken ordinary programmatic callers."""
    from hephaestus.aa_runtime import TrustedAAOrchestrator
    from hephaestus.catalog_runtime import CatalogRuntime
    from hephaestus.demo import TrustedDemoOrchestrator
    from hephaestus.measure import RunSettings

    monkeypatch.chdir(tmp_path)
    criteria = ROOT / "gates" / "default.yaml"
    capabilities = {
        "schema_version": 1,
        "torch_version": "2.13.0",
        "compiler_backends": ["inductor"],
        "inductor_modes": ["default", "max-autotune-no-cudagraphs", "reduce-overhead"],
        "inductor_options": ["epilogue_fusion"],
    }

    aa = TrustedAAOrchestrator(Path("relative"), criteria)
    demo = TrustedDemoOrchestrator(Path("relative"), criteria)
    runtime = CatalogRuntime(
        Path("relative") / "runs",
        criteria,
        RunSettings(schema_version=2, repeats=64),
        _host_state_sink=SimpleNamespace(
            append=lambda capture, bundle_path: None,
        ),
        capability_snapshot=capabilities,
    )

    assert aa._output_root == (tmp_path / "relative")
    assert demo._output_root == (tmp_path / "relative")
    assert runtime._runs_root == (tmp_path / "relative" / "runs")


def test_post_science_ledger_error_preserves_the_aa_verdict_and_parent_path(
    tmp_path: Path,
) -> None:
    """Converting a completed receipt to runtime.failure would discard scientific evidence."""
    output_root = tmp_path / "output"
    parent = output_root / "parent"
    source = f"""
class DurabilityError(ValueError):
    pass
class AttemptRoot:
    workflow_root = Path({str(output_root)!r})
    def __enter__(self):
        return self
    def __exit__(self, *args):
        return None
    def enter_worker_directory(self):
        return None
    def append_attempt(self, **kwargs):
        raise OSError('ledger unavailable after science')
module.DurabilityError = DurabilityError
module.prepare_attempt_output_root = lambda *args, **kwargs: AttemptRoot()
"""
    durability_source = source
    aa_source = f"""
parent = Path({str(parent)!r})
parent.mkdir(parents=True)
statistics = NS(
    signed_effects=(0.0,), bootstrap_absolute_medians=(0.0,), p95_noise_floor=0.125,
    speedup_lower_bound_a_over_b=1.0, speedup_lower_bound_b_over_a=1.0,
)
module.run_aa_test = lambda *args, **kwargs: NS(
    parent_path=parent, verdict='FAIL', driving_finding='methodology.noise_floor',
    statistics=statistics,
)
"""
    code = f"""
import sys, types
from pathlib import Path
from types import SimpleNamespace as NS
module = types.ModuleType('hephaestus.durability')
{durability_source}
sys.modules['hephaestus.durability'] = module
module = types.ModuleType('hephaestus.aa_runtime')
{aa_source}
sys.modules['hephaestus.aa_runtime'] = module
"""
    with tempfile.TemporaryDirectory() as temporary:
        Path(temporary, "sitecustomize.py").write_text(code, encoding="utf-8")
        environment = os.environ.copy()
        existing = environment.get("PYTHONPATH")
        environment["PYTHONPATH"] = (
            temporary if not existing else f"{temporary}{os.pathsep}{existing}"
        )
        completed = _run_direct_worker(
            [
                "aa-test",
                "mlp_stack",
                "--output-root",
                str(output_root),
                "--allow-volatile-output",
            ],
            environment=environment,
        )

    assert completed.returncode == 1
    payload = json.loads(completed.stdout)
    assert payload["verdict"] == "FAIL"
    assert payload["evidence_path"] == str(parent.resolve())
    assert "ledger unavailable after science" in completed.stderr

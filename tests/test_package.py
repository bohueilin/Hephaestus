from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
import tarfile
import zipfile
from pathlib import Path


def test_package_exposes_version() -> None:
    import hephaestus

    assert hephaestus.__version__ == "0.1.0"


def test_sdist_built_wheel_contains_exact_gate_and_resolves_outside_checkout(
    tmp_path: Path,
) -> None:
    """Release artifacts must carry canonical gate bytes without repository/dependency help."""
    root = Path(__file__).parents[1]
    canonical_gate = (root / "gates/default.yaml").read_bytes()
    uv = shutil.which("uv")
    assert uv is not None
    dist = tmp_path / "dist"
    dist.mkdir()

    subprocess.run(
        [uv, "build", "--sdist", "--out-dir", str(dist), str(root)],
        check=True,
        capture_output=True,
        text=True,
    )
    sdist = next(dist.glob("*.tar.gz"))
    with tarfile.open(sdist) as archive:
        assert any(name.endswith("/gates/default.yaml") for name in archive.getnames())
    wheel_dist = tmp_path / "wheel"
    wheel_dist.mkdir()
    subprocess.run(
        [uv, "build", "--wheel", "--out-dir", str(wheel_dist), str(sdist)],
        check=True,
        capture_output=True,
        text=True,
    )
    wheel = next(wheel_dist.glob("*.whl"))
    with zipfile.ZipFile(wheel) as archive:
        members = [name for name in archive.namelist() if name.endswith("/gates/default.yaml")]
        assert members == ["hephaestus/gates/default.yaml"]
        assert archive.read(members[0]) == canonical_gate

    venv = tmp_path / "isolated"
    subprocess.run(
        [str(root / ".venv/bin/python"), "-m", "venv", str(venv)],
        check=True,
        capture_output=True,
        text=True,
    )
    python = venv / "bin/python"
    subprocess.run(
        [str(python), "-m", "pip", "install", "--no-deps", str(wheel)],
        check=True,
        capture_output=True,
        text=True,
    )
    outside = tmp_path / "outside"
    outside.mkdir()
    environment = os.environ.copy()
    environment.pop("PYTHONPATH", None)
    environment["PYTHONNOUSERSITE"] = "1"
    help_result = subprocess.run(
        [str(venv / "bin/hephaestus"), "--help"],
        cwd=outside,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )
    assert "demo-planted-regressions" in help_result.stdout
    resolver = subprocess.run(
        [
            str(python),
            "-c",
            (
                "import hashlib, hephaestus; from pathlib import Path; "
                "from hephaestus.criteria import packaged_criteria_path; "
                "ctx=packaged_criteria_path(); p=ctx.__enter__(); "
                "print(Path(hephaestus.__file__).resolve()); "
                "print(hashlib.sha256(p.read_bytes()).hexdigest()); ctx.__exit__(None,None,None)"
            ),
        ],
        cwd=outside,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )
    imported_path, gate_hash = resolver.stdout.splitlines()
    assert Path(imported_path).is_relative_to(venv)
    assert gate_hash == hashlib.sha256(canonical_gate).hexdigest()

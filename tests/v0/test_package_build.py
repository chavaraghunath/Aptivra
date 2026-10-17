"""Historical wheel-resource and optional-dependency assertions for v0."""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

from aptivra.v0.resources import aptivra_resource

_PROJECT_ROOT = Path(__file__).resolve().parents[1] / ".."
_HUB_TEMPLATE_DIR = _PROJECT_ROOT / "aptivra" / "v0" / "resources" / "huggingface_hub"


def _stage_hub_templates_for_static_analysis(destination: Path) -> Path:
    package_dir = destination / "aptivra_hub_remote_code"
    package_dir.mkdir()
    (package_dir / "__init__.py").touch()
    for source in sorted(_HUB_TEMPLATE_DIR.glob("*.py.template")):
        shutil.copy2(source, package_dir / source.name.removesuffix(".template"))
    return package_dir


def _run_static_analysis(
    package_dir: Path,
    cache_dir: Path,
) -> tuple[subprocess.CompletedProcess[str], subprocess.CompletedProcess[str]]:
    ruff = subprocess.run(
        [sys.executable, "-m", "ruff", "check", "--no-cache", str(package_dir)],
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
        cwd=_PROJECT_ROOT,
    )
    mypy = subprocess.run(
        [
            sys.executable,
            "-m",
            "mypy",
            "--strict",
            "--config-file",
            str(_PROJECT_ROOT / "pyproject.toml"),
            "--cache-dir",
            str(cache_dir),
            str(package_dir),
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=60,
        cwd=_PROJECT_ROOT,
    )
    return ruff, mypy


def test_hub_remote_code_templates_pass_ruff_and_strict_mypy(tmp_path: Path) -> None:
    package_dir = _stage_hub_templates_for_static_analysis(tmp_path)

    ruff, mypy = _run_static_analysis(package_dir, tmp_path / "mypy-cache")

    assert ruff.returncode == 0, ruff.stdout + ruff.stderr
    assert mypy.returncode == 0, mypy.stdout + mypy.stderr


def test_hub_remote_code_static_analysis_rejects_lint_and_type_errors(
    tmp_path: Path,
) -> None:
    package_dir = _stage_hub_templates_for_static_analysis(tmp_path)
    configuration = package_dir / "configuration_aptivra.py"
    configuration.write_text(
        configuration.read_text(encoding="utf-8")
        + '\ninvalid_remote_type: int = "not-an-int"\nundefined_remote_name\n',
        encoding="utf-8",
    )

    ruff, mypy = _run_static_analysis(package_dir, tmp_path / "mypy-cache")

    assert ruff.returncode != 0
    assert "F821" in ruff.stdout
    assert "undefined_remote_name" in ruff.stdout
    assert mypy.returncode != 0
    assert "Incompatible types in assignment" in mypy.stdout
    assert "[assignment]" in mypy.stdout


def test_gguf_export_module_import_is_lazy_when_dependency_is_unavailable() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; "
                "sys.modules['gguf'] = None; "
                "import aptivra.v0.export.gguf_exporter"
            ),
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
        cwd=_PROJECT_ROOT,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_historical_release_gate_config_is_byte_identical() -> None:
    assert (_PROJECT_ROOT / "configs/v0/model_card_release_gate.yaml").read_bytes() == (
        aptivra_resource("configs", "model_card_release_gate.yaml").read_bytes()
    )

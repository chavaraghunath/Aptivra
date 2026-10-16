"""Regression tests for distribution discovery and packaged runtime resources."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

from aptivra.eval.config import load_experiment_settings
from aptivra.eval.model_card_eval import (
    DEFAULT_HELDOUT_PROMPTS_PATH,
    DEFAULT_HELDOUT_TEXT_PATH,
    load_prompt_answers,
)
from aptivra.eval.task_stream_audit import load_audit_plan
from aptivra.export.model_card import load_release_gate_settings
from aptivra.inference.standard_base import load_standard_base_settings
from aptivra.resources import aptivra_resource, require_resource
from aptivra.tasks.code_sandbox import load_sandbox_config
from aptivra.tasks.github_issues import load_github_issue_builder_config

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_HUB_TEMPLATE_DIR = _PROJECT_ROOT / "aptivra" / "resources" / "huggingface_hub"


def _mutation_runtime_path() -> str | None:
    if "MUTANT_UNDER_TEST" not in os.environ:
        return None

    import mutmut

    return str(Path(mutmut.__file__).resolve().parent.parent)


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


def test_wheel_contains_only_aptivra_packages_and_required_resources(
    tmp_path: Path,
) -> None:
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    for filename in ("pyproject.toml", "README.md", "LICENSE"):
        shutil.copy2(_PROJECT_ROOT / filename, source_dir / filename)
    shutil.copytree(_PROJECT_ROOT / "aptivra", source_dir / "aptivra")

    dist_dir = tmp_path / "dist"
    result = subprocess.run(
        [
            "uv",
            "build",
            "--wheel",
            "--offline",
            "--no-build-isolation",
            "--no-create-gitignore",
            "--python",
            sys.executable,
            "--out-dir",
            str(dist_dir),
            str(source_dir),
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr

    wheels = list(dist_dir.glob("aptivra-*.whl"))
    assert len(wheels) == 1
    with zipfile.ZipFile(wheels[0]) as wheel:
        members = set(wheel.namelist())
        metadata_name = next(name for name in members if name.endswith(".dist-info/METADATA"))
        metadata = wheel.read(metadata_name).decode("utf-8")

    expected_python = {
        path.relative_to(source_dir).as_posix()
        for path in (source_dir / "aptivra").rglob("*.py")
        if "__pycache__" not in path.parts
    }
    assert expected_python <= members
    assert {
        "aptivra/eval/model_card_heldout_v1.json",
        "aptivra/eval/model_card_heldout_text_v1.txt",
        "aptivra/resources/configs/model_card_release_gate.yaml",
        "aptivra/resources/configs/smollm2_base.yaml",
        "aptivra/resources/configs/github_issues_stream.yaml",
        "aptivra/resources/configs/task_stream.yaml",
        "aptivra/resources/configs/task_stream_audit.yaml",
        "aptivra/resources/configs/task_stream_v2.yaml",
        "aptivra/resources/configs/verifier_sandbox.yaml",
        "aptivra/resources/huggingface_hub/configuration_aptivra.py.template",
        "aptivra/resources/huggingface_hub/modeling_aptivra.py.template",
        "aptivra/resources/huggingface_hub/tokenization_aptivra.py.template",
    } <= members
    assert any(name.endswith(".dist-info/licenses/LICENSE") for name in members)
    for forbidden_prefix in ("configs/", "deploy/", "experiments/", "multitier_c/"):
        assert not any(name.startswith(forbidden_prefix) for name in members)

    assert "License-Expression: Apache-2.0" in metadata
    assert "License-File: LICENSE" in metadata
    assert "Requires-Python: >=3.11" in metadata
    assert "Project-URL: Repository, https://github.com/chavaraghunath/Aptivra" in metadata
    assert (
        "Project-URL: Model checkpoint, "
        "https://huggingface.co/raghunath1/aptivra-200m-tiered-memory"
    ) in metadata

    installed_dir = tmp_path / "installed"
    outside_dir = tmp_path / "outside-repository"
    installed_dir.mkdir()
    outside_dir.mkdir()
    with zipfile.ZipFile(wheels[0]) as wheel:
        wheel.extractall(installed_dir)
    import_arguments = [
        sys.executable,
        "-I",
        "-c",
        (
            "import importlib, os, pkgutil, sys; "
            "from pathlib import Path; "
            "outside = Path.cwd(); "
            "sys.path.insert(0, sys.argv[1]) if len(sys.argv) > 1 else None; "
            "os.chdir(sys.argv[2]) if len(sys.argv) > 2 else None; "
            "importlib.import_module('mutmut.configuration').config() "
            "if len(sys.argv) > 2 else None; "
            "os.chdir(outside) if len(sys.argv) > 2 else None; "
            f"installed = Path({str(installed_dir)!r}).resolve(); "
            "sys.path.insert(0, str(installed)); "
            "import aptivra; "
            "package_file = Path(aptivra.__file__).resolve(); "
            "assert installed in package_file.parents, package_file; "
            "modules = sorted(info.name for info in "
            "pkgutil.walk_packages(aptivra.__path__, aptivra.__name__ + '.')); "
            "[importlib.import_module(name) for name in modules]; "
            "from aptivra.resources import aptivra_resource; "
            "from aptivra.eval.config import load_experiment_settings; "
            "from aptivra.eval.task_stream_audit import load_audit_plan; "
            "from aptivra.tasks.github_issues import "
            "load_github_issue_builder_config; "
            "assert load_experiment_settings(aptivra_resource('configs', "
            "'task_stream_v2.yaml')).task_stream.version == 'v2'; "
            "assert len(load_audit_plan(aptivra_resource('configs', "
            "'task_stream_audit.yaml')).streams) == 2; "
            "assert load_github_issue_builder_config().source == "
            "'gh_archive_clickhouse'; "
            f"assert len(modules) + 1 == {len(expected_python)}; "
            "print(f'imported {len(modules) + 1} wheel modules')"
        ),
    ]
    mutation_runtime = _mutation_runtime_path()
    if mutation_runtime is not None:
        import_arguments.extend([mutation_runtime, str(_PROJECT_ROOT)])
    import_sweep = subprocess.run(
        import_arguments,
        check=False,
        capture_output=True,
        text=True,
        timeout=300,
        cwd=outside_dir,
    )
    assert import_sweep.returncode == 0, import_sweep.stdout + import_sweep.stderr
    assert import_sweep.stdout == f"imported {len(expected_python)} wheel modules\n"


def test_packaged_defaults_load_without_repository_relative_paths() -> None:
    release_gate = load_release_gate_settings()
    sandbox = load_sandbox_config()
    experiment = load_experiment_settings()
    v2_experiment = load_experiment_settings(
        aptivra_resource("configs", "task_stream_v2.yaml")
    )
    audit_plan = load_audit_plan(
        aptivra_resource("configs", "task_stream_audit.yaml")
    )
    github_issues = load_github_issue_builder_config()
    standard_base = load_standard_base_settings()
    heldout = load_prompt_answers(DEFAULT_HELDOUT_PROMPTS_PATH)

    assert release_gate.minimum_heldout_prompt_samples == len(heldout)
    assert sandbox.timeout_seconds == 1.0
    assert experiment.task_stream.seed == 42
    assert v2_experiment.task_stream.version == "v2"
    assert len(audit_plan.streams) == 2
    assert github_issues.source == "gh_archive_clickhouse"
    assert standard_base.feature.enabled is False
    assert standard_base.source.revision == "f8027fd0eaeea54caa13c31d31b9fdc459c38b49"
    assert len(DEFAULT_HELDOUT_TEXT_PATH.read_bytes()) >= 8192


@pytest.mark.parametrize(
    ("repository_path", "resource_parts"),
    [
        (
            "configs/github_issues_stream.yaml",
            ("configs", "github_issues_stream.yaml"),
        ),
        (
            "configs/model_card_release_gate.yaml",
            ("configs", "model_card_release_gate.yaml"),
        ),
        (
            "configs/smollm2_base.yaml",
            ("configs", "smollm2_base.yaml"),
        ),
        ("configs/task_stream.yaml", ("configs", "task_stream.yaml")),
        (
            "configs/task_stream_audit.yaml",
            ("configs", "task_stream_audit.yaml"),
        ),
        (
            "configs/task_stream_v2.yaml",
            ("configs", "task_stream_v2.yaml"),
        ),
        ("configs/verifier_sandbox.yaml", ("configs", "verifier_sandbox.yaml")),
    ],
)
def test_repository_config_copy_matches_packaged_default(
    repository_path: str,
    resource_parts: tuple[str, ...],
) -> None:
    assert (_PROJECT_ROOT / repository_path).read_bytes() == aptivra_resource(
        *resource_parts
    ).read_bytes()


def test_missing_packaged_resource_names_the_required_file() -> None:
    with pytest.raises(FileNotFoundError, match="missing-default.yaml"):
        require_resource("aptivra.resources", "configs", "missing-default.yaml")


def test_gguf_export_module_import_is_lazy_when_dependency_is_unavailable() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; "
                "sys.modules['gguf'] = None; "
                "import aptivra.export.gguf_exporter"
            ),
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
        cwd=_PROJECT_ROOT,
    )
    assert result.returncode == 0, result.stdout + result.stderr

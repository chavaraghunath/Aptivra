"""Pre-network ordering and fail-closed destination tests for lane 7."""

from __future__ import annotations

import json
import shutil
from collections.abc import Callable
from dataclasses import FrozenInstanceError, replace
from pathlib import Path

import pytest
from safetensors import safe_open
from safetensors.torch import save_file

import aptivra.export.hf_hub_artifacts as hub_artifacts
import aptivra.export.hf_push as hf_push
from aptivra.export.hf_push import (
    HfPublicationPlan,
    create_hf_repository,
    prepare_hf_publication,
    push_aptivra_model_to_hf,
    tag_hf_repository,
)
from tests.test_standard_packaging import (
    PackageFixture,
    _release_config,
    package_fixture,
)

__all__ = ["package_fixture"]


def _forbidden_client_factory(calls: list[str]) -> Callable[[], object]:
    def construct() -> object:
        calls.append("constructed")
        raise AssertionError("network client must not be constructed")

    return construct


def _rewrite_package_manifest(stage: Path, update: Callable[[dict[str, object]], None]) -> None:
    path = stage / hub_artifacts.STAGING_MANIFEST
    manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest.pop("manifest_sha256")
    update(manifest)
    manifest["manifest_sha256"] = hub_artifacts.canonical_sha256(manifest)
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _refresh_file_manifest(stage: Path) -> None:
    hub_artifacts._write_file_manifest(stage)


def _repository_config(fixture: PackageFixture, path: Path) -> Path:
    manifest = json.loads(
        (fixture.stage / hub_artifacts.STAGING_MANIFEST).read_text(encoding="utf-8")
    )
    return _release_config(path, repository_id=manifest["base"]["repository"])


class RecordingClient:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, object]]] = []

    def create_repo(self, **kwargs: object) -> None:
        self.calls.append(("create_repo", kwargs))

    def upload_folder(self, **kwargs: object) -> None:
        self.calls.append(("upload_folder", kwargs))

    def create_tag(self, **kwargs: object) -> None:
        self.calls.append(("create_tag", kwargs))


@pytest.mark.parametrize("operation", ["create", "push", "tag"])
def test_null_owner_repository_blocks_every_network_path_before_client_construction(
    package_fixture: PackageFixture,
    operation: str,
) -> None:
    calls: list[str] = []
    factory = _forbidden_client_factory(calls)
    with pytest.raises(RuntimeError, match="remains null"):
        if operation == "create":
            create_hf_repository(
                config_path=package_fixture.config_path,
                client_factory=factory,
            )
        elif operation == "push":
            push_aptivra_model_to_hf(
                package_fixture.stage,
                config_path=package_fixture.config_path,
                client_factory=factory,
            )
        else:
            tag_hf_repository(
                package_fixture.stage,
                "v1",
                config_path=package_fixture.config_path,
                client_factory=factory,
            )
    assert calls == []


def _mutate_filename(stage: Path) -> None:
    path = stage / "docs" / "github-note.txt"
    path.parent.mkdir()
    path.write_text("safe body", encoding="utf-8")


def _mutate_text(stage: Path) -> None:
    path = stage / "NOTICE"
    path.write_text(path.read_text(encoding="utf-8") + "\nGitHub\n", encoding="utf-8")


def _mutate_json_value(stage: Path) -> None:
    path = stage / "adapters" / "identity" / "adapter_config.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["auto_map"] = {"model": "forbidden"}
    path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")
    digest = hub_artifacts.sha256_file(path)

    def update(manifest: dict[str, object]) -> None:
        adapters = manifest["adapters"]
        assert isinstance(adapters, list)
        entry = adapters[0]
        assert isinstance(entry, dict)
        files = entry["files"]
        assert isinstance(files, dict)
        files["adapter_config.json"] = digest

    _rewrite_package_manifest(stage, update)


def _mutate_model_card_field(stage: Path) -> None:
    path = stage / "README.md"
    text = path.read_text(encoding="utf-8")
    path.write_text(text.replace("- research", "- github"), encoding="utf-8")


def _mutate_template(stage: Path) -> None:
    path = stage / hub_artifacts.CHAT_TEMPLATE
    path.write_text(path.read_text(encoding="utf-8") + "github", encoding="utf-8")


def _mutate_python_file(stage: Path) -> None:
    (stage / "modeling.py").write_text("pass\n", encoding="utf-8")


def _mutate_tensor_metadata(stage: Path) -> None:
    path = stage / hub_artifacts.ROUTER_WEIGHTS
    with safe_open(path, framework="pt", device="cpu") as handle:
        tensor_names = list(handle.keys())
        tensors = {name: handle.get_tensor(name) for name in tensor_names}
        metadata = dict(handle.metadata() or {})
    metadata["source"] = "GitHub"
    save_file(tensors, path, metadata=metadata)
    digest = hub_artifacts.sha256_file(path)

    def update(manifest: dict[str, object]) -> None:
        router = manifest["router"]
        assert isinstance(router, dict)
        router["weights_sha256"] = digest

    _rewrite_package_manifest(stage, update)


@pytest.mark.parametrize(
    "mutation",
    [
        _mutate_filename,
        _mutate_text,
        _mutate_json_value,
        _mutate_model_card_field,
        _mutate_template,
        _mutate_python_file,
        _mutate_tensor_metadata,
    ],
    ids=[
        "filename",
        "text",
        "json-value",
        "model-card-field",
        "template",
        "python-file",
        "safetensors-metadata",
    ],
)
def test_every_prohibited_surface_fails_before_client_construction(
    package_fixture: PackageFixture,
    tmp_path: Path,
    mutation: Callable[[Path], None],
) -> None:
    stage = tmp_path / "stage"
    shutil.copytree(package_fixture.stage, stage)
    mutation(stage)
    _refresh_file_manifest(stage)
    config = _repository_config(package_fixture, tmp_path / "release.yaml")
    calls: list[str] = []

    with pytest.raises(ValueError, match="prohibited|Python files"):
        push_aptivra_model_to_hf(
            stage,
            config_path=config,
            client_factory=_forbidden_client_factory(calls),
        )

    assert calls == []


def test_verified_create_push_and_tag_use_only_the_configured_repository(
    package_fixture: PackageFixture,
    tmp_path: Path,
) -> None:
    config = _repository_config(package_fixture, tmp_path / "release.yaml")
    repository_id = json.loads(
        (package_fixture.stage / hub_artifacts.STAGING_MANIFEST).read_text(
            encoding="utf-8"
        )
    )["base"]["repository"]

    create_client = RecordingClient()
    assert create_hf_repository(
        private=True,
        config_path=config,
        client_factory=lambda: create_client,
    ) == repository_id
    assert create_client.calls == [
        (
            "create_repo",
            {
                "repo_id": repository_id,
                "repo_type": "model",
                "exist_ok": False,
                "private": True,
            },
        )
    ]

    default_client = RecordingClient()
    create_hf_repository(
        config_path=config,
        client_factory=lambda: default_client,
    )
    assert default_client.calls[0][1]["private"] is False

    push_client = RecordingClient()
    assert push_aptivra_model_to_hf(
        package_fixture.stage,
        config_path=config,
        client_factory=lambda: push_client,
    ) == repository_id
    assert [name for name, _ in push_client.calls] == ["create_repo", "upload_folder"]
    assert push_client.calls[0][1] == {
        "repo_id": repository_id,
        "repo_type": "model",
        "exist_ok": False,
        "private": False,
    }
    upload = push_client.calls[1][1]
    staged = hub_artifacts.verify_staged_repository(
        package_fixture.stage, config_path=config
    )
    assert upload["allow_patterns"] == list(staged.files)
    assert upload["repo_id"] == repository_id

    tag_client = RecordingClient()
    assert tag_hf_repository(
        package_fixture.stage,
        "v1",
        config_path=config,
        client_factory=lambda: tag_client,
    ) == repository_id
    assert tag_client.calls == [
        (
            "create_tag",
            {"repo_id": repository_id, "tag": "v1", "repo_type": "model"},
        )
    ]


def test_publication_plan_detects_post_validation_change(
    package_fixture: PackageFixture,
    tmp_path: Path,
) -> None:
    stage = tmp_path / "stage"
    shutil.copytree(package_fixture.stage, stage)
    config = _repository_config(package_fixture, tmp_path / "release.yaml")
    plan = prepare_hf_publication(stage, config_path=config)
    assert isinstance(plan, HfPublicationPlan)
    (stage / "NOTICE").write_text("changed", encoding="utf-8")
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        plan.verify(config_path=config)


def test_publication_plan_rechecks_manifest_identity_and_each_file(
    package_fixture: PackageFixture,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stage = tmp_path / "stage"
    shutil.copytree(package_fixture.stage, stage)
    config = _repository_config(package_fixture, tmp_path / "release.yaml")
    plan = prepare_hf_publication(stage, config_path=config)
    with pytest.raises(FrozenInstanceError):
        plan.repository_id = "changed/value"
    original_verify = hf_push.verify_staged_repository

    monkeypatch.setattr(
        hf_push,
        "verify_staged_repository",
        lambda *args, **kwargs: replace(
            original_verify(*args, **kwargs), file_manifest_sha256="0" * 64
        ),
    )
    with pytest.raises(ValueError, match="changed after publication planning"):
        plan.verify(config_path=config)

    def verify_then_change(*args: object, **kwargs: object) -> object:
        current = original_verify(*args, **kwargs)
        (stage / "NOTICE").write_text("changed after verification", encoding="utf-8")
        return current

    monkeypatch.setattr(hf_push, "verify_staged_repository", verify_then_change)
    with pytest.raises(ValueError, match="staged publication file changed"):
        plan.verify(config_path=config)
    (stage / "NOTICE").write_bytes((package_fixture.stage / "NOTICE").read_bytes())

    changed_hashes = dict(plan.file_sha256)
    changed_hashes["NOTICE"] = "0" * 64
    hash_plan = replace(plan, file_sha256=changed_hashes)
    monkeypatch.setattr(hf_push, "verify_staged_repository", original_verify)
    with pytest.raises(ValueError, match="staged publication file changed"):
        hash_plan.verify(config_path=config)

    different_manifest = replace(plan.stage, file_manifest_sha256="f" * 64)
    monkeypatch.setattr(
        hf_push,
        "verify_staged_repository",
        lambda *args, **kwargs: different_manifest,
    )
    with pytest.raises(ValueError, match="changed after publication planning"):
        plan.verify(config_path=config)


def test_publication_path_arguments_remain_keyword_capable(
    package_fixture: PackageFixture,
    tmp_path: Path,
) -> None:
    config = _repository_config(package_fixture, tmp_path / "release.yaml")
    plan = prepare_hf_publication(
        stage_path=package_fixture.stage,
        config_path=config,
    )
    plan.verify(config_path=config)
    client = RecordingClient()
    push_aptivra_model_to_hf(
        stage_path=package_fixture.stage,
        config_path=config,
        client_factory=lambda: client,
    )
    tag_hf_repository(
        stage_path=package_fixture.stage,
        tag="v1",
        config_path=config,
        client_factory=lambda: client,
    )


@pytest.mark.parametrize("tag", ["", " v1", "v1 ", "release candidate"])
def test_invalid_tag_fails_before_client_construction(
    package_fixture: PackageFixture,
    tag: str,
) -> None:
    calls: list[str] = []
    with pytest.raises(ValueError, match="tag must"):
        tag_hf_repository(
            package_fixture.stage,
            tag,
            config_path=package_fixture.config_path,
            client_factory=_forbidden_client_factory(calls),
        )
    assert calls == []

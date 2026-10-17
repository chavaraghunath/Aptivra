"""Fail-closed network boundary for a verified standard PEFT repository."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Protocol, Union, cast

from huggingface_hub import HfApi

from aptivra.export.hf_hub_artifacts import (
    StagedRepository,
    sha256_file,
    verify_staged_repository,
)
from aptivra.export.model_card import (
    DEFAULT_RELEASE_GATE_CONFIG,
    ConfigSource,
    ReleaseGateSettings,
    load_release_gate_settings,
)


class HubClient(Protocol):
    """Narrow network surface used only after every local check passes."""

    def create_repo(self, **kwargs: Any) -> Any: ...

    def upload_folder(self, **kwargs: Any) -> Any: ...

    def create_tag(self, **kwargs: Any) -> Any: ...


@dataclass(frozen=True)
class HfPublicationPlan:
    """Immutable local evidence captured before a client can be constructed."""

    repository_id: str
    stage: StagedRepository
    file_sha256: dict[str, str]

    def verify(self, *, config_path: ConfigSource) -> None:
        """Re-run exact allowlist, hash, and prohibited-content checks."""
        current = verify_staged_repository(self.stage.path, config_path=config_path)
        if current.file_manifest_sha256 != self.stage.file_manifest_sha256:
            raise ValueError("staged repository changed after publication planning")
        for relative, expected in self.file_sha256.items():
            if sha256_file(self.stage.path / relative) != expected:
                raise ValueError(f"staged publication file changed: {relative}")


def _authorized_repository(settings: ReleaseGateSettings) -> str:
    repository_id = settings.repository.repository_id
    if repository_id is None:
        raise RuntimeError(
            "new Hub repository name is owner-controlled and remains null; "
            "create, push, and tag operations are disabled"
        )
    if not settings.feature.enabled:
        raise RuntimeError("ADR-0011 standard publication is disabled by configuration")
    return repository_id


def prepare_hf_publication(
    stage_path: Union[Path, str],  # noqa: UP007
    *,
    config_path: ConfigSource = DEFAULT_RELEASE_GATE_CONFIG,
) -> HfPublicationPlan:
    """Validate repository authority and every staged byte without network access."""
    settings = load_release_gate_settings(config_path)
    repository_id = _authorized_repository(settings)
    stage = verify_staged_repository(stage_path, config_path=config_path)
    file_sha256 = {
        relative: sha256_file(stage.path / relative) for relative in stage.files
    }
    return HfPublicationPlan(repository_id, stage, file_sha256)


def _client(factory: Optional[Callable[[], HubClient]]) -> HubClient:  # noqa: UP045
    return cast(HubClient, HfApi()) if factory is None else factory()


def create_hf_repository(
    *,
    private: bool = False,
    config_path: ConfigSource = DEFAULT_RELEASE_GATE_CONFIG,
    client_factory: Optional[Callable[[], HubClient]] = None,  # noqa: UP045
) -> str:
    """Create only the owner-configured repository after fail-closed authority checks."""
    settings = load_release_gate_settings(config_path)
    repository_id = _authorized_repository(settings)
    client = _client(client_factory)
    client.create_repo(
        repo_id=repository_id,
        repo_type="model",
        exist_ok=False,
        private=private,
    )
    return repository_id


def push_aptivra_model_to_hf(
    stage_path: Union[Path, str],  # noqa: UP007
    *,
    private: bool = False,
    config_path: ConfigSource = DEFAULT_RELEASE_GATE_CONFIG,
    client_factory: Optional[Callable[[], HubClient]] = None,  # noqa: UP045
) -> str:
    """Create and upload a verified exact allowlist; never infer a destination."""
    plan = prepare_hf_publication(stage_path, config_path=config_path)
    plan.verify(config_path=config_path)
    client = _client(client_factory)
    client.create_repo(
        repo_id=plan.repository_id,
        repo_type="model",
        exist_ok=False,
        private=private,
    )
    plan.verify(config_path=config_path)
    client.upload_folder(
        folder_path=str(plan.stage.path),
        repo_id=plan.repository_id,
        repo_type="model",
        allow_patterns=list(plan.stage.files),
        commit_message="Publish verified standard PEFT adapter bundle",
    )
    return plan.repository_id


def tag_hf_repository(
    stage_path: Union[Path, str],  # noqa: UP007
    tag: str,
    *,
    config_path: ConfigSource = DEFAULT_RELEASE_GATE_CONFIG,
    client_factory: Optional[Callable[[], HubClient]] = None,  # noqa: UP045
) -> str:
    """Tag only after the exact local staged bytes are reverified."""
    if not tag or tag.strip() != tag or any(character.isspace() for character in tag):
        raise ValueError("tag must be non-empty, trimmed, and contain no whitespace")
    plan = prepare_hf_publication(stage_path, config_path=config_path)
    plan.verify(config_path=config_path)
    client = _client(client_factory)
    client.create_tag(
        repo_id=plan.repository_id,
        tag=tag,
        repo_type="model",
    )
    return plan.repository_id

"""Offline acceptance tests for the ADR-0011 standard PEFT staged repository."""

from __future__ import annotations

import json
import shutil
from collections.abc import Callable
from dataclasses import FrozenInstanceError, dataclass
from pathlib import Path

import pytest
import torch
import yaml
from peft import PeftModel
from transformers import AutoModelForCausalLM

import aptivra.export.hf_hub_artifacts as hub_artifacts
from aptivra.experts.router import SequenceTop1Router
from aptivra.experts.standard import (
    PeftExpertBank,
    StandardExpertSettings,
    standard_base_fingerprint,
)
from aptivra.export.hf_hub_artifacts import (
    FILE_MANIFEST,
    PublishableAdapter,
    stage_huggingface_hub_artifacts,
    verify_staged_repository,
)
from aptivra.export.model_card import load_release_gate_settings
from aptivra.inference.chat_canonical import (
    identity_manifest_sha256,
    load_identity_manifest,
)
from aptivra.inference.standard_base import StandardBase, StandardBaseSettings
from tests.test_standard_experts import _enabled_settings, _tiny_base, _train_one_step


@dataclass(frozen=True)
class PackageFixture:
    stage: Path
    snapshot: Path
    config_path: Path
    base_logits: torch.Tensor
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    base_settings: StandardBaseSettings
    expert_settings: StandardExpertSettings
    identity_manifest: Path
    adapter_artifact: Path


def _release_config(path: Path, *, repository_id: str | None = None) -> Path:
    payload = {
        "model_card_release_gate": {
            "schema_version": "aptivra.model_card_release_gate.v1",
            "feature": {"enabled": True},
            "repository": {"repository_id": repository_id},
            "publication": {
                "allowed_adapter_kinds": ["synthetic", "identity"],
            },
            "forbidden_references": ["github"],
        }
    }
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return path


def _identity_manifest(path: Path, base: StandardBase) -> Path:
    source = base.settings.source
    raw = yaml.safe_load(Path("configs/identity_manifest.yaml").read_text(encoding="utf-8"))
    identity = raw["identity_manifest"]
    identity["base"] = {
        "repository": source.repository,
        "revision": source.revision,
        "config_sha256": source.config.sha256,
        "weights_sha256": source.weights.sha256,
        "fingerprint": standard_base_fingerprint(base.settings),
        "unique_parameter_count": base.settings.architecture.unique_parameter_count,
    }
    identity["content_sha256"] = identity_manifest_sha256(identity)
    path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
    load_identity_manifest(path, base_settings=base.settings)
    return path


def build_package_fixture(root: Path) -> PackageFixture:
    base = _tiny_base(root / "base")
    settings = _enabled_settings()
    bank = PeftExpertBank.start_candidate(base, settings, "identity")
    _train_one_step(bank, "identity")
    input_ids = torch.tensor([[1, 3, 4, 2]], dtype=torch.long)
    attention_mask = torch.ones_like(input_ids)
    with torch.no_grad():
        base_logits = bank.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
        ).logits.detach().cpu()
    artifact = root / "identity-artifact"
    bank.promote_candidate(artifact)
    router = SequenceTop1Router(
        base.settings.architecture.hidden_size,
        settings.router.confidence_threshold,
    )
    router.add_candidate("identity", "identity")
    router.promote_candidate()
    release_config = _release_config(root / "release.yaml")
    identity_manifest = _identity_manifest(root / "identity.yaml", base)
    stage = root / "stage"
    stage_huggingface_hub_artifacts(
        stage,
        (
            PublishableAdapter(
                "identity",
                "identity",
                "identity",
                artifact,
            ),
        ),
        router,
        base_settings=base.settings,
        expert_settings=settings,
        identity_manifest_path=identity_manifest,
        config_path=release_config,
    )
    return PackageFixture(
        stage,
        base.snapshot.path,
        release_config,
        base_logits,
        input_ids,
        attention_mask,
        base.settings,
        settings,
        identity_manifest,
        artifact,
    )


@pytest.fixture(scope="module")
def package_fixture(tmp_path_factory: pytest.TempPathFactory) -> PackageFixture:
    return build_package_fixture(tmp_path_factory.mktemp("standard-package"))


def _copy_stage(fixture: PackageFixture, destination: Path) -> Path:
    shutil.copytree(fixture.stage, destination)
    return destination


def _rewrite_hashed_json(
    path: Path,
    hash_field: str,
    update: Callable[[dict[str, object]], None],
) -> None:
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload.pop(hash_field)
    update(payload)
    payload[hash_field] = hub_artifacts.canonical_sha256(payload)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _refresh_stage_file_manifest(stage: Path) -> None:
    hub_artifacts._write_file_manifest(stage)


@pytest.mark.parametrize("content", [b"\xff", b"{"])
def test_json_reader_rejects_non_utf8_and_invalid_json(
    tmp_path: Path,
    content: bytes,
) -> None:
    path = tmp_path / "invalid.json"
    path.write_bytes(content)
    with pytest.raises(ValueError, match="not valid UTF-8 JSON"):
        hub_artifacts._read_json(path, "fixture")

    with pytest.raises(ValueError, match="not valid UTF-8 JSON"):
        hub_artifacts._read_json(tmp_path / "missing.json", "fixture")


def test_staged_bundle_is_exact_standard_peft_and_reproduces_logits_offline(
    package_fixture: PackageFixture,
) -> None:
    staged = verify_staged_repository(
        package_fixture.stage,
        config_path=package_fixture.config_path,
    )
    assert staged.files == tuple(sorted(path for path in staged.files))
    assert not any(
        path.endswith((".py", ".pyi", ".pyc", ".py.template"))
        for path in staged.files
    )
    for path in staged.files:
        if not path.endswith(".safetensors"):
            text = (staged.path / path).read_text(encoding="utf-8").casefold()
            assert "trust_remote_code" not in text
            if path.endswith(".json"):
                assert "auto_map" not in json.loads(text)

    base = AutoModelForCausalLM.from_pretrained(
        package_fixture.snapshot,
        local_files_only=True,
        trust_remote_code=False,
    )
    loaded = PeftModel.from_pretrained(
        base,
        package_fixture.stage / "adapters" / "identity",
        adapter_name="identity",
        local_files_only=True,
    )
    loaded.set_adapter("identity")
    with torch.no_grad():
        actual = loaded(
            input_ids=package_fixture.input_ids,
            attention_mask=package_fixture.attention_mask,
            use_cache=False,
        ).logits.detach().cpu()
    assert torch.equal(actual, package_fixture.base_logits)


def test_manifest_binds_base_versions_adapters_router_identity_and_absent_evaluation(
    package_fixture: PackageFixture,
) -> None:
    manifest = json.loads(
        (package_fixture.stage / "aptivra-manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["base"]["weights_included"] is False
    assert len(manifest["base"]["revision"]) == 40
    assert manifest["dependencies"] == {
        "peft_version": "0.21.2",
        "transformers_version": "5.18.0",
    }
    assert manifest["adapters"][0]["adapter_id"] == "identity"
    assert manifest["adapters"][0]["kind"] == "identity"
    assert manifest["adapters"][0]["rank"] == 16
    assert len(manifest["adapters"][0]["target_modules"]) == 7
    assert manifest["router"]["schema_version"] == "aptivra.router_sidecar.v1"
    assert manifest["identity"]["schema_version"] == "aptivra.identity_manifest.v1"
    assert manifest["evaluation"]["record"] is None
    assert manifest["evaluation"]["absence"]
    recorded = manifest.pop("manifest_sha256")
    assert recorded == hub_artifacts.canonical_sha256(manifest)


def test_standard_release_config_is_disabled_owner_named_and_byte_identical() -> None:
    settings = load_release_gate_settings()
    assert settings.feature.enabled is False
    # Owner-approved new repository name (2026-10-17, ADR-0011 §7).
    assert settings.repository.repository_id == "raghunath1/aptivra-360m-continual"
    assert settings.publication.allowed_adapter_kinds == ("synthetic", "identity")
    assert Path("configs/model_card_release_gate.yaml").read_bytes() == Path(
        "aptivra/resources/configs/model_card_release_gate.yaml"
    ).read_bytes()


@pytest.mark.parametrize("mutation", ["missing", "extra", "mismatch"])
def test_exact_allowlist_and_hash_manifest_rejects_any_drift(
    package_fixture: PackageFixture,
    tmp_path: Path,
    mutation: str,
) -> None:
    stage = _copy_stage(package_fixture, tmp_path / mutation)
    if mutation == "missing":
        (stage / "NOTICE").unlink()
    elif mutation == "extra":
        (stage / "extra.txt").write_text("extra", encoding="utf-8")
    else:
        (stage / "NOTICE").write_text("changed", encoding="utf-8")
    with pytest.raises(ValueError, match="allowlist mismatch|SHA-256 mismatch"):
        verify_staged_repository(stage, config_path=package_fixture.config_path)


def test_allowlist_mismatch_reports_exact_missing_and_extra_paths(
    package_fixture: PackageFixture,
    tmp_path: Path,
) -> None:
    missing_stage = _copy_stage(package_fixture, tmp_path / "missing")
    (missing_stage / "NOTICE").unlink()
    with pytest.raises(ValueError, match=r"missing=\['NOTICE'\], extra=\[\]"):
        verify_staged_repository(missing_stage, config_path=package_fixture.config_path)

    extra_stage = _copy_stage(package_fixture, tmp_path / "extra")
    (extra_stage / "extra.txt").write_text("extra", encoding="utf-8")
    with pytest.raises(ValueError, match=r"missing=\[\], extra=\['extra.txt'\]"):
        verify_staged_repository(extra_stage, config_path=package_fixture.config_path)


def test_file_manifest_self_hash_covers_itself_and_every_other_file(
    package_fixture: PackageFixture,
) -> None:
    payload = json.loads(
        (package_fixture.stage / FILE_MANIFEST).read_text(encoding="utf-8")
    )
    assert set(payload["allowlist"]) == {
        path.relative_to(package_fixture.stage).as_posix()
        for path in package_fixture.stage.rglob("*")
        if path.is_file()
    }
    assert FILE_MANIFEST not in payload["files"]
    recorded = payload.pop("file_manifest_sha256")
    assert recorded == hub_artifacts.canonical_sha256(payload)


def test_chronological_or_auxiliary_adapter_input_is_rejected(
    package_fixture: PackageFixture,
    tmp_path: Path,
) -> None:
    artifact = package_fixture.adapter_artifact
    router = SequenceTop1Router(16, 0.65)
    router.add_candidate("identity", "identity")
    router.promote_candidate()
    with pytest.raises(ValueError, match="non-publishable provenance"):
        stage_huggingface_hub_artifacts(
            tmp_path / "rejected-kind",
            (PublishableAdapter("identity", "chronological", "identity", artifact),),
            router,
            base_settings=package_fixture.base_settings,
            expert_settings=package_fixture.expert_settings,
            identity_manifest_path=package_fixture.identity_manifest,
            config_path=package_fixture.config_path,
        )
    copied_artifact = tmp_path / "artifact-with-raw-input"
    shutil.copytree(artifact, copied_artifact)
    (copied_artifact / "raw-records.json").write_text("{}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="raw data, configs"):
        stage_huggingface_hub_artifacts(
            tmp_path / "rejected-extra",
            (
                PublishableAdapter(
                    "identity", "identity", "identity", copied_artifact
                ),
            ),
            router,
            base_settings=package_fixture.base_settings,
            expert_settings=package_fixture.expert_settings,
            identity_manifest_path=package_fixture.identity_manifest,
            config_path=package_fixture.config_path,
        )


def test_staging_is_idempotent_for_the_same_verified_bundle(
    package_fixture: PackageFixture,
    tmp_path: Path,
) -> None:
    router = SequenceTop1Router(16, 0.65)
    router.add_candidate("identity", "identity")
    router.promote_candidate()
    output = tmp_path / "stage"

    first = stage_huggingface_hub_artifacts(
        output,
        (
            PublishableAdapter(
                "identity",
                "identity",
                "identity",
                package_fixture.adapter_artifact,
            ),
        ),
        router,
        base_settings=package_fixture.base_settings,
        expert_settings=package_fixture.expert_settings,
        identity_manifest_path=package_fixture.identity_manifest,
        config_path=package_fixture.config_path,
    )
    second = stage_huggingface_hub_artifacts(
        output,
        (
            PublishableAdapter(
                "identity",
                "identity",
                "identity",
                package_fixture.adapter_artifact,
            ),
        ),
        router,
        base_settings=package_fixture.base_settings,
        expert_settings=package_fixture.expert_settings,
        identity_manifest_path=package_fixture.identity_manifest,
        config_path=package_fixture.config_path,
    )

    assert second.file_manifest_sha256 == first.file_manifest_sha256


def test_staging_rejects_disabled_empty_duplicate_and_router_mismatch(
    package_fixture: PackageFixture,
    tmp_path: Path,
) -> None:
    router = SequenceTop1Router(16, 0.65)
    router.add_candidate("identity", "identity")
    router.promote_candidate()
    adapter = PublishableAdapter(
        "identity", "identity", "identity", package_fixture.adapter_artifact
    )
    with pytest.raises(RuntimeError, match="disabled by configuration"):
        stage_huggingface_hub_artifacts(tmp_path / "disabled", (adapter,), router)
    with pytest.raises(ValueError, match="at least one"):
        stage_huggingface_hub_artifacts(
            tmp_path / "empty",
            (),
            router,
            config_path=package_fixture.config_path,
        )


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda payload: payload.update(schema_version="wrong"), "schema is invalid"),
        (lambda payload: payload.update(router="wrong"), "router must be an object"),
        (
            lambda payload: payload["router"].update(weights_path=7),
            "router file binding is invalid",
        ),
        (lambda payload: payload.update(adapters="wrong"), "non-empty list"),
        (lambda payload: payload.update(adapters=[]), "non-empty list"),
        (lambda payload: payload.update(adapters=["wrong"]), "entry must be an object"),
        (
            lambda payload: payload["adapters"][0].update(path=7),
            "file binding is invalid",
        ),
        (
            lambda payload: payload["adapters"][0].update(files="wrong"),
            "file binding is invalid",
        ),
        (
            lambda payload: payload["adapters"][0].update(files={"name": 7}),
            "adapter hash is invalid",
        ),
    ],
)
def test_staging_manifest_rejects_every_invalid_shape(
    package_fixture: PackageFixture,
    tmp_path: Path,
    mutation: Callable[[dict[str, object]], None],
    message: str,
) -> None:
    stage = _copy_stage(package_fixture, tmp_path / "stage")
    _rewrite_hashed_json(
        stage / hub_artifacts.STAGING_MANIFEST,
        "manifest_sha256",
        mutation,
    )
    _refresh_stage_file_manifest(stage)
    with pytest.raises(ValueError, match=message):
        verify_staged_repository(stage, config_path=package_fixture.config_path)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda payload: payload.update(schema_version="wrong"), "schema is invalid"),
        (lambda payload: payload.update(allowlist="wrong"), "allowlist"),
        (lambda payload: payload.update(files="wrong"), "hashes must map"),
        (lambda payload: payload.update(files={7: "hash"}), "hash keys"),
        (lambda payload: payload.update(files={"NOTICE": 7}), "hashes must map"),
        (lambda payload: payload["files"].pop("NOTICE"), "hash keys"),
        (lambda payload: payload["files"].update(extra="0" * 64), "hash keys"),
    ],
)
def test_file_manifest_rejects_every_invalid_shape(
    package_fixture: PackageFixture,
    tmp_path: Path,
    mutation: Callable[[dict[str, object]], None],
    message: str,
) -> None:
    stage = _copy_stage(package_fixture, tmp_path / "stage")
    _rewrite_hashed_json(stage / FILE_MANIFEST, "file_manifest_sha256", mutation)
    with pytest.raises(ValueError, match=message):
        verify_staged_repository(stage, config_path=package_fixture.config_path)


def test_structured_yaml_and_first_model_card_field_are_scanned(
    package_fixture: PackageFixture,
    tmp_path: Path,
) -> None:
    yaml_stage = _copy_stage(package_fixture, tmp_path / "yaml")
    (yaml_stage / "metadata.yaml").write_text("source: GitHub\n", encoding="utf-8")
    with pytest.raises(ValueError, match="prohibited"):
        hub_artifacts.scan_staged_repository(yaml_stage, ("github",))

    card_stage = _copy_stage(package_fixture, tmp_path / "card")
    card = card_stage / "README.md"
    text = card.read_text(encoding="utf-8")
    card.write_text(text.replace("---\n", "---\ngithub: true\n", 1), encoding="utf-8")
    with pytest.raises(ValueError, match="prohibited"):
        hub_artifacts.scan_staged_repository(card_stage, ("github",))


def test_staging_records_are_immutable_and_output_parent_is_created(
    package_fixture: PackageFixture,
    tmp_path: Path,
) -> None:
    adapter = PublishableAdapter(
        "identity", "identity", "identity", package_fixture.adapter_artifact
    )
    with pytest.raises(FrozenInstanceError):
        adapter.kind = "synthetic"
    router = SequenceTop1Router(16, 0.65)
    router.add_candidate("identity", "identity")
    router.promote_candidate()
    staged = stage_huggingface_hub_artifacts(
        output_dir=tmp_path / "nested" / "stage",
        adapters=(adapter,),
        router=router,
        base_settings=package_fixture.base_settings,
        expert_settings=package_fixture.expert_settings,
        identity_manifest_path=package_fixture.identity_manifest,
        config_path=package_fixture.config_path,
    )
    with pytest.raises(FrozenInstanceError):
        staged.path = tmp_path
    with pytest.raises(ValueError, match="duplicate"):
        stage_huggingface_hub_artifacts(
            tmp_path / "duplicate",
            (adapter, adapter),
            router,
            base_settings=package_fixture.base_settings,
            expert_settings=package_fixture.expert_settings,
            identity_manifest_path=package_fixture.identity_manifest,
            config_path=package_fixture.config_path,
        )
    wrong_router = SequenceTop1Router(16, 0.65)
    wrong_router.add_candidate("other", "identity")
    wrong_router.promote_candidate()
    with pytest.raises(ValueError, match="IDs must match exactly"):
        stage_huggingface_hub_artifacts(
            tmp_path / "router-mismatch",
            (adapter,),
            wrong_router,
            base_settings=package_fixture.base_settings,
            expert_settings=package_fixture.expert_settings,
            identity_manifest_path=package_fixture.identity_manifest,
            config_path=package_fixture.config_path,
        )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("adapter_name", "aaa", "adapter ID"),
        ("adapter_name", "zzz", "adapter ID"),
        ("base_repository", "aaa/base", "pinned upstream base"),
        ("base_repository", "zzz/base", "pinned upstream base"),
        ("base_revision", "0" * 40, "pinned upstream base"),
        ("base_revision", "f" * 40, "pinned upstream base"),
        ("base_config_sha256", "0" * 64, "pinned upstream base"),
        ("base_config_sha256", "f" * 64, "pinned upstream base"),
        ("base_weights_sha256", "0" * 64, "pinned upstream base"),
        ("base_weights_sha256", "f" * 64, "pinned upstream base"),
        ("base_fingerprint", "0" * 64, "pinned upstream base"),
        ("base_fingerprint", "f" * 64, "pinned upstream base"),
        ("content_sha256", "0" * 64, "content SHA-256"),
        ("content_sha256", "f" * 64, "content SHA-256"),
    ],
)
def test_adapter_manifest_binding_rejects_every_mismatch_direction(
    package_fixture: PackageFixture,
    tmp_path: Path,
    field: str,
    value: object,
    message: str,
) -> None:
    artifact = tmp_path / "artifact"
    shutil.copytree(package_fixture.adapter_artifact, artifact)
    manifest_path = artifact / "expert_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest[field] = value
    if field == "adapter_name":
        manifest["adapter_subdir"] = value
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    router = SequenceTop1Router(16, 0.65)
    router.add_candidate("identity", "identity")
    router.promote_candidate()
    with pytest.raises(ValueError, match=message):
        stage_huggingface_hub_artifacts(
            tmp_path / "stage",
            (PublishableAdapter("identity", "identity", "identity", artifact),),
            router,
            base_settings=package_fixture.base_settings,
            expert_settings=package_fixture.expert_settings,
            identity_manifest_path=package_fixture.identity_manifest,
            config_path=package_fixture.config_path,
        )

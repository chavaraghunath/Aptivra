"""Tiny-fixture acceptance tests for ADR-0011 lane 4 PEFT experts."""

from __future__ import annotations

import copy
import hashlib
import json
import shutil
from importlib.metadata import metadata, version
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import yaml
from peft import PeftModel
from pydantic import ValidationError
from transformers import AutoConfig, LlamaConfig

import aptivra.experts.standard as standard_experts
from aptivra.experts.standard import (
    PeftExpertBank,
    expected_lora_parameter_count,
    load_promoted_expert,
    load_standard_expert_settings,
)
from aptivra.inference.standard_base import (
    StandardBase,
    load_standard_base,
    load_standard_base_settings,
)
from aptivra.resources import aptivra_resource
from tests.test_standard_base import _tiny_snapshot


def _tiny_base(tmp_path: Path) -> StandardBase:
    config_path, snapshot = _tiny_snapshot(tmp_path)
    return load_standard_base(
        load_standard_base_settings(config_path),
        snapshot_path=snapshot,
        device="cpu",
    )


def _enabled_settings() -> standard_experts.StandardExpertSettings:
    settings = load_standard_expert_settings()
    return settings.model_copy(
        update={"feature": settings.feature.model_copy(update={"enabled": True})}
    )


def _inputs() -> tuple[torch.Tensor, torch.Tensor]:
    input_ids = torch.tensor([[1, 3, 4, 2]], dtype=torch.long)
    return input_ids, torch.ones_like(input_ids)


def _logits(bank: PeftExpertBank) -> torch.Tensor:
    input_ids, attention_mask = _inputs()
    with torch.no_grad():
        return bank.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
        ).logits.detach().clone()


def _train_one_step(bank: PeftExpertBank, adapter_name: str) -> None:
    bank.activate(adapter_name)
    parameters = bank.adapter_parameters(adapter_name)
    optimizer = torch.optim.SGD(parameters, lr=0.25)
    optimizer.zero_grad(set_to_none=True)
    input_ids, attention_mask = _inputs()
    logits = bank.model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        use_cache=False,
    ).logits
    loss = torch.nn.functional.cross_entropy(
        logits[:, -1, :], torch.tensor([5], dtype=torch.long)
    )
    loss.backward()
    assert any(
        parameter.grad is not None and bool(torch.count_nonzero(parameter.grad).item())
        for parameter in parameters
    )
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)


def test_dependency_pin_license_and_import_compatibility() -> None:
    assert version("peft") == "0.21.2"
    peft_metadata = metadata("peft")
    assert peft_metadata["License"] == "Apache"
    assert "License :: OSI Approved :: Apache Software License" in peft_metadata.get_all(
        "Classifier"
    )
    assert version("torch") == "2.14.1"
    assert version("transformers") == "5.18.0"
    assert PeftModel.from_pretrained is not None


def test_packaged_expert_config_is_strict_disabled_and_byte_identical(
    tmp_path: Path,
) -> None:
    settings = load_standard_expert_settings()
    assert settings.feature.enabled is False
    assert settings.peft.version == "0.21.2"
    assert settings.peft.license == "Apache-2.0"
    assert settings.lora.rank == 16
    assert settings.lora.target_modules == [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ]
    assert Path("configs/smollm2_experts.yaml").read_bytes() == aptivra_resource(
        "configs", "smollm2_experts.yaml"
    ).read_bytes()

    raw = json.loads(json.dumps(settings.model_dump()))
    raw["lora"]["target_modules"][-1] = "other_proj"
    invalid = tmp_path / "invalid.yaml"
    invalid.write_text(
        yaml.safe_dump({"smollm2_experts": raw}, sort_keys=False),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="all seven"):
        load_standard_expert_settings(invalid)


def test_disabled_flag_rejects_before_peft_attachment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    base = _tiny_base(tmp_path)
    called = False

    def unexpected_attachment(*args: object, **kwargs: object) -> object:
        nonlocal called
        del args, kwargs
        called = True
        raise AssertionError("PEFT attachment must not run while disabled")

    monkeypatch.setattr(standard_experts, "get_peft_model", unexpected_attachment)
    with pytest.raises(RuntimeError, match="disabled"):
        PeftExpertBank.start_candidate(base, load_standard_expert_settings(), "candidate")
    assert called is False


def test_active_expert_changes_logits_and_receives_gradients_while_inactive_does_not(
    tmp_path: Path,
) -> None:
    bank = PeftExpertBank.start_candidate(
        _tiny_base(tmp_path), _enabled_settings(), "candidate"
    )
    assert bank.trainable_adapter == "candidate"
    assert bank.actual_parameter_count("candidate") == expected_lora_parameter_count(
        bank.base.settings.architecture,
        rank=16,
    )

    bank.deactivate()
    inactive_before = _logits(bank)
    assert all(parameter.grad is None for parameter in bank.adapter_parameters("candidate"))

    _train_one_step(bank, "candidate")
    active_after = _logits(bank)
    assert not torch.equal(active_after, inactive_before)

    bank.deactivate()
    inactive_after = _logits(bank)
    assert torch.equal(inactive_after, inactive_before)


def test_new_candidate_training_preserves_base_and_mature_adapter_bytes(
    tmp_path: Path,
) -> None:
    settings = _enabled_settings()
    bank = PeftExpertBank.start_candidate(_tiny_base(tmp_path), settings, "mature")
    _train_one_step(bank, "mature")
    artifact = tmp_path / "mature-artifact"
    manifest = bank.promote_candidate(artifact)
    assert manifest.adapter_name == "mature"
    assert bank.trainable_adapter is None
    assert all(
        parameter.requires_grad is False
        for parameter in bank.adapter_parameters("mature")
    )

    base_before = bank.base_parameter_sha256()
    mature_before = bank.adapter_parameter_sha256("mature")
    bank.add_candidate("candidate")
    _train_one_step(bank, "candidate")

    assert bank.base_parameter_sha256() == base_before
    assert bank.adapter_parameter_sha256("mature") == mature_before
    assert all(
        parameter.grad is None for parameter in bank.adapter_parameters("mature")
    )


def test_content_hash_load_and_standard_peft_reload_reproduce_exact_cpu_logits(
    tmp_path: Path,
) -> None:
    settings = _enabled_settings()
    base = _tiny_base(tmp_path / "source")
    bank = PeftExpertBank.start_candidate(base, settings, "skill")
    _train_one_step(bank, "skill")
    expected = _logits(bank)
    artifact = tmp_path / "artifact"
    manifest = bank.promote_candidate(artifact)

    reloaded = load_promoted_expert(_tiny_base(tmp_path / "aptivra"), settings, artifact)
    assert reloaded.trainable_adapter is None
    assert torch.equal(_logits(reloaded), expected)

    standard = PeftModel.from_pretrained(
        _tiny_base(tmp_path / "standard").model,
        artifact / manifest.adapter_subdir,
        adapter_name=manifest.adapter_name,
        is_trainable=False,
    )
    input_ids, attention_mask = _inputs()
    standard.eval()
    with torch.no_grad():
        standard_logits = standard(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
        ).logits
    assert torch.equal(standard_logits, expected)


@pytest.mark.parametrize(
    ("field", "replacement", "message"),
    [
        ("base_fingerprint", "0" * 64, "base fingerprint"),
        ("rank", 8, "rank"),
        (
            "target_modules",
            ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "x"],
            "target modules",
        ),
    ],
)
def test_wrong_manifest_contract_fails_before_peft_load(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    replacement: object,
    message: str,
) -> None:
    settings = _enabled_settings()
    bank = PeftExpertBank.start_candidate(_tiny_base(tmp_path / "source"), settings, "skill")
    _train_one_step(bank, "skill")
    original = tmp_path / "original"
    bank.promote_candidate(original)
    tampered = tmp_path / f"tampered-{field}"
    shutil.copytree(original, tampered)
    manifest_path = tampered / "expert_manifest.json"
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload[field] = replacement
    manifest_path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")

    called = False

    def unexpected_load(*args: object, **kwargs: object) -> object:
        nonlocal called
        del args, kwargs
        called = True
        raise AssertionError("PEFT load must happen only after manifest validation")

    monkeypatch.setattr(standard_experts.PeftModel, "from_pretrained", unexpected_load)
    with pytest.raises(ValueError, match=message):
        load_promoted_expert(_tiny_base(tmp_path / "target"), settings, tampered)
    assert called is False


def test_adapter_content_tampering_fails_before_peft_load(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings = _enabled_settings()
    bank = PeftExpertBank.start_candidate(_tiny_base(tmp_path / "source"), settings, "skill")
    artifact = tmp_path / "artifact"
    manifest = bank.promote_candidate(artifact)
    weights_path = artifact / manifest.adapter_subdir / "adapter_model.safetensors"
    weights = bytearray(weights_path.read_bytes())
    weights[-1] ^= 1
    weights_path.write_bytes(weights)

    called = False

    def unexpected_load(*args: object, **kwargs: object) -> object:
        nonlocal called
        del args, kwargs
        called = True
        raise AssertionError("PEFT load must happen only after content validation")

    monkeypatch.setattr(standard_experts.PeftModel, "from_pretrained", unexpected_load)
    with pytest.raises(ValueError, match="content SHA-256"):
        load_promoted_expert(_tiny_base(tmp_path / "target"), settings, artifact)
    assert called is False


def test_rank16_parameter_arithmetic_matches_cached_smollm2_config() -> None:
    settings = load_standard_base_settings()
    config = AutoConfig.from_pretrained(
        settings.source.repository,
        revision=settings.source.revision,
        local_files_only=True,
        trust_remote_code=False,
    )
    assert isinstance(config, LlamaConfig)
    assert expected_lora_parameter_count(config, rank=16) == 8_683_520


def test_base_fingerprint_is_canonical_and_sensitive(tmp_path: Path) -> None:
    base = _tiny_base(tmp_path)
    first = standard_experts.standard_base_fingerprint(base.settings)
    source = base.settings.source
    payload = {
        "schema_version": base.settings.schema_boundary.new_fingerprint_schema,
        "repository": source.repository,
        "revision": source.revision,
        "config_sha256": source.config.sha256,
        "weights_sha256": source.weights.sha256,
    }
    expected = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    assert first == expected
    assert len(first) == 64
    changed = copy.deepcopy(base.settings.model_dump())
    changed["source"]["weights"]["sha256"] = "0" * 64
    changed_settings = type(base.settings).model_validate(changed)
    assert standard_experts.standard_base_fingerprint(changed_settings) != first
    assert hashlib.sha256(bytes.fromhex(first)).hexdigest() != first


def test_config_arithmetic_and_adapter_guardrails_fail_closed(tmp_path: Path) -> None:
    non_mapping = tmp_path / "non-mapping.yaml"
    non_mapping.write_text("- invalid\n", encoding="utf-8")
    with pytest.raises(ValueError, match="YAML mapping"):
        load_standard_expert_settings(non_mapping)
    wrong_root = tmp_path / "wrong-root.yaml"
    wrong_root.write_text("other: {}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="contain only"):
        load_standard_expert_settings(wrong_root)

    settings = _enabled_settings()
    with pytest.raises(ValueError, match="dropout=0.0"):
        standard_experts.ExpertLoRASettings.model_validate(
            settings.lora.model_dump() | {"dropout": 0.1}
        )
    valid = SimpleNamespace(
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
    )
    assert expected_lora_parameter_count(valid, rank=16) == 4_096
    assert type(expected_lora_parameter_count(valid, rank=1)) is int
    assert expected_lora_parameter_count(valid, rank=1) == 256
    with pytest.raises(ValueError, match="positive integer"):
        expected_lora_parameter_count(valid, rank=0)
    with pytest.raises(ValueError, match="positive integer"):
        expected_lora_parameter_count(valid, rank=-1)
    with pytest.raises(ValueError, match="positive integer"):
        expected_lora_parameter_count(valid, rank=True)
    malformed = SimpleNamespace(
        hidden_size=15,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
    )
    with pytest.raises(ValueError, match="divisible"):
        expected_lora_parameter_count(malformed, rank=16)

    bank = PeftExpertBank.start_candidate(_tiny_base(tmp_path / "bank"), settings, "one")
    assert bank.adapter_names == ("one",)
    with pytest.raises(RuntimeError, match="only one trainable"):
        bank.add_candidate("two")
    with pytest.raises(KeyError, match="does not exist"):
        bank.activate("missing")
    with pytest.raises(KeyError, match="does not exist"):
        bank.adapter_parameter_sha256("missing")
    with pytest.raises(ValueError, match="letters"):
        bank.adapter_parameters("not.valid")

    bank.promote_candidate(tmp_path / "promoted")
    with pytest.raises(RuntimeError, match="no trainable"):
        bank.promote_candidate(tmp_path / "again")
    with pytest.raises(ValueError, match="already exists"):
        bank.add_candidate("one")
    bank.settings = settings.model_copy(
        update={"lora": settings.lora.model_copy(update={"max_adapters": 1})}
    )
    with pytest.raises(RuntimeError, match="adapter-count"):
        bank.add_candidate("two")


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        ("alpha", 0),
        ("alpha", -1),
        ("alpha", True),
        ("dropout", -0.1),
        ("dropout", 1.1),
        ("max_adapters", 0),
        ("max_adapters", -1),
        ("max_adapters", True),
        ("target_modules", ["q_proj"] * 6),
        ("target_modules", ["q_proj"] * 8),
    ],
)
def test_lora_configuration_boundaries_are_strict(
    field: str, replacement: object
) -> None:
    payload = load_standard_expert_settings().lora.model_dump()
    payload[field] = replacement
    with pytest.raises(ValidationError):
        standard_experts.ExpertLoRASettings.model_validate(payload)


def test_configuration_models_reject_extras_and_mutation() -> None:
    settings = load_standard_expert_settings()
    payload = settings.model_dump()
    payload["unexpected"] = True
    with pytest.raises(ValidationError):
        standard_experts.StandardExpertSettings.model_validate(payload)
    with pytest.raises(ValidationError):
        settings.lora.alpha = 32
    assert standard_experts.ExpertLoRASettings.model_validate(
        settings.lora.model_dump() | {"alpha": 1, "max_adapters": 1}
    ).alpha == 1


@pytest.mark.parametrize(
    ("field", "replacement", "message"),
    [
        ("r", 8, "rank"),
        ("r", 32, "rank"),
        ("target_modules", ["q_proj"], "target modules"),
        ("target_modules", ["q_proj"] * 7, "target modules"),
        ("lora_alpha", 8, "alpha"),
        ("lora_alpha", 32, "alpha"),
        ("lora_dropout", -1.0, "dropout"),
        ("lora_dropout", 1.0, "dropout"),
        ("bias", "all", "bias"),
        ("bias", "", "bias"),
        ("peft_type", "IA3", "causal-language-model"),
        ("task_type", "SEQ_CLS", "causal-language-model"),
    ],
)
def test_saved_peft_configuration_contract_rejects_each_wrong_field(
    tmp_path: Path,
    field: str,
    replacement: object,
    message: str,
) -> None:
    settings = _enabled_settings()
    adapter_dir = tmp_path / field
    adapter_dir.mkdir()
    payload: dict[str, object] = {
        "r": settings.lora.rank,
        "target_modules": settings.lora.target_modules,
        "lora_alpha": settings.lora.alpha,
        "lora_dropout": settings.lora.dropout,
        "bias": settings.lora.bias,
        "peft_type": "LORA",
        "task_type": "CAUSAL_LM",
    }
    payload[field] = replacement
    (adapter_dir / "adapter_config.json").write_text(
        json.dumps(payload), encoding="utf-8"
    )
    with pytest.raises(ValueError, match=message):
        standard_experts._adapter_config_contract(adapter_dir, settings)


def test_manifest_validators_reject_every_malformed_identity(tmp_path: Path) -> None:
    bank = PeftExpertBank.start_candidate(
        _tiny_base(tmp_path / "source"), _enabled_settings(), "skill"
    )
    manifest = bank._manifest("skill", "a" * 64)
    payload = manifest.model_dump(mode="json")
    invalid_values: list[tuple[str, object]] = [
        ("adapter_name", "not.valid"),
        ("adapter_subdir", "not/valid"),
        ("base_repository", ""),
        ("base_revision", "a" * 39),
        ("base_revision", "G" * 40),
        ("base_config_sha256", "a" * 63),
        ("base_weights_sha256", "G" * 64),
        ("base_fingerprint", "a" * 65),
        ("content_sha256", "z" * 64),
    ]
    for field, replacement in invalid_values:
        changed = copy.deepcopy(payload)
        changed[field] = replacement
        with pytest.raises(ValidationError):
            standard_experts.ExpertManifest.model_validate(changed)

    mismatched_subdir = copy.deepcopy(payload)
    mismatched_subdir["adapter_subdir"] = "other"
    with pytest.raises(ValidationError, match="subdirectory"):
        standard_experts.ExpertManifest.model_validate(mismatched_subdir)
    wrong_targets = copy.deepcopy(payload)
    wrong_targets["target_modules"] = list(reversed(payload["target_modules"]))
    with pytest.raises(ValidationError, match="all-seven"):
        standard_experts.ExpertManifest.model_validate(wrong_targets)
    wrong_targets["target_modules"] = ["z_proj", *payload["target_modules"][1:]]
    with pytest.raises(ValidationError, match="all-seven"):
        standard_experts.ExpertManifest.model_validate(wrong_targets)
    mismatched_subdir["adapter_subdir"] = "a"
    with pytest.raises(ValidationError, match="subdirectory"):
        standard_experts.ExpertManifest.model_validate(mismatched_subdir)


@pytest.mark.parametrize(
    ("field", "replacement", "message"),
    [
        ("base_repository", "a", "base identity"),
        ("base_repository", "z", "base identity"),
        ("base_revision", "0" * 40, "base identity"),
        ("base_revision", "f" * 40, "base identity"),
        ("base_config_sha256", "0" * 64, "base identity"),
        ("base_config_sha256", "f" * 64, "base identity"),
        ("base_weights_sha256", "0" * 64, "base identity"),
        ("base_weights_sha256", "f" * 64, "base identity"),
        ("base_fingerprint", "f" * 64, "base fingerprint"),
        ("rank", 8, "rank"),
        ("rank", 32, "rank"),
        ("target_modules", ("a_proj",), "target modules"),
        ("target_modules", ("z_proj",), "target modules"),
        ("parameter_count", 1, "parameter count"),
        ("parameter_count", 1_000_000, "parameter count"),
        ("peft_version", "0.0.0", "PEFT version"),
        ("peft_version", "99.0.0", "PEFT version"),
        ("peft_license", "A", "PEFT license"),
        ("peft_license", "Z", "PEFT license"),
    ],
)
def test_loaded_manifest_rejects_each_incompatible_identity_field(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    replacement: object,
    message: str,
) -> None:
    settings = _enabled_settings()
    bank = PeftExpertBank.start_candidate(_tiny_base(tmp_path), settings, "skill")
    manifest = bank._manifest("skill", "a" * 64).model_copy(update={field: replacement})
    monkeypatch.setattr(standard_experts, "_adapter_config_contract", lambda *_: None)
    monkeypatch.setattr(standard_experts, "_artifact_content_sha256", lambda *_: "a" * 64)
    with pytest.raises(ValueError, match=message):
        standard_experts._validate_manifest_for_base(
            manifest, bank.base, settings, tmp_path
        )


def test_promotion_manifest_is_exact_and_retry_is_idempotent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings = _enabled_settings()
    bank = PeftExpertBank.start_candidate(_tiny_base(tmp_path), settings, "skill")
    artifact = tmp_path / "nested" / "artifact"
    manifest = bank.promote_candidate(artifact)
    source = bank.base.settings.source
    assert manifest.model_dump(mode="json") == {
        "schema_version": "aptivra.peft_expert_manifest.v1",
        "adapter_name": "skill",
        "adapter_subdir": "skill",
        "base_repository": source.repository,
        "base_revision": source.revision,
        "base_config_sha256": source.config.sha256,
        "base_weights_sha256": source.weights.sha256,
        "base_fingerprint": standard_experts.standard_base_fingerprint(bank.base.settings),
        "rank": 16,
        "target_modules": list(settings.lora.target_modules),
        "parameter_count": 4_096,
        "peft_version": "0.21.2",
        "peft_license": "Apache-2.0",
        "content_sha256": manifest.content_sha256,
    }
    assert manifest.content_sha256 == standard_experts._artifact_content_sha256(
        artifact / "skill"
    )
    assert {path.name for path in (artifact / "skill").iterdir()} == {
        "adapter_config.json",
        "adapter_model.safetensors",
    }
    manifest_text = (artifact / "expert_manifest.json").read_text(encoding="utf-8")
    assert manifest_text == json.dumps(
        manifest.model_dump(mode="json"), sort_keys=True
    ) + "\n"
    bank.trainable_adapter = "skill"
    assert bank.promote_candidate(artifact) == manifest
    assert not list(artifact.parent.glob(".artifact.*"))

    original_content_hash = standard_experts._artifact_content_sha256
    for wrong_hash in ("0" * 64, "f" * 64):
        calls = iter((manifest.content_sha256, wrong_hash))
        monkeypatch.setattr(
            standard_experts,
            "_artifact_content_sha256",
            lambda *_, calls=calls: next(calls),
        )
        bank.trainable_adapter = "skill"
        with pytest.raises(ValueError, match="existing promoted expert content"):
            bank.promote_candidate(artifact)
    monkeypatch.setattr(
        standard_experts, "_artifact_content_sha256", original_content_hash
    )

    payload = json.loads((artifact / "expert_manifest.json").read_text(encoding="utf-8"))
    payload["base_repository"] = "different"
    (artifact / "expert_manifest.json").write_text(
        json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8"
    )
    bank.trainable_adapter = "skill"
    with pytest.raises(FileExistsError, match="different promoted expert"):
        bank.promote_candidate(artifact)


def test_digest_framing_is_exact(tmp_path: Path) -> None:
    tensors = [
        ("z", torch.tensor([1, 2], dtype=torch.int16)),
        ("a", torch.tensor([[3.0]], dtype=torch.float32)),
    ]
    digest = hashlib.sha256()
    for name, tensor in sorted(tensors):
        value = tensor.detach().cpu().contiguous()
        metadata = json.dumps(
            {"name": name, "dtype": str(value.dtype), "shape": list(value.shape)},
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
        raw = value.view(torch.uint8).numpy().tobytes()
        digest.update(len(metadata).to_bytes(8, "big"))
        digest.update(metadata)
        digest.update(len(raw).to_bytes(8, "big"))
        digest.update(raw)
    assert standard_experts._tensor_digest(tensors) == digest.hexdigest()

    adapter_dir = tmp_path / "adapter"
    adapter_dir.mkdir()
    expected_artifact = hashlib.sha256()
    for name, payload in (
        ("adapter_config.json", b"config"),
        ("adapter_model.safetensors", b"weights"),
    ):
        (adapter_dir / name).write_bytes(payload)
        expected_artifact.update(name.encode())
        expected_artifact.update(len(payload).to_bytes(8, "big"))
        expected_artifact.update(payload)
    assert standard_experts._artifact_content_sha256(adapter_dir) == (
        expected_artifact.hexdigest()
    )


@pytest.mark.parametrize("reported_version", ["0.0.0", "99.0.0"])
def test_version_mismatch_fails_before_start_or_load(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    reported_version: str,
) -> None:
    monkeypatch.setattr(standard_experts, "version", lambda _: reported_version)
    with pytest.raises(RuntimeError, match="exact pin"):
        PeftExpertBank.start_candidate(
            _tiny_base(tmp_path / "start"), _enabled_settings(), "skill"
        )
    monkeypatch.setattr(
        standard_experts,
        "_load_expert_manifest",
        lambda *_: (_ for _ in ()).throw(AssertionError("manifest read too early")),
    )
    with pytest.raises(RuntimeError, match="exact pin"):
        load_promoted_expert(
            _tiny_base(tmp_path / "load"), _enabled_settings(), tmp_path / "missing"
        )


def test_base_is_frozen_before_peft_attachment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    original = standard_experts.get_peft_model

    def checked_attachment(
        model: torch.nn.Module, *args: object, **kwargs: object
    ) -> object:
        assert all(not parameter.requires_grad for parameter in model.parameters())
        return original(model, *args, **kwargs)

    monkeypatch.setattr(standard_experts, "get_peft_model", checked_attachment)
    bank = PeftExpertBank.start_candidate(
        _tiny_base(tmp_path), _enabled_settings(), "skill"
    )
    base_parameter = next(
        parameter
        for name, parameter in bank.model.named_parameters()
        if "lora_" not in name
    )
    base_parameter.requires_grad_(True)
    bank.deactivate()
    assert base_parameter.requires_grad is False


def test_parameter_count_guard_rejects_both_directions(tmp_path: Path) -> None:
    bank = PeftExpertBank.start_candidate(
        _tiny_base(tmp_path), _enabled_settings(), "skill"
    )
    expected = bank.actual_parameter_count("skill")
    for wrong in (expected - 1, expected + 1):
        original = bank.actual_parameter_count
        bank.actual_parameter_count = lambda _name, value=wrong: value  # type: ignore[method-assign]
        with pytest.raises(ValueError, match="parameter count mismatch"):
            bank._verify_parameter_count("skill")
        bank.actual_parameter_count = original  # type: ignore[method-assign]


def test_adapter_limit_rejects_counts_above_limit(tmp_path: Path) -> None:
    settings = _enabled_settings()
    bank = PeftExpertBank.start_candidate(_tiny_base(tmp_path), settings, "skill")
    bank.promote_candidate(tmp_path / "artifact")
    bank.settings = settings.model_copy(
        update={"lora": settings.lora.model_copy(update={"max_adapters": 0})}
    )
    with pytest.raises(RuntimeError, match="adapter-count"):
        bank.add_candidate("another")


def test_malformed_or_incomplete_artifact_fails_before_peft_load(tmp_path: Path) -> None:
    settings = _enabled_settings()
    malformed = tmp_path / "malformed"
    malformed.mkdir()
    (malformed / "expert_manifest.json").write_text("not-json", encoding="utf-8")
    with pytest.raises(ValueError, match="valid UTF-8 JSON"):
        load_promoted_expert(_tiny_base(tmp_path / "bad-target"), settings, malformed)

    bank = PeftExpertBank.start_candidate(_tiny_base(tmp_path / "source"), settings, "skill")
    artifact = tmp_path / "artifact"
    manifest = bank.promote_candidate(artifact)
    weights = artifact / manifest.adapter_subdir / "adapter_model.safetensors"
    weights.rename(artifact / manifest.adapter_subdir / "adapter_model.missing")
    with pytest.raises(ValueError, match="missing required PEFT file"):
        load_promoted_expert(_tiny_base(tmp_path / "missing-target"), settings, artifact)

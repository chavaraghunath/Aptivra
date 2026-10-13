"""End-to-end tests for the self-contained Hugging Face layout."""

from __future__ import annotations

import builtins
import json
import pickle
from pathlib import Path
from typing import Any

import pytest
import torch
from safetensors.torch import load_file, save_file
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, dynamic_module_utils

from aptivra.experts.expert import ExpertModule
from aptivra.experts.router import GatingRouter
from aptivra.export import hf_conversion
from aptivra.export.hf_conversion import convert_legacy_export_to_huggingface
from aptivra.export.hf_hub_artifacts import HUB_LAYOUT_FILES, export_huggingface_hub_layout
from aptivra.export.hf_push import push_aptivra_model_to_hf
from aptivra.export.multitier import MultitierExporter
from aptivra.inference.causal_lm import AptivraCausalLM
from aptivra.model.config import ModelConfig
from aptivra.model.transformer import BackboneTransformer


def _tiny_components() -> tuple[
    ModelConfig,
    BackboneTransformer,
    GatingRouter,
    list[ExpertModule],
]:
    torch.manual_seed(7)
    config = ModelConfig(
        d_model=16,
        n_layers=1,
        n_heads=2,
        d_ff=32,
        max_seq_len=16,
        dropout=0.0,
        device="cpu",
    )
    backbone = BackboneTransformer(config)
    router = GatingRouter(config.d_model, initial_num_experts=2)
    experts = [ExpertModule(index, config.d_model) for index in range(2)]
    return config, backbone, router, experts


def _export_new_layout(path: Path) -> AptivraCausalLM:
    config, backbone, router, experts = _tiny_components()
    export_huggingface_hub_layout(path, backbone, router, experts, config)
    return AptivraCausalLM(config, backbone, router, experts).eval()


def _export_legacy(path: Path) -> ModelConfig:
    config, backbone, router, experts = _tiny_components()
    MultitierExporter(path).export_model(backbone, router, experts, config)
    return config


def _set_modules_cache(monkeypatch: pytest.MonkeyPatch, path: Path) -> None:
    monkeypatch.setenv("HF_MODULES_CACHE", str(path))
    monkeypatch.setattr(dynamic_module_utils, "HF_MODULES_CACHE", str(path))


def test_stock_transformers_load_is_self_contained_and_numerically_identical(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    export_dir = tmp_path / "hub"
    package_model = _export_new_layout(export_dir)
    _set_modules_cache(monkeypatch, tmp_path / "modules")

    original_import = builtins.__import__

    def reject_aptivra(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "aptivra" or name.startswith("aptivra."):
            raise ModuleNotFoundError("aptivra deliberately unavailable")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", reject_aptivra)
    config = AutoConfig.from_pretrained(export_dir, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(export_dir, trust_remote_code=True)
    hub_model = AutoModelForCausalLM.from_pretrained(export_dir, trust_remote_code=True).eval()

    assert config.max_seq_len == 16
    assert len(tokenizer) == tokenizer.vocab_size == 256
    assert sum(parameter.numel() for parameter in hub_model.parameters()) > 0
    assert hub_model.get_input_embeddings().weight.data_ptr() == (
        hub_model.get_output_embeddings().weight.data_ptr()
    )

    input_ids = torch.tensor([[65, 112, 116, 105, 118, 114, 97]], dtype=torch.long)
    for use_moe in (False, True):
        package_model.use_moe_at_inference = use_moe
        hub_model.config.use_moe_at_inference = use_moe
        with torch.no_grad():
            package_logits = package_model(input_ids)
            hub_logits = hub_model(input_ids).logits
        torch.testing.assert_close(hub_logits, package_logits, rtol=1e-6, atol=1e-7)


def test_transformers_generate_matches_package_greedy_bytes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    export_dir = tmp_path / "hub"
    package_model = _export_new_layout(export_dir)
    with torch.no_grad():
        package_model.backbone.token_embedding.weight.zero_()
    export_huggingface_hub_layout(
        export_dir,
        package_model.backbone,
        package_model.router,
        package_model.experts,
        package_model.config,
    )
    _set_modules_cache(monkeypatch, tmp_path / "modules")
    tokenizer = AutoTokenizer.from_pretrained(export_dir, trust_remote_code=True)
    hub_model = AutoModelForCausalLM.from_pretrained(export_dir, trust_remote_code=True).eval()

    prompt = "Aptivra"
    package_text = package_model.generate(prompt, max_new_tokens=4)
    encoded = tokenizer(prompt, return_tensors="pt")
    generated = hub_model.generate(**encoded, max_new_tokens=4, do_sample=False)
    suffix = generated[0, encoded["input_ids"].shape[1] :].tolist()
    assert bytes(suffix) == package_text.encode("utf-8")


@pytest.mark.parametrize("text", ["héllo 世界 🙂", "", "context-" * 8])
def test_byte_tokenizer_round_trip_without_added_tokens(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    text: str,
) -> None:
    export_dir = tmp_path / "hub"
    _export_new_layout(export_dir)
    _set_modules_cache(monkeypatch, tmp_path / "modules")
    tokenizer = AutoTokenizer.from_pretrained(export_dir, trust_remote_code=True)

    encoded = tokenizer.encode(text)
    assert tokenizer.decode(encoded, skip_special_tokens=False) == text
    assert len(tokenizer) == 256
    assert max(encoded, default=0) < 256


def test_package_loader_prefers_safetensors_and_legacy_fallback(tmp_path: Path) -> None:
    legacy_dir = tmp_path / "legacy"
    config = _export_legacy(legacy_dir)
    legacy = AptivraCausalLM.from_export_dir(legacy_dir, model_config=config).eval()
    convert_legacy_export_to_huggingface(legacy_dir, model_config=config)
    consolidated = AptivraCausalLM.from_export_dir(legacy_dir, model_config=config).eval()

    input_ids = torch.tensor([[1, 2, 3, 4]], dtype=torch.long)
    torch.testing.assert_close(consolidated(input_ids), legacy(input_ids), rtol=1e-6, atol=1e-7)


def test_conversion_is_transactional_and_rejects_missing_or_corrupt_shards(
    tmp_path: Path,
) -> None:
    missing_source = tmp_path / "missing"
    missing_config = _export_legacy(missing_source)
    (missing_source / "experts" / "expert_0.pt").unlink()
    missing_output = tmp_path / "missing-output"
    with pytest.raises(FileNotFoundError, match="Expert shard missing"):
        convert_legacy_export_to_huggingface(
            missing_source,
            missing_output,
            model_config=missing_config,
        )
    assert not (missing_output / "model.safetensors").exists()

    corrupt_source = tmp_path / "corrupt"
    corrupt_config = _export_legacy(corrupt_source)
    (corrupt_source / "dense" / "backbone.pt").write_bytes(b"not a torch checkpoint")
    corrupt_output = tmp_path / "corrupt-output"
    with pytest.raises((RuntimeError, OSError, EOFError, pickle.UnpicklingError)):
        convert_legacy_export_to_huggingface(
            corrupt_source,
            corrupt_output,
            model_config=corrupt_config,
        )
    assert not (corrupt_output / "model.safetensors").exists()


def test_conversion_rejects_perturbed_consolidated_tensor_without_installing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "legacy"
    config = _export_legacy(source)
    output = tmp_path / "output"
    real_export = hf_conversion.export_huggingface_hub_layout

    def export_then_perturb(*args: Any, **kwargs: Any) -> Path:
        result = real_export(*args, **kwargs)
        state = load_file(result)
        key = "backbone.pos_embedding.weight"
        state[key] = state[key].clone()
        state[key][0, 0] += 1.0
        save_file(state, result, metadata={"format": "pt"})
        return result

    monkeypatch.setattr(hf_conversion, "export_huggingface_hub_layout", export_then_perturb)
    with pytest.raises(RuntimeError, match="Converted logits mismatch"):
        convert_legacy_export_to_huggingface(source, output, model_config=config)
    assert not (output / "model.safetensors").exists()


class _RecordingApi:
    def __init__(self) -> None:
        self.create_calls: list[dict[str, object]] = []
        self.upload_calls: list[dict[str, object]] = []

    def create_repo(self, **kwargs: object) -> None:
        self.create_calls.append(kwargs)

    def upload_folder(self, **kwargs: object) -> None:
        self.upload_calls.append(kwargs)


def test_push_uploads_only_new_layout_with_injected_client(tmp_path: Path) -> None:
    config, backbone, router, experts = _tiny_components()
    export_dir = tmp_path / "upload"
    export_dir.mkdir()
    (export_dir / "legacy.pt").write_bytes(b"legacy")
    (export_dir / "legacy.gguf").write_bytes(b"legacy")
    (export_dir / "multitier_manifest.json").write_text("{}", encoding="utf-8")
    api = _RecordingApi()

    url = push_aptivra_model_to_hf(
        "owner/repo",
        backbone,
        router,
        experts,
        config,
        export_dir,
        api=api,
    )

    assert url == "https://huggingface.co/owner/repo"
    assert len(api.create_calls) == 1
    assert len(api.upload_calls) == 1
    allowed = api.upload_calls[0]["allow_patterns"]
    assert allowed == list(HUB_LAYOUT_FILES)
    assert all(not name.endswith((".pt", ".gguf")) for name in allowed)
    assert all("manifest" not in name for name in allowed)
    assert {path.name for path in export_dir.iterdir()} >= set(HUB_LAYOUT_FILES)


def test_layout_metadata_records_checkpoint_truth(tmp_path: Path) -> None:
    export_dir = tmp_path / "hub"
    package_model = _export_new_layout(export_dir)
    payload = json.loads((export_dir / "config.json").read_text(encoding="utf-8"))

    assert payload["max_seq_len"] == package_model.config.max_seq_len == 16
    assert payload["vocab_size"] == 256
    assert payload["use_moe_at_inference"] is False
    assert payload["parameter_count"] == sum(
        parameter.numel() for parameter in package_model.parameters()
    )
    assert sorted(path.name for path in export_dir.iterdir()) == sorted(HUB_LAYOUT_FILES)

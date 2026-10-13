"""End-to-end tests for the self-contained Hugging Face layout."""

from __future__ import annotations

import builtins
import hashlib
import json
import os
import pickle
import runpy
import sys
from pathlib import Path
from typing import Any

import pytest
import torch
from safetensors.torch import load_file, save_file
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    LogitsProcessor,
    dynamic_module_utils,
)

from aptivra.eval.model_card_eval import (
    DEFAULT_HELDOUT_PROMPTS_PATH,
    DEFAULT_HELDOUT_TEXT_PATH,
    PromptAnswer,
    evaluate_export_dir,
    sha256_file,
)
from aptivra.eval.model_card_schema import PROVENANCE_FILENAME
from aptivra.experts.expert import ExpertModule
from aptivra.experts.router import GatingRouter
from aptivra.export import hf_conversion, hf_push
from aptivra.export.hf_conversion import convert_legacy_export_to_huggingface
from aptivra.export.hf_hub_artifacts import HUB_LAYOUT_FILES, export_huggingface_hub_layout
from aptivra.export.hf_push import (
    HUB_PUBLICATION_FILES,
    LEGACY_HUB_DELETE_PATTERNS,
    prepare_hf_publication,
    push_aptivra_model_to_hf,
)
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


@pytest.mark.parametrize(
    "forced_bytes",
    ([0], [65, 0], [10], [65, 10], [65, 66], [32, 65, 32]),
)
def test_transformers_generate_matches_package_stop_byte_contract(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    forced_bytes: list[int],
) -> None:
    export_dir = tmp_path / "hub"
    package_model = _export_new_layout(export_dir)
    _set_modules_cache(monkeypatch, tmp_path / "modules")
    tokenizer = AutoTokenizer.from_pretrained(export_dir, trust_remote_code=True)
    hub_model = AutoModelForCausalLM.from_pretrained(export_dir, trust_remote_code=True).eval()

    prompt = "Aptivra"
    prompt_length = len(prompt.encode("utf-8"))

    def forced_forward(input_ids: torch.Tensor) -> torch.Tensor:
        generated = input_ids.shape[1] - prompt_length
        token = forced_bytes[min(generated, len(forced_bytes) - 1)]
        logits = torch.full(
            (1, input_ids.shape[1], package_model.config.vocab_size),
            float("-inf"),
        )
        logits[0, -1, token] = 0.0
        return logits

    monkeypatch.setattr(package_model, "forward", forced_forward)

    class ForceBytes(LogitsProcessor):
        def __call__(
            self, input_ids: torch.Tensor, scores: torch.Tensor
        ) -> torch.Tensor:
            generated = input_ids.shape[1] - prompt_length
            token = forced_bytes[min(generated, len(forced_bytes) - 1)]
            forced = torch.full_like(scores, float("-inf"))
            forced[:, token] = 0.0
            return forced

    package_text = package_model.generate(prompt, max_new_tokens=len(forced_bytes))
    encoded = tokenizer(prompt, return_tensors="pt")
    generated = hub_model.generate(
        **encoded,
        max_new_tokens=len(forced_bytes),
        do_sample=False,
        logits_processor=[ForceBytes()],
    )
    suffix = generated[0, encoded["input_ids"].shape[1] :].tolist()
    hub_text = tokenizer.decode(suffix, skip_special_tokens=False)
    assert hub_text == package_text == bytes(forced_bytes).decode("utf-8")


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
    assert not (missing_output / PROVENANCE_FILENAME).exists()

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
    assert not (corrupt_output / PROVENANCE_FILENAME).exists()


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
    assert not (output / PROVENANCE_FILENAME).exists()


@pytest.mark.parametrize("position_state", [None, torch.ones(16)])
def test_conversion_rejects_missing_or_malformed_position_embedding(
    tmp_path: Path,
    position_state: torch.Tensor | None,
) -> None:
    source = tmp_path / "legacy"
    config = _export_legacy(source)
    backbone_path = source / "dense" / "backbone.pt"
    state = torch.load(backbone_path, map_location="cpu", weights_only=True)
    if position_state is None:
        del state["pos_embedding.weight"]
    else:
        state["pos_embedding.weight"] = position_state
    torch.save(state, backbone_path)

    output = tmp_path / "output"
    with pytest.raises(RuntimeError, match="pos_embedding.weight"):
        convert_legacy_export_to_huggingface(source, output, model_config=config)
    assert not (output / "model.safetensors").exists()


def test_conversion_uses_checkpoint_context_instead_of_stale_config(tmp_path: Path) -> None:
    source = tmp_path / "legacy"
    config = _export_legacy(source)
    stale_config = ModelConfig(
        d_model=config.d_model,
        n_layers=config.n_layers,
        n_heads=config.n_heads,
        d_ff=config.d_ff,
        max_seq_len=8,
        dropout=config.dropout,
        device="cpu",
    )

    weights = convert_legacy_export_to_huggingface(
        source,
        tmp_path / "output",
        model_config=stale_config,
    )
    payload = json.loads(weights.with_name("config.json").read_text(encoding="utf-8"))
    assert payload["max_seq_len"] == config.max_seq_len == 16


def test_conversion_writes_source_shard_provenance_and_revision(tmp_path: Path) -> None:
    source = tmp_path / "legacy"
    config = _export_legacy(source)
    expected_backbone_hash = sha256_file(source / "dense" / "backbone.pt")
    output = tmp_path / "output"

    convert_legacy_export_to_huggingface(
        source,
        output,
        model_config=config,
        source_hub_revision="source-revision",
    )

    provenance_path = output / PROVENANCE_FILENAME
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    assert provenance["source_layout"] == "legacy_multitier"
    assert provenance["source_manifest_version"] == "1.0.0"
    assert provenance["source_hub_revision"] == "source-revision"
    assert provenance["source_shards_sha256"]["dense/backbone.pt"] == (
        expected_backbone_hash
    )
    assert provenance["source_shards_sha256"]["dense/router.pt"] == sha256_file(
        source / "dense" / "router.pt"
    )
    assert provenance["model_safetensors_sha256"] == sha256_file(
        output / "model.safetensors"
    )
    assert isinstance(provenance["converter_version"], str)
    record = evaluate_export_dir(
        output,
        trained_prompts=[PromptAnswer("a", "b")],
        heldout_prompts=[PromptAnswer("a", "b")],
        heldout_text="ab",
        hub_revision="source-revision",
    )
    checkpoint = record["checkpoint"]
    assert isinstance(checkpoint, dict)
    assert checkpoint["backbone_sha256"] == expected_backbone_hash
    assert checkpoint["source_hub_revision"] == "source-revision"


def test_conversion_rejects_non_finite_legacy_weights(tmp_path: Path) -> None:
    source = tmp_path / "legacy"
    config = _export_legacy(source)
    backbone_path = source / "dense" / "backbone.pt"
    state = torch.load(backbone_path, map_location="cpu", weights_only=True)
    state["token_embedding.weight"][0, 0] = torch.nan
    torch.save(state, backbone_path)

    output = tmp_path / "output"
    with pytest.raises(RuntimeError, match="non-finite tensor"):
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


class _SnapshottingApi(_RecordingApi):
    def __init__(self) -> None:
        super().__init__()
        self.uploaded_bytes: dict[str, bytes] = {}

    def upload_folder(self, **kwargs: object) -> None:
        super().upload_folder(**kwargs)
        folder_path = Path(str(kwargs["folder_path"]))
        allow_patterns = kwargs["allow_patterns"]
        assert isinstance(allow_patterns, list)
        self.uploaded_bytes = {
            str(name): (folder_path / str(name)).read_bytes()
            for name in allow_patterns
        }


def _write_results_record(
    export_dir: Path,
    output: Path,
    *,
    weights_hash: str | None = None,
) -> None:
    weights = export_dir / "model.safetensors"
    digest = weights_hash or hashlib.sha256(weights.read_bytes()).hexdigest()
    config = json.loads((export_dir / "config.json").read_text(encoding="utf-8"))
    provenance_path = export_dir / PROVENANCE_FILENAME
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    provenance["model_safetensors_sha256"] = digest
    state = load_file(weights)
    counts = {"backbone": 0, "router": 0, "experts": 0}
    for name, tensor in state.items():
        counts[name.split(".", 1)[0]] += tensor.numel()

    def metric(mean: float, samples: int, unigram: float) -> dict[str, object]:
        summary = {"mean": mean, "min": mean, "max": mean, "samples": samples}
        return {
            **summary,
            "evaluated_bytes": samples,
            "baselines": {
                "uniform_bpb": {
                    "mean": 8.0,
                    "min": 8.0,
                    "max": 8.0,
                    "samples": samples,
                },
                "unigram_bpb": {
                    "mean": unigram,
                    "min": unigram,
                    "max": unigram,
                    "samples": samples,
                },
                "unigram_fit": "evaluated_bytes",
            },
        }

    output.write_text(
        json.dumps(
            {
                "schema_version": "aptivra.model_card_eval.v4",
                "timestamp_utc": "2026-10-14T00:00:00+00:00",
                "checkpoint": {
                    "model_safetensors_sha256": digest,
                    "config_sha256": sha256_file(export_dir / "config.json"),
                    "provenance_sha256": sha256_file(provenance_path),
                    "backbone_sha256": provenance["source_shards_sha256"].get(
                        "dense/backbone.pt"
                    ),
                    "source_hub_revision": provenance["source_hub_revision"],
                    "provenance": provenance,
                },
                "model": {
                    "backbone_parameters": counts["backbone"],
                    "router_parameters": counts["router"],
                    "expert_parameters": counts["experts"],
                    "total_parameters": sum(counts.values()),
                    "config_parameter_count": config["parameter_count"],
                    "context_length": config["max_seq_len"],
                    "vocab_size": config["vocab_size"],
                    "position_embedding": "learned_absolute",
                },
                "evaluation": {
                    "moe_enabled": False,
                    "heldout_prompt_set": {
                        "name": DEFAULT_HELDOUT_PROMPTS_PATH.name,
                        "sha256": sha256_file(DEFAULT_HELDOUT_PROMPTS_PATH),
                        "samples": 24,
                    },
                    "heldout_text_set": {
                        "name": DEFAULT_HELDOUT_TEXT_PATH.name,
                        "sha256": sha256_file(DEFAULT_HELDOUT_TEXT_PATH),
                        "bytes": DEFAULT_HELDOUT_TEXT_PATH.stat().st_size,
                    },
                    "trained_prompt_answer_bpb": metric(5.0, 7, 4.0),
                    "heldout_prompt_answer_bpb": metric(5.0, 24, 4.0),
                    "heldout_text_bpb": {
                        "value": 5.0,
                        "evaluated_bytes": DEFAULT_HELDOUT_TEXT_PATH.stat().st_size - 1,
                        "baselines": {
                            "uniform_bpb": 8.0,
                            "unigram_bpb": 4.0,
                            "unigram_fit": "evaluated_bytes",
                        },
                    },
                },
                "tokenizer": {
                    "padding_token_id": 0,
                    "end_of_sequence_token_ids": [0, 10],
                    "newline_stop_token_id": 10,
                },
            }
        ),
        encoding="utf-8",
    )
    weights_mtime = weights.stat().st_mtime_ns
    os.utime(output, ns=(weights_mtime + 1_000_000_000, weights_mtime + 1_000_000_000))


def test_push_uploads_only_new_layout_with_injected_client(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, backbone, router, experts = _tiny_components()
    export_dir = tmp_path / "upload"
    export_huggingface_hub_layout(export_dir, backbone, router, experts, config)
    (export_dir / "legacy.pt").write_bytes(b"legacy")
    (export_dir / "legacy.gguf").write_bytes(b"legacy")
    (export_dir / "multitier_manifest.json").write_text("{}", encoding="utf-8")
    results = tmp_path / "results.json"
    _write_results_record(export_dir, results)
    fresh_record = json.loads(results.read_text(encoding="utf-8"))
    fresh_record["timestamp_utc"] = "2026-10-14T01:00:00+00:00"
    for field in ("mean", "min", "max"):
        fresh_record["evaluation"]["heldout_prompt_answer_bpb"][field] += 0.0000005
    monkeypatch.setattr(hf_push, "evaluate_export_dir", lambda _: fresh_record)
    api = _RecordingApi()

    url = push_aptivra_model_to_hf(
        "owner/repo",
        export_dir,
        results,
        api=api,
    )

    assert url == "https://huggingface.co/owner/repo"
    assert len(api.create_calls) == 1
    assert len(api.upload_calls) == 1
    staged_path = Path(str(api.upload_calls[0]["folder_path"]))
    assert not staged_path.exists()
    allowed = api.upload_calls[0]["allow_patterns"]
    assert allowed == list(HUB_PUBLICATION_FILES)
    assert api.upload_calls[0]["delete_patterns"] == list(LEGACY_HUB_DELETE_PATTERNS)
    assert all(not name.endswith((".pt", ".gguf")) for name in allowed)
    assert all("manifest" not in name for name in allowed)
    assert not (export_dir / "README.md").exists()
    assert PROVENANCE_FILENAME in allowed


def test_push_uploads_staged_bytes_when_source_changes_after_staging(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    export_dir = tmp_path / "hub"
    _export_new_layout(export_dir)
    original_weights = (export_dir / "model.safetensors").read_bytes()
    results = tmp_path / "results.json"
    _write_results_record(export_dir, results)
    fresh_record = json.loads(results.read_text(encoding="utf-8"))
    staged_paths: list[Path] = []

    def evaluate(staged_path: Path) -> dict[str, object]:
        staged_paths.append(staged_path)
        (export_dir / "model.safetensors").write_bytes(b"source changed after staging")
        return fresh_record

    monkeypatch.setattr(hf_push, "evaluate_export_dir", evaluate)
    api = _SnapshottingApi()

    push_aptivra_model_to_hf("owner/repo", export_dir, results, api=api)

    assert api.uploaded_bytes["model.safetensors"] == original_weights
    assert staged_paths and not staged_paths[0].exists()


def test_push_refuses_staging_mutation_after_evaluation_and_cleans_up(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    export_dir = tmp_path / "hub"
    _export_new_layout(export_dir)
    results = tmp_path / "results.json"
    _write_results_record(export_dir, results)
    fresh_record = json.loads(results.read_text(encoding="utf-8"))
    staged_paths: list[Path] = []

    def evaluate(staged_path: Path) -> dict[str, object]:
        staged_paths.append(staged_path)
        (staged_path / "tokenizer_config.json").write_text(
            "changed after evaluation", encoding="utf-8"
        )
        return fresh_record

    monkeypatch.setattr(hf_push, "evaluate_export_dir", evaluate)
    api = _RecordingApi()

    with pytest.raises(ValueError, match="staged publication file changed"):
        push_aptivra_model_to_hf("owner/repo", export_dir, results, api=api)

    assert api.create_calls == []
    assert api.upload_calls == []
    assert staged_paths and not staged_paths[0].exists()


@pytest.mark.parametrize(
    ("record_kind", "error_type", "message"),
    [
        ("missing", FileNotFoundError, "results record not found"),
        ("missing_hash", ValueError, "model_safetensors_sha256"),
        ("wrong_hash", ValueError, "does not match export"),
    ],
)
def test_push_refuses_invalid_supplied_evidence_before_network(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    record_kind: str,
    error_type: type[Exception],
    message: str,
) -> None:
    export_dir = tmp_path / "hub"
    _export_new_layout(export_dir)
    results = tmp_path / "results.json"
    if record_kind == "missing_hash":
        results.write_text(json.dumps({"checkpoint": {}}), encoding="utf-8")
    elif record_kind == "wrong_hash":
        _write_results_record(export_dir, results, weights_hash="0" * 64)
    trusted = tmp_path / "trusted.json"
    _write_results_record(export_dir, trusted)
    monkeypatch.setattr(
        hf_push,
        "evaluate_export_dir",
        lambda _: json.loads(trusted.read_text(encoding="utf-8")),
    )
    api = _RecordingApi()

    with pytest.raises(error_type, match=message):
        push_aptivra_model_to_hf("owner/repo", export_dir, results, api=api)

    assert api.create_calls == []
    assert api.upload_calls == []
    assert not (export_dir / "README.md").exists()


def test_prepare_refuses_incomplete_layout(tmp_path: Path) -> None:
    results = tmp_path / "results.json"
    results.write_text("{}", encoding="utf-8")
    with pytest.raises(FileNotFoundError, match="Hub layout is incomplete"):
        prepare_hf_publication("owner/repo", tmp_path / "missing-layout", results)


@pytest.mark.parametrize(
    ("supplied", "fresh"),
    [({"metric": 1.0}, {"other": 1.0}), ([1.0], [1.0, 2.0])],
)
def test_supplied_record_comparison_rejects_structural_mismatches(
    supplied: object,
    fresh: object,
) -> None:
    with pytest.raises(ValueError, match="differs from fresh evaluation"):
        hf_push._compare_record_values(
            supplied,
            fresh,
            path=("evaluation",),
            tolerance=0.000001,
        )


def test_push_without_supplied_record_uses_fresh_evaluation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    export_dir = tmp_path / "hub"
    _export_new_layout(export_dir)
    fresh_path = tmp_path / "fresh.json"
    _write_results_record(export_dir, fresh_path)
    calls: list[Path] = []

    def evaluate(path: Path) -> dict[str, object]:
        calls.append(path)
        return json.loads(fresh_path.read_text(encoding="utf-8"))

    monkeypatch.setattr(hf_push, "evaluate_export_dir", evaluate)
    api = _RecordingApi()

    url = push_aptivra_model_to_hf("owner/repo", export_dir, api=api)

    assert url == "https://huggingface.co/owner/repo"
    assert len(calls) == 1
    assert calls[0] != export_dir
    assert not calls[0].exists()
    assert len(api.create_calls) == 1
    assert len(api.upload_calls) == 1


@pytest.mark.parametrize(
    "record_kind",
    ["old_version", "forged_partial", "parameter_mismatch", "false_metrics"],
)
def test_push_refuses_untrusted_or_unreproduced_evidence_before_network(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    record_kind: str,
) -> None:
    export_dir = tmp_path / "hub"
    _export_new_layout(export_dir)
    results = tmp_path / "results.json"
    _write_results_record(export_dir, results)
    record = json.loads(results.read_text(encoding="utf-8"))
    fresh_record = json.loads(results.read_text(encoding="utf-8"))
    monkeypatch.setattr(hf_push, "evaluate_export_dir", lambda _: fresh_record)
    if record_kind == "old_version":
        record["schema_version"] = "aptivra.model_card_eval.v3"
    elif record_kind == "forged_partial":
        record = {
            "checkpoint": record["checkpoint"],
            "evaluation": {
                "heldout_prompt_answer_bpb": {"mean": 0},
                "baselines": {"unigram_bpb": 8},
            },
        }
    elif record_kind == "parameter_mismatch":
        record["model"]["backbone_parameters"] -= 1
        record["model"]["router_parameters"] += 1
    else:
        metric = record["evaluation"]["heldout_prompt_answer_bpb"]
        metric["mean"] = 0.0
        metric["min"] = 0.0
        metric["max"] = 0.0
    results.write_text(json.dumps(record), encoding="utf-8")
    api = _RecordingApi()

    with pytest.raises(ValueError):
        push_aptivra_model_to_hf("owner/repo", export_dir, results, api=api)

    assert api.create_calls == []
    assert api.upload_calls == []
    assert not (export_dir / "README.md").exists()


def test_push_cli_dry_run_prints_exact_plan_without_network(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    export_dir = tmp_path / "hub"
    _export_new_layout(export_dir)
    results = tmp_path / "results.json"
    _write_results_record(export_dir, results)
    fresh_record = json.loads(results.read_text(encoding="utf-8"))
    evaluation_calls: list[Path] = []

    def evaluate(path: Path) -> dict[str, object]:
        evaluation_calls.append(path)
        return fresh_record

    monkeypatch.setattr(hf_push, "evaluate_export_dir", evaluate)
    script = Path(__file__).resolve().parents[1] / "scripts" / "push_to_hf.py"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(script),
            "--repo-id",
            "owner/repo",
            "--export-dir",
            str(export_dir),
            "--results-record",
            str(results),
            "--dry-run",
        ],
    )

    runpy.run_path(str(script), run_name="__main__")

    lines = capsys.readouterr().out.splitlines()
    expected = ["Files to upload:"]
    expected.extend(
        f"  {name}  sha256:{sha256_file(export_dir / name)}"
        if name != "README.md"
        else f"  {name}  sha256:{lines[1 + len(HUB_LAYOUT_FILES)].split('sha256:', 1)[1]}"
        for name in HUB_PUBLICATION_FILES
    )
    expected += [
        "Remote files to delete:",
        *(f"  {pattern}" for pattern in LEGACY_HUB_DELETE_PATTERNS),
    ]
    assert lines == expected
    assert len(evaluation_calls) == 1
    assert evaluation_calls[0] != export_dir
    assert not evaluation_calls[0].exists()


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

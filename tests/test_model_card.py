"""Tests for measured-only model-card rendering and release gating."""

from __future__ import annotations

import copy
import json
import re
from pathlib import Path

import pytest

from aptivra.eval.model_card_eval import DEFAULT_HELDOUT_PROMPTS_PATH, sha256_file
from aptivra.eval.model_card_schema import validate_evaluation_record
from aptivra.export.model_card import (
    generate_model_card_readme,
    load_release_gate_config,
    load_results_record,
    release_gate_allows_general_language_model,
)
from aptivra.model.config import ModelConfig

DENIED_CARD_TEXT = (
    "89.4% average accuracy",
    "+0.042 BWT",
    "Dense 64.2%",
    "EWC 71.8%",
    "Standard MoE 82.1%",
    "AID-Bench 4-Phase Developmental Benchmark Results",
    "100.0% / PASSED",
    "200M parameter",
    "RoPE",
    "5,716+ MCP skills",
    "zero catastrophic forgetting",
    "C execution engine",
    "tool-use",
    "function-calling",
    "embodied-ai",
)


def test_generated_card_denies_unsupported_claims_and_marks_absent_fields() -> None:
    card = generate_model_card_readme(
        repo_id="example/aptivra",
        model_config=ModelConfig(),
        num_experts=8,
        results_record={"evaluation": {"baselines": {}}},
    )

    for denied in DENIED_CARD_TEXT:
        assert denied not in card
    assert "not measured" in card
    assert "Not a usable language model" in card


def test_release_gate_notice_present_below_gate_and_absent_above_gate() -> None:
    base_record = _valid_record(heldout_mean=6.0, heldout_unigram_mean=4.0)
    below_card = generate_model_card_readme(
        repo_id="example/aptivra",
        model_config=ModelConfig(),
        num_experts=1,
        results_record=base_record,
    )
    assert "Not a usable language model" in below_card

    above_record = _valid_record(heldout_mean=3.0, heldout_unigram_mean=4.0)
    above_card = generate_model_card_readme(
        repo_id="example/aptivra",
        model_config=ModelConfig(),
        num_experts=1,
        results_record=above_record,
    )
    assert "Not a usable language model" not in above_card
    assert release_gate_allows_general_language_model(above_record) is True


def test_forged_partial_record_fails_closed_and_never_suppresses_notice() -> None:
    forged = {
        "checkpoint": {"model_safetensors_sha256": "0" * 64},
        "evaluation": {
            "heldout_prompt_answer_bpb": {"mean": 0},
            "baselines": {"unigram_bpb": 8},
        },
    }

    assert release_gate_allows_general_language_model(forged) is False
    card = generate_model_card_readme(
        repo_id="example/aptivra-200m",
        model_config=ModelConfig(),
        num_experts=1,
        results_record=forged,
    )
    assert "Not a usable language model" in card
    assert "Held-out answer-span BpB | not measured" in card


@pytest.mark.parametrize(
    ("path", "value", "message"),
    [
        (("schema_version",), "aptivra.model_card_eval.v2", "literal_error"),
        (
            ("evaluation", "heldout_prompt_set", "sha256"),
            "f" * 64,
            "bundled set",
        ),
        (
            ("evaluation", "heldout_prompt_answer_bpb", "mean"),
            float("nan"),
            "finite_number",
        ),
        (
            ("evaluation", "heldout_prompt_answer_bpb", "min"),
            4.0,
            "min <= mean <= max",
        ),
        (("model", "total_parameters"), 16, "parameter components"),
    ],
)
def test_strict_record_schema_rejects_old_partial_or_inconsistent_evidence(
    path: tuple[str, ...], value: object, message: str
) -> None:
    record = _valid_record(heldout_mean=3.0, heldout_unigram_mean=4.0)
    _set_path(record, path, value)

    with pytest.raises(ValueError, match=message):
        validate_evaluation_record(
            record,
            minimum_trained_samples=7,
            minimum_heldout_samples=24,
        )


def test_strict_record_schema_enforces_configured_minimum_and_count_alignment() -> None:
    record = _valid_record(heldout_mean=3.0, heldout_unigram_mean=4.0)
    for path in (
        ("evaluation", "heldout_prompt_set", "samples"),
        ("evaluation", "heldout_prompt_answer_bpb", "samples"),
        (
            "evaluation",
            "heldout_prompt_answer_bpb",
            "baselines",
            "uniform_bpb",
            "samples",
        ),
        (
            "evaluation",
            "heldout_prompt_answer_bpb",
            "baselines",
            "unigram_bpb",
            "samples",
        ),
    ):
        _set_path(record, path, 23)

    with pytest.raises(ValueError, match="configured minimum"):
        validate_evaluation_record(
            record,
            minimum_trained_samples=7,
            minimum_heldout_samples=24,
        )


def test_strict_record_schema_rejects_cross_field_inconsistencies() -> None:
    def rejected(
        changes: tuple[tuple[tuple[str, ...], object], ...], message: str
    ) -> None:
        record = _valid_record(heldout_mean=3.0, heldout_unigram_mean=4.0)
        for path, value in changes:
            _set_path(record, path, value)
        with pytest.raises(ValueError, match=message):
            validate_evaluation_record(
                record,
                minimum_trained_samples=7,
                minimum_heldout_samples=24,
            )

    rejected(
        (
            (
                (
                    "evaluation",
                    "heldout_prompt_answer_bpb",
                    "baselines",
                    "uniform_bpb",
                    "samples",
                ),
                23,
            ),
        ),
        "uniform baseline sample count",
    )
    rejected(
        (
            (
                (
                    "evaluation",
                    "heldout_prompt_answer_bpb",
                    "baselines",
                    "unigram_bpb",
                    "samples",
                ),
                23,
            ),
        ),
        "unigram baseline sample count",
    )
    rejected(
        ((("evaluation", "heldout_prompt_set", "samples"), 25),),
        "prompt-set and metric sample counts",
    )
    rejected(
        ((('model', 'config_parameter_count'), 16),),
        "config parameter_count",
    )
    rejected(
        ((('checkpoint', 'provenance', 'source_shards_sha256'), {"../x": "0" * 64}),),
        "safe relative paths",
    )
    rejected(
        (
            (("checkpoint", "provenance", "source_layout"), "legacy_multitier"),
            (("checkpoint", "provenance", "source_manifest_version"), None),
        ),
        "source_manifest_version",
    )
    rejected(
        (
            (("checkpoint", "provenance", "source_layout"), "legacy_multitier"),
            (("checkpoint", "provenance", "source_manifest_version"), "1.0.0"),
        ),
        "dense/backbone.pt",
    )
    rejected(
        (
            (("checkpoint", "provenance", "source_manifest_version"), "1.0.0"),
        ),
        "native provenance",
    )
    rejected(
        ((('checkpoint', 'provenance', 'model_safetensors_sha256'), "f" * 64),),
        "model hashes do not match",
    )
    rejected(
        ((('checkpoint', 'provenance', 'source_hub_revision'), "other"),),
        "source revisions do not match",
    )
    rejected(
        ((('checkpoint', 'backbone_sha256'), "f" * 64),),
        "backbone hash does not match",
    )
    rejected(
        ((('tokenizer', 'end_of_sequence_token_ids'), []),),
        "must not be empty",
    )
    rejected(
        (
            (("tokenizer", "padding_token_id"), 1),
        ),
        "padding token",
    )
    rejected(
        (
            (("tokenizer", "newline_stop_token_id"), 11),
        ),
        "newline stop token",
    )
    rejected(((('timestamp_utc',), "not-a-date"),), "ISO-8601")
    rejected(((('timestamp_utc',), "2026-10-14T00:00:00"),), "timezone")
    rejected(
        ((('evaluation', 'heldout_prompt_set', 'name'), "other.json"),),
        "name does not match",
    )
    record = _valid_record(heldout_mean=3.0, heldout_unigram_mean=4.0)
    with pytest.raises(ValueError, match="trained prompt sample count"):
        validate_evaluation_record(
            record,
            minimum_trained_samples=8,
            minimum_heldout_samples=24,
        )


def test_model_card_source_has_no_result_literal_regressions() -> None:
    source = Path("aptivra/export/model_card.py").read_text(encoding="utf-8")
    forbidden_patterns = (
        r"\b\d+(?:\.\d+)?\s*%\b",
        r"[+-]\d+\.\d+\s*BWT",
        r"\baccuracy\b.*\d+\.\d+",
        r"\bBWT\b.*\d+\.\d+",
    )
    for pattern in forbidden_patterns:
        assert re.search(pattern, source, flags=re.IGNORECASE) is None


def test_default_release_gate_config_is_independent_of_working_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    assert load_release_gate_config() == 0.0


def test_card_describes_transformers_loading_generation_contract_and_measured_ranges() -> None:
    record = _valid_record(heldout_mean=6.0, heldout_unigram_mean=4.0)

    card = generate_model_card_readme(
        repo_id="example/aptivra-200m",
        model_config=ModelConfig(),
        num_experts=1,
        results_record=record,
    )

    assert "library_name: transformers" in card
    assert "Loading Interface Under Packaging Review" not in card
    assert "loads through the stock Transformers auto classes" in card
    assert "Always set `max_new_tokens`" in card
    assert "Byte 0 doubles as padding and end-of-sequence" in card
    assert "byte 10 (newline) is also a stop" in card
    assert "Byte\n0" not in card
    assert "retain the first emitted stop byte" in card
    assert "range 0.01-12.7; samples 7" in card
    assert "Trained-prompt answer-span BpB | range 0.01-12.7; samples 7" in card
    assert "model_card_heldout_v1.json" in card
    assert "Held-out prompt samples | 24" in card
    assert "repository name predates the measured 15-parameter checkpoint" in card
    assert "optimistic\nbaseline bound" in card


def test_results_record_must_be_a_json_object(tmp_path: Path) -> None:
    path = tmp_path / "results.json"
    path.write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError, match="must be a JSON object"):
        load_results_record(path)


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        ("- not\n- a mapping\n", "must be a YAML mapping"),
        ("other: {}\n", "missing model_card_release_gate"),
        (
            "model_card_release_gate:\n  heldout_bpb_unigram_margin_bits: true\n",
            "must be numeric",
        ),
        (
            "model_card_release_gate:\n  heldout_bpb_unigram_margin_bits: -0.1\n",
            "must be non-negative",
        ),
    ],
)
def test_release_gate_config_rejects_invalid_shapes_and_values(
    tmp_path: Path,
    payload: str,
    message: str,
) -> None:
    path = tmp_path / "release-gate.yaml"
    path.write_text(payload, encoding="utf-8")
    with pytest.raises(ValueError, match=message):
        load_release_gate_config(path)


def test_card_can_load_results_path_and_rejects_malformed_summaries(tmp_path: Path) -> None:
    record = {
        "evaluation": {
            "trained_prompt_answer_bpb": {"min": "low", "max": 2.0, "samples": 1},
            "heldout_prompt_answer_bpb": {
                "mean": 2.0,
                "min": 1.0,
                "max": 3.0,
                "samples": True,
            },
            "baselines": {},
        }
    }
    path = tmp_path / "results.json"
    path.write_text(json.dumps(record), encoding="utf-8")

    card = generate_model_card_readme(
        repo_id="example/path-loaded",
        model_config=ModelConfig(),
        num_experts=1,
        results_path=path,
    )

    assert "example/path-loaded" in card
    assert "Trained-prompt answer-span BpB | not measured" in card
    assert "Held-out answer-span BpB | not measured" in card


def _metric(
    mean: float,
    minimum: float,
    maximum: float,
    samples: int,
    unigram_mean: float,
) -> dict[str, object]:
    return {
        "mean": mean,
        "min": minimum,
        "max": maximum,
        "samples": samples,
        "evaluated_bytes": samples * 2,
        "baselines": {
            "uniform_bpb": {
                "mean": 8.0,
                "min": 8.0,
                "max": 8.0,
                "samples": samples,
            },
            "unigram_bpb": {
                "mean": unigram_mean,
                "min": unigram_mean,
                "max": unigram_mean,
                "samples": samples,
            },
            "unigram_fit": "evaluated_bytes",
        },
    }


def _valid_record(
    *, heldout_mean: float, heldout_unigram_mean: float
) -> dict[str, object]:
    provenance = {
        "schema_version": "aptivra.conversion_provenance.v1",
        "source_layout": "native_huggingface",
        "source_manifest_version": None,
        "source_shards_sha256": {},
        "source_hub_revision": None,
        "converter_version": "0.1.0",
        "model_safetensors_sha256": "0" * 64,
    }
    record: dict[str, object] = {
        "schema_version": "aptivra.model_card_eval.v3",
        "timestamp_utc": "2026-10-14T00:00:00+00:00",
        "checkpoint": {
            "model_safetensors_sha256": "0" * 64,
            "config_sha256": "1" * 64,
            "provenance_sha256": "2" * 64,
            "backbone_sha256": None,
            "source_hub_revision": None,
            "provenance": provenance,
        },
        "model": {
            "backbone_parameters": 10,
            "router_parameters": 2,
            "expert_parameters": 3,
            "total_parameters": 15,
            "config_parameter_count": 15,
            "context_length": 512,
            "vocab_size": 256,
            "position_embedding": "learned_absolute",
        },
        "evaluation": {
            "moe_enabled": False,
            "heldout_prompt_set": {
                "name": DEFAULT_HELDOUT_PROMPTS_PATH.name,
                "sha256": sha256_file(DEFAULT_HELDOUT_PROMPTS_PATH),
                "samples": 24,
            },
            "trained_prompt_answer_bpb": _metric(4.0, 0.01, 12.7, 7, 4.5),
            "heldout_prompt_answer_bpb": _metric(
                heldout_mean,
                heldout_mean,
                heldout_mean,
                24,
                heldout_unigram_mean,
            ),
            "heldout_text_bpb": {
                "value": 5.0,
                "evaluated_bytes": 20,
                "baselines": {
                    "uniform_bpb": 8.0,
                    "unigram_bpb": 4.2,
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
    return copy.deepcopy(record)


def _set_path(record: dict[str, object], path: tuple[str, ...], value: object) -> None:
    target: dict[str, object] = record
    for key in path[:-1]:
        nested = target[key]
        assert isinstance(nested, dict)
        target = nested
    target[path[-1]] = value

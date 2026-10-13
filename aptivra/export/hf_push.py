"""Validated publication of an already-evaluated Aptivra Hub layout."""

from __future__ import annotations

import hashlib
import hmac
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from huggingface_hub import HfApi

from aptivra.eval.model_card_schema import validate_evaluation_record
from aptivra.export.hf_hub_artifacts import HUB_LAYOUT_FILES
from aptivra.export.model_card import (
    generate_model_card_readme,
    load_release_gate_settings,
    load_results_record,
)
from aptivra.inference.hub_config import model_config_from_export_dir

HUB_PUBLICATION_FILES = (*HUB_LAYOUT_FILES, "README.md")

# These are the only obsolete remote paths a publication may remove. Keep this list
# explicit: broad cleanup globs could delete the stock-Transformers layout.
LEGACY_HUB_DELETE_PATTERNS = (
    "dense/*",
    "experts/*",
    "gguf/*",
    "*.gguf",
    "multitier_manifest.json",
    ".expert_usage",
)


@dataclass(frozen=True)
class HfPublicationPlan:
    """The complete, inspectable mutation sent to the Hub in one commit."""

    export_dir: Path
    upload_files: tuple[str, ...]
    delete_patterns: tuple[str, ...]


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _model_safetensors_hash(record: dict[str, Any]) -> str:
    checkpoint = record.get("checkpoint")
    if not isinstance(checkpoint, dict):
        raise ValueError("results record is missing the checkpoint object")
    recorded_hash = checkpoint.get("model_safetensors_sha256")
    if not isinstance(recorded_hash, str) or not recorded_hash:
        raise ValueError("results record is missing checkpoint.model_safetensors_sha256")
    return recorded_hash


def _num_experts(export_path: Path) -> int:
    config_path = export_path / "config.json"
    try:
        payload = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read Hub config: {config_path}") from exc
    value = payload.get("num_experts") if isinstance(payload, dict) else None
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError("config.json num_experts must be a non-negative integer")
    return value


def prepare_hf_publication(
    repo_id: str,
    export_dir: Path | str,
    results_record: Path | str | None = None,
) -> HfPublicationPlan:
    """Validate evidence and render the card without contacting the Hub.

    The layout must already exist. Publication never rewrites the evaluated weights.
    """
    if results_record is None:
        raise ValueError("a model-card results record is required for publication")

    export_path = Path(export_dir)
    missing = [name for name in HUB_LAYOUT_FILES if not (export_path / name).is_file()]
    if missing:
        raise FileNotFoundError(
            "Hub layout is incomplete; evaluate a complete layout before publishing. "
            f"Missing: {', '.join(missing)}"
        )

    results_path = Path(results_record)
    if not results_path.is_file():
        raise FileNotFoundError(f"model-card results record not found: {results_path}")

    weights_path = export_path / "model.safetensors"
    record = load_results_record(results_path)
    settings = load_release_gate_settings()
    validate_evaluation_record(
        record,
        minimum_trained_samples=settings.minimum_trained_prompt_samples,
        minimum_heldout_samples=settings.minimum_heldout_prompt_samples,
        export_dir=export_path,
    )
    recorded_hash = _model_safetensors_hash(record)
    actual_hash = _sha256_file(weights_path)
    if not hmac.compare_digest(recorded_hash.lower(), actual_hash):
        raise ValueError(
            "results record model.safetensors SHA-256 does not match the file being uploaded"
        )
    if results_path.stat().st_mtime_ns < weights_path.stat().st_mtime_ns:
        raise ValueError(
            "results record is older than model.safetensors; evaluate these weights again"
        )

    model_config = model_config_from_export_dir(export_path, device="cpu")
    readme = generate_model_card_readme(
        repo_id,
        model_config,
        _num_experts(export_path),
        results_record=record,
    )
    readme_path = export_path / "README.md"
    temporary = readme_path.with_name(f".{readme_path.name}.tmp")
    temporary.write_text(readme, encoding="utf-8")
    os.replace(temporary, readme_path)

    return HfPublicationPlan(
        export_dir=export_path,
        upload_files=HUB_PUBLICATION_FILES,
        delete_patterns=LEGACY_HUB_DELETE_PATTERNS,
    )


def push_aptivra_model_to_hf(
    repo_id: str,
    export_dir: Path | str,
    results_record: Path | str | None = None,
    *,
    private: bool = False,
    api: Any | None = None,
) -> str:
    """Publish one validated layout and remove only enumerated legacy artifacts."""
    plan = prepare_hf_publication(repo_id, export_dir, results_record)

    client = api or HfApi()
    print(f"Creating Hugging Face repository '{repo_id}'...")
    client.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True, private=private)

    print(f"Uploading Transformers layout to 'https://huggingface.co/{repo_id}'...")
    client.upload_folder(
        folder_path=str(plan.export_dir),
        repo_id=repo_id,
        repo_type="model",
        allow_patterns=list(plan.upload_files),
        delete_patterns=list(plan.delete_patterns),
        commit_message="Publish evaluated single-safetensors Transformers layout",
    )

    url = f"https://huggingface.co/{repo_id}"
    print(f"Upload complete! Model available at: {url}")
    return url

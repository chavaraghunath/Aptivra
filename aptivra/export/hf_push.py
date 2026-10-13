# ruff: noqa: E501
"""Hugging Face Hub Exporter for Aptivra Tiered Memory Models."""

from pathlib import Path
from typing import Any

from huggingface_hub import HfApi

from aptivra.experts.expert import ExpertModule
from aptivra.experts.router import GatingRouter
from aptivra.export.hf_hub_artifacts import HUB_LAYOUT_FILES, export_huggingface_hub_layout
from aptivra.export.model_card import generate_model_card_readme
from aptivra.model.config import ModelConfig
from aptivra.model.transformer import BackboneTransformer


def push_aptivra_model_to_hf(
    repo_id: str,
    model: BackboneTransformer,
    router: GatingRouter,
    experts: list[ExpertModule],
    model_config: ModelConfig,
    export_dir: Path | str = "scratch/hf_export",
    private: bool = False,
    api: Any | None = None,
) -> str:
    """Exports and uploads only the self-contained Transformers layout."""
    export_path = Path(export_dir)
    export_path.mkdir(parents=True, exist_ok=True)

    export_huggingface_hub_layout(
        export_path,
        model,
        router,
        experts,
        model_config,
    )

    client = api or HfApi()
    print(f"Creating Hugging Face repository '{repo_id}'...")
    client.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True, private=private)

    print(f"Uploading Transformers layout to 'https://huggingface.co/{repo_id}'...")
    client.upload_folder(
        folder_path=str(export_path),
        repo_id=repo_id,
        repo_type="model",
        allow_patterns=list(HUB_LAYOUT_FILES),
        commit_message="Publish verified single-safetensors Transformers layout",
    )

    url = f"https://huggingface.co/{repo_id}"
    print(f"Upload complete! Model available at: {url}")
    return url

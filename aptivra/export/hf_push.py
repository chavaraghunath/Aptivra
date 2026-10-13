# ruff: noqa: E501
"""Hugging Face Hub Exporter for Aptivra Tiered Memory Models."""

from pathlib import Path

from huggingface_hub import HfApi

from aptivra.experts.expert import ExpertModule
from aptivra.experts.router import GatingRouter
from aptivra.export.hf_hub_artifacts import stage_huggingface_hub_artifacts
from aptivra.export.model_card import generate_model_card_readme
from aptivra.export.multitier import MultitierExporter
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
) -> str:
    """Exports Aptivra model, router, and expert shards, and uploads folder to Hugging Face Hub."""
    export_path = Path(export_dir)
    export_path.mkdir(parents=True, exist_ok=True)

    # 1. Export Multitier model shards and manifest
    exporter = MultitierExporter(export_dir=export_path)
    exporter.export_model(model=model, router=router, experts=experts, model_config=model_config)

    # 2. Export GGUF binary models (FP16 & Q8_0)
    from aptivra.export.gguf_exporter import export_model_to_gguf

    gguf_dir = export_path / "gguf"
    gguf_dir.mkdir(parents=True, exist_ok=True)
    export_model_to_gguf(
        model=model,
        router=router,
        experts=experts,
        model_config=model_config,
        output_path=export_path / "aptivra-200m-fp16.gguf",
        quant_type="fp16",
    )
    export_model_to_gguf(
        model=model,
        router=router,
        experts=experts,
        model_config=model_config,
        output_path=export_path / "aptivra-200m-q8_0.gguf",
        quant_type="q8_0",
    )
    export_model_to_gguf(
        model=model,
        router=router,
        experts=experts,
        model_config=model_config,
        output_path=gguf_dir / "aptivra-200m-fp16.gguf",
        quant_type="fp16",
    )
    export_model_to_gguf(
        model=model,
        router=router,
        experts=experts,
        model_config=model_config,
        output_path=gguf_dir / "aptivra-200m-q8_0.gguf",
        quant_type="q8_0",
    )

    # 3. Write Enterprise Model Card README.md
    readme_content = generate_model_card_readme(
        repo_id=repo_id,
        model_config=model_config,
        num_experts=len(experts),
    )
    readme_path = export_path / "README.md"
    readme_path.write_text(readme_content, encoding="utf-8")

    stage_huggingface_hub_artifacts(
        export_dir=export_path,
        model_config=model_config,
        num_experts=len(experts),
    )

    # 4. Create repo and upload folder to Hugging Face Hub
    api = HfApi()
    print(f"Creating Hugging Face repository '{repo_id}'...")
    api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True, private=private)

    print(
        f"Uploading exported Aptivra model shards and GGUF binaries to 'https://huggingface.co/{repo_id}'..."
    )
    api.upload_folder(
        folder_path=str(export_path),
        repo_id=repo_id,
        repo_type="model",
        commit_message="Update model card, Transformers auto_map (H), and run options G–J",
    )

    url = f"https://huggingface.co/{repo_id}"
    print(f"Upload complete! Model available at: {url}")
    return url

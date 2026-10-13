"""CLI Script to push Aptivra 200M model to Hugging Face Hub."""

import argparse
from pathlib import Path

from aptivra.experts.expert import ExpertModule
from aptivra.experts.router import GatingRouter
from aptivra.export.hf_push import push_aptivra_model_to_hf
from aptivra.model.config import ModelConfig
from aptivra.model.transformer import BackboneTransformer


def main() -> None:
    parser = argparse.ArgumentParser(description="Push Aptivra 200M model to Hugging Face Hub")
    parser.add_argument("--repo-id", type=str, default="raghunath1/aptivra-200m-tiered-memory")
    parser.add_argument("--experts", type=int, default=8)
    parser.add_argument("--export-dir", type=str, default="scratch/hf_export")
    parser.add_argument("--regenerate", action="store_true", help="Force regenerate model shards")
    parser.add_argument("--private", action="store_true")
    args = parser.parse_args()

    model_config = ModelConfig(
        vocab_size=256,
        d_model=1024,
        n_layers=16,
        n_heads=16,
        d_ff=4096,
        device="cpu",
    )

    export_path = Path(args.export_dir)

    if not args.regenerate and (export_path / "aptivra-200m-fp16.gguf").exists():
        print(f"Existing model shards detected in '{export_path}'.")
        print("Updating Model Card and uploading to Hugging Face Hub...")
        from huggingface_hub import HfApi

        from aptivra.export.hf_hub_artifacts import stage_huggingface_hub_artifacts
        from aptivra.export.hf_push import generate_model_card_readme

        readme_content = generate_model_card_readme(
            repo_id=args.repo_id,
            model_config=model_config,
            num_experts=args.experts,
        )
        (export_path / "README.md").write_text(readme_content, encoding="utf-8")
        stage_huggingface_hub_artifacts(
            export_dir=export_path,
            model_config=model_config,
            num_experts=args.experts,
        )

        api = HfApi()
        print(f"Ensuring repository '{args.repo_id}' exists...")
        api.create_repo(
            repo_id=args.repo_id, repo_type="model", exist_ok=True, private=args.private
        )

        commit_msg = "Chat-aligned backbone: identity fine-tune, inference pipeline, hub sync"
        print(f"Uploading '{export_path}' to 'https://huggingface.co/{args.repo_id}'...")
        api.upload_folder(
            folder_path=str(export_path),
            repo_id=args.repo_id,
            repo_type="model",
            commit_message=commit_msg,
        )
        url = f"https://huggingface.co/{args.repo_id}"
    else:
        print(f"Initializing Aptivra 200M model with {args.experts} experts...")
        model = BackboneTransformer(model_config)
        router = GatingRouter(d_model=model_config.d_model, initial_num_experts=args.experts)
        experts = [
            ExpertModule(expert_id=i, d_model=model_config.d_model) for i in range(args.experts)
        ]

        url = push_aptivra_model_to_hf(
            repo_id=args.repo_id,
            model=model,
            router=router,
            experts=experts,
            model_config=model_config,
            export_dir=args.export_dir,
            private=args.private,
        )

    print(f"\nSUCCESS: Aptivra model successfully pushed to Hugging Face Hub!\nURL: {url}")


if __name__ == "__main__":
    main()

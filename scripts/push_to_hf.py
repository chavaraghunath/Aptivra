"""Push the Aptivra 271,361,024-parameter Transformers layout to Hugging Face Hub."""

import argparse
from pathlib import Path

from aptivra.export.hf_conversion import convert_legacy_export_to_huggingface
from aptivra.export.hf_push import push_aptivra_model_to_hf
from aptivra.inference.causal_lm import AptivraCausalLM
from aptivra.inference.hub_config import model_config_from_export_dir


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Push the Aptivra 271,361,024-parameter checkpoint to Hugging Face Hub"
    )
    parser.add_argument("--repo-id", type=str, default="raghunath1/aptivra-200m-tiered-memory")
    parser.add_argument("--export-dir", type=Path, default=Path("scratch/hf_export"))
    parser.add_argument(
        "--regenerate",
        action="store_true",
        help="Reconvert legacy shards even when model.safetensors already exists",
    )
    parser.add_argument("--private", action="store_true")
    args = parser.parse_args()

    export_path = args.export_dir
    weights_path = export_path / "model.safetensors"
    legacy_manifest = export_path / "multitier_manifest.json"
    if args.regenerate or not weights_path.is_file():
        if not legacy_manifest.is_file():
            raise SystemExit(
                "No model.safetensors or legacy multitier_manifest.json found; "
                "refusing to publish randomly initialized weights"
            )
        convert_legacy_export_to_huggingface(export_path)

    model_config = model_config_from_export_dir(export_path, device="cpu")
    lm = AptivraCausalLM.from_export_dir(export_path, model_config=model_config).eval()
    url = push_aptivra_model_to_hf(
        repo_id=args.repo_id,
        model=lm.backbone,
        router=lm.router,
        experts=list(lm.experts),
        model_config=model_config,
        export_dir=export_path,
        private=args.private,
    )
    print(f"SUCCESS: verified Aptivra layout pushed to {url}")


if __name__ == "__main__":
    main()

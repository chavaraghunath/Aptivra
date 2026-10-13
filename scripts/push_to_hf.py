"""Push the Aptivra 271,361,024-parameter Transformers layout to Hugging Face Hub."""

import argparse
from pathlib import Path

from aptivra.export.hf_conversion import convert_legacy_export_to_huggingface
from aptivra.export.hf_push import prepare_hf_publication, push_aptivra_model_to_hf


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Push the Aptivra 271,361,024-parameter checkpoint to Hugging Face Hub"
    )
    parser.add_argument("--repo-id", type=str, default="raghunath1/aptivra-200m-tiered-memory")
    parser.add_argument("--export-dir", type=Path, default=Path("scratch/hf_export"))
    parser.add_argument(
        "--results-record",
        type=Path,
        required=True,
        help="Fresh model-card evaluation JSON for the exact model.safetensors file",
    )
    parser.add_argument(
        "--regenerate",
        action="store_true",
        help="Reconvert legacy shards even when model.safetensors already exists",
    )
    parser.add_argument(
        "--source-hub-revision",
        default=None,
        help="Immutable source revision to record when converting legacy shards",
    )
    parser.add_argument("--private", action="store_true")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate and print the exact Hub mutation without contacting the network",
    )
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
        convert_legacy_export_to_huggingface(
            export_path,
            source_hub_revision=args.source_hub_revision,
        )

    if args.dry_run:
        plan = prepare_hf_publication(args.repo_id, export_path, args.results_record)
        print("Files to upload:")
        for filename in plan.upload_files:
            print(f"  {filename}")
        print("Remote files to delete:")
        for pattern in plan.delete_patterns:
            print(f"  {pattern}")
        return

    url = push_aptivra_model_to_hf(
        repo_id=args.repo_id,
        export_dir=export_path,
        results_record=args.results_record,
        private=args.private,
    )
    print(f"SUCCESS: verified Aptivra layout pushed to {url}")


if __name__ == "__main__":
    main()

"""Restore the self-contained Transformers export from Hugging Face Hub."""

from __future__ import annotations

import argparse
import shutil
from datetime import UTC, datetime
from pathlib import Path

from aptivra.export.hf_hub_artifacts import HUB_LAYOUT_FILES


def main() -> None:
    parser = argparse.ArgumentParser(description="Restore local export from Hugging Face")
    parser.add_argument("--repo-id", type=str, default="raghunath1/aptivra-200m-tiered-memory")
    parser.add_argument("--export-dir", type=Path, default=Path("scratch/hf_export"))
    parser.add_argument(
        "--backup",
        action="store_true",
        help="Backup current export before restore",
    )
    args = parser.parse_args()

    export_dir = args.export_dir
    if args.backup and export_dir.exists():
        stamp = datetime.now(tz=UTC).strftime("%Y%m%dT%H%M%SZ")
        backup = export_dir.parent / f"{export_dir.name}_backup_{stamp}"
        shutil.copytree(export_dir, backup)
        print(f"Backed up current export to {backup}")

    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise SystemExit('Install HF extras: pip install "huggingface_hub"') from exc

    local = snapshot_download(
        repo_id=args.repo_id,
        local_dir=str(export_dir),
        allow_patterns=list(HUB_LAYOUT_FILES),
    )
    print(f"Restored {args.repo_id} into {local}")


if __name__ == "__main__":
    main()

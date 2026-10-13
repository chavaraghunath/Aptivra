"""Convert legacy Aptivra shards into a verified stock-Transformers layout."""

from __future__ import annotations

import argparse
from pathlib import Path

from aptivra.export.hf_conversion import convert_legacy_export_to_huggingface


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_dir", type=Path, help="Directory containing legacy .pt shards")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Destination; defaults to adding the new layout beside the legacy shards",
    )
    args = parser.parse_args()

    weights = convert_legacy_export_to_huggingface(args.source_dir, args.output_dir)
    print(f"SUCCESS: verified Transformers checkpoint written to {weights}")


if __name__ == "__main__":
    main()

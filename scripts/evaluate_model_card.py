"""CLI for writing the measured-results JSON consumed by the model-card renderer."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from aptivra.eval.model_card_eval import (
    DEFAULT_HELDOUT_TEXT,
    evaluate_export_dir,
    load_prompt_answers,
    write_evaluation_record,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate an Aptivra export for model-card data")
    parser.add_argument("--export-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--hub-revision",
        type=str,
        default=None,
        help="Must match the source Hub revision already stored in conversion provenance",
    )
    parser.add_argument("--trained-prompts-json", type=Path, default=None)
    parser.add_argument("--heldout-prompts-json", type=Path, default=None)
    parser.add_argument("--heldout-text-file", type=Path, default=None)
    args = parser.parse_args()

    trained_prompts = (
        load_prompt_answers(args.trained_prompts_json)
        if args.trained_prompts_json is not None
        else None
    )
    heldout_text = (
        args.heldout_text_file.read_text(encoding="utf-8")
        if args.heldout_text_file is not None
        else None
    )

    record = evaluate_export_dir(
        args.export_dir,
        trained_prompts=trained_prompts,
        heldout_prompts_path=args.heldout_prompts_json,
        heldout_text=heldout_text if heldout_text is not None else DEFAULT_HELDOUT_TEXT,
        hub_revision=args.hub_revision,
    )
    write_evaluation_record(record, args.output)
    print(json.dumps({"output": str(args.output), "checkpoint": record["checkpoint"]}))


if __name__ == "__main__":
    main()

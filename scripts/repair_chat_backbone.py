"""Urgent repair: fine-tune backbone on chat data (UTF-8 + full transformer forward)."""

from __future__ import annotations

import argparse
from pathlib import Path

from aptivra.v0.inference.chat_finetune import finetune_backbone_for_chat


def main() -> None:
    parser = argparse.ArgumentParser(description="Repair chat-readable backbone weights")
    parser.add_argument("--export-dir", type=Path, default=Path("scratch/hf_export"))
    parser.add_argument("--epochs", type=int, default=45)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--early-stop-patience", type=int, default=10)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()

    out = finetune_backbone_for_chat(
        args.export_dir,
        epochs=args.epochs,
        lr=args.lr,
        device=args.device,
        repeats=args.repeats,
        include_conversational=False,
        early_stop_patience=args.early_stop_patience,
    )
    print(f"Re-exported chat-aligned backbone to {out}")


if __name__ == "__main__":
    main()

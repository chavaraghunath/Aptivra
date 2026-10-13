"""Phase A (multidomain train + export + eval) then Phase B (identity SFT + verify + HF push)."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def run(cmd: list[str], *, cwd: Path) -> None:
    print("\n>>", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=cwd, check=True)


def _last_identity_accuracy(eval_path: Path, tag: str) -> float | None:
    if not eval_path.is_file():
        return None
    acc: float | None = None
    for line in eval_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("tag") == tag:
            acc = float(row["identity_accuracy"])
    return acc


def main() -> None:
    parser = argparse.ArgumentParser(description="Run continual learning phase A then chat phase B")
    parser.add_argument("--export-dir", type=Path, default=Path("scratch/hf_export"))
    parser.add_argument("--steps-per-task", type=int, default=25)
    parser.add_argument("--train-seed", type=int, default=42)
    parser.add_argument("--repair-epochs", type=int, default=45)
    parser.add_argument("--repair-repeats", type=int, default=10)
    parser.add_argument("--skip-restore", action="store_true")
    parser.add_argument("--skip-a", action="store_true")
    parser.add_argument("--skip-b", action="store_true")
    parser.add_argument("--repo-id", type=str, default="raghunath1/aptivra-200m-tiered-memory")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    py = root / ".venv" / "bin" / "python"
    if not py.is_file():
        raise SystemExit("Missing .venv/bin/python — create venv and install aptivra first.")

    export = args.export_dir
    eval_path = root / "experiments" / "identity_export_eval.jsonl"

    if not args.skip_restore:
        print("=== Restore export from Hugging Face (backup local first) ===")
        run(
            [
                str(py),
                "scripts/restore_hf_export.py",
                "--repo-id",
                args.repo_id,
                "--export-dir",
                str(export),
                "--backup",
            ],
            cwd=root,
        )

    if not args.skip_a:
        print("=== Phase A: baseline identity eval ===")
        run(
            [
                str(py),
                "scripts/eval_identity_export.py",
                "--export-dir",
                str(export),
                "--tag",
                "pre_phase_a",
            ],
            cwd=root,
        )
        pre_acc = _last_identity_accuracy(eval_path, "pre_phase_a")
        print("=== Phase A: light multidomain (identity + skills, no dialogue/prune) ===")
        run(
            [
                str(py),
                "scripts/train_multidomain_conversational.py",
                "--no-hf-push",
                "--steps-per-task",
                str(args.steps_per_task),
                "--seeds",
                str(args.train_seed),
            ],
            cwd=root,
        )
        print("=== Phase A: post-train identity eval ===")
        run(
            [
                str(py),
                "scripts/eval_identity_export.py",
                "--export-dir",
                str(export),
                "--tag",
                "post_phase_a",
            ],
            cwd=root,
        )
        post_acc = _last_identity_accuracy(eval_path, "post_phase_a")
        if pre_acc is not None and post_acc is not None and post_acc < pre_acc * 0.75:
            raise SystemExit(
                f"Phase A regressed identity ({pre_acc:.2f} -> {post_acc:.2f}). "
                "Halting before Phase B; restore and investigate."
            )

    if not args.skip_b:
        print("=== Phase B: identity-only chat SFT ===")
        run(
            [
                str(py),
                "scripts/repair_chat_backbone.py",
                "--export-dir",
                str(export),
                "--epochs",
                str(args.repair_epochs),
                "--repeats",
                str(args.repair_repeats),
                "--device",
                "auto",
            ],
            cwd=root,
        )
        print("=== Phase B: raw chat quality gate ===")
        run([str(py), "scripts/verify_chat_quality.py", "--export-dir", str(export)], cwd=root)
        print("=== Phase B: Hugging Face push ===")
        run(
            [
                str(py),
                "scripts/push_to_hf.py",
                "--repo-id",
                args.repo_id,
                "--export-dir",
                str(export),
            ],
            cwd=root,
        )

    print("\nPhase A→B pipeline complete.")


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as exc:
        print(f"Pipeline failed with exit code {exc.returncode}", file=sys.stderr)
        sys.exit(exc.returncode)

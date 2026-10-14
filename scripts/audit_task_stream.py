"""Audit versioned synthetic continual-learning task streams."""

import argparse
from pathlib import Path

from aptivra.eval.task_stream_audit import load_audit_plan, write_or_check_plan


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--plan",
        type=Path,
        default=Path("configs/task_stream_audit.yaml"),
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Fail if recorded audit outputs differ from freshly computed records.",
    )
    args = parser.parse_args()
    plan = load_audit_plan(args.plan)
    write_or_check_plan(plan, check=args.check)
    action = "verified" if args.check else "recorded"
    print(f"Task stream audits {action} successfully.")


if __name__ == "__main__":
    main()

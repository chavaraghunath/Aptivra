"""Build the local public GitHub issues continual-learning stream."""

import argparse
from pathlib import Path

from aptivra.tasks.github_issues import (
    DEFAULT_GITHUB_ISSUES_CONFIG,
    GitHubIssueStreamBuilder,
    load_github_issue_builder_config,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build Aptivra's local GitHub issue continual-learning stream."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_GITHUB_ISSUES_CONFIG,
        help="Path to the GitHub issue stream builder YAML.",
    )
    args = parser.parse_args()

    config = load_github_issue_builder_config(args.config)
    result = GitHubIssueStreamBuilder(config).build()
    print(
        "Built GitHub issue stream: "
        f"{result.accepted_records} accepted, {result.dropped_records} dropped, "
        f"{result.elapsed_seconds:.2f}s elapsed"
    )
    print(f"Local dataset: {result.raw_dataset_path}")
    print(f"Manifest: {result.manifest_path}")
    print(f"Audit: {result.audit_path}")


if __name__ == "__main__":
    main()

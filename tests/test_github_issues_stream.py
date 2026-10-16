"""Tests for the public GitHub issues continual-learning stream."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest

from aptivra.eval.harness import ContinualEvalHarness
from aptivra.tasks.github_issues import (
    GitHubIssueBuilderConfig,
    GitHubIssueCoverageConfig,
    GitHubIssueRenameDetectionConfig,
    GitHubIssueRepositoryConfig,
    GitHubIssueSamplingConfig,
    GitHubIssuesContinualStream,
    GitHubIssueSplitConfig,
    GitHubIssueStreamBuilder,
    GitHubIssueWindow,
    canonical_label_string,
    content_hash,
    detect_label_rename_candidates,
    drift_against_resampling_baseline,
    events_per_month,
    load_github_issue_builder_config,
    sanitize_issue_text,
    split_counts,
    tfidf_centroid_baselines,
    validate_window_coverage,
)


def test_sanitize_masks_email_and_generic_token_but_drops_obvious_secret() -> None:
    masked = sanitize_issue_text(
        "contact dev@example.com with token=abc123456789xyz"  # gitleaks:allow
    )
    assert masked.drop_reason is None
    assert "[EMAIL]" in masked.text
    assert "[SECRET]" in masked.text

    dropped = sanitize_issue_text("accidentally posted ghp_abcdefghijklmnopqrstuvwxyz")
    assert dropped.drop_reason == "secret"


def test_canonical_label_string_is_stable() -> None:
    assert canonical_label_string(["terminal"]) == "terminal"


def test_split_counts_leave_chronological_test_suffix() -> None:
    assert split_counts(1, 0.7, 0.1) == (1, 0)
    assert split_counts(2, 0.7, 0.1) == (1, 0)
    assert split_counts(10, 0.7, 0.1) == (7, 1)


def test_builder_uses_handwritten_clickhouse_fixture_and_stream_verifies_hashes(
    tmp_path: Path,
) -> None:
    config = _fixture_config(tmp_path)
    queries: list[str] = []

    def fixture_query(query: str) -> dict[str, Any]:
        queries.append(query)
        if "GROUP BY month" in query:
            if "2019-01-01" in query:
                return {
                    "data": [
                        {"month": "2019-01", "eligible_rows": 10},
                        {"month": "2019-02", "eligible_rows": 10},
                        {"month": "2019-05", "eligible_rows": 10},
                    ],
                    "statistics": {"elapsed": 0.001},
                }
            return {
                "data": [
                    {"month": "2020-01", "eligible_rows": 10},
                    {"month": "2020-02", "eligible_rows": 10},
                    {"month": "2020-03", "eligible_rows": 10},
                ],
                "statistics": {"elapsed": 0.001},
            }
        if "action IN ('labeled', 'unlabeled')" in query:
            return {
                "data": [
                    {"number": 1, "edit_events": 2},
                    {"number": 2, "edit_events": 1},
                ],
                "statistics": {"elapsed": 0.001},
            }
        if "2019-01-01" in query:
            return {
                "data": [
                    _issue_row(
                        1,
                        "Terminal crashes on empty command",
                        "The terminal crashes when I run an empty shell command.",
                        ["integrated-terminal"],
                        "2019-01-05 00:00:00",
                    ),
                    _issue_row(
                        2,
                        "Debugger ignores breakpoint",
                        "Please reply to dev@example.com. token=abc123456789xyz",  # gitleaks:allow
                        ["debug"],
                        "2019-02-05 00:00:00",
                    ),
                    _issue_row(
                        3,
                        "Multi component report",
                        "The terminal debug integration is confused.",
                        ["terminal", "debug"],
                        "2019-03-05 00:00:00",
                    ),
                    _issue_row(
                        4,
                        "Secret-bearing report",
                        "The accidental key was ghp_abcdefghijklmnopqrstuvwxyz.",
                        ["terminal"],
                        "2019-04-05 00:00:00",
                    ),
                    _issue_row(
                        5,
                        "Notebook rendering fails",
                        "Notebook output rendering fails for markdown output.",
                        ["notebook"],
                        "2019-05-05 00:00:00",
                    ),
                ],
                "statistics": {"elapsed": 0.002},
            }
        return {
            "data": [
                _issue_row(
                    6,
                    "Terminal shell integration fails",
                    "The shell integration sequence is printed in the terminal.",
                    ["terminal-shell-integration"],
                    "2020-01-05 00:00:00",
                ),
                _issue_row(
                    7,
                    "Git decoration missing",
                    "Git decorations disappear after repository refresh.",
                    ["git"],
                    "2020-02-05 00:00:00",
                ),
                _issue_row(
                    8,
                    "Notebook execution failure",
                    "Notebook execution fails after kernel restart.",
                    ["notebook"],
                    "2020-03-05 00:00:00",
                ),
            ],
            "statistics": {"elapsed": 0.002},
        }

    result = GitHubIssueStreamBuilder(config, query_clickhouse=fixture_query).build()

    assert queries
    issue_queries = [query for query in queries if "row_number() OVER" in query]
    assert issue_queries
    assert all("PARTITION BY toYYYYMM(created_at)" in query for query in issue_queries)
    assert all("ORDER BY created_at ASC, number ASC" not in query for query in issue_queries)
    assert result.accepted_records == 6
    assert result.dropped_records == 2
    manifest_text = config.manifest_path.read_text(encoding="utf-8")
    assert "Terminal crashes on empty command" not in manifest_text
    assert "Debugger ignores breakpoint" not in manifest_text
    assert "raw_text" in manifest_text
    manifest = json.loads(manifest_text)
    assert manifest["schema_version"] == 3
    assert manifest["record_count"] == 6
    assert "records" not in manifest
    assert len(manifest["record_shards"]) == 1
    shard_path = config.manifest_path.parent / manifest["record_shards"][0]["path"]
    assert shard_path.stat().st_size < 500 * 1024

    raw_rows = [
        json.loads(line)
        for line in config.raw_dataset_path.read_text(encoding="utf-8").splitlines()
    ]
    assert any("[EMAIL]" in row["prompt"] for row in raw_rows)
    assert any("[SECRET]" in row["prompt"] for row in raw_rows)
    for row in raw_rows:
        assert (
            content_hash(row["prompt"], row["target_solution"])
            == row["metadata"]["content_hash"]
        )
    assert all(
        len(row["metadata"]["canonical_labels"]) == 1
        for row in raw_rows
    )
    aliased = next(row for row in raw_rows if row["metadata"]["number"] == 1)
    assert aliased["metadata"]["raw_target_labels"] == ["integrated-terminal"]
    assert aliased["metadata"]["canonical_labels"] == ["terminal"]
    assert aliased["target_solution"] == "terminal"

    stream = GitHubIssuesContinualStream(config.manifest_path, config.raw_dataset_path)
    family = stream.families[0]
    split = stream.get_split(family.family_id)
    assert (len(split.train), len(split.val), len(split.test)) == (2, 0, 1)
    assert family.verifier.verify(split.test[0], split.test[0].target_solution).passed
    assert not family.verifier.verify(split.test[0], "debug").passed

    harness = ContinualEvalHarness(stream)

    def oracle(prompt: str, *, stop_token_ids: tuple[int, ...]) -> str:
        del stop_token_ids
        for instance in split.test:
            if instance.prompt == prompt:
                return instance.target_solution
        return "wrong"

    assert harness.evaluate_task(oracle, 0) == 1.0
    audit = stream.audit_summary()[family.family_id]
    assert audit["distinct_targets"] == 3
    assert audit["overlap"] == {"train_test": 0, "train_val": 0, "val_test": 0}
    assert audit["label_noise"]["accepted_records_with_label_edit_events"] == 2
    assert "tfidf_centroid_baseline" in audit
    assert audit["query_rows_returned"] == 5
    assert audit["coverage"]["months_represented"] == 3
    assert audit["coverage"]["events_per_month"] == {
        "2019-01": 1,
        "2019-02": 1,
        "2019-05": 1,
    }
    assert audit["label_aliases"]["alias_counts"] == {
        "integrated-terminal->terminal": 1
    }

    truncated_path = tmp_path / "truncated.jsonl"
    truncated_path.write_text(
        "\n".join(config.raw_dataset_path.read_text(encoding="utf-8").splitlines()[:-1])
        + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="incomplete relative to the manifest"):
        GitHubIssuesContinualStream(config.manifest_path, truncated_path)

    duplicate_path = tmp_path / "duplicate.jsonl"
    duplicate_path.write_text(
        config.raw_dataset_path.read_text(encoding="utf-8")
        + config.raw_dataset_path.read_text(encoding="utf-8").splitlines()[0]
        + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="duplicate record_id"):
        GitHubIssuesContinualStream(config.manifest_path, duplicate_path)

    shard_path.write_text("[]\n", encoding="utf-8")
    with pytest.raises(ValueError, match="shard hash mismatch"):
        GitHubIssuesContinualStream(config.manifest_path, config.raw_dataset_path)


def test_drift_resampling_baseline_reports_observed_and_same_window_statistics() -> None:
    previous = [
        _raw_record("a", "terminal shell terminal", "terminal", "2019", "train"),
        _raw_record("b", "debug breakpoint debug", "debug", "2019", "test"),
        _raw_record("c", "terminal command shell", "terminal", "2019", "train"),
        _raw_record("d", "debug console breakpoint", "debug", "2019", "test"),
    ]
    current = [
        _raw_record("e", "notebook cell output", "notebook", "2020", "train"),
        _raw_record("f", "notebook kernel execution", "notebook", "2020", "test"),
        _raw_record("g", "git repository decoration", "git", "2020", "train"),
        _raw_record("h", "git branch source control", "git", "2020", "test"),
    ]

    drift = drift_against_resampling_baseline(previous, current, seed=3)

    assert drift["label_js_divergence"] > 0.9
    assert drift["vocabulary_jaccard_distance"] > 0.9
    assert "label_js_previous_window_resampling_baseline" in drift
    assert "vocabulary_jaccard_current_window_resampling_baseline" in drift


def test_tfidf_centroid_baseline_trains_current_and_earlier_window_models() -> None:
    by_family = {
        "github_issues_2019": [
            _raw_record("a", "terminal shell command", "terminal", "2019", "train"),
            _raw_record("b", "debug breakpoint console", "debug", "2019", "train"),
            _raw_record("c", "terminal shell panel", "terminal", "2019", "test"),
        ],
        "github_issues_2020": [
            _raw_record("d", "notebook cell output", "notebook", "2020", "train"),
            _raw_record("e", "git branch repository", "git", "2020", "train"),
            _raw_record("f", "notebook kernel output", "notebook", "2020", "test"),
        ],
    }

    baselines = tfidf_centroid_baselines(by_family)

    assert baselines["github_issues_2019"]["current_window_train_accuracy_on_test"] == 1.0
    assert baselines["github_issues_2019"]["earlier_windows_train_accuracy_on_test"] is None
    assert baselines["github_issues_2020"]["current_window_train_accuracy_on_test"] == 1.0
    assert baselines["github_issues_2020"]["earlier_windows_train_records"] == 2


def test_window_coverage_counts_months_and_fails_when_below_threshold() -> None:
    records = [
        _raw_record("a", "terminal shell command", "terminal", "2020", "train"),
        _raw_record("b", "debug breakpoint console", "debug", "2020", "test"),
    ]
    records[1]["metadata"]["event_created_at"] = "2020-02-01 00:00:00"

    assert events_per_month(records) == {"2020-01": 1, "2020-02": 1}
    validate_window_coverage(
        records,
        [GitHubIssueWindow("2020", "2020-01-01", "2021-01-01")],
        min_months_represented=2,
    )

    try:
        validate_window_coverage(
            records,
            [GitHubIssueWindow("2020", "2020-01-01", "2021-01-01")],
            min_months_represented=3,
        )
    except ValueError as exc:
        assert "2020: 2/3 months" in str(exc)
    else:
        raise AssertionError("coverage gate accepted an under-covered window")


def test_rename_detector_reports_temporal_handoff_with_similar_text() -> None:
    records = [
        _raw_record("old1", "terminal shell command panel", "terminal", "2021", "train"),
        _raw_record("old2", "terminal shell profile command", "terminal", "2021", "test"),
        _raw_record(
            "new1",
            "terminal shell integration command panel",
            "terminal-shell-integration",
            "2022",
            "train",
        ),
        _raw_record(
            "new2",
            "terminal shell integration profile",
            "terminal-shell-integration",
            "2022",
            "test",
        ),
        _raw_record("git1", "git branch commit repository", "git", "2021", "train"),
        _raw_record("git2", "git status repository branch", "git", "2021", "test"),
        _raw_record("scm1", "scm branch commit repository", "scm", "2022", "train"),
        _raw_record("scm2", "scm source control repository", "scm", "2022", "test"),
        _raw_record("debug1", "debug breakpoint console", "debug", "2020", "train"),
        _raw_record("debug2", "debug breakpoint watch", "debug", "2020", "test"),
    ]

    candidates = detect_label_rename_candidates(
        records,
        min_label_records=2,
        max_handoff_days=455,
        min_centroid_similarity=0.2,
    )
    pairs = {(item["old_label"], item["new_label"]) for item in candidates}

    assert ("terminal", "terminal-shell-integration") in pairs
    assert ("git", "scm") in pairs
    assert all("debug" not in pair for pair in pairs)


def test_config_loader_rejects_disabled_non_boolean(tmp_path: Path) -> None:
    config = tmp_path / "bad.yaml"
    config.write_text(
        """github_issues_stream:
  enabled: yes please
  source: gh_archive_clickhouse
  output_dir: scratch/github_issues_stream
  raw_dataset_path: scratch/github_issues_stream/issues.jsonl
  manifest_path: docs/github-issues-stream/manifest.json
  audit_path: docs/github-issues-stream/audit.json
  clickhouse_endpoint: https://play.clickhouse.com/
  clickhouse_user: play
  user_agent: aptivra
  query_timeout_seconds: 1.0
  max_records_per_window: 1000
  min_body_chars: 0
  max_body_chars: 4000
  near_duplicate_jaccard: 0.9
  drift_resampling_seed: 1
  canonical_label_aliases: {}
  sampling:
    method: month_stratified_hash
    seed: 1
  coverage:
    min_months_represented: 1
  rename_detection:
    min_label_records: 2
    max_handoff_days: 455
    min_centroid_similarity: 0.2
  split:
    train_fraction: 0.7
    val_fraction: 0.1
  windows:
    - id: "2019"
      start: "2019-01-01"
      end: "2020-01-01"
  repositories:
    - name: example/project
      license_spdx: MIT
      target_labels:
        - terminal
""",
        encoding="utf-8",
    )

    try:
        load_github_issue_builder_config(config)
    except ValueError as exc:
        assert "enabled" in str(exc)
    else:
        raise AssertionError("malformed enabled value was accepted")


def _fixture_config(tmp_path: Path) -> GitHubIssueBuilderConfig:
    return GitHubIssueBuilderConfig(
        enabled=True,
        source="gh_archive_clickhouse",
        output_dir=tmp_path / "scratch" / "github_issues_stream",
        raw_dataset_path=tmp_path / "scratch" / "github_issues_stream" / "issues.jsonl",
        manifest_path=tmp_path / "docs" / "github-issues-stream" / "manifest.json",
        audit_path=tmp_path / "docs" / "github-issues-stream" / "audit.json",
        clickhouse_endpoint="https://play.clickhouse.com/",
        clickhouse_user="play",
        user_agent="aptivra-test",
        query_timeout_seconds=1.0,
        max_records_per_window=1000,
        min_body_chars=5,
        max_body_chars=4000,
        near_duplicate_jaccard=0.95,
        drift_resampling_seed=7,
        canonical_label_aliases={"integrated-terminal": "terminal"},
        sampling=GitHubIssueSamplingConfig(method="month_stratified_hash", seed=23),
        coverage=GitHubIssueCoverageConfig(min_months_represented=2),
        rename_detection=GitHubIssueRenameDetectionConfig(
            min_label_records=2,
            max_handoff_days=455,
            min_centroid_similarity=0.2,
        ),
        windows=(
            GitHubIssueWindow("2019", "2019-01-01", "2020-01-01"),
            GitHubIssueWindow("2020", "2020-01-01", "2021-01-01"),
        ),
        split=GitHubIssueSplitConfig(train_fraction=0.7, val_fraction=0.1),
        repositories=(
            GitHubIssueRepositoryConfig(
                name="example/project",
                license_spdx="MIT",
                target_labels=(
                    "debug",
                    "git",
                    "integrated-terminal",
                    "notebook",
                    "terminal",
                    "terminal-shell-integration",
                ),
            ),
        ),
    )


def _issue_row(
    number: int,
    title: str,
    body: str,
    labels: list[str],
    event_created_at: str,
) -> dict[str, Any]:
    return {
        "repo_name": "example/project",
        "number": number,
        "title": title,
        "body": body,
        "labels": labels,
        "event_created_at": event_created_at,
        "issue_updated_at": event_created_at,
        "issue_closed_at": event_created_at,
    }


def _raw_record(
    record_id: str,
    prompt: str,
    target: str,
    window_id: str,
    split: str,
) -> dict[str, Any]:
    metadata: Mapping[str, Any] = {
        "record_id": record_id,
        "repo": "example/project",
        "number": len(record_id),
        "html_url": f"https://github.com/example/project/issues/{len(record_id)}",
        "event_created_at": f"{window_id}-01-01 00:00:00",
        "issue_updated_at": f"{window_id}-01-01 00:00:00",
        "issue_closed_at": f"{window_id}-01-01 00:00:00",
        "window_id": window_id,
        "split": split,
        "source_labels": [target],
        "raw_target_labels": [target],
        "canonical_labels": [target],
        "target": target,
        "content_hash": content_hash(prompt, target),
        "label_edit_event_count": 0,
    }
    return {
        "record_id": record_id,
        "prompt": prompt,
        "target_solution": target,
        "metadata": metadata,
    }

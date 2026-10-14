"""GitHub issue continual-learning stream and local dataset builder."""

from __future__ import annotations

import hashlib
import json
import math
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any, TypedDict

import yaml

from aptivra.tasks.base import (
    MechanicalVerifier,
    TaskFamily,
    TaskInstance,
    VerifierResult,
    exact_match_result,
)
from aptivra.tasks.stream import TaskStreamSplit

JsonObject = dict[str, Any]
ClickHouseQuery = Callable[[str], JsonObject]

EMAIL_RE = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
HIGH_CONFIDENCE_SECRET_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\bgh[pousr]_[A-Za-z0-9_]{20,}\b"),
    re.compile(r"\bgithub_pat_[A-Za-z0-9_]{20,}\b"),
    re.compile(r"\bsk-[A-Za-z0-9]{20,}\b"),
    re.compile(r"\bxox[baprs]-[A-Za-z0-9-]{20,}\b"),
    re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH |DSA )?PRIVATE KEY-----"),
    re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
)
MASKED_SECRET_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(
        r"(?i)\b(api[_-]?key|token|secret|password)\s*[:=]\s*"
        r"['\"]?[A-Za-z0-9._~+/=-]{12,}['\"]?"
    ),
    re.compile(r"(?i)\bAuthorization:\s*Bearer\s+[A-Za-z0-9._~+/=-]{12,}"),
)
TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9_+-]{1,}")


class ManifestRecord(TypedDict):
    """Manifest entry that intentionally excludes issue title and body text."""

    record_id: str
    repo: str
    number: int
    html_url: str
    event_created_at: str
    issue_updated_at: str
    issue_closed_at: str
    window_id: str
    split: str
    source_labels: list[str]
    raw_target_labels: list[str]
    canonical_labels: list[str]
    target: str
    content_hash: str
    label_edit_event_count: int


class RawDatasetRecord(TypedDict):
    """Local-only dataset row containing sanitized issue text."""

    record_id: str
    prompt: str
    target_solution: str
    metadata: ManifestRecord


@dataclass(frozen=True)
class GitHubIssueWindow:
    """Fixed calendar window for one continual-learning task."""

    window_id: str
    start: str
    end: str

    def contains(self, event_created_at: str) -> bool:
        """Return whether an ISO timestamp falls inside this half-open date window."""
        created_date = event_created_at[:10]
        return self.start <= created_date < self.end


@dataclass(frozen=True)
class GitHubIssueRepositoryConfig:
    """Repository and target labels used by the GitHub issue stream builder."""

    name: str
    license_spdx: str
    target_labels: tuple[str, ...]


@dataclass(frozen=True)
class GitHubIssueSplitConfig:
    """Chronological split fractions within each calendar window."""

    train_fraction: float
    val_fraction: float


@dataclass(frozen=True)
class GitHubIssueSamplingConfig:
    """Deterministic record sampling policy for each source window."""

    method: str
    seed: int


@dataclass(frozen=True)
class GitHubIssueCoverageConfig:
    """Minimum accepted-record month coverage required for a source window."""

    min_months_represented: int


@dataclass(frozen=True)
class GitHubIssueRenameDetectionConfig:
    """Heuristics used to report possible taxonomy label renames."""

    min_label_records: int
    max_handoff_days: int
    min_centroid_similarity: float


@dataclass(frozen=True)
class GitHubIssueBuilderConfig:
    """Validated configuration for building the local GitHub issue dataset."""

    enabled: bool
    source: str
    output_dir: Path
    raw_dataset_path: Path
    manifest_path: Path
    audit_path: Path
    clickhouse_endpoint: str
    clickhouse_user: str
    user_agent: str
    query_timeout_seconds: float
    max_records_per_window: int
    min_body_chars: int
    max_body_chars: int
    near_duplicate_jaccard: float
    drift_resampling_seed: int
    canonical_label_aliases: Mapping[str, str]
    sampling: GitHubIssueSamplingConfig
    coverage: GitHubIssueCoverageConfig
    rename_detection: GitHubIssueRenameDetectionConfig
    windows: tuple[GitHubIssueWindow, ...]
    split: GitHubIssueSplitConfig
    repositories: tuple[GitHubIssueRepositoryConfig, ...]


@dataclass(frozen=True)
class SanitizedText:
    """Sanitized issue text and the safety action taken."""

    text: str
    masked_email_count: int
    masked_secret_count: int
    drop_reason: str | None


@dataclass(frozen=True)
class BuilderResult:
    """Paths and summary returned by a completed dataset build."""

    raw_dataset_path: Path
    manifest_path: Path
    audit_path: Path
    accepted_records: int
    dropped_records: int
    elapsed_seconds: float
    disk_usage_bytes: int


class GitHubIssueVerifier(MechanicalVerifier):
    """Exact-match verifier for canonical GitHub issue component label strings."""

    def verify(self, instance: TaskInstance, model_output: str) -> VerifierResult:
        return exact_match_result(instance, model_output)


class GitHubIssueTriageTaskFamily(TaskFamily):
    """One calendar window of GitHub issue component-label examples."""

    def __init__(
        self,
        family_id: str,
        name: str,
        description: str,
        instances: Sequence[TaskInstance],
        audit: Mapping[str, Any],
    ) -> None:
        super().__init__(family_id, name, description)
        self._instances = list(instances)
        self.audit = dict(audit)
        self._verifier = GitHubIssueVerifier()

    @property
    def verifier(self) -> MechanicalVerifier:
        return self._verifier

    def generate_instances(self, count: int, seed: int) -> list[TaskInstance]:
        """Return materialized instances deterministically; generation is not synthetic."""
        del seed
        return self._instances[:count]


class GitHubIssuesContinualStream:
    """Manifest-backed GitHub issue stream matching the evaluation harness protocol."""

    def __init__(
        self, manifest_path: Path | str, raw_dataset_path: Path | str | None = None
    ) -> None:
        manifest = _read_json_object(Path(manifest_path))
        dataset_path = (
            Path(raw_dataset_path)
            if raw_dataset_path is not None
            else Path(_require_str(manifest, "raw_dataset_path"))
        )
        raw_records = _read_jsonl_dataset(dataset_path)
        manifest_records = _manifest_records(manifest)
        by_record_id = {record["record_id"]: record for record in manifest_records}
        if len(by_record_id) != len(manifest_records):
            raise ValueError("GitHub issue manifest contains duplicate record_id values")

        self.seed = int(manifest.get("seed", 0))
        self.num_train_per_task = 0
        self.num_val_per_task = 0
        self.num_test_per_task = 0
        self._splits: dict[str, TaskStreamSplit] = {}
        self._audit_by_family: dict[str, dict[str, Any]] = {}
        by_window: dict[str, dict[str, list[TaskInstance]]] = defaultdict(
            lambda: {"train": [], "val": [], "test": []}
        )

        for raw_record in raw_records:
            record_id = raw_record["record_id"]
            manifest_record = by_record_id.get(record_id)
            if manifest_record is None:
                raise ValueError(f"Local dataset has record absent from manifest: {record_id}")
            if raw_record["metadata"]["content_hash"] != manifest_record["content_hash"]:
                raise ValueError(f"Local dataset hash mismatch for {record_id}")
            expected_hash = content_hash(raw_record["prompt"], raw_record["target_solution"])
            if expected_hash != manifest_record["content_hash"]:
                raise ValueError(f"Local dataset content hash verification failed for {record_id}")
            family_id = _family_id(manifest_record["window_id"])
            instance = TaskInstance(
                task_id=f"github_issue:{record_id}",
                task_family=family_id,
                prompt=raw_record["prompt"],
                target_solution=raw_record["target_solution"],
                metadata=dict(manifest_record),
            )
            by_window[manifest_record["window_id"]][manifest_record["split"]].append(instance)

        audit = _require_mapping(manifest, "audit")
        windows = _require_list(manifest, "windows")
        self.families: list[TaskFamily] = []
        for window in windows:
            window_map = _ensure_mapping(window, "window")
            window_id = _require_str(window_map, "id")
            family_id = _family_id(window_id)
            split_map = by_window[window_id]
            split = TaskStreamSplit(
                train=split_map["train"], val=split_map["val"], test=split_map["test"]
            )
            self._splits[family_id] = split
            self.num_train_per_task = max(self.num_train_per_task, len(split.train))
            self.num_val_per_task = max(self.num_val_per_task, len(split.val))
            self.num_test_per_task = max(self.num_test_per_task, len(split.test))
            family_audit = dict(_ensure_mapping(audit.get(family_id, {}), family_id))
            self._audit_by_family[family_id] = family_audit
            self.families.append(
                GitHubIssueTriageTaskFamily(
                    family_id=family_id,
                    name=f"GitHub Issues {window_id}",
                    description=(
                        "Predict the canonical component label for public GitHub issues "
                        f"closed in calendar window {window_id}."
                    ),
                    instances=split.train + split.val + split.test,
                    audit=family_audit,
                )
            )

    def get_split(self, family_id: str) -> TaskStreamSplit:
        """Return materialized split for a GitHub issue calendar window."""
        try:
            return self._splits[family_id]
        except KeyError as exc:
            raise KeyError(f"Task family '{family_id}' not found in stream.") from exc

    def audit_summary(self) -> dict[str, dict[str, Any]]:
        """Return per-window audit fields committed with the manifest."""
        return {family_id: dict(audit) for family_id, audit in self._audit_by_family.items()}


class GitHubIssueStreamBuilder:
    """Build a local sanitized issue dataset and committed manifest from GH Archive."""

    def __init__(
        self,
        config: GitHubIssueBuilderConfig,
        *,
        query_clickhouse: ClickHouseQuery | None = None,
        monotonic: Callable[[], float] = time.monotonic,
    ) -> None:
        self.config = config
        self._query_clickhouse = query_clickhouse or self._query_clickhouse_remote
        self._monotonic = monotonic

    def build(self) -> BuilderResult:
        """Fetch public issue events read-only, sanitize, split, and write local outputs."""
        if not self.config.enabled:
            raise ValueError("github issue stream builder is disabled by configuration")
        if self.config.source != "gh_archive_clickhouse":
            raise ValueError("github issue stream source must be gh_archive_clickhouse")

        started = self._monotonic()
        records: list[RawDatasetRecord] = []
        dropped = Counter[str]()
        query_summaries: dict[str, Any] = {}
        seen_issues: set[tuple[str, int]] = set()
        seen_fingerprints: dict[tuple[str, ...], list[set[str]]] = defaultdict(list)

        for repo_config in self.config.repositories:
            for window in self.config.windows:
                rows_payload = self._query_issue_rows(repo_config, window)
                edit_counts = self._query_label_edit_counts(repo_config, window)
                coverage_payload = self._query_source_coverage(repo_config, window)
                rows = _require_list(rows_payload, "data")
                query_summaries[f"{repo_config.name}|{window.window_id}"] = {
                    "rows_returned": len(rows),
                    "statistics": rows_payload.get("statistics", {}),
                    "source_coverage": _coverage_rows_by_month(coverage_payload),
                }
                for item in rows:
                    issue = _ensure_mapping(item, "clickhouse issue row")
                    record, drop_reason = self._issue_to_record(
                        repo_config, window, issue, edit_counts
                    )
                    if drop_reason is not None:
                        dropped[drop_reason] += 1
                        continue
                    if record is None:
                        dropped["no_record"] += 1
                        continue
                    issue_key = (
                        record["metadata"]["repo"],
                        int(record["metadata"]["number"]),
                    )
                    if issue_key in seen_issues:
                        dropped["duplicate_issue"] += 1
                        continue
                    fingerprint = token_fingerprint(record["prompt"])
                    if self._is_near_duplicate(fingerprint, seen_fingerprints):
                        dropped["near_duplicate"] += 1
                        continue
                    seen_issues.add(issue_key)
                    seen_fingerprints[_fingerprint_bucket(fingerprint)].append(fingerprint)
                    records.append(record)

        records = self._assign_chronological_splits(records)
        validate_window_coverage(
            records,
            self.config.windows,
            min_months_represented=self.config.coverage.min_months_represented,
        )
        manifest, audit = self._build_manifest_and_audit(
            records, dropped, query_summaries, started
        )
        self._write_outputs(records, manifest, audit)
        disk_usage = _directory_size(self.config.output_dir)
        elapsed = self._monotonic() - started
        build_metrics = {
            "elapsed_seconds": round(elapsed, 3),
            "disk_usage_bytes": disk_usage,
            "raw_dataset_bytes": self.config.raw_dataset_path.stat().st_size,
            "manifest_bytes": self.config.manifest_path.stat().st_size,
            "audit_bytes": self.config.audit_path.stat().st_size,
        }
        manifest["build_metrics"] = build_metrics
        audit["_build_metrics"] = build_metrics
        manifest["audit"] = audit
        _write_json_atomic(self.config.manifest_path, manifest)
        _write_json_atomic(self.config.audit_path, audit)
        build_metrics = {
            **build_metrics,
            "manifest_bytes": self.config.manifest_path.stat().st_size,
            "audit_bytes": self.config.audit_path.stat().st_size,
        }
        manifest["build_metrics"] = build_metrics
        audit["_build_metrics"] = build_metrics
        manifest["audit"] = audit
        _write_json_atomic(self.config.manifest_path, manifest)
        _write_json_atomic(self.config.audit_path, audit)
        return BuilderResult(
            raw_dataset_path=self.config.raw_dataset_path,
            manifest_path=self.config.manifest_path,
            audit_path=self.config.audit_path,
            accepted_records=len(records),
            dropped_records=sum(dropped.values()),
            elapsed_seconds=elapsed,
            disk_usage_bytes=disk_usage,
        )

    def _query_issue_rows(
        self, repo_config: GitHubIssueRepositoryConfig, window: GitHubIssueWindow
    ) -> JsonObject:
        labels_sql = _clickhouse_string_array(self._source_query_labels(repo_config))
        sampling_seed = _clickhouse_quote(str(self.config.sampling.seed))
        month_quota = max(
            1,
            math.ceil(
                self.config.max_records_per_window
                / max(1, calendar_month_count(window.start, window.end))
            ),
        )
        if self.config.sampling.method != "month_stratified_hash":
            raise ValueError(
                "github issue stream sampling method must be month_stratified_hash"
            )
        query = f"""
SELECT
    repo_name,
    number,
    title,
    substring(body, 1, {self.config.max_body_chars}) AS body,
    labels,
    created_at AS event_created_at,
    updated_at AS issue_updated_at,
    closed_at AS issue_closed_at
FROM
(
    SELECT
        repo_name,
        number,
        title,
        body,
        labels,
        created_at,
        updated_at,
        closed_at,
        row_number() OVER (
            PARTITION BY toYYYYMM(created_at)
            ORDER BY
                cityHash64(concat(toString(number), ':', {sampling_seed})),
                number
        ) AS month_rank
    FROM github_events
    WHERE event_type = 'IssuesEvent'
      AND action = 'closed'
      AND repo_name = {_clickhouse_quote(repo_config.name)}
      AND created_at >= {_clickhouse_quote(window.start)}
      AND created_at < {_clickhouse_quote(window.end)}
      AND length(body) >= {self.config.min_body_chars}
      AND length(arrayFilter(label -> has({labels_sql}, label), labels)) >= 1
)
WHERE month_rank <= {month_quota}
ORDER BY toYYYYMM(event_created_at) ASC, month_rank ASC, number ASC
LIMIT {self.config.max_records_per_window}
FORMAT JSON
"""
        return self._query_clickhouse(query)

    def _query_source_coverage(
        self, repo_config: GitHubIssueRepositoryConfig, window: GitHubIssueWindow
    ) -> JsonObject:
        labels_sql = _clickhouse_string_array(self._source_query_labels(repo_config))
        query = f"""
SELECT
    substring(toString(toStartOfMonth(created_at)), 1, 7) AS month,
    count() AS eligible_rows
FROM github_events
WHERE event_type = 'IssuesEvent'
  AND action = 'closed'
  AND repo_name = {_clickhouse_quote(repo_config.name)}
  AND created_at >= {_clickhouse_quote(window.start)}
  AND created_at < {_clickhouse_quote(window.end)}
  AND length(body) >= {self.config.min_body_chars}
  AND length(arrayFilter(label -> has({labels_sql}, label), labels)) >= 1
GROUP BY month
ORDER BY month
FORMAT JSON
"""
        return self._query_clickhouse(query)

    def _query_label_edit_counts(
        self, repo_config: GitHubIssueRepositoryConfig, window: GitHubIssueWindow
    ) -> dict[int, int]:
        query = f"""
SELECT number, count() AS edit_events
FROM github_events
WHERE event_type = 'IssuesEvent'
  AND action IN ('labeled', 'unlabeled')
  AND repo_name = {_clickhouse_quote(repo_config.name)}
  AND created_at >= {_clickhouse_quote(window.start)}
  AND created_at < {_clickhouse_quote(window.end)}
GROUP BY number
FORMAT JSON
"""
        payload = self._query_clickhouse(query)
        counts: dict[int, int] = {}
        for row in _require_list(payload, "data"):
            mapping = _ensure_mapping(row, "label edit row")
            counts[int(mapping["number"])] = int(mapping["edit_events"])
        return counts

    def _source_query_labels(
        self, repo_config: GitHubIssueRepositoryConfig
    ) -> tuple[str, ...]:
        return tuple(
            sorted(set(repo_config.target_labels) | set(self.config.canonical_label_aliases))
        )

    def _query_clickhouse_remote(self, query: str) -> JsonObject:
        encoded_user = urllib.parse.urlencode({"user": self.config.clickhouse_user})
        separator = "&" if "?" in self.config.clickhouse_endpoint else "?"
        url = f"{self.config.clickhouse_endpoint}{separator}{encoded_user}&default_format=JSON"
        request = urllib.request.Request(
            url,
            data=query.encode("utf-8"),
            method="POST",
            headers={"User-Agent": self.config.user_agent},
        )
        try:
            with urllib.request.urlopen(
                request, timeout=self.config.query_timeout_seconds
            ) as response:
                payload = json.load(response)
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"ClickHouse query failed: {detail}") from exc
        if not isinstance(payload, dict):
            raise ValueError("ClickHouse response must be a JSON object")
        return payload

    def _issue_to_record(
        self,
        repo_config: GitHubIssueRepositoryConfig,
        window: GitHubIssueWindow,
        issue: Mapping[str, Any],
        label_edit_counts: Mapping[int, int],
    ) -> tuple[RawDatasetRecord | None, str | None]:
        event_created_at = _require_str(issue, "event_created_at")
        if not window.contains(event_created_at):
            return None, "outside_window"
        title = _require_str(issue, "title")
        body = str(issue.get("body") or "")
        if len(body.strip()) < self.config.min_body_chars:
            return None, "body_too_short"
        label_names = [_ensure_str(label, "label") for label in _require_list(issue, "labels")]
        raw_target_labels = sorted(
            label for label in label_names if label in self._source_query_labels(repo_config)
        )
        canonical_labels = sorted(
            {
                self.config.canonical_label_aliases.get(label, label)
                for label in raw_target_labels
            }
        )
        if not canonical_labels:
            return None, "no_canonical_label"
        if len(canonical_labels) > 1:
            return None, "conflicting_target_labels"
        title_sanitized = sanitize_issue_text(title)
        body_sanitized = sanitize_issue_text(body)
        drop_reason = title_sanitized.drop_reason or body_sanitized.drop_reason
        if drop_reason is not None:
            return None, drop_reason
        target = canonical_label_string(canonical_labels)
        repo_name = _require_str(issue, "repo_name")
        number = int(issue["number"])
        prompt = issue_prompt(
            repo=repo_name,
            closed_at=event_created_at,
            title=title_sanitized.text,
            body=body_sanitized.text,
        )
        record_id = content_hash(f"{repo_name}#{number}#{event_created_at}", target)[:24]
        metadata: ManifestRecord = {
            "record_id": record_id,
            "repo": repo_name,
            "number": number,
            "html_url": f"https://github.com/{repo_name}/issues/{number}",
            "event_created_at": event_created_at,
            "issue_updated_at": _require_str(issue, "issue_updated_at"),
            "issue_closed_at": _require_str(issue, "issue_closed_at"),
            "window_id": window.window_id,
            "split": "unassigned",
            "source_labels": label_names,
            "raw_target_labels": raw_target_labels,
            "canonical_labels": canonical_labels,
            "target": target,
            "content_hash": content_hash(prompt, target),
            "label_edit_event_count": int(label_edit_counts.get(number, 0)),
        }
        return {
            "record_id": record_id,
            "prompt": prompt,
            "target_solution": target,
            "metadata": metadata,
        }, None

    def _is_near_duplicate(
        self, fingerprint: set[str], prior: Mapping[tuple[str, ...], Sequence[set[str]]]
    ) -> bool:
        if not fingerprint:
            return False
        for other in prior.get(_fingerprint_bucket(fingerprint), ()):
            union_size = len(fingerprint | other)
            if union_size == 0:
                continue
            if len(fingerprint & other) / union_size >= self.config.near_duplicate_jaccard:
                return True
        return False

    def _assign_chronological_splits(
        self, records: Sequence[RawDatasetRecord]
    ) -> list[RawDatasetRecord]:
        by_window: dict[str, list[RawDatasetRecord]] = defaultdict(list)
        for record in records:
            by_window[record["metadata"]["window_id"]].append(record)

        assigned: list[RawDatasetRecord] = []
        for window_id in sorted(by_window):
            window_records = sorted(
                by_window[window_id],
                key=lambda record: (
                    record["metadata"]["event_created_at"],
                    record["metadata"]["repo"],
                    record["metadata"]["number"],
                ),
            )
            train_count, val_count = split_counts(
                len(window_records),
                self.config.split.train_fraction,
                self.config.split.val_fraction,
            )
            for idx, record in enumerate(window_records):
                split = "train"
                if idx >= train_count + val_count:
                    split = "test"
                elif idx >= train_count:
                    split = "val"
                metadata = dict(record["metadata"])
                metadata["split"] = split
                assigned.append(
                    {
                        "record_id": record["record_id"],
                        "prompt": record["prompt"],
                        "target_solution": record["target_solution"],
                        "metadata": _coerce_manifest_record(metadata),
                    }
                )
        return assigned

    def _build_manifest_and_audit(
        self,
        records: Sequence[RawDatasetRecord],
        dropped: Counter[str],
        query_summaries: Mapping[str, Any],
        started: float,
    ) -> tuple[JsonObject, dict[str, Any]]:
        manifest_records = [record["metadata"] for record in records]
        config_hash = hashlib.sha256(
            self.config.manifest_path.as_posix().encode("utf-8")
            + json.dumps(query_summaries, sort_keys=True).encode("utf-8")
        ).hexdigest()
        manifest: JsonObject = {
            "schema_version": 2,
            "source": self.config.source,
            "built_at": datetime.now(tz=UTC).isoformat(),
            "config_hash": config_hash,
            "seed": self.config.drift_resampling_seed,
            "sampling": {
                "method": self.config.sampling.method,
                "seed": self.config.sampling.seed,
                "max_records_per_window": self.config.max_records_per_window,
            },
            "coverage_gate": {
                "min_months_represented": self.config.coverage.min_months_represented
            },
            "canonical_label_aliases": dict(
                sorted(self.config.canonical_label_aliases.items())
            ),
            "rename_detection": {
                "min_label_records": self.config.rename_detection.min_label_records,
                "max_handoff_days": self.config.rename_detection.max_handoff_days,
                "min_centroid_similarity": (
                    self.config.rename_detection.min_centroid_similarity
                ),
            },
            "repositories": [
                {
                    "name": repo.name,
                    "license_spdx": repo.license_spdx,
                    "target_labels": list(repo.target_labels),
                }
                for repo in self.config.repositories
            ],
            "windows": [
                {"id": window.window_id, "start": window.start, "end": window.end}
                for window in self.config.windows
            ],
            "privacy_safety": {
                "email_addresses": "masked as [EMAIL]",
                "generic_secret_assignments": "masked as [SECRET]",
                "high_confidence_secrets": "dropped",
                "raw_text": "written only to the git-ignored raw_dataset_path",
                "dropped_counts": dict(sorted(dropped.items())),
            },
            "source_limitations": {
                "label_state": "labels captured from the GH Archive IssuesEvent close payload",
                "label_edits": (
                    "labeled/unlabeled IssuesEvent counts are queried when the mirror exposes "
                    "them; selected microsoft/vscode rows in this mirror expose opened, closed, "
                    "and reopened actions, so label-edit counts are reported as unavailable when "
                    "no edit events are observed"
                ),
                "deleted_issues": (
                    "GitHub issue deletion or later moderation state is not exposed by the "
                    "ClickHouse GH Archive mirror used here"
                ),
            },
            "query_summaries": dict(query_summaries),
            "records": manifest_records,
            "raw_dataset_path": self.config.raw_dataset_path.as_posix(),
            "elapsed_seconds_at_manifest": round(self._monotonic() - started, 3),
        }
        audit = build_audit(
            records,
            query_summaries,
            windows=self.config.windows,
            coverage_min_months_represented=(
                self.config.coverage.min_months_represented
            ),
            drift_resampling_seed=self.config.drift_resampling_seed,
            rename_detection=self.config.rename_detection,
        )
        audit["_dropped_counts"] = dict(sorted(dropped.items()))
        manifest["audit"] = audit
        return manifest, audit

    def _write_outputs(
        self,
        records: Sequence[RawDatasetRecord],
        manifest: Mapping[str, Any],
        audit: Mapping[str, Any],
    ) -> None:
        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        self.config.manifest_path.parent.mkdir(parents=True, exist_ok=True)
        self.config.audit_path.parent.mkdir(parents=True, exist_ok=True)
        _write_jsonl_atomic(self.config.raw_dataset_path, records)
        _write_json_atomic(self.config.manifest_path, manifest)
        _write_json_atomic(self.config.audit_path, audit)


def load_github_issue_builder_config(path: Path | str) -> GitHubIssueBuilderConfig:
    """Load and validate the GitHub issue stream builder YAML."""
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    root = _ensure_mapping(raw, "root")
    section = _require_mapping(root, "github_issues_stream")
    split = _require_mapping(section, "split")
    sampling = _require_mapping(section, "sampling")
    coverage = _require_mapping(section, "coverage")
    rename_detection = _require_mapping(section, "rename_detection")
    repositories_raw = _require_list(section, "repositories")
    windows_raw = _require_list(section, "windows")
    canonical_label_aliases = _parse_label_aliases(
        _require_mapping(section, "canonical_label_aliases")
    )
    repositories = tuple(_parse_repo(item) for item in repositories_raw)
    _validate_label_aliases(canonical_label_aliases, repositories)
    return GitHubIssueBuilderConfig(
        enabled=_require_bool(section, "enabled"),
        source=_require_str(section, "source"),
        output_dir=Path(_require_str(section, "output_dir")),
        raw_dataset_path=Path(_require_str(section, "raw_dataset_path")),
        manifest_path=Path(_require_str(section, "manifest_path")),
        audit_path=Path(_require_str(section, "audit_path")),
        clickhouse_endpoint=_require_str(section, "clickhouse_endpoint"),
        clickhouse_user=_require_str(section, "clickhouse_user"),
        user_agent=_require_str(section, "user_agent"),
        query_timeout_seconds=_bounded_float(
            section, "query_timeout_seconds", minimum=1.0, maximum=600.0
        ),
        max_records_per_window=_bounded_int(
            section, "max_records_per_window", minimum=300, maximum=100_000
        ),
        min_body_chars=_bounded_int(section, "min_body_chars", minimum=0, maximum=10_000),
        max_body_chars=_bounded_int(section, "max_body_chars", minimum=200, maximum=20_000),
        near_duplicate_jaccard=_bounded_float(
            section, "near_duplicate_jaccard", minimum=0.5, maximum=1.0
        ),
        drift_resampling_seed=_bounded_int(
            section, "drift_resampling_seed", minimum=0, maximum=2**31 - 1
        ),
        canonical_label_aliases=canonical_label_aliases,
        sampling=GitHubIssueSamplingConfig(
            method=_require_choice(
                sampling,
                "method",
                choices=("month_stratified_hash",),
            ),
            seed=_bounded_int(sampling, "seed", minimum=0, maximum=2**31 - 1),
        ),
        coverage=GitHubIssueCoverageConfig(
            min_months_represented=_bounded_int(
                coverage, "min_months_represented", minimum=1, maximum=12
            )
        ),
        rename_detection=GitHubIssueRenameDetectionConfig(
            min_label_records=_bounded_int(
                rename_detection, "min_label_records", minimum=2, maximum=100_000
            ),
            max_handoff_days=_bounded_int(
                rename_detection, "max_handoff_days", minimum=0, maximum=3650
            ),
            min_centroid_similarity=_bounded_float(
                rename_detection,
                "min_centroid_similarity",
                minimum=0.0,
                maximum=1.0,
            ),
        ),
        windows=tuple(_parse_window(item) for item in windows_raw),
        split=GitHubIssueSplitConfig(
            train_fraction=_bounded_float(
                split, "train_fraction", minimum=0.1, maximum=0.9
            ),
            val_fraction=_bounded_float(split, "val_fraction", minimum=0.0, maximum=0.5),
        ),
        repositories=repositories,
    )


def sanitize_issue_text(text: str) -> SanitizedText:
    """Mask low-risk identifiers and reject obvious secrets."""
    for pattern in HIGH_CONFIDENCE_SECRET_PATTERNS:
        if pattern.search(text):
            return SanitizedText(
                text="", masked_email_count=0, masked_secret_count=0, drop_reason="secret"
            )
    text, email_count = EMAIL_RE.subn("[EMAIL]", text)
    masked_secret_count = 0
    for pattern in MASKED_SECRET_PATTERNS:
        text, count = pattern.subn("[SECRET]", text)
        masked_secret_count += count
    return SanitizedText(
        text=text.strip(),
        masked_email_count=email_count,
        masked_secret_count=masked_secret_count,
        drop_reason=None,
    )


def canonical_label_string(labels: Sequence[str]) -> str:
    """Return the verifier target string for canonical labels."""
    if not labels:
        raise ValueError("at least one canonical label is required")
    return "+".join(sorted(labels))


def issue_prompt(repo: str, closed_at: str, title: str, body: str) -> str:
    """Format the supervised issue component-triage prompt."""
    return (
        f"Repository: {repo}\n"
        f"Closed: {closed_at[:10]}\n"
        "Issue title:\n"
        f"{title.strip()}\n\n"
        "Issue body:\n"
        f"{body.strip()}\n\n"
        "Component label:"
    )


def content_hash(prompt: str, target_solution: str) -> str:
    """Hash local sanitized content for manifest verification without exposing text."""
    return hashlib.sha256(f"{prompt}\n{target_solution}".encode()).hexdigest()


def token_fingerprint(text: str) -> set[str]:
    """Token set used for lightweight near-duplicate detection."""
    return {match.group(0).lower() for match in TOKEN_RE.finditer(text)}


def _fingerprint_bucket(fingerprint: set[str]) -> tuple[str, ...]:
    """Return a stable high-threshold near-duplicate bucket key."""
    return tuple(sorted(fingerprint)[:8])


def split_counts(total: int, train_fraction: float, val_fraction: float) -> tuple[int, int]:
    """Return chronological train/val counts leaving the remainder for test."""
    if total <= 0:
        return 0, 0
    if total == 1:
        return 1, 0
    if total == 2:
        return 1, 0
    train_count = max(1, int(math.floor(total * train_fraction)))
    val_count = int(math.floor(total * val_fraction))
    if val_fraction > 0 and val_count == 0:
        val_count = 1
    if train_count + val_count >= total:
        val_count = max(0, total - train_count - 1)
    if train_count + val_count >= total:
        train_count = total - val_count - 1
    return train_count, val_count


def calendar_month_count(start: str, end: str) -> int:
    """Return the number of calendar months touched by a half-open date window."""
    start_date = _parse_iso_date(start)
    end_date = _parse_iso_date(end)
    if start_date >= end_date:
        raise ValueError("window start must be before end")
    month_count = (end_date.year - start_date.year) * 12 + end_date.month - start_date.month
    if end_date.day > 1:
        month_count += 1
    return max(1, month_count)


def events_per_month(records: Sequence[RawDatasetRecord]) -> dict[str, int]:
    """Return accepted record counts keyed by YYYY-MM event month."""
    counts = Counter[str]()
    for record in records:
        counts[record["metadata"]["event_created_at"][:7]] += 1
    return dict(sorted(counts.items()))


def validate_window_coverage(
    records: Sequence[RawDatasetRecord],
    windows: Sequence[GitHubIssueWindow],
    *,
    min_months_represented: int,
) -> None:
    """Fail when accepted records do not cover enough months in each window."""
    by_window: dict[str, list[RawDatasetRecord]] = defaultdict(list)
    for record in records:
        by_window[record["metadata"]["window_id"]].append(record)

    failures: list[str] = []
    for window in windows:
        represented = len(events_per_month(by_window.get(window.window_id, [])))
        required = min(
            min_months_represented,
            calendar_month_count(window.start, window.end),
        )
        if represented < required:
            failures.append(f"{window.window_id}: {represented}/{required} months")
    if failures:
        raise ValueError(
            "GitHub issue stream coverage gate failed: " + "; ".join(failures)
        )


def build_audit(
    records: Sequence[RawDatasetRecord],
    query_summaries: Mapping[str, Any],
    *,
    windows: Sequence[GitHubIssueWindow] = (),
    coverage_min_months_represented: int = 10,
    drift_resampling_seed: int = 0,
    rename_detection: GitHubIssueRenameDetectionConfig | None = None,
) -> dict[str, dict[str, Any]]:
    """Compute per-window audit, drift, and baseline statistics."""
    by_family: dict[str, list[RawDatasetRecord]] = defaultdict(list)
    for record in records:
        by_family[_family_id(record["metadata"]["window_id"])].append(record)

    classifier_baselines = tfidf_centroid_baselines(by_family)
    windows_by_family = {_family_id(window.window_id): window for window in windows}
    audits: dict[str, dict[str, Any]] = {}
    previous_family_id: str | None = None
    previous_records: list[RawDatasetRecord] | None = None
    for family_id in sorted(by_family):
        family_records = by_family[family_id]
        targets = [record["target_solution"] for record in family_records]
        target_counts = Counter(targets)
        raw_target_counts = Counter(_raw_target_string(record) for record in family_records)
        split_counts_by_name = Counter(record["metadata"]["split"] for record in family_records)
        train_targets = [
            record["target_solution"]
            for record in family_records
            if record["metadata"]["split"] == "train"
        ]
        test_records = [
            record for record in family_records if record["metadata"]["split"] == "test"
        ]
        majority_target = Counter(train_targets).most_common(1)[0][0] if train_targets else ""
        label_edit_event_count = sum(
            record["metadata"]["label_edit_event_count"] for record in family_records
        )
        majority_accuracy = _baseline_accuracy(
            test_records, constant_baseline_prediction(majority_target)
        )
        keyword_accuracy = _baseline_accuracy(test_records, keyword_baseline_prediction)
        drift = None
        raw_drift = None
        if previous_records is not None and previous_family_id is not None:
            drift = drift_against_resampling_baseline(
                previous_records,
                family_records,
                seed=drift_resampling_seed,
            )
            drift["previous_family_id"] = previous_family_id
            drift["current_family_id"] = family_id
            raw_drift = drift_against_resampling_baseline(
                previous_records,
                family_records,
                seed=drift_resampling_seed,
                label_getter=_raw_target_string,
            )
            raw_drift["previous_family_id"] = previous_family_id
            raw_drift["current_family_id"] = family_id
        coverage = _accepted_coverage(
            family_records,
            windows_by_family.get(family_id),
            coverage_min_months_represented,
            query_summaries,
            family_id,
        )
        audits[family_id] = {
            "records": len(family_records),
            "split_counts": dict(sorted(split_counts_by_name.items())),
            "distinct_targets": len(target_counts),
            "chance_level": 1.0 / len(target_counts) if target_counts else 0.0,
            "target_counts": dict(sorted(target_counts.items())),
            "raw_target_counts": dict(sorted(raw_target_counts.items())),
            "label_aliases": _alias_summary(family_records),
            "coverage": coverage,
            "overlap": {
                "train_val": _split_overlap(family_records, "train", "val"),
                "train_test": _split_overlap(family_records, "train", "test"),
                "val_test": _split_overlap(family_records, "val", "test"),
            },
            "majority_class_baseline": {
                "target": majority_target,
                "test_accuracy": majority_accuracy,
            },
            "keyword_baseline": {"test_accuracy": keyword_accuracy},
            "tfidf_centroid_baseline": classifier_baselines.get(family_id, {}),
            "drift_from_previous_window": drift,
            "raw_label_drift_from_previous_window": raw_drift,
            "label_noise": {
                "accepted_records_with_label_edit_events": sum(
                    1
                    for record in family_records
                    if record["metadata"]["label_edit_event_count"] > 0
                ),
                "label_edit_event_count": label_edit_event_count,
                "label_edit_events_observable": label_edit_event_count > 0,
                "resolution": (
                    "records with more than one configured component label at close "
                    "are dropped as conflicting_target_labels"
                ),
            },
            "query_rows_returned": _query_rows_for_family(query_summaries, family_id),
        }
        previous_family_id = family_id
        previous_records = family_records
    detection_config = rename_detection or GitHubIssueRenameDetectionConfig(
        min_label_records=20,
        max_handoff_days=455,
        min_centroid_similarity=0.25,
    )
    audits["_label_rename_candidates"] = {
        "method": (
            "raw label pairs whose usage handoff is close in time and whose TF-IDF "
            "prompt centroids are similar"
        ),
        "parameters": {
            "min_label_records": detection_config.min_label_records,
            "max_handoff_days": detection_config.max_handoff_days,
            "min_centroid_similarity": detection_config.min_centroid_similarity,
        },
        "candidates": detect_label_rename_candidates(
            records,
            min_label_records=detection_config.min_label_records,
            max_handoff_days=detection_config.max_handoff_days,
            min_centroid_similarity=detection_config.min_centroid_similarity,
        ),
    }
    return audits


def drift_against_resampling_baseline(
    previous_records: Sequence[RawDatasetRecord],
    current_records: Sequence[RawDatasetRecord],
    *,
    seed: int,
    label_getter: Callable[[RawDatasetRecord], str] | None = None,
) -> dict[str, Any]:
    """Return observed adjacent-window drift and same-window resampling baselines."""
    get_label = label_getter or _target_solution
    previous_label_dist = _record_label_distribution(previous_records, get_label)
    current_label_dist = _record_label_distribution(current_records, get_label)
    previous_vocab = _record_vocab(previous_records)
    current_vocab = _record_vocab(current_records)
    previous_left, previous_right = _deterministic_halves(previous_records, seed)
    current_left, current_right = _deterministic_halves(current_records, seed + 1)
    return {
        "label_js_divergence": jensen_shannon_divergence(
            previous_label_dist, current_label_dist
        ),
        "label_js_previous_window_resampling_baseline": jensen_shannon_divergence(
            _record_label_distribution(previous_left, get_label),
            _record_label_distribution(previous_right, get_label),
        ),
        "label_js_current_window_resampling_baseline": jensen_shannon_divergence(
            _record_label_distribution(current_left, get_label),
            _record_label_distribution(current_right, get_label),
        ),
        "vocabulary_jaccard_distance": jaccard_distance(previous_vocab, current_vocab),
        "vocabulary_jaccard_previous_window_resampling_baseline": jaccard_distance(
            _record_vocab(previous_left), _record_vocab(previous_right)
        ),
        "vocabulary_jaccard_current_window_resampling_baseline": jaccard_distance(
            _record_vocab(current_left), _record_vocab(current_right)
        ),
    }


def tfidf_centroid_baselines(
    by_family: Mapping[str, Sequence[RawDatasetRecord]],
) -> dict[str, dict[str, float | int | None]]:
    """Train deterministic sparse TF-IDF centroid classifiers for each window."""
    results: dict[str, dict[str, float | int | None]] = {}
    earlier_train_records: list[RawDatasetRecord] = []
    for family_id in sorted(by_family):
        family_records = list(by_family[family_id])
        train_records = [
            record for record in family_records if record["metadata"]["split"] == "train"
        ]
        test_records = [
            record for record in family_records if record["metadata"]["split"] == "test"
        ]
        current_model = _TfidfCentroidClassifier.fit(train_records)
        current_accuracy = _baseline_accuracy(test_records, current_model.predict)
        previous_accuracy = None
        if earlier_train_records:
            previous_model = _TfidfCentroidClassifier.fit(earlier_train_records)
            previous_accuracy = _baseline_accuracy(test_records, previous_model.predict)
        results[family_id] = {
            "current_window_train_accuracy_on_test": current_accuracy,
            "earlier_windows_train_accuracy_on_test": previous_accuracy,
            "current_window_train_records": len(train_records),
            "earlier_windows_train_records": len(earlier_train_records),
        }
        earlier_train_records.extend(train_records)
    return results


class _TfidfCentroidClassifier:
    """Small deterministic non-neural text classifier for audit baselines."""

    def __init__(
        self,
        idf: Mapping[str, float],
        centroids: Mapping[str, Mapping[str, float]],
        fallback_label: str,
    ) -> None:
        self._idf = dict(idf)
        self._centroids = {label: dict(values) for label, values in centroids.items()}
        self._fallback_label = fallback_label

    @classmethod
    def fit(cls, records: Sequence[RawDatasetRecord]) -> _TfidfCentroidClassifier:
        if not records:
            return cls({}, {}, "")
        document_tokens = [token_fingerprint(record["prompt"]) for record in records]
        document_frequency = Counter[str]()
        for tokens in document_tokens:
            document_frequency.update(tokens)
        document_count = len(records)
        idf = {
            token: math.log((1 + document_count) / (1 + count)) + 1.0
            for token, count in document_frequency.items()
        }
        by_label: dict[str, list[dict[str, float]]] = defaultdict(list)
        for record in records:
            by_label[record["target_solution"]].append(_tfidf_vector(record["prompt"], idf))
        centroids = {
            label: _mean_vector(vectors) for label, vectors in by_label.items() if vectors
        }
        fallback_label = Counter(
            record["target_solution"] for record in records
        ).most_common(1)[0][0]
        return cls(idf, centroids, fallback_label)

    def predict(self, record: RawDatasetRecord) -> str:
        if not self._centroids:
            return self._fallback_label
        vector = _tfidf_vector(record["prompt"], self._idf)
        if not vector:
            return self._fallback_label
        best_label = self._fallback_label
        best_score = -1.0
        for label, centroid in self._centroids.items():
            score = _dot(vector, centroid)
            if score > best_score or (score == best_score and label < best_label):
                best_label = label
                best_score = score
        return best_label


def detect_label_rename_candidates(
    records: Sequence[RawDatasetRecord],
    *,
    min_label_records: int,
    max_handoff_days: int,
    min_centroid_similarity: float,
) -> list[dict[str, Any]]:
    """Detect raw label pairs with temporal handoff and similar issue text."""
    by_label: dict[str, list[RawDatasetRecord]] = defaultdict(list)
    for record in records:
        raw_label = _raw_target_string(record)
        if raw_label:
            by_label[raw_label].append(record)

    eligible = {
        label: label_records
        for label, label_records in by_label.items()
        if len(label_records) >= min_label_records
    }
    if len(eligible) < 2:
        return []

    document_frequency = Counter[str]()
    for record in records:
        document_frequency.update(token_fingerprint(record["prompt"]))
    idf = {
        token: math.log((1 + len(records)) / (1 + count)) + 1.0
        for token, count in document_frequency.items()
    }
    centroids = {
        label: _mean_vector(
            [_tfidf_vector(record["prompt"], idf) for record in label_records]
        )
        for label, label_records in eligible.items()
    }
    spans = {label: _label_span(label_records) for label, label_records in eligible.items()}
    candidates: list[dict[str, Any]] = []
    labels = sorted(eligible)
    for left_index, left_label in enumerate(labels):
        for right_label in labels[left_index + 1 :]:
            left_span = spans[left_label]
            right_span = spans[right_label]
            old_label, new_label = _orient_handoff(
                left_label, left_span, right_label, right_span
            )
            old_span = spans[old_label]
            new_span = spans[new_label]
            handoff_days = max(0, (new_span["first"] - old_span["last"]).days)
            if handoff_days > max_handoff_days:
                continue
            similarity = _dot(centroids[old_label], centroids[new_label])
            if similarity < min_centroid_similarity:
                continue
            candidates.append(
                {
                    "old_label": old_label,
                    "new_label": new_label,
                    "old_count": len(eligible[old_label]),
                    "new_count": len(eligible[new_label]),
                    "old_first_event": old_span["first"].isoformat(),
                    "old_last_event": old_span["last"].isoformat(),
                    "new_first_event": new_span["first"].isoformat(),
                    "new_last_event": new_span["last"].isoformat(),
                    "handoff_gap_days": handoff_days,
                    "overlap_days": _overlap_days(old_span, new_span),
                    "centroid_cosine_similarity": round(similarity, 6),
                }
            )
    return sorted(
        candidates,
        key=lambda item: (
            -float(item["centroid_cosine_similarity"]),
            int(item["handoff_gap_days"]),
            str(item["old_label"]),
            str(item["new_label"]),
        ),
    )


def keyword_baseline_prediction(record: RawDatasetRecord) -> str:
    """Deterministic keyword baseline for common VS Code component labels."""
    text = record["prompt"].lower()
    scores = {
        "accessibility": text.count("accessib") + text.count("screen reader"),
        "api": text.count("api") + text.count("extension host"),
        "debug": text.count("debug") + text.count("breakpoint"),
        "extensions": text.count("extension") + text.count("marketplace"),
        "file-explorer": text.count("explorer") + text.count("file tree"),
        "git": text.count("git") + text.count("repository"),
        "integrated-terminal": text.count("integrated terminal"),
        "javascript": text.count("javascript") + text.count("js "),
        "notebook": text.count("notebook") + text.count("jupyter"),
        "search": text.count("search") + text.count("replace"),
        "settings-sync": text.count("sync") + text.count("settings"),
        "tasks": text.count("task") + text.count("build task"),
        "terminal": text.count("terminal") + text.count("shell"),
        "terminal-shell-integration": text.count("shell integration"),
        "testing": text.count("test") + text.count("testing"),
        "typescript": text.count("typescript") + text.count("tsserver"),
        "workspace-trust": text.count("workspace trust") + text.count("trusted workspace"),
    }
    best_label, best_score = max(scores.items(), key=lambda item: (item[1], item[0]))
    if best_score <= 0:
        return "debug"
    return best_label


def constant_baseline_prediction(target: str) -> Callable[[RawDatasetRecord], str]:
    """Return a typed constant predictor for baseline scoring."""

    def predict(_record: RawDatasetRecord) -> str:
        return target

    return predict


def jaccard_distance(left: set[str], right: set[str]) -> float:
    """Return set Jaccard distance."""
    union = left | right
    if not union:
        return 0.0
    return 1.0 - (len(left & right) / len(union))


def jensen_shannon_divergence(
    left: Mapping[str, float],
    right: Mapping[str, float],
) -> float:
    """Return Jensen-Shannon divergence in bits for two sparse distributions."""
    labels = set(left) | set(right)
    midpoint = {label: (left.get(label, 0.0) + right.get(label, 0.0)) / 2.0 for label in labels}
    return (_kl_divergence(left, midpoint, labels) + _kl_divergence(right, midpoint, labels)) / 2.0


def _tfidf_vector(text: str, idf: Mapping[str, float]) -> dict[str, float]:
    counts = Counter(token_fingerprint(text))
    vector = {token: count * idf[token] for token, count in counts.items() if token in idf}
    norm = math.sqrt(sum(value * value for value in vector.values()))
    if norm == 0.0:
        return {}
    return {token: value / norm for token, value in vector.items()}


def _mean_vector(vectors: Sequence[Mapping[str, float]]) -> dict[str, float]:
    totals = Counter[str]()
    for vector in vectors:
        totals.update(vector)
    averaged = {token: value / len(vectors) for token, value in totals.items()}
    norm = math.sqrt(sum(value * value for value in averaged.values()))
    if norm == 0.0:
        return {}
    return {token: value / norm for token, value in averaged.items()}


def _dot(left: Mapping[str, float], right: Mapping[str, float]) -> float:
    if len(left) > len(right):
        left, right = right, left
    return sum(value * right.get(token, 0.0) for token, value in left.items())


def _kl_divergence(
    left: Mapping[str, float], right: Mapping[str, float], labels: Iterable[str]
) -> float:
    total = 0.0
    for label in labels:
        p_value = left.get(label, 0.0)
        q_value = right.get(label, 0.0)
        if p_value > 0.0 and q_value > 0.0:
            total += p_value * math.log2(p_value / q_value)
    return total


def _raw_target_string(record: RawDatasetRecord) -> str:
    raw_labels = record["metadata"].get("raw_target_labels", record["metadata"]["canonical_labels"])
    return canonical_label_string([_ensure_str(label, "raw target label") for label in raw_labels])


def _alias_summary(records: Sequence[RawDatasetRecord]) -> dict[str, Any]:
    alias_counts = Counter[str]()
    for record in records:
        raw = _raw_target_string(record)
        canonical = record["target_solution"]
        if raw != canonical:
            alias_counts[f"{raw}->{canonical}"] += 1
    return {
        "records_with_alias_applied": sum(alias_counts.values()),
        "alias_counts": dict(sorted(alias_counts.items())),
    }


def _accepted_coverage(
    records: Sequence[RawDatasetRecord],
    window: GitHubIssueWindow | None,
    min_months_represented: int,
    query_summaries: Mapping[str, Any],
    family_id: str,
) -> dict[str, Any]:
    per_month = events_per_month(records)
    first_event = min(
        (record["metadata"]["event_created_at"] for record in records),
        default=None,
    )
    last_event = max(
        (record["metadata"]["event_created_at"] for record in records),
        default=None,
    )
    expected_months = (
        calendar_month_count(window.start, window.end) if window is not None else len(per_month)
    )
    required_months = min(min_months_represented, expected_months)
    represented_months = len(per_month)
    return {
        "first_event": first_event,
        "last_event": last_event,
        "events_per_month": per_month,
        "months_represented": represented_months,
        "expected_months_in_window": expected_months,
        "min_months_required": required_months,
        "meets_threshold": represented_months >= required_months,
        "source_eligible_rows_per_month": _source_coverage_for_family(
            query_summaries, family_id
        ),
    }


def _coverage_rows_by_month(payload: Mapping[str, Any]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in _require_list(payload, "data"):
        mapping = _ensure_mapping(row, "coverage row")
        counts[_require_str(mapping, "month")] = int(mapping["eligible_rows"])
    return dict(sorted(counts.items()))


def _source_coverage_for_family(
    query_summaries: Mapping[str, Any], family_id: str
) -> dict[str, int]:
    window_id = family_id.removeprefix("github_issues_")
    counts = Counter[str]()
    for key, value in query_summaries.items():
        if not key.endswith(f"|{window_id}") or not isinstance(value, Mapping):
            continue
        source_coverage = value.get("source_coverage", {})
        if not isinstance(source_coverage, Mapping):
            continue
        for month, count in source_coverage.items():
            counts[str(month)] += int(count)
    return dict(sorted(counts.items()))


def _parse_iso_date(value: str) -> date:
    return date.fromisoformat(value[:10])


def _label_span(records: Sequence[RawDatasetRecord]) -> dict[str, date]:
    dates = [_parse_iso_date(record["metadata"]["event_created_at"]) for record in records]
    return {"first": min(dates), "last": max(dates)}


def _orient_handoff(
    left_label: str,
    left_span: Mapping[str, date],
    right_label: str,
    right_span: Mapping[str, date],
) -> tuple[str, str]:
    if left_span["first"] < right_span["first"]:
        return left_label, right_label
    if right_span["first"] < left_span["first"]:
        return right_label, left_label
    if left_span["last"] <= right_span["last"]:
        return left_label, right_label
    return right_label, left_label


def _overlap_days(
    left_span: Mapping[str, date], right_span: Mapping[str, date]
) -> int:
    latest_start = max(left_span["first"], right_span["first"])
    earliest_end = min(left_span["last"], right_span["last"])
    if earliest_end < latest_start:
        return 0
    return (earliest_end - latest_start).days + 1


def _target_solution(record: RawDatasetRecord) -> str:
    return record["target_solution"]


def _record_label_distribution(
    records: Sequence[RawDatasetRecord],
    label_getter: Callable[[RawDatasetRecord], str],
) -> dict[str, float]:
    return _distribution(Counter(label_getter(record) for record in records))


def _record_vocab(records: Sequence[RawDatasetRecord]) -> set[str]:
    vocab: set[str] = set()
    for record in records:
        vocab.update(token_fingerprint(record["prompt"]))
    return vocab


def _deterministic_halves(
    records: Sequence[RawDatasetRecord], seed: int
) -> tuple[list[RawDatasetRecord], list[RawDatasetRecord]]:
    keyed = sorted(
        records,
        key=lambda record: content_hash(f"{seed}:{record['record_id']}", record["target_solution"]),
    )
    midpoint = len(keyed) // 2
    return keyed[:midpoint], keyed[midpoint:]


def _distribution(counts: Counter[str]) -> dict[str, float]:
    total = sum(counts.values())
    if total == 0:
        return {}
    return {key: count / total for key, count in counts.items()}


def _baseline_accuracy(
    records: Sequence[RawDatasetRecord],
    predict: Callable[[RawDatasetRecord], str],
) -> float:
    if not records:
        return 0.0
    correct = sum(1 for record in records if predict(record) == record["target_solution"])
    return correct / len(records)


def _split_overlap(records: Sequence[RawDatasetRecord], left: str, right: str) -> int:
    left_hashes = {
        record["metadata"]["content_hash"]
        for record in records
        if record["metadata"]["split"] == left
    }
    right_hashes = {
        record["metadata"]["content_hash"]
        for record in records
        if record["metadata"]["split"] == right
    }
    return len(left_hashes & right_hashes)


def _query_rows_for_family(query_summaries: Mapping[str, Any], family_id: str) -> int:
    window_id = family_id.removeprefix("github_issues_")
    total = 0
    for key, value in query_summaries.items():
        if key.endswith(f"|{window_id}") and isinstance(value, Mapping):
            total += int(value.get("rows_returned", 0))
    return total


def _parse_window(value: object) -> GitHubIssueWindow:
    mapping = _ensure_mapping(value, "window")
    window = GitHubIssueWindow(
        window_id=_require_str(mapping, "id"),
        start=_require_str(mapping, "start"),
        end=_require_str(mapping, "end"),
    )
    if window.start >= window.end:
        raise ValueError(f"window {window.window_id} start must be before end")
    return window


def _parse_repo(value: object) -> GitHubIssueRepositoryConfig:
    mapping = _ensure_mapping(value, "repository")
    labels = tuple(
        _ensure_str(item, "target label")
        for item in _require_list(mapping, "target_labels")
    )
    if not labels:
        raise ValueError("repository target_labels must not be empty")
    if len(set(labels)) != len(labels):
        raise ValueError("repository target_labels must be unique")
    return GitHubIssueRepositoryConfig(
        name=_require_str(mapping, "name"),
        license_spdx=_require_str(mapping, "license_spdx"),
        target_labels=labels,
    )


def _parse_label_aliases(mapping: Mapping[str, Any]) -> dict[str, str]:
    aliases: dict[str, str] = {}
    for source, target in mapping.items():
        source_label = _ensure_str(source, "canonical label alias source")
        target_label = _ensure_str(target, "canonical label alias target")
        if source_label == target_label:
            raise ValueError("canonical label alias source and target must differ")
        aliases[source_label] = target_label
    return aliases


def _validate_label_aliases(
    aliases: Mapping[str, str],
    repositories: Sequence[GitHubIssueRepositoryConfig],
) -> None:
    configured_labels = set[str]()
    for repository in repositories:
        configured_labels.update(repository.target_labels)
    for source, target in aliases.items():
        if source not in configured_labels:
            raise ValueError(f"canonical label alias source is not configured: {source}")
        if target not in configured_labels:
            raise ValueError(f"canonical label alias target is not configured: {target}")


def _family_id(window_id: str) -> str:
    safe_window = re.sub(r"[^A-Za-z0-9_]+", "_", window_id)
    return f"github_issues_{safe_window}"


def _read_json_object(path: Path) -> JsonObject:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return data


def _read_jsonl_dataset(path: Path) -> list[RawDatasetRecord]:
    records: list[RawDatasetRecord] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            raw = json.loads(line)
            mapping = _ensure_mapping(raw, "dataset record")
            metadata = _ensure_mapping(mapping.get("metadata"), "metadata")
            records.append(
                {
                    "record_id": _require_str(mapping, "record_id"),
                    "prompt": _require_str(mapping, "prompt"),
                    "target_solution": _require_str(mapping, "target_solution"),
                    "metadata": _coerce_manifest_record(metadata),
                }
            )
    return records


def _manifest_records(manifest: Mapping[str, Any]) -> list[ManifestRecord]:
    return [
        _coerce_manifest_record(_ensure_mapping(item, "manifest record"))
        for item in _require_list(manifest, "records")
    ]


def _coerce_manifest_record(mapping: Mapping[str, Any]) -> ManifestRecord:
    return {
        "record_id": _require_str(mapping, "record_id"),
        "repo": _require_str(mapping, "repo"),
        "number": int(mapping["number"]),
        "html_url": _require_str(mapping, "html_url"),
        "event_created_at": _require_str(mapping, "event_created_at"),
        "issue_updated_at": _require_str(mapping, "issue_updated_at"),
        "issue_closed_at": _require_str(mapping, "issue_closed_at"),
        "window_id": _require_str(mapping, "window_id"),
        "split": _require_str(mapping, "split"),
        "source_labels": [_ensure_str(item, "source label") for item in mapping["source_labels"]],
        "raw_target_labels": [
            _ensure_str(item, "raw target label")
            for item in mapping.get("raw_target_labels", mapping["canonical_labels"])
        ],
        "canonical_labels": [
            _ensure_str(item, "canonical label") for item in mapping["canonical_labels"]
        ],
        "target": _require_str(mapping, "target"),
        "content_hash": _require_str(mapping, "content_hash"),
        "label_edit_event_count": int(mapping.get("label_edit_event_count", 0)),
    }


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    temp_path = path.with_suffix(f"{path.suffix}.tmp")
    temp_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temp_path.replace(path)


def _write_jsonl_atomic(path: Path, records: Sequence[RawDatasetRecord]) -> None:
    temp_path = path.with_suffix(f"{path.suffix}.tmp")
    with temp_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
    temp_path.replace(path)


def _directory_size(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def _clickhouse_quote(value: str) -> str:
    return "'" + value.replace("\\", "\\\\").replace("'", "\\'") + "'"


def _clickhouse_string_array(values: Sequence[str]) -> str:
    return "[" + ", ".join(_clickhouse_quote(value) for value in values) + "]"


def _ensure_mapping(value: object, name: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a mapping")
    return value


def _require_mapping(mapping: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    return _ensure_mapping(mapping.get(key), key)


def _require_list(mapping: Mapping[str, Any], key: str) -> list[Any]:
    value = mapping.get(key)
    if not isinstance(value, list):
        raise ValueError(f"{key} must be a list")
    return value


def _require_str(mapping: Mapping[str, Any], key: str) -> str:
    return _ensure_str(mapping.get(key), key)


def _ensure_str(value: object, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _require_bool(mapping: Mapping[str, Any], key: str) -> bool:
    value = mapping.get(key)
    if not isinstance(value, bool):
        raise ValueError(f"{key} must be a boolean")
    return value


def _require_choice(
    mapping: Mapping[str, Any], key: str, *, choices: Sequence[str]
) -> str:
    value = _require_str(mapping, key)
    if value not in choices:
        raise ValueError(f"{key} must be one of {', '.join(choices)}")
    return value


def _bounded_int(mapping: Mapping[str, Any], key: str, *, minimum: int, maximum: int) -> int:
    value = mapping.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{key} must be an integer")
    if not minimum <= value <= maximum:
        raise ValueError(f"{key} must be between {minimum} and {maximum}")
    return value


def _bounded_float(
    mapping: Mapping[str, Any], key: str, *, minimum: float, maximum: float
) -> float:
    value = mapping.get(key)
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"{key} must be numeric")
    numeric = float(value)
    if not minimum <= numeric <= maximum:
        raise ValueError(f"{key} must be between {minimum} and {maximum}")
    return numeric

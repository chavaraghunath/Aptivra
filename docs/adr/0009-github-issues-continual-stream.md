# ADR-0009: Public GitHub Issues Continual-Learning Stream

- **Status:** Proposed
- **Date:** 2026-10-14

## Context

Aptivra needs a real text stream that can support continual-learning claims. The earlier
REST-search stream had only 341 accepted examples and 16-20 test items per window. That is
underpowered: one test item moved accuracy by roughly five to six percentage points, and the
measured label drift was too small to make forgetting claims defensible.

The revised task remains issue triage, but the target is a VS Code component label at close time.
Component labels expose real taxonomy movement across 2019-2024: `notebook`, `settings-sync`,
`workspace-trust`, `terminal`, and `terminal-shell-integration` appear or rise in later windows,
while accessibility and SCM work become more prominent later. The old `integrated-terminal` label
is treated as a supported rename to `terminal`, not as a concept-drift target by itself.

The source must be public, read-only, credential-free, reproducible enough to audit, and safe to
share without redistributing raw issue text.

## Source Options Evaluated

| Option | Rate limits / gating | Label-state posture | Decision |
|---|---|---|---|
| GitHub REST Search API | Unauthenticated access is IP-limited; the previous build hit practical scale limits and produced only 341 records. | Returns current issue state, not label state at close. Reconstructing label edits requires per-issue timeline/event calls and does not scale without credentials. | Rejected for this stream: underpowered and not close-state faithful. |
| GH Archive hourly dumps | Public `.json.gz` hourly files with no per-request API rate limit. | Raw event payload can represent the issue object at event time, but local direct streaming was too slow for repeated wide searches on this machine. | Suitable source of record, but not the chosen access path for this build. |
| Public ClickHouse `github_events` mirror of GH Archive | Read-only public playground user; no credentials or writes. | Structured `IssuesEvent` rows expose `title`, `body`, `labels`, `created_at`, `updated_at`, `closed_at`, and `action`. The selected VS Code rows expose `opened`, `closed`, and `reopened`, not `labeled`/`unlabeled`, so label-edit counts are marked unavailable rather than inferred. | **Chosen.** It gives the GH Archive event-derived close-state rows at practical scale while preserving the manifest-only publication posture. |

Primary source references:

- GitHub Terms of Service, public repository access and public-content rights:
  <https://docs.github.com/en/site-policy/github-terms/github-terms-of-service>
- GitHub REST API rate limits:
  <https://docs.github.com/en/rest/using-the-rest-api/rate-limits-for-the-rest-api>
- GitHub rate-limit endpoint and separate search bucket:
  <https://docs.github.com/en/rest/rate-limit/rate-limit>
- GitHub Issues REST API shape:
  <https://docs.github.com/en/rest/issues/issues>
- GitHub Labels REST API shape:
  <https://docs.github.com/en/rest/issues/labels>
- GH Archive public timeline and hourly archives:
  <https://www.gharchive.org/>
- ClickHouse public playground read-only access:
  <https://clickhouse.com/docs/getting-started/playground>

## Decision

Use the public ClickHouse `github_events` mirror as the read-only GH Archive access path. Query
`IssuesEvent` rows where `action = 'closed'`, `repo_name = 'microsoft/vscode'`, and at least one
configured component label is present in the close-event label array.

The query samples deterministically across the whole configured window with
`github_issues_stream.sampling.method = month_stratified_hash` and
`github_issues_stream.sampling.seed = 23`: each calendar month gets an equal record quota, and
records inside the month are ordered by a seeded ClickHouse hash of the issue number. This replaces
the defective chronological `ORDER BY created_at ASC LIMIT ...` behavior, which selected only the
first slice of each year.

The build records accepted-record coverage for every window: first event, last event, and events
per month. The coverage gate fails when fewer than
`github_issues_stream.coverage.min_months_represented = 10` months are represented, capped by the
configured window length. The 2019 window is intentionally `2019-05-01` through `2020-01-01`
because an aggregate source query over `microsoft/vscode` closed `IssuesEvent` rows returned no
January-April 2019 rows in the public mirror, even before the label/body filters.

The build applies `github_issues_stream.canonical_label_aliases` before supervised targets are
assigned. The supported alias is `integrated-terminal -> terminal`. Evidence is the detector output
in the audit (`750` old-label records, `665` new-label records, one-day handoff, cosine `0.871837`)
plus public GitHub metadata: issue
[`microsoft/vscode#10023`](https://github.com/microsoft/vscode/issues/10023), titled
`Integrated Terminal : Allow Renaming sessions`, currently carries the `terminal` label, and the
public labels API returns `terminal` while `integrated-terminal` returns 404.

Commit only:

- `configs/github_issues_stream.yaml`
- `docs/github-issues-stream/manifest.json`
- `docs/github-issues-stream/audit.json`
- builder/stream code and offline tests

Raw/sanitized issue text remains only under `scratch/github_issues_stream/issues.jsonl`, which is
git-ignored. Published artifacts contain repo/name IDs, issue numbers, URLs, timestamps, source
label names, target labels, split/window assignments, aggregate audit statistics, and content
hashes. They do not contain issue titles or issue bodies.

The builder truncates issue bodies at `github_issues_stream.max_body_chars = 1000` before local
JSONL writing. That cap is a performance and privacy control; it is recorded in
`docs/configuration-register.md`.

## Repository and Target

The configured repository is `microsoft/vscode` (MIT). The target labels are component labels:
`accessibility`, `api`, `debug`, `electron`, `extensions`, `file-explorer`, `git`,
`integrated-terminal`, `javascript`, `notebook`, `remote`, `scm`, `search`, `settings-sync`,
`tasks`, `terminal`, `terminal-shell-integration`, `testing`, `typescript`, and
`workspace-trust`.

Rows with more than one configured component label at close are treated as conflicting target
labels after aliasing and dropped from supervised learning. This avoids pretending that
multi-component issues have a single unambiguous component annotation, while allowing old and new
names for the same supported component to collapse to one target.

## Measured Build

The 2026-10-14 rebuild accepted 10,065 records and dropped 1,603 records:

| Drop reason | Count |
|---|---:|
| Body below configured minimum | 1 |
| Conflicting component labels at close after aliasing | 542 |
| Duplicate issue returned across windows/queries | 848 |
| Near-duplicate prompt by high-threshold token Jaccard | 212 |

Build and disk metrics:

| Metric | Value |
|---|---:|
| Build time | 42.390 seconds |
| Raw local JSONL | 15,549,853 bytes |
| Manifest JSON | 7,532,994 bytes |
| Audit JSON | 101,529 bytes |
| Git-ignored output directory | 15 MB by `du -sh` |
| Committed manifest/audit directory | 7.3 MB by `du -sh` |

## Time-Ordered Splits

Windows are half-open source-coverage periods. The 2019 window is May-December because the source
mirror has no earlier closed VS Code issue events for this query family. Within each window,
records are sorted by close event time, repository, and issue number. The first 70% are train, the
next 10% validation, and the suffix 20% test.

| Family | Train | Validation | Test | Total |
|---|---:|---:|---:|---:|
| `github_issues_2019` | 1,192 | 170 | 341 | 1,703 |
| `github_issues_2020` | 1,266 | 180 | 363 | 1,809 |
| `github_issues_2021` | 1,201 | 171 | 345 | 1,717 |
| `github_issues_2022` | 1,194 | 170 | 343 | 1,707 |
| `github_issues_2023` | 1,162 | 166 | 333 | 1,661 |
| `github_issues_2024` | 1,027 | 146 | 295 | 1,468 |

Every full-year window covers all 12 months. The 2019 source-covered window covers all 8 configured
months. The 2024 test suffix has 295 records, below the earlier 300-item goal but still large enough
that one item moves accuracy by roughly 0.34 percentage points.

## Coverage Audit

| Family | First event | Last event | Months represented | Events per month |
|---|---|---|---:|---|
| 2019 | 2019-05-07 | 2019-12-31 | 8 | May 233, Jun 228, Jul 229, Aug 226, Sep 140, Oct 229, Nov 185, Dec 233 |
| 2020 | 2020-01-02 | 2020-12-31 | 12 | 148-157 per month |
| 2021 | 2021-01-02 | 2021-12-31 | 12 | 136-150 per month |
| 2022 | 2022-01-01 | 2022-12-30 | 12 | 137-150 per month |
| 2023 | 2023-01-02 | 2023-12-30 | 12 | 115-147 per month |
| 2024 | 2024-01-03 | 2024-12-29 | 12 | 82-147 per month |

## Drift Evidence

Label drift is Jensen-Shannon divergence in bits. Vocabulary drift is token-set Jaccard distance.
Each adjacent-window statistic is reported against deterministic same-window resampling baselines:
the same metric computed between two hash-shuffled halves of the previous and current windows.
`Label JS` is after aliasing; `Raw label JS` is before aliasing.

| Pair | Label JS | Raw label JS | Previous resample | Current resample | Vocabulary distance | Previous resample | Current resample |
|---|---:|---:|---:|---:|---:|---:|---:|
| 2019 -> 2020 | 0.1499 | 0.1499 | 0.0091 | 0.0027 | 0.7021 | 0.6670 | 0.6595 |
| 2020 -> 2021 | 0.1045 | 0.1597 | 0.0038 | 0.0076 | 0.7042 | 0.6604 | 0.6661 |
| 2021 -> 2022 | 0.1855 | 0.2082 | 0.0104 | 0.0092 | 0.6993 | 0.6747 | 0.6646 |
| 2022 -> 2023 | 0.0520 | 0.0520 | 0.0129 | 0.0076 | 0.6940 | 0.6625 | 0.6619 |
| 2023 -> 2024 | 0.0301 | 0.0301 | 0.0125 | 0.0091 | 0.6906 | 0.6573 | 0.6614 |

The 2021 -> 2022 pair remains the largest aliased label shift, but it is now moderate rather than
the previous rename-inflated `0.3229` claim. The 2020 -> 2021 raw/aliased gap shows why aliasing is
mandatory: much of the raw label movement is the `integrated-terminal -> terminal` rename, not a
new user-report concept. The 2023 -> 2024 label drift is weak after aliasing.

Vocabulary distances remain only moderately above same-window resampling baselines. The audit
records them, but this ADR does not treat vocabulary Jaccard alone as strong drift evidence because
same-window text diversity is already high.

## Baselines

The audit reports majority-class, keyword, current-window TF-IDF centroid, and all-earlier-window
TF-IDF centroid baselines on each chronological test suffix.

| Family | Majority accuracy | Keyword accuracy | Current-window TF-IDF | Earlier-window TF-IDF |
|---|---:|---:|---:|---:|
| 2019 | 0.2199 | 0.4633 | 0.5982 | n/a |
| 2020 | 0.2782 | 0.4380 | 0.5014 | 0.3967 |
| 2021 | 0.2174 | 0.4870 | 0.4725 | 0.4551 |
| 2022 | 0.2099 | 0.5394 | 0.4577 | 0.5102 |
| 2023 | 0.1832 | 0.4474 | 0.4294 | 0.4985 |
| 2024 | 0.1661 | 0.3966 | 0.4847 | 0.5119 |

The trained non-neural classifier provides a concrete floor for language-model claims. After
aliasing, earlier-window TF-IDF is competitive from 2021 onward and beats the current-window model
on 2022-2024 tests. That means the stream still has measurable label drift, but the classifier
comparison no longer supports a broad catastrophic-forgetting claim.

## Noise and Label Sanity

Computed sanity indicators:

- 542 rows had multiple configured component labels at close after aliasing and were dropped as
  conflicting.
- 750 accepted raw records used `integrated-terminal`; all are aliased to `terminal` before target
  assignment. Per-window alias counts are 377 in 2019, 194 in 2020, and 179 in 2021.
- The rename detector reports 190 high-similarity raw-label pairs. Only
  `integrated-terminal -> terminal` is applied as a supported alias. `git`/`scm` is reported with
  high text similarity but long overlap, and `terminal`/`terminal-shell-integration` is reported as
  related but overlapping; both remain ambiguous and are not aliased.
- The selected ClickHouse mirror does not expose `labeled` or `unlabeled` `IssuesEvent` actions for
  `microsoft/vscode` in 2019-2024; observed actions are `opened`, `closed`, and `reopened`.
  Therefore label-edit counts are marked unavailable in the audit rather than interpreted as zero
  human disagreement.
- Deleted or later-hidden issues are not observable through this mirror. The manifest records the
  limitation explicitly.

## Privacy and Safety Rules

The builder applies these rules before any local dataset row is written:

1. Mask email addresses as `[EMAIL]`.
2. Mask generic assignments such as `token=...`, `api_key=...`, `secret=...`, `password=...`, and
   `Authorization: Bearer ...` as `[SECRET]`.
3. Drop high-confidence secrets outright, including GitHub tokens, GitHub fine-grained PATs,
   OpenAI-style `sk-...` keys, Slack tokens, AWS access-key IDs, and private-key PEM blocks.
4. Drop bodies below the configured minimum length.
5. Drop duplicate issue IDs and near-duplicate prompts across all windows.

Raw text never enters committed files. The manifest contains hashes that let reviewers verify local
raw JSONL consistency without publishing issue text.

## Consequences

- The stream is large enough for window-level evaluation: 10,065 accepted examples and at least
  295 test records in every window.
- Month-stratified seeded sampling removes the earlier defect where each year represented only its
  first chronological slice.
- The task has measurable but moderate component-label drift after aliasing. The audit reports both
  raw and aliased label drift plus same-window resampling baselines, so taxonomy churn and ordinary
  sampling variation are visible.
- The 2019 window is not a full-year claim. It is a May-December source-covered window because the
  public mirror has no matching VS Code closed issue rows before May 2019.
- Rebuilds depend on the public ClickHouse GH Archive mirror. If that service changes schema,
  retention, quota, or availability, the manifest hashes will still detect local raw-text mismatch,
  but byte-identical rebuilding is not guaranteed.
- The stream is a research/evaluation artifact, not a redistributable raw-text dataset. Publication
  remains limited to the manifest, audit, code, and aggregate/model evaluation results unless a
  future ADR changes the legal posture.

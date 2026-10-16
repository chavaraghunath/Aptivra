"""Strict validation for Cosmic Ray mutation sessions.

Cosmic Ray's bundled survival-rate helper treats every result other than
``SURVIVED`` as a kill.  That includes lost workers and mutation-runner
exceptions.  This module reads the persisted session outcomes directly and
accepts a kill only when both the worker and test outcomes are valid.
"""

import argparse
import json
import re
import sqlite3
import tomllib
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass
class IncompetentCategory:
    """A documented reason that permits an incompetent mutant to be excluded."""

    name: str
    description: str
    operator_names: tuple[str, ...]
    reason_pattern: str

    def matches(self, operator_names: tuple[str, ...], reason: str) -> bool:
        """Return whether both the operator and complete recorded reason match."""
        return bool(set(operator_names).intersection(self.operator_names)) and re.fullmatch(
            self.reason_pattern,
            reason,
        ) is not None


@dataclass
class MutationGateConfig:
    """Validated strict-gate policy."""

    minimum_kill_rate: float
    incompetent_categories: tuple[IncompetentCategory, ...]


@dataclass
class ResultDetail:
    """A report entry for an invalid or incompetent work item."""

    job_id: str
    reason: str
    worker_outcome: str | None
    test_outcome: str | None
    operator_names: tuple[str, ...]
    category: str | None = None


@dataclass
class OutcomeCounts:
    """Outcome counts used to calculate and audit the strict score."""

    total: int
    killed: int
    survived: int
    incompetent: int
    excluded_incompetent: int
    counted_incompetent: int
    invalid: int
    unexecuted: int
    abnormal: int
    exception: int
    no_test: int
    skipped: int


@dataclass
class MutationGateReport:
    """Complete strict-gate result for one Cosmic Ray session."""

    session: str
    minimum_kill_rate: float
    kill_rate: float | None
    denominator: int
    passed: bool
    counts: OutcomeCounts
    incompetent_results: tuple[ResultDetail, ...]
    invalid_results: tuple[ResultDetail, ...]

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable structured report."""
        report = asdict(self)
        report["status"] = "pass" if self.passed else "fail"
        return report


@dataclass
class _MutableCounts:
    killed: int = 0
    survived: int = 0
    incompetent: int = 0
    excluded_incompetent: int = 0
    counted_incompetent: int = 0
    invalid: int = 0
    unexecuted: int = 0
    abnormal: int = 0
    exception: int = 0
    no_test: int = 0
    skipped: int = 0


@dataclass
class _SessionRow:
    job_id: str
    has_result: bool
    worker_outcome: str | None
    test_outcome: str | None
    output: str | None
    operator_names: tuple[str, ...]


def _required_string(mapping: dict[str, Any], key: str, context: str) -> str:
    value = mapping.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{context}.{key} must be a non-empty string")
    return value


def load_config(path: Path) -> MutationGateConfig:
    """Load and validate strict mutation-gate policy from TOML."""
    with path.open("rb") as handle:
        payload = tomllib.load(handle)

    gate = payload.get("mutation-gate")
    if not isinstance(gate, dict):
        raise ValueError("configuration requires a [mutation-gate] table")
    threshold = gate.get("minimum-kill-rate")
    if isinstance(threshold, bool) or not isinstance(threshold, (int, float)):
        raise ValueError("mutation-gate.minimum-kill-rate must be a number")
    threshold_value = float(threshold)
    if threshold_value < 0.0 or threshold_value > 100.0:
        raise ValueError("mutation-gate.minimum-kill-rate must be between 0 and 100")

    raw_categories = payload.get("incompetent-category", [])
    if not isinstance(raw_categories, list):
        raise ValueError("incompetent-category must be an array of tables")
    categories: list[IncompetentCategory] = []
    category_names: set[str] = set()
    for index, raw_category in enumerate(raw_categories):
        context = f"incompetent-category[{index}]"
        if not isinstance(raw_category, dict):
            raise ValueError(f"{context} must be a table")
        name = _required_string(raw_category, "name", context)
        if name in category_names:
            raise ValueError(f"duplicate incompetent category: {name}")
        description = _required_string(raw_category, "description", context)
        raw_operators = raw_category.get("operator-names")
        if (
            not isinstance(raw_operators, list)
            or not raw_operators
            or any(not isinstance(item, str) or not item for item in raw_operators)
        ):
            raise ValueError(f"{context}.operator-names must contain non-empty strings")
        reason_pattern = _required_string(raw_category, "reason-pattern", context)
        try:
            re.compile(reason_pattern)
        except re.error as error:
            raise ValueError(f"{context}.reason-pattern is invalid: {error}") from error
        categories.append(
            IncompetentCategory(
                name=name,
                description=description,
                operator_names=tuple(raw_operators),
                reason_pattern=reason_pattern,
            )
        )
        category_names.add(name)

    return MutationGateConfig(
        minimum_kill_rate=threshold_value,
        incompetent_categories=tuple(categories),
    )


def _normalized_outcome(value: str | None) -> str | None:
    if value is None:
        return None
    return value.strip().lower().replace("_", "-")


def _read_session(path: Path) -> tuple[_SessionRow, ...]:
    if not path.is_file():
        raise ValueError(f"Cosmic Ray session does not exist: {path}")
    try:
        with sqlite3.connect(f"file:{path}?mode=ro", uri=True) as connection:
            connection.row_factory = sqlite3.Row
            operator_names: dict[str, list[str]] = {}
            for row in connection.execute(
                "SELECT job_id, operator_name FROM mutation_specs "
                "ORDER BY job_id, operator_name"
            ):
                operator_names.setdefault(str(row["job_id"]), []).append(
                    str(row["operator_name"])
                )
            rows = connection.execute(
                "SELECT work_items.job_id, work_results.job_id AS result_job_id, "
                "work_results.worker_outcome, work_results.test_outcome, work_results.output "
                "FROM work_items LEFT JOIN work_results "
                "ON work_items.job_id = work_results.job_id "
                "ORDER BY work_items.job_id"
            ).fetchall()
    except sqlite3.Error as error:
        raise ValueError(f"invalid Cosmic Ray session {path}: {error}") from error

    return tuple(
        _SessionRow(
            job_id=str(row["job_id"]),
            has_result=row["result_job_id"] is not None,
            worker_outcome=_normalized_outcome(row["worker_outcome"]),
            test_outcome=_normalized_outcome(row["test_outcome"]),
            output=None if row["output"] is None else str(row["output"]),
            operator_names=tuple(operator_names.get(str(row["job_id"]), [])),
        )
        for row in rows
    )


def _detail(
    row: _SessionRow,
    reason: str,
    category: str | None = None,
) -> ResultDetail:
    return ResultDetail(
        job_id=row.job_id,
        reason=reason,
        worker_outcome=row.worker_outcome,
        test_outcome=row.test_outcome,
        operator_names=row.operator_names,
        category=category,
    )


def _incompetent_category(
    config: MutationGateConfig,
    row: _SessionRow,
    reason: str,
) -> IncompetentCategory | None:
    for category in config.incompetent_categories:
        if category.matches(row.operator_names, reason):
            return category
    return None


def evaluate_session(path: Path, config: MutationGateConfig) -> MutationGateReport:
    """Evaluate a Cosmic Ray session with fail-closed outcome semantics."""
    rows = _read_session(path)
    counts = _MutableCounts()
    invalid_results: list[ResultDetail] = []
    incompetent_results: list[ResultDetail] = []

    for row in rows:
        if not row.has_result:
            counts.unexecuted += 1
            counts.invalid += 1
            invalid_results.append(_detail(row, "unexecuted"))
            continue

        if row.test_outcome == "incompetent":
            counts.incompetent += 1

        if row.worker_outcome != "normal":
            match row.worker_outcome:
                case "abnormal":
                    counts.abnormal += 1
                case "exception":
                    counts.exception += 1
                case "no-test":
                    counts.no_test += 1
                case "skipped":
                    counts.skipped += 1
                case _:
                    pass
            counts.invalid += 1
            worker_reason = (
                "worker_outcome is null"
                if row.worker_outcome is None
                else f"worker_outcome={row.worker_outcome}"
            )
            invalid_results.append(_detail(row, worker_reason))
            if row.test_outcome == "incompetent":
                incompetent_results.append(
                    _detail(row, row.output or "<no output recorded>")
                )
            continue

        match row.test_outcome:
            case "killed":
                counts.killed += 1
            case "survived":
                counts.survived += 1
            case "incompetent":
                reason = row.output or "<no output recorded>"
                category = _incompetent_category(config, row, reason)
                incompetent_results.append(
                    _detail(row, reason, None if category is None else category.name)
                )
                if category is None:
                    counts.counted_incompetent += 1
                else:
                    counts.excluded_incompetent += 1
            case _:
                counts.invalid += 1
                test_reason = (
                    "test_outcome is null"
                    if row.test_outcome is None
                    else f"test_outcome={row.test_outcome}"
                )
                invalid_results.append(_detail(row, test_reason))

    denominator = counts.killed + counts.survived + counts.counted_incompetent
    kill_rate = None if denominator == 0 else counts.killed * 100.0 / denominator
    passed = (
        counts.invalid == 0
        and denominator > 0
        and kill_rate is not None
        and kill_rate >= config.minimum_kill_rate
    )
    frozen_counts = OutcomeCounts(
        total=len(rows),
        killed=counts.killed,
        survived=counts.survived,
        incompetent=counts.incompetent,
        excluded_incompetent=counts.excluded_incompetent,
        counted_incompetent=counts.counted_incompetent,
        invalid=counts.invalid,
        unexecuted=counts.unexecuted,
        abnormal=counts.abnormal,
        exception=counts.exception,
        no_test=counts.no_test,
        skipped=counts.skipped,
    )
    return MutationGateReport(
        session=str(path),
        minimum_kill_rate=config.minimum_kill_rate,
        kill_rate=kill_rate,
        denominator=denominator,
        passed=passed,
        counts=frozen_counts,
        incompetent_results=tuple(incompetent_results),
        invalid_results=tuple(invalid_results),
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Fail-closed validation of a completed Cosmic Ray session."
    )
    parser.add_argument("session", type=Path)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("mutation/strict-gate.toml"),
        help="Strict-gate TOML policy (default: mutation/strict-gate.toml).",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the strict gate and emit one structured JSON report."""
    arguments = _parser().parse_args(argv)
    try:
        config = load_config(arguments.config)
        report = evaluate_session(arguments.session, config)
    except OSError as error:
        print(json.dumps({"error": str(error), "status": "error"}))
        return 2
    except ValueError as error:
        print(json.dumps({"error": str(error), "status": "error"}))
        return 2
    print(json.dumps(report.to_dict()))
    return 0 if report.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())

"""Strict mutation-gate regression tests."""

from __future__ import annotations

import json
import sqlite3
import subprocess
import sys
from pathlib import Path

import pytest

from aptivra.mutation_gate import evaluate_session, load_config, main


def _write_config(
    path: Path,
    *,
    threshold: float = 90.0,
    categories: str = "",
) -> Path:
    path.write_text(
        "[mutation-gate]\n"
        f"minimum-kill-rate = {threshold}\n"
        f"{categories}",
        encoding="utf-8",
    )
    return path


def _write_raw_config(path: Path, content: str) -> Path:
    path.write_text(content, encoding="utf-8")
    return path


def _write_session(
    path: Path,
    rows: list[tuple[str, str | None, str | None, str | None, str]],
    *,
    pending: int = 0,
) -> Path:
    with sqlite3.connect(path) as connection:
        connection.executescript(
            """
            CREATE TABLE work_items (job_id TEXT PRIMARY KEY);
            CREATE TABLE mutation_specs (
                job_id TEXT NOT NULL,
                operator_name TEXT NOT NULL
            );
            CREATE TABLE work_results (
                job_id TEXT PRIMARY KEY,
                worker_outcome TEXT,
                test_outcome TEXT,
                output TEXT
            );
            """
        )
        for job_id, worker_outcome, test_outcome, output, operator_name in rows:
            connection.execute("INSERT INTO work_items VALUES (?)", (job_id,))
            connection.execute(
                "INSERT INTO mutation_specs VALUES (?, ?)",
                (job_id, operator_name),
            )
            connection.execute(
                "INSERT INTO work_results VALUES (?, ?, ?, ?)",
                (job_id, worker_outcome, test_outcome, output),
            )
        for index in range(pending):
            job_id = f"pending-{index}"
            connection.execute("INSERT INTO work_items VALUES (?)", (job_id,))
            connection.execute(
                "INSERT INTO mutation_specs VALUES (?, ?)",
                (job_id, "core/NumberReplacer"),
            )
    return path


def test_strict_gate_rejects_abnormal_result_that_cr_rate_counts_as_killed(
    tmp_path: Path,
) -> None:
    session = _write_session(
        tmp_path / "abnormal.sqlite",
        [("job-1", "ABNORMAL", None, "worker disappeared", "core/NumberReplacer")],
    )
    config = load_config(_write_config(tmp_path / "gate.toml"))

    report = evaluate_session(session, config)

    assert report.passed is False
    assert report.counts.abnormal == 1
    assert report.counts.killed == 0
    assert report.invalid_results[0].reason == "worker_outcome=abnormal"


def test_strict_gate_reports_every_outcome_without_conflating_counts(
    tmp_path: Path,
) -> None:
    rows = [
        ("01-killed", "NORMAL", "KILLED", "failed", "core/NumberReplacer"),
        ("02-survived", "normal", "survived", "passed", "core/NumberReplacer"),
        (
            "03-counted",
            "NORMAL",
            "INCOMPETENT",
            None,
            "core/NumberReplacer",
        ),
        (
            "04-excluded",
            "NORMAL",
            "INCOMPETENT",
            "documented defect",
            "core/ExceptionReplacer",
        ),
        ("05-abnormal", "ABNORMAL", None, "lost", "core/NumberReplacer"),
        (
            "06-exception",
            "EXCEPTION",
            "INCOMPETENT",
            None,
            "core/ExceptionReplacer",
        ),
        ("07-no-test", "NO_TEST", None, None, "core/NumberReplacer"),
        ("08-skipped", "SKIPPED", None, None, "core/NumberReplacer"),
        ("09-null-worker", None, None, None, "core/NumberReplacer"),
        ("10-null-test", "NORMAL", None, None, "core/NumberReplacer"),
        ("11-unknown-test", "NORMAL", "OTHER", None, "core/NumberReplacer"),
        ("12-unknown-worker", "OTHER", "KILLED", None, "core/NumberReplacer"),
    ]
    session = _write_session(tmp_path / "mixed.sqlite", rows, pending=1)
    categories = """

[[incompetent-category]]
name = "documented-defect"
description = "A recorded synthetic defect."
operator-names = ["core/ExceptionReplacer"]
reason-pattern = "documented defect"
"""
    config = load_config(
        _write_config(tmp_path / "gate.toml", threshold=30.0, categories=categories)
    )

    report = evaluate_session(session, config)

    assert report.passed is False
    assert report.denominator == 3
    assert report.kill_rate == pytest.approx(100.0 / 3.0)
    assert report.counts.total == 13
    assert report.counts.killed == 1
    assert report.counts.survived == 1
    assert report.counts.incompetent == 3
    assert report.counts.excluded_incompetent == 1
    assert report.counts.counted_incompetent == 1
    assert report.counts.invalid == 9
    assert report.counts.unexecuted == 1
    assert report.counts.abnormal == 1
    assert report.counts.exception == 1
    assert report.counts.no_test == 1
    assert report.counts.skipped == 1
    incompetent = {item.job_id: item for item in report.incompetent_results}
    assert incompetent["03-counted"].reason == "<no output recorded>"
    assert incompetent["03-counted"].category is None
    assert incompetent["04-excluded"].category == "documented-defect"
    assert incompetent["06-exception"].reason == "<no output recorded>"
    invalid = {item.job_id: item.reason for item in report.invalid_results}
    assert invalid == {
        "05-abnormal": "worker_outcome=abnormal",
        "06-exception": "worker_outcome=exception",
        "07-no-test": "worker_outcome=no-test",
        "08-skipped": "worker_outcome=skipped",
        "09-null-worker": "worker_outcome is null",
        "10-null-test": "test_outcome is null",
        "11-unknown-test": "test_outcome=other",
        "12-unknown-worker": "worker_outcome=other",
        "pending-0": "unexecuted",
    }


@pytest.mark.parametrize(
    ("worker_outcome", "test_outcome", "reason"),
    [
        ("EXCEPTION", "INCOMPETENT", "worker_outcome=exception"),
        ("NORMAL", None, "test_outcome is null"),
        ("NO_TEST", None, "worker_outcome=no-test"),
        ("SKIPPED", None, "worker_outcome=skipped"),
    ],
)
def test_strict_gate_rejects_invalid_completed_results(
    tmp_path: Path,
    worker_outcome: str,
    test_outcome: str | None,
    reason: str,
) -> None:
    session = _write_session(
        tmp_path / "invalid.sqlite",
        [("job-1", worker_outcome, test_outcome, "failure detail", "core/NumberReplacer")],
    )
    config = load_config(_write_config(tmp_path / "gate.toml"))

    report = evaluate_session(session, config)

    assert report.passed is False
    assert report.invalid_results[0].reason == reason


def test_strict_gate_rejects_unexecuted_work(tmp_path: Path) -> None:
    session = _write_session(tmp_path / "pending.sqlite", [], pending=2)
    config = load_config(_write_config(tmp_path / "gate.toml"))

    report = evaluate_session(session, config)

    assert report.passed is False
    assert report.counts.unexecuted == 2
    assert report.counts.total == 2


def test_empty_session_fails_without_inventing_a_score(tmp_path: Path) -> None:
    session = _write_session(tmp_path / "empty.sqlite", [])
    config = load_config(_write_config(tmp_path / "gate.toml"))

    report = evaluate_session(session, config)

    assert report.passed is False
    assert report.denominator == 0
    assert report.kill_rate is None
    assert report.counts.total == 0


def test_strict_gate_uses_configured_threshold(tmp_path: Path) -> None:
    rows = [
        *[
            (f"killed-{index}", "NORMAL", "KILLED", "failed", "core/NumberReplacer")
            for index in range(8)
        ],
        *[
            (f"survived-{index}", "NORMAL", "SURVIVED", "passed", "core/NumberReplacer")
            for index in range(2)
        ],
    ]
    session = _write_session(tmp_path / "threshold.sqlite", rows)

    passing = evaluate_session(
        session,
        load_config(_write_config(tmp_path / "passing.toml", threshold=80.0)),
    )
    failing = evaluate_session(
        session,
        load_config(_write_config(tmp_path / "failing.toml", threshold=80.01)),
    )

    assert passing.passed is True
    assert passing.kill_rate == 80.0
    assert failing.passed is False


def test_undocumented_incompetent_is_reported_and_counted_as_not_killed(
    tmp_path: Path,
) -> None:
    session = _write_session(
        tmp_path / "incompetent.sqlite",
        [("job-1", "NORMAL", "INCOMPETENT", "cannot launch tests", "core/NumberReplacer")],
    )
    config = load_config(_write_config(tmp_path / "gate.toml"))

    report = evaluate_session(session, config)

    assert report.passed is False
    assert report.kill_rate == 0.0
    assert report.counts.incompetent == 1
    assert report.counts.excluded_incompetent == 0
    assert report.incompetent_results[0].reason == "cannot launch tests"
    assert report.incompetent_results[0].category is None


def test_documented_incompetent_category_is_explicitly_excluded(
    tmp_path: Path,
) -> None:
    session = _write_session(
        tmp_path / "documented.sqlite",
        [
            ("job-1", "NORMAL", "KILLED", "failed", "core/NumberReplacer"),
            (
                "job-2",
                "NORMAL",
                "INCOMPETENT",
                "documented operator construction defect: AST mismatch",
                "core/ExceptionReplacer",
            ),
        ],
    )
    categories = """

[[incompetent-category]]
name = "operator-construction-defect"
description = "The configured operator cannot construct this generated mutant."
operator-names = ["core/ExceptionReplacer"]
reason-pattern = "^documented operator construction defect: .+$"
"""
    config = load_config(
        _write_config(tmp_path / "gate.toml", categories=categories)
    )

    report = evaluate_session(session, config)

    assert report.passed is True
    assert report.kill_rate == 100.0
    assert report.counts.incompetent == 1
    assert report.counts.excluded_incompetent == 1
    assert report.incompetent_results[0].category == "operator-construction-defect"
    assert "AST mismatch" in report.incompetent_results[0].reason


@pytest.mark.parametrize(
    ("operator_name", "reason"),
    [
        ("core/ExceptionReplacer", "different failure"),
        ("core/NumberReplacer", "documented operator construction defect: AST mismatch"),
        ("core/ExceptionReplacer", "documented operator construction defect: "),
        (
            "core/ExceptionReplacer",
            "prefix documented operator construction defect: AST mismatch",
        ),
    ],
)
def test_incompetent_category_requires_operator_and_complete_reason_match(
    tmp_path: Path,
    operator_name: str,
    reason: str,
) -> None:
    session = _write_session(
        tmp_path / "mismatch.sqlite",
        [
            (
                "job-1",
                "NORMAL",
                "INCOMPETENT",
                reason,
                operator_name,
            )
        ],
    )
    categories = """

[[incompetent-category]]
name = "operator-construction-defect"
description = "The configured operator cannot construct this generated mutant."
operator-names = ["core/ExceptionReplacer"]
reason-pattern = "^documented operator construction defect: .+$"
"""
    config = load_config(
        _write_config(tmp_path / "gate.toml", categories=categories)
    )

    report = evaluate_session(session, config)

    assert report.passed is False
    assert report.counts.excluded_incompetent == 0
    assert report.incompetent_results[0].category is None


@pytest.mark.parametrize("threshold", [-0.01, 100.01])
def test_gate_configuration_rejects_out_of_range_threshold(
    tmp_path: Path,
    threshold: float,
) -> None:
    path = _write_config(tmp_path / "gate.toml", threshold=threshold)

    with pytest.raises(ValueError, match="minimum-kill-rate"):
        load_config(path)


@pytest.mark.parametrize("threshold", [0.0, 100.0])
def test_gate_configuration_accepts_boundary_thresholds(
    tmp_path: Path,
    threshold: float,
) -> None:
    config = load_config(_write_config(tmp_path / "gate.toml", threshold=threshold))

    assert config.minimum_kill_rate == threshold


@pytest.mark.parametrize("threshold", ["true", '"90"'])
def test_gate_configuration_rejects_non_numeric_thresholds(
    tmp_path: Path,
    threshold: str,
) -> None:
    path = _write_raw_config(
        tmp_path / "gate.toml",
        f"[mutation-gate]\nminimum-kill-rate = {threshold}\n",
    )

    with pytest.raises(ValueError, match="must be a number"):
        load_config(path)


def test_gate_configuration_requires_gate_table(tmp_path: Path) -> None:
    path = _write_raw_config(tmp_path / "gate.toml", "minimum-kill-rate = 90.0\n")

    with pytest.raises(ValueError, match=r"\[mutation-gate\]"):
        load_config(path)


@pytest.mark.parametrize(
    ("category", "message"),
    [
        ('incompetent-category = "bad"\n', "array of tables"),
        ('incompetent-category = ["bad"]\n', "must be a table"),
        (
            "[[incompetent-category]]\n"
            'name = ""\n'
            'description = "description"\n'
            'operator-names = ["core/Operator"]\n'
            'reason-pattern = "reason"\n',
            "name must be a non-empty string",
        ),
        (
            "[[incompetent-category]]\n"
            'name = "category"\n'
            'description = ""\n'
            'operator-names = ["core/Operator"]\n'
            'reason-pattern = "reason"\n',
            "description must be a non-empty string",
        ),
        (
            "[[incompetent-category]]\n"
            'name = "category"\n'
            'description = "description"\n'
            "operator-names = []\n"
            'reason-pattern = "reason"\n',
            "operator-names must contain non-empty strings",
        ),
        (
            "[[incompetent-category]]\n"
            'name = "category"\n'
            'description = "description"\n'
            'operator-names = [""]\n'
            'reason-pattern = "reason"\n',
            "operator-names must contain non-empty strings",
        ),
        (
            "[[incompetent-category]]\n"
            'name = "category"\n'
            'description = "description"\n'
            "operator-names = [1]\n"
            'reason-pattern = "reason"\n',
            "operator-names must contain non-empty strings",
        ),
        (
            "[[incompetent-category]]\n"
            'name = "category"\n'
            'description = "description"\n'
            'operator-names = ["core/Operator"]\n'
            'reason-pattern = "["\n',
            "reason-pattern is invalid",
        ),
    ],
)
def test_gate_configuration_rejects_invalid_categories(
    tmp_path: Path,
    category: str,
    message: str,
) -> None:
    path = _write_raw_config(
        tmp_path / "gate.toml",
        category + "[mutation-gate]\nminimum-kill-rate = 90.0\n",
    )

    with pytest.raises(ValueError, match=message):
        load_config(path)


def test_gate_configuration_rejects_duplicate_category_names(tmp_path: Path) -> None:
    category = """
[[incompetent-category]]
name = "duplicate"
description = "first"
operator-names = ["core/First"]
reason-pattern = "first"

[[incompetent-category]]
name = "duplicate"
description = "second"
operator-names = ["core/Second"]
reason-pattern = "second"

[mutation-gate]
minimum-kill-rate = 90.0
"""
    path = _write_raw_config(tmp_path / "gate.toml", category)

    with pytest.raises(ValueError, match="duplicate incompetent category"):
        load_config(path)


def test_session_reader_rejects_missing_or_invalid_database(tmp_path: Path) -> None:
    config = load_config(_write_config(tmp_path / "gate.toml"))
    invalid = tmp_path / "invalid.sqlite"
    invalid.write_text("not sqlite", encoding="utf-8")

    with pytest.raises(ValueError, match="does not exist"):
        evaluate_session(tmp_path / "missing.sqlite", config)
    with pytest.raises(ValueError, match="invalid Cosmic Ray session"):
        evaluate_session(invalid, config)


def test_cli_prints_structured_report_and_returns_failure(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    session = _write_session(
        tmp_path / "abnormal.sqlite",
        [("job-1", "ABNORMAL", None, "worker disappeared", "core/NumberReplacer")],
    )
    config = _write_config(tmp_path / "gate.toml")

    exit_code = main([str(session), "--config", str(config)])
    output = json.loads(capsys.readouterr().out)

    assert exit_code == 1
    assert output["status"] == "fail"
    assert output["counts"]["abnormal"] == 1
    assert output["invalid_results"][0]["job_id"] == "job-1"


def test_cli_reports_input_errors_as_structured_output(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    missing_config = tmp_path / "missing.toml"

    exit_code = main([str(tmp_path / "missing.sqlite"), "--config", str(missing_config)])
    output = json.loads(capsys.readouterr().out)

    assert exit_code == 2
    assert output["status"] == "error"
    assert str(missing_config) in output["error"]


def test_module_entrypoint_propagates_gate_exit_status(tmp_path: Path) -> None:
    session = _write_session(
        tmp_path / "survived.sqlite",
        [("job-1", "NORMAL", "SURVIVED", "passed", "core/NumberReplacer")],
    )
    config = _write_config(tmp_path / "gate.toml")

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "aptivra.mutation_gate",
            str(session),
            "--config",
            str(config),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 1
    assert json.loads(result.stdout)["status"] == "fail"
    assert result.stderr == ""

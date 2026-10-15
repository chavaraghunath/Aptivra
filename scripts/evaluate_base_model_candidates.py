"""CLI for isolated, resumable Tier 3 base-model candidate measurements."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

from aptivra.eval.base_model_evaluation import (
    BaseModelEvaluationSettings,
    attach_environment_attempts,
    derive_amendment_3_analysis,
    derive_existing_records_analysis,
    load_base_model_evaluation_settings,
    prepare_run_manifest,
    rank_amendment_4g_records,
    rank_existing_candidate_records,
    run_amendment_4a_equivalence,
    run_amendment_4b_equivalence,
    run_evaluation,
    run_seed_variance_evaluation,
    system_free_memory_percentage,
    system_swap_used_bytes,
    write_environment_status,
    write_isolated_process_failure,
    write_json,
)


def _equivalence_allows_sweep(
    equivalence: dict[str, object], settings: BaseModelEvaluationSettings
) -> bool:
    """Keep failed gates closed unless the manifest-bound owner override is present."""
    return (
        equivalence.get("status") == "passed"
        or settings.equivalence_gate_owner_override is not None
    )


def _wait_until_environment_normal(
    *,
    minimum_free_memory_percent: float,
    timeout_seconds: float,
    poll_seconds: float,
    maximum_other_process_rss_bytes: int | None = None,
) -> dict[str, object]:
    """Wait for host pressure to become normal and return bounded evidence."""
    started = time.monotonic()
    samples: list[dict[str, float | int]] = []
    while True:
        sample: dict[str, object] = {
            "elapsed_seconds": time.monotonic() - started,
            "swap_used_bytes": system_swap_used_bytes(),
            "free_memory_percent": system_free_memory_percentage(),
        }
        heavy_processes = (
            []
            if maximum_other_process_rss_bytes is None
            else _other_processes_above_rss(maximum_other_process_rss_bytes)
        )
        sample["other_processes_above_rss_threshold"] = heavy_processes
        samples.append(sample)
        if (
            float(sample["free_memory_percent"]) >= minimum_free_memory_percent
            and not heavy_processes
        ):
            return {"normal": True, "samples": samples}
        if time.monotonic() - started >= timeout_seconds:
            return {"normal": False, "samples": samples}
        time.sleep(poll_seconds)


def _other_processes_above_rss(maximum_rss_bytes: int) -> list[dict[str, object]]:
    """List heavy processes while excluding this orchestrator and its descendants."""
    completed = subprocess.run(
        ["/bin/ps", "-axo", "pid=,ppid=,rss=,comm="],
        check=True,
        capture_output=True,
        text=True,
        timeout=5.0,
    )
    rows: list[tuple[int, int, int, str]] = []
    for line in completed.stdout.splitlines():
        fields = line.strip().split(maxsplit=3)
        if len(fields) != 4:
            continue
        try:
            pid, parent_pid, rss_kib = map(int, fields[:3])
        except ValueError:
            continue
        rows.append((pid, parent_pid, rss_kib * 1024, fields[3]))
    excluded = {os.getpid()}
    changed = True
    while changed:
        changed = False
        for pid, parent_pid, _, _ in rows:
            if parent_pid in excluded and pid not in excluded:
                excluded.add(pid)
                changed = True
    return [
        {"pid": pid, "rss_bytes": rss_bytes, "executable": executable}
        for pid, _, rss_bytes, executable in rows
        if pid not in excluded and rss_bytes > maximum_rss_bytes
    ]


def _run_child(
    config: Path,
    output_dir: Path,
    candidate_id: str,
    *,
    resume: bool,
    timeout_seconds: float,
    maximum_swap_growth_bytes: int,
    swap_poll_interval_seconds: float,
    minimum_free_memory_percent: float,
    environment_recovery_timeout_seconds: float,
    environment_max_attempts: int,
    prestart_minimum_free_memory_percent: float = 0.0,
    prestart_max_other_process_rss_bytes: int = 2**63 - 1,
    prestart_timeout_seconds: float = 0.0,
    prestart_poll_interval_seconds: float = 0.05,
    variance_seed: int | None = None,
    variance_primary_output_dir: Path | None = None,
) -> None:
    attempts: list[dict[str, object]] = []
    prestart = _wait_until_environment_normal(
        minimum_free_memory_percent=prestart_minimum_free_memory_percent,
        timeout_seconds=prestart_timeout_seconds,
        poll_seconds=prestart_poll_interval_seconds,
        maximum_other_process_rss_bytes=prestart_max_other_process_rss_bytes,
    )
    if prestart["normal"] is not True:
        monitor = {
            "attempt": 0,
            "status": "environment_invalid",
            "process_started": False,
            "environment_invalid_reasons": [
                "quiet-machine pre-start requirements did not pass before timeout"
            ],
            "prestart_check": prestart,
            "configured_prestart_minimum_free_memory_percent": (
                prestart_minimum_free_memory_percent
            ),
            "configured_prestart_max_other_process_rss_bytes": (
                prestart_max_other_process_rss_bytes
            ),
            "configured_prestart_timeout_seconds": prestart_timeout_seconds,
            "configured_prestart_poll_interval_seconds": (
                prestart_poll_interval_seconds
            ),
        }
        attempts.append(monitor)
        monitor_name = candidate_id.replace("/", "_")
        if variance_seed is not None:
            monitor_name += f"-seed-{variance_seed}"
        write_json(
            output_dir / "process_monitors" / f"{monitor_name}-prestart.json",
            monitor,
        )
        write_environment_status(
            config,
            output_dir,
            candidate_id,
            status="not_measured_environment",
            attempts=attempts,
            variance_seed=variance_seed,
        )
        print(
            json.dumps(
                {
                    "candidate_id": candidate_id,
                    "event": "quiet_machine_timeout",
                    "message": (
                        "candidate was not started because the quiet-machine check "
                        "did not pass before its configured timeout"
                    ),
                },
                sort_keys=True,
            ),
            flush=True,
        )
        return
    for attempt_number in range(1, environment_max_attempts + 1):
        recovery = _wait_until_environment_normal(
            minimum_free_memory_percent=minimum_free_memory_percent,
            timeout_seconds=environment_recovery_timeout_seconds,
            poll_seconds=swap_poll_interval_seconds,
        )
        if recovery["normal"] is not True:
            monitor = {
                "attempt": attempt_number,
                "status": "environment_invalid",
                "process_started": False,
                "environment_invalid_reasons": [
                    "system free-memory percentage did not recover before timeout"
                ],
                "recovery": recovery,
                "prestart_check": prestart,
                "configured_minimum_free_memory_percent": (
                    minimum_free_memory_percent
                ),
                "configured_recovery_timeout_seconds": (
                    environment_recovery_timeout_seconds
                ),
            }
            attempts.append(monitor)
            monitor_name = candidate_id.replace("/", "_")
            if variance_seed is not None:
                monitor_name += f"-seed-{variance_seed}"
            write_json(
                output_dir
                / "process_monitors"
                / f"{monitor_name}-attempt-{attempt_number}.json",
                monitor,
            )
            write_environment_status(
                config,
                output_dir,
                candidate_id,
                status="environment_invalid",
                attempts=attempts,
                variance_seed=variance_seed,
            )
            continue

        command = [
            sys.executable,
            "-m",
            "scripts.evaluate_base_model_candidates",
            "--config",
            str(config),
            "--output-dir",
            str(output_dir),
            "--candidate-id",
            candidate_id,
        ]
        command.append("--resume" if resume else "--no-resume")
        if variance_seed is not None:
            command.extend(("--variance-seed", str(variance_seed)))
            if variance_primary_output_dir is not None:
                command.extend(
                    (
                        "--variance-primary-output-dir",
                        str(variance_primary_output_dir),
                    )
                )
        swap_before = system_swap_used_bytes()
        swap_peak = swap_before
        free_before = system_free_memory_percentage()
        free_minimum = free_before
        started = time.monotonic()
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        environment_reasons: list[str] = []
        timeout_message: str | None = None
        while process.poll() is None:
            current_swap = system_swap_used_bytes()
            current_free = system_free_memory_percentage()
            swap_peak = max(swap_peak, current_swap)
            free_minimum = min(free_minimum, current_free)
            if current_free < minimum_free_memory_percent:
                environment_reasons.append(
                    "system free-memory percentage fell below the critical floor"
                )
            if _other_processes_above_rss(prestart_max_other_process_rss_bytes):
                environment_reasons.append(
                    "foreign heavy process appeared during candidate measurement"
                )
            if environment_reasons:
                process.terminate()
                break
            if time.monotonic() - started > timeout_seconds:
                timeout_message = f"candidate process exceeded {timeout_seconds} seconds"
                environment_reasons.append(timeout_message)
                process.terminate()
                break
            time.sleep(swap_poll_interval_seconds)
        if environment_reasons or timeout_message is not None:
            try:
                process.wait(timeout=10.0)
            except subprocess.TimeoutExpired:
                process.kill()
        stdout, stderr = process.communicate()
        swap_after = system_swap_used_bytes()
        free_after = system_free_memory_percentage()
        swap_peak = max(swap_peak, swap_after)
        free_minimum = min(free_minimum, free_after)
        if free_after < minimum_free_memory_percent:
            environment_reasons.append(
                "system free-memory percentage fell below the critical floor"
            )
        environment_reasons = list(dict.fromkeys(environment_reasons))
        throughput_compromised = (
            swap_peak - swap_before > maximum_swap_growth_bytes
        )
        monitor = {
            "attempt": attempt_number,
            "status": (
                "environment_invalid" if environment_reasons else "completed"
            ),
            "process_started": True,
            "system_swap_before_bytes": swap_before,
            "system_swap_peak_bytes": swap_peak,
            "system_swap_after_bytes": swap_after,
            "system_swap_peak_growth_bytes": max(0, swap_peak - swap_before),
            "system_swap_growth_bytes": max(0, swap_after - swap_before),
            "system_free_memory_before_percent": free_before,
            "system_free_memory_minimum_percent": free_minimum,
            "system_free_memory_after_percent": free_after,
            "configured_maximum_swap_growth_bytes": maximum_swap_growth_bytes,
            "configured_minimum_free_memory_percent": minimum_free_memory_percent,
            "throughput_compromised": throughput_compromised,
            "throughput_compromise_reason": (
                "system swap growth exceeded the configured reporting threshold"
                if throughput_compromised
                else None
            ),
            "poll_interval_seconds": swap_poll_interval_seconds,
            "elapsed_seconds": time.monotonic() - started,
            "returncode": process.returncode,
            "environment_invalid_reasons": environment_reasons,
            "forced_termination_reason": timeout_message,
            "recovery": recovery,
            "prestart_check": prestart,
        }
        attempts.append(monitor)
        monitor_name = candidate_id.replace("/", "_")
        if variance_seed is not None:
            monitor_name += f"-seed-{variance_seed}"
        write_json(
            output_dir
            / "process_monitors"
            / f"{monitor_name}-attempt-{attempt_number}.json",
            monitor,
        )
        if environment_reasons:
            write_environment_status(
                config,
                output_dir,
                candidate_id,
                status="environment_invalid",
                attempts=attempts,
                variance_seed=variance_seed,
            )
            return
        if process.returncode != 0:
            diagnostic = (stderr or stdout)[-2000:]
            write_isolated_process_failure(
                config,
                output_dir,
                candidate_id,
                returncode=(
                    process.returncode if process.returncode is not None else -1
                ),
                message=diagnostic or "candidate process exited without diagnostics",
                variance_seed=variance_seed,
                parent_process_memory=monitor,
            )
        attach_environment_attempts(
            output_dir,
            candidate_id,
            attempts,
            variance_seed=variance_seed,
        )
        return

    write_environment_status(
        config,
        output_dir,
        candidate_id,
        status="not_measured_environment",
        attempts=attempts,
        variance_seed=variance_seed,
    )


def _run_amendment_4g(
    config: Path,
    output_dir: Path,
    amendment_4c_output_dir: Path,
    amendment_4f_output_dir: Path,
    settings: BaseModelEvaluationSettings,
    *,
    resume: bool,
) -> dict[str, object]:
    """Run only the full-rule-derived Amendment 4g top-two seed repeats."""
    first_pass = rank_amendment_4g_records(
        config,
        output_dir,
        amendment_4c_output_dir,
        amendment_4f_output_dir,
    )
    ranked = first_pass.get("ranked_candidates")
    if not isinstance(ranked, list) or len(ranked) < 2:
        raise RuntimeError("Amendment 4g full rule did not produce two eligible leaders")
    leaders: list[str] = []
    for row in ranked[: settings.measurement.variance_top_candidates]:
        if not isinstance(row, dict) or not isinstance(row.get("candidate_id"), str):
            raise TypeError("Amendment 4g ranking candidate row is malformed")
        leaders.append(row["candidate_id"])
    if len(leaders) != 2:
        raise RuntimeError("Amendment 4g requires exactly two seed-repeat leaders")
    for candidate_id in leaders:
        primary_output_dir = (
            amendment_4f_output_dir
            if candidate_id == "qwen2.5-0.5b"
            else amendment_4c_output_dir
        )
        for seed in settings.measurement.variance_seeds[1:]:
            print(
                json.dumps(
                    {
                        "candidate_id": candidate_id,
                        "event": "seed_repeat_started",
                        "seed": seed,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
            _run_child(
                config,
                output_dir,
                candidate_id,
                resume=resume,
                timeout_seconds=settings.measurement.candidate_timeout_seconds,
                maximum_swap_growth_bytes=settings.decision.maximum_swap_growth_bytes,
                swap_poll_interval_seconds=(
                    settings.measurement.swap_sample_interval_seconds
                ),
                minimum_free_memory_percent=(
                    settings.measurement.environment_minimum_free_memory_percent
                ),
                environment_recovery_timeout_seconds=(
                    settings.measurement.environment_recovery_timeout_seconds
                ),
                environment_max_attempts=settings.measurement.environment_max_attempts,
                prestart_minimum_free_memory_percent=(
                    settings.measurement.prestart_minimum_free_memory_percent
                ),
                prestart_max_other_process_rss_bytes=(
                    settings.measurement.prestart_max_other_process_rss_bytes
                ),
                prestart_timeout_seconds=settings.measurement.prestart_timeout_seconds,
                prestart_poll_interval_seconds=(
                    settings.measurement.prestart_poll_interval_seconds
                ),
                variance_seed=seed,
                variance_primary_output_dir=primary_output_dir,
            )
    return rank_amendment_4g_records(
        config,
        output_dir,
        amendment_4c_output_dir,
        amendment_4f_output_dir,
    )


def main() -> None:
    orchestration_started = time.perf_counter()
    orchestration_started_at = datetime.now(UTC).isoformat()
    parser = argparse.ArgumentParser(
        description="Evaluate and rank configured Aptivra base-model candidates"
    )
    parser.add_argument(
        "--config", type=Path, default=Path("configs/base_model_candidates.yaml")
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument(
        "--resume", action=argparse.BooleanOptionalAction, default=None
    )
    parser.add_argument("--candidate-id", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--variance-seed", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument(
        "--variance-primary-output-dir",
        type=Path,
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--derive-existing-analysis",
        action="store_true",
        help="derive Amendment 2 analysis from frozen records without model work",
    )
    parser.add_argument(
        "--derive-amendment-3-analysis",
        action="store_true",
        help="combine prior powered evidence with Amendment 3 records without model work",
    )
    parser.add_argument(
        "--prior-output-dir",
        type=Path,
        default=Path("experiments/base_model_candidates"),
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--run-amendment-4g",
        action="store_true",
        help="run only Amendment 4g top-two repeats and its full-rule ranking",
    )
    parser.add_argument(
        "--amendment-4c-output-dir",
        type=Path,
        default=Path("experiments/base_model_candidates/amendment-4c"),
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--amendment-4f-output-dir",
        type=Path,
        default=Path("experiments/base_model_candidates/amendment-4f"),
        help=argparse.SUPPRESS,
    )
    args = parser.parse_args()

    settings = load_base_model_evaluation_settings(args.config)
    if settings.execution is None:
        raise RuntimeError("base-model evaluation config is missing execution settings")
    output_dir = args.output_dir or settings.execution.output_dir
    resume = settings.execution.resume if args.resume is None else args.resume
    if args.run_amendment_4g:
        if args.candidate_id is not None or args.variance_seed is not None:
            parser.error("Amendment 4g orchestration cannot be a candidate worker")
        ranking = _run_amendment_4g(
            args.config,
            output_dir,
            args.amendment_4c_output_dir,
            args.amendment_4f_output_dir,
            settings,
            resume=resume,
        )
        ranking["orchestration"] = {
            "started_at": orchestration_started_at,
            "finished_at": datetime.now(UTC).isoformat(),
            "actual_wall_time_seconds": time.perf_counter() - orchestration_started,
            "expected_wall_time_seconds": settings.measurement.expected_wall_time_seconds,
        }
        write_json(output_dir / "ranking.json", ranking)
        print(
            json.dumps(
                {
                    "output_dir": str(output_dir),
                    "decision_status": ranking["decision_status"],
                    "selected_candidate": ranking["selected_candidate"],
                },
                sort_keys=True,
            )
        )
        return
    if args.derive_existing_analysis:
        if args.candidate_id is not None or args.variance_seed is not None:
            parser.error("derived analysis cannot be combined with candidate worker options")
        analysis, ranking = derive_existing_records_analysis(
            args.config, output_dir
        )
        print(
            json.dumps(
                {
                    "analysis_schema_version": analysis["schema_version"],
                    "output_dir": str(output_dir),
                    "selected_candidate": ranking["selected_candidate"],
                },
                sort_keys=True,
            )
        )
        return
    if args.derive_amendment_3_analysis:
        if args.candidate_id is not None or args.variance_seed is not None:
            parser.error("derived analysis cannot be combined with candidate worker options")
        _, ranking = derive_amendment_3_analysis(
            args.config, output_dir, args.prior_output_dir
        )
        print(
            json.dumps(
                {
                    "output_dir": str(output_dir),
                    "selected_candidate": ranking["selected_candidate"],
                },
                sort_keys=True,
            )
        )
        return
    prepare_run_manifest(args.config, output_dir, settings)
    if settings.equivalence_gate_owner_override is not None:
        equivalence = {
            "status": "owner_overridden",
            "owner_override": settings.equivalence_gate_owner_override,
        }
        write_json(output_dir / "equivalence-owner-override.json", equivalence)
    elif settings.measurement.equivalence_protocol == "amendment_4b":
        equivalence = run_amendment_4b_equivalence(args.config, output_dir)
    else:
        equivalence = run_amendment_4a_equivalence(args.config, output_dir)
    if not _equivalence_allows_sweep(equivalence, settings):
        print(
            json.dumps(
                {
                    "equivalence_status": equivalence.get("status"),
                    "output_dir": str(output_dir),
                    "sweep_started": False,
                },
                sort_keys=True,
            )
        )
        raise SystemExit(2)
    if args.candidate_id is not None:
        known = {candidate.candidate_id for candidate in settings.candidates}
        if args.candidate_id not in known:
            parser.error(f"unknown candidate id: {args.candidate_id}")
        if args.variance_seed is None:
            records, _ = run_evaluation(
                args.config,
                output_dir,
                resume=resume,
                candidate_ids={args.candidate_id},
            )
            status = records[0]["status"] if records else "missing"
        else:
            variance_record = run_seed_variance_evaluation(
                args.config,
                output_dir,
                args.candidate_id,
                args.variance_seed,
                resume=resume,
                primary_output_dir=args.variance_primary_output_dir,
            )
            status = variance_record["status"]
        print(
            json.dumps(
                {
                    "candidate_id": args.candidate_id,
                    "variance_seed": args.variance_seed,
                    "status": status,
                },
                sort_keys=True,
            )
        )
        return

    for candidate in (
        item for item in settings.candidates if not item.reference_only
    ):
        print(
            json.dumps(
                {
                    "candidate_id": candidate.candidate_id,
                    "event": "candidate_started",
                },
                sort_keys=True,
            ),
            flush=True,
        )
        _run_child(
            args.config,
            output_dir,
            candidate.candidate_id,
            resume=resume,
            timeout_seconds=settings.measurement.candidate_timeout_seconds,
            maximum_swap_growth_bytes=settings.decision.maximum_swap_growth_bytes,
            swap_poll_interval_seconds=settings.measurement.swap_sample_interval_seconds,
            minimum_free_memory_percent=(
                settings.measurement.environment_minimum_free_memory_percent
            ),
            environment_recovery_timeout_seconds=(
                settings.measurement.environment_recovery_timeout_seconds
            ),
            environment_max_attempts=settings.measurement.environment_max_attempts,
            prestart_minimum_free_memory_percent=(
                settings.measurement.prestart_minimum_free_memory_percent
            ),
            prestart_max_other_process_rss_bytes=(
                settings.measurement.prestart_max_other_process_rss_bytes
            ),
            prestart_timeout_seconds=(
                settings.measurement.prestart_timeout_seconds
            ),
            prestart_poll_interval_seconds=(
                settings.measurement.prestart_poll_interval_seconds
            ),
        )
    first_pass_ranking = rank_existing_candidate_records(args.config, output_dir)
    ranked = first_pass_ranking["ranked_candidates"]
    if not isinstance(ranked, list):
        raise TypeError("ranking record has no ranked_candidates list")
    leaders = (
        []
        if first_pass_ranking["decision_status"] == "measurement_incomplete"
        else ranked[: settings.measurement.variance_top_candidates]
    )
    for row in leaders:
        if not isinstance(row, dict) or not isinstance(row.get("candidate_id"), str):
            raise TypeError("ranking candidate row is malformed")
        for seed in settings.measurement.variance_seeds[1:]:
            _run_child(
                args.config,
                output_dir,
                row["candidate_id"],
                resume=resume,
                timeout_seconds=settings.measurement.candidate_timeout_seconds,
                maximum_swap_growth_bytes=(
                    settings.decision.maximum_swap_growth_bytes
                ),
                swap_poll_interval_seconds=(
                    settings.measurement.swap_sample_interval_seconds
                ),
                minimum_free_memory_percent=(
                    settings.measurement.environment_minimum_free_memory_percent
                ),
                environment_recovery_timeout_seconds=(
                    settings.measurement.environment_recovery_timeout_seconds
                ),
                environment_max_attempts=(
                    settings.measurement.environment_max_attempts
                ),
                prestart_minimum_free_memory_percent=(
                    settings.measurement.prestart_minimum_free_memory_percent
                ),
                prestart_max_other_process_rss_bytes=(
                    settings.measurement.prestart_max_other_process_rss_bytes
                ),
                prestart_timeout_seconds=(
                    settings.measurement.prestart_timeout_seconds
                ),
                prestart_poll_interval_seconds=(
                    settings.measurement.prestart_poll_interval_seconds
                ),
                variance_seed=seed,
            )
    ranking = rank_existing_candidate_records(args.config, output_dir)
    ranking["orchestration"] = {
        "started_at": orchestration_started_at,
        "finished_at": datetime.now(UTC).isoformat(),
        "actual_wall_time_seconds": time.perf_counter() - orchestration_started,
        "expected_wall_time_seconds": settings.measurement.expected_wall_time_seconds,
    }
    write_json(output_dir / "ranking.json", ranking)
    print(
        json.dumps(
            {
                "output_dir": str(output_dir),
                "decision_status": ranking["decision_status"],
                "selected_candidate": ranking["selected_candidate"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()

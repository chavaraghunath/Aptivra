"""CLI for isolated, resumable Tier 3 base-model candidate measurements."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

from aptivra.eval.base_model_evaluation import (
    derive_existing_records_analysis,
    load_base_model_evaluation_settings,
    prepare_run_manifest,
    rank_existing_candidate_records,
    run_evaluation,
    run_seed_variance_evaluation,
    write_isolated_process_failure,
    write_json,
)


def _run_child(
    config: Path,
    output_dir: Path,
    candidate_id: str,
    *,
    resume: bool,
    timeout_seconds: float,
    variance_seed: int | None = None,
) -> None:
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
    try:
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
        )
    except subprocess.TimeoutExpired as error:
        write_isolated_process_failure(
            config,
            output_dir,
            candidate_id,
            returncode=-1,
            message=f"candidate process exceeded {timeout_seconds} seconds: {error}",
            variance_seed=variance_seed,
        )
        return
    if completed.returncode != 0:
        diagnostic = (completed.stderr or completed.stdout)[-2000:]
        write_isolated_process_failure(
            config,
            output_dir,
            candidate_id,
            returncode=completed.returncode,
            message=diagnostic or "candidate process exited without diagnostics",
            variance_seed=variance_seed,
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
        "--derive-existing-analysis",
        action="store_true",
        help="derive Amendment 2 analysis from frozen records without model work",
    )
    args = parser.parse_args()

    settings = load_base_model_evaluation_settings(args.config)
    if settings.execution is None:
        raise RuntimeError("base-model evaluation config is missing execution settings")
    output_dir = args.output_dir or settings.execution.output_dir
    resume = settings.execution.resume if args.resume is None else args.resume
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
    prepare_run_manifest(args.config, output_dir, settings)
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

    for candidate in settings.candidates:
        _run_child(
            args.config,
            output_dir,
            candidate.candidate_id,
            resume=resume,
            timeout_seconds=settings.measurement.candidate_timeout_seconds,
        )
    first_pass_ranking = rank_existing_candidate_records(args.config, output_dir)
    ranked = first_pass_ranking["ranked_candidates"]
    if not isinstance(ranked, list):
        raise TypeError("ranking record has no ranked_candidates list")
    leaders = ranked[: settings.measurement.variance_top_candidates]
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

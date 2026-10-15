"""Run Amendment 4e foreground checks or launch its detached MPS completion."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import cast

from aptivra.eval.base_model_evaluation import (
    load_base_model_evaluation_settings,
    prepare_run_manifest,
    rank_amendment_4e_records,
    run_amendment_4e_candidate_check,
    write_json,
)
from scripts.evaluate_base_model_candidates import _run_child

DEFAULT_CONFIG = Path("configs/base_model_candidates_amendment_4e.yaml")
DEFAULT_PRIOR_OUTPUT = Path("experiments/base_model_candidates/amendment-4c")
FRESH_CANDIDATES = ("qwen2.5-0.5b", "qwen2.5-1.5b", "smollm2-1.7b")
COMPLETED_4C_CANDIDATES = (
    "gpt2",
    "smollm2-135m",
    "pythia-160m",
    "smollm2-360m",
)


def _require_passed(result: dict[str, object]) -> None:
    """Make every foreground gate fail closed for shell automation."""
    if result.get("status") not in {"passed", "passed_documented_nondeterminism"}:
        raise SystemExit(2)


def _check_record(output_dir: Path, name: str) -> dict[str, object]:
    path = output_dir / "checks" / name
    if not path.is_file():
        raise RuntimeError(f"required foreground check is missing: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"foreground check is malformed: {path}")
    return cast(dict[str, object], value)


def run_measurement(config: Path, prior_output_dir: Path) -> dict[str, object]:
    """Measure the authorised candidates and write one mixed-provenance ranking."""
    settings = load_base_model_evaluation_settings(config)
    if settings.execution is None:
        raise RuntimeError("Amendment 4e config requires execution settings")
    output_dir = settings.execution.output_dir
    foreground_evidence_dir = (
        output_dir.parent / "amendment-4e"
        if output_dir.name == "amendment-4f"
        else output_dir.parent
    )
    placement = _check_record(foreground_evidence_dir, "placement-equivalence.json")
    if placement.get("status") != "passed_documented_nondeterminism":
        raise RuntimeError("SmolLM2-135M placement equivalence did not pass")
    consistency = _check_record(foreground_evidence_dir, "gpt2-consistency.json")
    if consistency.get("status") != "passed_documented_nondeterminism":
        raise RuntimeError("GPT-2 consistency check did not pass")
    manifest, _ = prepare_run_manifest(config, output_dir, settings)
    if settings.equivalence_gate_owner_override is not None:
        write_json(
            output_dir / "equivalence-owner-override.json",
            {
                "status": "owner_overridden",
                "owner_override": settings.equivalence_gate_owner_override,
                "correlation_id": manifest["correlation_id"],
            },
        )
    rerun_completed = False
    candidate_ids = FRESH_CANDIDATES
    for candidate_id in candidate_ids:
        print(
            json.dumps(
                {"candidate_id": candidate_id, "event": "candidate_started"},
                sort_keys=True,
            ),
            flush=True,
        )
        _run_child(
            config,
            output_dir,
            candidate_id,
            resume=settings.execution.resume,
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
        )
    first_pass_ranking = rank_amendment_4e_records(
        config,
        output_dir,
        prior_output_dir,
        rerun_completed_4c_candidates=rerun_completed,
    )
    ranked = first_pass_ranking.get("ranked_candidates")
    if not isinstance(ranked, list):
        raise TypeError("Amendment 4e ranking has no ranked_candidates list")
    leaders = (
        []
        if first_pass_ranking.get("decision_status") == "measurement_incomplete"
        else ranked[: settings.measurement.variance_top_candidates]
    )
    for row in leaders:
        if not isinstance(row, dict) or not isinstance(row.get("candidate_id"), str):
            raise TypeError("Amendment 4e ranking candidate row is malformed")
        candidate_id = cast(str, row["candidate_id"])
        primary_output_dir = (
            output_dir if candidate_id in FRESH_CANDIDATES else prior_output_dir
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
                resume=settings.execution.resume,
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
    ranking = rank_amendment_4e_records(
        config,
        output_dir,
        prior_output_dir,
        rerun_completed_4c_candidates=rerun_completed,
    )
    print(
        json.dumps(
            {
                "decision_status": ranking["decision_status"],
                "output_dir": str(output_dir),
                "selected_candidate": ranking["selected_candidate"],
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return ranking


def launch_detached(config: Path, prior_output_dir: Path) -> dict[str, object]:
    """Launch the bounded offline workflow under caffeinate and persist its PID."""
    settings = load_base_model_evaluation_settings(config)
    if settings.execution is None:
        raise RuntimeError("Amendment 4e config requires execution settings")
    output_dir = settings.execution.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    amendment_name = output_dir.name
    log_path = output_dir / f"{amendment_name}.log"
    pid_path = output_dir / f"{amendment_name}.pid"
    if pid_path.is_file():
        prior_pid_text = pid_path.read_text(encoding="utf-8").strip()
        if prior_pid_text.isdigit():
            try:
                os.kill(int(prior_pid_text), 0)
            except ProcessLookupError:
                pass
            else:
                raise RuntimeError(
                    f"Amendment 4e is already running with PID {prior_pid_text}"
                )
    caffeinate = shutil.which("caffeinate")
    if caffeinate is None:
        raise RuntimeError("caffeinate is required for the detached measurement")
    command = [
        caffeinate,
        "-dimsu",
        sys.executable,
        "-m",
        (
            "scripts.run_amendment_4f"
            if amendment_name == "amendment-4f"
            else "scripts.run_amendment_4e"
        ),
        "--config",
        str(config),
        "--prior-output-dir",
        str(prior_output_dir),
        "--measure-and-rank",
    ]
    environment = os.environ.copy()
    environment["HF_HUB_OFFLINE"] = "1"
    environment["TRANSFORMERS_OFFLINE"] = "1"
    environment["PYTHONPATH"] = str(Path.cwd())
    with log_path.open("a", encoding="utf-8") as log_handle:
        process = subprocess.Popen(
            command,
            cwd=Path.cwd(),
            env=environment,
            stdin=subprocess.DEVNULL,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            text=True,
        )
    pid_path.write_text(f"{process.pid}\n", encoding="utf-8")
    result = {
        "pid": process.pid,
        "pid_path": str(pid_path),
        "log_path": str(log_path),
    }
    print(json.dumps(result, sort_keys=True), flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--prior-output-dir", type=Path, default=DEFAULT_PRIOR_OUTPUT
    )
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--placement-equivalence", action="store_true")
    action.add_argument("--gpt2-consistency", action="store_true")
    action.add_argument("--measure-and-rank", action="store_true")
    action.add_argument("--launch-detached", action="store_true")
    args = parser.parse_args()
    settings = load_base_model_evaluation_settings(args.config)
    if settings.execution is None:
        raise RuntimeError("Amendment 4e config requires execution settings")
    if args.placement_equivalence:
        result = run_amendment_4e_candidate_check(
            args.config,
            settings.execution.output_dir,
            args.prior_output_dir,
            "smollm2-135m",
            placement_equivalence=True,
        )
        print(json.dumps(result, sort_keys=True))
        _require_passed(result)
    elif args.gpt2_consistency:
        result = run_amendment_4e_candidate_check(
            args.config,
            settings.execution.output_dir,
            args.prior_output_dir,
            "gpt2",
            placement_equivalence=False,
        )
        print(json.dumps(result, sort_keys=True))
        _require_passed(result)
    elif args.measure_and_rank:
        run_measurement(args.config, args.prior_output_dir)
    else:
        launch_detached(args.config, args.prior_output_dir)


if __name__ == "__main__":
    main()

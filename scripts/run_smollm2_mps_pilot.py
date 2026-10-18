"""Run the bounded, offline ADR-0011 lane-8b MPS pilot."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Literal, cast

from aptivra.eval.mps_pilot import (
    PilotMethod,
    _json_mapping,
    _write_json_atomic,
    load_mps_pilot_settings,
    run_calibration_worker,
    run_evaluation_profile_worker,
    run_pilot,
    run_repeatability_worker,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/h3_mps_pilot.yaml"))
    parser.add_argument("--authorize-pilot", action="store_true")
    parser.add_argument("--resume-amendment", type=Path)
    parser.add_argument("--worker-output", type=Path)
    parser.add_argument("--quiet-record", type=Path)
    parser.add_argument("--calibration-worker", nargs=3, metavar=("METHOD", "STREAM", "MB"))
    parser.add_argument(
        "--repeatability-worker",
        nargs=4,
        metavar=("METHOD", "STREAM", "SEED", "INDEX"),
    )
    parser.add_argument("--evaluation-profile-worker")
    parser.add_argument("--checkpointing", action="store_true")
    args = parser.parse_args()
    settings = load_mps_pilot_settings(args.config)
    if (
        args.calibration_worker is not None
        or args.repeatability_worker is not None
        or args.evaluation_profile_worker is not None
    ):
        if args.worker_output is None or args.quiet_record is None:
            parser.error("workers require --worker-output and --quiet-record")
        quiet = _json_mapping(args.quiet_record)
        if args.calibration_worker is not None:
            method, stream, micro_batch = args.calibration_worker
            record = run_calibration_worker(
                settings,
                cast(PilotMethod, method),
                stream,
                int(micro_batch),
                quiet,
                checkpointing=args.checkpointing,
            )
        elif args.repeatability_worker is not None:
            method, stream, seed, repeat_index = args.repeatability_worker
            if method not in {"naive_sequential", "aptivra_full"}:
                parser.error("repeatability method must be naive_sequential or aptivra_full")
            record = run_repeatability_worker(
                settings,
                cast(Literal["naive_sequential", "aptivra_full"], method),
                stream,
                int(seed),
                int(repeat_index),
                quiet,
            )
        else:
            record = run_evaluation_profile_worker(
                settings,
                str(args.evaluation_profile_worker),
                quiet,
            )
        _write_json_atomic(args.worker_output, record)
        return
    result = run_pilot(
        settings,
        authorized=args.authorize_pilot,
        config_path=args.config,
        resume_amendment_path=args.resume_amendment,
    )
    print(
        f"Pilot {result['status']}: {result['pilot_wall_time_seconds']:.3f} seconds; "
        f"evidence={settings.output_dir / 'lane8b-pilot-evidence.raw.json'}"
    )


if __name__ == "__main__":
    main()

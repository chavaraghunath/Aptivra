"""Launch Amendment 4f's detached, offline Mac measurement."""

from __future__ import annotations

import argparse
from pathlib import Path

from scripts.run_amendment_4e import launch_detached, run_measurement

DEFAULT_CONFIG = Path("configs/base_model_candidates_amendment_4f.yaml")
DEFAULT_PRIOR_OUTPUT = Path("experiments/base_model_candidates/amendment-4c")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--prior-output-dir", type=Path, default=DEFAULT_PRIOR_OUTPUT)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--measure-and-rank", action="store_true")
    action.add_argument("--launch-detached", action="store_true")
    args = parser.parse_args()
    if args.measure_and_rank:
        run_measurement(args.config, args.prior_output_dir)
    else:
        launch_detached(args.config, args.prior_output_dir)


if __name__ == "__main__":
    main()

"""Launch Amendment 4g's manifest-bound detached offline seed repeats."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

from aptivra.eval.base_model_evaluation import load_base_model_evaluation_settings

DEFAULT_CONFIG = Path("configs/base_model_candidates_amendment_4g.yaml")
DEFAULT_4C_OUTPUT = Path("experiments/base_model_candidates/amendment-4c")
DEFAULT_4F_OUTPUT = Path("experiments/base_model_candidates/amendment-4f")


def launch_detached(
    config: Path,
    amendment_4c_output_dir: Path,
    amendment_4f_output_dir: Path,
) -> dict[str, object]:
    """Launch the bounded evaluator CLI under caffeinate and persist its PID."""
    settings = load_base_model_evaluation_settings(config)
    if settings.execution is None:
        raise RuntimeError("Amendment 4g config requires execution settings")
    output_dir = settings.execution.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "amendment-4g.log"
    pid_path = output_dir / "amendment-4g.pid"
    if pid_path.is_file():
        prior_pid_text = pid_path.read_text(encoding="utf-8").strip()
        if prior_pid_text.isdigit():
            try:
                os.kill(int(prior_pid_text), 0)
            except ProcessLookupError:
                pass
            else:
                raise RuntimeError(
                    f"Amendment 4g is already running with PID {prior_pid_text}"
                )
    caffeinate = shutil.which("caffeinate")
    if caffeinate is None:
        raise RuntimeError("caffeinate is required for the detached measurement")
    command = [
        caffeinate,
        "-dimsu",
        sys.executable,
        "-m",
        "scripts.evaluate_base_model_candidates",
        "--config",
        str(config),
        "--output-dir",
        str(output_dir),
        "--amendment-4c-output-dir",
        str(amendment_4c_output_dir),
        "--amendment-4f-output-dir",
        str(amendment_4f_output_dir),
        "--run-amendment-4g",
        "--resume",
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
        "--amendment-4c-output-dir", type=Path, default=DEFAULT_4C_OUTPUT
    )
    parser.add_argument(
        "--amendment-4f-output-dir", type=Path, default=DEFAULT_4F_OUTPUT
    )
    args = parser.parse_args()
    launch_detached(
        args.config,
        args.amendment_4c_output_dir,
        args.amendment_4f_output_dir,
    )


if __name__ == "__main__":
    main()

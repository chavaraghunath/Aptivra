"""Transformers AutoModel smoke test for Aptivra Hub models (option H)."""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Aptivra via Hugging Face Transformers")
    parser.add_argument("--repo-id", type=str, default=None)
    parser.add_argument("--config", type=Path, default=Path("configs/inference.yaml"))
    parser.add_argument("--prompt", type=str, default="User: hello\nAssistant:")
    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    repo_id = args.repo_id or cfg["model_repo_id"]

    try:
        from transformers import AutoModelForCausalLM
    except ImportError as exc:
        raise SystemExit('Install HF extras: pip install -e ".[hf]"') from exc

    model = AutoModelForCausalLM.from_pretrained(repo_id, trust_remote_code=True)
    lm = model.lm
    if lm is None:
        raise SystemExit("Model failed to load AptivraCausalLM")

    text = lm.generate(args.prompt, max_new_tokens=int(cfg["max_new_tokens"]))
    print(text)


if __name__ == "__main__":
    main()

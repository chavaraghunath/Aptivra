"""Fail if identity chat raw outputs do not meet acceptance (no Gradio sanitization)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

from aptivra.inference.causal_lm import AptivraCausalLM
from aptivra.inference.chat_prompt import format_chat_prompt
from aptivra.inference.chat_canonical import IDENTITY_CHAT_TEMPLATES
from aptivra.inference.chat_quality import (
    canonical_prefix_ok,
    default_identity_cases,
    passes_case,
)


def _canonical_match(user_message: str, raw: str) -> bool:
    q = user_message.strip().lower()
    for label, _prompt, target in IDENTITY_CHAT_TEMPLATES:
        if label.lower().rstrip("?") == q.rstrip("?"):
            return canonical_prefix_ok(raw, target, min_ratio=0.55)
    return False
from aptivra.inference.hub_config import model_config_from_export_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify Aptivra identity chat quality")
    parser.add_argument("--config", type=Path, default=Path("configs/inference.yaml"))
    parser.add_argument("--export-dir", type=Path, default=None)
    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    export_path = args.export_dir or Path(cfg["local_export_dir"])
    model_config = model_config_from_export_dir(export_path, device=str(cfg.get("device", "auto")))
    lm = AptivraCausalLM.from_export_dir(export_path, model_config=model_config)
    lm.use_moe_at_inference = bool(cfg.get("use_moe_at_inference", False))
    max_new = int(cfg.get("max_new_tokens", 96))

    failed = 0
    for case in default_identity_cases():
        prompt = format_chat_prompt([], case.user_message)
        raw = lm.generate(
            prompt,
            max_new_tokens=max_new,
            stop_on_single_newline=False,
            stop_on_sentence_end=True,
        )
        ok = passes_case(raw, case) or _canonical_match(case.user_message, raw)
        status = "PASS" if ok else "FAIL"
        print(f"{status} {case.user_message!r} -> {raw[:160]!r}")
        if not ok:
            failed += 1

    if failed:
        print(f"\n{failed} case(s) failed. Re-run scripts/repair_chat_backbone.py before pushing to HF.")
        sys.exit(1)
    print("\nAll identity chat checks passed (raw model, no sanitization).")


if __name__ == "__main__":
    main()

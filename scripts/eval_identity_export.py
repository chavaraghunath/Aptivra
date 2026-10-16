"""Verifier accuracy for ModelIdentity prompts on exported AptivraCausalLM."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

from aptivra.eval.harness import ContinualEvalHarness
from aptivra.v0.inference.causal_lm import AptivraCausalLM
from aptivra.v0.inference.hub_config import model_config_from_export_dir
from aptivra.v0.tasks.identity import ModelIdentityTaskFamily


def main() -> None:
    parser = argparse.ArgumentParser(description="Identity verifier eval on export checkpoint")
    parser.add_argument("--export-dir", type=Path, default=Path("scratch/hf_export"))
    parser.add_argument("--tag", type=str, default="eval")
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("experiments/identity_export_eval.jsonl"),
    )
    args = parser.parse_args()

    model_config = model_config_from_export_dir(args.export_dir, device="auto")
    lm = AptivraCausalLM.from_export_dir(args.export_dir, model_config=model_config)
    lm.use_moe_at_inference = False

    family = ModelIdentityTaskFamily()
    instances = family.generate_instances(count=20, seed=42)
    passed = 0
    details: list[dict[str, str | bool]] = []

    def predict(prompt: str, stop_token_ids: tuple[int, ...]) -> str:
        if stop_token_ids != (0, ord("\n")):
            raise ValueError("identity evaluation requires the single-line stop contract")
        return lm.generate(
            prompt,
            max_new_tokens=args.max_new_tokens,
            stop_on_single_newline=True,
            stop_on_sentence_end=False,
        )

    for inst in instances:
        raw, res = ContinualEvalHarness.verify_prediction(
            predict,
            family,
            inst,
        )
        passed += int(res.passed)
        details.append({"task_id": inst.task_id, "passed": res.passed, "output": raw[:200]})

    accuracy = passed / max(len(instances), 1)
    record = {
        "tag": args.tag,
        "timestamp": datetime.now(tz=UTC).isoformat(),
        "export_dir": str(args.export_dir.resolve()),
        "identity_accuracy": accuracy,
        "passed": passed,
        "total": len(instances),
        "details": details,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")
    print(
        json.dumps(
            {
                "tag": args.tag,
                "identity_accuracy": accuracy,
                "passed": passed,
                "total": len(instances),
            }
        )
    )


if __name__ == "__main__":
    main()

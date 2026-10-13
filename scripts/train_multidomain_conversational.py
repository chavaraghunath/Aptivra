# ruff: noqa: E501
"""Training script for Aptivra Multi-Domain Expertise, Self-Identity, Conversation, and Multi-Tasking."""

import argparse
import json
import random
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim

from aptivra.baselines.base import BaselineTrainer
from aptivra.eval.model_card_eval import evaluate_export_dir, write_evaluation_record
from aptivra.export.hf_hub_artifacts import export_huggingface_hub_layout
from aptivra.export.hf_push import push_aptivra_model_to_hf
from aptivra.inference.hub_config import model_config_from_export_dir
from aptivra.memory.hybrid_store import HybridMemoryStore
from aptivra.tasks.conversational import ConversationalTaskFamily
from aptivra.tasks.identity import ModelIdentityTaskFamily
from aptivra.tasks.multitasking import MultiTaskingTaskFamily
from aptivra.tasks.skill_dataset import MCPSkillRepositoryLoader


def train_multidomain_conversational(
    config_path: str = "configs/backbone_200m.yaml",
    output_jsonl: str = "experiments/multidomain_conversational_results.jsonl",
    skills_dir: str = "/Users/raghunath/code/others/skills-mcp/skills",
    num_steps_per_task: int = 20,
    seeds: list[int] | None = None,
    push_hf: bool = True,
    hf_repo_id: str = "raghunath1/aptivra-200m-tiered-memory",
    *,
    hf_results_record: str | None = None,
    include_conversational: bool = False,
    include_multitasking: bool = False,
    run_sleep_cycle: bool = False,
) -> None:
    """Executes self-identity, multi-domain, conversational, and multi-tasking training for Aptivra 200M MoE."""
    if push_hf and hf_results_record is None:
        raise ValueError("hf_results_record is required when push_hf is enabled")
    seeds = seeds or [42]
    out_path = Path(output_jsonl)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print("==========================================================================")
    print("⚡ Starting Aptivra Self-Identity, Multi-Domain & Conversational Training")
    print("==========================================================================")

    # 1. Initialize Aptivra 200M Model Architecture
    export_dir = Path("scratch/hf_export")
    model_config = model_config_from_export_dir(export_dir, device="auto")
    dev = model_config.get_device()

    from aptivra.export.multitier import MultitierLoader

    loader = MultitierLoader(export_dir)
    backbone, router, experts = loader.load_model(model_config)
    backbone = backbone.to(dev)
    router = router.to(dev)
    experts = [exp.to(dev) for exp in experts]
    memory_store = HybridMemoryStore(
        embedding_dim=model_config.d_model,
        max_raw_replay_capacity=1000,
    )

    # 2. Register Task Families (Identity -> MCP Skills -> Conversation -> Multi-Tasking)
    task_families = []

    # Priority 1: Self-Identity & Capabilities
    task_families.append(ModelIdentityTaskFamily())

    # Priority 2: Multi-Domain MCP Skills
    skill_loader = MCPSkillRepositoryLoader(skills_dir=skills_dir)
    available_skills = skill_loader.list_available_skills()
    print(f"Discovered {len(available_skills)} MCP Skills in '{skills_dir}'")

    target_skills = [
        "ai.retrieval.rag_systems",
        "analysis.empirical.python_pipeline",
        "api.design.rest.graphql",
        "agent.framework.autogen_development",
        "academic.data.web_collection",
    ]

    for sk in target_skills:
        fam = skill_loader.load_skill_task_family(sk)
        if fam:
            task_families.append(fam)

    if include_conversational:
        task_families.append(ConversationalTaskFamily())
    if include_multitasking:
        task_families.append(MultiTaskingTaskFamily())

    print(f"Registered {len(task_families)} task families for continual update.")

    # Optimizer
    optimizer = optim.AdamW(backbone.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss(ignore_index=-100)

    results_log = []

    # 3. Continual Training Across Task Stream
    for seed in seeds:
        print(f"\n--- Training Seed {seed} ---")
        random.seed(seed)
        torch.manual_seed(seed)

        for task_idx, family in enumerate(task_families):
            print(f"Task [{task_idx + 1}/{len(task_families)}]: {family.name} ({family.family_id})")
            instances = family.generate_instances(count=num_steps_per_task, seed=seed)

            task_losses = []
            start_time = time.time()

            for _step, inst in enumerate(instances):
                optimizer.zero_grad()

                p_toks = BaselineTrainer.encode_text(inst.prompt)
                s_toks = BaselineTrainer.encode_text(inst.target_solution)
                full_seq = torch.cat([p_toks, s_toks])[: model_config.max_seq_len]
                if full_seq.numel() < 2:
                    continue
                target_seq = full_seq.clone()
                target_seq[: p_toks.numel()] = -100

                inputs = full_seq.unsqueeze(0).to(dev)
                targets = target_seq.unsqueeze(0).to(dev)

                backbone.train()
                logits, _ = backbone(inputs)
                shift_logits = logits[:, :-1, :].contiguous()
                shift_targets = targets[:, 1:].contiguous()
                loss = criterion(
                    shift_logits.reshape(-1, model_config.vocab_size),
                    shift_targets.reshape(-1),
                )

                loss.backward()
                optimizer.step()

                task_losses.append(loss.item())

                # Admit verified samples to HybridMemoryStore
                res = family.verifier.verify(inst, inst.target_solution)
                if res.passed:
                    memory_store.write_sample(
                        prompt=inst.prompt,
                        target_solution=inst.target_solution,
                        verifier_result=res,
                        task_family=inst.task_family,
                        input_ids=inputs.cpu(),
                        labels=targets.cpu(),
                    )

            avg_loss = sum(task_losses) / len(task_losses) if task_losses else 0.0
            elapsed = time.time() - start_time
            print(
                f"  Completed {len(instances)} steps in {elapsed:.2f}s | Avg Loss: {avg_loss:.4f} | Memory Size: {len(memory_store)}"
            )

            entry = {
                "seed": seed,
                "task_idx": task_idx,
                "task_id": family.family_id,
                "task_name": family.name,
                "steps": len(instances),
                "avg_loss": avg_loss,
                "memory_consolidated_count": len(memory_store),
                "elapsed_sec": elapsed,
            }
            results_log.append(entry)

            with open(out_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry) + "\n")

        # Re-anchor identity after dialogue / multitask families (prevents chat regression).
        print("Identity consolidation pass (post-stream)...")
        identity_family = ModelIdentityTaskFamily()
        id_instances = identity_family.generate_instances(count=num_steps_per_task, seed=seed)
        for inst in id_instances * 3:
            optimizer.zero_grad()
            p_toks = BaselineTrainer.encode_text(inst.prompt)
            s_toks = BaselineTrainer.encode_text(inst.target_solution)
            full_seq = torch.cat([p_toks, s_toks])[: model_config.max_seq_len]
            if full_seq.numel() < 2:
                continue
            target_seq = full_seq.clone()
            target_seq[: p_toks.numel()] = -100
            inputs = full_seq.unsqueeze(0).to(dev)
            targets = target_seq.unsqueeze(0).to(dev)
            backbone.train()
            logits, _ = backbone(inputs)
            shift_logits = logits[:, :-1, :].contiguous()
            shift_targets = targets[:, 1:].contiguous()
            loss = criterion(
                shift_logits.reshape(-1, model_config.vocab_size),
                shift_targets.reshape(-1),
            )
            loss.backward()
            optimizer.step()

    print("\n==========================================================================")
    print("✅ Self-Identity & Multi-Domain Training Complete! Results saved to:", out_path)
    print("==========================================================================")

    from aptivra.consolidation.sleep_cycle import GenerativeSleepEngine
    from aptivra.export.multitier import MultitierExporter
    from aptivra.memory.hexa_store import HexaTierMemoryStore

    if run_sleep_cycle:
        hexa = HexaTierMemoryStore(embedding_dim=model_config.d_model)
        sleep = GenerativeSleepEngine(backbone, hexa, model_config)
        sleep_report = sleep.execute_sleep_cycle(max_episodes=min(len(memory_store), 50))
        print(f"Generative sleep consolidation: {sleep_report}")
    else:
        print("Skipping generative sleep/prune (preserves backbone weights for chat).")

    backbone_cpu = backbone.cpu()
    router_cpu = router.cpu()
    experts_cpu = [exp.cpu() for exp in experts]
    MultitierExporter(export_dir).export_model(
        model=backbone_cpu,
        router=router_cpu,
        experts=experts_cpu,
        model_config=model_config,
    )
    print(f"Re-exported multitier shards to {export_dir.resolve()}")

    # 4. Build, evaluate, and publish the stock-Transformers layout.
    if push_hf:
        assert hf_results_record is not None
        export_huggingface_hub_layout(
            export_dir,
            backbone_cpu,
            router_cpu,
            experts_cpu,
            model_config,
        )
        results_path = Path(hf_results_record)
        record = evaluate_export_dir(export_dir)
        write_evaluation_record(record, results_path)
        print(f"\nPushing evaluated Aptivra Transformers layout to: '{hf_repo_id}'...")
        push_aptivra_model_to_hf(
            repo_id=hf_repo_id,
            export_dir=export_dir,
            results_record=results_path,
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train Aptivra Self-Identity, Multi-Domain & Conversational Model"
    )
    parser.add_argument("--config", type=str, default="configs/backbone_200m.yaml")
    parser.add_argument(
        "--output", type=str, default="experiments/multidomain_conversational_results.jsonl"
    )
    parser.add_argument(
        "--skills-dir", type=str, default="/Users/raghunath/code/others/skills-mcp/skills"
    )
    parser.add_argument("--steps-per-task", type=int, default=20)
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    parser.add_argument("--no-hf-push", action="store_true", help="Disable Hugging Face upload")
    parser.add_argument(
        "--hf-results-record",
        type=Path,
        help="Path where the required fresh pre-publication evaluation record is written",
    )
    parser.add_argument(
        "--include-conversational",
        action="store_true",
        help="Include conversational task family (can hurt byte-LM identity)",
    )
    parser.add_argument(
        "--include-multitasking",
        action="store_true",
        help="Include multitasking task family",
    )
    parser.add_argument(
        "--run-sleep-cycle",
        action="store_true",
        help="Run synaptic prune sleep (destructive to chat; off by default)",
    )
    args = parser.parse_args()
    if not args.no_hf_push and args.hf_results_record is None:
        parser.error("--hf-results-record is required unless --no-hf-push is set")

    train_multidomain_conversational(
        config_path=args.config,
        output_jsonl=args.output,
        skills_dir=args.skills_dir,
        num_steps_per_task=args.steps_per_task,
        seeds=args.seeds,
        push_hf=not args.no_hf_push,
        hf_results_record=(
            str(args.hf_results_record) if args.hf_results_record is not None else None
        ),
        include_conversational=args.include_conversational,
        include_multitasking=args.include_multitasking,
        run_sleep_cycle=args.run_sleep_cycle,
    )


if __name__ == "__main__":
    main()

"""Fine-tune exported backbone for chat (UTF-8 bytes + full transformer forward)."""

from __future__ import annotations

import copy
from pathlib import Path

import torch
import torch.nn as nn

from aptivra.baselines.base import BaselineTrainer
from aptivra.tasks.base import TaskFamily, TaskInstance
from aptivra.v0.export.multitier import MultitierExporter, MultitierLoader
from aptivra.v0.inference.chat_canonical import IDENTITY_CHAT_TEMPLATES
from aptivra.v0.inference.hub_config import model_config_from_export_dir
from aptivra.v0.tasks.conversational import ConversationalTaskFamily
from aptivra.v0.tasks.identity import ModelIdentityTaskFamily


def _build_chat_instances(
    repeats: int, *, include_conversational: bool = True
) -> list[TaskInstance]:
    families: list[TaskFamily] = [ModelIdentityTaskFamily()]
    if include_conversational:
        families.append(ConversationalTaskFamily())
    instances: list[TaskInstance] = []
    for family in families:
        batch = family.generate_instances(count=max(repeats, 1), seed=42)
        for _ in range(repeats):
            instances.extend(batch)

    for idx, (_label, prompt, answer) in enumerate(IDENTITY_CHAT_TEMPLATES):
        for _ in range(repeats * 8):
            instances.append(
                TaskInstance(
                    task_id=f"ui_extra_{idx}",
                    task_family="model_identity_capabilities",
                    prompt=prompt,
                    target_solution=answer,
                )
            )
    return instances


def finetune_backbone_for_chat(
    export_dir: Path | str,
    *,
    epochs: int = 40,
    lr: float = 3e-4,
    device: str = "auto",
    repeats: int = 8,
    include_conversational: bool = False,
    early_stop_patience: int = 10,
) -> Path:
    """Trains backbone lm_head path on identity + conversational data; re-exports shards."""
    export_path = Path(export_dir)
    model_config = model_config_from_export_dir(export_path, device=device)
    dev = model_config.get_device()

    loader = MultitierLoader(export_path)
    backbone, router, experts = loader.load_model(model_config)
    backbone = backbone.to(dev)
    router = router.to(dev)
    experts_list = [e.to(dev) for e in experts]

    backbone.train()
    optimizer = torch.optim.AdamW(backbone.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss(ignore_index=-100)

    instances = _build_chat_instances(
        repeats=repeats,
        include_conversational=include_conversational,
    )

    best_loss = float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    stale_epochs = 0

    for epoch in range(epochs):
        epoch_loss = 0.0
        steps = 0
        for inst in instances:
            p_toks = BaselineTrainer.encode_text(inst.prompt)
            s_toks = BaselineTrainer.encode_text(inst.target_solution)
            full_seq = torch.cat([p_toks, s_toks])[: model_config.max_seq_len]
            if full_seq.numel() < 2:
                continue

            target_seq = full_seq.clone()
            target_seq[: p_toks.numel()] = -100

            inputs = full_seq.unsqueeze(0).to(dev)
            targets = target_seq.unsqueeze(0).to(dev)

            optimizer.zero_grad()
            logits, _ = backbone(inputs)
            shift_logits = logits[:, :-1, :].contiguous()
            shift_targets = targets[:, 1:].contiguous()
            if shift_targets.numel() == 0:
                continue
            loss = criterion(
                shift_logits.reshape(-1, model_config.vocab_size),
                shift_targets.reshape(-1),
            )
            loss.backward()
            optimizer.step()
            epoch_loss += float(loss.item())
            steps += 1

        avg = epoch_loss / max(steps, 1)
        print(f"chat finetune epoch {epoch + 1}/{epochs} loss={avg:.4f}", flush=True)
        if avg + 1e-6 < best_loss:
            best_loss = avg
            best_state = copy.deepcopy(backbone.state_dict())
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= early_stop_patience:
                print(f"Early stop at epoch {epoch + 1} (best loss={best_loss:.4f})", flush=True)
                break

    if best_state is not None:
        backbone.load_state_dict(best_state)

    backbone.eval()
    backbone_cpu = backbone.cpu()
    router_cpu = router.cpu()
    experts_cpu = [exp.cpu() for exp in experts_list]
    MultitierExporter(export_path).export_model(
        model=backbone_cpu,
        router=router_cpu,
        experts=experts_cpu,
        model_config=model_config,
    )
    return export_path

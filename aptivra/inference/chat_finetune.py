"""Target-only PEFT identity-adapter training for the versioned chat template."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Union, cast

import torch
from transformers.modeling_outputs import CausalLMOutputWithPast

from aptivra.experts.standard import (
    ExpertManifest,
    PeftExpertBank,
    StandardExpertSettings,
)
from aptivra.inference.chat_canonical import IdentityManifest
from aptivra.inference.chat_prompt import (
    END_OF_TEXT,
    IM_END,
    ChatTokenizer,
    IdentitySettings,
    validate_chat_tokenizer,
)
from aptivra.inference.standard_base import StandardBase
from aptivra.tasks.base import TaskInstance
from aptivra.tasks.conversational import ConversationalTaskFamily
from aptivra.tasks.identity import ModelIdentityTaskFamily

_IGNORE_INDEX = -100

# Cosmic Ray treats postponed PEP 604 annotations as executable binary
# expressions, creating equivalent arithmetic mutants.  Keep the precise
# public union without producing inert operator substitutions.


@dataclass(frozen=True)
class IdentityTrainingBatch:
    """One exact prompt/assistant-span training example."""

    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    labels: torch.Tensor
    task_id: str
    prompt_tokens: int
    target_tokens: int


@dataclass(frozen=True)
class IdentityTrainingResult:
    """Observable isolation, optimization, and artifact evidence."""

    manifest: ExpertManifest
    base_sha256_before: str
    base_sha256_after: str
    optimizer_steps: int
    target_tokens: int
    final_loss: float


def identity_training_instances(
    manifest: IdentityManifest,
    *,
    expert_settings: StandardExpertSettings,
) -> tuple[TaskInstance, ...]:
    """Return single- and multi-turn records generated from the same manifest."""
    identity = ModelIdentityTaskFamily(manifest, expert_settings=expert_settings)
    conversational = ConversationalTaskFamily(manifest, expert_settings=expert_settings)
    record_count = 5
    return tuple(
        identity.generate_instances(record_count, seed=0)
        + conversational.generate_instances(record_count, seed=0)
    )


def materialize_identity_batch(
    base: StandardBase,
    instance: TaskInstance,
    settings: IdentitySettings,
) -> IdentityTrainingBatch:
    """Mask the prompt and supervise answer, message close, and EOS exactly."""
    token_ids = validate_chat_tokenizer(
        cast(ChatTokenizer, base.tokenizer),
        settings.template,
    )
    prompt_ids = base.tokenizer.encode(instance.prompt, add_special_tokens=False)
    target_text = f"{instance.target_solution}{IM_END}{END_OF_TEXT}"
    target_ids = base.tokenizer.encode(target_text, add_special_tokens=False)
    if not prompt_ids:
        raise ValueError(f"identity task {instance.task_id!r} has an empty tokenized prompt")
    expected_suffix = [token_ids[IM_END], token_ids[END_OF_TEXT]]
    if target_ids[-2:] != expected_suffix:
        raise ValueError("identity target must end with message delimiter followed by EOS")
    total_tokens = len(prompt_ids) + len(target_ids)
    limit = min(
        settings.training.max_sequence_tokens,
        base.settings.runtime.max_sequence_tokens,
    )
    if total_tokens > limit:
        raise ValueError(
            f"identity task {instance.task_id!r} exceeds configured token limit: "
            f"prompt={len(prompt_ids)}, target={len(target_ids)}, total={total_tokens}, "
            f"limit={limit}"
        )
    ids = [*prompt_ids, *target_ids]
    labels = [_IGNORE_INDEX] * len(prompt_ids) + target_ids
    input_tensor = torch.tensor([ids], dtype=torch.long, device=base.device)
    return IdentityTrainingBatch(
        input_ids=input_tensor,
        attention_mask=torch.ones_like(input_tensor),
        labels=torch.tensor([labels], dtype=torch.long, device=base.device),
        task_id=instance.task_id,
        prompt_tokens=len(prompt_ids),
        target_tokens=len(target_ids),
    )


def train_identity_adapter(
    base: StandardBase,
    expert_settings: StandardExpertSettings,
    identity_settings: IdentitySettings,
    identity_manifest: IdentityManifest,
    output_dir: Union[Path, str],  # noqa: UP007
) -> IdentityTrainingResult:
    """Train and atomically persist only the separate PEFT ``identity`` adapter."""
    if not identity_settings.feature.enabled:
        raise RuntimeError("ADR-0011 identity adapter training is disabled by configuration")
    if not expert_settings.feature.enabled:
        raise RuntimeError("ADR-0011 PEFT expert path is disabled by configuration")
    validate_chat_tokenizer(
        cast(ChatTokenizer, base.tokenizer),
        identity_settings.template,
    )
    bank = PeftExpertBank.start_candidate(
        base,
        expert_settings,
        identity_settings.training.adapter_name,
    )
    base_before = bank.base_parameter_sha256()
    parameters = bank.adapter_parameters(identity_settings.training.adapter_name)
    optimizer = torch.optim.AdamW(
        parameters,
        lr=identity_settings.training.learning_rate,
        weight_decay=identity_settings.training.weight_decay,
    )
    instances = identity_training_instances(
        identity_manifest,
        expert_settings=expert_settings,
    )
    steps = 0
    remaining_steps = identity_settings.training.max_optimizer_steps
    target_tokens = 0
    final_loss = math.nan
    bank.model.train()
    for _epoch in range(identity_settings.training.epochs):
        for instance in instances:
            if not remaining_steps:
                break
            batch = materialize_identity_batch(base, instance, identity_settings)
            optimizer.zero_grad(set_to_none=True)
            output = cast(
                CausalLMOutputWithPast,
                bank.model(
                    input_ids=batch.input_ids,
                    attention_mask=batch.attention_mask,
                    labels=batch.labels,
                    use_cache=False,
                    return_dict=True,
                ),
            )
            if output.loss is None or not bool(torch.isfinite(output.loss).item()):
                raise RuntimeError("identity adapter training produced a non-finite loss")
            loss = cast(torch.Tensor, output.loss)
            torch.autograd.backward(loss)
            if not any(parameter.grad is not None for parameter in parameters):
                raise RuntimeError("identity adapter received no gradients")
            optimizer.step()
            steps += 1
            remaining_steps -= 1
            target_tokens += batch.target_tokens
            final_loss = float(output.loss.detach().item())
        if not remaining_steps:
            break
    if not steps:
        raise RuntimeError("identity adapter training completed no optimizer steps")
    manifest = bank.promote_candidate(output_dir)
    base_after = bank.base_parameter_sha256()
    if base_after != base_before:
        raise RuntimeError("identity adapter training modified frozen base bytes")
    return IdentityTrainingResult(
        manifest=manifest,
        base_sha256_before=base_before,
        base_sha256_after=base_after,
        optimizer_steps=steps,
        target_tokens=target_tokens,
        final_loss=final_loss,
    )

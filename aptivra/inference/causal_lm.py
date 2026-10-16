"""Standard stock-Llama plus explicit PEFT identity-adapter chat inference."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Optional, Union, cast

import torch
from transformers import LlamaForCausalLM

from aptivra.experts.standard import PeftExpertBank, StandardExpertSettings
from aptivra.inference.chat_canonical import IdentityManifest, default_system_message
from aptivra.inference.chat_prompt import (
    IM_END,
    ChatTokenizer,
    IdentitySettings,
    format_chat_prompt,
    strip_assistant_turn_delimiter,
    validate_chat_tokenizer,
)
from aptivra.inference.standard_base import (
    GenerationResult,
    StandardBase,
    generate_task_answer,
)

# Cosmic Ray treats postponed PEP 604 annotations as executable binary
# expressions, creating equivalent arithmetic mutants.  The classic typing
# spellings keep the public types precise without generating inert mutants.


@dataclass
class AptivraCausalLM:
    """Explicit identity-adapter view over a verified stock Transformers base."""

    base: StandardBase
    expert_bank: PeftExpertBank
    identity_settings: IdentitySettings
    identity_manifest: IdentityManifest

    def __post_init__(self) -> None:
        if not self.identity_settings.feature.enabled:
            raise RuntimeError("ADR-0011 identity/chat path is disabled by configuration")
        adapter_name = self.identity_settings.training.adapter_name
        if adapter_name not in self.expert_bank.adapter_names:
            raise ValueError("identity adapter is not loaded in the PEFT expert bank")
        validate_chat_tokenizer(
            cast(ChatTokenizer, self.base.tokenizer),
            self.identity_settings.template,
        )
        self.expert_bank.activate(adapter_name)

    @property
    def device(self) -> torch.device:
        return self.base.device

    def eval(self) -> AptivraCausalLM:
        self.expert_bank.model.eval()
        return self

    def forward(self, **model_inputs: Any) -> Any:
        """Delegate to the stock PEFT-wrapped causal model."""
        return self.expert_bank.model(**model_inputs)

    def generate_chat(
        self,
        user_message: str,
        *,
        history: tuple[tuple[str, str], ...] = (),
        max_new_tokens: Optional[int] = None,  # noqa: UP045
    ) -> GenerationResult:
        """Generate one assistant turn, stopping on complete message close or EOS."""
        self.expert_bank.activate(self.identity_settings.training.adapter_name)
        prompt = format_chat_prompt(
            history,
            user_message,
            system_message=default_system_message(self.identity_manifest),
        )
        adapted_base = replace(
            self.base,
            model=cast(LlamaForCausalLM, self.expert_bank.model),
        )
        result = generate_task_answer(
            adapted_base,
            prompt,
            max_new_tokens=max_new_tokens,
            stop_sequences=(IM_END,),
        )
        return replace(result, text=strip_assistant_turn_delimiter(result.text))


def load_identity_chat_model(
    base: StandardBase,
    expert_settings: StandardExpertSettings,
    identity_settings: IdentitySettings,
    identity_manifest: IdentityManifest,
    adapter_artifact: Union[Path, str],  # noqa: UP007
) -> AptivraCausalLM:
    """Load a content-verified standard PEFT identity adapter for local inference."""
    from aptivra.experts.standard import load_promoted_expert

    bank = load_promoted_expert(base, expert_settings, adapter_artifact)
    return AptivraCausalLM(base, bank, identity_settings, identity_manifest)

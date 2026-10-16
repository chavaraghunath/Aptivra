"""Manifest-generated multi-turn identity conversation task family."""

from __future__ import annotations

from typing import Optional

from aptivra.experts.standard import StandardExpertSettings
from aptivra.inference.chat_canonical import (
    IdentityManifest,
    canonical_identity_records,
    default_system_message,
    load_identity_manifest,
)
from aptivra.inference.chat_prompt import CHAT_TEMPLATE_VERSION, format_chat_prompt
from aptivra.tasks.base import (
    MechanicalVerifier,
    TaskFamily,
    TaskInstance,
    VerifierResult,
    exact_match_result,
)

# Cosmic Ray treats postponed PEP 604 annotations as executable binary
# expressions, creating equivalent arithmetic mutants.  Classic Optional
# preserves the public types while keeping the mutation inventory behavioral.


class ConversationalVerifier(MechanicalVerifier):
    """Exact unsanitized verifier for one manifest-bound assistant turn."""

    def verify(self, instance: TaskInstance, model_output: str) -> VerifierResult:
        return exact_match_result(instance, model_output)


class ConversationalTaskFamily(TaskFamily):
    """Deterministic multi-turn records built only from canonical identity records."""

    def __init__(
        self,
        manifest: Optional[IdentityManifest] = None,  # noqa: UP045
        *,
        expert_settings: Optional[StandardExpertSettings] = None,  # noqa: UP045
    ) -> None:
        super().__init__(
            family_id="conversational_dialogue",
            name="Manifest-Bound Multi-Turn Conversation",
            description="Multi-turn identity dialogue with no facts beyond the manifest.",
        )
        self.manifest = manifest or load_identity_manifest()
        self.expert_settings = expert_settings
        self._verifier = ConversationalVerifier()

    @property
    def verifier(self) -> MechanicalVerifier:
        return self._verifier

    def generate_instances(self, count: int, seed: int = 42) -> list[TaskInstance]:
        """Generate a prior canonical turn followed by another canonical question."""
        records = canonical_identity_records(
            self.manifest,
            expert_settings=self.expert_settings,
        )
        system_message = default_system_message(self.manifest)
        instances: list[TaskInstance] = []
        for index in range(count):
            prior = records[(seed + index) % len(records)]
            current = records[(seed + index + 1) % len(records)]
            prompt = format_chat_prompt(
                [(prior.question, prior.answer)],
                current.question,
                system_message=system_message,
            )
            instances.append(
                TaskInstance(
                    task_id=f"conv_dialogue_{index}",
                    task_family=self.family_id,
                    prompt=prompt,
                    target_solution=current.answer,
                    metadata={
                        "seed": seed,
                        "turn": index,
                        "prompt_intent": current.prompt_intent,
                        "template_version": CHAT_TEMPLATE_VERSION,
                        "identity_manifest_sha256": self.manifest.content_sha256,
                    },
                )
            )
        return instances

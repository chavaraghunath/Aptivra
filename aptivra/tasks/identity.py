"""Evidence-generated Aptivra identity task family."""

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


class ModelIdentityVerifier(MechanicalVerifier):
    """Exact unsanitized verifier for evidence-generated identity answers."""

    def verify(self, instance: TaskInstance, model_output: str) -> VerifierResult:
        return exact_match_result(instance, model_output)


class ModelIdentityTaskFamily(TaskFamily):
    """Deterministic identity records generated from one validated manifest."""

    def __init__(
        self,
        manifest: Optional[IdentityManifest] = None,  # noqa: UP045
        *,
        expert_settings: Optional[StandardExpertSettings] = None,  # noqa: UP045
    ) -> None:
        super().__init__(
            family_id="model_identity_capabilities",
            name="Model Self-Identity and Evidence-Bound Capabilities",
            description=(
                "Manifest-bound Aptivra identity, base provenance, limitations, and "
                "release-evidence scope."
            ),
        )
        self.manifest = manifest or load_identity_manifest()
        self.expert_settings = expert_settings
        self._verifier = ModelIdentityVerifier()

    @property
    def verifier(self) -> MechanicalVerifier:
        return self._verifier

    def generate_instances(self, count: int, seed: int = 42) -> list[TaskInstance]:
        """Generate exact template prompts and targets without fabricated claims."""
        records = canonical_identity_records(
            self.manifest,
            expert_settings=self.expert_settings,
        )
        system_message = default_system_message(self.manifest)
        instances: list[TaskInstance] = []
        for index in range(count):
            record = records[(seed + index) % len(records)]
            instances.append(
                TaskInstance(
                    task_id=f"identity_{index}",
                    task_family=self.family_id,
                    prompt=format_chat_prompt(
                        [],
                        record.question,
                        system_message=system_message,
                    ),
                    target_solution=record.answer,
                    metadata={
                        "seed": seed,
                        "index": index,
                        "prompt_intent": record.prompt_intent,
                        "template_version": CHAT_TEMPLATE_VERSION,
                        "identity_manifest_sha256": self.manifest.content_sha256,
                    },
                )
            )
        return instances

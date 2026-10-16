# ruff: noqa: E501
"""Multi-Turn Conversational Task Family for Aptivra Dialogue Training."""

from aptivra.tasks.base import (
    MechanicalVerifier,
    TaskFamily,
    TaskInstance,
    VerifierResult,
    exact_match_result,
)


class ConversationalVerifier(MechanicalVerifier):
    """Mechanical verifier for multi-turn conversational dialog instances."""

    def verify(self, instance: TaskInstance, model_output: str) -> VerifierResult:
        return exact_match_result(instance, model_output)


class ConversationalTaskFamily(TaskFamily):
    """Task Family for multi-turn conversational interactions and context retention."""

    def __init__(self) -> None:
        super().__init__(
            family_id="conversational_dialogue",
            name="Multi-Turn Conversational Dialogue",
            description="Multi-turn interactive dialogue with context memory, clarification, and follow-ups.",
        )
        self._verifier = ConversationalVerifier()

    @property
    def verifier(self) -> MechanicalVerifier:
        return self._verifier

    def generate_instances(self, count: int, seed: int = 42) -> list[TaskInstance]:
        """Generates deterministic multi-turn conversational task instances."""
        instances: list[TaskInstance] = []

        dialog_templates = [
            (
                "User: Hello, can you help me write a Python function for binary search?\nAssistant: Sure! Binary search requires a sorted list. Here is the implementation:\nUser: Can you make it recursive?\nAssistant:",
                "def binary_search_recursive(arr, low, high, x):\n    if high >= low:\n        mid = (high + low) // 2\n        if arr[mid] == x: return mid\n        elif arr[mid] > x: return binary_search_recursive(arr, low, mid - 1, x)\n        else: return binary_search_recursive(arr, mid + 1, high, x)\n    else: return -1",
            ),
            (
                "User: What is the difference between PyTorch MoE and dense transformers?\nAssistant: MoE activates a subset of expert layers per token, reducing compute while scaling parameters.\nUser: How does Gating Router select experts?\nAssistant:",
                "Gating Router computes softmax probabilities over expert projection weights and selects top-k indices.",
            ),
            (
                "User: I have a bug in my database query script.\nAssistant: Please share the error traceback or query code.\nUser: Here is the query: SELECT * FROM users WHERE age > 'thirty'\nAssistant:",
                "The error occurs because string 'thirty' is passed to numeric column age. Use integer 30: SELECT * FROM users WHERE age > 30;",
            ),
            (
                "User: Explain Tiered Memory Runtime in Aptivra.\nAssistant: TMR is a C-engine execution ABI that shards MoE expert weights for low-latency NVMe/RAM offloading.\nUser: What is its throughput?\nAssistant:",
                "TMR achieves 3,142.21 MB/s shard export throughput and 433.25 MB/s load throughput.",
            ),
        ]

        for i in range(count):
            tpl_idx = (seed + i) % len(dialog_templates)
            prompt, target = dialog_templates[tpl_idx]
            instances.append(
                TaskInstance(
                    task_id=f"conv_dialogue_{i}",
                    task_family=self.family_id,
                    prompt=prompt,
                    target_solution=target,
                    metadata={"seed": seed, "turn": i},
                )
            )

        return instances

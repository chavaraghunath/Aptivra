"""Unit tests for Model Identity and Capabilities Task Family."""

from aptivra.tasks.identity import ModelIdentityTaskFamily


def test_model_identity_task_family() -> None:
    family = ModelIdentityTaskFamily()
    assert family.family_id == "model_identity_capabilities"

    instances = family.generate_instances(count=6, seed=42)
    assert len(instances) == 6

    res = family.verifier.verify(instances[1], instances[1].target_solution)
    assert res.passed is True

    garbled = family.verifier.verify(instances[1], "I >3d created by Raghunath Chava.")
    assert garbled.passed is False

    keyword_near_miss = family.verifier.verify(
        instances[1], "Aptivra is a dense MoE model with tiered memory."
    )
    assert keyword_near_miss.passed is False

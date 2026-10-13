"""Unit tests for Model Identity and Capabilities Task Family."""

from aptivra.tasks.identity import ModelIdentityTaskFamily


def test_model_identity_task_family() -> None:
    family = ModelIdentityTaskFamily()
    assert family.family_id == "model_identity_capabilities"

    instances = family.generate_instances(count=6, seed=42)
    assert len(instances) == 6

    # Test creator verification
    res = family.verifier.verify(instances[1], "I was created by Raghunath Chava.")
    assert res.passed is True

    # Test model name verification
    res_name = family.verifier.verify(instances[0], "My name is Aptivra.")
    assert res_name.passed is True

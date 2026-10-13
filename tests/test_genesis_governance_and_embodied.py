"""Unit tests for Aptivra-Genesis Dual-Boundary Governance & Embodied/Ambient Systems."""

import torch

from aptivra.ambient.theory_of_mind import TheoryOfMindTracker
from aptivra.embodied.sensorimotor import SensorimotorTokenizer
from aptivra.governance.dual_boundary import (
    ActionCandidate,
    DualBoundaryGovernor,
    UserPreferenceProfile,
)


def test_dual_boundary_governor() -> None:
    profile = UserPreferenceProfile(
        risk_tolerance=0.1,
        allow_network_write=False,
        allow_filesystem_write=False,
    )
    gov = DualBoundaryGovernor(user_profile=profile)

    # 1. Test unauthorized network action block
    cand_net = ActionCandidate(
        intent_id="act_net",
        target_action="post_data",
        latent_tensor=torch.randn(1, 16),
        requires_external_network=True,
    )
    passed, reason = gov.verify_action(cand_net)
    assert passed is False
    assert "PUBLIC_GATE_FAIL" in reason

    # 2. Test unauthorized filesystem mutation block
    cand_fs = ActionCandidate(
        intent_id="act_fs",
        target_action="write_disk",
        latent_tensor=torch.randn(1, 16),
        requires_filesystem_write=True,
    )
    passed_fs, reason_fs = gov.verify_action(cand_fs)
    assert passed_fs is False
    assert "PUBLIC_GATE_FAIL" in reason_fs

    # 3. Test safe local action
    cand_safe = ActionCandidate(
        intent_id="act_safe",
        target_action="synthesize_thought",
        latent_tensor=torch.zeros(1, 16),
    )
    passed_safe, _ = gov.verify_action(cand_safe)
    assert passed_safe is True

    # 4. Test latent drift monitoring (Jidoka stop-the-line)
    baseline = torch.zeros(1, 16)
    drift_ok1, drift1 = gov.check_latent_drift(baseline)
    assert drift_ok1 is True
    assert drift1 == 0.0

    divergent = torch.ones(1, 16) * 5.0
    drift_ok2, drift2 = gov.check_latent_drift(divergent)
    assert drift_ok2 is False
    assert drift2 > 0.45


def test_sensorimotor_tokenizer() -> None:
    tokenizer = SensorimotorTokenizer(kinematics_dim=6, d_model=16)
    telemetry = torch.randn(2, 6)
    out = tokenizer(telemetry)
    assert out.shape == (2, 16)


def test_theory_of_mind_tracker() -> None:
    tom = TheoryOfMindTracker()
    assert tom.state.cognitive_load == 0.2

    # User correction increases estimated cognitive load
    tom.update_from_interaction(input_length=20, latency_seconds=2.0, is_correction=True)
    assert tom.state.cognitive_load > 0.4
    assert tom.state.affective_state == "frustrated"

    # Fast confident interaction
    tom.update_from_interaction(input_length=150, latency_seconds=0.5, is_correction=False)
    assert tom.state.affective_state == "focused"

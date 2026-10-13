"""Rigorous mathematical and empirical verification of Genesis Phase 2 and Phase 3."""

import torch

from aptivra.ambient.peer_sync import CrossDeviceSyncProtocol
from aptivra.embodied.kinematic_actuator import KinematicActuatorEngine
from aptivra.memory.semantic_graph import CausalEdge, ConceptNode, SemanticGraphCore
from aptivra.research.auto_experimenter import AutoExperimenterEngine
from aptivra.research.hypothesis_engine import HypothesisCandidate, HypothesisEngine


def test_phase2_kinematic_actuator_rodrigues_so3_orthogonality() -> None:
    """Verifies that Rodrigues exponential mapping generates strictly valid SO(3) matrices."""
    engine = KinematicActuatorEngine(num_joints=12, d_model=32)
    kinematic_state = torch.randn(1, 12)
    target_latent = torch.randn(1, 32)

    cmd = engine.compute_actuation(kinematic_state, target_latent)

    # 1. Torque saturation bounds
    assert torch.all(cmd.joint_torques >= -100.0)
    assert torch.all(cmd.joint_torques <= 100.0)

    # 2. SE(3) Homogeneous Matrix structure: shape (4, 4)
    h_mat = cmd.target_pose_se3[0]
    assert h_mat.shape == (4, 4)
    assert float(h_mat[3, 3].item()) == 1.0
    assert torch.allclose(h_mat[3, :3], torch.zeros(3), atol=1e-5)

    # 3. SO(3) Orthogonality: R @ R.T == I and det(R) == 1.0
    r_mat = h_mat[:3, :3]
    eye = torch.eye(3)
    assert torch.allclose(r_mat @ r_mat.T, eye, atol=1e-4)
    det_r = float(torch.det(r_mat).item())
    assert abs(det_r - 1.0) < 1e-4


def test_phase2_p2p_sync_hmac_cryptographic_verification() -> None:
    """Verifies HMAC-SHA256 authenticated synchronization, tampering rejection, and clock skew."""
    sync_proto = CrossDeviceSyncProtocol(local_device_id="phone_node")
    sync_proto.register_peer("glasses_node", "glasses", "SECRET_KEY_FINGERPRINT_12345")

    records = [{"record_id": "mem_001", "content": "Visual perception of blue mug"}]
    packet = sync_proto.create_differential_sync_packet("glasses_node", records, timestamp=1000.0)
    assert packet is not None
    assert packet.encrypted_payload.startswith("ENC_ZK_HMAC:")

    # 1. Valid application on recipient
    receiver_proto = CrossDeviceSyncProtocol(local_device_id="glasses_node")
    receiver_proto.register_peer("phone_node", "phone", "SECRET_KEY_FINGERPRINT_12345")
    ok, count = receiver_proto.receive_and_apply_packet(packet, current_time=1050.0)
    assert ok is True
    assert count == 1
    assert "mem_001" in receiver_proto.peers["phone_node"].synced_record_ids

    # 2. Tampered payload rejection
    tampered_packet = packet
    tampered_packet.encrypted_payload = packet.encrypted_payload.replace("blue", "red")
    ok_tampered, _ = receiver_proto.receive_and_apply_packet(tampered_packet)
    assert ok_tampered is False

    # 3. Excessive clock skew rejection
    ok_skew, _ = receiver_proto.receive_and_apply_packet(
        packet, current_time=2000.0, max_clock_skew=300.0
    )
    assert ok_skew is False


def test_phase3_hypothesis_engine_multi_tier_abduction() -> None:
    """Verifies contradiction detection, transitive deduction, and analogy generation."""
    graph = SemanticGraphCore()
    c_a = ConceptNode("heat", "Heat", torch.tensor([1.0, 0.0, 0.0]))
    c_b = ConceptNode("expansion", "Expansion", torch.tensor([0.9, 0.1, 0.0]))
    c_c = ConceptNode("pressure", "Pressure", torch.tensor([0.8, 0.2, 0.0]))

    graph.add_concept(c_a)
    graph.add_concept(c_b)
    graph.add_concept(c_c)

    # Add transitive chain: heat -> expansion -> pressure
    graph.add_relation(CausalEdge("heat", "expansion", "causes", strength=0.95))
    graph.add_relation(CausalEdge("expansion", "pressure", "causes", strength=0.90))

    # Add contradiction on heat & expansion
    graph.add_relation(CausalEdge("heat", "expansion", "inhibits", strength=0.80))

    engine = HypothesisEngine(semantic_graph=graph)
    hypotheses = engine.discover_hypotheses()

    types = [h.hypothesis_type for h in hypotheses]
    assert "contradiction" in types
    assert "transitive_causality" in types

    transitive_hyp = next(h for h in hypotheses if h.hypothesis_type == "transitive_causality")
    assert transitive_hyp.target_concept == "pressure"
    assert "influences" in transitive_hyp.premise


def test_phase3_auto_experimenter_sandbox_execution() -> None:
    """Verifies that AutoExperimenter runs real sandboxed Python test code instead of a stub."""
    graph = SemanticGraphCore()
    c_energy = ConceptNode("energy", "Energy", torch.ones(32))
    graph.add_concept(c_energy)

    experimenter = AutoExperimenterEngine(semantic_graph=graph)
    hyp = HypothesisCandidate(
        hypothesis_id="hyp_kinetic_test",
        premise="E = 0.5 * m * v^2",
        proposed_relation="computational_law",
        target_concept="energy",
        prior_confidence=0.85,
        latent_embedding=torch.randn(32),
        hypothesis_type="computational",
    )

    outcome = experimenter.conduct_experiment(hyp)

    assert outcome.verified is True
    assert outcome.confidence_delta > 0.0
    assert outcome.execution_time_ms >= 0.0
    assert "Passed 4/4 empirical invariant test vectors" in outcome.empirical_evidence
    assert outcome.promoted_to_semantic is True
    assert "verify_kinetic_test" in experimenter.tool_synthesizer.registry

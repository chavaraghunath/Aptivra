"""End-to-End Test Suite for all 4 Developmental Phases of Aptivra-Genesis."""

import torch

from aptivra.agents.tool_synthesizer import ToolSynthesizerEngine
from aptivra.ambient.lifelong_keeper import LifelongNarrativeKeeper
from aptivra.ambient.peer_sync import CrossDeviceSyncProtocol
from aptivra.benchmarks.aid_bench import AIDevelopmentalBenchmark
from aptivra.embodied.kinematic_actuator import KinematicActuatorEngine
from aptivra.federation.collective_mind import FederatedMindNode
from aptivra.genesis import AptivraGenesis
from aptivra.governance.audit_kernel import MerkleAuditKernel
from aptivra.governance.drift_auditor import AutonomousDriftAuditor
from aptivra.model.config import ModelConfig
from aptivra.planning.counterfactual import CounterfactualPlanner
from aptivra.research.auto_experimenter import AutoExperimenterEngine
from aptivra.research.hypothesis_engine import HypothesisEngine


def test_phase2_embodied_and_ambient() -> None:
    # 1. Kinematic Actuator
    actuator = KinematicActuatorEngine(num_joints=6, d_model=16)
    curr_kin = torch.randn(2, 6)
    target_goal = torch.randn(2, 16)
    cmd = actuator.compute_actuation(curr_kin, target_goal)
    assert cmd.joint_torques.shape == (2, 6)
    assert cmd.target_pose_se3.shape == (2, 4, 4)

    # 2. Peer Sync Protocol
    sync = CrossDeviceSyncProtocol(local_device_id="phone_01")
    sync.register_peer("workstation_01", "workstation", "PUB_FP_01")
    records = [{"record_id": "r1", "concept": "gravity"}]
    packet = sync.create_differential_sync_packet("workstation_01", records)
    assert packet is not None
    assert packet.is_verified is True

    # 3. Counterfactual Planner
    planner = CounterfactualPlanner(world_model=actuator.world_model, max_horizon=5)
    s0 = torch.randn(1, 16)
    actions = [torch.randn(1, 6) for _ in range(3)]
    goal = torch.zeros(1, 16)
    traj = planner.evaluate_what_if_rollout(s0, actions, goal)
    assert len(traj.predicted_states) == 3
    assert traj.goal_alignment_score >= 0.0


def test_phase3_autonomous_research_and_tools() -> None:
    # 1. Dynamic Tool Synthesis
    tool_synth = ToolSynthesizerEngine()
    code = "def multiply(a, b):\n    return a * b\n"
    tool_synth.synthesize_tool("multiply", "(a, b) -> int", code)
    assert len(tool_synth) == 1
    assert tool_synth.execute_tool("multiply", 6, 7) == 42

    # 2. Hypothesis & Auto-Experimenter
    from aptivra.memory.semantic_graph import ConceptNode, SemanticGraphCore

    graph = SemanticGraphCore(embedding_dim=8)
    graph.add_concept(
        ConceptNode(
            concept_id="speed",
            label="Speed",
            prototype_vector=torch.ones(8),
        )
    )
    graph.add_concept(
        ConceptNode(
            concept_id="time",
            label="Time",
            prototype_vector=torch.ones(8) * 0.8,
        )
    )

    hyp_engine = HypothesisEngine(semantic_graph=graph)
    hyps = hyp_engine.discover_hypotheses()
    assert len(hyps) >= 1

    exp_engine = AutoExperimenterEngine(semantic_graph=graph)
    outcome = exp_engine.conduct_experiment(hyps[0])
    assert outcome.verified is True
    assert outcome.promoted_to_semantic is True

    # 3. Drift Auditor
    auditor = AutonomousDriftAuditor(max_allowed_divergence=0.40)
    auditor.register_goal_anchor("task_01", torch.zeros(1, 8))
    report = auditor.audit_step("task_01", torch.zeros(1, 8), step_number=1)
    assert report.is_safe_to_continue is True


def test_phase4_collective_mind_and_merkle_audit() -> None:
    # 1. Federated Mind
    node1 = FederatedMindNode(node_id="node_us_east", epsilon=1.0)
    insight = node1.export_private_insight("climate_model", torch.randn(10))
    node1.receive_contribution(insight)
    consensus = node1.aggregate_consensus("climate_model")
    assert consensus is not None
    assert consensus.participating_nodes_count == 1
    assert consensus.reliability_index >= 0.5

    # 2. Lifelong Keeper
    keeper = LifelongNarrativeKeeper(d_model=16)
    keeper.compress_epoch(
        epoch_id="epoch_1",
        start_time=0.0,
        end_time=10.0,
        milestones=["First Steps"],
        discovered_values=["Curiosity"],
        representative_latents=[torch.randn(16)],
    )
    assert len(keeper.epochs) == 1
    reflection = keeper.generate_symbiotic_reflection()
    assert "Curiosity" in reflection

    # 3. Merkle Audit Kernel
    kernel = MerkleAuditKernel()
    kernel.log_constitutional_event("WEIGHT_UPDATE", "SleepEngine", {"pruned": 100})
    block = kernel.commit_epoch_audit_block(timestamp=1.0)
    assert block.block_index == 1
    assert kernel.verify_chain_integrity() is True


def test_full_developmental_genesis_all_phases() -> None:
    cfg = ModelConfig(d_model=16, n_layers=2, n_heads=2, d_ff=32)
    agent = AptivraGenesis(config=cfg, node_id="agent_alpha")

    # Verify stage progression across all 6 developmental stages
    assert agent.developmental_stage == 0
    for stage_expected in range(1, 6):
        advanced = agent.advance_developmental_stage()
        assert advanced == stage_expected

    # Benchmark all 4 phases
    bench = AIDevelopmentalBenchmark()
    report = bench.evaluate_four_phases_e2e(agent)

    assert report.phase1_sensorimotor_score > 0.8
    assert report.phase1_sleep_consolidation_passed is True
    assert report.phase2_kinematic_control_fidelity > 0.8
    assert report.phase2_p2p_sync_success_rate == 1.0
    assert report.phase2_counterfactual_safety is True
    assert report.phase3_hypotheses_discovered >= 1
    assert report.phase3_tool_synthesis_verified is True
    assert report.phase3_drift_audit_passed is True
    assert report.phase4_collective_consensus_reliability > 0.5
    assert report.phase4_lifelong_epochs_recorded >= 1
    assert report.phase4_merkle_audit_integrity is True
    assert report.overall_developmental_quotient == 1.0

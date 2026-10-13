"""Milestone Genesis: Unified Cognitive Operating System Architecture Across All 4 Phases."""

from typing import Any

import torch
import torch.nn as nn

from aptivra.agents.tool_synthesizer import ToolSynthesizerEngine
from aptivra.ambient.lifelong_keeper import LifelongNarrativeKeeper
from aptivra.ambient.peer_sync import CrossDeviceSyncProtocol
from aptivra.ambient.theory_of_mind import TheoryOfMindTracker
from aptivra.consolidation.sleep_cycle import GenerativeSleepEngine
from aptivra.embodied.kinematic_actuator import (
    KinematicActuatorEngine,
    KinematicCommand,
)
from aptivra.embodied.sensorimotor import SensorimotorTokenizer
from aptivra.federation.collective_mind import FederatedMindNode
from aptivra.governance.audit_kernel import MerkleAuditKernel
from aptivra.governance.drift_auditor import AutonomousDriftAuditor
from aptivra.governance.dual_boundary import (
    ActionCandidate,
    DualBoundaryGovernor,
    UserPreferenceProfile,
)
from aptivra.learning.predictive_coding import LocalPredictiveLayer
from aptivra.memory.hexa_store import HexaTierMemoryStore
from aptivra.model.config import ModelConfig
from aptivra.planning.active_inference import ActiveInferencePlanner
from aptivra.planning.counterfactual import CounterfactualPlanner
from aptivra.research.auto_experimenter import AutoExperimenterEngine
from aptivra.research.hypothesis_engine import HypothesisEngine
from aptivra.world_model.jepa import LatentWorldModel


class AptivraGenesis(nn.Module):
    """Unified Aptivra-Genesis Cognitive Operating System spanning all 4 Developmental Phases.

    Phases & Stages:
      - Phase 1 (Infant): Sensorimotor exploration, Hexa-Tier Memory, Sleep Consolidation.
      - Phase 2 (Child/Adolescent): Kinematic control, Counterfactual planning, P2P Sync.
      - Phase 3 (Adult/Expert): Autonomous research, Tool synthesis, Long-horizon drift audit.
      - Phase 4 (Master): Federated collective intelligence, Lifelong narrative, Merkle audit.
    """

    def __init__(
        self,
        config: ModelConfig | None = None,
        user_profile: UserPreferenceProfile | None = None,
        node_id: str = "node_local_001",
    ) -> None:
        super().__init__()
        self.config = config or ModelConfig()
        self.device = self.config.get_device()
        self.node_id = node_id

        # --- PHASE 1: COGNITIVE SUBSTRATE ---
        self.hexa_memory = HexaTierMemoryStore(
            embedding_dim=self.config.d_model, device=self.device
        )
        self.world_model = LatentWorldModel(latent_dim=self.config.d_model, action_dim=64).to(
            self.device
        )
        self.planner = ActiveInferencePlanner(self.world_model)
        self.governor = DualBoundaryGovernor(user_profile=user_profile)
        self.theory_of_mind = TheoryOfMindTracker()
        self.sensorimotor = SensorimotorTokenizer(
            kinematics_dim=12, d_model=self.config.d_model
        ).to(self.device)
        self.predictive_layer = LocalPredictiveLayer(
            in_features=self.config.d_model, out_features=self.config.d_model
        ).to(self.device)
        self.sleep_engine = GenerativeSleepEngine(
            model=self, hexa_store=self.hexa_memory, config=self.config
        )

        # --- PHASE 2: EMBODIED CHILD & AMBIENT MESH ---
        self.kinematic_actuator = KinematicActuatorEngine(
            num_joints=12, d_model=self.config.d_model, world_model=self.world_model
        ).to(self.device)
        self.peer_sync = CrossDeviceSyncProtocol(local_device_id=node_id)
        self.counterfactual_planner = CounterfactualPlanner(
            world_model=self.world_model, max_horizon=10
        )

        # --- PHASE 3: AUTONOMOUS ADULT & SCIENTIFIC RESEARCHER ---
        self.hypothesis_engine = HypothesisEngine(semantic_graph=self.hexa_memory.semantic_graph)
        self.auto_experimenter = AutoExperimenterEngine(
            semantic_graph=self.hexa_memory.semantic_graph
        )
        self.tool_synthesizer = ToolSynthesizerEngine()
        self.drift_auditor = AutonomousDriftAuditor()

        # --- PHASE 4: SYMBIOTIC MASTER INTELLIGENCE ---
        self.federated_mind = FederatedMindNode(node_id=node_id, epsilon=1.0)
        self.lifelong_keeper = LifelongNarrativeKeeper(d_model=self.config.d_model)
        self.audit_kernel = MerkleAuditKernel()

        # Stage: 0=Infant, 1=Child, 2=Adolescent, 3=Adult, 4=Expert, 5=Master
        self.developmental_stage = 0

    def step(self, observation: str) -> dict[str, Any]:
        """Executes sensory-cognitive-action step with multi-stage capability branching."""
        # 1. Update user mental state
        self.theory_of_mind.update_from_interaction(
            input_length=len(observation), latency_seconds=0.5, is_correction=False
        )

        # 2. Formulate candidate action conditioned on observation
        obs_val = float(sum(ord(c) for c in observation) % 100) / 100.0
        action_latent = torch.full((1, self.config.d_model), obs_val, device=self.device)
        candidate = ActionCandidate(
            intent_id=f"act_genesis_{self.developmental_stage}",
            target_action="generate_response",
            latent_tensor=action_latent,
        )

        # 3. Dual-Boundary Safety Interlock Verification
        verified, reason = self.governor.verify_action(candidate)
        if not verified:
            self.audit_kernel.log_constitutional_event(
                event_type="SAFETY_INTERLOCK_BLOCKED",
                actor="DualBoundaryGovernor",
                payload={"reason": reason, "stage": self.developmental_stage},
            )
            return {"status": "BLOCKED_BY_SAFETY", "reason": reason}

        # 4. Latent representation drift check
        drift_ok, drift_score = self.governor.check_latent_drift(action_latent)
        if not drift_ok:
            return {
                "status": "HALTED_BY_JIDOKA",
                "reason": f"Latent drift threshold exceeded: {drift_score:.3f}",
            }

        # 5. Log verified constitutional action to Merkle chain
        self.audit_kernel.log_constitutional_event(
            event_type="ACTION_EXECUTED",
            actor="AptivraGenesis",
            payload={"stage": self.developmental_stage, "obs_len": len(observation)},
        )

        stage_names = ["Infant", "Child", "Adolescent", "Adult", "Expert", "Master"]
        stage_name = stage_names[min(self.developmental_stage, 5)]

        out_msg = (
            f"Aptivra-Genesis [Stage {self.developmental_stage}: {stage_name}] "
            f"processed observation: {observation[:60]}"
        )
        return {
            "status": "SUCCESS",
            "stage": self.developmental_stage,
            "stage_name": stage_name,
            "output": out_msg,
        }

    def execute_physical_trajectory(
        self, current_kinematics: torch.Tensor, goal_latent: torch.Tensor
    ) -> KinematicCommand:
        """Executes Phase 2 closed-loop physical actuation."""
        return self.kinematic_actuator.compute_actuation(current_kinematics, goal_latent)

    def trigger_sleep_consolidation(self) -> dict[str, Any]:
        """Triggers offline generative sleep replay and synaptic pruning."""
        return self.sleep_engine.execute_sleep_cycle()

    def advance_developmental_stage(self) -> int:
        """Advances the model along the developmental trajectory (Infant -> Master)."""
        if self.developmental_stage < 5:
            self.developmental_stage += 1
            self.audit_kernel.log_constitutional_event(
                event_type="STAGE_ADVANCED",
                actor="DevelopmentalController",
                payload={"new_stage": self.developmental_stage},
            )
        return self.developmental_stage

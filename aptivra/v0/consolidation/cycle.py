"""Consolidation Cycle execution with transactional promotion rollback."""

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, override

import numpy as np
import torch
import torch.nn as nn

from aptivra.baselines.base import BaselineTrainer
from aptivra.memory.store import EpisodicRecord, EpisodicStore
from aptivra.model.config import ModelConfig
from aptivra.tasks.base import TaskInstance
from aptivra.v0.experts.expert import ExpertModule
from aptivra.v0.experts.router import GatingRouter
from aptivra.v0.experts.spawner import ExpertSpawner


@dataclass
class ConsolidationConfig:
    """Configuration and ablation toggles for Aptivra consolidation cycle."""

    use_t1_memory: bool = True
    use_expert_growth: bool = True
    use_replay_buffer: bool = True
    use_ewc_penalty: bool = True
    use_promotion_gate: bool = True
    lr: float = 1e-4
    epochs: int = 2
    ewc_lambda: float = 400.0


class ConsolidationCycle(BaselineTrainer):
    """Executes targeted consolidation training passes over memory, replay, and experts."""

    def __init__(
        self,
        model_config: ModelConfig,
        consolidation_config: ConsolidationConfig | None = None,
    ) -> None:
        super().__init__(model_config)
        self.cons_config = consolidation_config or ConsolidationConfig()

        self.t1_store = EpisodicStore(embedding_dim=64, device=self.device)
        self.router = GatingRouter(d_model=model_config.d_model, initial_num_experts=1).to(
            self.device
        )
        self.experts = nn.ModuleList(
            [ExpertModule(expert_id=0, d_model=model_config.d_model).to(self.device)]
        )
        self.spawner = ExpertSpawner()

        self.replay_buffer: list[TaskInstance] = []
        self.fisher_matrices: list[dict[str, torch.Tensor]] = []
        self.optpar_matrices: list[dict[str, torch.Tensor]] = []

        params = (
            list(self.model.parameters())
            + list(self.router.parameters())
            + list(self.experts.parameters())
        )
        self.optimizer = torch.optim.AdamW(params, lr=self.cons_config.lr)
        self.criterion = nn.CrossEntropyLoss(ignore_index=-100)

    def _reset_optimizer(self) -> None:
        """Rebuild the optimizer against the currently active module parameters."""
        params = (
            list(self.model.parameters())
            + list(self.router.parameters())
            + list(self.experts.parameters())
        )
        self.optimizer = torch.optim.AdamW(params, lr=self.cons_config.lr)

    def save_promotion_snapshot(self, path: Path) -> None:
        """Persist every learned state component at the promotion boundary.

        ``total_flops`` is intentionally excluded: rejected training still consumed compute.
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        expert_metadata = [
            {
                "expert_id": expert.expert_id,
                "d_expert": expert.d_expert,
                "is_mature": expert.is_mature,
            }
            for expert in self.experts
            if isinstance(expert, ExpertModule)
        ]
        torch.save(
            {
                "model_state": self.model.state_dict(),
                "router_state": self.router.state_dict(),
                "router_frozen_mask": self.router.frozen_mask,
                "router_top_k": self.router.top_k,
                "experts_state": self.experts.state_dict(),
                "expert_metadata": expert_metadata,
                "model_training": self.model.training,
                "router_training": self.router.training,
                "expert_training": [expert.training for expert in self.experts],
                "optimizer_state": self.optimizer.state_dict(),
                "replay_buffer": [self._task_to_dict(item) for item in self.replay_buffer],
                "fisher_matrices": self.fisher_matrices,
                "optpar_matrices": self.optpar_matrices,
                "t1_records": {
                    key: {
                        "record_id": record.record_id,
                        "task_family": record.task_family,
                        "prompt": record.prompt,
                        "target_solution": record.target_solution,
                        "verifier_passed": record.verifier_passed,
                        "retrieval_count": record.retrieval_count,
                        "verification_count": record.verification_count,
                        "metadata": record.metadata,
                        "answer_stop_characters": record.answer_stop_characters,
                        "training_terminator": record.training_terminator,
                    }
                    for key, record in self.t1_store.records.items()
                },
                "t1_embedding_dim": self.t1_store.index.embedding_dim,
                "t1_encoder_state": self.t1_store.index.encoder.state_dict(),
                "t1_encoder_training": self.t1_store.index.encoder.training,
                "t1_optimizer_state": self.t1_store.index.optimizer.state_dict(),
                "t1_vectors": self.t1_store.index.vectors,
                "t1_record_ids": self.t1_store.index.record_ids,
                "rng_state": self._capture_rng_state(),
            },
            path,
        )

    def restore_promotion_snapshot(self, path: Path) -> None:
        """Restore learned and RNG state while retaining compute already spent."""
        # The archive is decoded on CPU so host-owned data is never relocated as a
        # side effect. Module.load_state_dict copies parameters and buffers into each
        # module's existing device, and Optimizer.load_state_dict applies PyTorch's
        # parameter-aware policy (including keeping non-capturable step counters on CPU).
        snapshot: dict[str, Any] = torch.load(path, map_location="cpu", weights_only=True)
        expert_metadata: list[dict[str, Any]] = snapshot["expert_metadata"]
        self.model.load_state_dict(snapshot["model_state"])

        self.router = GatingRouter(
            d_model=self.config.d_model,
            initial_num_experts=len(expert_metadata),
            top_k=int(snapshot["router_top_k"]),
        ).to(self.device)
        self.router.load_state_dict(snapshot["router_state"])
        for expert_idx, is_frozen in enumerate(snapshot["router_frozen_mask"]):
            if is_frozen:
                self.router.freeze_expert_row(expert_idx)

        restored_experts = nn.ModuleList()
        for metadata in expert_metadata:
            expert = ExpertModule(
                expert_id=int(metadata["expert_id"]),
                d_model=self.config.d_model,
                d_expert=int(metadata["d_expert"]),
            ).to(self.device)
            if bool(metadata["is_mature"]):
                expert.freeze()
            restored_experts.append(expert)
        self.experts = restored_experts
        self.experts.load_state_dict(snapshot["experts_state"])
        self._reset_optimizer()
        self.optimizer.load_state_dict(snapshot["optimizer_state"])

        self.replay_buffer = [
            TaskInstance(**item) for item in snapshot["replay_buffer"]
        ]
        # EWC tensors participate directly in model-device arithmetic and therefore
        # need an explicit transfer after the deliberately CPU-only archive load.
        self.fisher_matrices = self._tensor_mappings_to_device(
            snapshot["fisher_matrices"]
        )
        self.optpar_matrices = self._tensor_mappings_to_device(
            snapshot["optpar_matrices"]
        )

        self.t1_store = EpisodicStore(
            embedding_dim=int(snapshot["t1_embedding_dim"]), device=self.device
        )
        self.t1_store.records = {
            key: EpisodicRecord(**record) for key, record in snapshot["t1_records"].items()
        }
        self.t1_store.index.encoder.load_state_dict(snapshot["t1_encoder_state"])
        self.t1_store.index.optimizer.load_state_dict(snapshot["t1_optimizer_state"])
        self.t1_store.index.vectors = snapshot["t1_vectors"].to(self.device)
        self.t1_store.index.record_ids = list(snapshot["t1_record_ids"])

        self.model.train(bool(snapshot["model_training"]))
        self.router.train(bool(snapshot["router_training"]))
        for expert_module, training in zip(
            self.experts, snapshot["expert_training"], strict=True
        ):
            expert_module.train(bool(training))
        self.t1_store.index.encoder.train(bool(snapshot["t1_encoder_training"]))

        self._restore_rng_state(snapshot["rng_state"])

    def save_weight_snapshot(self, path: Path) -> None:
        """Compatibility alias for the now-complete promotion snapshot."""
        self.save_promotion_snapshot(path)

    def restore_weight_snapshot(self, path: Path) -> None:
        """Compatibility alias for the now-complete promotion rollback."""
        self.restore_promotion_snapshot(path)

    @staticmethod
    def _task_to_dict(item: TaskInstance) -> dict[str, Any]:
        return {
            "task_id": item.task_id,
            "task_family": item.task_family,
            "prompt": item.prompt,
            "target_solution": item.target_solution,
            "metadata": item.metadata,
            "answer_stop_characters": item.answer_stop_characters,
            "training_terminator": item.training_terminator,
        }

    def _tensor_mappings_to_device(
        self, mappings: list[dict[str, torch.Tensor]]
    ) -> list[dict[str, torch.Tensor]]:
        return [
            {name: tensor.to(self.device) for name, tensor in mapping.items()}
            for mapping in mappings
        ]

    def _capture_rng_state(self) -> dict[str, Any]:
        """Capture every global generator that can affect a consolidation run.

        Python and NumPy are orchestration/task-generation globals. The torch CPU
        generator is used during module construction, while dropout and random
        tensors consume the active CUDA or MPS generator when the model uses that
        accelerator. No custom ``torch.Generator`` exists in this training path.

        Generator states are serialized as bytes or Python primitives, never tensors,
        so checkpoint device mapping cannot alter their meaning or storage device.
        Only the active accelerator generator is captured because other accelerator
        generators do not participate in this single-device execution path.
        """
        cuda_state: bytes | None = None
        mps_state: bytes | None = None
        if self.device.type == "cuda" and torch.cuda.is_available():
            cuda_state = self._torch_rng_state_to_bytes(
                torch.cuda.get_rng_state(self.device)
            )
        elif self.device.type == "mps" and torch.backends.mps.is_available():
            mps_state = self._torch_rng_state_to_bytes(torch.mps.get_rng_state())

        return {
            "python": random.getstate(),
            "numpy": self._numpy_rng_state(),
            "torch_cpu": self._torch_rng_state_to_bytes(torch.get_rng_state()),
            "torch_cuda": cuda_state,
            "torch_mps": mps_state,
        }

    def _restore_rng_state(self, state: dict[str, Any]) -> None:
        random.setstate(state["python"])
        self._set_numpy_rng_state(state["numpy"])
        torch.set_rng_state(self._torch_rng_state_from_bytes(state["torch_cpu"]))

        cuda_state: bytes | None = state["torch_cuda"]
        if (
            cuda_state is not None
            and self.device.type == "cuda"
            and torch.cuda.is_available()
        ):
            torch.cuda.set_rng_state(
                self._torch_rng_state_from_bytes(cuda_state), self.device
            )

        mps_state: bytes | None = state["torch_mps"]
        if (
            mps_state is not None
            and self.device.type == "mps"
            and torch.backends.mps.is_available()
        ):
            torch.mps.set_rng_state(self._torch_rng_state_from_bytes(mps_state))

    @staticmethod
    def _torch_rng_state_to_bytes(state: torch.Tensor) -> bytes:
        return state.detach().cpu().contiguous().numpy().tobytes()

    @staticmethod
    def _torch_rng_state_from_bytes(state: bytes) -> torch.Tensor:
        return torch.tensor(tuple(state), dtype=torch.uint8)

    @staticmethod
    def _numpy_rng_state() -> dict[str, Any]:
        algorithm, keys, position, has_gauss, cached_gaussian = np.random.get_state()
        return {
            "algorithm": algorithm,
            "keys": keys.tolist(),
            "position": position,
            "has_gauss": has_gauss,
            "cached_gaussian": cached_gaussian,
        }

    @staticmethod
    def _set_numpy_rng_state(state: dict[str, Any]) -> None:
        np.random.set_state(
            (
                state["algorithm"],
                np.asarray(state["keys"], dtype=np.uint32),
                state["position"],
                state["has_gauss"],
                state["cached_gaussian"],
            )
        )

    @override
    def train_task(self, task_idx: int, train_instances: list[TaskInstance]) -> dict[str, Any]:
        """Baseline interface compatibility wrapper."""
        task_fam = train_instances[0].task_family if train_instances else f"task_{task_idx}"
        return self.run_cycle(task_idx, task_fam, train_instances)

    def run_cycle(
        self,
        task_idx: int,
        task_family: str,
        train_instances: list[TaskInstance],
    ) -> dict[str, Any]:
        """Executes one consolidation training cycle for current task family."""
        self.model.train()
        self.router.train()
        for exp in self.experts:
            exp.train()

        total_loss = 0.0
        steps = 0

        # Construct consolidation training batch pool
        batch_pool = list(train_instances)

        # Mix T1 frequently retrieved verified records if enabled
        if self.cons_config.use_t1_memory:
            candidates = self.t1_store.get_consolidation_candidates(min_retrievals=1)
            for cand in candidates:
                batch_pool.append(
                    TaskInstance(
                        task_id=cand.record_id,
                        task_family=cand.task_family,
                        prompt=cand.prompt,
                        target_solution=cand.target_solution,
                        answer_stop_characters=cand.answer_stop_characters,
                        training_terminator=cand.training_terminator,
                    )
                )

        # Mix Replay buffer if enabled
        if self.cons_config.use_replay_buffer and self.replay_buffer:
            batch_pool.extend(self.replay_buffer[: len(train_instances) // 2])

        batch_size = 16
        for _ in range(self.cons_config.epochs):
            for i in range(0, len(batch_pool), batch_size):
                batch = batch_pool[i : i + batch_size]
                if not batch:
                    continue

                inputs_list: list[torch.Tensor] = []
                targets_list: list[torch.Tensor] = []

                for inst in batch:
                    p_toks = self.encode_text(inst.prompt)
                    s_toks = self.encode_training_target(inst)
                    full_seq = torch.cat([p_toks, s_toks])[: self.config.max_seq_len]

                    target_seq = full_seq.clone()
                    target_seq[: len(p_toks)] = -100

                    inputs_list.append(full_seq)
                    targets_list.append(target_seq)

                max_len = max(len(s) for s in inputs_list)
                padded_inputs = torch.zeros(
                    (len(batch), max_len), dtype=torch.long, device=self.device
                )
                padded_targets = torch.full(
                    (len(batch), max_len), -100, dtype=torch.long, device=self.device
                )

                for b_idx, (inp, tgt) in enumerate(zip(inputs_list, targets_list, strict=True)):
                    padded_inputs[b_idx, : len(inp)] = inp.to(self.device)
                    padded_targets[b_idx, : len(tgt)] = tgt.to(self.device)

                self.optimizer.zero_grad()
                logits, _ = self.model(padded_inputs)

                shift_logits = logits[:, :-1, :].contiguous()
                shift_targets = padded_targets[:, 1:].contiguous()
                ce_loss = self.criterion(
                    shift_logits.view(-1, self.config.vocab_size), shift_targets.view(-1)
                )

                loss = ce_loss

                # Calculate EWC penalty if enabled
                if self.cons_config.use_ewc_penalty and self.fisher_matrices:
                    ewc_pen = torch.tensor(0.0, device=self.device)
                    for fisher, optpar in zip(
                        self.fisher_matrices, self.optpar_matrices, strict=True
                    ):
                        for name, param in self.model.named_parameters():
                            if name in fisher and param.requires_grad:
                                ewc_pen += (fisher[name] * (param - optpar[name]).pow(2)).sum()
                    loss = loss + (self.cons_config.ewc_lambda / 2.0) * ewc_pen

                loss.backward()
                self.optimizer.step()

                self.total_flops += 6 * self.config.param_count_estimate * padded_inputs.numel()
                total_loss += loss.item()
                steps += 1

        # Check expert spawning criteria if enabled
        if self.cons_config.use_expert_growth:
            dummy_x = torch.randn(1, 10, self.config.d_model, device=self.device)
            _, entropy_tensor, _lb_loss = self.router(dummy_x)
            entropy_float = float(entropy_tensor.item())
            should_spawn, _reason = self.spawner.should_spawn(
                current_num_experts=len(self.experts),
                entropy_val=entropy_float,
                verifier_failure_rate=0.0,
            )
            if should_spawn:
                # Freeze current active expert and router row
                curr_exp_idx = len(self.experts) - 1
                expert_to_freeze = self.experts[curr_exp_idx]
                if isinstance(expert_to_freeze, ExpertModule):
                    expert_to_freeze.freeze()
                self.router.freeze_expert_row(curr_exp_idx)

                # Spawn new expert block and router row
                new_exp_idx = self.router.spawn_expert_row()
                new_exp = ExpertModule(expert_id=new_exp_idx, d_model=self.config.d_model).to(
                    self.device
                )
                self.experts.append(new_exp)

                # Recreate optimizer with new parameters
                self._reset_optimizer()

        # Update replay buffer reservoir
        if self.cons_config.use_replay_buffer:
            self.replay_buffer.extend(train_instances[:20])

        avg_loss = total_loss / max(1, steps)
        return {"task_idx": task_idx, "loss": avg_loss, "flops": self.total_flops}

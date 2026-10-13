"""Consolidation Cycle execution with ablation toggles."""

from dataclasses import dataclass
from typing import Any, override

import torch
import torch.nn as nn

from aptivra.baselines.base import BaselineTrainer
from aptivra.experts.expert import ExpertModule
from aptivra.experts.router import GatingRouter
from aptivra.experts.spawner import ExpertSpawner
from aptivra.memory.store import EpisodicStore
from aptivra.model.config import ModelConfig
from aptivra.tasks.base import TaskInstance


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
                    s_toks = self.encode_text(inst.target_solution + "\n")
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
                new_params = (
                    list(self.model.parameters())
                    + list(self.router.parameters())
                    + list(self.experts.parameters())
                )
                self.optimizer = torch.optim.AdamW(new_params, lr=self.cons_config.lr)

        # Update replay buffer reservoir
        if self.cons_config.use_replay_buffer:
            self.replay_buffer.extend(train_instances[:20])

        avg_loss = total_loss / max(1, steps)
        return {"task_idx": task_idx, "loss": avg_loss, "flops": self.total_flops}

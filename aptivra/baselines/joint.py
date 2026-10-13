"""Baseline 4: Joint Multitask Training Baseline Trainer (Upper Bound)."""

import random
from typing import Any, override

import torch
import torch.nn as nn

from aptivra.baselines.base import BaselineTrainer
from aptivra.model.config import ModelConfig
from aptivra.tasks.base import TaskInstance


class JointMultitaskBaselineTrainer(BaselineTrainer):
    """Joint multitask baseline trainer (Upper bound baseline)."""

    def __init__(
        self,
        model_config: ModelConfig,
        lr: float = 1e-4,
        total_epochs: int = 5,
        batch_size: int = 16,
    ) -> None:
        super().__init__(model_config)
        self.lr = lr
        self.total_epochs = total_epochs
        self.batch_size = batch_size
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr)
        self.criterion = nn.CrossEntropyLoss(ignore_index=-100)

    @override
    def train_task(self, task_idx: int, train_instances: list[TaskInstance]) -> dict[str, Any]:
        """For Joint Multitask baseline, train_instances contains ALL task instances combined."""
        self.model.train()
        total_loss = 0.0
        steps = 0

        # Shuffle joint multitask dataset
        all_instances = list(train_instances)
        random.shuffle(all_instances)

        for _ in range(self.total_epochs):
            for i in range(0, len(all_instances), self.batch_size):
                batch = all_instances[i : i + self.batch_size]
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
                loss = self.criterion(
                    shift_logits.view(-1, self.config.vocab_size), shift_targets.view(-1)
                )

                loss.backward()
                self.optimizer.step()

                self.total_flops += 6 * self.config.param_count_estimate * padded_inputs.numel()
                total_loss += loss.item()
                steps += 1

        avg_loss = total_loss / max(1, steps)
        return {"task_idx": task_idx, "loss": avg_loss, "flops": self.total_flops}

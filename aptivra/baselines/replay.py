"""Baseline 2: Experience Replay Baseline Trainer."""

import random
from typing import Any, override

import torch
import torch.nn as nn

from aptivra.baselines.base import BaselineTrainer
from aptivra.model.config import ModelConfig
from aptivra.tasks.base import TaskInstance


class ReplayBaselineTrainer(BaselineTrainer):
    """Replay-only baseline trainer maintaining a fixed experience replay buffer."""

    def __init__(
        self,
        model_config: ModelConfig,
        replay_capacity: int = 500,
        replay_ratio: float = 0.5,
        lr: float = 1e-4,
        epochs_per_task: int = 2,
        batch_size: int = 16,
    ) -> None:
        super().__init__(model_config)
        self.replay_capacity = replay_capacity
        self.replay_ratio = replay_ratio
        self.lr = lr
        self.epochs_per_task = epochs_per_task
        self.batch_size = batch_size
        self.replay_buffer: list[TaskInstance] = []
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr)
        self.criterion = nn.CrossEntropyLoss(ignore_index=-100)

    @override
    def train_task(self, task_idx: int, train_instances: list[TaskInstance]) -> dict[str, Any]:
        self.model.train()
        total_loss = 0.0
        steps = 0

        for _ in range(self.epochs_per_task):
            for i in range(0, len(train_instances), self.batch_size):
                batch = list(train_instances[i : i + self.batch_size])

                # Sample from replay buffer if available
                if self.replay_buffer:
                    replay_sample_count = int(len(batch) * self.replay_ratio)
                    if replay_sample_count > 0:
                        sampled_replay = random.sample(
                            self.replay_buffer, min(replay_sample_count, len(self.replay_buffer))
                        )
                        batch.extend(sampled_replay)

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

        # Populate replay buffer reservoir
        k_samples = min(len(train_instances), max(1, self.replay_capacity // 8))
        new_samples = random.sample(train_instances, k_samples)
        self.replay_buffer.extend(new_samples)
        if len(self.replay_buffer) > self.replay_capacity:
            self.replay_buffer = random.sample(self.replay_buffer, self.replay_capacity)

        avg_loss = total_loss / max(1, steps)
        return {"task_idx": task_idx, "loss": avg_loss, "flops": self.total_flops}

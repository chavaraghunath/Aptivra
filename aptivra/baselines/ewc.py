"""Baseline 3: Elastic Weight Consolidation (EWC) Baseline Trainer."""

from typing import Any, override

import torch
import torch.nn as nn

from aptivra.baselines.base import BaselineTrainer
from aptivra.model.config import ModelConfig
from aptivra.tasks.base import TaskInstance


class EWCBaselineTrainer(BaselineTrainer):
    """Elastic Weight Consolidation (EWC) trainer with Fisher Information Matrix regularization."""

    def __init__(
        self,
        model_config: ModelConfig,
        ewc_lambda: float = 400.0,
        lr: float = 1e-4,
        epochs_per_task: int = 2,
        batch_size: int = 16,
    ) -> None:
        super().__init__(model_config)
        self.ewc_lambda = ewc_lambda
        self.lr = lr
        self.epochs_per_task = epochs_per_task
        self.batch_size = batch_size
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr)
        self.criterion = nn.CrossEntropyLoss(ignore_index=-100)

        self.fisher_matrices: list[dict[str, torch.Tensor]] = []
        self.optpar_matrices: list[dict[str, torch.Tensor]] = []

    def compute_fisher(self, train_instances: list[TaskInstance], sample_size: int = 100) -> None:
        """Computes diagonal Fisher Information Matrix over current task data."""
        self.model.eval()
        fisher: dict[str, torch.Tensor] = {
            name: torch.zeros_like(param)
            for name, param in self.model.named_parameters()
            if param.requires_grad
        }
        optpar: dict[str, torch.Tensor] = {
            name: param.data.clone()
            for name, param in self.model.named_parameters()
            if param.requires_grad
        }

        subset = train_instances[:sample_size]
        for inst in subset:
            p_toks = self.encode_text(inst.prompt)
            s_toks = self.encode_text(inst.target_solution + "\n")
            full_seq = (
                torch.cat([p_toks, s_toks])[: self.config.max_seq_len].unsqueeze(0).to(self.device)
            )

            self.model.zero_grad()
            logits, _ = self.model(full_seq)
            log_probs = torch.log_softmax(logits, dim=-1)

            pseudo_target = log_probs.argmax(dim=-1)
            loss = nn.functional.nll_loss(
                log_probs.view(-1, self.config.vocab_size), pseudo_target.view(-1)
            )
            loss.backward()  # type: ignore[no-untyped-call]

            for name, param in self.model.named_parameters():
                if param.requires_grad and param.grad is not None:
                    fisher[name] += param.grad.data.pow(2) / len(subset)

        self.fisher_matrices.append(fisher)
        self.optpar_matrices.append(optpar)

    def ewc_loss(self) -> torch.Tensor:
        """Calculates EWC quadratic loss penalty across all previous tasks."""
        loss = torch.tensor(0.0, device=self.device)
        for fisher, optpar in zip(self.fisher_matrices, self.optpar_matrices, strict=True):
            for name, param in self.model.named_parameters():
                if name in fisher and param.requires_grad:
                    loss += (fisher[name] * (param - optpar[name]).pow(2)).sum()
        return (self.ewc_lambda / 2.0) * loss

    @override
    def train_task(self, task_idx: int, train_instances: list[TaskInstance]) -> dict[str, Any]:
        self.model.train()
        total_loss = 0.0
        steps = 0

        for _ in range(self.epochs_per_task):
            for i in range(0, len(train_instances), self.batch_size):
                batch = train_instances[i : i + self.batch_size]
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

                penalty = self.ewc_loss()
                loss = ce_loss + penalty

                loss.backward()
                self.optimizer.step()

                self.total_flops += 6 * self.config.param_count_estimate * padded_inputs.numel()
                total_loss += loss.item()
                steps += 1

        self.compute_fisher(train_instances)
        avg_loss = total_loss / max(1, steps)
        return {"task_idx": task_idx, "loss": avg_loss, "flops": self.total_flops}

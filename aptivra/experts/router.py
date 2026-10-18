"""Sequence-level learned top-1 routing for the ADR-0011 standard-base path."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal, Protocol, cast

import torch
import torch.nn as nn
import torch.nn.functional as functional


class _EmbeddingModel(Protocol):
    def get_input_embeddings(self) -> nn.Module:
        """Return the frozen base token-embedding module."""


class _ExpertActivator(Protocol):
    def activate(self, adapter_name: str) -> None:
        """Activate one named adapter."""

    def deactivate(self) -> None:
        """Deactivate adapters and expose the frozen-base path."""


@dataclass(frozen=True)
class RouteDecision:
    """One observable sequence-level routing decision."""

    route_status: Literal["routed", "unrouted"]
    expert_id: str | None
    confidence: float
    entropy: float
    probabilities: tuple[float, ...]
    fallback_probability: float = 0.0


@dataclass(frozen=True)
class RouterFitResult:
    """Bounded candidate-row fitting evidence."""

    loss: float
    optimizer_steps: int
    current_examples: int
    replay_examples: int


def masked_mean_frozen_embeddings(
    model: _EmbeddingModel,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """Mean-pool real frozen-base token embeddings using the supplied attention mask."""
    if input_ids.ndim != 2 or attention_mask.shape != input_ids.shape:
        raise ValueError("router inputs and attention mask must have equal rank-2 shapes")
    if input_ids.dtype != torch.long:
        raise TypeError("router input IDs must use torch.long")
    if attention_mask.dtype == torch.bool:
        mask = attention_mask
    elif attention_mask.dtype in (torch.int32, torch.int64, torch.long):
        if not bool(((attention_mask == 0) | (attention_mask == 1)).all().item()):
            raise ValueError("router attention mask must contain only zero or one")
        mask = attention_mask.bool()
    else:
        raise TypeError("router attention mask must be boolean or integer")
    counts = mask.sum(dim=1)
    if bool((counts == 0).any().item()):
        raise ValueError("router cannot pool an all-padding sequence")
    embedding = model.get_input_embeddings()
    if any(parameter.requires_grad for parameter in embedding.parameters()):
        raise RuntimeError("router features require a frozen base embedding")
    with torch.no_grad():
        token_features = cast(torch.Tensor, embedding(input_ids))
        weights = mask.to(dtype=token_features.dtype).unsqueeze(-1)
        pooled = (token_features * weights).sum(dim=1) / counts.to(
            dtype=token_features.dtype
        ).unsqueeze(-1)
    return pooled.detach()


class SequenceTop1Router(nn.Module):
    """Linear 960→E router with stable expert IDs and independently frozen rows."""

    def __init__(self, hidden_size: int, confidence_threshold: float) -> None:
        super().__init__()
        if isinstance(hidden_size, bool) or hidden_size <= 0:
            raise ValueError("router hidden size must be a positive integer")
        if not 0.0 < confidence_threshold <= 1.0:
            raise ValueError("router confidence threshold must be in (0, 1]")
        self.hidden_size = hidden_size
        self.confidence_threshold = confidence_threshold
        self.weights = nn.ParameterList()
        self.biases = nn.ParameterList()
        self.expert_ids: list[str] = []
        self.task_families: list[str] = []
        self.frozen_rows: list[bool] = []

    @property
    def num_experts(self) -> int:
        return len(self.expert_ids)

    def add_candidate(self, expert_id: str, task_family: str) -> int:
        """Append one deterministic zero-initialized trainable row without touching old rows."""
        if not expert_id or not task_family:
            raise ValueError("expert ID and task-family provenance must be non-empty")
        if expert_id in self.expert_ids:
            raise ValueError(f"router expert ID '{expert_id}' already exists")
        for row, bias in zip(self.weights, self.biases, strict=True):
            row.requires_grad_(False)
            bias.requires_grad_(False)
        self.weights.append(nn.Parameter(torch.zeros(self.hidden_size)))
        self.biases.append(nn.Parameter(torch.zeros(())))
        self.expert_ids.append(expert_id)
        self.task_families.append(task_family)
        self.frozen_rows.append(False)
        return self.num_experts - 1

    def promote_candidate(self) -> None:
        """Freeze the final candidate row after promotion."""
        if not self.expert_ids or self.frozen_rows[-1]:
            raise RuntimeError("router has no trainable candidate row")
        self.weights[-1].requires_grad_(False)
        self.biases[-1].requires_grad_(False)
        self.frozen_rows[-1] = True

    def candidate_parameters(self) -> tuple[nn.Parameter, nn.Parameter]:
        """Return only the final unfrozen row and bias."""
        if not self.expert_ids or self.frozen_rows[-1]:
            raise RuntimeError("router has no trainable candidate row")
        return self.weights[-1], self.biases[-1]

    def logits(self, features: torch.Tensor) -> torch.Tensor:
        """Score rank-2 frozen-base sequence features."""
        if self.num_experts == 0:
            raise RuntimeError("router has no expert rows")
        if features.ndim != 2 or features.shape[1] != self.hidden_size:
            raise ValueError("router features have the wrong shape")
        weight = torch.stack(tuple(self.weights))
        bias = torch.stack(tuple(self.biases))
        return features @ weight.t() + bias

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Return open-set expert probabilities; residual mass is frozen fallback."""
        probabilities = functional.softmax(self.classification_logits(features), dim=-1)
        return probabilities[:, 1:]

    def classification_logits(self, features: torch.Tensor) -> torch.Tensor:
        """Prepend a fixed reject logit so one-expert routing remains calibratable."""
        expert_logits = self.logits(features)
        fallback = torch.zeros(
            (features.shape[0], 1),
            dtype=expert_logits.dtype,
            device=expert_logits.device,
        )
        return torch.cat((fallback, expert_logits), dim=1)

    def route(self, features: torch.Tensor) -> tuple[RouteDecision, ...]:
        """Route above threshold and otherwise expose frozen-base fallback telemetry."""
        probabilities = self(features)
        fallback_probabilities = 1.0 - probabilities.sum(dim=1)
        decisions: list[RouteDecision] = []
        for row, fallback_tensor in zip(
            probabilities.detach().cpu(),
            fallback_probabilities.detach().cpu(),
            strict=True,
        ):
            confidence_tensor, index_tensor = torch.max(row, dim=0)
            confidence = float(confidence_tensor.item())
            fallback_probability = float(fallback_tensor.item())
            index = int(index_tensor.item())
            all_probabilities = torch.cat((fallback_tensor.reshape(1), row))
            entropy = float(
                (-(all_probabilities * torch.log(all_probabilities.clamp_min(1e-12))).sum()).item()
            )
            routed = (
                confidence >= self.confidence_threshold
                and confidence > fallback_probability
            )
            decisions.append(
                RouteDecision(
                    route_status="routed" if routed else "unrouted",
                    expert_id=self.expert_ids[index] if routed else None,
                    confidence=confidence,
                    entropy=entropy,
                    probabilities=tuple(float(value.item()) for value in row),
                    fallback_probability=fallback_probability,
                )
            )
        return tuple(decisions)

    def fit_candidate(
        self,
        current_features: torch.Tensor,
        replay_features: torch.Tensor,
        replay_labels: torch.Tensor,
        *,
        learning_rate: float,
        weight_decay: float,
        adam_beta1: float,
        adam_beta2: float,
        adam_epsilon: float,
        max_steps: int,
    ) -> tuple[RouterFitResult, torch.optim.Optimizer]:
        """Fit only the new row using current positives and stratified prior replay."""
        if current_features.ndim != 2 or current_features.shape[1] != self.hidden_size:
            raise ValueError("current router features have the wrong shape")
        if current_features.shape[0] == 0:
            raise ValueError("router fitting requires current-task examples")
        if replay_features.ndim != 2 or replay_features.shape[1] != self.hidden_size:
            raise ValueError("replay router features have the wrong shape")
        if replay_labels.ndim != 1 or replay_labels.shape[0] != replay_features.shape[0]:
            raise ValueError("replay labels must align one-to-one with replay features")
        candidate_index = self.num_experts - 1
        if candidate_index > 0 and replay_features.shape[0] == 0:
            raise ValueError("a new router row requires replay from prior experts")
        if replay_labels.numel() and (
            int(replay_labels.min().item()) < 0
            or int(replay_labels.max().item()) >= candidate_index
        ):
            raise ValueError("replay labels must name only prior frozen router rows")
        if not math.isfinite(learning_rate) or learning_rate <= 0.0:
            raise ValueError("router learning rate must be finite and positive")
        if not math.isfinite(weight_decay) or not 0.0 <= weight_decay <= 1.0:
            raise ValueError("router weight decay must be finite and in [0, 1]")
        if not math.isfinite(adam_beta1) or not 0.0 < adam_beta1 < 1.0:
            raise ValueError("router Adam beta1 must be finite and in (0, 1)")
        if not math.isfinite(adam_beta2) or not 0.0 < adam_beta2 < 1.0:
            raise ValueError("router Adam beta2 must be finite and in (0, 1)")
        if not math.isfinite(adam_epsilon) or not 0.0 < adam_epsilon <= 1.0:
            raise ValueError("router Adam epsilon must be finite and in (0, 1]")
        if isinstance(max_steps, bool) or max_steps <= 0:
            raise ValueError("router max steps must be a positive integer")
        device = self.weights[-1].device
        current = current_features.detach().to(device)
        replay = replay_features.detach().to(device)
        labels = torch.cat(
            (
                torch.full(
                    (current.shape[0],), candidate_index, dtype=torch.long, device=device
                ),
                replay_labels.detach().to(device=device, dtype=torch.long),
            )
        )
        features = torch.cat((current, replay), dim=0)
        old_rows = tuple(row.detach().clone() for row in self.weights[:-1])
        old_biases = tuple(bias.detach().clone() for bias in self.biases[:-1])
        optimizer = torch.optim.AdamW(
            self.candidate_parameters(),
            lr=learning_rate,
            betas=(adam_beta1, adam_beta2),
            eps=adam_epsilon,
            weight_decay=weight_decay,
        )
        final_loss = math.inf
        for _ in range(max_steps):
            optimizer.zero_grad(set_to_none=True)
            loss = functional.cross_entropy(
                self.classification_logits(features),
                labels + 1,
            )
            if not bool(torch.isfinite(loss).item()):
                raise FloatingPointError("router fitting produced a non-finite loss")
            torch.autograd.backward(loss)
            optimizer.step()
            final_loss = float(loss.detach().cpu().item())
        for actual, expected in zip(self.weights[:-1], old_rows, strict=True):
            if not torch.equal(actual.detach(), expected):
                raise RuntimeError("frozen router weight row changed during replay fitting")
        for actual, expected in zip(self.biases[:-1], old_biases, strict=True):
            if not torch.equal(actual.detach(), expected):
                raise RuntimeError("frozen router bias changed during replay fitting")
        return (
            RouterFitResult(
                loss=final_loss,
                optimizer_steps=max_steps,
                current_examples=current.shape[0],
                replay_examples=replay.shape[0],
            ),
            optimizer,
        )


def route_and_activate_fallback(
    bank: _ExpertActivator,
    router: SequenceTop1Router,
    feature: torch.Tensor,
) -> RouteDecision:
    """Apply one route, explicitly deactivating adapters on abstention."""
    if feature.ndim != 2 or feature.shape[0] != 1:
        raise ValueError("runtime routing accepts exactly one sequence feature")
    decision = router.route(feature)[0]
    if decision.route_status == "unrouted":
        bank.deactivate()
    else:
        bank.activate(cast(str, decision.expert_id))
    return decision

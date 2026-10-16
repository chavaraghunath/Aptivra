"""ADR-0011 lane-3 baselines on the verified stock-Llama path."""

from __future__ import annotations

import hashlib
import json
import math
import random
import resource
import sys
import time
from collections import defaultdict
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from importlib.resources.abc import Traversable
from pathlib import Path
from typing import Annotated, Literal, Protocol, cast

import torch
import torch.nn as nn
import torch.nn.functional as functional
import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator
from transformers import LlamaForCausalLM

from aptivra.eval.config import seed_experiment
from aptivra.eval.harness import (
    BaselineCompute,
    StandardBaselineRecord,
    StandardContinualEvalHarness,
    TaskStream,
)
from aptivra.inference.standard_base import (
    CausalBatch,
    StandardBase,
    collate_task_instances,
    generate_task_instance_answer,
)
from aptivra.resources import aptivra_resource
from aptivra.tasks.base import TaskInstance

DEFAULT_STANDARD_BASELINE_CONFIG = aptivra_resource("configs", "smollm2_baselines.yaml")
BaselineMethod = Literal[
    "frozen_base",
    "naive_sequential",
    "replay",
    "online_ewc",
    "manual_lora",
]
_ALL_METHODS: tuple[BaselineMethod, ...] = (
    "frozen_base",
    "naive_sequential",
    "replay",
    "online_ewc",
    "manual_lora",
)
_ALL_LORA_TARGETS = frozenset(
    {"q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"}
)
_IGNORE_INDEX = -100

PositiveInt = Annotated[int, Field(strict=True, gt=0)]
NonNegativeInt = Annotated[int, Field(strict=True, ge=0)]
PositiveFloat = Annotated[float, Field(strict=True, gt=0.0)]


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


class BaselineFeatureSettings(_StrictModel):
    """Fail-closed switch for the unaccepted lane-3 training path."""

    enabled: bool


class BaselineTrainingSettings(_StrictModel):
    """Shared optimizer and generation budget for all lane-3 methods."""

    learning_rate: PositiveFloat = Field(le=0.01)
    epochs_per_task: PositiveInt = Field(le=100)
    micro_batch_size: PositiveInt = Field(le=64)
    gradient_accumulation_steps: PositiveInt = Field(le=64)
    effective_batch_size: PositiveInt = Field(le=64)
    max_optimizer_steps_per_task: PositiveInt
    max_new_tokens: PositiveInt = Field(le=512)

    @model_validator(mode="after")
    def validate_effective_batch(self) -> BaselineTrainingSettings:
        calculated = self.micro_batch_size * self.gradient_accumulation_steps
        if calculated != self.effective_batch_size:
            raise ValueError(
                "micro_batch_size * gradient_accumulation_steps must equal effective_batch_size"
            )
        if self.effective_batch_size != 8:
            raise ValueError("ADR-0011 lane 3 requires effective_batch_size=8")
        return self


class ReplaySettings(_StrictModel):
    """Bounded canonical-text replay policy."""

    capacity: PositiveInt = Field(le=50_000)
    ratio: float = Field(strict=True, gt=0.0, le=1.0)
    sampling_seed: NonNegativeInt = Field(le=2**31 - 1)
    per_family_minimum: PositiveInt
    byte_limit: PositiveInt
    eviction_policy: Literal["stratified_hash_reservoir"]

    @model_validator(mode="after")
    def validate_minimum(self) -> ReplaySettings:
        if self.per_family_minimum > self.capacity:
            raise ValueError("replay per_family_minimum cannot exceed capacity")
        return self


class OnlineEWCSettings(_StrictModel):
    """Single-state online EWC policy."""

    penalty_lambda: PositiveFloat = Field(le=100_000.0)
    decay: float = Field(strict=True, gt=0.0, le=1.0)
    fisher_sample_size: PositiveInt
    state_byte_limit: PositiveInt


class ManualLoRASettings(_StrictModel):
    """PEFT-free task-ID LoRA baseline contract."""

    rank: Literal[16]
    alpha: PositiveFloat
    dropout: float = Field(strict=True, ge=0.0, le=1.0)
    target_modules: list[str] = Field(min_length=7, max_length=7)
    max_adapters: PositiveInt

    @model_validator(mode="after")
    def validate_targets(self) -> ManualLoRASettings:
        if self.dropout != 0.0:
            raise ValueError("lane-3 manual LoRA requires dropout=0.0")
        if len(set(self.target_modules)) != len(self.target_modules):
            raise ValueError("manual LoRA target_modules must be unique")
        if set(self.target_modules) != _ALL_LORA_TARGETS:
            raise ValueError("manual LoRA must target all seven Llama projection families")
        return self


class StandardBaselineSettings(_StrictModel):
    """Complete disabled-by-default lane-3 configuration."""

    schema_version: Literal["aptivra.standard_baseline_config.v1"]
    feature: BaselineFeatureSettings
    methods: list[BaselineMethod] = Field(min_length=5, max_length=5)
    seeds: list[NonNegativeInt] = Field(min_length=5, max_length=5)
    training: BaselineTrainingSettings
    replay: ReplaySettings
    online_ewc: OnlineEWCSettings
    manual_lora: ManualLoRASettings

    @model_validator(mode="after")
    def validate_matrix(self) -> StandardBaselineSettings:
        if tuple(self.methods) != _ALL_METHODS:
            raise ValueError(
                "baseline methods must list frozen, sequential, replay, online-EWC, "
                "and manual-LoRA in the registered order"
            )
        if len(set(self.seeds)) != 5:
            raise ValueError("baseline seeds must contain exactly five unique values")
        return self


def load_standard_baseline_settings(
    path: Path | str | Traversable = DEFAULT_STANDARD_BASELINE_CONFIG,
) -> StandardBaselineSettings:
    """Load and strictly validate the packaged lane-3 configuration."""
    source = Path(path) if isinstance(path, str) else path
    raw: object = yaml.safe_load(source.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("standard baseline config root must be a YAML mapping")
    if set(raw) != {"smollm2_baselines"}:
        raise ValueError("standard baseline config root must contain only 'smollm2_baselines'")
    return StandardBaselineSettings.model_validate(raw["smollm2_baselines"])


@dataclass(frozen=True)
class TrainingSchedule:
    """Immutable per-task/per-epoch data order for one declared seed."""

    seed: int
    ordered_task_ids: tuple[tuple[tuple[str, ...], ...], ...]
    sha256: str

    def order(self, task_idx: int, epoch: int) -> tuple[str, ...]:
        """Return the precomputed order, rejecting an undeclared coordinate."""
        try:
            return self.ordered_task_ids[task_idx][epoch]
        except IndexError as error:
            raise IndexError(f"training schedule has no task={task_idx}, epoch={epoch}") from error


def _derived_seed(seed: int, task_idx: int, epoch: int) -> int:
    payload = f"{seed}:{task_idx}:{epoch}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def build_training_schedule(
    stream: TaskStream,
    *,
    seed: int,
    epochs_per_task: int,
) -> TrainingSchedule:
    """Create deterministic data orders without mutating stream-owned split lists."""
    if isinstance(seed, bool) or seed < 0:
        raise ValueError("schedule seed must be a non-negative integer")
    if isinstance(epochs_per_task, bool) or epochs_per_task <= 0:
        raise ValueError("epochs_per_task must be a positive integer")
    task_orders: list[tuple[tuple[str, ...], ...]] = []
    for task_idx, family in enumerate(stream.families):
        instances = stream.get_split(family.family_id).train
        if not instances:
            raise ValueError(f"task '{family.family_id}' has an empty training split")
        task_ids = [instance.task_id for instance in instances]
        if len(task_ids) != len(set(task_ids)):
            raise ValueError(f"task '{family.family_id}' has duplicate training task IDs")
        epoch_orders: list[tuple[str, ...]] = []
        for epoch in range(epochs_per_task):
            ordered = list(task_ids)
            random.Random(_derived_seed(seed, task_idx, epoch)).shuffle(ordered)
            epoch_orders.append(tuple(ordered))
        task_orders.append(tuple(epoch_orders))
    frozen_orders = tuple(task_orders)
    payload = json.dumps(frozen_orders, separators=(",", ":")).encode()
    return TrainingSchedule(
        seed=seed,
        ordered_task_ids=frozen_orders,
        sha256=hashlib.sha256(payload).hexdigest(),
    )


def build_five_seed_schedules(
    settings: StandardBaselineSettings,
    stream: TaskStream,
) -> tuple[TrainingSchedule, ...]:
    """Build five schedules and fail if distinct seeds do not change data order."""
    schedules = tuple(
        build_training_schedule(
            stream,
            seed=seed,
            epochs_per_task=settings.training.epochs_per_task,
        )
        for seed in settings.seeds
    )
    hashes = {schedule.sha256 for schedule in schedules}
    if len(hashes) != len(schedules):
        raise ValueError(
            "configured seeds produced colliding data orders; use seeds with distinct schedules"
        )
    return schedules


def _target_mask(batch: CausalBatch) -> torch.Tensor:
    mask = batch.labels[:, 1:] != _IGNORE_INDEX
    if not bool(mask.any().item()):
        raise ValueError("target-only loss requires at least one supervised target token")
    return mask


def target_only_loss_from_hidden(
    model: LlamaForCausalLM,
    hidden_states: torch.Tensor,
    labels: torch.Tensor,
) -> torch.Tensor:
    """Project only supervised shifted positions through the tied LM head."""
    shifted_labels = labels[:, 1:]
    mask = shifted_labels != _IGNORE_INDEX
    if not bool(mask.any().item()):
        raise ValueError("target-only loss requires at least one supervised target token")
    selected_hidden = hidden_states[:, :-1, :][mask]
    selected_labels = shifted_labels[mask]
    logits = model.lm_head(selected_hidden)
    return functional.cross_entropy(logits, selected_labels, reduction="mean")


def full_logit_loss_from_hidden(
    model: LlamaForCausalLM,
    hidden_states: torch.Tensor,
    labels: torch.Tensor,
) -> torch.Tensor:
    """Reference full-projection loss over the identical supervised positions."""
    shifted_labels = labels[:, 1:]
    mask = shifted_labels != _IGNORE_INDEX
    if not bool(mask.any().item()):
        raise ValueError("full-logit reference requires at least one target token")
    logits = model.lm_head(hidden_states[:, :-1, :])
    return functional.cross_entropy(logits[mask], shifted_labels[mask], reduction="mean")


def target_only_causal_loss(base: StandardBase, batch: CausalBatch) -> torch.Tensor:
    """Run the decoder once and avoid allocating prompt/padding vocabulary logits."""
    outputs = base.model.model(
        input_ids=batch.input_ids,
        attention_mask=batch.attention_mask,
        use_cache=False,
        return_dict=True,
    )
    return target_only_loss_from_hidden(
        base.model,
        outputs.last_hidden_state,
        batch.labels,
    )


@dataclass(frozen=True)
class ReplayRecord:
    """Canonical CPU text record retained by the replay comparator."""

    task_id: str
    task_family: str
    prompt: str
    target_solution: str
    answer_stop_characters: tuple[str, ...]
    training_terminator: str

    @classmethod
    def from_instance(cls, instance: TaskInstance) -> ReplayRecord:
        return cls(
            task_id=instance.task_id,
            task_family=instance.task_family,
            prompt=instance.prompt,
            target_solution=instance.target_solution,
            answer_stop_characters=instance.answer_stop_characters,
            training_terminator=instance.training_terminator,
        )

    def to_instance(self) -> TaskInstance:
        return TaskInstance(
            task_id=self.task_id,
            task_family=self.task_family,
            prompt=self.prompt,
            target_solution=self.target_solution,
            answer_stop_characters=self.answer_stop_characters,
            training_terminator=self.training_terminator,
        )

    def canonical_bytes(self) -> bytes:
        payload = {
            "answer_stop_characters": self.answer_stop_characters,
            "prompt": self.prompt,
            "target_solution": self.target_solution,
            "task_family": self.task_family,
            "task_id": self.task_id,
            "training_terminator": self.training_terminator,
        }
        return json.dumps(
            payload,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")


class BoundedReplayBuffer:
    """Deterministic stratified priority reservoir with count and byte caps."""

    def __init__(self, settings: ReplaySettings) -> None:
        self.settings = settings
        self._records: tuple[ReplayRecord, ...] = ()

    @property
    def records(self) -> tuple[ReplayRecord, ...]:
        return self._records

    @property
    def size_bytes(self) -> int:
        return sum(len(record.canonical_bytes()) for record in self._records)

    def _priority(self, record: ReplayRecord, context: str) -> str:
        return hashlib.sha256(
            f"{self.settings.sampling_seed}:{context}:".encode() + record.canonical_bytes()
        ).hexdigest()

    def assert_within_bounds(self) -> None:
        """Fail closed if retained state violates either registered bound."""
        if any(isinstance(value, torch.Tensor) for value in self._records):
            raise RuntimeError("replay buffer must not retain accelerator tensors")
        if len(self._records) > self.settings.capacity:
            raise RuntimeError("replay record capacity bound violated")
        if self.size_bytes > self.settings.byte_limit:
            raise RuntimeError("replay canonical-byte bound violated")

    def add_instances(self, instances: Iterable[TaskInstance]) -> None:
        """Select the next bounded reservoir completely before mutating live state."""
        incoming = tuple(ReplayRecord.from_instance(instance) for instance in instances)
        if not incoming:
            raise ValueError("cannot update replay with an empty instance collection")
        if any(len(record.canonical_bytes()) > self.settings.byte_limit for record in incoming):
            raise MemoryError("one replay record exceeds the configured canonical-byte bound")
        candidates = {
            hashlib.sha256(record.canonical_bytes()).hexdigest(): record
            for record in (*self._records, *incoming)
        }
        grouped: dict[str, list[ReplayRecord]] = defaultdict(list)
        for record in candidates.values():
            grouped[record.task_family].append(record)
        required = self.settings.per_family_minimum * len(grouped)
        if required > self.settings.capacity:
            raise MemoryError("replay capacity cannot preserve the configured per-family minimum")
        selected: list[ReplayRecord] = []
        for family_id in sorted(grouped):
            group = sorted(
                grouped[family_id],
                key=lambda record: self._priority(record, "retain"),
            )
            if len(group) < self.settings.per_family_minimum:
                raise MemoryError(f"replay family '{family_id}' cannot meet per-family minimum")
            selected.extend(group[: self.settings.per_family_minimum])
        selected_bytes = sum(len(record.canonical_bytes()) for record in selected)
        if selected_bytes > self.settings.byte_limit:
            raise MemoryError("replay byte bound cannot preserve the configured per-family minimum")
        selected_keys = {
            hashlib.sha256(record.canonical_bytes()).hexdigest() for record in selected
        }
        remaining = sorted(
            (record for key, record in candidates.items() if key not in selected_keys),
            key=lambda record: self._priority(record, "retain"),
        )
        for record in remaining:
            record_bytes = len(record.canonical_bytes())
            if len(selected) >= self.settings.capacity:
                break
            if selected_bytes + record_bytes <= self.settings.byte_limit:
                selected.append(record)
                selected_bytes += record_bytes
        next_records = tuple(
            sorted(selected, key=lambda record: (record.task_family, record.task_id))
        )
        if len(next_records) > self.settings.capacity or selected_bytes > self.settings.byte_limit:
            raise RuntimeError("replay selection produced an out-of-bounds state")
        self._records = next_records
        self.assert_within_bounds()

    def sample(self, count: int, *, context: str) -> list[TaskInstance]:
        """Return a deterministic sample without changing reservoir state."""
        if isinstance(count, bool) or count < 0:
            raise ValueError("replay sample count must be a non-negative integer")
        ordered = sorted(
            self._records,
            key=lambda record: self._priority(record, f"sample:{context}"),
        )
        return [record.to_instance() for record in ordered[:count]]


class ManualLoRAAdapter(nn.Module):
    """One minimal low-rank delta for a single wrapped projection."""

    def __init__(self, in_features: int, out_features: int, *, rank: int, alpha: float) -> None:
        super().__init__()
        self.scale = alpha / rank
        self.lora_a = nn.Linear(in_features, rank, bias=False)
        self.lora_b = nn.Linear(rank, out_features, bias=False)
        nn.init.kaiming_uniform_(self.lora_a.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_b.weight)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return cast(torch.Tensor, self.lora_b(self.lora_a(inputs))) * self.scale


class ManualLoRALinear(nn.Module):
    """Frozen stock linear layer plus one explicitly selected task adapter."""

    def __init__(self, base_layer: nn.Linear, settings: ManualLoRASettings) -> None:
        super().__init__()
        self.base_layer = base_layer
        self.settings = settings
        self.adapters = nn.ModuleDict()
        self.active_adapter: str | None = None
        for parameter in self.base_layer.parameters():
            parameter.requires_grad_(False)

    def add_adapter(self, name: str) -> None:
        if not name or "." in name:
            raise ValueError("manual LoRA adapter names must be non-empty module-safe keys")
        if name in self.adapters:
            raise ValueError(f"manual LoRA adapter '{name}' already exists")
        if len(self.adapters) >= self.settings.max_adapters:
            raise RuntimeError("manual LoRA adapter-count bound exceeded")
        self.adapters[name] = ManualLoRAAdapter(
            self.base_layer.in_features,
            self.base_layer.out_features,
            rank=self.settings.rank,
            alpha=self.settings.alpha,
        ).to(device=self.base_layer.weight.device, dtype=self.base_layer.weight.dtype)

    def set_active(self, name: str | None) -> None:
        if name is not None and name not in self.adapters:
            raise KeyError(f"manual LoRA adapter '{name}' does not exist")
        self.active_adapter = name

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        output = cast(torch.Tensor, self.base_layer(inputs))
        if self.active_adapter is None:
            return output
        delta = cast(torch.Tensor, self.adapters[self.active_adapter](inputs))
        return output + delta


@dataclass
class _Counters:
    target_tokens: int = 0
    optimizer_steps: int = 0
    fisher_steps: int = 0
    estimated_flops: int = 0
    peak_accelerator_memory_bytes: int = 0
    peak_rss_bytes: int = 0
    peak_trainable_parameters: int = 0


def _rss_bytes() -> int:
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value if sys.platform == "darwin" else value * 1024


def _accelerator_memory(device: torch.device) -> int:
    if device.type == "mps":
        return max(
            int(torch.mps.current_allocated_memory()),
            int(torch.mps.driver_allocated_memory()),
        )
    if device.type == "cuda":
        return int(torch.cuda.max_memory_allocated(device))
    return 0


class StandardBaselineTrainer:
    """Shared target-only optimizer loop for standard-base lane-3 methods."""

    method: BaselineMethod

    def __init__(
        self,
        base: StandardBase,
        settings: StandardBaselineSettings,
        schedule: TrainingSchedule,
    ) -> None:
        self.base = base
        self.settings = settings
        self.schedule = schedule
        self.counters = _Counters()
        self.optimizer: torch.optim.Optimizer | None = None
        self._sample_memory()

    def _sample_memory(self) -> None:
        self.counters.peak_accelerator_memory_bytes = max(
            self.counters.peak_accelerator_memory_bytes,
            _accelerator_memory(self.base.device),
        )
        self.counters.peak_rss_bytes = max(
            self.counters.peak_rss_bytes,
            _rss_bytes(),
        )
        trainable = sum(
            parameter.numel()
            for parameter in self.base.model.parameters()
            if parameter.requires_grad
        )
        self.counters.peak_trainable_parameters = max(
            self.counters.peak_trainable_parameters,
            trainable,
        )

    def _regularization_loss(self) -> torch.Tensor:
        return torch.zeros((), device=self.base.device, dtype=torch.float32)

    def _augment_batch(
        self,
        batch: list[TaskInstance],
        *,
        task_idx: int,
        epoch: int,
        batch_idx: int,
    ) -> list[TaskInstance]:
        del task_idx, epoch, batch_idx
        return batch

    def _after_task(
        self,
        task_idx: int,
        train_instances: list[TaskInstance],
    ) -> None:
        del task_idx, train_instances

    def _ordered_instances(
        self,
        task_idx: int,
        epoch: int,
        train_instances: list[TaskInstance],
    ) -> list[TaskInstance]:
        by_id = {instance.task_id: instance for instance in train_instances}
        if len(by_id) != len(train_instances):
            raise ValueError("training instances must have unique task IDs")
        order = self.schedule.order(task_idx, epoch)
        if set(order) != set(by_id) or len(order) != len(by_id):
            raise ValueError("training instances do not match the frozen schedule")
        return [by_id[task_id] for task_id in order]

    def train_task(
        self,
        task_idx: int,
        train_instances: list[TaskInstance],
    ) -> float:
        """Train one chronological task within the registered step cap."""
        if not train_instances:
            raise ValueError("cannot train a baseline on an empty task split")
        if self.optimizer is None:
            raise RuntimeError(f"baseline method '{self.method}' has no optimizer")
        family_ids = {instance.task_family for instance in train_instances}
        if len(family_ids) != 1:
            raise ValueError("one train_task call must contain exactly one task family")
        micro_size = self.settings.training.micro_batch_size
        accumulation = self.settings.training.gradient_accumulation_steps
        micro_batches_per_epoch = math.ceil(len(train_instances) / micro_size)
        required_steps = self.settings.training.epochs_per_task * math.ceil(
            micro_batches_per_epoch / accumulation
        )
        if required_steps > self.settings.training.max_optimizer_steps_per_task:
            raise RuntimeError(
                "registered optimizer-step cap is smaller than the deterministic schedule"
            )
        self.base.model.train()
        loss_total = 0.0
        loss_count = 0
        for epoch in range(self.settings.training.epochs_per_task):
            ordered = self._ordered_instances(task_idx, epoch, train_instances)
            micro_batches = [
                ordered[offset : offset + micro_size]
                for offset in range(0, len(ordered), micro_size)
            ]
            for group_start in range(0, len(micro_batches), accumulation):
                group = micro_batches[group_start : group_start + accumulation]
                self.optimizer.zero_grad(set_to_none=True)
                for group_offset, current in enumerate(group):
                    batch_idx = group_start + group_offset
                    augmented = self._augment_batch(
                        list(current),
                        task_idx=task_idx,
                        epoch=epoch,
                        batch_idx=batch_idx,
                    )
                    batch = collate_task_instances(self.base, augmented)
                    target_tokens = int(_target_mask(batch).sum().item())
                    causal_loss = target_only_causal_loss(self.base, batch)
                    loss = causal_loss + self._regularization_loss()
                    if not bool(torch.isfinite(loss).item()):
                        raise FloatingPointError(
                            f"non-finite {self.method} loss on task {task_idx}"
                        )
                    torch.autograd.backward(loss / len(group))
                    loss_total += float(loss.detach().cpu().item())
                    loss_count += 1
                    self.counters.target_tokens += target_tokens
                    trainable = sum(
                        parameter.numel()
                        for parameter in self.base.model.parameters()
                        if parameter.requires_grad
                    )
                    self.counters.estimated_flops += 6 * trainable * target_tokens
                    self._sample_memory()
                self.optimizer.step()
                self.counters.optimizer_steps += 1
                self._sample_memory()
        self._after_task(task_idx, train_instances)
        if loss_count == 0:
            raise RuntimeError("training schedule produced no loss measurements")
        average = loss_total / loss_count
        if not math.isfinite(average):
            raise FloatingPointError("baseline average loss is not finite")
        return average

    def predict_instance(self, instance: TaskInstance) -> str:
        """Generate through the lane-1/2 stock task path."""
        cast(nn.Module, self.base.model).eval()
        result = generate_task_instance_answer(
            self.base,
            instance,
            max_new_tokens=self.settings.training.max_new_tokens,
        )
        self._sample_memory()
        return result.text

    @property
    def replay_bytes(self) -> int:
        return 0

    @property
    def ewc_state_bytes(self) -> int:
        return 0

    @property
    def adapter_bytes(self) -> int:
        return 0

    def compute_record(
        self,
        *,
        generations: int,
        wall_time_seconds: float,
    ) -> BaselineCompute:
        """Freeze complete finite compute/state evidence for the run."""
        self._sample_memory()
        return BaselineCompute(
            target_tokens=self.counters.target_tokens,
            optimizer_steps=self.counters.optimizer_steps,
            fisher_steps=self.counters.fisher_steps,
            generations=generations,
            estimated_flops=self.counters.estimated_flops,
            wall_time_seconds=wall_time_seconds,
            peak_accelerator_memory_bytes=(self.counters.peak_accelerator_memory_bytes),
            peak_rss_bytes=self.counters.peak_rss_bytes,
            replay_bytes=self.replay_bytes,
            ewc_state_bytes=self.ewc_state_bytes,
            adapter_bytes=self.adapter_bytes,
            total_parameters=sum(parameter.numel() for parameter in self.base.model.parameters()),
            peak_trainable_parameters=self.counters.peak_trainable_parameters,
        )


class FrozenBaseTrainer(StandardBaselineTrainer):
    """Frozen pretrained base with no parameter update."""

    method: BaselineMethod = "frozen_base"

    def __init__(
        self,
        base: StandardBase,
        settings: StandardBaselineSettings,
        schedule: TrainingSchedule,
    ) -> None:
        for parameter in base.model.parameters():
            parameter.requires_grad_(False)
        super().__init__(base, settings, schedule)

    def train_task(
        self,
        task_idx: int,
        train_instances: list[TaskInstance],
    ) -> float:
        if not train_instances:
            raise ValueError("cannot advance frozen baseline over an empty task split")
        self._ordered_instances(task_idx, 0, train_instances)
        return 0.0


class NaiveSequentialTrainer(StandardBaselineTrainer):
    """Naive chronological full fp32 fine-tuning."""

    method: BaselineMethod = "naive_sequential"

    def __init__(
        self,
        base: StandardBase,
        settings: StandardBaselineSettings,
        schedule: TrainingSchedule,
    ) -> None:
        for parameter in base.model.parameters():
            parameter.requires_grad_(True)
        super().__init__(base, settings, schedule)
        self.optimizer = torch.optim.AdamW(
            base.model.parameters(), lr=settings.training.learning_rate
        )


class ReplayTrainer(NaiveSequentialTrainer):
    """Full fine-tuning with bounded deterministic canonical-text replay."""

    method: BaselineMethod = "replay"

    def __init__(
        self,
        base: StandardBase,
        settings: StandardBaselineSettings,
        schedule: TrainingSchedule,
    ) -> None:
        super().__init__(base, settings, schedule)
        self.buffer = BoundedReplayBuffer(settings.replay)

    def _augment_batch(
        self,
        batch: list[TaskInstance],
        *,
        task_idx: int,
        epoch: int,
        batch_idx: int,
    ) -> list[TaskInstance]:
        if not self.buffer.records:
            return batch
        replay_count = max(1, math.ceil(len(batch) * self.settings.replay.ratio))
        replay = self.buffer.sample(
            replay_count,
            context=f"{self.schedule.seed}:{task_idx}:{epoch}:{batch_idx}",
        )
        return [*batch, *replay]

    def _after_task(
        self,
        task_idx: int,
        train_instances: list[TaskInstance],
    ) -> None:
        del task_idx
        self.buffer.add_instances(train_instances)

    @property
    def replay_bytes(self) -> int:
        self.buffer.assert_within_bounds()
        return self.buffer.size_bytes


class OnlineEWCTrainer(NaiveSequentialTrainer):
    """Full fine-tuning with one consolidated diagonal Fisher and anchor."""

    method: BaselineMethod = "online_ewc"

    def __init__(
        self,
        base: StandardBase,
        settings: StandardBaselineSettings,
        schedule: TrainingSchedule,
    ) -> None:
        super().__init__(base, settings, schedule)
        required_bytes = 2 * sum(
            parameter.numel() * parameter.element_size()
            for parameter in base.model.parameters()
            if parameter.requires_grad
        )
        if required_bytes > settings.online_ewc.state_byte_limit:
            raise MemoryError("online-EWC Fisher+anchor state exceeds the configured byte bound")
        self.fisher: dict[str, torch.Tensor] | None = None
        self.anchor: dict[str, torch.Tensor] | None = None

    def _regularization_loss(self) -> torch.Tensor:
        if self.fisher is None or self.anchor is None:
            return torch.zeros((), device=self.base.device, dtype=torch.float32)
        penalty = torch.zeros((), device=self.base.device, dtype=torch.float32)
        for name, parameter in self.base.model.named_parameters():
            if parameter.requires_grad:
                penalty = (
                    penalty + (self.fisher[name] * (parameter - self.anchor[name]).square()).sum()
                )
        return 0.5 * self.settings.online_ewc.penalty_lambda * penalty

    def _after_task(
        self,
        task_idx: int,
        train_instances: list[TaskInstance],
    ) -> None:
        ordered = self._ordered_instances(task_idx, 0, train_instances)
        subset = ordered[: self.settings.online_ewc.fisher_sample_size]
        if not subset:
            raise ValueError("online-EWC requires at least one Fisher sample")
        current = {
            name: torch.zeros_like(parameter)
            for name, parameter in self.base.model.named_parameters()
            if parameter.requires_grad
        }
        cast(nn.Module, self.base.model).eval()
        for instance in subset:
            self.base.model.zero_grad(set_to_none=True)
            batch = collate_task_instances(self.base, [instance])
            loss = target_only_causal_loss(self.base, batch)
            if not bool(torch.isfinite(loss).item()):
                raise FloatingPointError("online-EWC Fisher loss is not finite")
            torch.autograd.backward(loss)
            self._sample_memory()
            for name, parameter in self.base.model.named_parameters():
                if parameter.requires_grad and parameter.grad is not None:
                    current[name].add_(parameter.grad.detach().square())
            self.counters.fisher_steps += 1
        for value in current.values():
            value.div_(len(subset))
        if self.fisher is None:
            consolidated = current
        else:
            consolidated = {
                name: self.settings.online_ewc.decay * self.fisher[name] + value
                for name, value in current.items()
            }
        anchor = {
            name: parameter.detach().clone()
            for name, parameter in self.base.model.named_parameters()
            if parameter.requires_grad
        }
        state_bytes = sum(
            tensor.numel() * tensor.element_size()
            for tensor in (*consolidated.values(), *anchor.values())
        )
        if state_bytes > self.settings.online_ewc.state_byte_limit:
            raise MemoryError("online-EWC consolidated state violated its byte bound")
        self.fisher = consolidated
        self.anchor = anchor
        self.base.model.zero_grad(set_to_none=True)
        self._sample_memory()

    @property
    def state_tensor_count(self) -> int:
        if self.fisher is None or self.anchor is None:
            return 0
        return len(self.fisher) + len(self.anchor)

    @property
    def ewc_state_bytes(self) -> int:
        if self.fisher is None or self.anchor is None:
            return 0
        state_bytes = sum(
            tensor.numel() * tensor.element_size()
            for tensor in (*self.fisher.values(), *self.anchor.values())
        )
        if state_bytes > self.settings.online_ewc.state_byte_limit:
            raise RuntimeError("online-EWC live state exceeds its registered byte bound")
        return state_bytes


class ManualLoRATrainer(StandardBaselineTrainer):
    """Frozen-base, per-task rank-16 LoRA with manual task-ID selection."""

    method: BaselineMethod = "manual_lora"

    def __init__(
        self,
        base: StandardBase,
        settings: StandardBaselineSettings,
        schedule: TrainingSchedule,
    ) -> None:
        for parameter in base.model.parameters():
            parameter.requires_grad_(False)
        self.wrapped_layers = self._inject(base.model, settings.manual_lora)
        self.family_adapters: dict[str, str] = {}
        super().__init__(base, settings, schedule)

    @staticmethod
    def _inject(
        model: LlamaForCausalLM,
        settings: ManualLoRASettings,
    ) -> tuple[ManualLoRALinear, ...]:
        wrapped: list[ManualLoRALinear] = []
        found_targets: set[str] = set()
        for module_name, module in list(model.named_modules()):
            target_name = module_name.rsplit(".", maxsplit=1)[-1]
            if target_name not in settings.target_modules:
                continue
            if not isinstance(module, nn.Linear):
                raise TypeError(f"LoRA target '{module_name}' is not a linear layer")
            parent_name, _, child_name = module_name.rpartition(".")
            parent = model.get_submodule(parent_name) if parent_name else model
            replacement = ManualLoRALinear(module, settings)
            setattr(parent, child_name, replacement)
            wrapped.append(replacement)
            found_targets.add(target_name)
        if found_targets != _ALL_LORA_TARGETS or not wrapped:
            missing = sorted(_ALL_LORA_TARGETS - found_targets)
            raise ValueError(
                "manual LoRA could not find every configured Llama target: " + ", ".join(missing)
            )
        return tuple(wrapped)

    def _set_active(self, adapter_name: str | None) -> None:
        for layer in self.wrapped_layers:
            layer.set_active(adapter_name)

    def _set_only_adapter_trainable(self, adapter_name: str) -> None:
        for layer in self.wrapped_layers:
            for name, adapter in layer.adapters.items():
                for parameter in adapter.parameters():
                    parameter.requires_grad_(name == adapter_name)
        self._sample_memory()

    def train_task(
        self,
        task_idx: int,
        train_instances: list[TaskInstance],
    ) -> float:
        if not train_instances:
            raise ValueError("cannot train manual LoRA on an empty task split")
        family_ids = {instance.task_family for instance in train_instances}
        if len(family_ids) != 1:
            raise ValueError("manual LoRA task batch must contain one family")
        family_id = next(iter(family_ids))
        if family_id in self.family_adapters:
            raise ValueError(f"manual LoRA family '{family_id}' already has an adapter")
        adapter_name = f"task_{task_idx:04d}"
        for layer in self.wrapped_layers:
            layer.add_adapter(adapter_name)
        self.family_adapters[family_id] = adapter_name
        self._set_active(adapter_name)
        self._set_only_adapter_trainable(adapter_name)
        parameters = [
            parameter for parameter in self.base.model.parameters() if parameter.requires_grad
        ]
        if not parameters:
            raise RuntimeError("manual LoRA exposed no trainable adapter parameters")
        self.optimizer = torch.optim.AdamW(
            parameters,
            lr=self.settings.training.learning_rate,
        )
        loss = super().train_task(task_idx, train_instances)
        for parameter in parameters:
            parameter.requires_grad_(False)
        self.optimizer = None
        return loss

    def predict_instance(self, instance: TaskInstance) -> str:
        self._set_active(self.family_adapters.get(instance.task_family))
        return super().predict_instance(instance)

    @property
    def adapter_bytes(self) -> int:
        return sum(
            parameter.numel() * parameter.element_size()
            for layer in self.wrapped_layers
            for adapter in layer.adapters.values()
            for parameter in adapter.parameters()
        )


_TRAINER_TYPES: dict[BaselineMethod, type[StandardBaselineTrainer]] = {
    "frozen_base": FrozenBaseTrainer,
    "naive_sequential": NaiveSequentialTrainer,
    "replay": ReplayTrainer,
    "online_ewc": OnlineEWCTrainer,
    "manual_lora": ManualLoRATrainer,
}


def create_standard_baseline_trainer(
    method: BaselineMethod,
    base: StandardBase,
    settings: StandardBaselineSettings,
    schedule: TrainingSchedule,
) -> StandardBaselineTrainer:
    """Construct one declared method or reject an unknown value."""
    try:
        trainer_type = _TRAINER_TYPES[method]
    except KeyError as error:
        raise ValueError(f"unknown standard baseline method: {method}") from error
    return trainer_type(base, settings, schedule)


def run_standard_baseline(
    method: BaselineMethod,
    base: StandardBase,
    stream: TaskStream,
    settings: StandardBaselineSettings,
    *,
    seed: int,
    schedule: TrainingSchedule | None = None,
) -> StandardBaselineRecord:
    """Run one method/seed through the real v5 continual-learning harness."""
    if not settings.feature.enabled:
        raise RuntimeError("ADR-0011 standard baselines are disabled by configuration")
    if method not in settings.methods:
        raise ValueError(f"baseline method '{method}' is not configured")
    if seed not in settings.seeds:
        raise ValueError(f"baseline seed {seed} is not one of the five declared seeds")
    seed_experiment(seed)
    active_schedule = schedule or build_training_schedule(
        stream,
        seed=seed,
        epochs_per_task=settings.training.epochs_per_task,
    )
    if active_schedule.seed != seed:
        raise ValueError("training schedule seed does not match the run seed")
    trainer = create_standard_baseline_trainer(method, base, settings, active_schedule)
    harness = StandardContinualEvalHarness(stream)
    evaluation = harness.start_run()
    generations = 0

    def predict(instance: TaskInstance) -> str:
        nonlocal generations
        generations += 1
        return trainer.predict_instance(instance)

    started = time.perf_counter()
    evaluation.measure_frozen(predict)
    losses: list[float] = []
    for task_idx, family in enumerate(stream.families):
        train_instances = stream.get_split(family.family_id).train
        losses.append(trainer.train_task(task_idx, train_instances))
        evaluation.measure_post_cycle(task_idx, predict)
    wall_time = time.perf_counter() - started
    compute = trainer.compute_record(
        generations=generations,
        wall_time_seconds=wall_time,
    )
    return evaluation.finalize(
        method=method,
        seed=seed,
        schedule_sha256=active_schedule.sha256,
        task_losses=tuple(losses),
        compute=compute,
        metadata={
            "target_only_loss": True,
            "dtype": str(next(base.model.parameters()).dtype).removeprefix("torch."),
            "device": str(base.device),
            "manual_lora_uses_peft": False,
        },
    )


class StandardBaseFactory(Protocol):
    """Fresh verified base factory; called once per method and seed."""

    def __call__(self, method: BaselineMethod, seed: int) -> StandardBase:
        """Return a pristine independently loaded standard base."""
        ...


def run_standard_baseline_suite(
    settings: StandardBaselineSettings,
    *,
    base_factory: StandardBaseFactory,
    stream_factory: Callable[[int], TaskStream],
) -> tuple[StandardBaselineRecord, ...]:
    """Execute the exact five-seed/method matrix, checking flags before allocation."""
    if not settings.feature.enabled:
        raise RuntimeError("ADR-0011 standard baselines are disabled by configuration")
    records: list[StandardBaselineRecord] = []
    schedule_hashes: set[str] = set()
    for seed in settings.seeds:
        stream = stream_factory(seed)
        schedule = build_training_schedule(
            stream,
            seed=seed,
            epochs_per_task=settings.training.epochs_per_task,
        )
        if schedule.sha256 in schedule_hashes:
            raise ValueError("different seeds produced the same training data order")
        schedule_hashes.add(schedule.sha256)
        for method in settings.methods:
            base = base_factory(method, seed)
            records.append(
                run_standard_baseline(
                    method,
                    base,
                    stream,
                    settings,
                    seed=seed,
                    schedule=schedule,
                )
            )
    return tuple(records)

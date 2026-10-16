"""Transactional ADR-0011 consolidation over real PEFT, router, and verifier state."""

from __future__ import annotations

import copy
import hashlib
import json
import os
import random
import shutil
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch

from aptivra.baselines.standard import BoundedReplayBuffer, ReplayRecord, ReplaySettings
from aptivra.experts.router import RouteDecision, RouterFitResult, SequenceTop1Router
from aptivra.experts.spawner import (
    EvidenceGrowthPolicy,
    GrowthDecision,
    GrowthSettings,
    VerifierInfrastructureError,
    VerifierOutcome,
    build_growth_evidence,
)
from aptivra.experts.standard import (
    ExpertManifest,
    PeftExpertBank,
    StandardExpertSettings,
    load_additional_promoted_expert,
    load_promoted_expert,
    standard_base_fingerprint,
)
from aptivra.gate.promotion_gate import PromotionGate, PromotionResult
from aptivra.inference.standard_base import StandardBase, collate_task_instances
from aptivra.tasks.base import TaskInstance

_STATE_SCHEMA = "aptivra.consolidation_state.v1"
_MANIFEST_SCHEMA = "aptivra.consolidation_manifest.v1"


@dataclass(frozen=True)
class VerifiedMemoryRecord:
    """Canonical verifier-gated text memory retained by consolidation."""

    task_id: str
    task_family: str
    prompt: str
    target_solution: str


@dataclass(frozen=True)
class ConsolidationTrainingResult:
    """Measured bounded candidate and router training outcome."""

    adapter_loss: float
    adapter_optimizer_steps: int
    target_tokens: int
    router: RouterFitResult


@dataclass(frozen=True)
class ConsolidationSnapshot:
    """Deep transaction snapshot plus hashes used to prove exact rollback."""

    payload: dict[str, Any]
    hashes: dict[str, str]


@dataclass(frozen=True)
class AcceptedCycle:
    """State reloaded from an accepted-cycle artifact in a fresh object graph."""

    bank: PeftExpertBank
    router: SequenceTop1Router
    replay: BoundedReplayBuffer
    memory: tuple[VerifiedMemoryRecord, ...]
    optimizer_state: dict[str, Any]
    rng_state: dict[str, Any]
    state_hashes: dict[str, str]


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _adapters_enabled(bank: PeftExpertBank) -> bool:
    states = {
        bool(module.disable_adapters)
        for module in bank.model.modules()
        if isinstance(getattr(module, "disable_adapters", None), bool)
    }
    if len(states) != 1:
        raise RuntimeError("PEFT adapter layers have inconsistent enablement state")
    return not states.pop()


def _update_digest(digest: Any, value: object) -> None:
    if isinstance(value, torch.Tensor):
        tensor = value.detach().cpu().contiguous()
        digest.update(b"tensor")
        digest.update(str(tensor.dtype).encode())
        digest.update(json.dumps(list(tensor.shape)).encode())
        digest.update(tensor.numpy().tobytes())
    elif isinstance(value, np.ndarray):
        digest.update(b"ndarray")
        digest.update(str(value.dtype).encode())
        digest.update(json.dumps(list(value.shape)).encode())
        digest.update(value.tobytes())
    elif isinstance(value, dict):
        digest.update(b"dict")
        for key in sorted(value, key=str):
            _update_digest(digest, key)
            _update_digest(digest, value[key])
    elif isinstance(value, list):
        digest.update(b"list")
        for item in value:
            _update_digest(digest, item)
    elif isinstance(value, tuple):
        digest.update(b"tuple")
        for item in value:
            _update_digest(digest, item)
    elif isinstance(value, set):
        digest.update(b"set")
        for item in sorted(value, key=str):
            _update_digest(digest, item)
    elif value is None or isinstance(value, bool | int | float | str | bytes):
        digest.update(type(value).__name__.encode())
        digest.update(repr(value).encode())
    else:
        raise TypeError(f"cannot hash consolidation state value {type(value).__name__}")


def state_sha256(value: object) -> str:
    """Hash nested tensor/Python state without serializing executable objects."""
    digest = hashlib.sha256()
    _update_digest(digest, value)
    return digest.hexdigest()


def _capture_rng_state(device: torch.device) -> dict[str, Any]:
    algorithm, keys, position, has_gauss, cached_gaussian = np.random.get_state()
    mps_state: torch.Tensor | None = None
    cuda_state: torch.Tensor | None = None
    if device.type == "mps" and torch.backends.mps.is_available():
        mps_state = torch.mps.get_rng_state().cpu()
    elif device.type == "cuda" and torch.cuda.is_available():
        cuda_state = torch.cuda.get_rng_state(device).cpu()
    return {
        "python": random.getstate(),
        "numpy": {
            "algorithm": algorithm,
            "keys": keys.tolist(),
            "position": position,
            "has_gauss": has_gauss,
            "cached_gaussian": cached_gaussian,
        },
        "torch_cpu": torch.get_rng_state().cpu(),
        "torch_mps": mps_state,
        "torch_cuda": cuda_state,
    }


def _restore_rng_state(device: torch.device, state: dict[str, Any]) -> None:
    random.setstate(state["python"])
    numpy_state = state["numpy"]
    np.random.set_state(
        (
            numpy_state["algorithm"],
            np.asarray(numpy_state["keys"], dtype=np.uint32),
            numpy_state["position"],
            numpy_state["has_gauss"],
            numpy_state["cached_gaussian"],
        )
    )
    torch.set_rng_state(state["torch_cpu"])
    if device.type == "mps" and state["torch_mps"] is not None:
        torch.mps.set_rng_state(state["torch_mps"])
    elif device.type == "cuda" and state["torch_cuda"] is not None:
        torch.cuda.set_rng_state(state["torch_cuda"], device)


def _verifier_accuracy(outcomes: tuple[VerifierOutcome, ...]) -> float:
    if not outcomes:
        raise ValueError("promotion requires non-empty verifier evidence")
    passed = 0
    for outcome in outcomes:
        if outcome.infrastructure_error is not None:
            raise VerifierInfrastructureError(outcome.infrastructure_error)
        if outcome.passed is None:
            raise RuntimeError("validated verifier outcome unexpectedly lacks a result")
        passed += int(outcome.passed)
    return passed / len(outcomes)


class ConsolidationCycle:
    """One bounded candidate transaction with exact rejection rollback."""

    def __init__(
        self,
        bank: PeftExpertBank,
        router: SequenceTop1Router,
        settings: StandardExpertSettings,
        *,
        replay: BoundedReplayBuffer | None = None,
        memory: tuple[VerifiedMemoryRecord, ...] = (),
    ) -> None:
        if not settings.feature.enabled:
            raise RuntimeError("ADR-0011 router/consolidation path is disabled by configuration")
        if bank.trainable_adapter is None:
            raise ValueError("consolidation requires one trainable PEFT candidate")
        if not router.expert_ids or router.expert_ids[-1] != bank.trainable_adapter:
            raise ValueError("router candidate row must match the trainable PEFT adapter")
        if set(router.expert_ids) != set(bank.adapter_names):
            raise ValueError("router rows must map exactly to the attached PEFT adapters")
        if router.frozen_rows[-1]:
            raise ValueError("router candidate row must be trainable")
        self.bank = bank
        self.router = router
        self.settings = settings
        self.replay = replay or BoundedReplayBuffer(self._replay_settings())
        self.memory = tuple(memory)
        self.adapter_optimizer = torch.optim.AdamW(
            bank.adapter_parameters(bank.trainable_adapter),
            lr=settings.consolidation.adapter_learning_rate,
            betas=(
                settings.consolidation.adapter_adam_beta1,
                settings.consolidation.adapter_adam_beta2,
            ),
            eps=settings.consolidation.adapter_adam_epsilon,
            weight_decay=settings.consolidation.adapter_weight_decay,
        )
        self.router_optimizer: torch.optim.Optimizer | None = None

    def _replay_settings(self) -> ReplaySettings:
        config = self.settings.consolidation
        return ReplaySettings(
            capacity=config.replay_capacity,
            ratio=config.replay_ratio,
            sampling_seed=config.replay_sampling_seed,
            per_family_minimum=config.replay_per_family_minimum,
            byte_limit=config.replay_byte_limit,
            eviction_policy=config.replay_eviction_policy,
        )

    def _router_structure(self) -> dict[str, Any]:
        return {
            "hidden_size": self.router.hidden_size,
            "confidence_threshold": self.router.confidence_threshold,
            "expert_ids": list(self.router.expert_ids),
            "task_families": list(self.router.task_families),
            "frozen_rows": list(self.router.frozen_rows),
        }

    def _optimizer_state(self) -> dict[str, Any]:
        return {
            "adapter": self.adapter_optimizer.state_dict(),
            "router": (
                None if self.router_optimizer is None else self.router_optimizer.state_dict()
            ),
        }

    def _payload(self) -> dict[str, Any]:
        return {
            "schema_version": _STATE_SCHEMA,
            "model_state": copy.deepcopy(self.bank.model.state_dict()),
            "model_requires_grad": {
                name: parameter.requires_grad
                for name, parameter in self.bank.model.named_parameters()
            },
            "model_training": self.bank.model.training,
            "active_adapters": tuple(self.bank.model.active_adapters),
            "adapters_enabled": _adapters_enabled(self.bank),
            "mature_adapters": set(self.bank.mature_adapters),
            "trainable_adapter": self.bank.trainable_adapter,
            "router_state": copy.deepcopy(self.router.state_dict()),
            "router_structure": self._router_structure(),
            "router_training": self.router.training,
            "optimizer_state": copy.deepcopy(self._optimizer_state()),
            "replay_records": tuple(self.replay.records),
            "memory": tuple(self.memory),
            "settings": self.settings.model_dump(mode="python"),
            "rng_state": _capture_rng_state(self.bank.base.device),
        }

    @staticmethod
    def _hash_payload(payload: dict[str, Any]) -> dict[str, str]:
        return {
            "weights_adapters": state_sha256(
                (
                    payload["model_state"],
                    payload["model_requires_grad"],
                    payload["model_training"],
                    payload["active_adapters"],
                    payload["adapters_enabled"],
                )
            ),
            "router": state_sha256(
                (
                    payload["router_state"],
                    payload["router_structure"],
                    payload["router_training"],
                )
            ),
            "optimizer": state_sha256(payload["optimizer_state"]),
            "replay": state_sha256(payload["replay_records"]),
            "memory": state_sha256(payload["memory"]),
            "configuration": state_sha256(payload["settings"]),
            "rng": state_sha256(payload["rng_state"]),
        }

    def snapshot(self) -> ConsolidationSnapshot:
        """Capture every mutable state before candidate fitting."""
        payload = self._payload()
        return ConsolidationSnapshot(payload, self._hash_payload(payload))

    def state_hashes(self) -> dict[str, str]:
        """Return live transaction hashes, including all RNG generators."""
        payload = self._payload()
        return self._hash_payload(payload)

    def restore(self, snapshot: ConsolidationSnapshot) -> dict[str, str]:
        """Restore and verify every snapshotted state component."""
        payload = snapshot.payload
        self.settings = StandardExpertSettings.model_validate(payload["settings"])
        self.bank.settings = self.settings
        self.bank.model.load_state_dict(payload["model_state"])
        self.bank.mature_adapters = set(payload["mature_adapters"])
        self.bank.trainable_adapter = payload["trainable_adapter"]
        active_adapters = tuple(payload["active_adapters"])
        if len(active_adapters) != 1:
            raise RuntimeError("rollback requires exactly one snapshotted active adapter")
        self.bank.activate(active_adapters[0])
        if not payload["adapters_enabled"]:
            self.bank.deactivate()
        for name, parameter in self.bank.model.named_parameters():
            parameter.requires_grad_(payload["model_requires_grad"][name])
        self.bank.model.train(bool(payload["model_training"]))
        self.router.load_state_dict(payload["router_state"])
        structure = payload["router_structure"]
        self.router.expert_ids = list(structure["expert_ids"])
        self.router.task_families = list(structure["task_families"])
        self.router.frozen_rows = list(structure["frozen_rows"])
        for index, frozen in enumerate(self.router.frozen_rows):
            self.router.weights[index].requires_grad_(not frozen)
            self.router.biases[index].requires_grad_(not frozen)
        self.router.train(bool(payload["router_training"]))
        if self._router_structure() != structure:
            raise RuntimeError("router topology changed outside the rollback boundary")
        self.adapter_optimizer.load_state_dict(payload["optimizer_state"]["adapter"])
        router_state = payload["optimizer_state"]["router"]
        if router_state is None:
            self.router_optimizer = None
        else:
            self.router_optimizer = torch.optim.AdamW(
                self.router.candidate_parameters(),
                lr=self.settings.router.learning_rate,
                betas=(
                    self.settings.router.adam_beta1,
                    self.settings.router.adam_beta2,
                ),
                eps=self.settings.router.adam_epsilon,
                weight_decay=self.settings.router.weight_decay,
            )
            self.router_optimizer.load_state_dict(router_state)
        replay = BoundedReplayBuffer(self._replay_settings())
        records = cast(tuple[ReplayRecord, ...], payload["replay_records"])
        if records:
            replay.add_instances(record.to_instance() for record in records)
        self.replay = replay
        self.memory = tuple(payload["memory"])
        _restore_rng_state(self.bank.base.device, payload["rng_state"])
        restored = self.state_hashes()
        if restored != snapshot.hashes:
            raise RuntimeError("atomic consolidation rollback hash verification failed")
        return restored

    def evaluate_growth(
        self,
        routes: tuple[RouteDecision, ...],
        verifier_outcomes: tuple[VerifierOutcome, ...],
        expected_expert_ids: tuple[str | None, ...],
    ) -> GrowthDecision:
        """Evaluate registered growth thresholds from real held-out evidence."""
        config = self.settings.growth
        policy = EvidenceGrowthPolicy(
            GrowthSettings(
                verifier_failure_threshold=config.verifier_failure_threshold,
                mean_entropy_threshold=config.mean_entropy_threshold,
                minimum_observations=config.minimum_observations,
                max_experts=self.settings.lora.max_adapters,
            )
        )
        evidence = build_growth_evidence(routes, verifier_outcomes, expected_expert_ids)
        return policy.decide(self.router.num_experts, evidence)

    def train_candidate(
        self,
        current_instances: list[TaskInstance],
        current_features: torch.Tensor,
        replay_features: torch.Tensor,
        replay_labels: torch.Tensor,
    ) -> ConsolidationTrainingResult:
        """Fit the real active PEFT adapter and new router row within registered caps."""
        if not current_instances:
            raise ValueError("candidate adapter training requires current-task examples")
        self.bank.activate(cast(str, self.bank.trainable_adapter))
        total_loss = 0.0
        steps = 0
        target_tokens = 0
        mature_hashes = {
            name: self.bank.adapter_parameter_sha256(name)
            for name in self.bank.mature_adapters
        }
        base_hash = self.bank.base_parameter_sha256()
        for instance in current_instances:
            if steps >= self.settings.consolidation.max_adapter_optimizer_steps:
                break
            batch = collate_task_instances(self.bank.base, [instance])
            batch_tokens = int((batch.labels[:, 1:] != -100).sum().item())
            if target_tokens + batch_tokens > self.settings.consolidation.max_target_tokens:
                raise RuntimeError("candidate target-token cap would be exceeded")
            self.adapter_optimizer.zero_grad(set_to_none=True)
            output = self.bank.model(
                input_ids=batch.input_ids,
                attention_mask=batch.attention_mask,
                labels=batch.labels,
                use_cache=False,
                return_dict=True,
            )
            loss = output.loss
            if loss is None or not bool(torch.isfinite(loss).item()):
                raise FloatingPointError("candidate adapter loss is absent or non-finite")
            loss.backward()
            self.adapter_optimizer.step()
            total_loss += float(loss.detach().cpu().item())
            target_tokens += batch_tokens
            steps += 1
        if steps == 0:
            raise RuntimeError("candidate adapter schedule produced no optimizer step")
        candidate_index = self.router.num_experts - 1
        if candidate_index > 0:
            expected_labels = set(range(candidate_index))
            actual_labels = {int(value) for value in replay_labels.tolist()}
            if actual_labels != expected_labels:
                raise ValueError("router replay must contain every prior expert row")
            expected_count = self.settings.router.replay_examples_per_expert
            if any(
                int((replay_labels == label).sum().item()) != expected_count
                for label in expected_labels
            ):
                raise ValueError(
                    "router replay count must match replay_examples_per_expert for every row"
                )
        router_result, self.router_optimizer = self.router.fit_candidate(
            current_features,
            replay_features,
            replay_labels,
            learning_rate=self.settings.router.learning_rate,
            weight_decay=self.settings.router.weight_decay,
            adam_beta1=self.settings.router.adam_beta1,
            adam_beta2=self.settings.router.adam_beta2,
            adam_epsilon=self.settings.router.adam_epsilon,
            max_steps=self.settings.router.max_optimizer_steps,
        )
        if self.bank.base_parameter_sha256() != base_hash:
            raise RuntimeError("frozen base changed during candidate training")
        for name, digest in mature_hashes.items():
            if self.bank.adapter_parameter_sha256(name) != digest:
                raise RuntimeError("mature PEFT expert changed during candidate training")
        return ConsolidationTrainingResult(
            adapter_loss=total_loss / steps,
            adapter_optimizer_steps=steps,
            target_tokens=target_tokens,
            router=router_result,
        )

    def admit_verified_memory(
        self,
        instances: tuple[TaskInstance, ...],
        outcomes: tuple[VerifierOutcome, ...],
    ) -> None:
        """Admit only real verifier passes; infrastructure failure aborts the cycle."""
        if len(instances) != len(outcomes):
            raise ValueError("memory instances and verifier outcomes must align")
        records = list(self.memory)
        existing = {record.task_id for record in records}
        for instance, outcome in zip(instances, outcomes, strict=True):
            if outcome.infrastructure_error is not None:
                raise VerifierInfrastructureError(outcome.infrastructure_error)
            if outcome.passed and instance.task_id not in existing:
                records.append(
                    VerifiedMemoryRecord(
                        instance.task_id,
                        instance.task_family,
                        instance.prompt,
                        instance.target_solution,
                    )
                )
                existing.add(instance.task_id)
        self.memory = tuple(sorted(records, key=lambda record: record.task_id))

    def evaluate_promotion(
        self,
        current_outcomes: tuple[VerifierOutcome, ...],
        candidate_outcomes: tuple[VerifierOutcome, ...],
        current_losses: np.ndarray,
        candidate_losses: np.ndarray,
    ) -> PromotionResult:
        """Apply the fail-closed promotion rule to real verifier outcomes."""
        if len(current_outcomes) != len(candidate_outcomes):
            raise ValueError("current and candidate verifier evidence must align")
        config = self.settings.consolidation
        gate = PromotionGate(
            alpha=config.promotion_alpha,
            minimum_regression_samples=config.minimum_regression_samples,
            minimum_validation_gain=config.minimum_validation_gain,
            loss_noninferiority_margin=config.loss_noninferiority_margin,
        )
        return gate.evaluate_measurements(
            current_val_accuracy=_verifier_accuracy(current_outcomes),
            candidate_val_accuracy=_verifier_accuracy(candidate_outcomes),
            current_losses=current_losses,
            candidate_losses=candidate_losses,
        )

    def finalize(
        self,
        snapshot: ConsolidationSnapshot,
        promotion: PromotionResult,
        current_instances: list[TaskInstance],
        output_dir: Path,
    ) -> dict[str, str]:
        """Rollback rejection or atomically persist an accepted restartable cycle."""
        if not promotion.promoted:
            return self.restore(snapshot)
        self.replay.add_instances(current_instances)
        output_dir.parent.mkdir(parents=True, exist_ok=True)
        if output_dir.exists():
            raise FileExistsError("accepted consolidation artifact is immutable")
        stage = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.", dir=output_dir.parent))
        try:
            optimizer_state = copy.deepcopy(self._optimizer_state())
            rng_state = _capture_rng_state(self.bank.base.device)
            candidate = self.bank.trainable_adapter
            if candidate is None:
                raise RuntimeError("accepted consolidation has no candidate to promote")
            experts_dir = stage / "experts"
            expert_manifests: list[ExpertManifest] = []
            for expert_id in self.router.expert_ids:
                artifact = experts_dir / expert_id
                if expert_id == candidate:
                    expert_manifests.append(self.bank.promote_candidate(artifact))
                else:
                    expert_manifests.append(self.bank.persist_adapter(expert_id, artifact))
            self.router.promote_candidate()
            checkpoint = {
                "schema_version": _STATE_SCHEMA,
                "router_state": self.router.state_dict(),
                "router_structure": self._router_structure(),
                "optimizer_state": optimizer_state,
                "rng_state": rng_state,
            }
            state_path = stage / "state.pt"
            torch.save(checkpoint, state_path)
            replay_payload = [asdict(record) for record in self.replay.records]
            memory_payload = [asdict(record) for record in self.memory]
            replay_path = stage / "replay.json"
            memory_path = stage / "memory.json"
            config_path = stage / "config.json"
            promotion_path = stage / "promotion.json"
            replay_path.write_text(
                json.dumps(replay_payload, sort_keys=True) + "\n", encoding="utf-8"
            )
            memory_path.write_text(
                json.dumps(memory_payload, sort_keys=True) + "\n", encoding="utf-8"
            )
            config_path.write_text(
                json.dumps(self.settings.model_dump(mode="json"), sort_keys=True) + "\n",
                encoding="utf-8",
            )
            promotion_path.write_text(
                json.dumps(asdict(promotion), sort_keys=True) + "\n", encoding="utf-8"
            )
            hashes = {
                "experts": state_sha256(
                    tuple(
                        (manifest.adapter_name, manifest.content_sha256)
                        for manifest in expert_manifests
                    )
                ),
                "state_file": _sha256_file(state_path),
                "router": state_sha256(
                    (checkpoint["router_state"], checkpoint["router_structure"])
                ),
                "optimizer": state_sha256(optimizer_state),
                "rng": state_sha256(rng_state),
                "replay": _sha256_file(replay_path),
                "memory": _sha256_file(memory_path),
                "config": _sha256_file(config_path),
                "promotion_evidence": _sha256_file(promotion_path),
            }
            manifest = {
                "schema_version": _MANIFEST_SCHEMA,
                "base_fingerprint": standard_base_fingerprint(self.bank.base.settings),
                "expert_manifests": [
                    expert_manifest.model_dump(mode="json")
                    for expert_manifest in expert_manifests
                ],
                "hashes": hashes,
            }
            (stage / "manifest.json").write_text(
                json.dumps(manifest, sort_keys=True) + "\n", encoding="utf-8"
            )
            os.replace(stage, output_dir)
            return hashes
        except Exception:
            self.restore(snapshot)
            raise
        finally:
            if stage.exists():
                shutil.rmtree(stage)


def load_accepted_cycle(
    base: StandardBase,
    settings: StandardExpertSettings,
    artifact_dir: Path,
) -> AcceptedCycle:
    """Hash-verify and reload an accepted cycle without relying on live process state."""
    manifest_raw: object = json.loads(
        (artifact_dir / "manifest.json").read_text(encoding="utf-8")
    )
    if not isinstance(manifest_raw, dict) or manifest_raw.get("schema_version") != _MANIFEST_SCHEMA:
        raise ValueError("accepted consolidation manifest schema is invalid")
    if manifest_raw.get("base_fingerprint") != standard_base_fingerprint(base.settings):
        raise ValueError("accepted consolidation base fingerprint mismatch")
    hashes = cast(dict[str, str], manifest_raw["hashes"])
    state_path = artifact_dir / "state.pt"
    replay_path = artifact_dir / "replay.json"
    memory_path = artifact_dir / "memory.json"
    config_path = artifact_dir / "config.json"
    promotion_path = artifact_dir / "promotion.json"
    if _sha256_file(state_path) != hashes["state_file"]:
        raise ValueError("accepted consolidation state-file hash mismatch")
    if _sha256_file(replay_path) != hashes["replay"]:
        raise ValueError("accepted consolidation replay hash mismatch")
    if _sha256_file(memory_path) != hashes["memory"]:
        raise ValueError("accepted consolidation memory hash mismatch")
    if _sha256_file(config_path) != hashes["config"]:
        raise ValueError("accepted consolidation config hash mismatch")
    if _sha256_file(promotion_path) != hashes["promotion_evidence"]:
        raise ValueError("accepted consolidation promotion-evidence hash mismatch")
    persisted_settings = StandardExpertSettings.model_validate_json(
        config_path.read_text(encoding="utf-8")
    )
    if persisted_settings != settings:
        raise ValueError("accepted consolidation settings do not match the requested runtime")
    checkpoint: dict[str, Any] = torch.load(
        state_path, map_location="cpu", weights_only=True
    )
    structure = checkpoint["router_structure"]
    expert_manifests = tuple(
        ExpertManifest.model_validate(item)
        for item in manifest_raw["expert_manifests"]
    )
    if not expert_manifests:
        raise ValueError("accepted consolidation contains no promoted experts")
    expert_ids = tuple(str(item) for item in structure["expert_ids"])
    if tuple(manifest.adapter_name for manifest in expert_manifests) != expert_ids:
        raise ValueError("accepted consolidation expert/router order mismatch")
    first, *additional = expert_manifests
    bank = load_promoted_expert(base, settings, artifact_dir / "experts" / first.adapter_name)
    loaded_manifests = [first]
    for expected in additional:
        loaded = load_additional_promoted_expert(
            bank, artifact_dir / "experts" / expected.adapter_name
        )
        if loaded != expected:
            raise ValueError("accepted consolidation expert manifest mismatch")
        loaded_manifests.append(loaded)
    expert_hash = state_sha256(
        tuple(
            (manifest.adapter_name, manifest.content_sha256)
            for manifest in loaded_manifests
        )
    )
    if expert_hash != hashes["experts"]:
        raise ValueError("accepted consolidation expert hash mismatch")
    router = SequenceTop1Router(
        hidden_size=int(structure["hidden_size"]),
        confidence_threshold=float(structure["confidence_threshold"]),
    )
    for expert_id, task_family, frozen in zip(
        structure["expert_ids"],
        structure["task_families"],
        structure["frozen_rows"],
        strict=True,
    ):
        router.add_candidate(str(expert_id), str(task_family))
        if frozen:
            router.promote_candidate()
    router.load_state_dict(checkpoint["router_state"])
    router_hash = state_sha256((router.state_dict(), structure))
    if router_hash != hashes["router"]:
        raise ValueError("accepted consolidation router hash mismatch")
    if state_sha256(checkpoint["optimizer_state"]) != hashes["optimizer"]:
        raise ValueError("accepted consolidation optimizer hash mismatch")
    if state_sha256(checkpoint["rng_state"]) != hashes["rng"]:
        raise ValueError("accepted consolidation RNG hash mismatch")
    replay_json = json.loads(replay_path.read_text(encoding="utf-8"))
    replay = BoundedReplayBuffer(
        ReplaySettings(
            capacity=settings.consolidation.replay_capacity,
            ratio=settings.consolidation.replay_ratio,
            sampling_seed=settings.consolidation.replay_sampling_seed,
            per_family_minimum=settings.consolidation.replay_per_family_minimum,
            byte_limit=settings.consolidation.replay_byte_limit,
            eviction_policy=settings.consolidation.replay_eviction_policy,
        )
    )
    records = tuple(
        ReplayRecord(
            task_id=item["task_id"],
            task_family=item["task_family"],
            prompt=item["prompt"],
            target_solution=item["target_solution"],
            answer_stop_characters=tuple(item["answer_stop_characters"]),
            training_terminator=item["training_terminator"],
        )
        for item in replay_json
    )
    if records:
        replay.add_instances(record.to_instance() for record in records)
    memory_json = json.loads(memory_path.read_text(encoding="utf-8"))
    memory = tuple(VerifiedMemoryRecord(**item) for item in memory_json)
    return AcceptedCycle(
        bank=bank,
        router=router,
        replay=replay,
        memory=memory,
        optimizer_state=checkpoint["optimizer_state"],
        rng_state=checkpoint["rng_state"],
        state_hashes=hashes,
    )

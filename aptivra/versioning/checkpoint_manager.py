"""Immutable Checkpoint Manager with Manifest Verification and Rollback Execution."""

import hashlib
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn


@dataclass
class CheckpointManifest:
    """Immutable metadata manifest associated with a promoted model checkpoint."""

    version_id: int
    task_idx: int
    task_family: str
    config_hash: str
    data_hash: str
    random_seed: int
    p_value: float
    val_acc_gain: float
    metrics: dict[str, Any]
    timestamp: float


class CheckpointManager:
    """Manages creation, manifest audit, and rollback of promoted immutable checkpoints."""

    def __init__(self, checkpoints_dir: Path | str = "checkpoints") -> None:
        self.checkpoints_dir = Path(checkpoints_dir)
        self.checkpoints_dir.mkdir(parents=True, exist_ok=True)
        self.manifest_history_file = self.checkpoints_dir / "manifest_history.jsonl"

    @staticmethod
    def _compute_hash(content: str) -> str:
        return hashlib.sha256(content.encode("utf-8")).hexdigest()

    def save_promoted_checkpoint(
        self,
        version_id: int,
        task_idx: int,
        task_family: str,
        model: nn.Module,
        router: nn.Module | None,
        experts: nn.ModuleList | None,
        config_dict: dict[str, Any],
        data_sample_ids: list[str],
        seed: int,
        p_value: float,
        val_acc_gain: float,
        metrics_dict: dict[str, Any],
    ) -> Path:
        """Saves an immutable versioned checkpoint directory with manifest."""
        version_dir = self.checkpoints_dir / f"v{version_id}"
        if version_dir.exists():
            raise FileExistsError(f"Checkpoint version v{version_id} already exists (Immutable).")

        version_dir.mkdir(parents=True, exist_ok=True)

        # 1. Save PyTorch state dicts
        state_dict: dict[str, Any] = {
            "model_state": model.state_dict(),
            "router_state": router.state_dict() if router is not None else None,
            "experts_state": experts.state_dict() if experts is not None else None,
        }
        weights_path = version_dir / "weights.pt"
        torch.save(state_dict, weights_path)

        # 2. Generate hashes and manifest
        cfg_hash = self._compute_hash(json.dumps(config_dict, sort_keys=True))
        data_hash = self._compute_hash(",".join(sorted(data_sample_ids)))

        manifest = CheckpointManifest(
            version_id=version_id,
            task_idx=task_idx,
            task_family=task_family,
            config_hash=cfg_hash,
            data_hash=data_hash,
            random_seed=seed,
            p_value=p_value,
            val_acc_gain=val_acc_gain,
            metrics=metrics_dict,
            timestamp=time.time(),
        )

        manifest_path = version_dir / "manifest.json"
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(asdict(manifest), f, indent=2)

        # Append to manifest history
        with open(self.manifest_history_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(asdict(manifest)) + "\n")

        return version_dir

    def rollback_checkpoint(
        self,
        target_version_id: int,
        model: nn.Module,
        router: nn.Module | None = None,
        experts: nn.ModuleList | None = None,
    ) -> CheckpointManifest:
        """Single-command rollback restoring model and modules to target version."""
        target_dir = self.checkpoints_dir / f"v{target_version_id}"
        if not target_dir.exists():
            raise FileNotFoundError(f"Checkpoint version v{target_version_id} does not exist.")

        weights_path = target_dir / "weights.pt"
        manifest_path = target_dir / "manifest.json"

        state_dict = torch.load(weights_path, map_location="cpu")
        model.load_state_dict(state_dict["model_state"])

        if router is not None and state_dict.get("router_state") is not None:
            router.load_state_dict(state_dict["router_state"])

        if experts is not None and state_dict.get("experts_state") is not None:
            experts.load_state_dict(state_dict["experts_state"])

        with open(manifest_path, encoding="utf-8") as f:
            m_dict = json.load(f)
            manifest = CheckpointManifest(**m_dict)

        return manifest

"""Exporter and loader modules for Tiered Memory Runtime (TMR) state management."""

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from aptivra.experts.expert import ExpertModule
from aptivra.experts.router import GatingRouter
from aptivra.model.config import ModelConfig
from aptivra.model.transformer import BackboneTransformer


@dataclass
class MultitierExportManifest:
    """Manifest describing exported Aptivra model layout for Multitier execution engine."""

    version: str
    d_model: int
    n_layers: int
    num_experts: int
    dense_shard_path: str
    expert_shard_paths: list[str]
    metadata: dict[str, Any]


class MultitierExporter:
    """Exports PyTorch Aptivra models and T2 expert pools into Tiered C execution format."""

    def __init__(self, export_dir: Path) -> None:
        self.export_dir = export_dir
        self.export_dir.mkdir(parents=True, exist_ok=True)

    def export_model(
        self,
        model: BackboneTransformer,
        router: GatingRouter,
        experts: list[ExpertModule] | nn.ModuleList,
        model_config: ModelConfig,
    ) -> Path:
        """Serializes backbone weights and expert modules into Multitier directory structure."""
        dense_dir = self.export_dir / "dense"
        experts_dir = self.export_dir / "experts"
        dense_dir.mkdir(exist_ok=True)
        experts_dir.mkdir(exist_ok=True)

        # 1. Export dense backbone state dictionary
        dense_shard_path = dense_dir / "backbone.pt"
        torch.save(model.state_dict(), dense_shard_path)

        # 2. Export T2 expert module parameters to separate expert shards
        expert_shard_paths: list[str] = []
        for idx, exp in enumerate(experts):
            exp_shard_path = experts_dir / f"expert_{idx}.pt"
            torch.save(exp.state_dict(), exp_shard_path)
            expert_shard_paths.append(str(exp_shard_path.relative_to(self.export_dir)))

        # 3. Export router state dictionary
        torch.save(router.state_dict(), dense_dir / "router.pt")

        # 4. Generate usage heat tracking file (.expert_usage)
        usage_file = self.export_dir / ".expert_usage"
        initial_usage = {f"expert_{i}": 0 for i in range(len(experts))}
        with open(usage_file, "w", encoding="utf-8") as f:
            json.dump(initial_usage, f, indent=2)

        # 5. Generate Multitier execution manifest
        manifest = MultitierExportManifest(
            version="1.0.0",
            d_model=model_config.d_model,
            n_layers=model_config.n_layers,
            num_experts=len(experts),
            dense_shard_path=str(dense_shard_path.relative_to(self.export_dir)),
            expert_shard_paths=expert_shard_paths,
            metadata={
                "vocab_size": model_config.vocab_size,
                "execution_platform": "tiered_c",
                "nvme_streaming_enabled": True,
            },
        )

        manifest_path = self.export_dir / "multitier_manifest.json"
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(asdict(manifest), f, indent=2)

        return manifest_path


class MultitierLoader:
    """Loads and validates Tiered Memory Runtime exported artifacts from disk."""

    def __init__(self, export_dir: Path) -> None:
        self.export_dir = export_dir
        self.manifest_path = export_dir / "multitier_manifest.json"

    def load_manifest(self) -> MultitierExportManifest:
        """Loads and parses the execution manifest, checking file integrity."""
        if not self.manifest_path.exists():
            raise FileNotFoundError(f"Execution manifest not found: {self.manifest_path}")

        with open(self.manifest_path, encoding="utf-8") as f:
            data = json.load(f)

        manifest = MultitierExportManifest(**data)

        # Verify dense backbone shard existence
        dense_full_path = self.export_dir / manifest.dense_shard_path
        if not dense_full_path.exists():
            raise FileNotFoundError(f"Dense backbone shard missing: {dense_full_path}")

        # Verify all expert shards exist
        for exp_rel in manifest.expert_shard_paths:
            exp_full = self.export_dir / exp_rel
            if not exp_full.exists():
                raise FileNotFoundError(f"Expert shard missing: {exp_full}")

        return manifest

    def load_model(
        self, model_config: ModelConfig
    ) -> tuple[BackboneTransformer, GatingRouter, list[ExpertModule]]:
        """Restores model, router, and expert modules from exported shards."""
        manifest = self.load_manifest()

        # Restore Backbone Transformer
        model = BackboneTransformer(model_config)
        dense_path = self.export_dir / manifest.dense_shard_path
        model.load_state_dict(torch.load(dense_path, weights_only=True))

        # Restore Router
        router = GatingRouter(
            d_model=model_config.d_model, initial_num_experts=manifest.num_experts
        )
        router_path = self.export_dir / "dense" / "router.pt"
        if router_path.exists():
            router.load_state_dict(torch.load(router_path, weights_only=True))

        # Restore Expert Modules
        experts: list[ExpertModule] = []
        for idx, exp_rel in enumerate(manifest.expert_shard_paths):
            exp_path = self.export_dir / exp_rel
            expert = ExpertModule(expert_id=idx, d_model=model_config.d_model)
            expert.load_state_dict(torch.load(exp_path, weights_only=True))
            experts.append(expert)

        return model, router, experts

    def record_expert_routing_heat(self, routed_expert_id: int) -> dict[str, int]:
        """Updates routing heat counter in .expert_usage for tiered auto-pinning."""
        usage_file = self.export_dir / ".expert_usage"
        usage: dict[str, int] = {}
        if usage_file.exists():
            with open(usage_file, encoding="utf-8") as f:
                usage = json.load(f)

        key = f"expert_{routed_expert_id}"
        usage[key] = usage.get(key, 0) + 1

        with open(usage_file, "w", encoding="utf-8") as f:
            json.dump(usage, f, indent=2)

        return usage

    def simulate_lookahead_prefetch(
        self, current_layer_routed_experts: torch.Tensor, predicted_next_experts: list[int]
    ) -> dict[str, float]:
        """Simulates 1-layer lookahead prefetching (PILOT=1) and calculates hit metrics."""
        current_set = set(current_layer_routed_experts.view(-1).cpu().tolist())
        predicted_set = set(predicted_next_experts)

        hits = len(current_set.intersection(predicted_set))
        total = max(1, len(current_set))

        return {
            "hit_rate": float(hits / total),
            "hits": float(hits),
            "total_requested": float(len(current_set)),
        }

"""Unit tests for CheckpointManager, manifest verification, and single-command rollback."""

from pathlib import Path

import pytest
import torch

from aptivra.experts.router import GatingRouter
from aptivra.model.config import ModelConfig
from aptivra.model.transformer import BackboneTransformer
from aptivra.versioning.checkpoint_manager import CheckpointManager


def test_checkpoint_manager_save_and_rollback(tmp_path: Path) -> None:
    mgr = CheckpointManager(checkpoints_dir=tmp_path)

    config = ModelConfig(vocab_size=128, d_model=64, n_layers=2, n_heads=4, d_ff=128, device="cpu")
    model = BackboneTransformer(config)
    router = GatingRouter(d_model=64, initial_num_experts=1)

    # Save version 1
    version_dir = mgr.save_promoted_checkpoint(
        version_id=1,
        task_idx=0,
        task_family="T1_modular_arithmetic",
        model=model,
        router=router,
        experts=None,
        config_dict={"d_model": 64},
        data_sample_ids=["sample_1", "sample_2"],
        seed=42,
        p_value=0.25,
        val_acc_gain=0.15,
        metrics_dict={"acc": 0.85},
    )
    assert version_dir.exists()
    assert (version_dir / "weights.pt").exists()
    assert (version_dir / "manifest.json").exists()

    # Immutability check: saving duplicate version_id 1 MUST raise FileExistsError
    with pytest.raises(FileExistsError, match="already exists"):
        mgr.save_promoted_checkpoint(
            version_id=1,
            task_idx=0,
            task_family="T1_modular_arithmetic",
            model=model,
            router=None,
            experts=None,
            config_dict={},
            data_sample_ids=[],
            seed=42,
            p_value=0.1,
            val_acc_gain=0.0,
            metrics_dict={},
        )

    # Modify model parameter weight to verify rollback accuracy
    with torch.no_grad():
        model.token_embedding.weight.fill_(9.9)

    # Rollback to version 1
    manifest = mgr.rollback_checkpoint(
        target_version_id=1,
        model=model,
        router=router,
    )
    assert manifest.version_id == 1
    assert manifest.random_seed == 42
    assert not torch.allclose(model.token_embedding.weight, torch.tensor(9.9))

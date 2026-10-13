"""Model Configuration and Device Resolution."""

from dataclasses import dataclass

import torch


@dataclass
class ModelConfig:
    """Configuration dataclass for backbone transformer and device resolution."""

    vocab_size: int = 256
    d_model: int = 512
    n_layers: int = 8
    n_heads: int = 8
    d_ff: int = 2048
    max_seq_len: int = 512
    dropout: float = 0.1
    device: str = "auto"
    use_pretrained_embeddings: bool = False
    pretrained_embedding_dim: int = 512
    freeze_pretrained_embeddings: bool = False

    def get_device(self) -> torch.device:
        """Resolves auto device setting to CUDA, MPS, or CPU in priority order."""
        if self.device == "cuda" and torch.cuda.is_available():
            return torch.device("cuda")
        if self.device == "mps" and torch.backends.mps.is_available():
            return torch.device("mps")
        if self.device == "cpu":
            return torch.device("cpu")

        # Auto detection
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    @property
    def param_count_estimate(self) -> int:
        """Estimates total parameter count for decoder transformer."""
        emb_params = self.vocab_size * self.d_model
        # Per layer: Attn (4 * d_model^2), FFN (3 * d_model * d_ff), 2x LayerNorm (2 * d_model)
        layer_params = (4 * self.d_model**2) + (3 * self.d_model * self.d_ff) + (4 * self.d_model)
        total = emb_params + (self.n_layers * layer_params) + emb_params
        return total

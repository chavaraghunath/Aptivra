"""Hub-side Transformers config (uploaded next to multitier shards)."""

from transformers import PretrainedConfig


class AptivraConfig(PretrainedConfig):
    """Configuration for Aptivra tiered-memory MoE causal LMs."""

    model_type = "aptivra"

    def __init__(
        self,
        vocab_size: int = 256,
        d_model: int = 1024,
        n_layers: int = 16,
        n_heads: int = 16,
        d_ff: int = 4096,
        max_seq_len: int = 256,
        num_experts: int = 8,
        top_k_experts: int = 2,
        dropout: float = 0.1,
        execution_platform: str = "pytorch",
        **kwargs: object,
    ) -> None:
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.d_ff = d_ff
        self.max_seq_len = max_seq_len
        self.num_experts = num_experts
        self.top_k_experts = top_k_experts
        self.dropout = dropout
        self.execution_platform = execution_platform

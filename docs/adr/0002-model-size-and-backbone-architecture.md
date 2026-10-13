# ADR-0002: Model Size and Backbone Architecture

- **Status:** Accepted
- **Date:** 2026-09-27
- **Authors:** Aptivra Lead Research Engineer

## Context

Phase 1 of Aptivra investigates Hypothesis H1 within a parameter scale of 50M to 200M parameters. The backbone (Tier 3, T3) serves as the core decoder-only transformer representation model. Additionally, Tier 0 (T0, context layer) requires a modular abstraction around the attention mechanism so that standard multi-head self-attention with KV caching can be seamlessly swapped with alternative fast-weight / delta-rule layers (such as KDA-style linear attention) in future phases.

## Decision

1. **Backbone Scale & Architecture:**
   - Default target size: ~85M parameters (Configurable via YAML between 50M and 200M).
   - Architecture: Decoder-only autoregressive Transformer with RMSNorm / LayerNorm, Rotary Position Embeddings (RoPE) or learned positional embeddings, and SwiGLU / Gated MLP blocks.
   - Hyperparameter ranges:
     - `d_model`: 512 (50M) to 1024 (200M)
     - `n_layers`: 8 to 16
     - `n_heads`: 8 to 16
     - `d_ff`: $4 \times d_{model}$ (or $2.67 \times d_{model}$ for SwiGLU)
2. **T0 Attention Module Isolation (`AttentionInterface`):**
   - An abstract base class `AttentionInterface(nn.Module)` defines the standard contract:
     - `forward(x, context=None, attention_mask=None, kv_cache=None) -> (output, new_kv_cache)`
   - The default implementation `StandardCausalAttention` implements scaled dot-product self-attention with standard causal masking and key-value caching.
   - All attention calls in T3 backbone layers must route through `AttentionInterface`, ensuring zero code changes in the backbone when a delta-rule layer (e.g. KDA) is plugged in.

## Consequences

### Positive
- Strict isolation of attention via `AttentionInterface` enforces modularity and future-proofs the codebase for non-standard attention / state-space / linear-attention mechanics.
- The 50M–200M scale ensures fast training iteration cycles on local dev hardware (MPS) while being expressive enough to learn non-trivial symbolic and algorithmic tasks in the 8-task continual learning stream.
- Parameter count is fully config-driven, enabling precise compute matching across baselines and ablation variants.

### Negative / Trade-offs
- The abstraction layer adds a minor indirection in the forward pass, though overhead in PyTorch is negligible.
- Small models (<200M) are known to suffer more severely from catastrophic forgetting than multi-billion parameter models, placing high demands on the efficiency of T1 episodic memory and T2 expert consolidation.

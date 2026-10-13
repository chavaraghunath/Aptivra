# ADR-0004: Episodic Store Retrieval Method

- **Status:** Accepted
- **Date:** 2026-09-27
- **Authors:** Aptivra Lead Research Engineer

## Context

Tier 1 (T1) acts as an external episodic store holding verified records of solved problems. T1 prevents contaminated or hallucinated data from entering long-term memory and provides retrieved samples for both prompt-based in-context retrieval and target consolidation cycles.

## Decision

1. **Gated Write Filter (Strict Verification Gate):**
   - A record $R = (x, y, v\_result, \text{meta})$ is written to T1 **if and only if** the task verifier returns $v\_result = 1$ (Passed). Unverified or failing model outputs are strictly rejected at the store write interface.
2. **Representation & Indexing:**
   - Text inputs $x$ are encoded into dense vectors using an embedding module (e.g. lightweight pretrained transformer encoder or frozen backbone token embeddings pooled across sequence length).
   - Vector indexing uses normalized cosine similarity: $S(q, d) = \frac{e(q) \cdot e(d)}{\|e(q)\| \|e(d)\|}$.
   - For fast in-memory execution in PyTorch (given store sizes typical of 50M–200M parameter model experiments, e.g., $10^3 - 10^5$ items), cosine matrix multiplication via PyTorch tensors on MPS/CUDA will be used, with an abstraction layer `VectorStore` to allow FAISS/HNSW indexing if store size exceeds memory limits.
3. **Retrieval & Consolidation Selection:**
   - In-context retrieval fetches top-$k$ nearest neighbors for prompt enrichment.
   - Every record tracks `retrieval_count` and `verification_count`.
   - Consolidation selection picks records with high retrieval frequency and high verification ratio ($v\_result = 1$) to populate consolidation training batches.

## Consequences

### Positive
- Zero risk of training memory consolidation on corrupt or incorrect outputs, enforcing strict input-to-verifier correctness.
- Vector retrieval enables semantic clustering of task inputs, matching incoming problems with structurally similar past verified solutions.
- Simple, high-speed PyTorch tensor vector operations keep Phase 1 dependencies lightweight without requiring external database services.

### Negative / Trade-offs
- Embedding model quality directly impacts retrieval accuracy. If input encoding fails to capture task similarity, retrieval effectiveness drops.
- Memory store grows linearly with solved samples; maximum store capacity and pruning/eviction strategies (e.g. LRU or utility-based pruning) must be configured.

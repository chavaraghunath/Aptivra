# ADR-0003: Expert Block Design and Router Architecture

- **Status:** Accepted
- **Date:** 2026-09-27
- **Authors:** Aptivra Lead Research Engineer

## Context

To support dynamic capacity expansion during continual learning without corrupting previously acquired skills, Aptivra uses a Tier 2 (T2) growable pool of expert modules controlled by a module-level router. As new task families are encountered, existing expert weights must remain stable while new experts are initialized to absorb novel task distributions.

## Decision

1. **Expert Block Architecture:**
   - Each expert in T2 is implemented as a modular Feed-Forward / Adapter Block (`ExpertModule`) inserted at designated backbone layers (or as parallel FFN branches).
   - Each expert consists of a down-projection, non-linear activation (e.g., SiLU/GELU), and up-projection: $E_k(x) = W_{up}^{(k)} \sigma(W_{down}^{(k)} x)$.
2. **Router Mechanics & Freezing Policy:**
   - A module-level gating router computes a softmax probability distribution over active experts: $g(x) = \text{softmax}(W_{router} x)$.
   - When an expert $k$ reaches maturity (defined at the end of a task consolidation cycle), its parameters $E_k$ and its corresponding routing parameter vector $W_{router}[k, :]$ are **frozen** (`requires_grad = False`).
   - When a new expert $k_{new}$ is spawned, a new row is appended to $W_{router}$. Only $E_{k_{new}}$ and $W_{router}[k_{new}, :]$ (plus un-frozen experts if explicitly allowed by config) receive gradient updates during the new task cycle.
3. **Dynamic Spawning Triggers:**
   - Spawning is controlled by two configurable metric thresholds:
     - **Router Entropy Threshold ($H_{thresh}$):** High routing entropy across samples indicates that existing experts lack high confidence / specialization for the current input distribution.
     - **Verifier Failure Rate Threshold ($V_{fail\_thresh}$):** High error/failure rate on recent verified inputs from T1 triggers the allocation of dedicated capacity.
   - Threshold values are defined entirely in configuration files and never hard-coded.

## Consequences

### Positive
- Parameter freezing of mature experts guarantees zero direct interference (catastrophic forgetting) on the sub-network capacity allocated to earlier tasks.
- Spawning new experts provides bounded parameter growth proportional to the difficulty and novelty of the task stream.
- Router entropy and verifier failure metrics provide principled, mechanism-driven triggers for network expansion.

### Negative / Trade-offs
- Memory footprint and parameter count increase monotonically as new experts are spawned across 8 task families.
- Freezing router rows for mature experts may limit forward transfer if a new task could benefit from reusing a frozen expert's routing logits; top-$k$ soft routing or residual routing blending must handle reuse.

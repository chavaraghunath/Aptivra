# ADR-0006: Tiered Memory Runtime (TMR), NVMe Expert Streaming, and Pure C Execution Backend for $T_2$ Expert Modules

- **Status:** Approved
- **Date:** September 2026
- **Deciders:** Lead Research Engineer, Systems Architect

---

## Context and Problem Statement

Aptivra's continual learning architecture relies on a growable pool of expert modules ($T_2$) that dynamically spawns specialized sub-networks when router entropy or verifier failure rates cross target thresholds. Over long task streams, mature experts are frozen (`requires_grad = False`) and preserved to eliminate catastrophic forgetting.

As the expert pool grows (e.g. dozens or hundreds of specialized experts across layers), keeping 100% of $T_2$ expert parameters concurrently resident in fast GPU VRAM or system RAM creates a severe memory footprint barrier on consumer hardware and edge devices.

---

## Decision Outcome

We adopt the **Tiered Memory Runtime (TMR)** AI Memory Multitiering architecture and pure C execution runtime for Aptivra:

1. **AI Memory Multitiering ($VRAM \to RAM \to NVMe$):**
   - **Resident Dense Tier:** $T_0$ working context attention and $T_3$ consolidated decoder backbone weights remain permanently resident in fast memory (VRAM or system RAM).
   - **Streamed Sparse Tier:** Frozen $T_2$ expert modules live on NVMe storage and are read JIT on-demand during forward passes.

2. **Usage-Heat Expert Pinning (`.expert_usage`):**
   - The engine logs routing frequency across execution turns and automatically pins the highest-frequency expert weights into RAM/VRAM.

3. **1-Layer Lookahead Expert Prefetching (`PILOT=1`):**
   - Uses layer $L$ top-$k$ router probability vectors to asynchronously prefetch layer $L+1$ expert parameters from NVMe into memory while layer $L$ is computing.

4. **Pure C Export & Native Execution Engine:**
   - Historical Aptivra provides an export module (`aptivra.v0.export.multitier`) that serializes PyTorch models and $T_2$ expert pools into `.safetensors` structures for zero-dependency standalone execution via the pure C `tmr` binary.

---

## Technical Invariants & Guardrails

- **Zero Semantic Deviation:** Placement policy (VRAM vs RAM vs NVMe) controls execution speed only — model output logits, target verifications, and parameter precisions must remain token-exact.
- **Configurability:** Multi-tier streaming is exposed as a runtime configuration setting (`execution.platform: "tiered_c"`) and registered in `docs/configuration-register.md`.

---

## Consequences

- **Positive:** Enables execution of large growable expert pools on consumer hardware and laptops with limited VRAM.
- **Positive:** Zero runtime Python/PyTorch dependency for standalone high-speed inference deployment.
- **Negative / Tradeoff:** Staging experts from NVMe storage incurs read latency if NVMe bandwidth is constrained or if lookahead prediction misses.

---

## References

- [ADR-0003: Expert Block Design & Router Architecture](0003-expert-block-design-and-router-architecture.md)

# ADR-0001: Framework and Device Selection

- **Status:** Accepted
- **Date:** 2026-09-27
- **Authors:** Aptivra Lead Research Engineer

## Context

Aptivra requires a deep learning framework capable of supporting dynamic computational graphs, custom attention module abstractions, growable parameter pools, and flexible training loops (such as custom EWC regularization and promotion gate validation). The system must run seamlessly across Apple Silicon (MPS execution on macOS during local development) and NVIDIA GPUs (CUDA execution in cloud/cluster environments) without code modifications.

## Decision

We select **PyTorch 2.x** (Python 3.11+) as the foundational deep learning framework.

1. **Device Abstraction:** All model components, tensors, and trainers will query a centralized configuration parameter `device` (`"auto"`, `"cuda"`, `"mps"`, or `"cpu"`). When `"auto"` is specified, the system detects backend availability in priority order: CUDA -> MPS -> CPU.
2. **Device Neutrality:** Device assignment occurs strictly at object instantiation or data loading using `.to(device)` or PyTorch device context handlers. No hardcoded `.cuda()` or `.to("mps")` calls are permitted anywhere in code.
3. **PyTorch 2.0 Features:** Torch compile (`torch.compile`) will be optionally configurable for CUDA, while maintaining full fallback compatibility for MPS where dynamic graphing and compilation support may vary.

## Consequences

### Positive
- Identical Python code executes on Apple Silicon laptops (MPS) for unit tests and rapid prototyping, and on Linux CUDA servers for benchmark runs.
- Full access to PyTorch's rich ecosystem for custom gradient hooks, dynamic parameter registration (crucial for growable expert pools), and low-level matrix operation control.
- PyTorch 2.x dynamic graph support makes routing and variable-size memory buffers straightforward to implement cleanly.

### Negative / Trade-offs
- MPS backend occasionally has missing or non-deterministic PyTorch op implementations compared to CUDA (e.g., certain sparse matrix operations or specialized linear algebra ops).
- Fallbacks for non-supported ops on MPS must be guarded with CPU fallback options (`PYTORCH_ENABLE_MPS_FALLBACK=1`) during dev testing.

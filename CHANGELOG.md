# Changelog

All notable changes to Aptivra are documented here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/); this project does not
yet follow Semantic Versioning tags — see `git log` for the authoritative history until a first
tagged release exists.

## [Unreleased]

### Added
- A configurable, deterministic, resumable base-model candidate evaluator with isolated MPS
  processes, byte-normalized likelihood, 12-family learnability probes, structured failures,
  tokenizer/expert fitness checks, and a pre-registered ranking rule.
- Proposed ADR-0008 plus machine-readable per-candidate evidence and an immutable run manifest for
  the measured pretrained Tier 3 selection.
- Powered ADR-0008 evidence with 100 test items per family, warmup, batch-8 probes, batched
  generation, top-two three-seed variance, early hard-constraint exits, and archived first-sweep
  evidence. Post-result Amendment 2 adds paired family-clustered intervals and sign tests,
  saturated-family sensitivity, computed stream properties, validation-only future grid probes,
  and an explicit owner decision. The data give a cautious preference for SmolLM2-360M; clustered
  accuracy and gain tie before the ranking falls through to bits per byte.
- A stock-Transformers Hub layout with one `model.safetensors`, self-contained remote code, a
  reversible byte tokenizer, and verified legacy-layout conversion.
- Extended run options **G–J** (chat OpenAI bridge, stock Transformers with remote code, legacy
  MLX NPZ export, Docker CUDA) documented in `docs/run-options.md`.
- `AptivraCausalLM` inference module with Top-2 MoE forward and `configs/inference.yaml` defaults.
- Measured-results model-card evaluation JSON generator and release gate for held-out bits per
  byte versus a unigram baseline.
- Per-cycle continual-learning evaluation so BWT, FWT, and forgetting use the model state that
  existed at each cycle rather than the final state.
- Strict v3 model-card evidence validation, like-for-like metric baselines, and conversion
  provenance uploaded beside consolidated weights.
- Subprocess-isolated generated-code verification with configured wall-clock and memory limits.

### Changed
- Replaced unsupported public README/model-card claims with measured-only research prototype
  language and marked invalid historical results in `docs/RESULTS.md`.
- AID-Bench compatibility properties now return unmeasured values instead of hard-coded
  capability scores.
- Corrected the published architecture metadata to 271,361,024 unique parameters and a
  512-byte context window.
- Hub publication now excludes all GGUF and legacy shard artifacts; local GGUF export supports
  FP32 and FP16 only.
- Promotion rejection now restores optimizer, T1, replay, EWC, topology, and RNG state while
  retaining FLOPs already spent; package generation now preserves raw stop bytes like
  Transformers generation.

### Fixed
- Constrained setuptools discovery to the `aptivra` package tree so top-level configuration,
  evidence, deployment, and native-runtime directories cannot break wheel builds.

## [v0.1.0] - 2026-09-29

### Added
- **Multi-Domain, Self-Identity & Skill Expertise Training:** Added the identity, dialogue,
  multitasking, and MCP-skill training pipeline. Later audit found the MCP portion used five
  placeholder skill task families and did not demonstrate skill capability.
- **Local GGUF Export:** Added the GGUF v3 exporter
  (`aptivra/export/gguf_exporter.py` and `scripts/export_gguf.py`). The maintained exporter
  supports FP32 and FP16; it does not implement quantized export.
- **Local C-Engine GGUF CLI (`aptivra-cli`):** High-performance C inference CLI (`multitier_c/gguf_cli.c`) with zero external dependencies to parse GGUF magic headers, versioning, tensor metadata (172 tensors), and execution parameters.
- **Initial Hugging Face Hub Release:** Published a legacy artifact layout. The current Hub
  publication contract is the single-safetensors Transformers layout described under
  Unreleased; local GGUF files are not Hub artifacts.
- **Tiered Memory System ($T_0, T_1, T_2$):** Pre-aligned token embeddings, verifier-gated contrastive memory store with micro-replay, soft top-2 MoE routing, and Tiered Memory Runtime (TMR) C engine.

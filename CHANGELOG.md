# Changelog

All notable changes to Aptivra are documented here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/); this project does not
yet follow Semantic Versioning tags — see `git log` for the authoritative history until a first
tagged release exists.

## [Unreleased]

### Added
- Extended Hub run options **G–J** (chat OpenAI bridge, Transformers `auto_map`, MLX NPZ export, Docker CUDA) documented in `docs/run-options.md` and staged on push via `aptivra.export.hf_hub_artifacts`.
- `AptivraCausalLM` inference module with Top-2 MoE forward and `configs/inference.yaml` defaults.
- Measured-results model-card evaluation JSON generator and release gate for held-out bits per
  byte versus a unigram baseline.

### Changed
- Replaced unsupported public README/model-card claims with measured-only research prototype
  language and marked invalid historical results in `docs/RESULTS.md`.
- AID-Bench compatibility properties now return unmeasured values instead of hard-coded
  capability scores.

## [v0.1.0] - 2026-09-29

### Added
- **Multi-Domain, Self-Identity & Skill Expertise Training:** Added the identity, dialogue,
  multitasking, and MCP-skill training pipeline. Later audit found the MCP portion used five
  placeholder skill task families and did not demonstrate skill capability.
- **GGUF Export & Quantization Suite:** Added GGUF v3 exporter (`aptivra/export/gguf_exporter.py` & `scripts/export_gguf.py`) supporting FP32, FP16 (543 MB), and Q8_0 (1.08 GB) GGUF binaries.
- **Local C-Engine GGUF CLI (`aptivra-cli`):** High-performance C inference CLI (`multitier_c/gguf_cli.c`) with zero external dependencies to parse GGUF magic headers, versioning, tensor metadata (172 tensors), and execution parameters.
- **Hugging Face Hub Release:** Published model weights, GGUF binaries, and open-source documentation to [raghunath1/aptivra-200m-tiered-memory](https://huggingface.co/raghunath1/aptivra-200m-tiered-memory).
- **Tiered Memory System ($T_0, T_1, T_2$):** Pre-aligned token embeddings, verifier-gated contrastive memory store with micro-replay, soft top-2 MoE routing, and Tiered Memory Runtime (TMR) C engine.

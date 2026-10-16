# REPO-TREE.md — Aptivra Directory & File Purpose Map

> The block between AUTO-TREE:START and AUTO-TREE:END is regenerated
> automatically by a Stop hook (`.claude/hooks/sync-repo-tree.sh`, running
> `tree -L 2 --gitignore -I '.git'`) every time an agent session stops — do
> not hand-edit it, edits there are overwritten on the next stop.
>
> Everything under "## Purpose" is agent-maintained: update it in the same
> commit that adds, removes, renames, or repurposes a top-level or
> second-level directory, or adds a file whose role isn't obvious from its
> name (see `AGENTS.md` -> "Repo Tree Maintenance").

## Structure

<!-- AUTO-TREE:START -->
```text
.
├── AGENTS.md
├── CHANGELOG.md
├── CLAUDE.md
├── CODE_OF_CONDUCT.md
├── CONTRIBUTING.md
├── LICENSE
├── README.md
├── REPO-TREE.md
├── SUPPORT.md
├── aptivra
│   ├── __init__.py
│   ├── __pycache__
│   ├── agents
│   ├── ambient
│   ├── baselines
│   ├── benchmarks
│   ├── consolidation
│   ├── embodied
│   ├── eval
│   ├── experts
│   ├── export
│   ├── federation
│   ├── gate
│   ├── genesis.py
│   ├── governance
│   ├── huggingface
│   ├── inference
│   ├── learning
│   ├── memory
│   ├── model
│   ├── planning
│   ├── research
│   ├── tasks
│   ├── versioning
│   └── world_model
├── aptivra.egg-info
│   ├── PKG-INFO
│   ├── SOURCES.txt
│   ├── dependency_links.txt
│   ├── requires.txt
│   └── top_level.txt
├── configs
│   ├── backbone_200m.yaml
│   ├── backbone_50m.yaml
│   ├── backbone_85m.yaml
│   └── inference.yaml
├── deploy
│   ├── docker
│   ├── huggingface_hub
│   └── ollama
├── docs
│   ├── DESIGN.md
│   ├── RESULTS.md
│   ├── adr
│   ├── chat-ui.md
│   ├── configuration-register.md
│   └── run-options.md
├── experiments
│   ├── ablation_200m_results.jsonl
│   ├── ablation_results.jsonl
│   ├── aptivra_200m_results.jsonl
│   ├── aptivra_results.jsonl
│   ├── baseline_200m_results.jsonl
│   ├── baseline_results.jsonl
│   ├── identity_export_eval.jsonl
│   ├── multidomain_conversational_results.jsonl
│   └── tmr_benchmark_results.json
├── multitier_c
│   ├── Makefile
│   ├── aptivra-cli
│   ├── gguf_cli.c
│   ├── gguf_cli.o
│   ├── multitier.c
│   ├── multitier.h
│   └── multitier.o
├── pyproject.toml
├── scratch
│   ├── hf_export
│   ├── hf_export_backup_20261004T150310Z
│   ├── hf_export_backup_20261004T155439Z
│   └── repair_phase_b.log
├── scripts
│   ├── __init__.py
│   ├── __pycache__
│   ├── benchmark_tmr_scaling.py
│   ├── eval_identity_export.py
│   ├── export_gguf.py
│   ├── gradio_chat.py
│   ├── push_to_hf.py
│   ├── repair_chat_backbone.py
│   ├── restore_hf_export.py
│   ├── run_ablations.py
│   ├── run_aid_bench.py
│   ├── run_aptivra.py
│   ├── run_baselines.py
│   ├── run_chat_server.py
│   ├── run_mlx_inference.py
│   ├── run_phase_a_then_b.py
│   ├── run_transformers_inference.py
│   ├── summarize_results.py
│   ├── train_multidomain_conversational.py
│   └── verify_chat_quality.py
├── tests
│   ├── __pycache__
│   ├── conftest.py
│   ├── test_all_phases_e2e.py
│   ├── test_baselines.py
│   ├── test_chat_prompt.py
│   ├── test_chat_quality.py
│   ├── test_chat_sanitize.py
│   ├── test_code_synth_tasks.py
│   ├── test_consolidation.py
│   ├── test_contrastive_memory.py
│   ├── test_embeddings.py
│   ├── test_eval.py
│   ├── test_experts.py
│   ├── test_gate.py
│   ├── test_genesis_governance_and_embodied.py
│   ├── test_genesis_learning_and_planning.py
│   ├── test_genesis_memory.py
│   ├── test_genesis_phases_deep.py
│   ├── test_genesis_system_and_bench.py
│   ├── test_gguf_exporter.py
│   ├── test_hybrid_store.py
│   ├── test_identity.py
│   ├── test_inference_causal_lm.py
│   ├── test_memory.py
│   ├── test_model.py
│   ├── test_multitier_export.py
│   ├── test_skill_dataset.py
│   ├── test_smoke_e2e.py
│   ├── test_soft_router.py
│   ├── test_tasks.py
│   └── test_versioning.py
└── uv.lock

43 directories, 93 files
```
<!-- AUTO-TREE:END -->

## Purpose

- `AGENTS.md` — Supreme agent operating instructions for this repository.
- `CLAUDE.md` — Verbatim mirror of `AGENTS.md`.
- `.github/workflows/` — Change-driven CI definitions for pre-commit, coverage, and partitioned Cosmic Ray mutation gates.
- `.pre-commit-config.yaml` — Pinned commit-time lint, type, whitespace, conflict, size, and model-artifact checks.
- `CHANGELOG.md` — Project version changelog.
- `CODE_OF_CONDUCT.md` — Contributor code of conduct.
- `CONTRIBUTING.md` — Guidelines for contributing to Aptivra.
- `LICENSE` — Software license terms.
- `SUPPORT.md` — Support channels and guidelines.
- `pyproject.toml` — Build system, dependency specification, and tool configuration for ruff, mypy, pytest, and mutmut.
- `.claude/` — Repository-local Claude Code automation.
  - `.claude/hooks/` — Stop hooks that regenerate the protected `REPO-TREE.md` structure block.
- `configs/` — Editable YAML copies for versioned task streams, task-stream audit policy, GitHub issues stream building, measured-results release gates, and verifier sandbox limits, plus repository-only model architecture, ablation, inference/chat, pretrained base-model candidate, decision-threshold, Vertex spot-execution, and GCS artifact-lifecycle presets; installed canonical runtime defaults live under `aptivra/resources/configs/`.
- `mutation/` — Repo-wide Cosmic Ray configurations partitioned into focused, parallelizable source sessions.
- `deploy/` — Deployment bridges for Docker CUDA/CPU and Ollama/LM Studio run options.
  - `deploy/docker/` — CPU and CUDA container deployment definitions.
  - `deploy/ollama/` — Ollama and LM Studio bridge configuration and usage notes.
- `docs/` — Architectural documentation, design specifications, and ADRs.
  - `docs/run-options.md` — G–J run paths, Hub-layout constraints, and measured-results publication workflow for `raghunath1/aptivra-200m-tiered-memory`.
  - `docs/DESIGN.md` — Master Phase 1 architecture and design specification.
  - `docs/configuration-register.md` — Register of all runtime-tunable configurations.
  - `docs/adr/` — Architecture Decision Records capturing foundational technical decisions.
  - `docs/github-issues-stream/` — Committed manifest index, bounded hash-verified record shards, and aggregate audit for the local-only public GitHub issues continual-learning stream; raw issue text stays under git-ignored `scratch/`.
- `aptivra/` — Core Python package containing models, memory systems, experts, task stream, and eval harness.
  - `aptivra/model/` — Decoder-only backbone transformer architecture, RMSNorm, SwiGLU, and isolated AttentionInterface.
  - `aptivra/baselines/` — Continual learning baseline trainers (Naive Sequential, Replay-Only, EWC-Only, Joint Multitask).
  - `aptivra/experts/` — Historical soft Top-2 modules plus the disabled ADR-0011 standard PEFT expert bank, content-addressed promotion, and fail-closed reload boundary.
  - `aptivra/gate/` — Promotion criteria and statistical decision helpers for continual-learning updates.
  - `aptivra/tasks/` — V1/v2 sequential task family generators, normalized-AST fingerprinting, subprocess-isolated mechanical verifiers, and versioned task stream manager.
  - `aptivra/eval/` — Continual learning evaluation harness, role-aware headline aggregates, deterministic task-stream audit gates, accuracy matrix logger, model-card measurement records, pretrained base-model comparison and selection logic, local/GCS artifact stores, the resumable Vertex worker/spec renderer, and the versioned held-out prompt set.
  - `aptivra/export/` — Self-contained Transformers packaging and legacy conversion, Tiered C and local-format export utilities, and measured-results model-card rendering.
  - `aptivra/inference/` — MoE causal LM loader, byte tokenizer, and generation for Hub checkpoints.
  - `aptivra/huggingface/` — In-package Transformers config, model, and tokenizer wrappers.
  - `aptivra/resources/` — Installed YAML defaults and non-importable, self-contained Hugging Face remote-code templates consumed by package APIs through `importlib.resources`.
  - `aptivra/agents/` — Dynamic tool synthesis and sandboxed execution engine (Phase 3).
  - `aptivra/ambient/` — Theory-of-Mind modeling, P2P sync, and lifelong narrative keeper (Phases 2 & 4).
  - `aptivra/benchmarks/` — 4-Phase Developmental Benchmark Suite (AID-Bench).
  - `aptivra/consolidation/` — Generative sleep and REM counterfactual dream consolidation engine with synaptic pruning.
  - `aptivra/embodied/` — Sensorimotor tokenizer and closed-loop SE(3) kinematic actuator (Phase 2).
  - `aptivra/federation/` — Decentralized collective intelligence and federated mind node (Phase 4).
  - `aptivra/governance/` — Dual-Boundary Constitutional Governor, Merkle audit kernel, and drift auditor.
  - `aptivra/learning/` — Local predictive coding and forward-only plasticity for backprop-free edge adaptation.
  - `aptivra/memory/` — Hexa-tier memory continuum (working, episodic narrative, semantic hypergraph, procedural skills, somatic valence).
  - `aptivra/planning/` — Active inference planning and counterfactual rollout planner.
  - `aptivra/research/` — Autonomous scientific hypothesis discovery and empirical experimenter (Phase 3).
  - `aptivra/versioning/` — Checkpoint manifests, history, integrity verification, and restoration.
  - `aptivra/world_model/` — Continuous latent state-space H-JEPA world model predicting transitions in representation space.
  - `aptivra/genesis.py` — Top-level unified cognitive architecture facade orchestrating all 10 engines across all 4 phases.
- `multitier_c/` — High-throughput C ABI sharding runtime with zero-copy mmap hot-swapping and HDC associative search.
- `experiments/` — Immutable and append-only measured experiment outputs, including recorded v1/v2 task-stream audit evidence.
  - `experiments/base_model_candidates/` — Frozen powered and amendment evidence, including immutable candidate/repeat-seed records, family-clustered analysis, the failed Amendment 4 proof, the manifested Amendment 4a interruption and noise-floor failure, the failed Amendment 4b outcome gate, and Amendment 4e foreground checks, detached-run state, candidate measurements, and mixed-provenance ranking.
- `scripts/` — Execution, benchmarking, evaluation, task-stream audit, base-model evaluation, Vertex dry-run rendering, and export scripts.
  - `scripts/run-project-tool` — Portable `uv` launcher for locked development tools in primary checkouts, linked worktrees, and CI.
- `tests/` — Pytest unit and integration suite; its task-stream audit check is the repository's CI-time policy gate.

# Configuration Register — Aptivra

Every runtime-tunable setting in this repository is listed here. **An
unregistered tunable does not exist**: if a value that a user, operator, or
environment should own is not in this table, that is a Configurability Mandate
violation and it stops the line (see `AGENTS.md`).

A row is added in the **same commit** that introduces the setting. Removing a
row requires the setting to be genuinely gone, not merely undocumented.

| Setting | Scope | Type | Default | Safe limits | RBAC role | Env-specific | Feature flag | API / config-as-code path | Requires restart |
|---------|-------|------|---------|-------------|-----------|--------------|--------------|---------------------------|------------------|
| `task_stream.num_train_per_task` | global | integer | `500` | 10–10000 | researcher | yes | — | `configs/task_stream.yaml` | no |
| `task_stream.num_val_per_task` | global | integer | `100` | 5–2000 | researcher | yes | — | `configs/task_stream.yaml` | no |
| `task_stream.num_test_per_task` | global | integer | `300` | 5–2000 | researcher | yes | — | `configs/task_stream.yaml` | no |
| `task_stream.seed` | global | integer | `42` | 0–2^31-1 | researcher | yes | — | `configs/task_stream.yaml` | no |
| `task_stream.max_generation_attempts` | global | integer | `100` | 1–1000 | researcher | yes | — | `configs/task_stream.yaml` | no |
| `evaluation.max_new_tokens` | global | integer | `32` | 1–512 | researcher | yes | — | `configs/task_stream.yaml` | no |
| `model.d_model` | global | integer | `512` | 64–2048 | researcher | yes | — | `configs/backbone_50m.yaml` | no |
| `model.n_layers` | global | integer | `8` | 1–32 | researcher | yes | — | `configs/backbone_50m.yaml` | no |
| `model.n_heads` | global | integer | `8` | 1–32 | researcher | yes | — | `configs/backbone_50m.yaml` | no |
| `model.d_ff` | global | integer | `2048` | 256–8192 | researcher | yes | — | `configs/backbone_50m.yaml` | no |
| `model.max_seq_len` | global | integer | `512` | 1–512 | researcher | yes | — | `configs/backbone_200m.yaml` | no |
| `baseline.learning_rate` | service | float | `0.0001` | 1e-6–1e-2 | researcher | yes | — | `configs/backbone_50m.yaml` | no |
| `baseline.replay_capacity` | service | integer | `500` | 10–50000 | researcher | yes | — | `configs/backbone_50m.yaml` | no |
| `baseline.ewc_lambda` | service | float | `400.0` | 0.1–10000.0 | researcher | yes | — | `configs/backbone_50m.yaml` | no |
| `execution.platform` | service | enum | `pytorch` | `pytorch`, `tiered_c` | researcher | yes | — | `configs/backbone_50m.yaml` | no |
| `multitier.enable_nvme_streaming` | service | boolean | `true` | `true`, `false` | researcher | yes | — | `configs/backbone_50m.yaml` | no |
| `multitier.prefetch_lookahead_layers` | service | integer | `1` | 0–4 | researcher | yes | — | `configs/backbone_50m.yaml` | no |
| `multitier.hot_expert_pin_gb` | service | float | `4.0` | 0.0–128.0 | researcher | yes | — | `configs/backbone_50m.yaml` | no |
| `inference.model_repo_id` | global | string | `raghunath1/aptivra-200m-tiered-memory` | non-empty HF repo id | operator | yes | — | `configs/inference.yaml` | no |
| `inference.local_export_dir` | global | path | `scratch/hf_export` | directory with the Transformers layout or legacy multitier shards | operator | yes | — | `configs/inference.yaml` | no |
| `inference.device` | global | enum | `auto` | `auto`, `cuda`, `mps`, `cpu` | operator | yes | — | `configs/inference.yaml` | no |
| `inference.max_new_tokens` | global | integer | `96` | 1–512 | operator | yes | — | `configs/inference.yaml` | no |
| `inference.use_moe_at_inference` | global | boolean | `false` | `true`, `false` | operator | yes | — | `configs/inference.yaml` | no |
| `inference.server.host` | service | string | `127.0.0.1` | valid bind address | operator | yes | — | `configs/inference.yaml` | no |
| `inference.server.port` | service | integer | `8080` | 1024–65535 | operator | yes | — | `configs/inference.yaml` | no |
| `inference.server.model_name` | service | string | `aptivra-200m-tiered-memory` | non-empty | operator | yes | — | `configs/inference.yaml` | no |
| `inference.chat.gradio_host` | service | string | `127.0.0.1` | valid bind address | operator | yes | — | `configs/inference.yaml` | no |
| `inference.chat.gradio_port` | service | integer | `7860` | 1024–65535 | operator | yes | — | `configs/inference.yaml` | no |
| `inference.chat.title` | service | string | `Aptivra Chat` | non-empty | operator | yes | — | `configs/inference.yaml` | no |
| `inference.chat.single_turn` | service | boolean | `true` | `true`, `false` | operator | yes | — | `configs/inference.yaml` | no |
| `inference.chat.sanitize_replies` | service | boolean | `true` | `true`, `false` | operator | yes | — | `configs/inference.yaml` | no |
| `inference.chat.max_reply_chars` | service | integer | `280` | 32–2048 | operator | yes | — | `configs/inference.yaml` | no |
| `model_card_release_gate.heldout_bpb_unigram_margin_bits` | global | float | `0.0` | 0.0–8.0 | researcher | yes | — | `configs/model_card_release_gate.yaml` | no |

## Column meanings

- **Scope** — global, per-environment, per-tenant, per-user, or per-service.
- **Safe limits** — the range/enum the UI and API both enforce. "None" is not
  an acceptable value; state the guardrail.
- **RBAC role** — the role permitted to write it. Read access is stated
  separately only when it differs.
- **Env-specific** — whether the value is scoped per environment and promoted
  explicitly between them.
- **Feature flag** — the flag gating the capability, if it is behind phased
  rollout.
- **Requires restart** — `no` is the expected answer. `yes` requires a note
  explaining why hot-apply is not technically feasible, surfaced in the UI
  before the change is applied.

## Excluded by the scope test

Values that alter the system's contract, data model, or security boundary are
**core platform changes**, not configuration. Record each such decision as an
ADR and link it below.

| Value | Why it is not configurable | ADR |
|---|---|---|
| Deep Learning Framework & Compute Abstraction | Framework engine choice (PyTorch 2.x) defines core runtime model and execution semantics | [ADR-0001](file:///Users/raghunath/code/personal/Aptivra/docs/adr/0001-framework-and-device-selection.md) |
| Backbone Architecture & Attention Isolation Contract | Structural contract of decoder transformer and T0 attention interface boundary | [ADR-0002](file:///Users/raghunath/code/personal/Aptivra/docs/adr/0002-model-size-and-backbone-architecture.md) |
| Expert Block Topology & Router Freezing Mechanics | Fundamental expert module parameter isolation and router parameter freezing rules | [ADR-0003](file:///Users/raghunath/code/personal/Aptivra/docs/adr/0003-expert-block-design-and-router-architecture.md) |
| Episodic Store Write Gate Policy | Hard invariant requiring 100% verifier correctness ($v\_result = 1$) for episodic admission | [ADR-0004](file:///Users/raghunath/code/personal/Aptivra/docs/adr/0004-episodic-store-retrieval-method.md) |
| Statistical Test for Promotion Gate | Mathematical decision boundary ($\alpha = 0.05$ paired statistical test) for model promotion | [ADR-0005](file:///Users/raghunath/code/personal/Aptivra/docs/adr/0005-statistical-test-for-promotion-gate.md) |
| Tiered Memory Runtime & Pure C Engine | Architecture for NVMe expert streaming, usage-heat expert pinning, and pure C runtime export | [ADR-0006](file:///Users/raghunath/code/personal/Aptivra/docs/adr/0006-multitier-nvme-expert-tiering-and-c-engine.md) |

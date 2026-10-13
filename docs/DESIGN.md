# Aptivra Architecture & System Design — Phase 1

> **Project:** Aptivra: Continual Learning with Tiered Memory, Growable Expert Modules, and Verifiable Consolidation  
> **Phase:** Phase 1 (Core Hypothesis H1 Evaluation)  
> **Author:** Aptivra Lead Research Engineer  
> **Date:** September 2026  

---

## 1. Executive Summary & Core Hypothesis

### 1.1 Hypothesis H1
> **Hypothesis H1:** At 50M–200M parameters, Aptivra's tiered memory architecture with gated consolidation achieves superior performance compared to a naive sequential fine-tuning baseline across a continual learning stream of task families, demonstrating both higher new-task adaptation accuracy and significantly higher retention of previously learned tasks.

### 1.2 Kill Criterion
If, after the full ablation study in Milestone M5, Aptivra does **not** surpass the **Replay-Only** baseline on backward transfer / retention at equal compute budget (measured in total FLOPs), Hypothesis H1 is falsified. If falsified, this finding will be reported transparently and development will halt. Hyperparameters will not be manually tuned to force a positive outcome.

### 1.3 Scope Boundary for Phase 1
- **In-Scope:** Decoder-only transformer backbone (50M–200M), isolated attention interface (T0), verifier-gated episodic store (T1), growable expert module pool with gating router (T2), periodic consolidation with replay and EWC penalties, verifiable promotion gate with statistical regression testing, versioned immutable checkpoints, and an 8-task mechanical benchmark stream.
- **Explicitly Out of Scope:** Reinforcement learning (RL), self-training task proposers, dynamic-k token-level routing, dynamic deliberation compute budgets, multimodality, and distributed multi-GPU/multi-node training. Clean abstraction interfaces are provided for future integration.

---

## 2. Tiered Memory & Modular Architecture

Aptivra decouples working context, episodic memory, parameter growth, and foundational representations across four distinct tiers.

```
+-------------------------------------------------------------------------+
|                              Aptivra Tiered System                      |
|                                                                         |
|  +-------------------------------------------------------------------+  |
|  | Tier 0 (T0): Context Layer & Attention Interface                   |  |
|  | - Standard Multi-Head Causal Attention with KV Cache               |  |
|  | - Abstract `AttentionInterface` (Pluggable Delta-Rule / KDA)      |  |
|  +-------------------------------------------------------------------+  |
|                                    |                                    |
|  +---------------------------------+---------------------------------+  |
|  |                                                                   |  |
|  v                                                                   v  |
|  +----------------------------------+   +----------------------------+  |
|  | Tier 3 (T3): Transformer Backbone|   | Tier 1 (T1): Episodic Store|  |
|  | - 50M–200M Decoder Transformer   |   | - Verified (x, y, v) Tuples|  |
|  | - RoPE, RMSNorm, SwiGLU FFN      |   | - Vector Index & Retrieval |  |
|  +----------------------------------+   +----------------------------+  |
|                  |                                    |                 |
|                  v                                    v                 |
|  +-------------------------------------------------------------------+  |
|  | Tier 2 (T2): Growable Expert Module Pool & Router                 |  |
|  | - Expert Blocks E_1, ..., E_K (Mature Experts Frozen)            |  |
|  | - Module Router g(x) with Frozen Router Rows for Mature Experts    |  |
|  +-------------------------------------------------------------------+  |
|                                    |                                    |
|                                    v                                    |
|  +-------------------------------------------------------------------+  |
|  | Gated Consolidation Cycle & Verifiable Promotion Gate             |  |
|  | - Replay + EWC Penalty + T1 Memory Batch Training                 |  |
|  | - Statistical Test on Held-out Regression Suite (p >= 0.05)       |  |
|  +-------------------------------------------------------------------+  |
+-------------------------------------------------------------------------+
```

### 2.1 Tier 0 (T0): Working Context & Attention Interface
- **Role:** Handles sequence-level self-attention and working context memory during inference and training.
- **Decoupled Interface (`AttentionInterface`):**
  - Defines contract: `forward(x, context=None, attention_mask=None, kv_cache=None) -> (output, new_kv_cache)`.
  - Primary Implementation: `StandardCausalAttention` with scaled dot-product attention and past KV-cache management.
  - Future Extensibility: Provides a clean drop-in interface for delta-rule / KDA linear attention layers without modifying backbone transformer code.

### 2.2 Tier 1 (T1): Verifier-Gated Episodic Store
- **Role:** External non-parametric memory of verified problem-solution executions.
- **Write Filter (Verification Gate):**
  - Only tuples $(x, y, v\_result, \text{meta})$ where $v\_result = 1$ (Passed) are admitted to T1. Failed or unverified attempts are strictly discarded.
- **Retrieval Engine:**
  - Dense input embeddings via cosine similarity search.
  - Returns top-$k$ nearest neighbors for prompt enrichment or targeted consolidation batching.
  - Tracks `retrieval_count` and `verification_count` per item for consolidation selection.

### 2.3 Tier 2 (T2): Growable Expert Module Pool & Gating Router
- **Role:** Modular parameter capacity expansion to absorb new task distributions without altering mature task parameters.
- **Structure:**
  - Pool of FFN/Adapter blocks $E_1, E_2, \dots, E_K$.
  - Module Router $g(x) = \text{softmax}(W_{router} x)$.
- **Freezing & Spawning Rules:**
  - When expert $k$ completes its consolidation phase, $E_k$ and $W_{router}[k, :]$ are frozen.
  - Spawning Trigger: A new expert $E_{K+1}$ is instantiated when:
    1. **Router Entropy:** $H(g(x)) = -\sum_i g_i(x) \log g_i(x) > H_{thresh}$ across a batch, OR
    2. **Verifier Failure Rate:** Error rate on task family verification exceeds $V_{fail\_thresh}$.
  - Thresholds live exclusively in configuration files.

### 2.4 Tier 3 (T3): Backbone Transformer
- **Role:** Base language and representation foundation.
- **Config-Driven Architecture:**
  - Decoder-only architecture configurable between 50M and 200M parameters.
  - RMSNorm pre-normalization, Rotary Position Embeddings (RoPE), SwiGLU activation.

---

## 3. Consolidation Cycle & Promotion Gate

### 3.1 Consolidation Cycle Mechanics
Consolidation occurs at task boundaries or after fixed intervals:
1. **Sample Selection:** High-utility T1 records (high retrieval frequency, high verification ratio).
2. **Batch Construction:** Mix of candidate T1 records + Replay Buffer samples from earlier tasks.
3. **Loss Function with EWC Regularization:**
   $$\mathcal{L}_{\text{cons}}(\theta) = \mathcal{L}_{\text{CE}}(D_{\text{cons}}; \theta) + \frac{\lambda_{\text{ewc}}}{2} \sum_{i} F_i (\theta_i - \theta_{i, \text{prev}})^2$$
   where $F_i$ is the diagonal Fisher Information Matrix entry computed over previous task data.
4. **Ablation Toggles:** Configurable boolean switches for:
   - `use_t1_memory`
   - `use_expert_growth`
   - `use_replay_buffer`
   - `use_ewc_penalty`
   - `use_promotion_gate`

### 3.2 Verifiable Promotion Gate
Every candidate checkpoint generated by consolidation must pass the Promotion Gate:
1. **Validation Check:** Candidate must show positive accuracy gain on new task validation split:
   $$\Delta \text{Acc}_{\text{new}} = \text{Acc}_{\text{val}}(\theta_{\text{cand}}, \text{Task}_{\text{new}}) - \text{Acc}_{\text{val}}(\theta_{\text{curr}}, \text{Task}_{\text{new}}) > 0$$
2. **Statistical Regression Test:**
   - Evaluated on the held-out regression test suite $D_{\text{regress}}$ covering all prior tasks.
   - Paired per-instance loss difference: $d_i = L(x_i, y_i; \theta_{\text{cand}}) - L(x_i, y_i; \theta_{\text{curr}})$.
   - One-sided paired t-test (or Wilcoxon signed-rank test): Null Hypothesis $H_0: \mu_d \le 0$ vs. $H_1: \mu_d > 0$.
   - **Gate Threshold:** Promote candidate if $p \ge 0.05$ (no statistically significant degradation).
   - Reject candidate if $p < 0.05$, log exact diagnostic payload (p-value, per-task delta, sample sizes).

### 3.3 Checkpoint Versioning & Rollback
- Promoted checkpoints are stored immutably under `checkpoints/v{version_id}/`.
- Manifest payload: `config_hash`, `data_hash`, `random_seed`, `eval_metrics`, `p_value`, `timestamp`.
- Rollback Command: Single CLI / Python interface: `aptivra rollback --to-version <version_id>`.

---

## 4. Benchmark Stream & Mechanical Verifiers

The evaluation task stream comprises 8 sequential task families, each governed by a deterministic, non-model programmatic verifier.

| ID | Task Family | Description | Programmatic Verifier Mechanism |
|---|---|---|---|
| T1 | Modular Arithmetic | Multi-step modular operations (add, mul, pow) | Direct mathematical computation check |
| T2 | String Transformations | Run-length encoding, string inversion, Caesar cipher | Exact string match with deterministic transformation |
| T3 | List Sorting & Filtering | Numerical sorting, even/odd filtering, top-K selection | Structural evaluation against list invariants |
| T4 | Syntax & Bracket Matching | Nested parentheses, brackets, braces balancing & completion | Pushdown automaton parser verification |
| T5 | Boolean Logic Synthesis | Evaluating & simplifying boolean logic expressions | Truth table evaluation matching ground truth |
| T6 | Sequence Completion | Arithmetic, geometric, polynomial, and Fibonacci progressions | Formula verification on next N elements |
| T7 | State Machine Execution | Deterministic Finite Automaton (DFA) state traversal | State transition graph execution check |
| T8 | Python Micro-Synthesis | Small deterministic functions (e.g. factorial, GCD, string count) | Isolated AST parsing & sandboxed execution verification |

### 4.1 Data Isolation Assertion
- Train, validation, and held-out regression splits are strictly disjoint.
- **Enforcement in Code:** An explicit runtime assertion checks dataset hashes during training/tuning loops:
  `assert not dataset.intersects(held_out_regression_suite), "DATA CONTAMINATION DETECTED"`

---

## 5. Evaluation Metrics & Baselines

### 5.1 Metrics Suite (Lopez-Paz & Ranzato, 2017)
Let $R_{i,j}$ be the test accuracy on task $j$ after training on task $i$ (for $i, j \in \{1, \dots, T\}$).

1. **Average Accuracy ($A_T$):**
   $$A_T = \frac{1}{T} \sum_{j=1}^T R_{T,j}$$
2. **Backward Transfer (BWT):**
   $$\text{BWT} = \frac{1}{T-1} \sum_{j=1}^{T-1} (R_{T,j} - R_{j,j})$$
3. **Forward Transfer (FWT):**
   $$\text{FWT} = \frac{1}{T-1} \sum_{j=2}^T (R_{j-1,j} - \bar{b}_j)$$
   where $\bar{b}_j$ is the accuracy on task $j$ at random initialization.
4. **Maximum Forgetting ($F_j$):**
   $$F_j = \max_{l \in \{1, \dots, T-1\}} (R_{l,j} - R_{T,j})$$
5. **Compute & Resource Metrics:** Total training FLOPs per consolidation cycle and T1 Store Size (record count & memory MB).
6. **Confidence Intervals:** 95% confidence intervals computed over $\ge 3$ distinct random seeds.

### 5.2 Baselines
1. **Naive Sequential Fine-Tuning:** Standard autoregressive training sequentially across tasks without memory, replay, or parameter protection.
2. **Replay-Only:** Fine-tuning with a fixed-size experience replay buffer sampled uniformly from past task training data.
3. **EWC-Only:** Sequential training with Elastic Weight Consolidation quadratic penalty on parameter drift based on Fisher information.
4. **Joint Multitask Training:** Upper bound baseline trained simultaneously on all 8 task families with equal batch distribution.

---

## 6. Software Architecture & Directory Layout

```text
Aptivra Workspace
├── AGENTS.md
├── CLAUDE.md
├── REPO-TREE.md
├── docs/
│   ├── configuration-register.md
│   ├── DESIGN.md
│   └── adr/
│       ├── 0001-framework-and-device-selection.md
│       ├── 0002-model-size-and-backbone-architecture.md
│       ├── 0003-expert-block-design-and-router-architecture.md
│       ├── 0004-episodic-store-retrieval-method.md
│       └── 0005-statistical-test-for-promotion-gate.md
├── aptivra/
│   ├── __init__.py
│   ├── model/           # T3 Backbone & T0 Attention Interface
│   ├── memory/          # T1 Episodic Store & Embedding Retrieval
│   ├── experts/         # T2 Expert Module Pool & Gating Router
│   ├── consolidation/   # Consolidation Trainer, Replay & EWC
│   ├── gate/            # Verifiable Promotion Gate & Statistical Tests
│   ├── eval/            # Continual Learning Eval Harness & Metrics
│   ├── tasks/           # 8 Task Families & Mechanical Verifiers
│   └── baselines/       # Baseline implementations
├── configs/             # YAML configurations (model, task, ablation)
├── experiments/         # Output logs, metrics JSONL, and manifests
├── tests/               # Pytest unit tests & e2e laptop smoke test
└── scripts/             # Run & evaluation scripts
```

---

## 7. Risk Analysis & Mitigation Strategies

| Risk / Failure Mode | Potential Impact | Prevention & Mitigation Strategy |
|---|---|---|
| **MPS Backend Op Incompatibilities** | Test failure on Apple Silicon dev environments | Explicit CPU fallback guard, device abstraction layers, unit test validation on both MPS and CPU. |
| **Data Contamination of Regression Suite** | Invalid baseline/Aptivra retention metrics | Runtime code assertions checking data sample unique identifiers and dataset hash non-overlap. |
| **Router Entropy Collapse** | Expert pool fails to spawn new experts, causing interference | Minimum entropy threshold tuning via config, verifier failure rate as dual trigger for spawning. |
| **High Compute Overhead in Gate Evaluation** | Slow consolidation loops | Subsampled evaluation batches for interim validation, full statistical test on designated evaluation checkpoints. |

---

## 8. Open Questions for Alignment (Nemawashi)

Before proceeding to Milestone M1 (Task stream & Verifiers), the following design questions are presented for explicit alignment:

1. **Backbone Size Default:** Is ~85M parameters (12 layers, 768 hidden dim, 12 heads) acceptable as the default backbone size for local MPS development and CUDA benchmarking, or would a smaller ~50M baseline be preferred for laptop smoke tests?
2. **T1 Embedding Model Selection:** For input vector retrieval in T1, should we default to a lightweight frozen transformer encoder (e.g. `all-MiniLM-L6-v2` via SentenceTransformers or PyTorch equivalent) or use a mean-pooled projection of the backbone's own embedding layer?
3. **Statistical Test Choice:** For the Promotion Gate regression check, should we default to the **one-sided paired t-test** or the non-parametric **Wilcoxon signed-rank test**? (Currently, ADR-0005 specifies using Shapiro-Wilk to pick t-test if normal, Wilcoxon if non-normal).
4. **Task Sequence Order:** For the 8 sequential task families, should the task presentation order be fixed across all random seeds, or should task order permutation be evaluated as an additional experimental condition?

---

# ADR-0005: Statistical Test for Promotion Gate

- **Status:** Accepted
- **Date:** 2026-09-27
- **Authors:** Aptivra Lead Research Engineer

## Context

Aptivra's promotion gate evaluates candidate model checkpoints generated after a consolidation cycle. To prevent silent regression on previously learned tasks (catastrophic forgetting), candidate checkpoints must pass a strict statistical test on a held-out regression test suite before being promoted to immutable model versions.

## Decision

1. **Promotion Rules:**
   A candidate checkpoint $\theta_{cand}$ is promoted if and only if both conditions are met:
   - **Condition 1 (New-Task Gain):** $\text{Acc}_{\text{val}}(\theta_{cand}, \text{Task}_{new}) > \text{Acc}_{\text{val}}(\theta_{curr}, \text{Task}_{new})$.
   - **Condition 2 (No Statistically Significant Regression):** Evaluated on the held-out regression suite $D_{regress}$ across all previously completed task families.
2. **Statistical Test Protocol:**
   - Compute paired per-sample accuracy (or cross-entropy loss) vectors $S_{curr}$ and $S_{cand}$ over all instances in $D_{regress}$.
   - Difference vector: $d_i = L(x_i, y_i; \theta_{cand}) - L(x_i, y_i; \theta_{curr})$.
   - Perform a **one-sided paired t-test** (or non-parametric **Wilcoxon signed-rank test** if normality is violated via Shapiro-Wilk test) testing the null hypothesis $H_0: \mu_d \le 0$ against the alternative hypothesis $H_1: \mu_d > 0$ (candidate has significantly higher loss / lower performance than current baseline).
   - Significance threshold: $\alpha = 0.05$.
   - **Promotion Decision:** Promote if $p \ge 0.05$ (fail to reject $H_0$, meaning no statistically significant degradation). If $p < 0.05$, candidate is rejected.
3. **Audit & Diagnostics:**
   - Every gate decision is logged into an immutable manifest JSON file containing the test statistic, $p$-value, sample size, per-task accuracy breakdown, decision (`PROMOTED` or `REJECTED`), and diagnostic failure analysis if rejected.

## Consequences

### Positive
- Rigorous statistical guardrail against catastrophic forgetting: random noise or small evaluation fluctuations will not trigger false rejections, while systematic regressions will reliably reject candidate checkpoints.
- Fully transparent and auditable model promotion history with exact $p$-values logged per cycle.

### Rejection state semantics

A rejected candidate is treated as if its learning never became promoted state. The rollback
therefore restores backbone parameters and buffers; router/expert weights, topology, maturity,
and training modes; the main optimizer; T1 records, retrieval index, encoder, and optimizer;
the replay buffer; EWC Fisher and optimal-parameter state; and Python, NumPy, Torch CPU, CUDA,
and MPS random-number generator state. Restoring RNG state makes the next cycle deterministic
relative to a run that never trained the rejected candidate.

`total_flops` is deliberately not restored because the rejected attempt consumed that compute.
The gate decision and diagnostics likewise remain as immutable audit history. Transient
gradients are not promoted state and are cleared by the next optimizer step.

### Negative / Trade-offs
- Requires evaluating candidate checkpoints over the full held-out regression suite at every consolidation cycle, incurring evaluation compute overhead.
- A strict $\alpha = 0.05$ threshold may occasionally reject candidate checkpoints that have minor trade-offs but high new-task gain; hyperparameter tuning during consolidation must focus on preservation.

# ADR-0012: H3 Continual-Learning Pre-registration

- **Status:** Proposed
- **Date:** 2026-10-18
- **Decider:** Repository owner (acceptance required before any H3 training)
- **Depends on:** ADR-0005, ADR-0008, ADR-0011, proposed ADR-0009, and proposed ADR-0010

> **Review status: not ready for acceptance.** Lane 8b replaces lane 8a's censored
> repeatability fixture with the actual H3 cycle caps and 64 held-out examples per task. The
> issue-stream pair produced a usable noise floor. Aptivra did not learn the first synthetic-v2
> task in any seed, however, and both methods scored zero on template-OOD. V2 paired variance,
> template-OOD variance, and their confirmatory power are therefore not estimable. The main H3
> experiment remains forbidden until a prospective amendment demonstrates learning on the v2
> pilot pair without changing the H3 metric after observing results.

## Context and evidence boundary

ADR-0011 requires H3 to be specified before its results exist. This ADR neither accepts H3 nor
contains an H3 result. Lane 8a's calibration evidence remains byte-identical at
[`0012-evidence/pilot-evidence.json`](0012-evidence/pilot-evidence.json), SHA-256
`439894ff14b5d6774a1f24d0bf0323971fd9b9535ab5a7206afddb36d4c7b686`. It is historical
evidence, not a noise floor.

Lane 8b is recorded in
[`0012-evidence/lane8b-pilot-evidence.json`](0012-evidence/lane8b-pilot-evidence.json), schema
`aptivra.h3_mps_pilot_evidence.v2`, with losslessly sharded quiet records below
`0012-evidence/lane8b-quiet-records/`. Raw chronological issue text remains local and is absent
from portable evidence. The run used only cached
`HuggingFaceTB/SmolLM2-360M@f8027fd0eaeea54caa13c31d31b9fdc459c38b49`, with both offline
flags set, on MPS.

The lane 8b evidence selectors are:

| ID | Selector | Measurement |
| --- | --- | --- |
| A1 | `preflight` and `run_manifest*` | Stream, model, configuration, code, runner, and resume-segment hashes. |
| A2 | `evaluation_profiles[*]` | Scalar/batched fixed-set output identity and measured evaluation cost. |
| A3 | `calibration[*]` | EWC, manual-LoRA, and Aptivra anomaly cells at issue micro-batch 8. |
| A4 | `repeatability[*].metrics` | Five runs per method/stream: one duplicate seed and four unique seeds. |
| A5 | `noise_floor` | Same-seed variation, four-seed sample SD, learning qualification, and paired-difference SD. |
| A6 | `compute_estimate` | Measured batched generation cost and revised H3 cell arithmetic. |

## Amendment to lane 8a measurements

### Protocol and bounded execution

The task pairs are synthetic-v2 indices `[7, 8]` and chronological-issue indices `[0, 1]`.
Every task used the real one-epoch stream up to the first cycle cap: 100 optimizer steps or 8,192
target tokens. V2 had 500 available training examples per task, consumed 200, ran 25 steps, and
stopped at 8,107/8,119 target tokens. Issues had 1,192/1,266 available examples for naive full
fine-tuning, consumed 800 per task, ran 100 steps per task, and stopped at 2,742/3,034 target
tokens. Aptivra's second issue task included replay in its 1,899 available examples but consumed
the same 800 current/replay-ordered examples under the same step cap.

Evaluation used 64 held-out examples per task. V2 additionally used 64 held-out template-OOD
examples per task. The seeds were `[20261017, 20261017, 20261018, 20261019, 20261020]`; the first
seed was repeated exactly and the remaining four values provide the unique-seed sample SD.

The pilot completed 20/20 cells in 24,818.404 seconds (6 h 53 min 38 s), below its eight-hour
limit. A worker was stopped after an MPS command-buffer stall. The resume amendment preserves the
old and new manifests, charges 4,905 interrupted seconds to the same wall budget, and changes only
the execution guard so every subprocess timeout is the remaining global budget. No completed cell
was replaced. The regression test is
`test_remaining_wall_time_limits_each_worker_to_global_budget`.

### Anomaly findings

| Anomaly | Root cause and disposition | Defect-proof test |
| --- | --- | --- |
| Aptivra 0.0 on lane 8a issues | The adapter was trained, but two steps did not improve validation accuracy; the promotion gate correctly rejected and deleted it. Evaluation then used the frozen base, which scored zero. This was a correct consequence of the lane 8a budget, not an adapter-activation defect. The independent router defect was real: a one-row softmax assigned probability 1.0, making threshold 0.65 incapable of abstention. A fixed zero fallback logit now makes abstention an actual alternative. | `test_routed_evaluation_activates_adapter_and_abstention_uses_base`; `test_router_uses_masked_real_frozen_embeddings_and_abstains` |
| Online EWC non-finite issue loss at micro-batch 8 | Lane 8a discarded worker stderr and mislabeled an asynchronous Metal out-of-memory failure as a non-finite loss. Its pilot Fisher estimator also used squared mean gradients rather than mean per-example squared gradients, omitted conventional `0.5 * lambda` scaling, and lacked squared-gradient overflow checks. Those defects are fixed in the pilot comparator; the production comparator already used mean per-example squared gradients and half scaling. The rerun produced finite losses `[1.849941, 0.663562, 0.272266]`, Fisher initialization loss `5.500088`, and a 2,894,568,960-byte EWC state, but peaked at 35,744,890,880 bytes and grew swap by 8,974,761,984 bytes. Micro-batch 8 is therefore correctly rejected; H3 retains calibrated micro-batch 4. | `test_diagonal_fisher_uses_mean_of_per_example_squared_gradients`; `test_diagonal_fisher_rejects_squared_gradient_overflow`; `test_ewc_penalty_uses_conventional_half_scaling`; `test_worker_spawn_classifies_metal_driver_oom_without_persisting_stderr` |
| Manual LoRA 4.99 s versus Aptivra 3.33 s | The earlier three-sample gap was not reproducible. With a seed fixed before model construction, manual LoRA and Aptivra medians were 2.151841 s and 2.140674 s (0.52% apart), with 8,683,520 versus 8,695,052 trainable parameters and nearly identical 25.800/25.801 GB driver peaks. Router work occurs outside the adapter step. Micro-batch-8 LoRA exceeds micro-batch-4 full-FT driver memory because 726-token activation storage scales with batch size; freezing base weights removes gradient/optimizer state but not forward activations or the backward path needed for LoRA. | `test_manual_lora_trains_only_the_last_resident_adapter`; seeded anomaly-cell replay in A3 |
| Evaluation cost | Scalar generation spent 2.531950 s/example on v2 and 0.101641 s/example on issues. Batched generation with KV-cache use and per-row first-complete-answer truncation is output-identical on the fixed set and costs 1.184370 and 0.088269 s/example: 2.1378x and 1.1515x speedups. The metric and verifier are unchanged. | `test_batched_generation_is_output_identical_to_scalar`; `test_batched_generation_removes_padding_after_per_row_terminator`; A2 `output_identity=true` |

### Measured noise floor

All same-seed absolute variations were `0.0`; unlike lane 8a, every row marked **valid** also
learned each trained task above its frozen-base accuracy. Values below are sample SD across four
unique seeds. “OOD” is template-OOD final average; the acquisition OOD SD is also zero.

| Stream / method | Learning qualification | Per-task accuracy SD | BWT SD | Forgetting SD | Final-average SD | Acquisition SD | OOD SD |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| v2 / naive full FT | valid | `[0.152761, 0.007812]` | 0.147957 | 0.147957 | 0.078902 | 0.013532 | 0.000000 |
| v2 / Aptivra full | **non-learning finding** (first task failed 5/5 runs) | not a noise floor | not a noise floor | not a noise floor | not a noise floor | not a noise floor | not a noise floor |
| issues / naive full FT | valid | `[0.044194, 0.032526]` | 0.077076 | 0.046875 | 0.025911 | 0.035516 | n/a |
| issues / Aptivra full | valid | `[0.007812, 0.029920]` | 0.000000 | 0.000000 | 0.016106 | 0.016106 | n/a |
| issues / paired `Aptivra - naive` | valid planning SD | `[0.041093, 0.028527]` | 0.077076 | 0.046875 | 0.030840 | 0.023329 | n/a |

The descriptive v2 Aptivra values were final-average/acquisition SD `0.006379` and zero BWT,
forgetting, and OOD variation, but they are resolution-censored non-learning outputs and are not
used as a noise floor. Naive v2 learned strongly in-template but scored zero OOD in every seed;
therefore zero OOD SD is also censoring, not evidence of perfect repeatability.

## Pre-flight and method contract

No model may be loaded until both stream audits reproduce the exact committed manifests and the
chronological raw hash is
`2f4d6f6554f4e762ff7a779fb81ca579d0a38db119952f67786e52997d3ab9ec`.
A missing or mismatched input halts the experiment. Every measured process starts from pristine
cached weights, passes the registered quiet-machine gate, records its own MPS/RSS/swap peaks, and
runs offline on MPS. The hard own-footprint cap remains 28,991,029,248 bytes. Foreign processes
are observed but never killed or reniced; swap growth above 1,073,741,824 bytes invalidates
throughput, not accuracy.

The six methods remain frozen base, naive sequential full FT, deterministic replay full FT,
bounded online EWC full FT, manual task-ID rank-16 LoRA, and Aptivra full. The selected issue
micro-batches remain 8 frozen, 4 naive, 4 replay, 4 EWC, 8 manual LoRA, and 8 Aptivra. V2 uses 8
for all methods. Online EWC uses a mean of per-example squared gradients, finite Fisher/state
checks, one consolidated diagonal state, and `0.5 * 400.0 * F * (theta - theta*)^2`.

## Confirmatory thresholds and power

The five main seeds remain exactly `20261017` through `20261021`, paired across methods within a
stream. Results from v2, issues, and the one-seed v1 regression are never pooled. The primary
decision remains an intersection-union test at one-sided alpha 0.05; every conjunct must pass on
both claim-bearing streams. Exploratory contrasts use Holm correction within endpoint/stream.

The practical effects are not shrunk to rescue five seeds. For a paired SD `s`, effect `delta`,
and `n` pairs, planning power is
`1 - F_nct(t_0.95,n-1; (delta/s)*sqrt(n))`. The valid issue-stream arithmetic is:

| Endpoint | Registered effect/margin | Paired SD | SD units | Power at n=5 | Pairs for 80% / 90% |
| --- | ---: | ---: | ---: | ---: | ---: |
| BWT superiority | +0.125 | 0.077076 | 1.6218 | 0.9036 | 5 / 5 |
| Forgetting superiority | +0.125 | 0.046875 | 2.6667 | 0.9990 | 3 / 4 |
| Final-average non-inferiority | -0.0625 | 0.030840 | 2.0266 | 0.9774 | 4 / 4 |
| Acquisition non-inferiority | -0.125 | 0.023329 | 5.3582 | >0.9999 | 3 / 3 |

These margins are respectively 1.62, 2.67, 2.03, and 5.36 measured paired SDs on issues. The
superiority effects remain practically meaningful eight-example-equivalents at the 64-example
evaluation size; the non-inferiority margins permit at most four final-average or eight per-task
correct-answer equivalents, rather than being invented from an eight-example pilot quantum.

V2 has no paired SD because Aptivra did not learn. Using only the valid naive-v2 BWT/forgetting SD
`0.147957` as a conservative sensitivity calculation, `delta=0.125` is `0.8448 SD`: five seeds
have power `0.4686`, 11 seeds reach 80%, and 14 reach 90%. This is not paired-H3 power and cannot
authorize H3. At the revised main-cell cost of 11.9213 Mac-hours per added cross-method seed, 11
seeds add 71.528 hours and raise the total design estimate to 202.192 hours; 14 add 107.292 hours
and raise it to 237.956 hours. The effect is not reduced to make five seeds sufficient.

Template-OOD power is not estimable because every pilot value was zero. The provisional practical
margin remains `-0.125`, but it has no noise-floor defense and therefore cannot be accepted. A
prospective learning pilot must establish non-degenerate OOD performance and paired SD before the
template-OOD conjunct is confirmatory.

## Comparator, routing, and kill criterion

Before Aptivra is unblinded, select the strongest eligible non-routing comparator separately per
stream by five-seed mean final average, then BWT, then forgetting, then the fixed order manual
LoRA, replay, EWC, naive FT, frozen base. Eligibility requires learning, finite completion,
memory-cap compliance, compute matching, rollback, and five paired seeds. An incomplete required
baseline kills H3 even if another comparator exists.

H3 can pass only if each claim-bearing stream satisfies all of the following:

1. The lower one-sided 95% paired bound for `Aptivra BWT - comparator BWT` is at least `+0.125`.
2. The lower bound for `comparator forgetting - Aptivra forgetting` is at least `+0.125`.
3. The lower bound for `Aptivra final average - comparator` is at least `-0.0625`.
4. The lower bound for `Aptivra acquisition - comparator` is at least `-0.125`.
5. On v2, the template-OOD lower bound is at least `-0.125`, but only after a prospective
   amendment establishes a non-degenerate paired OOD SD and adequate power.
6. Every trained task in every required method/seed exceeds its frozen-base accuracy. The lane 8b
   v2 Aptivra result fails this prerequisite and currently blocks H3.
7. Routing at a validation-calibrated, test-frozen threshold has eligible coverage at least
   `0.875`, eligible misroute rate at most `0.0625`, fallback false-route rate at most `0.0625`,
   and oracle-minus-routed accuracy at most `0.125`. These permit at most 8 missed routes, 4
   misroutes, 4 false routes, and 8 correct-answer equivalents per 64-prompt denominator.
8. A cap breach, missing seed, non-finite value, worker timeout, CPU fallback, failed rollback,
   audit mismatch, gate bypass, or incomplete evidence kills H3 and remains in the table.

The routing bounds are deliberately not inferred from the broken lane 8a router. With the fixed
fallback logit, lane 8b still observed first-task fallback false-route `1.0`; after the second issue
candidate was rejected, coverage ranged `0.171875–0.640625`, eligible misroute
`0.326087–0.514286`, and oracle gaps `0.21875–0.4921875`. Threshold `0.65` is therefore not
calibrated adequately and fails every relevant bound. Confidence calibration, not a threshold
change after test results, is required before H3.

## Compute matching and revised estimate

All updating methods receive identical serialized order, one epoch per task, effective batch 8,
target-only loss, evaluation schedule, and the first-applicable 100-step/8,192-target-token cap.
Evaluation remains capped at 64 examples per seen task per cycle; v2's template-OOD cell is
separate and never pooled. Actual steps, tokens, generations, wall time, peak memory, swap growth,
and cap utilization are mandatory.

A2's output-identical batching reduces measured generation cost to 1.184370 s/example on v2 and
0.088269 s/example on issues. With 60 main cells, six v1 regression cells, and 66 three-seed
ablation stream-cells, A6 estimates:

| Component | Seconds |
| --- | ---: |
| Main training | 33,653.741 |
| Main evaluation | 180,930.299 |
| V1 training | 705.049 |
| V1 evaluation | 19,059.051 |
| Ablation training | 37,019.115 |
| Ablation evaluation | 199,023.329 |
| **Total** | **470,390.583 (130.664 h)** |

This replaces lane 8a's 606.234-hour unreduced scalar-generation estimate. It is larger than lane
8a's proposed 82.225-hour one-seed-ablation reduction because this estimate preserves the
registered three seeds for each ablation. The 59.607-hour five-seed main matrix is 11.9213 hours
per cross-method seed. This is a planning estimate, not permission to execute H3.

## Execution order and failure handling

After a prospective amendment resolves the v2 learning/OOD blockers: rebind audits and hashes;
reproduce calibration; run frozen and non-routing baselines for all paired seeds on one stream;
select the comparator while Aptivra stays blinded; run Aptivra; seal that stream; then run v1 and
ablations. Every cell checkpoints atomically. Failed, timed-out, OOM, or swap-compromised workers
remain visible. No replacement seed is permitted. Every worker timeout is bounded by the remaining
global run budget.

## Consequences and retrospective

Lane 8b turns lane 8a's zero-score fixture into a real issue-stream variance estimate and exposes a
stronger negative result: the current H3 recipe does not make Aptivra learn the selected v2 pair.
The correct response is to keep ADR-0012 Proposed and block the main experiment, not to call zero
variation reproducibility or shrink the effect. Batched evaluation materially lowers cost without
changing outputs, while explicit manifest-transition evidence and remaining-budget timeouts prevent
a stalled worker from silently escaping the bounded pilot. The next amendment must be prospective,
must retain the metric, and must prove both in-template and template-OOD learning before power is
claimed.

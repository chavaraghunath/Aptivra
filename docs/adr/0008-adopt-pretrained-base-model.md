# ADR-0008: Adopt a Pretrained Causal Language Model as the Tier 3 Base

- **Status:** Proposed
- **Date:** 2026-10-14

## Context

The published Aptivra checkpoint was not pretrained: its weight spreads remain at their
initialisation values, a separate pre-existing checkpoint evaluation—not the powered protocol—
measures 12.5 bits per byte on held-out text against 8.0 bits per byte for a uniform byte
predictor, and it recites only a few memorised sentences. Consequently, the
continual-learning comparisons that started from it produced roughly 4--5% accuracy for every
method and do not establish useful language-model learning.

This evidence reverses the premise in
[ADR-0002](0002-model-size-and-backbone-architecture.md) that Aptivra should train a custom
50M--200M byte-level Tier 3 backbone. If accepted, Aptivra will instead begin from an existing
small pretrained causal language model and layer its memory, routed experts, and consolidation
system on top. Integration remains outside this ADR.

## Original pre-registered rule — superseded

The first rule required permissive licensing, immutable standard-Transformers loading without
remote code, 25% headroom on a 36 GiB M4 Max, exact UTF-8 round-trip, a valid EOS token, and a
consistent feed-forward attachment boundary. Passing candidates were ordered exactly by:

1. mean learned accuracy over 12 families, descending;
2. mean accuracy gain, descending;
3. combined held-out bits per byte, ascending;
4. median probe training tokens per second, descending; and
5. unique parameters, ascending.

The rule used exact point estimates with no noise floor. Its measurement used seed `20261014`,
eight optimizer steps at batch size one, eight train items, one validation item, and one test item
per family at learning rate `5e-4`. Thus one changed test item moved the candidate mean by
`1/12 = 0.0833` and controlled the first ranking criterion. That rule and result are retained as
history, but are superseded because their precision could not support a model decision.

## First sweep — superseded history

The first sweep used manifest SHA-256
`d14c1d7d8c917ee434345f75f0832d4bec4250da69ea27e88dcbe7268fc1ad04` and took about 349
candidate-seconds. Its complete config, manifest, ranking, and candidate records are archived in
`experiments/base_model_candidates/archive/first_underpowered_sweep/`.

| Candidate | Hard constraints | Combined BPB | Zero-shot mean | Learned mean | Mean gain |
| --- | --- | ---: | ---: | ---: | ---: |
| SmolLM2-135M | pass | 0.9284 | 0.0000 | 0.0833 | 0.0833 |
| SmolLM2-360M | pass | 0.8262 | 0.0000 | 0.0000 | 0.0000 |
| GPT-2 | pass | 1.3839 | 0.0000 | 0.0000 | 0.0000 |
| Pythia-160M | reject: UTF-8 | 1.0964 | 0.0000 | 0.0000 | 0.0000 |
| Qwen2.5-0.5B | reject: UTF-8; point-estimate negative gain | 0.8414 | 0.0833 | 0.0000 | -0.0833 |
| Aptivra byte reference | reference only | 12.7898 | 0.0000 | 0.0000 | 0.0000 |

The exact first rule selected SmolLM2-135M because it passed one T12 test item after training.
Qwen lost its one initially correct T7 item. Those were single-item observations, not stable
capability estimates. The earlier proposed selection is withdrawn.

## Amendment 1 — powered measurement and uncertainty-aware rule

Amendment 1 was encoded in `configs/base_model_candidates.yaml` before the rerun and was not
changed afterwards. The powered manifest records its SHA-256 as
`819479ce62ee1fc8a7a916cf84ff190dc39aa9186a78efdcb744f737ae8d69f0`, the evaluator SHA-256
as `60f9f623a3a08dd6ffdf98016da427b9b277d871d3c359d34559cfff7338e330`, and the orchestrator
SHA-256 as `c78e62a8da79f145ce2c12d3bf687c432ae8de2b3814b73b125abe37ed90d5ed`.

### Amended hard-constraint order

Licence, immutable standard loading, exact round-trip, EOS, and structural attachment checks run
before likelihood, learnability, or throughput work. A failure writes its diagnostic and skips the
probe. The reference-only Aptivra checkpoint is excluded before probe work. Memory remains a
post-measurement constraint because its process peak cannot be known before executing a passing
candidate.

### Amended ranking rule

Amendment 1 retained the original criterion order, but stopped comparing learned accuracy and gain
as exact values:

- every accuracy stores successes, denominator, and a two-sided 95% Wilson interval;
- two learned-accuracy results tie when their point estimates differ by at most `0.02` **or**
  their 95% intervals overlap, then the rule falls through to gain;
- gain uses the same `0.02` floor and overlap rule with a Newcombe interval formed from Wilson
  component intervals, then falls through to bits per byte;
- a minimum accuracy or gain threshold rejects only when the interval's upper bound is below the
  threshold; and
- bits per byte, throughput, and parameter count retain the original order.

This was the rule frozen before powered measurement. Its use of separate pooled Wilson intervals
was later found to overstate precision; Amendment 2 below supersedes that part of the rule after
the result was reviewed.

### Powered protocol

- 12 families; 256 train, 100 validation, and 100 test items per family;
- 200 full-fine-tuning optimizer steps per family at batch size 8 and weight decay 0;
- 20 linear warmup steps;
- candidate-wide learning rate selected only from validation scores over `[1e-5, 3e-5, 1e-4]`,
  with lower rate breaking an exact validation tie;
- right-padded batched greedy evaluation at batch size 16 and a 48-token bound;
- primary seed `20261014`; first-pass top two repeated at seeds `20261015` and `20261016` using
  the already selected rate; and
- fresh isolated candidate processes, atomic records, and no remote access.

Expected wall time was 10,800 seconds (three hours), based on a real MPS pilot. Actual wall time
was 12,402.99 seconds (3 h 26 min 43 s), 1,602.99 seconds or 14.8% over estimate. The 360M workload
was slower than the pilot extrapolation. The frozen workload was not reduced after measurement
began.

The existing powered records were produced by code that evaluated test after every learning-rate
probe, although only validation entered the selection calculation. There was therefore no leakage
into the learning-rate choice, but test was not held untouched until the final rate. Future v4
runs use validation-only grid probes, then retrain the selected rate from pristine base weights
before evaluating test once.

## Amendment 2 — post-result family-clustered reanalysis

Independent review of the powered result found that treating 1,200 items as independent Bernoulli
trials ignored clustering inside twelve task families, four of which were saturated. This
amendment was introduced **after the result was seen**. It is not pre-registered and must not be
presented as such.

The criterion order and configured practical margin remain unchanged. The revised tie test pairs
candidates by family and forms a two-sided Student-t interval over the twelve family differences.
A criterion ties if its point difference is at most `0.02`, or if that paired interval intersects
the practical tie region `[-0.02, +0.02]`. The sign test is diagnostic, not an additional gate.
Minimum-threshold checks likewise use the upper bound of the family-clustered interval. Pooled
Wilson figures remain descriptive and are explicitly labelled optimistic.

The analysis-only evaluator read the frozen v3 candidate and seed records, wrote
`amendment-2-analysis.json` under schema `pretrained-base-derived-analysis-v1`, and regenerated
`ranking.json` under schema `pretrained-base-ranking-v4`. It did not load, train, or evaluate a
model. The original powered config is preserved byte-for-byte as `powered-sweep-config.yaml`; its
SHA-256 still matches the frozen manifest.

## Amendment 3 — pre-measurement NFC contract and expanded candidate slate

Amendment 3 was recorded on 2026-10-14 before any learnability measurement of
Pythia-160M, Qwen2.5-0.5B, SmolLM2-1.7B, or Qwen2.5-1.5B. It is a legitimate
pre-measurement amendment for these candidates: Pythia and Qwen2.5-0.5B were rejected before
their powered probes, and the two larger candidates had not previously been in the candidate
slate. No learnability result for any of the four was available when this amendment was made.

The tokenizer hard constraint is amended from exact code-point round-trip for arbitrary UTF-8
to both of the following requirements:

1. configured UTF-8 diagnostic samples must round-trip exactly after the input is NFC-normalised;
2. every prompt, target, and training target in the measured task stream must round-trip exactly
   as stored.

The verifier contract now NFC-normalises expected and generated text before comparison. The
cost of this relaxation is explicit: canonically equivalent but non-NFC input is altered, so
the system no longer promises preservation of the caller's original Unicode code-point
sequence. The measured task stream is ASCII, so the relaxation changes candidate eligibility
without changing any measured task string.

The candidate slate adds `HuggingFaceTB/SmolLM2-1.7B` at immutable revision
`effd688a12921b4cc83e3312b6feb579f70f9c71` (1,711,376,384 parameters, Apache-2.0) and
`Qwen/Qwen2.5-1.5B` at immutable revision
`8faed761d45a263340a0528343f099c05c9a4323` (1,543,714,304 parameters, Apache-2.0).
All four new measurements use the unchanged Amendment 1 powered budgets and the Amendment 2
family-clustered tie rule. The new evidence is written under
`experiments/base_model_candidates/amendment-3/`; prior candidate and seed records remain
byte-identical, and the combined ranking records the provenance and digest of every source
record.

## Amendment 4 — post-hoc memory-recipe correction, pre-registered rerun

Amendment 4 was recorded at **2026-10-14T18:23:17+05:30**, before any Amendment 4
equivalence or candidate measurement. It is explicitly **post hoc**: the trigger was the observed
Amendment 3 rejection of Qwen2.5-0.5B at a 43.17 GB MPS driver peak despite an estimated
9.88 GB full-fine-tuning parameter state. The excess was dominated by avoidable activation
materialisation, particularly full-sequence fp32 logits over Qwen's 151,936-token vocabulary,
and the host had also entered heavy swap. That result diagnoses a defect in the measurement
recipe, not permission to weaken the memory rule. No Amendment 4 result was available when this
amendment was recorded.

The effective batch size of 8, 200 optimizer steps per family, learning-rate grid
`[1e-5, 3e-5, 1e-4]`, 20 warmup steps, fp32 weights and optimizer state, test items, verifier
contracts including Amendment 3, decision rule, `0.02` tie margin, 36 GiB target, and 25%
headroom remain unchanged.

Only the memory recipe changes:

1. Training uses gradient accumulation with a configurable micro-batch. For each candidate the
   evaluator selects the largest of `[8, 4, 2, 1]` that remains under the unchanged memory cap,
   and records every attempted size, its measured peak and swap evidence, and the selected size.
2. The language-model head and cross-entropy are computed only for positions carrying a target:
   hidden states are sliced before `lm_head` (or an equivalent chunked cross-entropy is used), so
   prompt-position full-vocabulary fp32 logits are never materialised.
3. Gradient checkpointing is disabled by default and is enabled only when a candidate still does
   not fit at micro-batch 1. Its use is recorded.

Before the candidate sweep, the evaluator must prove equivalence on SmolLM2-135M and GPT-2. From
identical pristine weights and deterministic example order, the legacy batch-8/full-logit recipe
and the Amendment 4 target-only recipe run three optimizer steps at each proof model's selected
largest fitting micro-batch. At every step, the accumulated mean loss must differ by no more than
`5e-5` absolute. After every step, the parameter-update vectors must differ by no more than `1e-6`
maximum absolute and `1e-4` relative L2 norm. The proof record includes per-step values, selected
micro-batches, and model revisions. Any failed tolerance stops the workflow before candidate
measurement; tolerances are not revised after observing the proof.

A pre-measurement clarification was recorded at **2026-10-14T18:26:38+05:30**, still before any
equivalence result. The proof micro-batch is selected rather than forced to 4 because GPT-2 has
training-time dropout: splitting a batch changes stochastic-mask ordering even when the
accumulated objective is mathematically identical. A proof model that selects a micro-batch below
8 must nevertheless pass the same parameter-update tolerance or stop the sweep. Offline
deterministic tiny-model tests separately verify accumulation at smaller micro-batches.

One recipe applies to every selectable candidate. GPT-2, SmolLM2-135M, Pythia-160M,
SmolLM2-360M, Qwen2.5-0.5B, Qwen2.5-1.5B, and SmolLM2-1.7B are rerun in actual
unique-parameter-count order, smallest-first, into
`experiments/base_model_candidates/amendment-4/`. Earlier amendment directories remain immutable
history, and the Amendment 4 ranking uses only Amendment 4 candidate and repeat-seed records.

The evaluator records system swap used immediately before and after each candidate in addition to
process and MPS peak memory. Swap growth has a configurable 1 GiB default limit. Exceeding that
limit aborts and records the candidate as memory-infeasible because a machine that must increase
swap does not actually hold the workload. If a candidate still exceeds the unchanged cap at
micro-batch 1 with checkpointing, it is honestly rejected; the cap is not raised.

### Amendment 4 equivalence result — failed; sweep halted

The equivalence gate ran under manifest SHA-256
`e468d0700f55425a6ae0d189216d7acbfaf7152fd4c2a9c3e43c7d78786b9a47` and correlation ID
`107cc59a-ab49-4217-a52d-ce9ab8185ed1`. SmolLM2-135M selected micro-batch 8 without
checkpointing. Its one-step worst-case calibration peaked at 4,623,613,952 bytes, below the
28,991,029,248-byte cap, and system swap remained 7,500,726,272 bytes before and after with zero
growth.

The first T1 optimizer step then failed the pre-registered parameter-update equivalence tolerance:

| Measure | Legacy batch 8 | Amendment 4 batch 8 | Difference | Tolerance | Result |
| --- | ---: | ---: | ---: | ---: | --- |
| Loss | 4.8913888931 | 4.8913890294 | 0.0000001362 absolute | 0.00005 | pass |
| Parameter update, maximum absolute | — | — | 0.0000009239 | 0.000001 | pass |
| Parameter update, relative L2 | — | — | 0.0018891581 | 0.0001 | **fail** |

The proof used train mode and did not disable dropout. It stopped after step 1 of 3 and before the
GPT-2 proof, exactly as the registered fail-closed rule requires. No candidate worker, repeat-seed
worker, Amendment 4 ranking, per-family result, or pairwise comparison was produced. The failed
proof is preserved in `experiments/base_model_candidates/amendment-4/equivalence.json`; revising
the tolerance after seeing it would invalidate the preregistration. The older results below remain
history and are not an Amendment 4 ranking.

### Amendment 4a — equivalence against a measured numerical noise floor

Amendment 4a was recorded at **2026-10-14T18:44:10+05:30**, after the failed Amendment 4 gate
above and before any Amendment 4a measurement. The original `1e-4` relative-L2 parameter-update
tolerance is not changed: moving it after observing a failure would violate preregistration. The
failed `equivalence.json` remains immutable. Amendment 4a tests the review hypothesis that the
recipes implement the same mathematical objective, while float32 reduction-order noise is
amplified by AdamW's first update for parameters whose gradients are near zero. The new evidence is
written separately to
`experiments/base_model_candidates/amendment-4a/equivalence-amendment-4a.json`; Amendment 4a uses
its own manifest and candidate records because changed evaluator and configuration digests cannot
be attached to the frozen Amendment 4 manifest.

The replacement equivalence gate has three mandatory checks. All three must pass before any
Amendment 4 candidate worker starts:

1. **Exact mathematics (float64, CPU).** For SmolLM2-135M and GPT-2, use the same batch of eight
   examples and identical pristine weights. Compare the Amendment 1 batch-8 full-logit gradient
   with the Amendment 4 target-only gradient accumulated at the micro-batch selected by the sweep,
   and separately at micro-batch 1. Relative-L2 gradient difference must be at most `1e-10` and
   absolute loss difference at most `1e-12` for every comparison. This check measures gradients
   before any optimizer step. Failure proves a real recipe defect and requires root-cause repair
   plus a regression test before the gate may be rerun.
2. **Measured noise floor (float32, MPS).** For at least three deterministic batches per proof
   model, measure the legacy recipe against itself with the examples permuted within the same
   batch. Record relative-L2 differences for both the full gradient vector and the first AdamW
   parameter-update vector. On each batch and for each quantity separately, the target-only recipe
   versus the unpermuted legacy recipe must be no greater than twice that batch's legacy-versus-
   permuted-legacy difference. A zero control floor therefore permits no non-zero recipe gap.
3. **Training trajectory (float32, MPS).** On SmolLM2-135M, run 50 optimizer steps for the legacy
   recipe, the target-only recipe, and a legacy control whose examples are permuted within each
   otherwise identical batch. All runs use identical pristine weights, seed, batch sequence, and
   optimizer settings. The maximum absolute target-only-versus-legacy loss-trajectory difference
   must be at most twice the maximum absolute permuted-legacy-versus-legacy trajectory difference.
   Final test accuracy on every configured probe family, and the mean across families, must agree
   between target-only and legacy within the decision rule's unchanged `0.02` tie margin.

All equivalence gradients are computed with dropout disabled while parameters remain trainable;
otherwise a within-batch permutation assigns different stochastic masks to examples and is not a
mathematically identical control. Batches are formed deterministically by cycling family order
first and training-item index second, eight examples per optimizer step. The control permutation
is the exact reversal of each batch. Each comparison variant resets the same configured RNG seed
before model execution. The trajectory uses the first pre-registered learning rate, unchanged
weight decay and warmup scheduler, and the same round-robin batch sequence for all three variants.

The evidence records devices, dtypes, model revisions, batch identities and permutations,
micro-batches, seeds, optimizer settings, per-batch/per-step measurements, aggregate maxima, and
the pass/fail decision for each check. Any check that still fails after a genuine source-level bug
investigation halts the workflow. If all checks pass, the full seven-candidate Amendment 4 sweep
resumes unchanged: smallest actual unique parameter count first, one target-only recipe, swap
self-detection, honest memory rejection, and the pre-registered decision rule.

#### Amendment 4a infrastructure interruption and pre-registered retry

The first Amendment 4a attempt started after preregistration under correlation ID
`1feec53e-47f1-4942-a0e5-fbd7c65b7ae3` but produced no serializable proof result. The loader had
placed SmolLM2-135M on MPS; the exact-proof helper then requested a combined transfer to CPU and
float64. MPS does not support float64, and the combined operation produced non-finite losses and
gradients before JSON's finite-number guard stopped serialization. This is an evaluator transport
defect, not a pass or fail of either recipe. Its `running` manifest and evidence remain preserved
under `experiments/base_model_candidates/amendment-4a/`.

At **2026-10-14T18:57:54+05:30**, before a corrected measurement, the retry was registered with no
change to data, models, thresholds, multiplier, step count, optimizer, or decision rule. The only
code correction is an ordered MPS-to-CPU transfer followed by a CPU float64 cast, plus mandatory
finite-value validation that records a structured failure instead of crashing serialization. A
regression test fixes that transfer order. The separately manifested retry writes to
`experiments/base_model_candidates/amendment-4a/attempt-2/`; any numerical result from it is still
judged solely by the already registered Amendment 4a criteria above.

#### Amendment 4a result — exact mathematics passed; noise floor failed

The corrected retry ran under manifest SHA-256
`74eb5e51d7ec04187f59e4a6d9e321f31fd7eddda8a5b69361043f664db9771a` and correlation ID
`98c783a2-9730-405c-9d28-09af5a9d9cc5`. Exact float64 CPU mathematics passed for both proof
models. SmolLM2-135M had zero loss and gradient difference at micro-batch 8; at micro-batch 1 its
loss difference was `1.33e-15` and relative-L2 gradient difference was `4.17e-16`. GPT-2 had zero
loss difference at both micro-batches, zero gradient difference at micro-batch 8, and `2.46e-16`
relative-L2 gradient difference at micro-batch 1. These are below the registered `1e-12` and
`1e-10` limits.

The float32 MPS noise-floor stage then failed on SmolLM2-135M batch 1:

| Quantity | Old vs old, reversed | New vs old | Registered limit | Result |
| --- | ---: | ---: | ---: | --- |
| Gradient relative L2 | `2.6106e-7` | `9.8403e-6` | `5.2212e-7` | **fail** |
| Step-1 AdamW update relative L2 | `2.0470e-5` | `7.2646e-4` | `4.0941e-5` | **fail** |

Batches 2 and 3 passed both quantities, but the rule is batchwise and requires every batch to
pass. The bug hunt removed boolean advanced-index gradient scatter in favour of validated
contiguous target slices and aligned target-only cross-entropy with legacy mean reduction plus
exact token-fraction accumulation. The MPS warning disappeared, but a direct remeasurement of the
same three batches retained the batch-1 failure (`9.8403e-6` gradient and `7.2646e-4` update,
within rounding of the persisted result). The remaining difference is therefore attributable to
the float32 MPS projection/reduction shape change itself, not defective target selection, token
weighting, or accumulation.

The gate stopped before GPT-2's MPS noise measurement, the 50-step trajectory check, and every
candidate worker. No Amendment 4a ranking or base-model selection exists. The `1e-4` historical
tolerance and every Amendment 4a criterion remain unchanged.

### Amendment 4b — outcome-gated rerun after a mis-specified control

Amendment 4b was recorded at **2026-10-14T19:10:13+05:30**, after the failed Amendment 4a gate
above and before any Amendment 4b measurement. The float32 MPS control in Amendment 4a was
**mis-specified by the orchestrator**. Permuting examples within a batch preserves every tensor
shape and kernel path and therefore measures only a lower bound of float32 noise. The target-only
recipe changes tensor shapes because the language-model head and log-softmax run only on target
positions; MPS can consequently select different kernels, tiling, and reduction orders. A
shape-preserving control cannot bound the rounding of a shape-changing computation. The failed
Amendment 4a control therefore says nothing about whether the recipes train differently, and it
does not justify raising a tolerance. Amendment 4a remains recorded as failed, and its evidence
and criteria remain immutable.

Amendment 4b uses the following pre-registered protocol:

1. **Mathematical identity.** The passed Amendment 4a float64 CPU proof stands. At the start of
   the Amendment 4b run, rerun that proof once for SmolLM2-135M and GPT-2 so its complete evidence
   is attached to the Amendment 4b manifest. The unchanged limits remain `1e-12` absolute loss
   difference and `1e-10` gradient relative-L2 difference.
2. **Shape-changing control, informational only and never a gate.** In float32 on MPS, compare
   the old recipe with the old recipe computed by applying the language-model head in sequence
   chunks. This preserves the old mathematical objective while changing tensor shapes. Report
   its gradient and first AdamW update differences beside the new-versus-old differences so the
   record shows whether new-versus-old lies within the observed noise of shape changes. This
   comparison is diagnostic only: no result from it may halt the run.
3. **Outcome gate.** On SmolLM2-135M, train 50 optimizer steps from identical pristine weights
   with the old recipe at seeds `0` and `1`, and with the new target-only recipe at seed `0`. All
   three runs use the identical deterministic data order and otherwise identical optimizer and
   scheduler settings. The maximum absolute loss difference between new seed `0` and old seed
   `0` must be no greater than the maximum absolute difference between old seed `0` and old seed
   `1`. Final probe test accuracy for new seed `0` and old seed `0` must differ by no more than
   the larger of the decision rule's unchanged `0.02` tie margin and the absolute final probe test
   accuracy difference between old seeds `0` and `1`. If either criterion fails, the workflow
   stops and reports the failure; no candidate sweep may start.
4. **Sweep after a passed gate only.** If and only if the outcome gate passes, rerun all seven
   selectable candidates into the new `experiments/base_model_candidates/amendment-4b/` output
   directory, ordered smallest first by actual unique parameter count. One target-only recipe
   applies to every candidate. The existing 1 GiB swap-growth self-detection, honest memory
   rejection, unchanged decision rule, and its authorised base-model selection remain mandatory.
   The result reports the full ranking, per-family table, pairwise clustered comparisons, memory,
   selected micro-batch, and swap evidence for every candidate.

The Amendment 4b evidence records the manifest and correlation ID, devices, dtypes, immutable
model revisions, batch identities and data order, literal seeds, optimizer and scheduler settings,
all per-step losses, final probe test accuracies, gradient and update comparisons, aggregate gate
bounds, and the pass/fail decision. No Amendment 4 or Amendment 4a evidence file is modified.

#### Amendment 4b result — accuracy passed; trajectory failed; sweep halted

Amendment 4b ran under manifest SHA-256
`2d9afbd704d182f3e7147029a05f9807458605da19732f80403ea70eb59ed82b` and correlation ID
`74729fe9-9a75-4194-b085-2f15d6de0ecf`. The repeated float64 CPU proof passed with the same
results as Amendment 4a: SmolLM2-135M had zero loss and gradient difference at micro-batch 8 and
`1.33e-15` loss difference with `4.17e-16` gradient relative L2 at micro-batch 1; GPT-2 had zero
loss difference at both sizes, zero gradient difference at micro-batch 8, and `2.46e-16` gradient
relative L2 at micro-batch 1. Both selected micro-batch 8 without checkpointing.

The non-gating float32 MPS shape-changing diagnostic completed for all three batches on both proof
models. Every new-versus-old gradient and update difference was smaller than its same-objective
chunked-old-versus-old comparison:

| Model | Batch | Chunked old vs old gradient rel-L2 | New vs old gradient rel-L2 | Chunked old vs old update rel-L2 | New vs old update rel-L2 |
| --- | ---: | ---: | ---: | ---: | ---: |
| SmolLM2-135M | 1 | `1.6874e-5` | `9.8403e-6` | `1.3284e-3` | `7.2646e-4` |
| SmolLM2-135M | 2 | `1.7845e-5` | `2.8646e-9` | `9.1987e-4` | `0` |
| SmolLM2-135M | 3 | `2.0419e-5` | `4.1884e-9` | `1.1063e-3` | `0` |
| GPT-2 | 1 | `1.7937e-5` | `1.0594e-5` | `5.6938e-3` | `3.9179e-3` |
| GPT-2 | 2 | `4.4837e-5` | `1.0414e-9` | `7.2348e-3` | `0` |
| GPT-2 | 3 | `4.4062e-5` | `9.9898e-10` | `6.9973e-3` | `2.2554e-9` |

This diagnostic is informational and did not decide the run. The mandatory 50-step outcome gate
then failed only its trajectory criterion. The maximum old-seed-0 versus old-seed-1 loss
difference was `2.6226043701171875e-6`; the maximum new-seed-0 versus old-seed-0 difference was
`3.5762786865234375e-6`, which exceeds the registered seed-to-seed bound. Final mean probe test
accuracy was exactly `0.2875` for all three runs, and every family accuracy was identical, so the
accuracy difference was `0`, passing the `max(0.02, 0) = 0.02` accuracy limit.

The outcome measurement peaked at 10,076,864,512 bytes by the configured hard-constraint metric,
selected micro-batch 8 without checkpointing, and recorded zero swap peak growth; system swap fell
from 7,265,845,248 to 7,148,404,736 bytes. The fail-closed gate stopped before every candidate
worker. No Amendment 4b ranking, per-family candidate table, pairwise candidate comparison,
repeat-seed run, or base-model selection exists.

#### Owner override of Amendment 4b (2026-10-14)

Repository owner **raghunath** reviewed the Amendment 4b evidence and authorised an explicit
override of its failed trajectory gate. The float64 mathematical-identity proof passed at about
`1e-15`; the non-gating shape-change diagnostic showed that new-versus-old differences were
within the observed shape-change noise; and the final probe accuracy gate passed at exactly
`0.2875` for every run and every family. The trajectory criterion failed by
`9.5367431640625e-7`: new-versus-old reached `3.5762786865234375e-6` against the
`2.6226043701171875e-6` old-seed variation bound, a difference of a few float32 ULPs. Because
the seed control held data order fixed, it measured only rounding variation rather than data-order
variation.

The owner accepted this narrow discrepancy because every candidate is evaluated under the same
target-only recipe, preserving the fairness of the candidate comparison. The failed Amendment 4b
gate and all Amendment 4, 4a, and 4b evidence remain unchanged. The override is explicit in the
run configuration and is copied into the new run manifest and every candidate record.

A confirmation is pre-registered: when cloud GPUs become available, rerun the top two candidates
under the original Amendment 1 recipe on an NVIDIA A100 80GB. If that confirmation changes the
winner, the original-recipe A100 result decides the base-model selection.

### Amendment 4c — candidate-footprint feasibility and environment-invalid retries

Amendment 4c was recorded at **2026-10-14T23:07:45+05:30**, after the orchestrator stopped the
owner-override sweep at about 23:05 IST and before any Amendment 4c measurement. The orchestrator
specified Amendment 4b's per-candidate rule to abort and reject a candidate when **system** swap
grew by more than 1 GiB. That rule was defective: system swap measures the whole host, macOS keeps
swap allocated after pressure subsides, and the IDE, browser, and MCP servers contribute memory
pressure outside the candidate process. The preserved `amendment-4b-override` evidence therefore
does not establish candidate infeasibility. In particular, SmolLM2-135M was killed after 1,429
seconds at about an 8 GB peak class with 2.0 GB system-swap growth, while SmolLM2-360M completed at
a 15.7 GB peak with 0.655 accuracy; Qwen2.5-0.5B and Qwen2.5-1.5B were killed at 1.9 GB and 1.3 GB
swap growth respectively. System swap was already about 6.3–7.0 GB of 7.2 GB. A smaller
candidate cannot honestly be declared memory-infeasible while a larger candidate completes merely
because unrelated host activity changed a global counter.

The following rule is pre-registered for one fresh seven-candidate run, smallest actual unique
parameter count first, using the Amendment 4 target-only recipe and the existing owner override:

1. Candidate feasibility is decided only from the candidate's own measured
   `max(process RSS peak, MPS driver allocation peak)`. The cap remains 36 GiB with 25% headroom
   (28,991,029,248 bytes). A peak above that cap is a hard-constraint rejection; system swap and
   host pressure can never produce that verdict.
2. System swap and the system free-memory percentage reported by `memory_pressure` are recorded
   before and after every attempt. Swap growth above the configured limit or free memory below the
   configured normal threshold makes only that attempt `environment_invalid`. The orchestrator
   waits for the configured normal-pressure threshold, subject to a configured timeout, then
   retries up to the configured maximum number of attempts (default two). If every attempt is
   environment-invalid, the candidate is `not_measured_environment`, never rejected or failed.
3. Ranking fails closed unless every non-reference candidate either completed or was rejected for
   its own footprint. Any `not_measured_environment`, missing, failed, or otherwise incomplete
   candidate produces no selection and names every missing candidate. A base cannot win because a
   competitor was not measured.
4. Amendment 4c uses a new config, evaluator digest, manifest, correlation ID, output directory,
   candidate records, attempt monitors, and ranking. No Amendment 4b or
   `amendment-4b-override` record or log is reused, modified, or deleted.

The implementation is accepted only with deterministic tests proving footprint-over-cap rejection,
swap-growth invalidation and retry, persistent environment invalidation with fail-closed ranking,
and an unaffected normal completion. The full pytest coverage gate, Ruff, and strict mypy must pass
before the detached sweep starts.

## Historical powered results and derived analysis

The sweep completed under manifest hash
`c332b3bcf758985d7a89174b151bebb87cab1bc2a5d10834fc2ba70066d8ba41` and correlation ID
`b8f8b4a3-f0c5-48d2-82c1-ae78a4f9a03c`.

| Candidate | Result | Selected LR | Combined BPB | Zero-shot mean | Learned mean (pooled 95% Wilson, optimistic) | Mean gain | Primary seconds |
| --- | --- | ---: | ---: | ---: | --- | ---: | ---: |
| SmolLM2-360M | pass, rank 1 | 3e-5 | 0.8262 | 0.0033 | 0.6567 [0.6293, 0.6830] | 0.6533 | 3,973.27 |
| SmolLM2-135M | pass, rank 2 | 3e-5 | 0.9284 | 0.0025 | 0.5900 [0.5619, 0.6175] | 0.5875 | 2,013.15 |
| GPT-2 | pass, rank 3 | 1e-5 | 1.3839 | 0.0000 | 0.5300 [0.5017, 0.5581] | 0.5300 | 1,350.68 |
| Pythia-160M | reject: exact UTF-8 | — | — | — | probe skipped | — | 1.00 |
| Qwen2.5-0.5B | reject: exact UTF-8 | — | — | — | probe skipped | — | 1.41 |
| Aptivra byte reference | reference only | — | — | — | probe skipped | — | 8.31 |

The pooled Wilson intervals are too narrow for selection because they treat clustered items as
independent. The paired family difference, 360M minus 135M, is `+0.0667` with approximate 95%
interval `[0.0025, 0.1309]`; the sign test is 7 positive, 0 negative, and 5 tied families
(`p = 0.015625`, two-sided, ties excluded). That interval intersects the configured practical tie
region, so learned accuracy is a tie under Amendment 2. The paired gain difference is `+0.0658`
with approximate interval `[0.0004, 0.1313]`; it also ties. The rule therefore falls through to
combined bits per byte, where `0.8262` places 360M ahead of `0.9284`. The selected ID does not
change, but the reason does.

### Aggregate sensitivity to saturated families

T9--T12 contribute 400 successes to every eligible probed candidate and therefore carry no
selection signal. The derived record reports both views:

| Candidate | All families | Excluding saturated T9--T12 |
| --- | ---: | ---: |
| SmolLM2-360M | 788/1,200 = 0.6567 | 388/800 = 0.4850 |
| SmolLM2-135M | 708/1,200 = 0.5900 | 308/800 = 0.3850 |
| GPT-2 | 636/1,200 = 0.5300 | 236/800 = 0.2950 |

All Wilson intervals attached to these pooled counts are labelled optimistic in the derived
record. T9 uses three solution templates; T10, T11, and T12 use two each. Train and test strings
are disjoint only because generated identifiers are renamed; every test template occurs in train.
These families need genuinely unseen test templates before they can support any
continual-learning claim. That redesign and a new sweep are follow-up work, not part of this ADR.

### Learning-rate selection evidence

Pooled validation accuracies for `[1e-5, 3e-5, 1e-4]` were respectively:

- SmolLM2-360M: `0.6200`, `0.6400`, `0.6250` → `3e-5`;
- SmolLM2-135M: `0.5925`, `0.5958`, `0.5817` → `3e-5`; and
- GPT-2: `0.5450`, `0.5408`, `0.5300` → `1e-5`.

Every grid mean and family validation accuracy has a Wilson interval in the candidate record.

### Per-family held-out learned accuracy

Each cell is learned accuracy followed by its 95% Wilson interval; every denominator is 100.
Zero-shot accuracies, gains, success counts, validation results, and gain intervals remain in the
machine-readable candidate records.

| Family | SmolLM2-360M | SmolLM2-135M | GPT-2 |
| --- | --- | --- | --- |
| T1 modular arithmetic | 0.07 [0.03, 0.14] | 0.02 [0.01, 0.07] | 0.07 [0.03, 0.14] |
| T2 string transforms | 0.01 [0.00, 0.05] | 0.01 [0.00, 0.05] | 0.00 [0.00, 0.04] |
| T3 list operations | 0.16 [0.10, 0.24] | 0.11 [0.06, 0.19] | 0.03 [0.01, 0.08] |
| T4 syntax matching | 0.89 [0.81, 0.94] | 0.83 [0.74, 0.89] | 0.69 [0.59, 0.77] |
| T5 boolean logic | 0.89 [0.81, 0.94] | 0.80 [0.71, 0.87] | 0.73 [0.64, 0.81] |
| T6 sequence completion | 0.79 [0.70, 0.86] | 0.45 [0.36, 0.55] | 0.08 [0.04, 0.15] |
| T7 state machines | 0.51 [0.41, 0.61] | 0.48 [0.38, 0.58] | 0.47 [0.38, 0.57] |
| T8 Python microsynth | 0.56 [0.46, 0.65] | 0.38 [0.29, 0.48] | 0.29 [0.21, 0.39] |
| T9 algorithm synthesis | 1.00 [0.96, 1.00] | 1.00 [0.96, 1.00] | 1.00 [0.96, 1.00] |
| T10 docstring to code | 1.00 [0.96, 1.00] | 1.00 [0.96, 1.00] | 1.00 [0.96, 1.00] |
| T11 bug fixer | 1.00 [0.96, 1.00] | 1.00 [0.96, 1.00] | 1.00 [0.96, 1.00] |
| T12 type inferrer | 1.00 [0.96, 1.00] | 1.00 [0.96, 1.00] | 1.00 [0.96, 1.00] |

T6 and T8 supply the largest 360M-versus-135M differences (`+0.34` and `+0.18`); the cautious
preference is driven mostly by those two families. The task stream also limits interpretation:

- T5 has two distinct test targets and a uniform-target chance level of `0.50`; T7 has four and a
  chance level of `0.25`. Their accuracies must be read against those baselines.
- Test targets already occur in train for 97 of 100 T1 items, 73 of 100 T6 items, and 95 of 100
  T8 items, even though prompts are disjoint. These are transfer over familiar answers, not clean
  unseen-target tests.
- T2 remains at 0--1% for every candidate. Review found no stop-rule or budget defect. A plausible,
  unproven explanation is that subword tokenisation is poorly matched to exact character-level
  transforms.
- T9--T12 measure renamed template reproduction, as described above, rather than unseen program
  synthesis.

The counts, overlaps, chance levels, and template signatures are computed by the evaluator and
stored in the derived record; they are not hand-entered annotations.

### Seed-to-seed variance

| Candidate | Seed means (`20261014`, `20261015`, `20261016`) | Across-seed mean | Sample SD |
| --- | --- | ---: | ---: |
| SmolLM2-360M | 0.6567, 0.6208, 0.6367 | 0.6381 | 0.0180 |
| SmolLM2-135M | 0.5900, 0.5875, 0.5983 | 0.5919 | 0.0057 |

The repeat seeds preserve the point ordering but do not turn three runs into strong replication.
360M has greater seed sensitivity; its three-seed mean is `0.0461` above 135M's. These spreads are
descriptive and do not replace the paired family analysis.

### Exact UTF-8 diagnosis

Pythia's `GPTNeoXTokenizer` and Qwen's `Qwen2Tokenizer` both report backend normalizer `NFC()`.
For input `e` + `U+0301` + ` and é`, both decode to two precomposed `U+00E9` characters. The
failure persists with `clean_up_tokenization_spaces` both `false` and `true`, encoding uses
`add_special_tokens=False`, and decoding uses `skip_special_tokens=False`. Pythia emitted token
IDs `[860, 285, 3974]`; Qwen emitted `[963, 323, 3958]`. This is genuine tokenizer Unicode
normalisation, not cleanup, BOS, or added-token contamination. This supports excluding both under
the chosen exact code-point round-trip constraint. Because neither candidate was probed, it does
**not** support a claim that SmolLM2 learns better than either would.

The measured task stream is ASCII. The owner may instead define the contract as exact round-trip
on NFC-normalised text. That would make Pythia-160M and Qwen2.5-0.5B eligible, at the cost of an
explicit normalisation step in the verifier contract and a new powered probe before selection.

## Proposed decision

Amendment 4b produces **no valid selection** because its mandatory outcome gate failed before the
sweep. The owner authorised the pre-registered rule to choose the base if the gate passed, but that
authority does not override a failed gate. The earlier evidence gave a cautious historical
preference for SmolLM2-360M, but it cannot answer the expanded seven-candidate question under the
corrected memory recipe and is not promoted into the Amendment 4b decision. The ADR stays
**Proposed**.

Amendment 4c supersedes only the invalid environment-verdict mechanism, not the frozen 4b
measurements or owner override. It has no result at preregistration time. The ADR remains
**Proposed** until all seven fresh 4c candidate records satisfy the completeness rule and the
resulting ranking receives owner review.

## Decisions for the owner

1. Decide whether to abandon the target-only memory recipe or authorise a future, separately
   pre-registered outcome rule. Amendment 4b's observed trajectory bound must not be changed in
   place after its failure.
2. Do not select a base model from Amendment 4b: no Amendment 4b candidate ranking exists.

## Tokenizer and stop-rule contract changes

Acceptance would replace the assumption that token ID equals a UTF-8 byte. Aptivra task families
continue to own decoded answer stop strings, while the adapter maps those strings to tokenized
generation behavior and also stops on model EOS. Existing `stop_token_ids` byte ordinals remain a
legacy bridge; integration code must not treat them as subword token IDs.

Training targets append a validated model EOS terminator after the family answer terminator.
Prompt length, target length, truncation, and padding are evaluated in model-token units. Examples
that cannot retain prompt plus target within context are rejected, not silently truncated.

## Existing checkpoint and Hub repository

`raghunath1/aptivra-200m-tiered-memory` remains an immutable byte-level historical reference. The
powered run checked its cheap compatibility constraints and excluded it before probe work because
it is reference-only. Its `12.7898` combined bits per byte in this ADR comes only from the
archived first underpowered sweep; the powered protocol did not measure its likelihood or
learnability. No Hub repository, tag, model card, or checkpoint was changed.

## Licence and attribution

All seven selectable candidates are Apache-2.0 or MIT. A derived Aptivra release must retain the
selected upstream licence text, copyright and attribution notices, identify the exact upstream
repository and revision, mark Aptivra modifications, and preserve required NOTICE content. Legal
acceptance remains with the repository owner.

## Alternatives considered

1. **Pretrain a custom 50M model.** This preserves architectural control but requires a corpus,
   contamination controls, substantial compute, and a separate validation campaign.
2. **Keep the byte backbone and train longer.** This does not correct missing pretraining by
   itself and retains longer byte sequences.
3. **Select the smallest model without measurement.** Parameter count does not establish language
   quality, learnability, tokenizer correctness, MPS cost, or attachment structure.
4. **Select a public benchmark winner.** Unrelated benchmarks do not prove compatibility with
   Aptivra's verifier stream, hardware, stop rules, and expert topology.

## Consequences

### Positive

- The eligible probed candidates now have an auditable, family-clustered comparison and an
  explicit cost trade-off for owner review.
- The ranking exposes pooled and unsaturated aggregates instead of hiding the 400 non-informative
  successes.
- Bits per byte remains tokenizer-independent, while immutable records make the run auditable.

### Negative / trade-offs

- Aptivra inherits a third-party tokenizer, architecture, licence, and attribution duty.
- The Amendment 4b target-only recipe passed exact mathematics and final probe accuracy but
  exceeded the registered seed-to-seed loss-trajectory bound, so none of its configured
  candidates was measured.
- Historically, the 360M base cost more memory and time than 135M; its Amendment 1 primary peak
  was 20,584,579,072 bytes, inside the configured headroom limit.
- The result has only twelve independent family units, T9--T12 are renamed-template tasks, several
  other families reuse training targets, and 360M shows more seed variance than 135M.
- Pythia-160M and Qwen2.5-0.5B have Amendment 3 learnability results, but those measurements use
  the defective full-logit memory recipe and do not form a complete ranking.
- ADR-0002's byte-backbone assumptions no longer govern Tier 3 if this proposal is accepted.

## Retrospective

The first sweep spent only 349 candidate-seconds and let one test item decide the headline. The
powered rerun added data but still overstated precision by pooling clustered items, exposed test
for unselected learning rates, and allowed renamed templates to inflate the mean. Independent
review caught those analysis defects. Prevention is structural: future runs keep test untouched
until the selected-rate retrain; records include stream properties, pooled metrics labelled
optimistic, paired family intervals, sign tests, saturated-family sensitivity, and seed spread;
tests enforce clustered fallthrough and source-record immutability. The pilot also underestimated
end-to-end workflow time by 14.8%; future estimates must pilot a complete family mix and include
validation/test decoding, not extrapolate optimizer cost alone.

Amendment 4 correctly stopped itself before propagating an unproven recipe. The loss and maximum
absolute update checks passed, while the relative-L2 update check exposed a small distributed
update difference that the aggregate loss alone would have hidden. Prevention is structural: the
equivalence record is manifest-bound, candidate workers refuse to start unless it passes, and
tests cover target-only projection, token-weighted accumulation, descending memory trials,
checkpoint fallback, terminal memory rejection, and swap-growth detection. A future protocol must
be registered separately; this failed tolerance must not be edited in place.

Amendment 4a then used a shape-preserving permutation control that was mis-specified by the
orchestrator for a shape-changing computation. Amendment 4b corrected that methodological defect:
the same-objective chunked-head diagnostic bounded every observed new-versus-old gradient and
update difference, and the final accuracy outcome was identical across all three trajectories.
The registered trajectory criterion nevertheless failed by `9.5367431640625e-7`. Jidoka held:
the evaluator stopped before the sweep, preserved the failed evidence, and made no selection.
Prevention is encoded in separate protocol selection, manifest-bound 4b evidence, tests proving
that informational diagnostics cannot gate, and direct tests of both outcome inequalities.

The stopped owner-override sweep exposed a second measurement defect: the orchestrator-specified
swap rule confused whole-host state with candidate footprint. Amendment 4c prevents recurrence by
making swap and `memory_pressure` environment-health evidence only, retaining every invalid
attempt, retrying after bounded recovery, and refusing selection whenever a competitor remains
unmeasured. Candidate infeasibility now requires the already established absolute
`max(process RSS peak, MPS driver allocation peak)` to exceed the unchanged cap.

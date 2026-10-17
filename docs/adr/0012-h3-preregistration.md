# ADR-0012: H3 Continual-Learning Pre-registration

- **Status:** Proposed
- **Date:** 2026-10-17
- **Decider:** Repository owner (acceptance required before any H3 training)
- **Depends on:** ADR-0005, ADR-0008, ADR-0011, proposed ADR-0009, and proposed ADR-0010

> **Review status (2026-10-17): not ready for acceptance.** The lane-8a calibration evidence is
> valid. The repeatability runs are not: they trained 2 optimizer steps on 16 examples and
> evaluated 8. At that budget Aptivra scored 0.0 on both streams and naive fine-tuning 0.0 on v2,
> so the observed zero variation is resolution censoring, not a noise floor. The thresholds below
> (multiples of 1/8) and the power figure therefore come from the evaluation's resolution, not
> from measured variation. Lane 8b re-measures the noise floor at the H3 per-cycle budget,
> investigates the Aptivra-zero, EWC non-finite and LoRA step-time anomalies, and revises this
> ADR before it is offered for acceptance.

## Context and evidence boundary

ADR-0011 requires H3 to be specified before its results exist. This ADR does not accept H3 and
does not contain an H3 result. It freezes the proposed design from the bounded real-MPS lane-8a
pilot in [`0012-evidence/pilot-evidence.json`](0012-evidence/pilot-evidence.json), schema
`aptivra.h3_mps_pilot_evidence.v1` (the **pilot evidence**). The pilot completed in
7,438.015 seconds on arm64 macOS with Python 3.12.13, Torch 2.14.1, Transformers 5.18.0, PEFT
0.21.2, MPS available, and both Hugging Face offline flags set. It used only the cached
`HuggingFaceTB/SmolLM2-360M` revision
`f8027fd0eaeea54caa13c31d31b9fdc459c38b49`.

The evidence references below are JSON selectors, not copied chronological-stream text:

| ID | Pilot evidence selector | Measurement |
| --- | --- | --- |
| E1 | `preflight.synthetic_v2` | v2 audit passed; 12 families; config, audit-plan and audit hashes bound. |
| E2 | `preflight.chronological_issues` | Issue audit passed; 10,065 records in 17 shards; raw/config/manifest/audit hashes bound. |
| E3 | `calibration[*]` plus each row's hash-bound `quiet_machine.record_file` | All 48 attempts, including non-fitting and failed attempts, sequence lengths, memory, loss, timing, swap and losslessly sharded quiet samples. |
| E4 | `noise_floor[*].same_seed_absolute_variation` | All four decision metrics varied by 0.0 in every method/stream duplicated-seed pair. |
| E5 | `noise_floor[*].different_seed_max_absolute_variation` | All four decision metrics varied by 0.0 in every method/stream distinct-seed comparison. |
| E6 | `repeatability[*].metrics` | Eight examples per task make the per-task accuracy quantum 0.125 and the two-task final-average quantum 0.0625. |
| E7 | `repeatability[method=aptivra_full].metrics.promotions[*].routing_probe` | Sixteen-prompt probes: 1.0 eligible coverage, 0.50 overall misroute rate, 1.0 fallback false-route rate and 0.0 oracle gap; eligible accuracy quantum 0.125 and all-prompt routing quantum 0.0625. |
| E8 | `compute_estimate` | Exact stream token counts, median valid ladder seconds/token, generation time, cell arithmetic and 606.234-hour unreduced estimate. |
| E9 | `quiet_gate_events[0]` | One 900.067-second foreign-load window expired after 19 samples; the runner checkpointed it and retried without starting a measured worker. |

Zero observed metric variation is resolution-censored evidence, not bit reproducibility. For a
metric with pilot quantum `q`, this ADR defines measured self-variation
`v* = max(observed same-seed variation, observed different-seed variation, q / 2)`. Every
equivalence or consistency check is `new-vs-old <= k × v*`; no check uses tensor, logit, token or
bitwise identity.

## Pre-flight and method contract

No model may be loaded until E1 and E2 are reproduced from the exact committed manifests and the
chronological raw file hash is `2f4d6f6554f4e762ff7a779fb81ca579d0a38db119952f67786e52997d3ab9ec`.
Raw chronological text remains local and is excluded from every Hub-facing artifact. A missing or
mismatched path halts the experiment.

Every measured process starts from pristine cached weights and passes the registered quiet-machine
gate. Foreign processes are observed and recorded but never killed or reniced. Swap growth above
1,073,741,824 bytes invalidates throughput only. The hard own-footprint cap is 28,991,029,248
bytes. MPS is the only permitted device; CPU fallback, mixed precision, quantization, offload and
network access are forbidden.

The six methods are frozen base, naive sequential full FT, deterministic replay full FT, bounded
online EWC full FT, manual task-ID rank-16 LoRA, and Aptivra full. Frozen-base calibration is
forward-only under `no_grad()` and eval mode, uses target-only loss, and exercises maximum-answer
generation; backward, optimizer, accumulation and checkpointing are recorded as
`not_applicable`. Every other method performs forward, backward and AdamW update steps. Online EWC
initializes its single consolidated diagonal Fisher from gradients on the same real worst-case
batch before warm-up, rejects an identically zero Fisher, and applies the registered penalty
`400.0`. Aptivra full backpropagates only through the active LoRA adapter and candidate router row.

For each stream and micro-batch `k`, calibration selects the `k` longest distinct real examples by
tokenized length with stable-record-ID tie-breaking, then pads all sequences to that stream's
single longest accepted example. The ladder is `[8, 4, 2, 1]`, effective batch is 8, and each
attempt has one warm-up plus three measured steps. Gradient checkpointing is permitted only after
a failed micro-batch-1 training attempt. The largest accepted non-checkpointed micro-batch is used:

| Stream | Frozen | Naive FT | Replay FT | Online EWC | Manual LoRA | Aptivra full |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| synthetic v2 (167 tokens) | 8 | 8 | 8 | 8 | 8 | 8 |
| chronological issues (726 tokens) | 8 | 4 | 4 | 4 | 8 | 8 |

The chronological micro-batch-8 naive and replay attempts peaked at 31,236,161,536 and
31,169,052,672 bytes and are rejected; their swap growth of 4,514,518,138 and 2,218,189,128
bytes also suppresses throughput. Online EWC micro-batch 8 initialized a nonzero Fisher
(initialization loss `5.296557`) but produced a non-finite training loss and is failed. These rows
remain in E3. At the selected micro-batches the chronological peaks are 19,079,888,896 (naive),
19,079,888,896 (replay), 23,644,340,224 (EWC), 25,800,343,552 (manual LoRA), and
25,801,474,048 bytes (Aptivra). Frozen base peaked at 2,743,943,168 bytes. All selected attempts
had finite losses, zero swap growth and no CPU fallback.

The exact parameter arithmetic remains: 361,821,120 tied base parameters; 8,683,520 parameters
per rank-16 all-seven-projection adapter; 104,202,240 adapter parameters for twelve resident
adapters; and `961E` router parameters (`11,532` at `E=12`). Base fp32 weights are
1,447,284,480 bytes. Twelve adapter weights add 416,808,960 bytes; the active adapter's gradient
and two AdamW moments add 104,202,240 bytes, for 1,968,295,680 bytes before negligible router
state and activations. E3, not this static arithmetic, is the fit decision.

## Confirmatory design and power

The five main seeds are exactly `20261017`, `20261018`, `20261019`, `20261020`, and `20261021`,
paired across every method within each stream. Results from v2, chronological issues and v1 are
never pooled. V1 is six one-seed regression cells only and cannot support a new claim.

E4 and E5 observed zero decision-metric variation. E6 gives `q=0.125` for per-task accuracy,
BWT and forgetting on a two-task pair, and `q=0.0625` for final average. The preregistered
standard-deviation guards are therefore `q/2`: 0.0625 for BWT, forgetting, new-task accuracy and
template-OOD, and 0.03125 for final average. A one-sided paired t test with five pairs, alpha 0.05,
and a threshold equal to `2 × SD` has noncentral-t power 0.9748 (`df=4`, noncentrality
`2*sqrt(5)`). This is an explicit planning assumption, not a post-hoc estimate. If the observed
five-seed SD is larger, power is lower and the claim is not rescued by changing seeds or margins.

The primary decision is an intersection-union test: every conjunct below must pass independently
on both claim-bearing streams. An intersection-union claim needs no alpha division because the
claim is accepted only when every component rejects its own null at one-sided alpha 0.05.
Exploratory method-by-method and ablation contrasts use Holm correction within each reported
endpoint/stream family and are labelled exploratory. Confidence intervals are paired across seeds;
normality is checked before a paired t interval, otherwise the exact paired randomization interval
is used. No unpaired replacement is permitted.

## Comparator and kill criterion

The comparator is selected separately per claim-bearing stream before Aptivra results are
unblinded. An eligible non-learned-routing baseline must pass its audit, memory, finite-loss,
completion, five-seed, rollback and compute-matching gates. Manual task-ID LoRA is eligible because
it has oracle adapter selection but no learned router. Among eligible baselines choose the highest
five-seed mean final average; a difference at or below 0.0625 ties (E6), then choose higher BWT,
then lower forgetting, then the fixed conservative order manual LoRA, replay, online EWC, naive
FT, frozen base. Any incomplete required baseline independently kills H3 even if another comparator
can be selected.

H3 passes only if all of the following pass on **each** claim-bearing stream:

1. **BWT superiority:** the lower one-sided 95% paired bound for `Aptivra - comparator` is at least
   `+0.125` (one E6 per-task quantum; `2 × 0.0625` SD guard).
2. **Forgetting superiority:** the lower bound for `comparator forgetting - Aptivra forgetting` is
   at least `+0.125`, from the same E6 quantum and power assumption.
3. **Final-average non-inferiority:** the lower bound for `Aptivra - comparator` is at least
   `-0.0625`, one E6 final-average quantum.
4. **New-task acquisition non-inferiority:** the lower bound is at least `-0.125`, one E6
   per-task quantum.
5. **Template-OOD non-inferiority (v2 only):** the lower bound is at least `-0.125`. The margin is
   transferred from E6's exact-match per-task quantum; template-OOD remains a separate cell and is
   never pooled with in-template accuracy. The chronological stream has no template-OOD cell.
6. **Routing reality:** at confidence threshold 0.65, eligible-adapter coverage is at least 0.875
   (E7 observed 1.0; one eligible miss is 0.125), overall misroute rate including false routing of
   fallback prompts is at most 0.0625 (one E7 all-prompt error quantum), fallback false-route rate
   is at most 0.0625, and task-ID-oracle accuracy minus learned-routing accuracy is at most 0.125
   (one E7 eligible-accuracy quantum). E7's 0.50 pilot misroute and 1.0 false-route rates do not
   satisfy these bounds; the short pilot is diagnostic, not an H3 pass.
7. **Consistency:** any loader/restart/new-vs-old metric check must satisfy
   `absolute(new - old) <= 2 × v*` for its measured metric. With E4/E5 resolution censoring this is
   0.125 for final average and 0.25 for per-task/BWT/forgetting, never bitwise identity.
8. **Operability:** a cap breach, missing seed, failed/OOM required method, verifier infrastructure
   error, non-finite value, CPU fallback, failed rollback, audit mismatch, gate bypass or incomplete
   evidence kills H3. Failed methods remain in every table.

## Compute matching and caps

All updating methods receive the same serialized order, one epoch per task, effective batch 8,
target-only loss, evaluation schedule, optimizer-step ceiling and target-token ceiling. A method
stops at the first applicable cap; unused budget cannot be moved to another method. Frozen base has
no training budget. Hyperparameter selection gets the same declared number of candidates and uses
only training/validation data. Router-only work is reported separately and cannot buy extra adapter
steps. Actual steps, tokens, generations, wall time, FLOPs method, peak memory and cap utilization
are mandatory.

Per consolidation cycle the adapter caps are 100 optimizer steps and 8,192 target tokens; router
caps are 100 optimizer steps and 8 replay examples per mature expert. Evaluation is capped at 64
examples per seen task per cycle: v2 uses 32 stratified in-template plus 32 stratified template-OOD
examples, while chronological and v1 use 64 stratified in-template examples. Sampling is fixed by
the experiment seed and task ID and is identical across methods. Final evaluation uses the same
frozen subsets. A cap overrun aborts and rolls back.

E8 counted exactly 534,755 synthetic-v2, 2,015,255 chronological and 263,315 v1 training sequence
tokens (157,271, 25,700 and 95,637 target tokens). At the original 363-evaluation cap, the exact
cell arithmetic is 60 main cells, 6 v1 cells and 66 ablation seed-stream cells. Measured training
is 34,265.722 + 713.045 + 37,692.294 seconds; capped evaluation is 955,213.578 + 103,824.200 +
1,050,734.936 seconds, totaling 2,182,443.775 seconds or 606.234 hours. That is not a reasonable
local budget.

The preregistered reduction leaves all 60 main cells at five seeds, reduces only the eleven named
ablations to one seed on each stream (22 labelled diagnostic cells), and uses the 64-example cap
above. Scaling E8's measured generation cost and exact cell counts gives 168,412.311 seconds main
evaluation, 18,305.093 seconds v1 evaluation, 61,751.181 seconds ablation evaluation, 34,265.722
seconds main training, 713.045 seconds v1 training and 12,564.098 seconds ablation training:
296,011.450 seconds or 82.225 hours total. This is the proposed-design estimate, not a promise.
Main H3 cells may not be reduced below five seeds. Any expansion requires an amendment before the
additional result is observed.

## Execution order and failure handling

1. Re-run and bind both stream audits, environment, code, config, model and raw/manifest hashes.
2. Reproduce calibration in fresh processes and freeze each method/stream micro-batch above.
3. Run frozen-base vectors, then non-routing baselines in the fixed order naive, replay, EWC and
   manual LoRA, for all five seeds on one stream before opening Aptivra results for that stream.
4. Select the strongest eligible comparator using the frozen rule, still blinded to Aptivra.
5. Run Aptivra full for the same five paired seeds; finish and seal one stream before the other.
6. Run the six one-seed v1 regression cells; they cannot rescue either claim-bearing stream.
7. Run the eleven one-seed-per-stream ablations in the order listed in
   `configs/h3_mps_pilot.yaml`, restoring pristine state after each.
8. Apply corrected intervals, the intersection-union kill criterion and release gates once, then
   write immutable evidence. No threshold, comparator or exclusion changes after step 1.

Every cell checkpoints atomically after completion and resumes only when its run manifest matches.
OOM, non-finite, timeout and failed methods remain in the table with quiet samples and failure
evidence. There is no replacement seed. A rejected Aptivra candidate atomically restores adapter,
router, optimizer, replay, memory, configuration and RNG state; rollback failure kills H3.

## Consequences and retrospective

The design is falsifiable and locally bounded, but its 82.225-hour estimate is still substantial. The
pilot also showed that a one-row router confidently false-routes fallback prompts; routing cannot
be inferred from adapter isolation. Kaizen action: retain the pre-model raw-data/hash check, the
persisted quiet-window samples, fresh-process calibration and explicit fallback false-route metric
as permanent gates. The owner must accept this Proposed ADR before any main H3 cell is run.

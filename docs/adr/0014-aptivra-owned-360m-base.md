# ADR-0014: Pretrain an Aptivra-Owned Small Base Under a Fixed Budget

- **Status:** Proposed
- **Date:** 2026-10-18
- **Decider:** Repository owner
- **Depends on:** ADR-0001, ADR-0005, ADR-0008, ADR-0011, ADR-0012, and proposed
  ADR-0013
- **Primary release target:** private repository `raghunath1/aptivra-360m-continual`

## Context

The original Aptivra base was self-built but under-trained. ADR-0008 measured approximately
12.5 bits per byte (BPB) on held-out text, worse than the 8.0-BPB uniform-byte baseline. Training
another base for only 20 tokens per parameter would repeat the failure mechanism rather than
correct it. Modern useful small language models are trained far beyond the Chinchilla compute
optimum: the completed research found examples in the hundreds to thousands of tokens per
parameter and recommends approximately 150 billion tokens for this 361.8 million-parameter
design.[R1][R2]

The owner wants a base whose weights and training provenance are Aptivra's, but that phrase has
multiple legitimate meanings. A model initialised and pretrained independently has the strongest
provenance claim. Pruning transfers another model's weights; logit distillation transfers another
model's behaviour even if the student starts from random weights. This ADR therefore makes the
meaning of “own” an explicit owner decision rather than hiding it inside an optimisation choice.

The financial constraint is absolute. The research recorded ₹28,796.63 of promotional credit,
expiring on 2027-01-13.[R1] This ADR authorises no training, download, quota request, paid upgrade,
billing change, or remote write. Those are later, separately authorised operations after every
gate below passes.

## Decision

Subject to the owner decisions at the end of this ADR, Aptivra will use a 361,821,120-parameter
base initialised and pretrained from scratch for approximately 150 billion tokens. The run uses a
TPU v6e single-chip Spot allocation only if a measured, approximately one-hour pilot proves the
complete TPU path can finish inside the ₹20,000 main-run allocation. A 135M-class base trained on
the M4 Max is the mandatory fallback if suitable cloud quota has not been granted by 2026-11-20
or if the pilot fails.[R1][R3]

### 1. “Owned” means independent weights and provenance

For this ADR, the recommended ownership contract is:

1. Aptivra specifies the architecture and tokenizer, starts from independently sampled weights,
   and performs pretraining without importing pruned or otherwise transformed parent weights.
2. Aptivra does not use teacher logits. MIT- or Apache-2.0-licensed model outputs may be used only
   as auditable post-training data, never to claim that pretraining was independent.
3. Every input shard, tokenizer artefact, checkpoint, post-training example, and evaluation result
   has a reproducible licence and provenance record.
4. “Owned” does not mean ownership of facts, source texts, the Llama architecture, or third-party
   model behaviour. It means independent model weights and a controlled, defensible training
   lineage.

The owner may instead choose one of these alternatives:

| Method | Quality/cost consequence | Ownership and licence consequence |
| --- | --- | --- |
| From-scratch pretraining (**recommended**) | Highest token requirement and execution risk; cleanest experiment | Independent weights and strongest provenance claim |
| Minitron-style structured pruning | NVIDIA reports up to 40× fewer training tokens for derived compact models.[R4] | The output is derived from the parent weights. Apache-2.0 parents are permissive, but provenance is not independent; Gemma or Llama parents retain their special terms.[R5][R6] |
| Logit distillation | Can improve small-model quality; the research estimates about 1.56× training compute for a 360M student with a 600M teacher and identifies tokenizer mismatch as an additional constraint.[R1] | Student weights start fresh, but behaviour comes from the teacher. Apache-2.0 Qwen has no special model licence obligations; Gemma defines derivatives broadly, and Llama imposes attribution/naming obligations.[R5][R6][R7] |

No method change is permitted merely to rescue a failing run. It requires the owner's recorded
choice and an ADR amendment before execution.

### 2. Primary architecture

The primary model is a stock-compatible Llama causal language model with no custom remote code.
Its geometry follows the proven SmolLM2-360M class while the tokenizer, initial weights, training
data, and checkpoints remain Aptivra-controlled.[R8]

| Field | Decision |
| --- | ---: |
| Layers | 32 |
| Hidden width | 960 |
| Query heads / KV heads | 15 / 5 |
| Head dimension | 64 |
| SwiGLU intermediate width | 2,560 |
| Vocabulary | 49,152, including reserved tokens |
| Embeddings | Tied input/output |
| Maximum context | 8,192; 4,096 device default |
| Position encoding | RoPE, theta 100,000, with no pretraining scaling |
| Norm / bias | RMSNorm / no linear bias |

The derived parameter count is auditable rather than an external estimate:

```text
head_dim = 960 / 15 = 64
kv_width = 5 * 64 = 320

attention/layer = 2*(960*960) + 2*(960*320) = 2,457,600
SwiGLU/layer = 3*960*2,560                 = 7,372,800
RMSNorm/layer = 2*960                     =     1,920
32 transformer layers                     = 314,634,240
tied embedding = 49,152*960               =  47,185,920
final RMSNorm                              =         960
total                                      = 361,821,120
```

The 135M fallback must preserve the same tokenizer and stock runtime contract where measurement
allows, but its exact depth, width, head geometry, and token target are a new pre-registered design
chosen from the Mac memory/throughput pilot. Calling that fallback “360m” would be false; its model
repository and all manifests must use an accurate 135M-class name.

### 3. Token budget and cost comparison

The primary target is **150B non-padding pretraining tokens**, about 415 tokens per parameter. It
is a quality target, not permission to exceed the spend ceiling.[R1] The previous 7.2B/20N plan is
rejected because it reproduces v0's under-training failure.[R1][R2]

The following comparison is a planning model, not a benchmark. It assumes a 361.8M dense model,
the standard `6ND` training-compute estimate, 40% model-FLOP utilisation, and a 1.35 attention and
systems overhead divisor from the completed research.[R1][R9] Prices were rechecked on 2026-10-18
against the Google Cloud Spot price page and Cloud Billing Catalog; the conversion uses the
research planning rate of ₹84.8/USD rather than claiming a live exchange rate.[R1][R10][R11]

| Device, Spot in `us-central1` | Current planning price | Estimated tokens/s | Estimated tokens/₹ | Tokens purchasable for ₹20,000 |
| --- | ---: | ---: | ---: | ---: |
| TPU v6e, one chip | $0.583830/chip-hour | 126k | 9.2M | 183B |
| NVIDIA L4, `g2-standard-8` | $0.512112/hour | 16.6k | 1.38M | 27.5B |
| NVIDIA A100 40GB, `a2-highgpu-1g` | $2.20401/hour | 42.8k | 0.825M | 16.5B |
| NVIDIA H100 80GB, one-GPU A3 estimate | $6.55479/hour | 135.6k | 0.878M | 17.6B |

At the estimated 126k tokens/s, 150B tokens require approximately 331 chip-hours. Raw v6e cost is
about ₹16,400; 20–25% operational overhead produces approximately ₹19,700–₹20,500.[R1][R10] That
range straddles the cap, which is precisely why projected throughput cannot authorise the run.
The pilot's measured end-to-end rate, recovery overhead, and current price must do so.

The table shows that v6e is estimated at approximately 6.7× the L4, 11.1× the A100, and 10.4× the
H100 in tokens per rupee for this workload.[R1][R10] Those ratios do not establish actual v6e
performance; the estimate is unmeasured and the devices use different software stacks.

### 4. The paid pilot is a hard gate

Before a production allocation, run one bounded, approximately one-hour v6e pilot with an
all-in maximum cost of ₹1,000. It must exercise the production tokenizer, streaming/packing path,
model geometry, optimiser, bf16 numerics, checkpoint save/restore, preemption recovery, evaluation,
structured cost ledger, and billing-stop signal. Warm-up and compilation time are reported
separately but included in the cost projection.[R1][R3]

The pilot decision rule is fail-closed:

1. Recompute projected main-run cost from measured sustained non-padding tokens/s, observed
   compilation/checkpoint/recovery overhead, and the current SKU price. Proceed with 150B only if
   the upper-bound projection is no more than ₹20,000.
2. Approximately 126k tokens/s with no more than about 20% overhead is the current expected pass
   region; the cost equation, not this indicative number, controls.[R1]
3. From 90k to below the cost-qualified rate, the 360M/150B run fails. Recalculate an explicitly
   smaller token target and accept it only by owner decision if it remains at least 120B; otherwise
   select the 135M fallback.[R1]
4. Below 90k tokens/s, reject the 360M run and shrink the model or the token count; the recommended
   action is the 135M fallback. Below 40k tokens/s, abandon TPU for this plan; consider L4 plus
   135M only after its own measured value and cost gate.[R1]
5. Any non-finite value, irreproducible resume, missing provenance row, unbounded input stall,
   unhandled preemption, or failure of the billing-stop simulation rejects the run regardless of
   throughput.

This gate deliberately allows the pilot to disprove the plan.

### 5. TPU and Mac software contracts

The primary TPU implementation is JAX with MaxText because it is Google's supported reference
stack for training Llama-class models on v6e. PyTorch/XLA is the evaluated alternative if the
repository's PyTorch model contract materially reduces conversion risk; either choice must be
fixed before the paid pilot and tested through its real TPU/XLA path.[R12][R13][R14]

The Mac rehearsal uses the repository's PyTorch/MPS path. It can prove tokenizer determinism,
data filtering and mixture accounting, loss masks, checkpoint schema, evaluation semantics,
provenance generation, exact token counting, tiny-run learning, and fail-closed controls. It
cannot prove TPU kernels, XLA compilation, JAX/PyTorch-XLA numerical equivalence, sharding, v6e
bf16 stability, TPU throughput/MFU, host-to-device input performance, cloud checkpoint bandwidth,
Spot preemption recovery, quota, or price. Therefore a successful local dry run never waives the
paid TPU pilot.[R1][R3][R12]

### 6. Commercially usable data and WSD schedule

Only sources whose licences permit commercial model training are eligible. Every shard is
licence-allowlisted before tokenisation; “publicly downloadable” is not a licence. The stable phase
uses FineWeb-Edu and DCLM. FineMath and Stack-Edu are concentrated in the final 15% decay phase of
a warmup-stable-decay (WSD) schedule, inside the research-recommended final 10–20% window.[R1][R15]

| Phase | Tokens | Mixture | Whole-run contribution |
| --- | ---: | --- | --- |
| Stable | 127.5B (85%) | 60% FineWeb-Edu, 40% DCLM | 76.5B FineWeb-Edu; 51.0B DCLM |
| Decay | 22.5B (15%) | 25% each FineWeb-Edu, DCLM, FineMath 4+, and licence-allowlisted Stack-Edu | 5.625B from each source |
| Total | 150B | 54.75% FineWeb-Edu; 37.75% DCLM; 3.75% FineMath; 3.75% Stack-Edu | 82.125B; 56.625B; 5.625B; 5.625B respectively |

FineWeb-Edu and FineMath are ODC-By; DCLM is CC-BY-4.0.[R16][R17][R18] Stack-Edu has no single
dataset-wide licence, so a file is eligible only when its recorded upstream licence is on the
commercial allowlist. Missing, unknown, mixed, non-commercial, no-derivatives, or incompatible
copyleft rows are rejected rather than guessed.[R19]

Excluded from pretraining are:

- PersonaHub, EmpatheticDialogues, APIGen-MT-5k, and other non-commercial datasets;
- OpenHermes-2.5, Mixture-of-Thoughts, Magpie Ultra/Qwen, PersonaChat, LMSYS conversation data,
  and any source without a sufficiently clear training licence;
- Magpie-Llama and other Llama-derived outputs carrying Llama-specific terms;
- MobileLLM-R1 under the FAIR non-commercial research licence;
- Books3, BookCorpus, pirated books or code, private user documents and logs, secrets or PII;
- benchmark evaluation data and any shard with unresolved contamination; and
- teacher logits, pruned weights, or synthetic teacher output during owned-base pretraining.[R1][R5]

Post-training may use outputs from MIT- or Apache-2.0-licensed models only after provenance,
licence, contamination, privacy, and quality checks. This exception does not change the
from-scratch pretraining claim.

### 7. Spend governance and automatic stop

The all-in programme ceiling is approximately ₹24,000:[R1]

| Envelope | Maximum |
| --- | ---: |
| Measured v6e pilot | ₹1,000 |
| Main run | ₹20,000 |
| Contingency, including recovery and egress | ₹3,000 nominal; no more than ₹2,996.63 may be drawn |
| **Enforceable total** | **₹23,996.63 (approximately ₹24,000)** |
| Recorded credit left untouched | **At least ₹4,800** |

The ₹3.37 difference between the nominal envelopes and the enforceable total is intentionally
unavailable: ₹28,796.63 minus ₹23,996.63 leaves exactly ₹4,800.[R1]

Before any paid-account upgrade, the project must have a simulated and reviewed automatic billing
kill switch: Cloud Billing budget notifications publish to Pub/Sub; an idempotent authenticated
Cloud Run function consumes the event; it verifies project and threshold; then it
disables billing on the dedicated training project and emits a redacted structured audit event
with `correlation_id`.[R20][R21] Deployment and destructive test procedures require separate owner
authorisation.

Cloud Billing notifications are delayed and delivered at least once, and disabling billing is a
destructive emergency brake rather than an exact spending cap.[R20][R21] Therefore the run also
requires a local spend ledger, time and token ceilings, restricted quota, no unrelated project
resources, duplicate-event handling, and manual shutdown instructions. A budget alert alone is
not a kill switch. The controls must be installed and dry-run before the paid upgrade, not after
the first charge.

### 8. Execution, checkpoint, and quality gates

The production recipe is frozen before the paid pilot. It must include exact non-padding-token
accounting, deterministic shard cursors, atomic checkpoints containing model/optimiser/scheduler/
RNG/data-mixture state, structured JSON logs with `correlation_id`, and fail-fast checks for
non-finite losses, gradients, parameters, and optimiser state. Spot recovery must resume without
replaying charged tokens or skipping data.

Held-out BPB, perplexity, licence/provenance coverage, contamination, downstream capability,
safety, and conversion to the selected device runtime are measured at pre-registered checkpoints.
The final acceptance rule compares the owned base with size-appropriate open baselines using the
same tokenizer-independent evaluations. Finishing the token target is not success if held-out
quality is still unusable.

The fallback trigger is calendar-bound: if the required cloud quota is not granted by
2026-11-20, stop waiting and execute the separately approved 135M Mac plan. The M4 Max measurement
of approximately 2,452 tokens/s applies to the 361.8M full-training path at sequence length 167;
the research's estimated 15–17B tokens/month for 135M is extrapolated, not measured, and requires
its own pilot before a schedule is promised.[R1][R3]

### 9. Ownership-versus-quality checkpoint

After ADR-0013's personal memory system exists, run a paired comparison on the owner's held-out
documents, preferences, and tasks:

- the Aptivra-owned base with the same frozen memory snapshot, retrieval policy, context budget,
  prompt, adapters, tools, and evaluator; and
- Apache-2.0 Qwen3-0.6B with that identical memory system and evaluation protocol.[R7]

Report quality, calibration/deferral, latency, energy, memory, privacy failures, and provenance.
The owner then explicitly chooses whether release priority is independent ownership, measured
quality, or a documented hybrid. The comparison cannot silently replace the owned-base objective,
and “same memory” forbids giving either base a richer index or context.

## Owner decisions

These decisions are taken in order; a later decision cannot waive an earlier gate.

1. **Meaning of ownership.** Options: from-scratch independent pretraining; Minitron-style
   pruning; logit distillation. **Recommendation:** from scratch, with MIT/Apache model outputs
   allowed only as post-training data and no logits or parent weights.
2. **Model size and repository name.** Options: gated 360M v6e run under the 360M name; accurately
   renamed 135M Mac fallback. **Recommendation:** 360M primary and automatic 135M fallback if quota
   is absent by 2026-11-20 or the pilot fails.[R1]
3. **Billing upgrade.** Options: accept paid-account card exposure after controls pass; remain on
   the current account and stop. **Recommendation:** no upgrade until the automatic kill switch and
   runtime controls are installed, simulated, and reviewed.
4. **Accelerator, region, and framework.** Options: v6e Spot in `us-central1` with JAX/MaxText;
   v6e with PyTorch/XLA; measured L4 fallback for 135M; Mac-only 135M. **Recommendation:** v6e with
   JAX/MaxText for 360M, unless rehearsal proves PyTorch/XLA materially lowers end-to-end risk.
5. **Spend cap and pilot stop rule.** Options: authorise 150B; authorise a disclosed 120–150B
   reduction; select 135M; stop. **Recommendation:** enforce ₹23,996.63 overall and authorise 150B
   only when the measured upper-bound main cost is at most ₹20,000; otherwise select 135M rather
   than under-train 360M.[R1]
6. **Licence posture.** Options: commercial-safe allowlist; broader research-only sources.
   **Recommendation:** commercial-safe inputs only, with per-file Stack-Edu review and all listed
   exclusions applied fail-closed.
7. **Cloud escalation of personal queries.** Options: prohibited; consented redacted escalation.
   **Recommendation:** this is decided under ADR-0013, independently of cloud pretraining; no base-
   training approval permits personal data to leave the device.
8. **Release and post-memory product base.** Options: release no model; release the owned base and
   derived private artefacts; use Apache-2.0 Qwen3-0.6B; document a hybrid. **Recommendation:**
   release nothing until all model-card, licence, evaluation, and conversion gates pass, then make
   the product-base choice from the same-memory owner-data comparison while retaining the owned
   base as a separate research artefact.[R7]

Until the owner records these decisions and all relevant gates pass, this ADR remains Proposed and
no cloud execution is authorised.

## Consequences

- The plan buys roughly twenty times more tokens than the rejected 7.2B plan while remaining
  bounded by measured economics rather than projected accelerator marketing.[R1]
- Independent ownership is prioritised over the cheapest route to benchmark quality; that trade
  is explicit and reversible only by a new owner decision.
- v6e is the only evaluated device likely to reach 150B inside the cap, but its unmeasured
  throughput is a first-class risk rather than an assumption.
- The 135M fallback protects the project from quota and pilot failure, at the cost of a different
  capability ceiling and an accurately renamed repository.
- The paired Qwen checkpoint prevents ownership ideology from silently overriding product quality.

## Retrospective

The earlier plan optimised an exact 20N number before asking whether that number could produce a
useful small model. The correction is structural: quality derives the token target; current prices
identify a candidate device; and a measured, fail-closed pilot decides whether the target is
affordable. Licence clarity, spend shutdown, fallback timing, and the meaning of ownership are now
decisions rather than assumptions.

## Sources

- **[R1]** *Aptivra personal small model*, completed deep-research report and its notes,
  especially *pretraining budget and distillation*, *small model architecture and data*, and
  *on-device runtimes and hardware* (read 2026-10-18).
- **[R2]** Muennighoff et al., [Scaling Data-Constrained Language Models](https://arxiv.org/abs/2305.16264).
- **[R3]** Repository measurement recorded by the completed research on the Apple M4 Max; see the
  report's throughput and feasibility sections.
- **[R4]** NVIDIA, [Compact Language Models via Pruning and Knowledge Distillation](https://arxiv.org/abs/2407.14679).
- **[R5]** Meta, [Llama 3.1 Community License](https://www.llama.com/llama3_1/license/).
- **[R6]** Google, [Gemma Terms of Use](https://ai.google.dev/gemma/terms).
- **[R7]** Qwen, [Qwen3-0.6B model card and Apache-2.0 licence](https://huggingface.co/Qwen/Qwen3-0.6B).
- **[R8]** Hugging Face, [SmolLM2-360M configuration](https://huggingface.co/HuggingFaceTB/SmolLM2-360M/blob/main/config.json).
- **[R9]** Hoffmann et al., [Training Compute-Optimal Large Language Models](https://arxiv.org/abs/2203.15556).
- **[R10]** Google Cloud, [Spot VM pricing](https://cloud.google.com/spot-vms/pricing), checked
  2026-10-18.
- **[R11]** Google Cloud Billing,
  [Cloud Billing Catalog API](https://cloud.google.com/billing/docs/reference/rest/v1/services.skus/list),
  queried read-only on 2026-10-18.
- **[R12]** Google Cloud, [Train models on Cloud TPU v6e](https://docs.cloud.google.com/tpu/docs/v6e-training).
- **[R13]** Google, [MaxText](https://github.com/AI-Hypercomputer/maxtext).
- **[R14]** PyTorch, [PyTorch/XLA documentation](https://docs.pytorch.org/xla/master/).
- **[R15]** Hu et al., [MiniCPM: Unveiling the Potential of Small Language Models with Scalable Training Strategies](https://arxiv.org/abs/2404.06395).
- **[R16]** Hugging Face, [FineWeb-Edu dataset card](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu).
- **[R17]** MLCommons, [DCLM dataset card](https://huggingface.co/datasets/mlfoundations/dclm-baseline-1.0).
- **[R18]** Hugging Face, [FineMath dataset card](https://huggingface.co/datasets/HuggingFaceTB/finemath).
- **[R19]** Hugging Face, [Stack-Edu dataset card](https://huggingface.co/datasets/HuggingFaceTB/stack-edu).
- **[R20]** Google Cloud Billing,
  [Programmatic budget notifications](https://docs.cloud.google.com/billing/docs/how-to/budgets-programmatic-notifications).
- **[R21]** Google Cloud Billing,
  [Disable billing from notifications](https://docs.cloud.google.com/billing/docs/how-to/disable-billing-with-notifications).

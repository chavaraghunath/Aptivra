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

## Powered results and derived analysis

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

Among the eligible probed candidates, the evidence gives a **cautious preference** for
**HuggingFaceTB/SmolLM2-360M** at revision
`f8027fd0eaeea54caa13c31d31b9fdc459c38b49` over SmolLM2-135M. The preference is driven mostly
by T6 and T8 and costs 3,973.27 primary seconds versus 2,013.15 seconds—roughly twice the training
time. Amendment 2 leaves the ranking first place unchanged, but reaches it only after learned
accuracy and gain tie and the rule falls through to bits per byte.

This is evidence for a preference, not authority to adopt. Owner acceptance remains pending, so
the ADR stays **Proposed**.

## Decisions for the owner

1. Choose 360M or 135M: accept 360M's cautious quality preference and roughly twofold measured
   training time, or choose 135M for lower cost.
2. Keep exact code-point round-trip, or relax the contract to exact round-trip after NFC
   normalisation and probe Pythia-160M and Qwen2.5-0.5B before choosing a base model.

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

All five selectable candidates are Apache-2.0 or MIT. A derived Aptivra release must retain the
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
- The 360M base costs more memory and time than 135M; its primary peak was 20,584,579,072 bytes,
  still inside the configured headroom limit.
- The result has only twelve independent family units, T9--T12 are renamed-template tasks, several
  other families reuse training targets, and 360M shows more seed variance than 135M.
- Pythia-160M and Qwen2.5-0.5B have no learnability result under this protocol.
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

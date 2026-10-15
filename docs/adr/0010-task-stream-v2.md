# ADR-0010: Version the Synthetic Continual-Learning Stream and Adopt V2

- **Status:** Proposed
- **Date:** 2026-10-14

## Context

The configured v1 stream cannot support continual-learning capability claims. A powered
base-model sweep found that T9--T12 repeat two or three programs under renamed identifiers and
reach 100% after 200 steps for every candidate, including GPT-2. T5 has two targets, T7 has four,
and T1, T6, and T8 reuse many training targets in test. T2 remains at 0--1% for every candidate
and is plausibly dominated by exact character editing under subword tokenization.

String-level split isolation did not detect these failures. V1 considered a renamed function a
new instance even when its normalized program and behavior were unchanged. Existing experiment
records depend on the exact v1 bytes, so correcting v1 in place would destroy reproducibility.

A follow-up prompt-shape audit found that the first v2 code test measured parameter
generalisation inside known templates, not unseen-template code synthesis. After replacing
generated function names, numbers, and quoted literals, T8--T12 had no test-only template except
one T11 shape. This is a valid in-template measure, but it cannot support a broader claim that the
model writes code for previously unseen program shapes.

## Decision

Introduce an explicitly selected `v2` alongside byte-stable `v1`. Omitting
`task_stream.version` selects v1; `configs/task_stream_v2.yaml` selects v2. Both retain the same
T1--T12 identifiers, order, split sizes, seed derivation, exact-match contracts, task-aware stop
rules, and per-cycle evaluation schedule. Run metadata records the selected stream version,
family roles, and headline-eligible families.
For v2, `task_stream.roles`, `task_stream.family_roles`, and `task_stream.headline_roles` are the
validated source of those classifications. An undeclared role, unknown or missing family ID, or a
configuration selecting no headline family halts loading.

V2 makes these changes:

1. Every family rejects a validation or test target already present in an earlier split. T1 uses
   large modular answer spaces; T4 emits mixed-bracket completions; T5 emits 32-bit truth tables;
   T6 emits three-value recurrence continuations; and T7 emits complete DFA traces.
2. T8--T12 additionally require distinct normalized Python AST hashes within and across the
   original train, validation, and in-template test splits.
   The normalizer erases function, argument, and local identifier spelling while retaining
   control flow, operators, literals, annotations, and statement structure. Invalid Python halts
   construction. T8 uses five executable snippet grammars; T9--T11 use eight parameterized
   algorithm or fault classes; T10 uses eight string-program classes; and T12 uses eight scalar,
   collection, text, and Boolean typing patterns. Their combinatorial parameters are executable
   behavior, not renamed identifiers.
3. Every v2 prompt carries a complete answer contract. Direct-computation families state their
   operator, ordering, trace, or serialization semantics. T6 names the bounded simple-rule library
   and emits an item only when that library yields one continuation. T9--T11 show two examples and
   retain additional held-out cases; T11 also states the intended behavior separately from the
   buggy source, and both shown examples discriminate the bug. T12 supplies two typed examples
   from which one most-specific PEP 585 signature is inferable and requires the body to be
   preserved. T8--T11 verifiers accept any program that passes the prompt-visible and held-out
   executions; they do not require the target source string.
4. Each code family has a second test cell named `template-OOD`, configurable with
   `task_stream.num_template_ood_test_per_code_family` and defaulting to 300 items. Two new,
   executable program shapes per family are reserved exclusively for this cell; their normalized
   prompt templates cannot occur in train, validation, or the in-template test. Exact target and
   normalized-AST isolation, determinacy execution, contamination, trivial-baseline, and byte
   budget checks apply independently. The harness records a separate template-OOD matrix and
   never pools it into the in-template matrix or headline metrics.
5. T2 keeps the exact-character reverse/RLE/Caesar workload. There is no model-free evidence that
   redesigning it would preserve the character-editing skill while making it learnable to a
   subword model, and this task forbids model training. It is therefore labelled
   `tokenizer_sensitivity_diagnostic`, remains measured in every accuracy-matrix cycle, and is
   excluded from v2 average accuracy, BWT, FWT, and aggregate forgetting. Per-family T2 results
   remain reportable and must not be described as a learned skill.
6. Generated-code verification continues through the resource-limited child-process runner.
   Audit checks are batched within that same containment boundary to avoid process-startup cost.
   T12 compares parsed ASTs, accepting formatting differences while enforcing the uniquely evidenced
   annotations and unchanged body. No verifier accepts approximate prose.

## Audit and acceptance evidence

`aptivra.eval.task_stream_audit` and `scripts/audit_task_stream.py` materialize configured streams
and record deterministic JSON under `experiments/`. For each family and split they report prompt
and target cardinality, UTF-8 target-length distribution, uniform-target chance, and empirical
most-frequent rate. They also report exact prompt/target train-test overlap, normalized-AST
train-test overlap for T8--T12, exact matches against identity, conversational, and canonical chat
training strings, and three train-only baselines: most-frequent answer, prompt copy, and nearest
training prompt by character-trigram Jaccard similarity.
Every family additionally reports normalized template count and top-template share per split plus
the number of test-only templates. Code families report the `template-OOD` cell separately,
including OOD-only template count and in-template overlap. Any configured template-OOD overlap is
an unconditional audit failure.

The audit also performs a family-specific determinacy gate over every generated item. Direct
families independently reconstruct the answer and exact prompt contract. T6 enumerates arithmetic,
integer-geometric, constant-second-difference quadratic, and bounded second-order recurrence rules
and counts items with multiple continuations. T8 executes the target and a naive snippet. T9--T11
execute the target over shown plus held-out cases and execute the naive or buggy program against
each shown example. T12 derives the signature from runtime example types, checks the AST
annotations, and executes the preserved body. Ambiguous and invalid counts are separate in the
record; their sum is enforced against `max_ambiguous_item_count` per family.

The checked-in v1 audit records the failure rather than laundering it: target overlap is 67 for
T1, 107 for T6, and 14 for T8; T5/T7 uniform chance is 0.50/0.25; and structural overlap is 3, 2,
2, and 2 for T9--T12. Its nearest-neighbour baseline reaches 0.64 on T5 and 0.40 on T8 in the
configured deterministic 25-item sample.

The v2 audit records 300 distinct in-template test targets for every family, uniform chance
`1/300`, zero
exact train-test target overlap, zero normalized-AST overlap for every code family, and zero
identity/chat contamination. Each code family has 500/100/300 distinct train/validation/test AST
hashes. Every most-frequent and nearest-neighbour baseline scores 0; prompt copy scores 0 after
the repaired bug corpus is regenerated. The configured v2 gate allows at most 0.05 uniform chance,
0.10 trivial-baseline accuracy, and zero target, structural, identity/chat, ambiguous, or invalid
items. All 900 in-template items in each v2 family pass the determinacy gate. The v1 record marks all 900 items
per family as lacking the new machine-checkable prompt contract and permits that frozen legacy
defect only in the v1-specific policy; v1 remains ineligible for new claims.

The new v2 template-OOD audit records 300 distinct targets and 300 distinct AST hashes for each
code family, zero exact-target, structural, or normalized-template overlap with in-template data,
and 300/300 determinate items per family. Each family contributes two OOD-only templates. Template
counts and top-template shares are:

| family | train | val | in-template test | test-only | template-OOD | top share train/val/test/OOD |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| T8 | 5 | 5 | 5 | 0 | 2 | 0.208 / 0.270 / 0.240 / 0.500 |
| T9 | 13 | 13 | 12 | 0 | 2 | 0.148 / 0.200 / 0.173 / 0.523 |
| T10 | 8 | 8 | 8 | 0 | 2 | 0.158 / 0.210 / 0.170 / 0.500 |
| T11 | 9 | 9 | 10 | 1 | 2 | 0.140 / 0.170 / 0.157 / 0.520 |
| T12 | 10 | 9 | 10 | 0 | 2 | 0.154 / 0.210 / 0.147 / 0.517 |

The prompt-template normalizer replaces generated `f_<seed>_<index>` names, signed numeric
literals, and non-docstring quoted literals while retaining list shape, operators, and
prose. These counts therefore describe prompt shapes under that declared normalizer, not a proof
of semantic distance.

The v1 byte snapshot remains `81c88937debbc55c6e34e9ce42d89749301d0d2ea37829e514cbabcc740b4a19`
(3,894,503 bytes). The original v2 in-template snapshot is unchanged before and after this
decision: `f3a64ba884ef2c7b42f1b6b3218b9e8ad694eab0eac28e28b5c6dd67d5adf2b5`
(9,674,700 bytes). Template-OOD records are deliberately excluded from both snapshots.

`tests/test_task_stream_audit.py` recomputes v1 and v2 at configured split sizes, compares them
byte-for-byte with the recorded JSON, and enforces the version-specific thresholds. V1 thresholds
freeze its known baseline; v2 thresholds are claim-readiness gates. Any regression fails pytest.

## Alternatives considered

1. **Edit v1 in place.** Rejected because historical records would no longer identify their
   generating stream.
2. **Delete T2.** Rejected because the persistent failure is useful tokenizer diagnostic evidence;
   deletion would hide rather than explain it.
3. **Claim a learnable T2 redesign without training.** Rejected because learnability is an
   empirical model claim and this change deliberately performs no model training.
4. **Use string hashes for code isolation.** Rejected because identifier renaming already proved
   that byte-level uniqueness is not program diversity.
5. **Chance-correct the binary families.** Rejected for v2 because a broad answer space produces a
   more direct exact-match measure; the audit still reports chance explicitly.

## Consequences

### Positive

- Historical v1 experiments remain reproducible and visibly non-claim-capable.
- V2 test answers and program structures cannot be recalled from its training split.
- In-template parameter generalisation and held-out-template generalisation are reported as
  distinct cells, preventing one from being described as the other.
- Trivial train-only predictors and contamination are measured before any expensive model run.
- Family roles and headline inclusion can be reviewed or changed in configuration without a code
  edit; invalid role graphs fail before stream generation.

### Negative and limitations

- AST identity is a syntactic guarantee, not proof that two different ASTs have unrelated
  semantics. The executable algorithm classes and hidden cases provide an additional behavioral
  check, but only a future preregistered model experiment can establish learning difficulty.
- Two OOD templates per family broaden the measurement but remain a small synthetic template
  library. A high template-OOD score supports only performance on these declared shapes, not
  unrestricted program synthesis.
- The 25-item baseline sample is deterministic and inexpensive, not a confidence interval.
- The simple-rule library bounds what “unambiguous” means for T6; it does not claim that a finite
  sequence has a unique continuation over all imaginable mathematical rules.
- V2 invalidates comparisons to v1 accuracy levels; reports must name the stream version.

## Rollback

Select `configs/task_stream.yaml` or set `task_stream.version: v1`. This restores historical stream
generation and all-family headline aggregation without deleting v2 evidence. Recorded results must
retain their original version and must never be relabelled during rollback.

## Retrospective

The original isolation invariant tested records, not skills. The first v2 audit repeated that
mistake twice in subtler forms: cardinality, overlap, and chance could pass while prompts failed
to determine labels, and AST isolation could pass while train and test reused the same prompt
templates. Human inspection exposed both gaps. The prevention is now layered:
prompt-visible specifications and examples, held-out behavioral verification for non-unique code,
per-family determinacy checks, explicit template metrics, a separately materialized template-OOD
cell with a zero-overlap assertion, an ambiguity threshold, and a mandatory five-item human review
per family. Batching the exhaustive code checks retains child-process containment.

## Evidence appendix: human-read test samples

The following are the first five materialized test items per family for seed 42 after split
isolation. Each line records the task-ID suffix, decisive prompt content, target, and the author's
manual judgement; combine the suffix with the family ID in its heading to reproduce the full task
ID. This review supplements, rather than substitutes for, the executable audit.

### T1 modular arithmetic

- `...s42003_i0` — `(61663 * 207083 + 40) mod 99581` → `87439` — Determinate: operands, operator, offset, and modulus are explicit.
- `...s42003_i1` — integer power `(480223 ** 2) mod 37447` → `671` — Determinate: `**` and “integer power” remove XOR ambiguity.
- `...s42003_i2` — `(89543 * 554457 + 84) mod 82633` → `18909` — Determinate by direct modular evaluation.
- `...s42003_i3` — `(53381 * 427410 + 34) mod 73813` → `48757` — Determinate by direct modular evaluation.
- `...s42003_i4` — `(260293 + 165577) mod 60068` → `5394` — Determinate by direct modular evaluation.

### T2 string transformations

- `...s42013_i0` — Caesar-shift `hnstzn` forward 8 with wrap → `pvabhv` — Determinate: direction, alphabet, and wrap are stated.
- `...s42013_i1` — Caesar-shift `koykgxo` forward 7 with wrap → `rvfrnev` — Determinate under the same explicit contract.
- `...s42013_i2` — Caesar-shift `slajlwendr` forward 8 with wrap → `atirtemvlz` — Determinate under the same explicit contract.
- `...s42013_i3` — Caesar-shift `jivdturllyw` forward 6 with wrap → `pobjzaxrrec` — Determinate under the same explicit contract.
- `...s42013_i4` — reverse `ujabdjogpghm` exactly → `mhgpgojdbaju` — Determinate: output-only exact reversal is stated.

### T3 list operations

- `...s42023_i0` — descending sort of `[33,-39,-44,70,-18,80,37,76,-25,42,57,4]` → `80 76 70 57 42 37 33 4 -18 -25 -39 -44` — Determinate: ordering and space-only serialization are explicit.
- `...s42023_i1` — ascending sort of `[50,51,23,95,-40,24,20]` → `-40 20 23 24 50 51 95` — Determinate under the stated serialization.
- `...s42023_i2` — stable even filter of `[96,-8,54,79,-45,49,0,-4,28,51,66]` → `96 -8 54 0 -4 28 66` — Determinate: original order is explicit.
- `...s42023_i3` — ascending sort of `[-29,-28,22,74,17,-4,88,59]` → `-29 -28 -4 17 22 59 74 88` — Determinate under the stated serialization.
- `...s42023_i4` — ascending sort of `[-18,37,6,-16,-6,-44,-3]` → `-44 -18 -16 -6 -3 6 37` — Determinate under the stated serialization.

### T4 syntax matching

- `...s42033_i0` — append the proper-nesting suffix for `{(({(([(` → `)]))}))}` — Determinate by reversing the opening stack.
- `...s42033_i1` — append the suffix for `([[({{(({[[` → `]]}))}})]])` — Determinate by reversing the opening stack.
- `...s42033_i2` — append the suffix for `{{({[[{(([` → `]))}]]})}}` — Determinate by reversing the opening stack.
- `...s42033_i3` — append the suffix for `{{({{{([[{{[{` → `}]}}]])}}})}}` — Determinate by reversing the opening stack.
- `...s42033_i4` — append the suffix for `{[[[[[({[{[{[[{` → `}]]}]}]})]]]]]}` — Determinate by reversing the opening stack.

### T5 Boolean truth tables

- `...s42043_i0` — fully parenthesized expression beginning `((((~E)&D)|(B|D))|D)...` → `01111011111111110111101111111111` — Determinate: the operator legend, row order, and bit encoding are explicit.
- `...s42043_i2` — `(((~B)^C)^(A&D))&(A|E)` → `01010000000001011100001100111100` — Determinate under explicit parentheses, legend, and row order.
- `...s42043_i7` — `(((~B)&B)^(A&D))^...` → `01010101010101011100011011000110` — Determinate under explicit parentheses, legend, and row order.
- `...s42043_i8` — nested `D/C/B/E/A` expression → `00010000000100000011001100110000` — Determinate under explicit parentheses and row order.
- `...s42043_i9` — `A|((((~B)^C)^(A|C))&...)` → `00010100000000001111111111111111` — Determinate under explicit parentheses, legend, and row order.

### T6 sequence completion

- `...s42053_i0` — `-269,-266,-263,-260,-257,-254,-251` → `-248,-245,-242` — Determinate: the bounded rule library yields only the `+3` continuation.
- `...s42053_i1` — `-442,-430,-418,-406,-394,-382,-370` → `-358,-346,-334` — Determinate: only the `+12` continuation fits the library.
- `...s42053_i2` — `-26,24,110,232,390,584,814` → `1080,1382,1720` — Determinate: one nonzero constant-second-difference continuation fits.
- `...s42053_i4` — `82,88,106,136,178,232,298` → `376,466,568` — Determinate: one nonzero constant-second-difference continuation fits.
- `...s42053_i5` — `-83,-72,-61,-50,-39,-28,-17` → `-6,5,16` — Determinate: only the `+11` continuation fits the library.

### T7 state machines

- `...s42063_i0` — eight-state table, start `S0`, input `10110100011` → `S0->S4->S7->S0->S4->S7->S0->S3->S1->S0->S4->S2` — Determinate: table, input, inclusion of start, and delimiter are explicit.
- `...s42063_i1` — start `S1`, input `0001101111` → `S1->S5->S1->S5->S3->S3->S2->S7->S0->S6->S2` — Determinate by the displayed transition table.
- `...s42063_i2` — start `S3`, input `01111010` → `S3->S3->S4->S3->S4->S3->S3->S4->S3` — Determinate by the displayed transition table.
- `...s42063_i3` — start `S3`, input `0010111` → `S3->S4->S6->S2->S1->S7->S7->S7` — Determinate by the displayed transition table.
- `...s42063_i4` — start `S4`, input `101110111` → `S4->S1->S2->S2->S2->S2->S2->S2->S2->S2` — Determinate by the displayed transition table.

### T8 Python micro-synthesis

- `...s42073_i0` — set `res` to the count in `1..246` congruent to `14 mod 15` → the executable comprehension over `range(1,247)` — Determinate; target executes to the independently computed result and `res=None` fails.
- `...s42073_i1` — count in `1..405` congruent to `3 mod 5` → the executable comprehension over `range(1,406)` — Determinate by execution.
- `...s42073_i2` — set `res=max-min` for `[36,-29,3,50,-65,24,-13]` → `res = max(...) - min(...)` — Determinate by execution.
- `...s42073_i3` — weighted sum of `[33,-21,-9,45,30]` with weight `i+18` → executable indexed sum — Determinate by execution.
- `...s42073_i4` — sum squares of strictly positive values in `[21,-10,18,6,1,-9,-10]` → filtered executable sum — Determinate by execution.

### T9 algorithm synthesis

- `...s42083_i0` — clamp to `[-9,9]`, add `736`; examples `[0]→[736]`, `[1]→[737]` — Determinate: target passes shown and held-out cases; identity-list naive fails both shown cases.
- `...s42083_i1` — sorted squares plus `842` below `0`; examples `[0]→[]`, `[1]→[]` — Determinate from the exact predicate plus held-out negative inputs.
- `...s42083_i2` — alternating signed values, scale `5`, offset `207`; examples `[0]→[207]`, `[1]→[212]` — Determinate by specification plus shown and held-out execution.
- `...s42083_i3` — clamp to `[-11,11]`, add `456`; examples `[0]→[456]`, `[1]→[457]` — Determinate by shown plus held-out execution.
- `...s42083_i4` — alternating signed values, scale `13`, offset `961`; examples `[0]→[961]`, `[1]→[974]` — Determinate by specification plus shown and held-out execution.

### T10 docstring-to-code

- `...s42093_i0` — rotate left by `47`; `'a b'→'ba '`, `'a b c'→'b ca '` — Determinate: precise docstring plus examples; target passes longer held-out strings.
- `...s42093_i1` — replace vowels with `f` in first `550`; `'a b'→'f b'`, `'a b c'→'f b c'` — Determinate by visible and held-out execution.
- `...s42093_i2` — reverse and return at most first `453` words; `'a b'→'b a'`, `'a b c'→'c b a'` — Determinate; a longer-than-limit held-out case rejects reversing the entire input.
- `...s42093_i3` — every `56`th character from index `24`; both short examples return empty — Determinate from the exact slice contract; a long held-out string rejects constant-empty implementations.
- `...s42093_i6` — rotate left by `88`; `'a b'→' ba'`, `'a b c'→' ca b'` — Determinate by visible and held-out execution.

### T11 bug fixing

- `...s42103_i0` — spec explicitly states even `i:2*x+789`, odd `i:789-x`; examples `[1]→[791]`, `[-1]→[787]`; buggy code flips parity — Determinate: both visible examples fail the bug and pass the repair.
- `...s42103_i1` — sum values congruent to `9 mod 13`, scaled by `2`; buggy predicate is `!=` — Determinate: both visible examples discriminate and held-out cases verify generality.
- `...s42103_i2` — count values with `11*x>840`; buggy predicate is `<=` — Determinate: both visible examples discriminate and held-out cases verify the boundary.
- `...s42103_i3` — adjacent differences start at index `1`; buggy range starts at `2` — Determinate: both visible examples expose the missing first difference.
- `...s42103_i4` — sum values congruent to `13 mod 18`, scaled by `10`; buggy predicate is `!=` — Determinate: both visible examples discriminate and held-out cases verify matches.

### T12 type inference

- `...s42113_i0` — examples `[1,-2]→[78861,78858]` and `[]→[]`, body adds an integer — target `list[int] -> list[int]` — Determinate: first example fixes element and result types.
- `...s42113_i1` — examples `[1,-2]→[66757,66754]` and `[]→[]` — target `list[int] -> list[int]` — Determinate from runtime types plus body.
- `...s42113_i2` — examples `['a','b']→('a','b')` and `[]→()` — target `list[str] -> tuple[str, ...]` — Determinate from the non-empty example.
- `...s42113_i3` — examples `[1,2]→95506` and `[]→95504` — target `list[int] -> int` — Determinate from runtime types plus `len` body.
- `...s42113_i4` — examples `[1,-2]→[23857,23854]` and `[]→[]` — target `list[int] -> list[int]` — Determinate from runtime types plus body.

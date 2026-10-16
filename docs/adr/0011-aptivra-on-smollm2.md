# ADR-0011: Aptivra Continual Learning on SmolLM2-360M

- **Status:** Accepted — approved by repository owner raghunath on 2026-10-16
- **Date:** 2026-10-16
- **Deciders:** Repository owner, Aptivra lead research engineer
- **Supersedes for the rebuilt model:** ADR-0002, ADR-0003, and the model-execution and
  export applicability of ADR-0006
- **Depends on:** ADR-0005, ADR-0008, proposed ADR-0009, and proposed ADR-0010

## Context

[ADR-0008](0008-adopt-pretrained-base-model.md) accepted
`HuggingFaceTB/SmolLM2-360M` at immutable revision
`f8027fd0eaeea54caa13c31d31b9fdc459c38b49`, with its `GPT2Tokenizer`, as the Tier 3 base.
That decision was conditional on full-fp32 fine-tuning on the project's only training machine, a
36 GiB Apple M4 Max, and fixed the usable peak at 28,991,029,248 bytes. It did not decide how the
pretrained model participates in Aptivra's continual-learning system.

The current Aptivra runtime cannot simply receive different weights:

- `aptivra.model` is a 256-token byte-level transformer with architecture and state names that do
  not match SmolLM2's stock Llama implementation.
- The current expert path is not a routed layer. `ConsolidationCycle` computes its training loss
  through the dense backbone only, never through `router` or `experts`; it evaluates router entropy
  on random `dummy_x` and supplies a verifier-failure rate of zero. The exported inference wrapper
  can add one post-backbone adapter mixture, but that is not the trained architecture claimed by
  ADR-0003.
- Byte ordinals are treated as token IDs throughout generation, stopping, padding, replay, and
  export. Those assumptions are invalid for a subword tokenizer even when a particular character
  happens to encode to one token.
- The current identity corpus states a 200M Top-2 MoE, zero catastrophic forgetting, thousands of
  skills, a throughput number, and a working Tiered Memory Runtime C engine. Those claims are not
  established for the rebuilt model and some are not established for the historical model.
- ADR-0006 says the exporter emits `.safetensors` and that a pure-C engine executes the model.
  The implementation emits PyTorch `.pt` expert shards; the C programs parse headers and print
  statistics but do not tokenize, load model tensors, run transformer/router layers, produce
  logits, or generate. There is no C/Python logit-equivalence test. This is a stop-the-line defect,
  not an integration starting point.

The design must preserve the historical v0-byte-level artifacts without laundering them into the
new architecture, make the expert mechanism real, fit on the sole machine, and leave a clean
experiment capable of falsifying the continual-learning hypothesis.

## Local evidence

The arithmetic in [`0011-evidence/smollm2-arithmetic.json`](0011-evidence/smollm2-arithmetic.json)
is reproduced by
[`0011-evidence/compute_smollm2_arithmetic.py`](0011-evidence/compute_smollm2_arithmetic.py).
It reads the already-cached immutable snapshot offline and validates the following facts:

| Fact | Value |
| --- | ---: |
| Architecture | stock `LlamaForCausalLM` |
| Hidden / FFN size | 960 / 2,560 |
| Layers / query heads / KV heads | 32 / 15 / 5 |
| Vocabulary / maximum positions | 49,152 / 8,192 |
| Exact tied-weight parameters | 361,821,120 |
| Cached weight dtype / bytes | BF16 / 723,674,912-byte file |
| FP32 base weights | 1,447,284,480 bytes |
| FP32 weights + gradients + two AdamW moments | 5,789,137,920 bytes |
| Config SHA-256 | `34f7801487078de7e434e19162c497e5cc6ff397080e40e8586627cb68a5168a` |
| Weight SHA-256 | `7aaff6661428bed033abba9522bec81938678642cca3181fe752b6ca9e1e540f` |

The accepted selection run measured a 18,273,697,792-byte hard-constraint peak for the full-fp32
SmolLM2-360M candidate at micro-batch 8 without checkpointing in
`experiments/base_model_candidates/amendment-4c/candidates/smollm2-360m-3156f262fc.json`.
That is evidence about the ADR-0008 probe, not a prediction of this architecture's peak.

The cached tokenizer has EOS/BOS/UNK `<|endoftext|>` at ID 0, no pad token, and no chat template.
The cache contains an explicit negative lookup for `chat_template.jinja`. Although IDs 1 and 2
are the added special tokens `<|im_start|>` and `<|im_end|>`, the selected base assigns them no
conversation semantics. Offline diagnostics encode LF as `[198]` and NUL as `[190]`; these are
observations, not values that runtime code may hard-code.

## Relationship to ADR-0008's selection premise

ADR-0008 selected the base under full fp32 fine-tuning, while Aptivra full now freezes the base
and trains rank-16 LoRA experts. That change does make every candidate look affordable when only
Aptivra is counted. It does **not** change the H3 selection constraint: naive sequential full
fine-tuning, replay full fine-tuning, and online EWC are claim-bearing comparators. Running those
methods on a smaller base than Aptivra, or on different hardware, would confound architecture with
base capacity or machine behavior. Every method in the main comparison therefore uses the same
base, tokenizer, device, fp32 contract, examples, and budget.

The offline candidate arithmetic is generated by
[`0011-evidence/compute_candidate_matrix.py`](0011-evidence/compute_candidate_matrix.py) in
[`0011-evidence/candidate-matrix-arithmetic.json`](0011-evidence/candidate-matrix-arithmetic.json).
For base parameters `P` and one rank-16 expert with `L` parameters, frozen-base Aptivra starts at
`4P + 16L` bytes: fp32 base weights plus LoRA weights, gradients, and two AdamW moments. The
twelve-expert design with one trainable expert is `4P + 60L`. Full-fp32 AdamW is `16P`; replay adds
at most `32,768,000` bytes for the optional cached IDs and labels; online EWC adds an fp32 Fisher
and anchor, `8P`, for `24P` static bytes before activations.

The table uses decimal GB. **M** is ADR-0008's measured hard-constraint peak. **A** is arithmetic
added to that measured peak. **S** is a config-derived static lower bound because ADR-0008 has no
completed peak; it omits activations and runtime allocations, so a sub-cap S value is not a fit
claim. An S value above the cap is a definitive failure because omitted allocations are
non-negative.

| Candidate | Aptivra: one expert / 12 resident (S) | Naive full FT | Replay full FT | Online EWC | Every comparator fits? |
| --- | ---: | ---: | ---: | ---: | --- |
| GPT-2 | 0.536 / 0.639 GB | 8.553 GB M | 8.586 GB A | 9.549 GB A | Yes, arithmetic projection |
| SmolLM2-135M | 0.616 / 0.831 GB | 9.186 GB M | 9.219 GB A | 10.263 GB A | Yes, arithmetic projection |
| Pythia-160M | 0.687 / 0.791 GB | 8.539 GB M | 8.572 GB A | 9.837 GB A | Yes, arithmetic projection |
| SmolLM2-360M | 1.586 / 1.968 GB | 18.274 GB M | 18.306 GB A | 21.168 GB A | Yes, arithmetic projection |
| Qwen2.5-0.5B | 2.117 / 2.504 GB | 37.610 GB M | 37.643 GB A | 41.563 GB A | **No**; naive already exceeds cap |
| Qwen2.5-1.5B | 6.470 / 7.283 GB | >=24.699 GB S; unmeasured | >=24.732 GB S; unmeasured | >=37.049 GB S | **No**; EWC lower bound exceeds cap |
| SmolLM2-1.7B | 7.135 / 7.931 GB | >=27.382 GB S; unmeasured | >=27.415 GB S; unmeasured | >=41.073 GB S | **No**; EWC lower bound exceeds cap |

Thus all seven bases fit the **static** frozen-base LoRA design, including a twelve-expert bank,
but only GPT-2, SmolLM2-135M, Pythia-160M, and SmolLM2-360M support the complete method-comparison
matrix under the available evidence and 28,991,029,248-byte cap. The first three are not newly
eligible alternatives: ADR-0008 already measured and ranked them below SmolLM2-360M. None of the
three footprint-excluded candidates becomes eligible for the full matrix. SmolLM2-360M therefore
remains the correct H3 base; this arithmetic does not justify reopening ADR-0008. The projected
comparator peaks remain subject to ADR-0011's measured calibration gate before H3 starts.

Residual risk remains: Aptivra's quality is bounded by the representations of its frozen 360M
base. A larger frozen base could outperform it in an Aptivra-only deployment even though that base
cannot support the controlled full-fine-tuning comparison matrix on this machine. Evaluating a
larger-base, Aptivra-only deployment is future work requiring a separate ADR with its own selection
rule, hardware budget, baselines, and claims; it is outside H3.

## Decision

### 1. Backbone integration

The rebuilt path loads the exact base and tokenizer with stock Transformers, an explicit revision,
`local_files_only`/offline mode when required, and `trust_remote_code=False`. `AutoConfig` and
`AutoModelForCausalLM` are authoritative for architecture. Before use, the loader verifies the
model ID, revision, config/weight hashes, tokenizer asset manifest, architecture, special-token
contract, and tied-weight parameter count.
It fails rather than accepting `main`, a floating revision, an incompatible adapter, or silent
embedding resize.

The continual-learning path does not subclass or copy `LlamaForCausalLM`. It attaches standard
PEFT LoRA adapters, keeps the base object's forward/generation semantics, and places Aptivra's
router and promotion orchestration outside the Transformers architecture. A stock base plus one
explicitly selected adapter remains independently loadable without Aptivra code. The complete
multi-expert routing policy requires the installed Aptivra orchestrator and is never misrepresented
as behavior of `AutoModelForCausalLM` alone.

#### Complete module disposition

“Retire” means excluded from every SmolLM2 loader, config, evaluator, exporter, and package
allowlist. It does not mean delete: the file remains available to reproduce and roll back v0.
Package `__init__.py` files follow the disposition of the symbols they export. Every current
Python module under `aptivra/` is covered by an exact path or glob below.

| Current module(s) | Disposition | Reason / required replacement |
| --- | --- | --- |
| `aptivra/__init__.py` | Survive | Package/version identity is architecture-neutral; its narrative must not claim the v0 architecture. |
| `aptivra/model/{attention,config,embeddings,transformer}.py`, `aptivra/model/__init__.py` | Retire | The scratch 256-vocabulary transformer is not the accepted base. Stock Llama config and layers are authoritative. |
| `aptivra/huggingface/{configuration_aptivra,modeling_aptivra,tokenization_aptivra}.py`, `aptivra/huggingface/__init__.py` | Retire | Custom auto classes and byte tokenizer require remote code and describe the wrong model. |
| `aptivra/baselines/{base,sequential,replay,ewc,joint}.py`, `aptivra/baselines/__init__.py` | Re-implement | Preserve the experimental methods, but rebuild loading, collation, target-only loss, attention masks, accumulation, measurement, and generation on the pinned base/tokenizer. |
| `aptivra/experts/{expert,router,spawner}.py`, `aptivra/experts/__init__.py` | Re-implement | Replace unattached adapters, random entropy, and zero verifier input with PEFT experts in the real forward path, evidence-based growth, and the router defined below. |
| `aptivra/consolidation/cycle.py`, `aptivra/consolidation/__init__.py` | Re-implement | Keep transactional promotion/rollback intent and complete RNG capture; rebuild state around base reference, adapter bank, router, optimizer, replay, and evidence manifests. |
| `aptivra/consolidation/sleep_cycle.py` | Retire | Global magnitude pruning of a pretrained base is destructive and unevidenced. “Sleep” becomes the bounded replay/promotion phase, not synaptic pruning. |
| `aptivra/memory/store.py` | Re-implement | Preserve the verifier-gated canonical record schema, but replace UUID/random retrieval and mutable access counts in the claim-bearing path with content-addressed IDs, deterministic sampling, and separately audited index statistics. Terminators remain text, not token IDs. |
| `aptivra/memory/retrieval.py` | Survive, out of main path | It indexes text independently of the LM tokenizer, but its weak heuristic and mutable scoring are neither deterministic replay nor semantic proof. It cannot enter the claim-bearing path until re-implemented and tested deterministically. |
| `aptivra/memory/hybrid_store.py` | Re-implement | Store canonical text, task/stream identity, hashes, and tokenizer/base fingerprint. Never replay v0 byte tensors; tokenization is regenerated lazily. |
| `aptivra/memory/{hexa_store,semantic_graph,somatic_valence,working_memory}.py`, `aptivra/memory/__init__.py` | Survive, out of path | These utilities are not the SmolLM2 KV cache or evidence for the main continual-learning claim. |
| `aptivra/eval/{config,metrics,task_stream_audit,artifact_store}.py` | Survive | Their validation, CL formulas, stream audit, and local artifact contracts remain useful. Local artifacts are mandatory for this machine-only experiment. |
| `aptivra/eval/harness.py`, `aptivra/eval/__init__.py` | Re-implement boundary | Preserve chronological matrices, verifier failure semantics, headline-role filtering, and separate template-OOD cells; replace byte stop IDs, rename the “random” baseline to frozen-base baseline, and add method/compute provenance. |
| `aptivra/eval/base_model_evaluation.py` | Survive as frozen ADR-0008 evidence tooling | Reuse its correct subword BPB primitives where extracted without changing immutable selection records; it is not the new trainer. |
| `aptivra/eval/{model_card_eval,model_card_schema}.py` | Re-implement | New schema must describe RoPE/Llama, base revision, PEFT adapters, router, seeds, stream, license, and aggregate subword BPB. V0 records remain on their old schema. |
| `aptivra/eval/vertex_base_sweep.py` | Retire | Model selection is complete and cloud training is unavailable; preserve it only with ADR-0008 history. |
| `aptivra/eval/model_card_heldout_v1.json`, `aptivra/eval/model_card_heldout_text_v1.txt` | Retire from the new gate, preserve as v0 | Their hashes and v0 claims remain immutable. Implementation creates versioned v2 evidence after contamination review. |
| `aptivra/gate/{promotion_gate,stats}.py`, `aptivra/gate/__init__.py` | Re-implement | Preserve the promote/rollback boundary, but replace the current `p >= alpha` “no regression” inference and automatic pass below five samples with a pre-registered paired non-inferiority/effect rule that fails closed on insufficient data or infrastructure failure. |
| `aptivra/versioning/checkpoint_manager.py`, `aptivra/versioning/__init__.py` | Re-implement | Atomically reference the immutable base and save adapter/router/optimizer/RNG/config/data/evidence hashes; verify all hashes before activation. |
| `aptivra/inference/{causal_lm,chat_prompt,chat_finetune,chat_canonical,chat_quality}.py`, `aptivra/inference/__init__.py` | Re-implement | Load stock model plus PEFT, apply versioned formatting/stopping, train identity through an adapter, and verify only evidence-bound claims. |
| `aptivra/inference/{tokenizer,hub_config,chat_sanitize}.py` | Retire | The byte tokenizer and v0 manifest are invalid. Output substitution can fabricate a passing identity response and is forbidden; only transparent delimiter removal survives in new code. |
| `aptivra/export/{hf_hub_artifacts,hf_push,model_card}.py`, `aptivra/export/__init__.py` | Re-implement | Keep isolated staging, hashing, fresh evaluation, schema generation, forbidden-reference checks, and pre-network failure; emit the standard/PEFT layout below. |
| `aptivra/export/{hf_conversion,multitier,gguf_exporter,mlx_export}.py` | Retire | These consume the custom v0 manifest or emit non-standard/incomplete formats. They cannot execute or faithfully export this Llama+LoRA model. |
| `aptivra/resources/huggingface_hub/*.template` | Retire | No custom configuration/model/tokenizer Python enters the new repository. |
| `aptivra/resources/configs/{task_stream,task_stream_v2,task_stream_audit,verifier_sandbox}.yaml`, `aptivra/resources/__init__.py` | Survive | Preserve frozen v1 and audited v2 task data and sandbox contracts. |
| `aptivra/resources/configs/model_card_release_gate.yaml` | Re-implement schema | Preserve fresh-evidence and forbidden-reference gates; add the new base, PEFT, BPB, identity, and package-layout invariants. |
| `aptivra/tasks/base.py` | Re-implement stopping only | Preserve task instances, mechanical verification, NFC exact matching, and decoded extraction. Replace `ord(character)` token IDs with typed decoded terminators plus EOS policy. |
| `aptivra/tasks/{stream,v2,modular_arithmetic,string_transforms,list_operations,syntax_matching,boolean_logic,sequence_completion,state_machines,python_microsynth,algo_synth,docstring_code,bug_fixer,type_inferrer}.py` | Survive | Generators/verifiers are model-independent. V1 stays frozen; v2 roles and template-OOD semantics remain binding. |
| `aptivra/tasks/{code_sandbox,sandbox_worker,sandbox_policy,structure}.py` | Survive | Contained execution and structural audits remain mandatory. |
| `aptivra/tasks/{identity,conversational}.py` | Re-implement completely | Remove obsolete/fabricated claims and use the identity/template contract below. |
| `aptivra/tasks/github_issues.py`, `aptivra/resources/configs/github_issues_stream.yaml` | Survive repository-side only | Required for the proposed second research stream; raw records and source-identifying material are excluded from publication staging. |
| `aptivra/tasks/{multitasking,skill_dataset}.py`, `aptivra/tasks/__init__.py` | Survive, out of main path | Their existence is not a demonstrated model capability and cannot appear as one in identity data. |
| `aptivra/agents/**`, `aptivra/ambient/**`, `aptivra/embodied/**`, `aptivra/federation/**`, `aptivra/governance/**`, `aptivra/learning/**`, `aptivra/planning/**`, `aptivra/research/**`, `aptivra/world_model/**` | Survive, out of path | Orthogonal prototypes are not attached to SmolLM2 and provide no capability evidence. |
| `aptivra/genesis.py`, `aptivra/benchmarks/**` | Retire from the SmolLM2 path | They aggregate/measure the legacy `ModelConfig` prototype, not the accepted base or experiment. |

### 2. Real routed experts

#### Selected expert

Each promoted expert is a PEFT LoRA adapter attached to all seven bias-free projection families in
every Llama layer: `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, and
`down_proj`. The starting rank is 16. The base is frozen; promoted adapters are frozen and
content-addressed. Only the candidate adapter and router update are trainable in a cycle.

For a linear weight of shape `out × in`, LoRA adds `r(in + out)` parameters. SmolLM2's grouped
query attention makes K/V outputs 320 rather than 960. Therefore:

```text
attention per rank = 32 * [(960+960) + 2*(960+320) + (960+960)] = 204,800
MLP per rank       = 32 * 3*(960+2,560)                         = 337,920
all projections    = 542,720 * r
rank 16 expert     = 8,683,520 parameters
```

One rank-16 expert is 2.40% of the base, 34,734,080 bytes in fp32, and 138,936,320 bytes for its
fp32 weights, gradients, and two AdamW moments. Twelve resident fp32 experts, with only one carrying
gradient/moment state, plus the frozen fp32 base require 1,968,295,680 static bytes before
activations and runtime. This is not a fit claim; it explains why the design is measurable under
the cap.

The starting rank and target modules are hypotheses, not facts. Rank 8/16/32 and MLP-only versus
all-seven targeting are mandatory ablations. An adapter is promoted only if its actual validation
gain and retained-task gate pass; parameter isolation does not prove adequate capacity or useful
transfer.

#### Selected router

The runtime router is a learned, sequence-level, top-1 classifier over a stable feature: masked
mean pooling of the frozen base token embeddings before any adapter is active. A linear
`960 → E` classifier has `961E` parameters. It runs once, activates one named PEFT adapter for the
whole sequence, and then performs the normal model forward. It is not token-level MoE and must
never be called Top-2 MoE.

Each expert row has a stable ID and task-family provenance. A new row is trained with current-task
examples plus deterministic, stratified replay from every prior expert. Old expert weights and
old router rows are frozen; replay teaches the new row not to steal prior inputs. During adapter
training the candidate is activated explicitly, so router mistakes cannot corrupt expert
specialization; router training and learned-route evaluation are separate phases. Router entropy,
confidence, misroute rate, load distribution, and verifier failure are measured on real held-out
prompts. A below-threshold route falls back to frozen-base generation while returning the normal
response type plus observable `route_status=unrouted` telemetry; it never guesses, silently changes
the API shape, or spawns during inference. Growth is proposed between cycles from
registered novelty/failure thresholds and still requires promotion-gate approval.

Task ID is used only to label router training and as an oracle routing ablation. The
LoRA-per-task-without-routing baseline manually selects its known adapter. This distinction is
necessary: otherwise “routing” contributes no behavior beyond an experiment harness lookup.

#### Evidence versus judgment

- Hu et al., *LoRA: Low-Rank Adaptation of Large Language Models* (2022), established that
  frozen pretrained weights can be adapted with low-rank trainable updates and showed strong
  parameter efficiency. It did not establish rank 16, these target modules, or continual-learning
  superiority for Aptivra.
- Houlsby et al., *Parameter-Efficient Transfer Learning for NLP* (2019), established compact
  per-task bottleneck adapters on a frozen transformer. It did not establish that a parallel
  adapter is preferable to LoRA here.
- Aljundi, Chakravarty, and Tuytelaars, *Expert Gate: Lifelong Learning With a Network of Experts*
  (2017), established learned expert selection in vision. Araujo, Moens, and Tuytelaars,
  *Learning to Route for Dynamic Adapter Composition in Continual Learning with Language Models*
  (2024), established that isolated PEFT modules plus learned, input-dependent adapter routing can
  be useful in language-model continual learning. Neither paper proves this deliberately simpler
  replay-trained top-1 router on Aptivra's streams.
- Komatsuzaki et al., *Sparse Upcycling: Training Mixture-of-Experts from Dense Checkpoints*
  (2022/2023) and Fedus, Zoph, and Shazeer, *Switch Transformers* (2022), establish dense-to-sparse
  reuse and sparse-routing behavior at much larger pretraining scales. They do not establish
  per-family continual learning on a 360M model or this machine.
- Kirkpatrick et al., *Overcoming Catastrophic Forgetting in Neural Networks* (2017), established
  EWC's diagonal-Fisher regularization; Chaudhry et al., *On Tiny Episodic Memories in Continual
  Learning* (2019), established strong replay baselines. They justify comparators, not assuming
  that either wins here.

The LoRA rank, all-seven targets, frozen-feature linear router, replay policy, and fallback are
Aptivra engineering judgments. H3 must falsify them.

#### Rejected expert alternatives

| Alternative | Exact arithmetic | Decision |
| --- | --- | --- |
| Parallel SwiGLU adapters | Projection-only lower bound `32 * 3 * 960 * b`; 11,796,480 parameters at `b=128`, 23,592,960 at `b=256` per expert, excluding router, branch norms, and biases | Rejected as default. Similar static cost buys custom layer surgery, an extra nonlinear branch, and non-standard packaging without Aptivra evidence. Keep `b=128` as an architecture ablation only if the implementation can remain standard-loadable. |
| Upcycled full FFNs | One all-layer FFN copy is 235,929,600 parameters. With per-layer router term `32 * 961 * E`, eight total experts produce 2,013,574,336 parameters and 32,217,189,376 bytes of full fp32 AdamW tensor state, already above the cap before activations. | Rejected. Four total experts are arithmetically smaller but leave exactly 11,875,302,400 bytes under the cap when all are trainable, break stock Llama structure, and depend on large continued-pretraining evidence that does not transfer to these streams. |
| Task-ID-only routing | Zero parameters and no routing error | Rejected for the full method because it reduces routing to an oracle lookup. Retained as the required upper-bound ablation and deterministic fallback for controlled evaluation. |
| Full fine-tuning per task | 361,821,120 mutable parameters; 5.789 GB of fp32 AdamW tensor state before activations | Rejected for Aptivra full because every task rewrites shared weights. Required as the naive sequential, replay, and EWC comparators. |

### 3. Tokenizer, stopping, BPB, and the release baseline

#### Tokenization and stopping

The byte tokenizer is retired from the new path. Prompt, target, padding, truncation, and maximum
generation lengths are measured in SmolLM2 token units. Oversize prompt-plus-target examples fail
materialization with task ID and measured lengths; they are never silently truncated. Because the
tokenizer has no pad token, collation explicitly uses EOS as the pad value while masking every pad
position in both attention and loss. Pad-as-EOS never makes a padding position a training target.

Task-family terminators remain decoded text so v1 and v2 records do not change:

- single-line target: `answer + LF + EOS`;
- multiline target: `answer + NUL + EOS`;
- generation stops at the first complete declared decoded terminator sequence or model EOS;
- the harness removes only the declared text terminator before the existing mechanical verifier.

The generator may pre-tokenize stop strings as an optimization, but it must also scan decoded byte
suffixes across token boundaries. It may not assume LF is token 198, NUL is token 190, or a stop
string is one token. EOS ID comes from the loaded tokenizer/config and must agree with the pinned
record.

#### Bits per byte

For fixed tokenization of examples `j`, corpus BPB is:

```text
BPB = [sum_j sum_i -ln p(token_ji | prefix_ji)]
      / [ln(2) * sum_j len(scored_text_j.encode("utf-8"))]
```

The numerator includes only the declared scored target token region; the denominator is the exact
original UTF-8 payload bytes represented by that region, not tokens, Unicode code points, prompt
bytes, vocabulary size, or a mean of per-example BPBs. Prompt and answer are encoded separately
then concatenated so a BPE merge cannot straddle the scoring boundary. Aggregate bits and bytes
are summed before division. Protocol special tokens, including EOS, are excluded from both the
target-language numerator and payload denominator; their termination accuracy is reported
separately. Exact encode/decode validation is a precondition.

The release baseline remains a **byte-domain unigram**, so it is tokenizer-independent. Its 256
probabilities are fitted once on the declared training-target UTF-8 bytes with a configured,
schema-recorded smoothing rule, frozen, and cross-entropy-scored on the exact held-out payload
bytes used by model BPB. Fitting a unigram on held-out bytes is an optimistic empirical-entropy
diagnostic, not the predictive release baseline. The new release passes only when fresh staged
model BPB plus the configured margin is strictly lower than the frozen training-fitted unigram
BPB. The uniform-byte diagnostic remains 8 BPB; `log2(49,152)` is not a byte baseline.

The v0 model-card schema and values are immutable. Implementation creates:

1. a versioned prompt/answer held-out v2 set formatted with the new prompt contract;
2. a versioned whole-text set, which may reuse v1 bytes only after a recorded contamination audit;
3. immutable hashes, record IDs, raw UTF-8 byte counts, model-token counts, and scoring masks; and
4. audits against identity/chat, v2, replay, and the second research stream.

Pretraining contamination of a public base is unknown. These sets support claims only about the
named evaluation, never a claim of uncontaminated general knowledge.

#### ADR-0006 disposition

ADR-0006 is retired for this model. Its v0 exporter and C binaries remain historical artifacts but
are not selectable, packaged, or described as SmolLM2 support. Twelve fp32 rank-16 adapters add
only 416,808,960 resident bytes, so NVMe streaming is not required by this experiment. A future
native backend requires a separate ADR covering the GPT2Tokenizer byte-BPE algorithm, Llama GQA,
RoPE, RMSNorm, SwiGLU, tied embeddings, LoRA application, KV cache, safetensors, and token-by-token
logit/generation equivalence across precision and placement. Header parsing is not inference.

### 4. Identity and conversation data

The selected base has no upstream chat template. Aptivra therefore owns and versions a minimal
template using already-present special tokens; it must never be described as “the SmolLM2 chat
template”:

```text
<|im_start|>system\n{system}<|im_end|>\n
<|im_start|>user\n{user}<|im_end|>\n
<|im_start|>assistant\n{answer}<|im_end|><|endoftext|>
```

`<|im_end|>` closes a message and EOS closes the sample. At inference, the assistant turn stops
on either complete `<|im_end|>` or EOS and strips the turn delimiter. Raw task-stream prompts are
not silently wrapped in this template; changing their prompt distribution would invalidate the
stream audit.

Identity is trained in a separate `identity` LoRA adapter, never by modifying the base. Permitted
facts are generated from a signed/schema-validated identity manifest:

- model name and owner-approved creator attribution;
- exact upstream base repository and revision;
- “frozen base plus routed LoRA adapters” only after those components exist;
- research status, known hardware/recipe qualification, and explicit limitations; and
- measured capabilities only when copied from an immutable release-evaluation record with metric,
  stream version, seed set, denominator, and date.

Required limitation examples include: an ordinary conversation changes only ephemeral context and
does not persist weights, replay, or episodic memory; it may be wrong; under automatic routing an
adapter is active only when selected (manual adapter selection is an explicit evaluation/runtime
mode); and capability is limited to the named evaluations. Forbidden targets include “autonomous,”
“Top-2 MoE,” 200M, zero forgetting,
skill counts, native/C-engine execution, throughput, or any metric not bound to release evidence.
The output sanitizer may remove delimiters; it may not insert or replace identity facts.

The initial canonical records are deliberately modest:

| Prompt intent | Evidence-bound target shape |
| --- | --- |
| “What are you?” | “I am Aptivra, a research language model built on `HuggingFaceTB/SmolLM2-360M` at revision `f8027f…` and adapted with separately routed LoRA modules. My measured abilities and limitations are listed in my model card.” |
| “What is your architecture?” | “My frozen base is a 361,821,120-parameter Llama causal language model. Aptivra adds rank-16 LoRA experts and a sequence-level top-1 router; it is not a Top-2 mixture-of-experts model.” |
| “Do you learn from this chat?” | “This conversation changes my temporary context, but it does not persistently update my weights, replay buffer, or episodic memory. Persistent learning requires the explicit verifier-gated training and promotion process.” |
| “Who created Aptivra?” | The owner-approved creator attribution, without an architecture or capability suffix. |
| “What can you do?” | A generated answer containing only metrics and scope copied from the bound release record, or an explicit statement that no release evidence is available. |

The displayed revision may be shortened for conversation only if the full revision remains in the
bound manifest and model card; stored training provenance always uses the full value.

### 5. Training recipe on the 36 GiB M4 Max

All methods use the same pinned base, tokenizer, serialized examples, ordering, target-only causal
loss, optimizer-step budget, evaluation schedule, and declared hyperparameter-selection budget.
The accepted hardware contract remains fp32. BF16 source weights are explicitly cast to fp32 before
training; no mixed precision, quantization, CPU offload, or cloud result may be substituted into
the primary experiment without a new pre-registered amendment.

| Method/state | Mutable parameters | Persistent fp32 tensor-state consequence |
| --- | ---: | ---: |
| Frozen base | 0 | 1.447 GB base weights |
| Naive sequential / replay full fine-tune | 361,821,120 | 5.789 GB weights+grads+AdamW moments before activations |
| Online EWC full fine-tune | same | adds one consolidated Fisher and anchor: 2.895 GB; per-task Fishers are forbidden because 12 copies would add 34.7 GB |
| One active rank-16 LoRA expert | 8,683,520 | 138.9 MB for that expert's weights+grads+moments |
| Twelve resident rank-16 experts, one trainable | 8,683,520 active | 1.968 GB including fp32 base, all expert weights, and active expert optimizer state |
| Learned router | `961E` | negligible, but fully checkpointed and measured |

Aptivra full freezes the base and every promoted adapter. It trains one new adapter and the new
router row with current examples plus replay. EWC is not part of Aptivra full because there is no
mutable shared backbone to protect and frozen prior rows/adapters cannot drift; retaining an EWC
penalty there would be ceremonial. Online EWC remains a required full-fine-tuning comparator.

The initial effective batch is 8, matching ADR-0008. A measured calibration tries micro-batches
`[8, 4, 2, 1]` in descending order and sets accumulation to preserve effective batch 8. It measures
the actual maximum token-length buckets present in each stream, including the longest accepted
example, with one warm-up and at least three forward/backward/optimizer steps. Target-only LM-head
projection is used only through the already-proven mathematical contract; gradient checkpointing
is a recorded fallback only at micro-batch 1. Every attempt synchronizes MPS and records MPS current
and driver allocation, process RSS, system pressure, step time, tokens/second, sequence lengths,
micro-batch, accumulation, and checkpointing. The recipe is accepted only if the candidate's own
peak is at most 28,991,029,248 bytes, no CPU fallback occurs, losses are finite, and a repeat run is
within the pre-registered numerical tolerance. No fit is inferred from the static table.

Replay uses a deterministic, stratified reservoir with a starting capacity of 500 canonical text
records. It stores no permanent accelerator tensors. Even an optional maximum-size int32 cache of
both IDs and labels is bounded by `500 * 8,192 * 2 * 4 = 32,768,000` bytes (31.25 MiB), plus masks
and metadata; the normal path retokenizes text and is smaller. Capacity, ratio, sampling seed,
per-family minimum, byte limit, and eviction policy become validated registered configuration in
the implementation phase.

Consolidation consists of bounded replay/router fitting, current-adapter training, chronological
evaluation, promotion decision, and atomic freeze or rollback. It performs no generative dreaming
and no pruning. A rejected candidate restores adapter bank, router, optimizer/scheduler, replay,
episodic records/index, configuration, and Python/NumPy/Torch CPU/MPS RNG state while retaining the
immutable rejection evidence and consumed compute.

Each consolidation cycle is bounded before execution by registered current-plus-replay target-token
and optimizer-step caps for adapter fitting, replay-example and step caps for router fitting, and an
evaluation budget proportional to the number of seen tasks times their held-out and template-OOD
examples. The H3 ADR pre-registers those caps rather than choosing them after results are visible.
The cycle records actual target tokens, steps, generations, wall time, peak MPS/RSS, and cap usage;
any overrun aborts and rolls back. This makes consolidation cost observable without inventing an
unmeasured duration for this machine.

### 6. Evaluation, baselines, and H3 boundary

The claim-bearing experiment runs separately on the proposed v2 synthetic stream and the proposed
chronological issue stream. Neither proposal becomes accepted through this ADR; each must pass its
own audit first. Frozen v1 is executed only as a compatibility/regression cell and cannot support
new capability claims. Results from v1, v2, and the chronological stream are never pooled.

Each method runs with at least five pre-declared seeds. The exact seeds, power analysis, margins,
multiple-comparison policy, and compute-matching rule belong in the separate H3 preregistration and
must be frozen before training.

Required baselines are:

1. frozen pretrained base with no updates;
2. naive sequential full fp32 fine-tuning;
3. sequential full fp32 fine-tuning with deterministic replay;
4. sequential full fp32 fine-tuning with bounded online EWC;
5. rank-16 LoRA per task with manual task-ID adapter selection and no learned router;
6. Aptivra full: frozen base, isolated LoRA growth, learned router, replay, verifier-gated memory,
   and transactional promotion/rollback.

A joint multitask run may be reported as an offline upper bound but is not a continual baseline.
Failed/OOM methods remain in the table with their failure evidence; they are not silently dropped.

Required ablations include LoRA rank 8/16/32; MLP-only versus all-seven projections; learned versus
task-ID oracle routing; router without replay; no replay; no expert freezing; fixed one-adapter
capacity; no promotion gate; no verifier-gated memory; and no growth trigger. Removing freezing is
allowed only in an ablation and must restore from a pristine state afterwards.

For every seed/method/stream, record:

- the chronological in-template accuracy matrix, final average accuracy, BWT, FWT relative to the
  frozen pretrained base vector, and per-task/aggregate maximum forgetting;
- v2 template-OOD accuracy matrix and the same metrics as a separate, never-pooled cell;
- T2 as a tokenizer-sensitivity diagnostic, excluded from v2 headline aggregates as ADR-0010
  requires;
- router accuracy, confusion matrix, abstention, entropy, confidence calibration, expert load,
  and task-ID-oracle gap;
- new-task validation/test accuracy and promotion rejection/rollback counts;
- held-out text and answer BPB, EOS/terminator success, and verifier infrastructure failures; and
- trainable/total/resident parameters, adapter bytes, replay bytes, optimizer steps, target tokens,
  FLOPs estimate with method, wall time, peak MPS/RSS, and tokens/second.

Lopez-Paz and Ranzato's *Gradient Episodic Memory for Continual Learning* (2017) establishes the
accuracy-matrix/BWT/FWT evaluation family; it does not dictate Aptivra's statistical decision.

H3 must test whether learned routed, frozen LoRA growth with replay and promotion improves
retention over the strongest eligible non-routing baseline without materially reducing new-task,
final-average, or template-OOD quality under the fixed machine budget. Its kill criterion has this
shape:

1. **superiority:** paired multi-seed intervals must show BWT improvement above zero and forgetting
   reduction—defined as comparator forgetting minus Aptivra forgetting—above zero, and both must
   clear pre-registered practically meaningful thresholds;
2. **non-inferiority:** final average, new-task acquisition, and template-OOD must stay within
   pre-registered negative margins;
3. **routing reality:** among prompts above the pre-registered confidence threshold, routed coverage
   (the fraction actually assigned an adapter), misroute rate, and the accuracy gap to task-ID oracle
   must meet pre-registered bounds; end-task accuracy is compared separately against frozen-base
   fallback rather than treating fallback as “coverage”;
4. **replication:** both claim-bearing streams must pass independently—one cannot rescue the other;
5. **operability:** any cap breach, incomplete baseline, failed rollback, gate bypass, or missing
   seed kills the claim.

Failure of any conjunct rejects H3 and leaves the architecture experimental. Exact thresholds and
power are intentionally not chosen after this design review; the H3 ADR must pre-register them.

### 7. Hub packaging and attribution

The new model goes to a new repository whose exact name remains an owner decision. That missing
name blocks every create/push/tag operation but does not block local implementation or evaluation.
The historical byte-level repository is immutable and is never repurposed.

The package is a standard PEFT multi-adapter bundle, not remote code:

- no `auto_map`, `trust_remote_code`, custom model/config/tokenizer Python, or copied Llama code;
- an immutable manifest naming the upstream base, revision, hashes, Transformers/PEFT versions,
  adapter IDs, target modules/ranks, router schema, identity schema, evaluation record, and license;
- one standard PEFT adapter directory per publishable expert plus a router safetensors/JSON sidecar;
- a model card that shows stock `AutoModelForCausalLM` loading of the pinned base and standard PEFT
  loading of an explicitly selected adapter, then separately states that automatic routing needs
  the installed Aptivra orchestrator; and
- an allowlist and hash manifest. Stock loading is tested offline with `trust_remote_code=False`.

This is more honest than a one-call custom architecture: Transformers loads the actual standard
base, PEFT loads actual adapters, and the card does not pretend the sidecar router is a built-in
Llama layer. A merged checkpoint may be produced only as a separately named, single-adapter
derivative; it cannot represent dynamic Aptivra routing.

PEFT is not currently a project dependency. Before the expert lane, implementation must select and
pin a reviewed PEFT version compatible with the already-pinned Transformers and Torch versions,
update the lock through the repository's normal dependency workflow, record its license, and pass
an offline import plus tiny-adapter save/load smoke test. This ADR authorizes that future dependency
evaluation; it does not claim compatibility or install anything now.

SmolLM2 is Apache-2.0. The new repository retains the upstream license, copyright and NOTICE
obligations, names the exact base and revision, marks Aptivra modifications, and identifies every
adapter as a derivative. Legal approval remains with the owner.

The existing case-insensitive forbidden-reference gate is retained and strengthened to scan every
staged path, filename, UTF-8 text file, JSON/YAML value, model-card field, tokenizer/template file,
and safetensors metadata before any network client is constructed. The configured prohibited term
remains absent from all Hub content. Raw chronological-stream data, its configuration, provenance,
source-identifying names, and adapters trained from it are excluded from the public bundle unless a
later owner-authorized legal/publication decision explicitly changes that boundary. Binary tensor
payloads cannot be meaningfully substring-scanned; their metadata and provenance are scanned, and
only allowlisted synthetic/identity adapters may be staged.

The public staged bundle is therefore a strict subset of the locally evaluated two-stream research
system. Fresh publication evaluation of the exact staged bytes covers only its publishable adapters
and claims. The full two-stream H3 remains local, and neither the staged bundle nor its model card
may claim to be that full system unless a separate owner-authorized legal/publication decision adds
the excluded adapter and the exact expanded staged bytes are re-evaluated.

Publication still requires schema validation, a strict unigram-gate pass, capability/identity
evidence checks, a mutation check after fresh evaluation, the owner-confirmed new repository name,
and separate authorization for the outward operation.

### 8. Migration, tests, and rollback

V0 checkpoints, byte tokenizer files, remote-code packages, TMR manifests, experimental records,
and the historical repository retain a `v0-byte-level` identity. They are never in-place converted.
The new loader rejects them by architecture/tokenizer fingerprint; the v0 loader remains explicit
and cannot be selected by a SmolLM2 config. New records use new schema versions and cannot be
compared to v0 capability metrics without an explicitly defined bridge. BPB itself is
tokenizer-comparable only when both models score the exact same UTF-8 held-out bytes, boundary and
special-token policy; BPB values from different corpora or scoring protocols are not comparable.

Implementation proceeds as small vertical slices. Each lane is independently revertible and the
new feature remains disabled until the final local acceptance gate passes. V0 is quarantined for
historical reproducibility, not used as a production fallback, because its identity and TMR claims
are known false. If explicitly invoked for historical evaluation, those claims are suppressed and
the output is labelled `v0-byte-level historical evaluation only`.

| Lane | Deliverable | Acceptance tests before next lane |
| --- | --- | --- |
| 0. Freeze and fingerprint | Record v0/new schema boundaries and pinned local base/tokenizer facts. | Existing v0 artifact hashes and tests unchanged; wrong revision/hash/architecture fails before allocation. |
| 1. Standard base slice | Offline stock load, fp32 cast, tokenizer, masked collation, raw task forward/generation. | Tiny local Llama fixture; no network/remote code; tied weights; pad mask; oversize fails; strict deterministic EOS/stop behavior on CPU and correct outcomes within pre-registered numerical/repeatability tolerances on MPS. |
| 2. Token/verifier/eval slice | Decoded LF/NUL/EOS stopping, v5 evidence schema, aggregate subword BPB, training-fitted byte unigram. | Analytic byte/subword/multibyte BPB equality; multi-token stop across boundaries; no stop leakage; held-out contamination/hash failures; unigram gate pass/fail boundaries. |
| 3. Baseline slice | Frozen, sequential, replay, online-EWC, and manual LoRA baselines through the real harness. | Tiny-model end-to-end matrices; deterministic five-seed scheduling; target-only loss equivalence; replay bound; EWC online-state bound; finite loss and compute/memory records. |
| 4. Expert slice | Pin the reviewed PEFT dependency; put a rank-16 all-seven expert in the actual forward and freeze/load it by content hash. | Lock/license/import compatibility passes; old code would fail: expert gradients/logits change when active, base and mature adapters remain byte-identical, wrong base/target/rank fails, standard PEFT reload reproduces logits. |
| 5. Router/consolidation slice | Learned top-1 router, replay calibration, real growth inputs, promotion, atomic rollback. | Misroute/abstain paths; no random dummy input; real verifier-failure propagation; row/expert freeze; rejected cycle restores every state/RNG/hash; accepted cycle survives restart. |
| 6. Identity/chat slice | Versioned Aptivra template and evidence-generated identity adapter. | No unsupported claim; exact message/EOS termination; unsanitized evaluation; stale/missing evidence fails generation; raw task prompts unchanged. |
| 7. Packaging slice | Local standard/PEFT staged repository and new model card. | Offline `trust_remote_code=False`; no Python/`auto_map`; exact allowlist/hashes; base license/attribution; prohibited filename/text/metadata fails before a fake network client is called. |
| 8. Measured pilot and H3 | Real-MPS memory calibration, then separately pre-registered experiment. | Peak ≤ cap, no fallback, repeatability tolerance, all baselines/seeds/streams complete, release gate and H3 rule evaluated fail-closed. |

### Lane 0–1 implementation notes

The owner accepted this ADR on 2026-10-16. Lane 0 records the immutable
`v0-byte-level` boundary and the new `smollm2-standard-base-v1` fingerprint schema in the
versioned standard-base configuration; no v0 artifact or schema is converted in place. Lane 1
remains feature-disabled by default pending the final local acceptance gate.

The MPS tolerance uses a non-degenerate stock `LlamaForCausalLM`: tied embeddings, non-zero random
weights from CPU generator seed `20261016`, and three inputs with distinct, non-tied top-1 tokens.
The narrow case has a CPU prompt-decision margin of `0.0036088228`; its generation reaches a
smaller `0.0002851635` margin while still producing six tokens rather than terminating immediately.
The unchanged load/collate/forward/generate path was executed five times on the Apple M4 Max under
torch `2.14.1` and transformers `5.18.0`. MPS generated IDs, top-1 IDs, margins, and losses were
identical across all five runs. CPU fp32 produced the same decision IDs; the largest tiny-fixture
CPU/MPS margin and loss deltas were `5.9604645e-8` and `2.3841858e-7`.

The cached immutable SmolLM2-360M was measured on three short prompts by the same procedure. Its
five MPS runs were identical, and every generated and prompt top-1 ID matched CPU fp32. Its largest
CPU/MPS margin and loss deltas were `4.3869019e-5` and `4.9591064e-5`. Scalar tolerances are
`max(1e-6, 2 × measured delta)`. A token mismatch is accepted only when the corresponding CPU
top-1/top-2 margin is no greater than the fixture's doubled measured CPU/MPS margin delta (or the
`1e-6` resolution guard when larger); this yields decision epsilons of `1e-6` for the tiny fixture
and `8.7738037e-5` for SmolLM2. The complete per-run observations, CPU references, environment,
floor, and tolerance derivation are frozen in
[`tests/standard_base_mps_tolerances.json`](../../tests/standard_base_mps_tolerances.json).

Only these existing assertions encode behavior this ADR deliberately replaces and qualify for the
test-rewrite exception:

- `test_model_identity_task_family`: replace expectations derived from fabricated v0 targets with
  evidence-manifest targets and explicit rejection of every forbidden claim;
- `test_format_chat_prompt_multiturn` and the prompt-format assertions in
  `test_normalize_gradio6_message_history`: replace `User:/Assistant:` expectations with the
  versioned Aptivra delimiter/EOS contract while preserving history-normalization assertions;
- `test_canonical_prefix_ok`: replace its explicit acceptance of “autonomous” with rejection; and
- `test_sanitize_identity_name`: replace fact-substitution expectations with delimiter-only,
  no-fact-insertion behavior.

The generic SQL-drift, first-sentence, identity-bundle, held-out contamination, and stream-audit
assertions remain. Their fixtures/hashes may be versioned to the new identity corpus, including in
`test_versioned_heldout_sets_have_required_coverage_and_no_training_strings` and
`test_audit_detects_identity_training_string_contamination`, but their safety assertions may not be
weakened. V0 compatibility assertions in `tests/test_multitier_export.py`,
`tests/test_hf_packaging.py`, and `tests/test_inference_causal_lm.py` remain under an explicit
legacy path; new SmolLM2 tests are added rather than disguising v0 as the new model.

Before final acceptance there is no production promotion to roll back: disable the new feature and
retain every candidate/evidence artifact. After acceptance, rollback selects only the immediately
prior verified SmolLM2 base+adapter+router manifest, never v0. Because the base is immutable and
adapters are separate, rollback is pointer replacement after hash verification, followed by the
release smoke/evaluation subset. Training candidates are never promoted in place.

### Lane 2 implementation notes

Lane 2 keeps the immutable v0 `aptivra.model_card_eval.v4` implementation intact and introduces
the standard-base evidence boundary as `aptivra.model_card_eval.v5`. A v5 metric is independently
tagged `aptivra.bpb_metric.v5`; release comparison validates both tags and exact corpus hash and
UTF-8 byte count before comparing values, so a v0 dictionary cannot enter a v5 comparison.

The lane-1 EOS-only generator call remains available for diagnostics. Task-aware generation is an
additive entry point that receives the task instance's existing decoded terminators, scans the
decoded generated UTF-8 after every token across token boundaries, and removes the earliest
complete terminator before returning text. This choice avoids changing any v1/v2 serialized task
field or audit hash while making LF/NUL semantics independent of token IDs.

The open byte-unigram smoothing choice is add-k over all 256 byte values with `k = 1.0`. It is
fitted once from the ordered, declared training-target texts, records the canonical JSON corpus
hash and raw training byte count, and is frozen before held-out scoring. Both held-out answers and
held-out whole text must strictly beat their like-for-like training-fitted baseline; equality is a
failure. Corpus evidence hashes ordered UTF-8 text arrays as compact JSON, while BPB denominators
remain the sum of the raw text payload bytes rather than serialization bytes. Contamination audits
require identity/chat, v2 stream, replay, and chronological-issue inputs, validate every supplied
hash first, and reject containment in either direction. Missing sources, stale hashes, or overlaps
fail closed.

### Lane 4 implementation notes

The reviewed dependency is PEFT `0.21.2`, pinned exactly. It was selected because its declared
runtime bounds include Torch 2.14.1, Transformers 5.18.0, Hugging Face Hub 1.33.0, and
Safetensors 0.8.0 already locked by this repository, and an offline import plus tiny-Llama
adapter save/load/reload test passes against that exact closure. Installed package metadata records
PEFT and its newly locked Accelerate 1.15.0 transitive as Apache licensed, and psutil 7.2.2 as
BSD-3-Clause; those terms are compatible with Aptivra's Apache-2.0 distribution. No other direct
runtime dependency was added.

The standard expert path remains disabled by default and is additive to the lane-3 manual-LoRA
oracle comparator. A promoted artifact contains the standard PEFT adapter directory plus an
`aptivra.peft_expert_manifest.v1` sidecar binding the base repository, revision, config and weight
hashes, rank, target set, parameter count, PEFT identity, and the exact adapter config/weight bytes.
Every field and content hash is verified before standard PEFT loading. Promotion stages the bytes
on the destination filesystem, reuses an identical existing artifact, refuses replacement of a
different artifact, and only then freezes the promoted adapter.

A fresh Cosmic Ray run generated 630 mutants for `aptivra/experts/standard.py`: 539 were killed,
90 survived, and one `ExceptionReplacer` mutant was incompetent because Cosmic Ray raised an
`AttributeError` while applying it, before a mutated program or test result existed. The raw kill
rate across all 629 executable mutants is therefore 85.69% (539/629); counting the incompetent
operator failure conservatively in all 630 generated entries gives 85.56%. Of the 90 survivors,
55 are equivalent binary-operator substitutions in the four PEP 604 annotations: 22 for
`Path | str | Traversable`, and 11 each for `_LlamaArchitecture | ArchitectureFingerprint`,
`Path | str` on promotion, and `Path | str` on reload. `from __future__ import annotations`
postpones these annotations, and this module never evaluates them at runtime, so replacing their
`|` tokens cannot change execution. Removing only those 55 equivalent mutants leaves 574
executable, non-equivalent mutants and an equivalence-adjusted kill rate of 93.90% (539/574),
which passes the 90% Lane 4 threshold. No source structure, lint suppression, mutation operator,
or mutation configuration was changed to alter the generated universe.

## Consequences

### Positive

- Aptivra uses the accepted pretrained base through its supported stock architecture.
- Experts become real low-rank changes in the forward path, with frozen prior capacity and small,
  measurable growth.
- Learned routing is evaluated as a mechanism rather than hidden behind task IDs.
- V0 remains reproducible while false byte-token, Top-2, remote-code, and C-runtime claims cannot
  enter the rebuilt model.
- Standard Transformers plus PEFT minimizes package code and supply-chain surface.
- The experiment can reject the architecture through explicit baselines, ablations, and H3 gates.

### Negative and limitations

- A multi-adapter router is not a one-call stock Transformers model; Aptivra orchestration is
  required for automatic selection and must be installed separately.
- Frozen experts reduce direct parameter interference but do not prevent routing errors, base
  limitations, distribution shift, capacity failure, or negative transfer.
- LoRA rank 16 and mean-embedding routing are unproven on both streams.
- Adapter count and storage grow monotonically; promotion and configured limits bound growth but
  do not eliminate it.
- The primary experiment remains fp32/MPS-specific. It does not establish behavior under mixed
  precision, quantization, CUDA, cloud hardware, or a native runtime.
- Proposed ADR-0009 and ADR-0010 remain proposals; this ADR cannot convert their datasets into
  accepted evidence.
- Retiring ADR-0006 for the rebuilt model removes the advertised native/TMR deployment path until
  a real implementation passes equivalence.

## Alternatives considered

1. **Wrap SmolLM2 in the current `AptivraCausalLM`.** Rejected because it preserves byte IDs,
   custom state names, remote code, and experts outside the trained loss.
2. **Continue full fine-tuning and rely on replay/EWC.** Rejected as Aptivra full because it does
   not provide isolated growable capacity. Retained as required baselines.
3. **Use only task-ID LoRA adapters.** Rejected as the full architecture because the purported
   router would be an oracle lookup; retained as the clean parameter-isolation upper bound.
4. **Upcycle SmolLM2's FFNs into token-level MoE.** Rejected on static memory, custom architecture,
   training-data demand, packaging, and absence of evidence for these streams.
5. **Keep remote code for a one-call Hub experience.** Rejected because convenience would obscure
   which behavior is stock Llama, PEFT, and Aptivra routing while increasing trust and audit cost.
6. **Adapt ADR-0006 now.** Rejected because the current C implementation is not an inference
   engine and adapter size does not require tiering. A claim without execution evidence is false.
7. **Overwrite v0 artifacts or the existing Hub repository.** Rejected because it destroys
   reproducibility and violates ADR-0008 and the owner decision.

## References

- Vladimir Araujo, Marie-Francine Moens, and Tinne Tuytelaars (2024),
  [*Learning to Route for Dynamic Adapter Composition in Continual Learning with Language
  Models*](https://aclanthology.org/2024.findings-emnlp.38/), Findings of EMNLP 2024,
  doi:10.18653/v1/2024.findings-emnlp.38.
- Rahaf Aljundi, Punarjay Chakravarty, and Tinne Tuytelaars (2017),
  [*Expert Gate: Lifelong Learning With a Network of Experts*](https://openaccess.thecvf.com/content_cvpr_2017/html/Aljundi_Expert_Gate_Lifelong_CVPR_2017_paper.html),
  CVPR 2017.
- Arslan Chaudhry, Marcus Rohrbach, Mohamed Elhoseiny, Thalaiyasingam Ajanthan, Puneet K.
  Dokania, Philip H. S. Torr, and Marc'Aurelio Ranzato (2019),
  [*On Tiny Episodic Memories in Continual Learning*](https://arxiv.org/abs/1902.10486),
  arXiv:1902.10486.
- William Fedus, Barret Zoph, and Noam Shazeer (2022),
  [*Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient
  Sparsity*](https://jmlr.org/papers/v23/21-0998.html), JMLR 23(120).
- Neil Houlsby, Andrei Giurgiu, Stanislaw Jastrzebski, Bruna Morrone, Quentin de Laroussilhe,
  Andrea Gesmundo, Mona Attariyan, and Sylvain Gelly (2019),
  [*Parameter-Efficient Transfer Learning for NLP*](https://proceedings.mlr.press/v97/houlsby19a.html),
  ICML 2019.
- Edward J. Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang,
  and Weizhu Chen (2022),
  [*LoRA: Low-Rank Adaptation of Large Language Models*](https://arxiv.org/abs/2106.09685),
  ICLR 2022.
- James Kirkpatrick et al. (2017),
  [*Overcoming Catastrophic Forgetting in Neural Networks*](https://doi.org/10.1073/pnas.1611835114),
  PNAS 114(13):3521–3526.
- Aran Komatsuzaki, Joan Puigcerver, James Lee-Thorp, Carlos Riquelme Ruiz, Basil Mustafa,
  Joshua Ainslie, Yi Tay, Mostafa Dehghani, and Neil Houlsby (2022/2023),
  [*Sparse Upcycling: Training Mixture-of-Experts from Dense Checkpoints*](https://arxiv.org/abs/2212.05055),
  arXiv:2212.05055v2.
- David Lopez-Paz and Marc'Aurelio Ranzato (2017),
  [*Gradient Episodic Memory for Continual Learning*](https://papers.nips.cc/paper/2017/hash/f87522788a2be2d171666752f97ddebb-Abstract.html),
  NeurIPS 2017.

## Retrospective

The original architecture named an expert system before proving that experts participated in
learning, and named a native runtime before it executed a model. The pretrained-base decision made
those gaps impossible to ignore. Prevention is now structural: actual forward-path gradient tests,
content-addressed standard artifacts, evidence-generated identity, explicit historical schemas,
learned-router versus oracle measurements, and a killable H3 preregistration. Parameter arithmetic
narrows the design space, but only measured MPS peaks and end-to-end outcomes can accept it.
The selection follow-up exposed a second prevention rule: base eligibility is evaluated against
the entire claim-bearing method matrix, not only the proposed method's cheapest training state.

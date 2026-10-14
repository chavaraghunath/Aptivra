# Aptivra

Aptivra is a research codebase for byte-level causal-model experiments, continual-learning
harnesses, tiered export formats, and prototype inference wrappers. The repository contains a
decoder-only Transformer backbone, a soft Top-2 MoE router and expert modules, evaluation
harnesses, a self-contained Transformers export, local export utilities, and documentation of
negative results. The packaged checkpoint contains 271,361,024 unique parameters and has a
512-byte context window.

The published checkpoint must not be described as a usable general language model unless a
machine-readable v4 evaluation record shows held-out answer-span bits per byte beating the
unigram baseline fitted to those same answer bytes. The current audit found low loss on some
seen identity prompts but poor held-out byte
prediction; see `docs/RESULTS.md` for the erratum and historical context.

## What Is Measured

- The exported backbone/router/expert parameter count is measured by
  `aptivra.eval.model_card_eval.count_export_parameters`.
- Model-card metrics are generated from a strict v4 JSON record. Hub publication copies the exact
  allowlisted export bytes into private staging, measures that copy against the bundled versioned
  8+ KB held-out prose/code corpus, and hash-verifies it immediately before upload; an optional
  record is only checked for reproducibility and never controls the card or notice.
- Hub packaging consists of one `model.safetensors`, configuration/tokenizer metadata,
  `conversion-provenance.json`, and self-contained remote code loaded by stock Transformers
  with `trust_remote_code=True`.
- The code uses learned absolute position embeddings in `aptivra/model/transformer.py`.
- GGUF is a local FP32/FP16 export option only. GGUF files are not part of the Hub release, and
  the C CLI currently parses their headers rather than providing a measured inference engine.
- AID-Bench checks subsystem invariants in `aptivra/benchmarks/aid_bench.py`; it is not a model
  capability score.
- Generated Python used by mechanical verifiers and synthesized research tools runs in a
  resource-limited child process with a wall-clock timeout and process-group termination. This
  contains hangs and accidental failures during evaluation; it is not a security boundary and
  must not be used to execute hostile code.

## Repository Layout

```text
aptivra/                  Python package
  benchmarks/             AID-Bench subsystem checks
  eval/                   Continual-learning, model-card, and base-model evaluation
  export/                 Hugging Face, GGUF, MLX, and multitier export helpers
  experts/                MoE router and expert modules
  inference/              Local inference wrappers
  model/                  Transformer backbone and config
configs/                  Runtime, release-gate, and base-candidate configuration
docs/                     ADRs, run options, results, and configuration register
experiments/              Recorded experiment outputs
scripts/                  Training, export, evaluation, and local runtime CLIs
tests/                    Unit and integration tests
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

Use the shared project virtualenv when working in the maintained workspace:

```bash
PYTHONPATH=$PWD /Users/raghunath/code/personal/Aptivra/.venv/bin/python -m pytest -q -p no:cacheprovider
PYTHONPATH=$PWD /Users/raghunath/code/personal/Aptivra/.venv/bin/python -m ruff check .
PYTHONPATH=$PWD /Users/raghunath/code/personal/Aptivra/.venv/bin/python -m mypy --strict aptivra
```

## Model-Card Evaluation

Model weights are expected in a Hub-layout directory such as `scratch/hf_export`, containing
`model.safetensors`, `config.json`, `conversion-provenance.json`, tokenizer metadata, and the
three remote-code modules. Convert a legacy shard layout first; conversion records source shard
hashes, source layout, optional source Hub revision, and converter version in the provenance file.

```bash
PYTHONPATH=$PWD python -m scripts.evaluate_model_card \
  --export-dir scratch/hf_export \
  --output experiments/model_card_eval.json \
  --hub-revision f55718d
```

Render the Hugging Face README from that record through
`aptivra.export.model_card.generate_model_card_readme`. The release gate threshold is configured
in `configs/model_card_release_gate.yaml` and registered in `docs/configuration-register.md`.
The recorded losses establish byte prediction only on the named, hashed prompt and text sets.
They do not establish factual accuracy, generated-answer correctness, instruction following,
safety, or performance on other inputs. Whole-text scoring uses fixed-context chunks and retains
the preceding byte at each chunk boundary.

## Base-Model Candidate Evaluation

The proposed [ADR-0008](docs/adr/0008-adopt-pretrained-base-model.md) compares seven permissively
licensed pretrained causal models and the current byte-level checkpoint as a reference. Candidate
identities, immutable revisions, powered MPS budgets, learning-rate grid, seeds, confidence level,
and Amendment 2 family-clustered decision rule live in `configs/base_model_candidates.yaml`; the
exact powered-run config is preserved beside the evidence as `powered-sweep-config.yaml`. The
Amendment 4a noise-floor and trajectory controls are frozen in
`configs/base_model_candidates_amendment_4a.yaml`; the current fresh-run protocol is
`configs/base_model_candidates_amendment_4c.yaml`. The
evaluator writes an immutable run manifest, one atomic JSON record per candidate, qualifying
repeat-seed records, a versioned equivalence record, and `ranking.json`. Amendment 3 evidence
remains historical under `experiments/base_model_candidates/amendment-3/`. Amendment 4 and 4a
evidence lives under `experiments/base_model_candidates/amendment-4/` and
`experiments/base_model_candidates/amendment-4a/`; neither protocol produced a ranking.
The stopped `amendment-4b-override` directory and log are immutable evidence. Amendment 4c writes
only to `experiments/base_model_candidates/amendment-4c/`.
Each candidate runs in a fresh process so an MPS allocator failure cannot contaminate later
measurements; completed matching records are resumable. The rejected one-item sweep is preserved under
`experiments/base_model_candidates/archive/first_underpowered_sweep/`.

```bash
PYTHONPATH=$PWD /Users/raghunath/code/personal/Aptivra/.venv/bin/python \
  -m scripts.evaluate_base_model_candidates \
  --config configs/base_model_candidates_amendment_4c.yaml
```

That command uses the existing manifested owner override, selects the largest fitting micro-batch
from `8, 4, 2, 1`, projects through the LM head only at target positions, and uses gradient
checkpointing only after micro-batch 1 fails. Candidate feasibility is determined only by the
measured maximum of process RSS and MPS driver allocation against the 36 GiB target with 25%
headroom. Parent monitors record swap and the `memory_pressure` free-memory percentage before and
after every attempt. More than 1 GiB swap growth or less than 10% free memory makes the attempt
`environment_invalid`; the orchestrator waits up to 900 seconds for recovery and retries up to two
total attempts. Exhaustion records `not_measured_environment`, and ranking refuses to select while
any non-reference candidate remains unmeasured. Frozen manifests fail closed rather than accept
changed code or configuration.

The current Amendment 4 equivalence record is failed. SmolLM2-135M step 1 matched loss within
`1.37e-7` and maximum absolute parameter update within `9.24e-7`, but its relative-L2 update
difference was `0.001889` against the pre-registered `0.0001` tolerance. The gate stopped before
GPT-2 and before every candidate worker. There is therefore no Amendment 4 ranking; rerunning the
command reuses the immutable failed proof and exits without a sweep.

Amendment 4a retained that failure and tested equivalence against a measured noise floor without
moving any tolerance. Exact float64 CPU loss and gradients passed on SmolLM2-135M and GPT-2 at the
selected micro-batch and micro-batch 1. SmolLM2-135M then failed MPS batch 1: gradient relative L2
was `9.84e-6` against a `5.22e-7` limit, and step-1 AdamW update relative L2 was `7.26e-4` against
a `4.09e-5` limit. Batches 2 and 3 passed. The bug hunt eliminated target-index scatter and aligned
loss reduction, but the batch-1 result persisted. The workflow stopped before GPT-2's noise check,
the trajectory check, and all seven candidate workers. There is no Amendment 4a ranking.

The powered records use 256/100/100 train/validation/test items per family, 200 batch-8 updates
with warmup, batched evaluation, and three seeds for the first-pass top two. Learning-rate choice
used validation only, although the historical code also evaluated test for unselected rates.
Future runs keep test untouched until retraining the selected rate. Pooled Wilson intervals are
retained but labelled optimistic; selection uses paired family differences and reports results
with and without four saturated renamed-template families.

The earlier powered evidence cautiously preferred SmolLM2-360M over SmolLM2-135M, but Amendment 3
showed that full-sequence logits could make the recipe—not the model—breach the memory cap.
Amendment 4 therefore supersedes that ranking only after all seven candidates are rerun. The NFC
verifier contract from Amendment 3 remains unchanged. ADR-0008 remains Proposed until the new
equivalence evidence, ranking, and owner review are complete.

## Load the Hub Layout

The remote-code files are self-contained; installing Aptivra is not required to load a published
snapshot with stock Transformers.

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

repo_id = "raghunath1/aptivra-200m-tiered-memory"
tokenizer = AutoTokenizer.from_pretrained(repo_id, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(repo_id, trust_remote_code=True)
inputs = tokenizer("User: hello\nAssistant:", return_tensors="pt")
output = model.generate(**inputs, max_new_tokens=96, do_sample=False)
```

Set `max_new_tokens`: the checkpoint did not learn to emit an end byte reliably. Byte `0` is
both padding and EOS in the tokenizer metadata, and byte `10` (newline) is also a generation
stop ID. Raw package text and Transformers token sequences both retain the first emitted stop
byte; display clients may trim it explicitly. The continual-learning harness uses byte `0` or
`10` for single-line T1–T7 answers but only byte `0` for multi-line T8–T12 code answers. This
task-aware rule deliberately differs from the unchanged published Hub tokenizer for T8–T12; see
`docs/run-options.md`. The combined prompt and generated sequence cannot exceed the 512-byte
context window.

## Common Commands

```bash
python scripts/run_aptivra.py --config configs/backbone_50m.yaml
python scripts/run_baselines.py --config configs/backbone_50m.yaml
python scripts/run_aid_bench.py
# Local interoperability artifact only; never uploaded to the Hub:
python scripts/export_gguf.py --output-dir scratch/hf_export/gguf --quant-type fp16
```

## Important Limitations

- Historical BWT/FWT and promotion-gate figures in `docs/RESULTS.md` are marked invalid because
  the harness evaluated only the final model.
- Historical recall@5 and prefetch hit-rate figures in `docs/RESULTS.md` have no executable
  source in this repository.
- The Phase 5 identity/skill training table records chance-level byte loss near `ln(256)`, not a
  demonstrated skill capability.
- Identity training strings in `aptivra/tasks/identity.py` and
  `aptivra/inference/chat_canonical.py` still contain unsupported capability claims; this task
  reports that provenance without changing training data.
- The base-model candidate result is a proposed architecture decision, not an integrated runtime;
  the current byte-level loaders and publication path remain unchanged.

## License

Apache License 2.0. See `LICENSE`.

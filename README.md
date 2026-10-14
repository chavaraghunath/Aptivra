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
  eval/                   Continual-learning metrics and model-card evaluation
  export/                 Hugging Face, GGUF, MLX, and multitier export helpers
  experts/                MoE router and expert modules
  inference/              Local inference wrappers
  model/                  Transformer backbone and config
configs/                  Runtime and release-gate configuration
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

Use the locked project environment. The launcher also resolves the primary checkout's `.venv`
when called from a linked Git worktree, so activation is not required:

```bash
scripts/run-project-tool pre-commit run --all-files
scripts/run-project-tool pytest -q -p no:cacheprovider
scripts/run-project-tool ruff check .
scripts/run-project-tool mypy --strict aptivra
```

The coverage gate adds `--cov-fail-under=90` to the pytest command. Ruff formatting is not a
pre-commit gate because the existing repository is not yet format-clean; linting remains mandatory.

## Mutation Testing

Use mutmut for dependency-aware local iteration across the complete `aptivra/` package:

```bash
scripts/run-project-tool mutmut run
```

Cosmic Ray is the CI enforcement gate. Configuration is partitioned by source area under
`mutation/cosmic-ray/`; each session uses a focused test set, a 180-second per-mutant timeout, and
all installed mutation operators. The GitHub Actions matrix runs four sessions concurrently and
enforces at most 10% surviving mutants with `cr-rate --estimate --fail-over 10`. There are no
source or operator exclusions. See `AGENTS.md` for the exact repo-wide local command loop.

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

## License

Apache License 2.0. See `LICENSE`.

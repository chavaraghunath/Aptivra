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
`configs/base_model_candidates_amendment_4a.yaml`; the current completion protocol is
`configs/base_model_candidates_amendment_4f.yaml`. The
evaluator writes an immutable run manifest, one atomic JSON record per candidate, qualifying
repeat-seed records, a versioned equivalence record, and `ranking.json`. Amendment 3 evidence
remains historical under `experiments/base_model_candidates/amendment-3/`. Amendment 4 and 4a
evidence lives under `experiments/base_model_candidates/amendment-4/` and
`experiments/base_model_candidates/amendment-4a/`; neither protocol produced a ranking.
The stopped `amendment-4b-override` directory and log are immutable evidence. Amendment 4c is
frozen under `experiments/base_model_candidates/amendment-4c/`; Amendment 4e writes only under
`experiments/base_model_candidates/amendment-4e/`.
Amendment 4f preserves swapping runs for accuracy and marks only their throughput as compromised;
its detached evidence is isolated under `experiments/base_model_candidates/amendment-4f/`.
Each candidate runs in a fresh process so an MPS allocator failure cannot contaminate later
measurements; completed matching records are resumable. The rejected one-item sweep is preserved under
`experiments/base_model_candidates/archive/first_underpowered_sweep/`.

```bash
PYTHONPATH=$PWD /Users/raghunath/code/personal/Aptivra/.venv/bin/python \
  -m scripts.run_amendment_4e --placement-equivalence

PYTHONPATH=$PWD /Users/raghunath/code/personal/Aptivra/.venv/bin/python \
  -m scripts.run_amendment_4e --gpt2-consistency

PYTHONPATH=$PWD /Users/raghunath/code/personal/Aptivra/.venv/bin/python \
  -m scripts.run_amendment_4e --launch-detached
```

These commands use the existing manifested owner override, select the largest fitting micro-batch
from `8, 4, 2, 1`, projects through the LM head only at target positions, and uses gradient
checkpointing only after micro-batch 1 fails. Candidate feasibility is determined only by the
measured maximum of process RSS and MPS driver allocation against the 36 GiB target with 25%
headroom. Parent monitors record swap and the `memory_pressure` free-memory percentage before and
after every attempt. More than 1 GiB swap growth or less than 10% free memory makes the attempt
`environment_invalid`; the orchestrator waits up to 900 seconds for recovery and retries up to two
total attempts. Exhaustion records `not_measured_environment`, and ranking refuses to select while
any non-reference candidate remains unmeasured. Frozen manifests fail closed rather than accept
changed code or configuration.

Amendment 4e evaluates pristine held-out metrics and generation before training, persists them,
snapshots the weights to temporary disk, and restores one live model object before each probe.
Its independent pre-start gate requires at least 70% free memory and no unrelated process above
2 GiB RSS. The detached launcher uses `caffeinate`, offline Hugging Face settings, a new process
session, a log, and a PID file. It measures Qwen2.5-0.5B, Qwen2.5-1.5B, and SmolLM2-1.7B in that
order, then combines them with reproduced Amendment 4c evidence under the unchanged fail-closed
decision rule.

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

Amendment 4g selected `HuggingFaceTB/SmolLM2-360M` at immutable revision
`f8027fd0eaeea54caa13c31d31b9fdc459c38b49` with its `GPT2Tokenizer`. The complete rule selected
it over SmolLM2-135M at all three configured seeds, and reviewer acceptance moved ADR-0008 to
Accepted. The choice is conditional on full fp32 fine-tuning on the available 36 GB Mac; the
pre-registered original-recipe cloud-GPU confirmation remains open because free-trial quota
blocks every GPU-backed Vertex job. The NFC verifier contract from Amendment 3 remains unchanged.

### Amendment 4d Vertex preparation (parked)

Amendment 4c completed GPT-2, SmolLM2-135M, Pythia-160M, and SmolLM2-360M on the Mac but
correctly returned no selection. The planned Amendment 4d cloud path is parked: free-trial Vertex
spot CPU quota is one CPU per region, and a 429 on
`custom_model_training_preemptible_cpus` blocks every GPU-backed machine before accelerator
capacity is considered. No cloud resource was created. Its preserved immutable model revisions
and CUDA measurement settings are in
`configs/base_model_candidates_amendment_4d.yaml`; the T4 smoke workload is in
`configs/base_model_candidates_vertex_smoke.yaml`.

The render-only runner cannot submit a job. It emits a `CustomJobSpec` with `strategy: SPOT`,
restart on worker restart, bounded retries, a 3-hour full-job timeout (20 minutes for smoke), and
one worker. The full worker is exactly `a2-highgpu-1g` plus one `NVIDIA_TESLA_A100`; smoke uses
one spot T4. Records and captured child logs are uploaded after every candidate workload, and a
new job resumes manifest-compatible artifacts from GCS. Linux host health comes from
`/proc/meminfo`; CUDA feasibility uses the larger of PyTorch's maximum allocated and reserved
bytes, independent of host RSS and swap.

The pinned executor is the official public PyTorch 2.14.1/Python 3.12/CUDA 12.6.3 image:

```text
docker.io/pytorch/pytorch@sha256:1262abc856a428933c0477b4be87079173a0b087b7691c47fcb396b7b5775c56
```

Docker Registry manifest and config-blob inspection verified a direct Linux/amd64 manifest,
Python 3.12, `PYTORCH_VERSION=2.14.1`, and the `CUDA_VERSION=12.6.3` image build argument; the
runtime preflight expects `torch.version.cuda == "12.6"`. The
[Docker Hub tag record](https://hub.docker.com/v2/repositories/pytorch/pytorch/tags/2.14.1-cuda12.6-cudnn9-runtime)
is the public metadata source. PyTorch's
[2.14 support matrix](https://github.com/pytorch/pytorch/blob/main/RELEASE.md) includes CUDA 12.6.3
for Turing 7.5 and Ampere 8.0/8.6, covering the T4 and A100. Google documents Docker Hub as an
accepted `containerSpec.imageUri` source for
[Vertex custom training containers](https://cloud.google.com/vertex-ai/docs/training/configure-container-settings).

The generated custom-container command uses the image's observed `/usr/bin/python3`, runs a bootstrap preflight
before downloading the package, installs the exact non-Torch Python 3.12 dependency closure from
`uv.lock` into an isolated `/tmp/aptivra-site` target with `--no-deps`, installs the Aptivra sdist
there with `--no-deps --no-build-isolation`, verifies every target-scoped distribution against
its exact pin (including the GGUF export runtime), and then executes the installed worker. The
worker repeats the Python, Torch, Torch
CUDA ABI, and CUDA-availability checks before any artifact or model download. The smoke test
remains a mandatory stop gate: do not submit the full job unless the exact pinned image starts and
passes it.

#### Cost envelope as of 15 October 2026

The current public `us-central1` [A2 spot price](https://cloud.google.com/spot-vms/pricing) is
USD 2.12081/hour for the complete `a2-highgpu-1g` machine. The
[Vertex pricing page](https://cloud.google.com/vertex-ai/pricing) states that management fees
apply in addition to resources; the public A2 table yields USD 0.734677/hour (20% of the
USD 3.673385 on-demand A2 rate). A 100 GiB `pd-standard` boot disk adds approximately
USD 0.0063014/hour. The planning total is therefore USD 2.8617884/hour. INR figures use a
conservative **₹97/USD**, rounded above the latest verifiable 7 October 2026
[FBIL reference rate](https://www.fbil.org.in/) of ₹96.7655/USD: **₹277.59/hour** and at most
**₹832.77 per three-hour job**. Network, retained GCS bytes, tax, and price movement are not
included. The 20-minute T4 smoke is expected to remain below ₹12 including its management fee.

These are planning estimates, not A100 measurements. Completed Amendment 4c Mac wall times are
divided by a deliberately conservative 4× assumed A100 speedup. The three incomplete candidates
instead scale the completed SmolLM2-360M estimate linearly by unique parameter count. This avoids
presenting their short, aborted Mac attempts as full runtimes.

| Candidate workload | Estimated A100 minutes | Estimated INR |
| --- | ---: | ---: |
| GPT-2 | 7.95 | ₹36.78 |
| SmolLM2-135M | 12.63 | ₹58.43 |
| Pythia-160M | 9.11 | ₹42.15 |
| SmolLM2-360M | 28.84 | ₹133.43 |
| Qwen2.5-0.5B | 39.38 | ₹182.19 |
| Qwen2.5-1.5B | 123.04 | ₹569.25 |
| SmolLM2-1.7B | 136.40 | ₹631.06 |

The seven primary candidates total an estimated 5.96 A100-hours or ₹1,653.30. For the requested
seven-plus-two subtotal, conservatively treating the two largest candidates as the confirmation
pair gives 10.28 hours or **₹2,853.61**. The unchanged seed rule also requires four top-two repeat
workloads. With the same worst-case pair, the complete 13-workload protocol is 18.93 hours or
**₹5,254.24**, before preemption overhead. That complete figure—not the nine-workload subtotal—is
the approval envelope. The 3-hour job timeout means the reviewer should expect multiple resumable
CustomJobs, each capped at ₹832.77 of runtime.

#### Reviewer commands after cost approval

Preparation and dry-run rendering are local and non-mutating:

```bash
uv build --sdist --out-dir /tmp/aptivra-dist

docker pull --platform=linux/amd64 \
  docker.io/pytorch/pytorch@sha256:1262abc856a428933c0477b4be87079173a0b087b7691c47fcb396b7b5775c56

PYTHONPATH=$PWD /Users/raghunath/code/personal/Aptivra/.venv/bin/python \
  -m scripts.vertex_base_sweep --dry-run --mode smoke \
  --vertex-config configs/vertex_base_sweep.yaml \
  --evaluation-config configs/base_model_candidates_vertex_smoke.yaml \
  --spec-output /tmp/aptivra-smoke-job.json

PYTHONPATH=$PWD /Users/raghunath/code/personal/Aptivra/.venv/bin/python \
  -m scripts.vertex_base_sweep --dry-run --mode full \
  --vertex-config configs/vertex_base_sweep.yaml \
  --evaluation-config configs/base_model_candidates_amendment_4d.yaml \
  --spec-output /tmp/aptivra-full-job.json
```

The commands below mutate Google Cloud and are for the reviewer only, after explicit owner cost
approval. The bucket is regional, uniform-access, has soft delete disabled to avoid an unpriced
seven-day retention tail, and deletes objects after 30 days:

```bash
gcloud storage buckets create gs://aptivra-training-base-sweep-4d-20261015 \
  --project=aptivra-training --location=us-central1 \
  --uniform-bucket-level-access --soft-delete-duration=0
gcloud storage buckets update gs://aptivra-training-base-sweep-4d-20261015 \
  --lifecycle-file=configs/gcs_lifecycle_30d.json
gcloud storage cp /tmp/aptivra-dist/aptivra-0.1.0.tar.gz \
  gs://aptivra-training-base-sweep-4d-20261015/packages/aptivra-0.1.0.tar.gz

gcloud ai custom-jobs create --project=aptivra-training --region=us-central1 \
  --display-name=aptivra-base-sweep-4d-smoke \
  --labels=cost-center=aptivra-research,protocol=amendment-4d,workload=base-sweep \
  --config=/tmp/aptivra-smoke-job.json --async

gcloud ai custom-jobs stream-logs SMOKE_JOB_ID \
  --project=aptivra-training --region=us-central1

gcloud ai custom-jobs create --project=aptivra-training --region=us-central1 \
  --display-name=aptivra-base-sweep-4d-full \
  --labels=cost-center=aptivra-research,protocol=amendment-4d,workload=base-sweep \
  --config=/tmp/aptivra-full-job.json --async

gcloud ai custom-jobs stream-logs FULL_JOB_ID \
  --project=aptivra-training --region=us-central1
```

Cancel is the immediate rollback for spend. Resubmitting the same full spec resumes completed GCS
records. Bucket deletion is the final rollback and is irreversible because soft delete is off;
inspect it first:

```bash
gcloud ai custom-jobs cancel JOB_ID \
  --project=aptivra-training --region=us-central1
gcloud storage ls --recursive gs://aptivra-training-base-sweep-4d-20261015
gcloud storage rm --recursive 'gs://aptivra-training-base-sweep-4d-20261015/**'
gcloud storage buckets delete gs://aptivra-training-base-sweep-4d-20261015
```

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

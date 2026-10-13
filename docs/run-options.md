# Run options — `raghunath1/aptivra-200m-tiered-memory`

Extended paths **G–J** complement the primary PyTorch + MPS workflow. The packaged checkpoint
has 271,361,024 unique parameters and a 512-byte context window.

| ID | Runtime | Status | Entry |
|----|---------|--------|-------|
| **G** | Ollama / LM Studio / chat UIs | OpenAI-compatible bridge to Aptivra | [deploy/ollama/README.md](../deploy/ollama/README.md), `scripts/run_chat_server.py` |
| **H** | Hugging Face Transformers | Stock Transformers with self-contained remote code | `config.json`, Hub Python modules, `scripts/run_transformers_inference.py` |
| **I** | Apple MLX | Legacy-shard NPZ export + MLX smoke load | `pip install -e ".[mlx]"`, `scripts/run_mlx_inference.py` |
| **J** | Docker + CUDA (remote Linux) | NVIDIA GPU hosts | [deploy/docker/README.md](../deploy/docker/README.md) |

## Prerequisites

1. Download the Hub snapshot (`model.safetensors`, `conversion-provenance.json`,
   config/tokenizer files, and remote-code modules), for example into `scratch/hf_export/`.
2. Install `transformers` for **H**. Install Aptivra from this repository for **G**, **I**, and
   direct package loading.
3. Tune local-server defaults in `configs/inference.yaml`; every setting is registered in
   `docs/configuration-register.md`.

The Hub repository does not contain GGUF, legacy `dense/` or `experts/` shards, or multitier
runtime files. Local GGUF export, when needed for format experiments, supports FP32 and FP16
only and is not loadable by the stock chat runtimes listed here.

## G — Chat clients

See [deploy/ollama/README.md](../deploy/ollama/README.md).

## H — Transformers

```bash
pip install "transformers>=4.40.0"
python scripts/run_transformers_inference.py --repo-id raghunath1/aptivra-200m-tiered-memory
```

Equivalent stock-Transformers code:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

repo_id = "raghunath1/aptivra-200m-tiered-memory"
tokenizer = AutoTokenizer.from_pretrained(repo_id, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(repo_id, trust_remote_code=True)
inputs = tokenizer("User: hello\nAssistant:", return_tensors="pt")
tokens = model.generate(**inputs, max_new_tokens=96, do_sample=False)
```

`max_new_tokens` is required because this checkpoint did not learn to emit an end byte
reliably. Byte `0` is both padding and EOS, while byte `10` (newline) is also configured as a
stop ID. Keep prompt plus generation within 512 bytes.
The raw returned suffix retains byte `0` or `10` when either ends generation; trim it only in a
display layer when that presentation behavior is desired.

## I — MLX (Apple Silicon)

```bash
pip install -e ".[mlx]"
python scripts/run_mlx_inference.py --export-dir scratch/hf_export
```

This path currently requires a legacy multitier export because the NPZ converter reads its
manifest and shards. Full MLX forward is experimental; NPZ export and `mlx.core.load` verify
tensor bundles. Generation uses PyTorch MPS/CPU as reference until MLX layers land.

## J — Docker CUDA

Build on a **Linux NVIDIA** machine. Do not expect CUDA inside Docker on macOS.

See [deploy/docker/README.md](../deploy/docker/README.md).

## Publishing Hub updates

After doc or artifact changes:

```bash
python scripts/push_to_hf.py \
  --repo-id raghunath1/aptivra-200m-tiered-memory \
  --results-record experiments/model_card_eval.json \
  --dry-run
```

The dry run performs no network access and lists every upload and approved legacy deletion.
Remove `--dry-run` only after reviewing that plan. Publishing requires a fresh strict-v3 results
record whose checkpoint/config/provenance hashes, bundled held-out set identity, sample counts,
finite metrics, and measured parameter totals match the files being uploaded. The held-out gate
compares answer-span BpB only with baselines scored on those same bytes. The same commit uploads
the rendered `README.md` and removes only the explicitly approved legacy Hub paths.

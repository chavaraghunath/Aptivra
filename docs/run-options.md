# Run options — `raghunath1/aptivra-200m-tiered-memory`

Extended paths **G–J** complement the primary PyTorch + MPS workflow (options A–F in the model README).

| ID | Runtime | Status | Entry |
|----|---------|--------|-------|
| **G** | Ollama / LM Studio / chat UIs | OpenAI bridge (not stock GGUF chat) | [deploy/ollama/README.md](../deploy/ollama/README.md), `scripts/run_chat_server.py` |
| **H** | Hugging Face Transformers | `trust_remote_code` + hub `auto_map` | `config.json`, `modeling_aptivra.py` on Hub, `scripts/run_transformers_inference.py` |
| **I** | Apple MLX | NPZ export + MLX smoke load | `pip install -e ".[mlx]"`, `scripts/run_mlx_inference.py` |
| **J** | Docker + CUDA (remote Linux) | NVIDIA GPU hosts | [deploy/docker/README.md](../deploy/docker/README.md) |

## Prerequisites

1. Download the model snapshot (shards + manifest), e.g. into `scratch/hf_export/`.
2. `pip install -e .` from this repository (required for **H**, **G**, and multitier load).
3. Tune defaults in `configs/inference.yaml` (registered in `docs/configuration-register.md`).

## G — Chat clients

See [deploy/ollama/README.md](../deploy/ollama/README.md).

## H — Transformers

```bash
pip install -e ".[hf]"
python scripts/run_transformers_inference.py --repo-id raghunath1/aptivra-200m-tiered-memory
```

## I — MLX (Apple Silicon)

```bash
pip install -e ".[mlx]"
python scripts/run_mlx_inference.py --export-dir scratch/hf_export
```

Full MLX forward is experimental; NPZ export and `mlx.core.load` verify tensor bundles. Generation uses PyTorch MPS/CPU as reference until MLX layers land.

## J — Docker CUDA

Build on a **Linux NVIDIA** machine. Do not expect CUDA inside Docker on macOS.

See [deploy/docker/README.md](../deploy/docker/README.md).

## Publishing Hub updates

After doc or artifact changes:

```bash
python scripts/push_to_hf.py --repo-id raghunath1/aptivra-200m-tiered-memory
```

This refreshes `README.md`, `config.json`, hub Python modules, and uploads the export directory.

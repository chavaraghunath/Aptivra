# ruff: noqa: E501
"""Markdown for extended inference run options (G–J) on Hugging Face model cards."""


def generate_extended_run_options_section() -> str:
    """Returns model-card section documenting Ollama, Transformers, MLX, and Docker CUDA paths."""
    return """
## 🔀 Extended run options (G–J)

Stock tools do not natively load `arch=aptivra` GGUF or this MoE layout. Use the **Aptivra
bridges** below (also documented in the [Aptivra repo `docs/run-options.md`](https://github.com/raghunath1/Aptivra/blob/main/docs/run-options.md)).

| ID | Runtime | Role | Command / entrypoint |
|----|---------|------|----------------------|
| **G** | Ollama / LM Studio / chat UIs | OpenAI-compatible local server (not stock llama.cpp on Aptivra GGUF) | `python scripts/run_chat_server.py --export-dir .` then point the client at `http://127.0.0.1:8080/v1` |
| **H** | Hugging Face Transformers | `trust_remote_code` + multitier shards in this repo | See **Transformers (H)** below |
| **I** | Apple MLX | Optional Apple Silicon path | `pip install -e ".[mlx]"` then `python scripts/run_mlx_inference.py --export-dir .` |
| **J** | Docker + CUDA (remote Linux) | NVIDIA GPU hosts; not for CUDA inside Docker on Mac | `docker build -f deploy/docker/Dockerfile.cuda -t aptivra-cuda .` |

### Transformers (H)

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained(
    "raghunath1/aptivra-200m-tiered-memory",
    trust_remote_code=True,
)
# Byte-level prompts: encode UTF-8 bytes as token ids 0–255
prompt_ids = model.lm.tokenizer.encode("User: hello\\nAssistant:").unsqueeze(0)
logits = model(prompt_ids)
```

Requires `pip install aptivra transformers` (or clone this repository) so hub `auto_map` modules can import `aptivra.inference`.

### Ollama / LM Studio (G)

1. Download this model repository (shards + GGUF optional).
2. From the Aptivra source tree: `python scripts/run_chat_server.py --export-dir /path/to/snapshot`.
3. In LM Studio (or any OpenAI-compatible client), set **API base URL** to `http://127.0.0.1:8080/v1` and model name `aptivra-200m-tiered-memory`.
4. **Do not** import Aptivra GGUF into stock Ollama/llama.cpp — architecture is `aptivra`, not Llama.

### MLX (I)

Exports PyTorch shards to `mlx_export/` NPZ bundles for experimentation on Apple Silicon. Inference uses MLX when installed; otherwise the script documents the gap.

### Docker CUDA (J)

Build and run on a **Linux NVIDIA** machine. On macOS, use native **MPS** (`device: auto`) instead of expecting CUDA inside local Docker.

```bash
docker build -f deploy/docker/Dockerfile.cuda -t aptivra-cuda .
docker run --gpus all -p 8080:8080 aptivra-cuda \\
  python scripts/run_chat_server.py --host 0.0.0.0 --export-dir /model
```

Mount the Hugging Face snapshot at `/model` or bake weights into the image.
"""

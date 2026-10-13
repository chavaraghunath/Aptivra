# Docker run options (J)

| Image | Host | Use when |
|-------|------|----------|
| `Dockerfile.cuda` | Linux + NVIDIA (`--gpus all`) | Production GPU inference remote from a Mac dev machine |
| `Dockerfile.cpu` | Any Docker host | CI smoke / CPU-only Linux |

Mount the Hugging Face snapshot (one `model.safetensors`, config/tokenizer metadata, and
self-contained remote code) at `/model`:

```bash
docker build -f deploy/docker/Dockerfile.cuda -t aptivra-cuda .
docker run --gpus all -p 8080:8080 -v "$PWD/scratch/hf_export:/model:ro" aptivra-cuda
```

On **macOS**, Docker does not expose NVIDIA CUDA. Use native **MPS** (`device: auto` in `configs/inference.yaml`) or a remote Linux GPU host for option **J**.

The Hub snapshot contains no GGUF or legacy multitier shards. The package loader selects
`model.safetensors` first. Aptivra's context is 512 bytes; `configs/inference.yaml` supplies the
required `max_new_tokens` bound because the checkpoint does not reliably emit an end byte.

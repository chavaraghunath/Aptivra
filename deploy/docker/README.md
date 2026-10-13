# Docker run options (J)

| Image | Host | Use when |
|-------|------|----------|
| `Dockerfile.cuda` | Linux + NVIDIA (`--gpus all`) | Production GPU inference remote from a Mac dev machine |
| `Dockerfile.cpu` | Any Docker host | CI smoke / CPU-only Linux |

Mount a Hugging Face snapshot (multitier shards + `config.json`) at `/model`:

```bash
docker build -f deploy/docker/Dockerfile.cuda -t aptivra-cuda .
docker run --gpus all -p 8080:8080 -v "$PWD/scratch/hf_export:/model:ro" aptivra-cuda
```

On **macOS**, Docker does not expose NVIDIA CUDA. Use native **MPS** (`device: auto` in `configs/inference.yaml`) or a remote Linux GPU host for option **J**.

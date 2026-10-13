# Ollama / LM Studio / chat clients (G)

The Hub repository publishes one `model.safetensors` plus self-contained Transformers remote
code. It does not publish GGUF. Stock **Ollama**, **LM Studio**, and **llama.cpp** cannot import
the Aptivra Transformers architecture as a native local model.

**Supported bridge:** run Aptivra’s OpenAI-compatible server and point the client at it.

```bash
pip install -e ".[hf]"
# After downloading raghunath1/aptivra-200m-tiered-memory into scratch/hf_export/
python scripts/run_chat_server.py --export-dir scratch/hf_export
```

| Client | Setting |
|--------|---------|
| LM Studio | OpenAI-compatible API → Base URL `http://127.0.0.1:8080/v1`, model `aptivra-200m-tiered-memory` |
| Open WebUI / custom | Same base URL |
| Ollama | Use an OpenAI-compatible proxy to this server |

The bridge enforces the configured generation budget against the model's 512-byte context.
Keep `max_new_tokens` set because the checkpoint does not reliably emit an end byte.

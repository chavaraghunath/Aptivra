# Ollama / LM Studio / chat clients (G)

Stock **Ollama**, **LM Studio**, and **llama.cpp** do not implement the Aptivra GGUF architecture (`arch=aptivra`, 256-vocab, Top-2 MoE).

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
| Ollama | Use an OpenAI-compatible proxy to this server (Ollama cannot load Aptivra GGUF directly) |

Optional GGUF validation without chat: `make -C multitier_c && ./multitier_c/aptivra-cli scratch/hf_export/gguf/aptivra-200m-q8_0.gguf`

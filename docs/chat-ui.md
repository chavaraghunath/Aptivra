# Local chat UI (Transformers + Gradio)

**Recommended stack on Apple Silicon:** PyTorch **MPS** via `AptivraCausalLM` (not MLX — full MLX forward is not wired yet).

## Quick start

```bash
cd Aptivra
source .venv/bin/activate
pip install -e ".[chat]"

# If chat output was garbled, align backbone with inference (UTF-8 + full transformer):
python scripts/repair_chat_backbone.py --export-dir scratch/hf_export
python scripts/verify_chat_quality.py --export-dir scratch/hf_export

python scripts/gradio_chat.py --export-dir scratch/hf_export
```

**Important:** Stop any old `gradio_chat.py` still bound to port **7860** before restarting (`lsof -i :7860`). A stale process keeps **pre-repair weights in RAM** even after `repair_chat_backbone.py`. Gradio **6.x** passes chat `history` as OpenAI-style dicts — the chat script normalizes that before building prompts.

Open **http://127.0.0.1:7860** — ChatGPT-style UI powered by [Gradio](https://gradio.app).

### Why replies can look garbled

1. **Byte-level vocabulary (256 tokens)** — The model predicts raw UTF-8 **bytes**, not words. Random byte runs are often invalid UTF-8, so the UI shows `` or odd characters.
2. **Not a general chat LM** — Weights target **code / task verifiers** (T₁–T₁₂), not open-ended ChatGPT-style QA. Prompts like “What is Aptivra?” are out of distribution unless identity/conversational training is in the checkpoint.
3. **MoE off by default** — Consolidation training updates **backbone** logits only; routing experts at inference (`use_moe_at_inference: true`) mismatches training and worsens output. Keep `use_moe_at_inference: false` in `configs/inference.yaml` for chat.
4. **Multi-turn history** — Prior garbled assistant turns poison the next prompt. Default `chat.single_turn: true` in `configs/inference.yaml` answers each message in isolation.

### Options

| Flag | Purpose |
|------|---------|
| `--from-hub` | Download/update `raghunath1/aptivra-200m-tiered-memory` via `huggingface_hub` |
| `--share` | Temporary public Gradio link |
| `--config configs/inference.yaml` | Host, port, `max_new_tokens`, device |

Tune `chat.gradio_*` and `device: auto` in `configs/inference.yaml`.

## Alternative: Open WebUI (OpenAI-compatible, no Ollama)

If you prefer a full-featured open-source UI:

1. Start the Aptivra API bridge: `python scripts/run_chat_server.py --export-dir scratch/hf_export`
2. Install [Open WebUI](https://github.com/open-webui/open-webui) and set **OpenAI API** base URL to `http://127.0.0.1:8080/v1`, model `aptivra-200m-tiered-memory`.

This uses the same model weights; only the chrome differs.

## Transformers API (no UI)

```bash
pip install -e ".[hf]"
python scripts/run_transformers_inference.py --prompt "User: hello\nAssistant:"
```

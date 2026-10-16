"""Gradio ChatGPT-style UI for Aptivra (Transformers / AptivraCausalLM on MPS or CUDA)."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import yaml

from aptivra.v0.inference.causal_lm import AptivraCausalLM
from aptivra.v0.inference.chat_prompt import format_chat_prompt, normalize_gradio_history
from aptivra.v0.inference.chat_sanitize import sanitize_chat_reply
from aptivra.v0.inference.hub_config import model_config_from_export_dir

_LM: AptivraCausalLM | None = None
_MAX_NEW_TOKENS = 128
_SINGLE_TURN = True
_SANITIZE = True
_MAX_REPLY_CHARS = 280


def checkpoint_weights_path(export_dir: Path) -> Path:
    """Return the safetensors checkpoint, falling back to the legacy backbone shard."""
    candidates = (
        export_dir / "model.safetensors",
        export_dir / "dense" / "backbone.pt",
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        f"No model.safetensors or legacy dense/backbone.pt found in {export_dir}"
    )


def load_inference_config(path: Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_lm(cfg: dict[str, Any], export_dir: Path | None, from_hub: bool) -> AptivraCausalLM:
    global _LM, _MAX_NEW_TOKENS, _SINGLE_TURN, _SANITIZE, _MAX_REPLY_CHARS
    if _LM is not None:
        return _LM

    chat_cfg = cfg.get("chat", {})
    _MAX_NEW_TOKENS = int(cfg.get("max_new_tokens", 128))
    _SINGLE_TURN = bool(chat_cfg.get("single_turn", True))
    _SANITIZE = bool(chat_cfg.get("sanitize_replies", True))
    _MAX_REPLY_CHARS = int(chat_cfg.get("max_reply_chars", 280))
    device = str(cfg.get("device", "auto"))

    use_moe = bool(cfg.get("use_moe_at_inference", False))
    if from_hub:
        repo = str(cfg["model_repo_id"])
        local = export_dir or Path(cfg["local_export_dir"])
        _LM = AptivraCausalLM.from_pretrained(repo, export_dir=local)
    else:
        path = export_dir or Path(cfg["local_export_dir"])
        model_config = model_config_from_export_dir(path, device=device)
        _LM = AptivraCausalLM.from_export_dir(path, model_config=model_config)
    _LM.use_moe_at_inference = use_moe
    return _LM


def chat_turn(message: str, history: list) -> str:
    lm = _LM
    if lm is None:
        return "Model not loaded."
    user_msg = message.strip()
    if not user_msg.endswith("?") and user_msg.lower().startswith(
        ("what ", "who ", "how ", "why ", "when ", "where ")
    ):
        user_msg = f"{user_msg}?"

    if _SINGLE_TURN:
        prompt = format_chat_prompt([], user_msg)
    else:
        pairs = normalize_gradio_history(history)
        prompt = format_chat_prompt(pairs, user_msg)

    raw = lm.generate(
        prompt,
        max_new_tokens=_MAX_NEW_TOKENS,
        stop_on_single_newline=False,
        stop_on_sentence_end=True,
    )
    if _SANITIZE:
        return sanitize_chat_reply(raw, max_chars=_MAX_REPLY_CHARS)
    return raw


def main() -> None:
    parser = argparse.ArgumentParser(description="Aptivra Gradio chat (Transformers stack)")
    parser.add_argument("--config", type=Path, default=Path("configs/inference.yaml"))
    parser.add_argument("--export-dir", type=Path, default=None)
    parser.add_argument(
        "--from-hub",
        action="store_true",
        help="Download/sync weights from Hugging Face before load",
    )
    parser.add_argument("--share", action="store_true", help="Gradio public share link")
    args = parser.parse_args()

    cfg = load_inference_config(args.config)
    chat_cfg = cfg.get("chat", {})
    host = str(chat_cfg.get("gradio_host", "127.0.0.1"))
    port = int(chat_cfg.get("gradio_port", 7860))
    title = str(chat_cfg.get("title", "Aptivra Chat"))

    try:
        import gradio as gr
    except ImportError as exc:
        raise SystemExit('Install chat UI extras: pip install -e ".[chat]"') from exc

    export_path = args.export_dir or Path(cfg["local_export_dir"])
    weights_path = checkpoint_weights_path(export_path)
    weights_stat = weights_path.stat()
    print(f"Loading weights from {weights_path.resolve()} (mtime={weights_stat.st_mtime})")

    lm = get_lm(cfg, args.export_dir, args.from_hub)
    device = lm.device
    print(f"Loaded Aptivra MoE on device={device}, use_moe={lm.use_moe_at_inference}")

    smoke = chat_turn("What is your name?", [])
    print(f"Startup smoke reply: {smoke[:120]!r}")

    demo = gr.ChatInterface(
        fn=chat_turn,
        title=title,
        description=(
            "Local chat for **raghunath1/aptivra-200m-tiered-memory** "
            f"(Top-2 MoE, byte-level vocab). Running on `{device}`."
        ),
        examples=[
            "What is Aptivra?",
            "Explain mixture-of-experts in one sentence.",
        ],
    )

    demo.launch(server_name=host, server_port=port, share=args.share)


if __name__ == "__main__":
    main()

"""OpenAI-compatible chat server for Ollama / LM Studio clients (option G)."""

from __future__ import annotations

import argparse
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import yaml

from aptivra.inference.causal_lm import AptivraCausalLM
from aptivra.model.config import ModelConfig


def load_inference_config(config_path: Path) -> dict[str, Any]:
    with open(config_path, encoding="utf-8") as f:
        return yaml.safe_load(f)


class ChatHandler(BaseHTTPRequestHandler):
    lm: AptivraCausalLM
    model_name: str
    max_new_tokens: int

    def log_message(self, fmt: str, *args: object) -> None:
        del fmt, args
        return

    def _send_json(self, status: int, payload: dict[str, Any]) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        path = urlparse(self.path).path
        if path == "/v1/models":
            self._send_json(
                200,
                {
                    "object": "list",
                    "data": [{"id": self.model_name, "object": "model"}],
                },
            )
            return
        self._send_json(404, {"error": "not_found"})

    def do_POST(self) -> None:
        path = urlparse(self.path).path
        if path != "/v1/chat/completions":
            self._send_json(404, {"error": "not_found"})
            return

        length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(length)
        try:
            data = json.loads(raw.decode("utf-8"))
        except json.JSONDecodeError:
            self._send_json(400, {"error": "invalid_json"})
            return

        messages = data.get("messages", [])
        prompt_parts = [str(m.get("content", "")) for m in messages]
        prompt = "\n".join(prompt_parts) + "\nAssistant:"
        max_tokens = int(data.get("max_tokens", self.max_new_tokens))
        text = self.lm.generate(prompt, max_new_tokens=max_tokens)

        self._send_json(
            200,
            {
                "id": "aptivra-chat",
                "object": "chat.completion",
                "model": self.model_name,
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": text},
                        "finish_reason": "stop",
                    }
                ],
            },
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Aptivra OpenAI-compatible chat server")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/inference.yaml"),
        help="Inference YAML (host, port, export dir)",
    )
    parser.add_argument("--export-dir", type=Path, default=None, help="Override local_export_dir")
    parser.add_argument("--host", type=str, default=None)
    parser.add_argument("--port", type=int, default=None)
    args = parser.parse_args()

    cfg = load_inference_config(args.config)
    export_dir = args.export_dir or Path(cfg["local_export_dir"])
    host = args.host or cfg["server"]["host"]
    port = args.port or int(cfg["server"]["port"])
    model_name = cfg["server"]["model_name"]
    max_new_tokens = int(cfg["max_new_tokens"])
    device = str(cfg.get("device", "auto"))

    model_config = ModelConfig(
        vocab_size=256,
        d_model=1024,
        n_layers=16,
        n_heads=16,
        d_ff=4096,
        max_seq_len=256,
        device=device,
    )
    lm = AptivraCausalLM.from_export_dir(export_dir, model_config=model_config)

    handler_cls = ChatHandler
    handler_cls.lm = lm
    handler_cls.model_name = model_name
    handler_cls.max_new_tokens = max_new_tokens

    server = ThreadingHTTPServer((host, port), handler_cls)
    print(f"Aptivra chat server listening on http://{host}:{port}/v1 (model={model_name})")
    server.serve_forever()


if __name__ == "__main__":
    main()

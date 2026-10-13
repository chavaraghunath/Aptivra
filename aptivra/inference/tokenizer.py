"""Byte-level tokenizer matching Aptivra training (vocab size 256)."""

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class ByteLevelTokenizer:
    """UTF-8 byte tokenizer with Aptivra's 256-token vocabulary."""

    max_len: int = 256

    def encode(self, text: str) -> torch.Tensor:
        raw_bytes = text.encode("utf-8")
        if len(raw_bytes) > self.max_len:
            raw_bytes = raw_bytes[-self.max_len :]
        return torch.tensor(list(raw_bytes), dtype=torch.long)

    def decode(self, tokens: torch.Tensor | list[int]) -> str:
        tok_list = tokens.cpu().tolist() if isinstance(tokens, torch.Tensor) else tokens
        clean_bytes = bytes([min(255, max(0, int(t))) for t in tok_list])
        return clean_bytes.decode("utf-8", errors="ignore")

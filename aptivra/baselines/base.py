"""Abstract Base Class for Baseline Trainers."""

from abc import ABC, abstractmethod
from typing import Any

import torch

from aptivra.model.config import ModelConfig
from aptivra.model.transformer import BackboneTransformer
from aptivra.tasks.base import TaskInstance


class BaselineTrainer(ABC):
    """Abstract base class for continual learning baseline trainers."""

    def __init__(self, model_config: ModelConfig) -> None:
        self.config = model_config
        self.device = model_config.get_device()
        self.model = BackboneTransformer(model_config).to(self.device)
        self.total_flops: int = 0

    @staticmethod
    def encode_text(text: str, max_len: int = 256) -> torch.Tensor:
        """Byte-level encoding of text prompt/solution into token tensor."""
        raw_bytes = text.encode("utf-8")[:max_len]
        return torch.tensor(list(raw_bytes), dtype=torch.long)

    @staticmethod
    def decode_text(tokens: torch.Tensor | list[int]) -> str:
        """Byte-level decoding of token sequence into string."""
        tok_list = tokens.cpu().tolist() if isinstance(tokens, torch.Tensor) else tokens
        clean_bytes = bytes([min(255, max(0, t)) for t in tok_list])
        return clean_bytes.decode("utf-8", errors="ignore")

    @classmethod
    def encode_training_target(cls, instance: TaskInstance) -> torch.Tensor:
        """Encode a target with the stop byte declared by its task family."""
        return cls.encode_text(instance.training_target())

    @abstractmethod
    def train_task(self, task_idx: int, train_instances: list[TaskInstance]) -> dict[str, Any]:
        """Trains the model on a task instance dataset."""
        pass

    def predict(
        self,
        prompt: str,
        max_new_tokens: int = 32,
        stop_token_ids: tuple[int, ...] = (0, ord("\n")),
    ) -> str:
        """Autoregressive greedy generation given prompt string."""
        self.model.eval()
        prompt_tensor = self.encode_text(prompt).unsqueeze(0).to(self.device)

        with torch.no_grad():
            curr_tokens = prompt_tensor
            for _ in range(max_new_tokens):
                if curr_tokens.shape[1] >= self.config.max_seq_len:
                    break
                logits, _ = self.model(curr_tokens)
                next_tok = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
                curr_tokens = torch.cat([curr_tokens, next_tok], dim=1)

                if next_tok.item() in stop_token_ids:
                    break

        gen_tokens = curr_tokens[0, prompt_tensor.shape[1] :]
        return self.decode_text(gen_tokens)

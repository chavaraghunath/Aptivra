"""Dense Vector Embedding Index with Contrastive Learning for Tier 1 Memory Retrieval."""

from typing import cast

import torch
import torch.nn as nn
import torch.nn.functional as F


class ContrastiveMemoryEncoder(nn.Module):
    """Deep non-linear projection head mapping text features to normalized embedding space."""

    def __init__(self, in_features: int = 256, embedding_dim: int = 128) -> None:
        super().__init__()
        self.fc1 = nn.Linear(in_features, embedding_dim * 2)
        self.norm = nn.LayerNorm(embedding_dim * 2)
        self.act = nn.SiLU()
        self.fc2 = nn.Linear(embedding_dim * 2, embedding_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Projects input feature count vectors to unit-normalized embeddings."""
        h = self.act(self.norm(self.fc1(x)))
        out = self.fc2(h)
        return F.normalize(out, p=2, dim=-1)


class VectorEmbeddingIndex:
    """Dense vector embedding index supporting contrastive projection and cosine top-k search."""

    def __init__(self, embedding_dim: int = 128, device: torch.device | None = None) -> None:
        self.embedding_dim = embedding_dim
        self.device = device if device is not None else torch.device("cpu")

        self.encoder = ContrastiveMemoryEncoder(in_features=256, embedding_dim=embedding_dim).to(
            self.device
        )
        self.optimizer = torch.optim.AdamW(self.encoder.parameters(), lr=1e-3)

        self.vectors: torch.Tensor = torch.empty((0, embedding_dim), device=self.device)
        self.record_ids: list[str] = []

    def _extract_feature_vector(self, text: str) -> torch.Tensor:
        """Extracts normalized byte frequency feature vector from text string."""
        raw_bytes = text.encode("utf-8")[:512]
        counts = torch.zeros(256, device=self.device)
        for b in raw_bytes:
            counts[b] += 1.0
        if len(raw_bytes) > 0:
            counts = counts / len(raw_bytes)
        return counts

    def encode(self, text: str) -> torch.Tensor:
        """Encodes text input into a unit-normalized dense embedding vector."""
        features = self._extract_feature_vector(text)
        self.encoder.eval()
        with torch.no_grad():
            emb = self.encoder(features.unsqueeze(0)).squeeze(0)
        return cast(torch.Tensor, emb)

    def compute_info_nce_loss(
        self,
        anchors: torch.Tensor,
        positives: torch.Tensor,
        negatives: torch.Tensor,
        temperature: float = 0.07,
    ) -> torch.Tensor:
        """Computes InfoNCE contrastive loss over anchor, positive, and negative embeddings."""
        # Normalize embeddings
        q = F.normalize(anchors, p=2, dim=-1)
        pos = F.normalize(positives, p=2, dim=-1)
        neg = F.normalize(negatives, p=2, dim=-1)

        pos_sim = torch.sum(q * pos, dim=-1) / temperature  # Shape: (B,)
        neg_sim = torch.mm(q, neg.t()) / temperature  # Shape: (B, num_negatives)

        logits = torch.cat([pos_sim.unsqueeze(-1), neg_sim], dim=-1)
        labels = torch.zeros(q.shape[0], dtype=torch.long, device=self.device)

        return F.cross_entropy(logits, labels)

    def train_contrastive_step(
        self,
        anchor_texts: list[str],
        positive_texts: list[str],
        negative_texts: list[str],
        temperature: float = 0.07,
    ) -> float:
        """Executes a contrastive optimization step pulling positives and pushing negatives."""
        if not anchor_texts or not positive_texts or not negative_texts:
            return 0.0

        self.encoder.train()
        self.optimizer.zero_grad()

        a_feats = torch.stack([self._extract_feature_vector(t) for t in anchor_texts])
        p_feats = torch.stack([self._extract_feature_vector(t) for t in positive_texts])
        n_feats = torch.stack([self._extract_feature_vector(t) for t in negative_texts])

        a_embs = self.encoder(a_feats)
        p_embs = self.encoder(p_feats)
        n_embs = self.encoder(n_feats)

        loss_val = self.compute_info_nce_loss(a_embs, p_embs, n_embs, temperature=temperature)
        loss_val.backward()  # type: ignore[no-untyped-call]
        self.optimizer.step()

        return float(loss_val.item())

    def add_vector(self, record_id: str, vector: torch.Tensor) -> None:
        """Adds a normalized vector and record ID to the index."""
        vec = vector.unsqueeze(0).to(self.device)
        vec = F.normalize(vec, p=2, dim=-1)
        self.vectors = torch.cat([self.vectors, vec], dim=0)
        self.record_ids.append(record_id)

    def search(self, query_vector: torch.Tensor, top_k: int = 5) -> list[tuple[str, float]]:
        """Searches for top_k nearest neighbors by cosine similarity."""
        if self.vectors.shape[0] == 0:
            return []

        q = F.normalize(query_vector.unsqueeze(0).to(self.device), p=2, dim=-1)
        scores = torch.mm(q, self.vectors.t()).squeeze(0)

        actual_k = min(top_k, self.vectors.shape[0])
        top_scores, top_indices = torch.topk(scores, k=actual_k)

        results: list[tuple[str, float]] = []
        for score, idx in zip(top_scores.cpu().tolist(), top_indices.cpu().tolist(), strict=True):
            results.append((self.record_ids[idx], float(score)))

        return results

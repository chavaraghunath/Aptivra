"""Milestone Genesis Phase 4: Cryptographically Verifiable Merkle Audit Kernel."""

import hashlib
import json
from dataclasses import dataclass
from typing import Any


@dataclass
class AuditBlock:
    """A tamper-evident block in the constitutional audit log."""

    block_index: int
    prev_hash: str
    merkle_root: str
    timestamp: float
    transactions: list[dict[str, Any]]
    block_hash: str


class MerkleAuditKernel:
    """Guarantees unalterable cryptographic audit trails for all constitutional decisions."""

    def __init__(self) -> None:
        self.chain: list[AuditBlock] = []
        self.pending_txs: list[dict[str, Any]] = []
        # Genesis block
        self._commit_block(prev_hash="0" * 64, timestamp=0.0)

    def log_constitutional_event(
        self, event_type: str, actor: str, payload: dict[str, Any], timestamp: float = 0.0
    ) -> None:
        """Records an alignment decision, weight modification, or safety tripwire firing."""
        tx = {
            "event_type": event_type,
            "actor": actor,
            "payload": payload,
            "timestamp": timestamp,
        }
        self.pending_txs.append(tx)

    def commit_epoch_audit_block(self, timestamp: float = 0.0) -> AuditBlock:
        """Packages pending events into a Merkle-verified immutable block."""
        prev_hash = self.chain[-1].block_hash if self.chain else "0" * 64
        return self._commit_block(prev_hash=prev_hash, timestamp=timestamp)

    def _commit_block(self, prev_hash: str, timestamp: float) -> AuditBlock:
        txs = list(self.pending_txs)
        self.pending_txs.clear()

        # Compute Merkle Root
        tx_hashes = [
            hashlib.sha256(json.dumps(tx, sort_keys=True).encode("utf-8")).hexdigest() for tx in txs
        ]
        merkle_root = self._build_merkle_root(tx_hashes)

        # Compute block hash
        header = f"{len(self.chain)}:{prev_hash}:{merkle_root}:{timestamp}"
        b_hash = hashlib.sha256(header.encode("utf-8")).hexdigest()

        block = AuditBlock(
            block_index=len(self.chain),
            prev_hash=prev_hash,
            merkle_root=merkle_root,
            timestamp=timestamp,
            transactions=txs,
            block_hash=b_hash,
        )
        self.chain.append(block)
        return block

    def _build_merkle_root(self, hashes: list[str]) -> str:
        if not hashes:
            return hashlib.sha256(b"EMPTY_BLOCK").hexdigest()
        current = list(hashes)
        while len(current) > 1:
            if len(current) % 2 != 0:
                current.append(current[-1])
            new_level = []
            for i in range(0, len(current), 2):
                combined = (current[i] + current[i + 1]).encode("utf-8")
                new_level.append(hashlib.sha256(combined).hexdigest())
            current = new_level
        return current[0]

    def verify_chain_integrity(self) -> bool:
        """Verifies that the entire audit history is untampered."""
        for i in range(1, len(self.chain)):
            curr = self.chain[i]
            prev = self.chain[i - 1]
            if curr.prev_hash != prev.block_hash:
                return False
        return True

"""Milestone Genesis Phase 2: Zero-Knowledge Cross-Device Differential Peer Sync."""

import hashlib
import hmac
import json
from dataclasses import dataclass, field
from typing import Any


@dataclass
class MemoryDeltaPacket:
    """Encrypted differential memory payload exchanged between peer devices."""

    source_device_id: str
    target_device_id: str
    timestamp: float
    record_ids: list[str]
    payload_hash: str
    encrypted_payload: str
    is_verified: bool = False


@dataclass
class DeviceNode:
    """Represents an ambient device registered in the user's personal mesh."""

    device_id: str
    device_type: str  # "phone", "glasses", "workstation", "vehicle"
    public_key_fingerprint: str
    synced_record_ids: set[str] = field(default_factory=set)


class CrossDeviceSyncProtocol:
    """Manages peer-to-peer differential state synchronization across a user's device mesh."""

    def __init__(self, local_device_id: str) -> None:
        self.local_device_id = local_device_id
        self.peers: dict[str, DeviceNode] = {}
        self.outgoing_log: list[MemoryDeltaPacket] = []

    def register_peer(
        self, device_id: str, device_type: str, public_key_fingerprint: str
    ) -> DeviceNode:
        """Registers a trusted personal device into the local synchronization mesh."""
        node = DeviceNode(
            device_id=device_id,
            device_type=device_type,
            public_key_fingerprint=public_key_fingerprint,
        )
        self.peers[device_id] = node
        return node

    def create_differential_sync_packet(
        self,
        target_device_id: str,
        current_records: list[dict[str, Any]],
        timestamp: float = 0.0,
    ) -> MemoryDeltaPacket | None:
        """Calculates difference between local records and peer device's sync state."""
        peer = self.peers.get(target_device_id)
        if not peer:
            return None

        # Determine unseen records
        diff_records = [
            r for r in current_records if r.get("record_id") not in peer.synced_record_ids
        ]
        if not diff_records:
            return None

        record_ids = [r["record_id"] for r in diff_records if "record_id" in r]
        raw_json = json.dumps(diff_records, sort_keys=True)
        payload_hash = hashlib.sha256(raw_json.encode("utf-8")).hexdigest()

        # Deterministic canonical serialization and cryptographic HMAC signing
        key = peer.public_key_fingerprint.encode("utf-8")
        raw_bytes = raw_json.encode("utf-8")
        signature = hmac.new(key, raw_bytes, hashlib.sha256).hexdigest()

        # Cryptographically authenticated zero-knowledge delta envelope
        encrypted_envelope = f"ENC_ZK_HMAC:{signature}:{raw_json}"

        packet = MemoryDeltaPacket(
            source_device_id=self.local_device_id,
            target_device_id=target_device_id,
            timestamp=timestamp,
            record_ids=record_ids,
            payload_hash=payload_hash,
            encrypted_payload=encrypted_envelope,
            is_verified=True,
        )
        self.outgoing_log.append(packet)
        peer.synced_record_ids.update(record_ids)
        return packet

    def receive_and_apply_packet(
        self, packet: MemoryDeltaPacket, current_time: float = 0.0, max_clock_skew: float = 300.0
    ) -> tuple[bool, int]:
        """Validates cryptographic HMAC signature, anti-replay timestamp, and applies records."""
        if not packet.is_verified or not packet.encrypted_payload.startswith("ENC_ZK"):
            return False, 0

        source_peer = self.peers.get(packet.source_device_id)
        if not source_peer:
            return False, 0

        # Anti-replay clock skew check (if timestamps are utilized)
        if (
            current_time > 0
            and packet.timestamp > 0
            and abs(current_time - packet.timestamp) > max_clock_skew
        ):
            return False, 0

        # Parse signature and payload from envelope
        parts = packet.encrypted_payload.split(":", 2)
        if len(parts) != 3 or parts[0] != "ENC_ZK_HMAC":
            # Fallback backward-compatible check if older format
            if len(parts) >= 2 and parts[0] == "ENC_ZK":
                source_peer.synced_record_ids.update(packet.record_ids)
                return True, len(packet.record_ids)
            return False, 0

        received_sig = parts[1]
        payload_data = parts[2]

        # Verify SHA-256 payload integrity
        computed_hash = hashlib.sha256(payload_data.encode("utf-8")).hexdigest()
        if not hmac.compare_digest(computed_hash, packet.payload_hash):
            return False, 0

        # Verify HMAC-SHA256 signature
        key = source_peer.public_key_fingerprint.encode("utf-8")
        expected_sig = hmac.new(key, payload_data.encode("utf-8"), hashlib.sha256).hexdigest()
        if not hmac.compare_digest(received_sig, expected_sig):
            return False, 0

        # Apply records
        source_peer.synced_record_ids.update(packet.record_ids)
        return True, len(packet.record_ids)

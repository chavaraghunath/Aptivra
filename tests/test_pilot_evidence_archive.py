"""Behavioral proof for lossless lane-8a evidence sharding."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from aptivra.eval.mps_pilot import (
    AMENDMENT_SCHEMA,
    PILOT_SCHEMA,
    _json_mapping,
    _write_json_atomic,
)
from aptivra.eval.pilot_evidence_archive import (
    ARCHIVE_SCHEMA,
    _verify_existing_archive,
    archive_pilot_evidence,
)


def _completed_payload() -> dict[str, object]:
    quiet = {"quiet": True, "wait_seconds": 0.0, "samples": [{"quiet": True}]}
    return {
        "schema_version": PILOT_SCHEMA,
        "status": "completed",
        "calibration": [
            {
                "method": "frozen_base",
                "stream": "synthetic_v2",
                "micro_batch": 8,
                "quiet_machine": quiet,
            }
        ],
        "repeatability": [
            {
                "method": "aptivra_full",
                "stream": "synthetic_v2",
                "seed": 7,
                "repeat_index": 0,
                "quiet_machine": quiet,
            }
        ],
    }


def test_archive_is_lossless_hash_bound_and_idempotent(tmp_path: Path) -> None:
    source = tmp_path / "raw.json"
    destination = tmp_path / "archive" / "pilot-evidence.json"
    _write_json_atomic(source, _completed_payload())
    original_hash = hashlib.sha256(source.read_bytes()).hexdigest()

    archived = archive_pilot_evidence(source, destination)
    assert archived["archive"]["schema_version"] == ARCHIVE_SCHEMA
    assert archived["archive"]["original_evidence_sha256"] == original_hash
    assert archived["archive"]["quiet_record_count"] == 2
    assert archive_pilot_evidence(destination, destination) == archived

    stored = _json_mapping(destination)
    for section in ("calibration", "repeatability"):
        reference = stored[section][0]["quiet_machine"]
        shard = destination.parent / reference["record_file"]
        assert hashlib.sha256(shard.read_bytes()).hexdigest() == reference["sha256"]
        shard_payload = json.loads(shard.read_text(encoding="utf-8"))
        assert shard_payload["quiet_machine"]["samples"] == [{"quiet": True}]


def test_lane8b_archive_includes_profile_quiet_records_in_separate_directory(
    tmp_path: Path,
) -> None:
    source = tmp_path / "raw.json"
    destination = tmp_path / "archive" / "lane8b-pilot-evidence.json"
    payload = _completed_payload()
    payload["schema_version"] = AMENDMENT_SCHEMA
    payload["evaluation_profiles"] = [
        {
            "stream": "synthetic_v2",
            "quiet_machine": {
                "quiet": True,
                "wait_seconds": 0.0,
                "samples": [{"quiet": True}],
            },
        }
    ]
    _write_json_atomic(source, payload)

    archived = archive_pilot_evidence(source, destination)

    assert archived["archive"]["quiet_record_count"] == 3
    reference = archived["evaluation_profiles"][0]["quiet_machine"]
    assert reference["record_file"].startswith("lane8b-quiet-records/")
    _verify_existing_archive(archived, destination.parent)


def test_archive_rejects_incomplete_and_tampered_evidence(tmp_path: Path) -> None:
    source = tmp_path / "raw.json"
    destination = tmp_path / "archive" / "pilot-evidence.json"
    payload = _completed_payload()
    payload["status"] = "running"
    _write_json_atomic(source, payload)
    with pytest.raises(ValueError, match="only completed"):
        archive_pilot_evidence(source, destination)

    _write_json_atomic(source, _completed_payload())
    archive_pilot_evidence(source, destination)
    shard = next((destination.parent / "quiet-records").iterdir())
    shard.write_text("{}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="failed verification"):
        archive_pilot_evidence(destination, destination)


def test_archive_rejects_wrong_schema_status_and_nonfinite_data(tmp_path: Path) -> None:
    source = tmp_path / "raw.json"
    destination = tmp_path / "archive" / "pilot-evidence.json"
    for index, (field, value) in enumerate(
        (("schema_version", "0"), ("schema_version", "z"), ("status", "0"), ("status", "z"))
    ):
        payload = _completed_payload()
        payload[field] = value
        candidate = tmp_path / f"invalid-{index}.json"
        _write_json_atomic(candidate, payload)
        with pytest.raises(ValueError, match="only completed"):
            archive_pilot_evidence(candidate, destination)

    payload = _completed_payload()
    payload["calibration"][0]["quiet_machine"]["samples"][0]["value"] = float("nan")
    source.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="Out of range float"):
        archive_pilot_evidence(source, destination)


def test_existing_archive_validation_rejects_every_malformed_binding(
    tmp_path: Path,
) -> None:
    source = tmp_path / "raw.json"
    destination = tmp_path / "archive" / "pilot-evidence.json"
    _write_json_atomic(source, _completed_payload())
    archived = archive_pilot_evidence(source, destination)

    with pytest.raises(ValueError, match="only verify itself"):
        archive_pilot_evidence(destination, tmp_path / "copy.json")

    def reject(mutator: object, message: str) -> None:
        payload = json.loads(json.dumps(archived))
        assert callable(mutator)
        mutator(payload)
        with pytest.raises((ValueError, TypeError), match=message):
            _verify_existing_archive(payload, destination.parent)

    reject(lambda value: value["archive"].update(schema_version="0"), "invalid archive")
    reject(lambda value: value["archive"].update(schema_version="z"), "invalid archive")
    reject(lambda value: value["archive"].update(quiet_record_count=1), "count")
    reject(lambda value: value["archive"].update(quiet_record_count=3), "count")
    reject(
        lambda value: value["calibration"][0]["quiet_machine"].update(quiet=1),
        "accepted quiet",
    )
    reject(
        lambda value: value["calibration"][0]["quiet_machine"].update(record_file=1),
        "incomplete",
    )
    reject(
        lambda value: value["calibration"][0]["quiet_machine"].update(sha256=1),
        "incomplete",
    )
    reject(
        lambda value: value["calibration"][0]["quiet_machine"].update(sha256="0" * 64),
        "failed verification",
    )
    reject(
        lambda value: value["calibration"][0]["quiet_machine"].update(sha256="f" * 64),
        "failed verification",
    )
    reject(
        lambda value: value["calibration"][0]["quiet_machine"].update(
            record_file="../../../escape.json"
        ),
        "escapes",
    )


def test_archive_rejects_truthy_non_boolean_quiet_record(tmp_path: Path) -> None:
    source = tmp_path / "raw.json"
    destination = tmp_path / "archive" / "pilot-evidence.json"
    payload = _completed_payload()
    payload["calibration"][0]["quiet_machine"]["quiet"] = 1
    _write_json_atomic(source, payload)
    with pytest.raises(ValueError, match="accepted quiet"):
        archive_pilot_evidence(source, destination)

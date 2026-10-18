"""Losslessly shard large lane-8 pilot quiet records for repository storage."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast

from aptivra.eval.mps_pilot import (
    AMENDMENT_SCHEMA,
    PILOT_SCHEMA,
    _json_mapping,
    _write_json_atomic,
)

ARCHIVE_SCHEMA = "aptivra.h3_mps_pilot_archive.v1"


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _canonical_json(payload: Mapping[str, Any]) -> bytes:
    return (json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")


def _evidence_sections(payload: Mapping[str, Any]) -> tuple[str, ...]:
    return (
        ("evaluation_profiles", "calibration", "repeatability")
        if payload.get("schema_version") == AMENDMENT_SCHEMA
        else ("calibration", "repeatability")
    )


def _verify_existing_archive(payload: Mapping[str, Any], output_dir: Path) -> None:
    archive = payload.get("archive")
    if not isinstance(archive, Mapping) or archive.get("schema_version") != ARCHIVE_SCHEMA:
        raise ValueError("pilot evidence has an invalid archive record")
    expected_count = archive.get("quiet_record_count")
    references = 0
    for section in _evidence_sections(payload):
        rows = payload.get(section)
        if not isinstance(rows, list):
            raise ValueError(f"archived pilot evidence has invalid {section} rows")
        for row in rows:
            if not isinstance(row, Mapping):
                raise ValueError(f"archived pilot evidence has invalid {section} row")
            quiet = row.get("quiet_machine")
            if not isinstance(quiet, Mapping) or quiet.get("quiet") is not True:
                raise ValueError("archived pilot row lacks an accepted quiet reference")
            relative = quiet.get("record_file")
            digest = quiet.get("sha256")
            if not isinstance(relative, str) or not isinstance(digest, str):
                raise ValueError("archived pilot quiet reference is incomplete")
            shard = (output_dir / relative).resolve()
            try:
                shard.relative_to(output_dir.resolve())
            except ValueError as error:
                raise ValueError(
                    "archived pilot quiet shard escapes its output directory"
                ) from error
            if not shard.is_file() or _sha256_bytes(shard.read_bytes()) != digest:
                raise ValueError(f"archived pilot quiet shard failed verification: {relative}")
            references += 1
    if references != expected_count:
        raise ValueError("archived pilot quiet-record count does not match its references")


def archive_pilot_evidence(source: Path, destination: Path) -> dict[str, Any]:
    """Create a compact, idempotent evidence index plus lossless quiet JSON shards."""
    payload = _json_mapping(source)
    output_dir = destination.parent
    if payload.get("archive") is not None:
        if source.resolve() != destination.resolve():
            raise ValueError("an existing pilot archive can only verify itself in place")
        _verify_existing_archive(payload, output_dir)
        return payload
    schema = payload.get("schema_version")
    if schema not in {PILOT_SCHEMA, AMENDMENT_SCHEMA} or payload.get("status") != "completed":
        raise ValueError("only completed lane-8 pilot evidence can be archived")

    original_sha256 = _sha256_bytes(source.read_bytes())
    quiet_count = 0
    shard_directory = "lane8b-quiet-records" if schema == AMENDMENT_SCHEMA else "quiet-records"
    for section in _evidence_sections(payload):
        rows = payload.get(section)
        if not isinstance(rows, list):
            raise ValueError(f"pilot evidence has invalid {section} rows")
        for index, raw_row in enumerate(rows):
            if not isinstance(raw_row, dict):
                raise ValueError(f"pilot evidence has invalid {section} row")
            row = cast(dict[str, Any], raw_row)
            quiet = row.get("quiet_machine")
            if not isinstance(quiet, dict) or quiet.get("quiet") is not True:
                raise ValueError("pilot evidence row lacks an accepted quiet record")
            shard_payload = {
                "schema_version": ARCHIVE_SCHEMA,
                "section": section,
                "index": index,
                "cell": {
                    key: row[key]
                    for key in ("method", "stream", "micro_batch", "seed", "repeat_index")
                    if key in row
                },
                "quiet_machine": quiet,
            }
            shard_bytes = _canonical_json(shard_payload)
            relative = Path(shard_directory) / f"{section}-{index:03d}.json"
            shard = output_dir / relative
            _write_json_atomic(shard, shard_payload)
            row["quiet_machine"] = {
                "quiet": True,
                "record_file": relative.as_posix(),
                "sha256": _sha256_bytes(shard_bytes),
            }
            quiet_count += 1

    payload["archive"] = {
        "schema_version": ARCHIVE_SCHEMA,
        "original_evidence_sha256": original_sha256,
        "archiver_sha256": _sha256_bytes(Path(__file__).read_bytes()),
        "quiet_record_count": quiet_count,
        "lossless_fields": ["quiet_machine"],
    }
    _write_json_atomic(destination, payload)
    _verify_existing_archive(payload, output_dir)
    return payload

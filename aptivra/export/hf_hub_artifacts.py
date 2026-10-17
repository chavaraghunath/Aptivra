"""Atomic standard-PEFT staging for the ADR-0011 lane-7 release boundary."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import tempfile
from dataclasses import dataclass
from importlib.metadata import files as distribution_files
from importlib.metadata import version
from pathlib import Path
from typing import Any, Optional, Union, cast

import torch
import yaml
from safetensors import safe_open
from safetensors.torch import save_file

from aptivra.experts.router import SequenceTop1Router
from aptivra.experts.standard import (
    ExpertManifest,
    StandardExpertSettings,
    load_standard_expert_settings,
    standard_base_fingerprint,
)
from aptivra.export.model_card import (
    DEFAULT_RELEASE_GATE_CONFIG,
    ConfigSource,
    ReleaseGateSettings,
    generate_model_card_readme,
    load_release_gate_settings,
)
from aptivra.inference.chat_canonical import IdentityManifest, load_identity_manifest
from aptivra.inference.standard_base import (
    StandardBaseSettings,
    load_standard_base_settings,
)

STAGING_MANIFEST = "aptivra-manifest.json"
FILE_MANIFEST = "file-manifest.json"
ROUTER_WEIGHTS = "router/router.safetensors"
ROUTER_CONFIG = "router/router.json"
CHAT_TEMPLATE = "chat_template.jinja"
_STAGING_SCHEMA = "aptivra.standard_peft_bundle.v1"
_FILE_MANIFEST_SCHEMA = "aptivra.file_manifest.v1"
_ROUTER_SCHEMA = "aptivra.router_sidecar.v1"
_ADAPTER_FILES = ("adapter_config.json", "adapter_model.safetensors")
_SAFE_ID = re.compile(r"[A-Za-z0-9_-]+\Z")
_STRUCTURAL_FORBIDDEN = ("auto_map", "trust_remote_code")


@dataclass(frozen=True)
class PublishableAdapter:
    """One promoted PEFT artifact with closed publication provenance."""

    adapter_id: str
    kind: str
    task_family: str
    artifact_dir: Path


@dataclass(frozen=True)
class StagedRepository:
    """Verified local repository and its immutable root hashes."""

    path: Path
    manifest_sha256: str
    file_manifest_sha256: str
    files: tuple[str, ...]


def sha256_file(path: Path) -> str:
    """Hash one file without loading tensor payloads into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(payload: object) -> str:
    """Hash canonical UTF-8 JSON for self-authenticating manifests."""
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _equal(actual: object, expected: object) -> bool:
    return actual == expected


def _require_equal(actual: object, expected: object, message: str) -> None:
    if not _equal(actual, expected):
        raise ValueError(message)


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _read_json(path: Path, description: str) -> dict[str, Any]:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as error:
        raise ValueError(f"{description} is not valid UTF-8 JSON: {path}") from error
    except UnicodeDecodeError as error:
        raise ValueError(f"{description} is not valid UTF-8 JSON: {path}") from error
    try:
        raw: object = json.loads(text)
    except json.JSONDecodeError as error:
        raise ValueError(f"{description} is not valid UTF-8 JSON: {path}") from error
    if not isinstance(raw, dict):
        raise ValueError(f"{description} must contain a JSON object: {path}")
    return cast(dict[str, Any], raw)


def _manifest_with_hash(payload: dict[str, Any], hash_field: str) -> dict[str, Any]:
    if hash_field in payload:
        raise ValueError(f"manifest payload already contains {hash_field}")
    result = dict(payload)
    result[hash_field] = canonical_sha256(payload)
    return result


def _verify_self_hash(
    payload: dict[str, Any], hash_field: str, description: str
) -> str:
    recorded = payload.get(hash_field)
    if not isinstance(recorded, str):
        raise ValueError(f"{description} is missing {hash_field}")
    content = dict(payload)
    del content[hash_field]
    actual = canonical_sha256(content)
    _require_equal(actual, recorded, f"{description} self hash mismatch")
    return recorded


def _artifact_content_sha256(adapter_dir: Path) -> str:
    digest = hashlib.sha256()
    for filename in _ADAPTER_FILES:
        path = adapter_dir / filename
        if not path.is_file():
            raise ValueError(f"publishable adapter is missing required PEFT file: {filename}")
        payload = path.read_bytes()
        digest.update(filename.encode())
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)
    return digest.hexdigest()


def _validate_adapter_source(
    adapter: PublishableAdapter,
    base: StandardBaseSettings,
    settings: ReleaseGateSettings,
) -> tuple[ExpertManifest, Path]:
    if _SAFE_ID.fullmatch(adapter.adapter_id) is None:
        raise ValueError("publishable adapter ID must use letters, digits, dash, or underscore")
    if not adapter.task_family or _SAFE_ID.fullmatch(adapter.task_family) is None:
        raise ValueError("publishable adapter task family must be a safe non-empty identifier")
    if adapter.kind not in settings.publication.allowed_adapter_kinds:
        raise ValueError(
            f"adapter '{adapter.adapter_id}' has non-publishable provenance kind "
            f"'{adapter.kind}'; only synthetic and identity adapters may be staged"
        )
    artifact = adapter.artifact_dir
    manifest = ExpertManifest.model_validate(
        _read_json(artifact / "expert_manifest.json", "expert manifest")
    )
    _require_equal(
        manifest.adapter_name,
        adapter.adapter_id,
        "publishable adapter ID does not match its expert manifest",
    )
    source = base.source
    base_error = "publishable adapter does not match the pinned upstream base"
    _require_equal(manifest.base_repository, source.repository, base_error)
    _require_equal(manifest.base_revision, source.revision, base_error)
    _require_equal(manifest.base_config_sha256, source.config.sha256, base_error)
    _require_equal(manifest.base_weights_sha256, source.weights.sha256, base_error)
    _require_equal(manifest.base_fingerprint, standard_base_fingerprint(base), base_error)
    adapter_dir = artifact / manifest.adapter_subdir
    _require_equal(
        _artifact_content_sha256(adapter_dir),
        manifest.content_sha256,
        "publishable adapter content SHA-256 mismatch",
    )
    actual_source_files = {
        path.relative_to(artifact).as_posix()
        for path in artifact.rglob("*")
        if path.is_file()
    }
    expected_source_files = {
        "README.md",
        "expert_manifest.json",
        *(f"{manifest.adapter_subdir}/{filename}" for filename in _ADAPTER_FILES),
    }
    if not _equal(actual_source_files, expected_source_files):
        raise ValueError(
            "publishable adapter artifact contains non-PEFT files; raw data, configs, "
            "and auxiliary provenance are excluded"
        )
    return manifest, adapter_dir


def _router_payload(
    router: SequenceTop1Router,
    adapters: tuple[PublishableAdapter, ...],
    base_fingerprint: str,
) -> tuple[dict[str, Any], dict[str, torch.Tensor]]:
    adapter_ids = tuple(adapter.adapter_id for adapter in adapters)
    task_families = tuple(adapter.task_family for adapter in adapters)
    _require_equal(
        tuple(router.expert_ids),
        adapter_ids,
        "router expert IDs must exactly match staged adapter order",
    )
    _require_equal(
        tuple(router.task_families),
        task_families,
        "router task-family provenance must match staged adapters",
    )
    if not router.frozen_rows or not all(router.frozen_rows):
        raise ValueError("only fully promoted, frozen router rows may be staged")
    tensors = {
        "weights": torch.stack(tuple(router.weights)).detach().cpu().contiguous(),
        "biases": torch.stack(tuple(router.biases)).detach().cpu().contiguous(),
    }
    payload: dict[str, Any] = {
        "schema_version": _ROUTER_SCHEMA,
        "base_fingerprint": base_fingerprint,
        "hidden_size": router.hidden_size,
        "confidence_threshold": router.confidence_threshold,
        "adapter_ids": list(adapter_ids),
        "task_families": list(task_families),
        "frozen_rows": list(router.frozen_rows),
        "weights_file": Path(ROUTER_WEIGHTS).name,
    }
    return payload, tensors


def _chat_template_text() -> str:
    return """{%- for message in messages %}
<|im_start|>{{ message['role'] }}
{{ message['content'] }}<|im_end|>
{%- endfor %}
{%- if add_generation_prompt %}
<|im_start|>assistant
{%- endif %}
"""


def _license_text() -> str:
    source_license = Path(__file__).resolve().parents[2] / "LICENSE"
    if source_license.is_file():
        text = source_license.read_text(encoding="utf-8")
        if "Apache License" in text:
            return text
    packaged = distribution_files("aptivra")
    if packaged is not None:
        for entry in packaged:
            normalized = str(entry).replace("\\", "/")
            if normalized.endswith(".dist-info/licenses/LICENSE"):
                path = Path(entry.locate())
                text = path.read_text(encoding="utf-8")
                if "Apache License" in text:
                    return text
    raise FileNotFoundError("the Apache-2.0 licence text is unavailable locally")


def _notice_text(
    identity: IdentityManifest,
    adapters: tuple[PublishableAdapter, ...],
) -> str:
    names = ", ".join(adapter.adapter_id for adapter in adapters)
    return (
        "Aptivra standard PEFT bundle\n\n"
        f"Upstream base: {identity.base.repository}\n"
        f"Immutable revision: {identity.base.revision}\n"
        "Upstream licence: Apache-2.0\n"
        "Upstream copyright and attribution notices must be retained; no holder or year "
        "is invented where the offline source evidence supplies none.\n\n"
        "Aptivra modifications: separately loaded PEFT LoRA adapters, a sequence-level "
        "router sidecar, and the aptivra-chat-v1 template.\n"
        f"Derivative adapters: {names}\n"
        f"Owner-approved creator attribution: {identity.model.creator.name}\n"
    )


def _adapter_manifest_entry(
    adapter: PublishableAdapter,
    manifest: ExpertManifest,
    staged_adapter_dir: Path,
) -> dict[str, Any]:
    return {
        "adapter_id": adapter.adapter_id,
        "kind": adapter.kind,
        "task_family": adapter.task_family,
        "path": f"adapters/{adapter.adapter_id}",
        "rank": manifest.rank,
        "target_modules": list(manifest.target_modules),
        "parameter_count": manifest.parameter_count,
        "derivative_of": f"{manifest.base_repository}@{manifest.base_revision}",
        "content_sha256": manifest.content_sha256,
        "files": {
            filename: sha256_file(staged_adapter_dir / filename)
            for filename in _ADAPTER_FILES
        },
    }


def _staging_manifest_payload(
    base: StandardBaseSettings,
    experts: StandardExpertSettings,
    identity: IdentityManifest,
    adapters: list[dict[str, Any]],
    router_sha256: str,
    router_config_sha256: str,
) -> dict[str, Any]:
    release_record = identity.release_evaluation.record
    evaluation = (
        {
            "record": None,
            "absence": "The identity manifest contains no bound release-evaluation record.",
        }
        if release_record is None
        else {"record": release_record.model_dump(mode="json"), "absence": None}
    )
    return {
        "schema_version": _STAGING_SCHEMA,
        "base": {
            "repository": base.source.repository,
            "revision": base.source.revision,
            "config_sha256": base.source.config.sha256,
            "weights_sha256": base.source.weights.sha256,
            "fingerprint": standard_base_fingerprint(base),
            "architecture": base.architecture.model_type,
            "unique_parameter_count": base.architecture.unique_parameter_count,
            "weights_included": False,
        },
        "dependencies": {
            "transformers_version": version("transformers"),
            "peft_version": experts.peft.version,
        },
        "adapters": adapters,
        "router": {
            "schema_version": _ROUTER_SCHEMA,
            "weights_path": ROUTER_WEIGHTS,
            "weights_sha256": router_sha256,
            "config_path": ROUTER_CONFIG,
            "config_sha256": router_config_sha256,
        },
        "chat_template": {"version": "aptivra-chat-v1", "path": CHAT_TEMPLATE},
        "identity": {
            "schema_version": identity.schema_version,
            "content_sha256": identity.content_sha256,
            "model_name": identity.model.name,
            "creator": identity.model.creator.model_dump(mode="json"),
            "research_status": identity.research_status,
            "limitations": list(identity.required_limitations),
        },
        "evaluation": evaluation,
        "licence": {
            "spdx": "Apache-2.0",
            "upstream_base": base.source.repository,
            "upstream_revision": base.source.revision,
            "copyright_notice_policy": (
                "Retain all supplied upstream copyright and attribution notices."
            ),
            "notice_policy": (
                "Retain readable upstream NOTICE attributions when supplied and mark modifications."
            ),
            "aptivra_modifications_marked": True,
            "all_adapters_are_derivatives": True,
            "legal_approval_owner": identity.model.creator.approval,
        },
    }


def _all_regular_files(root: Path) -> set[str]:
    return {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file()
    }


def _find_term(value: object, normalized_terms: tuple[str, ...]) -> Optional[str]:  # noqa: UP045
    if isinstance(value, str):
        folded = value.casefold()
        return next((term for term in normalized_terms if term in folded), None)
    if isinstance(value, dict):
        for key, child in value.items():
            found = _find_term(key, normalized_terms) or _find_term(
                child, normalized_terms
            )
            if found is not None:
                return found
    if isinstance(value, (list, tuple)):
        for child in value:
            found = _find_term(child, normalized_terms)
            if found is not None:
                return found
    return None


def _scan_model_card_front_matter(text: str, normalized_terms: tuple[str, ...]) -> None:
    if not text.startswith("---\n"):
        raise ValueError("model card must start with YAML front matter")
    closing = text.find("\n---\n", 4)
    if closing < 0:
        raise ValueError("model card YAML front matter is not terminated")
    front_matter: object = yaml.safe_load(text[4:closing])
    found = _find_term(front_matter, normalized_terms)
    if found is not None:
        raise ValueError(f"model-card field contains prohibited term: {found}")


def scan_staged_repository(
    stage_path: Path,
    forbidden_references: tuple[str, ...],
) -> None:
    """Scan paths, all text/structured values, templates, and tensor metadata."""
    terms = tuple(dict.fromkeys(term.casefold() for term in forbidden_references))
    for relative in sorted(_all_regular_files(stage_path)):
        path = stage_path / relative
        folded_path = relative.casefold()
        if path.suffix.casefold() in {".py", ".pyi", ".pyc"} or ".py." in folded_path:
            raise ValueError(f"Python files are prohibited from staged repositories: {relative}")
        path_term = next(
            (
                term
                for term in (*_STRUCTURAL_FORBIDDEN, *terms)
                if term in folded_path
            ),
            None,
        )
        if path_term is not None:
            raise ValueError(
                f"staged path or filename contains prohibited term '{path_term}': {relative}"
            )
        if path.suffix.casefold() == ".safetensors":
            try:
                with safe_open(path, framework="pt", device="cpu") as handle:
                    metadata = handle.metadata() or {}
            except Exception as error:
                raise ValueError(f"staged safetensors file is invalid: {relative}") from error
            found = _find_term(metadata, terms)
            serialized_metadata = json.dumps(metadata, sort_keys=True).casefold()
            structural = next(
                (
                    term
                    for term in _STRUCTURAL_FORBIDDEN
                    if re.search(rf"\b{re.escape(term)}\b", serialized_metadata)
                ),
                None,
            )
            found = found or structural
            if found is not None:
                raise ValueError(
                    f"safetensors metadata contains prohibited term '{found}': {relative}"
                )
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError as error:
            raise ValueError(f"staged non-tensor file is not UTF-8 text: {relative}") from error
        folded_text = text.casefold()
        text_term = next((term for term in terms if term in folded_text), None)
        structural = next(
            (
                term
                for term in _STRUCTURAL_FORBIDDEN
                if re.search(rf"\b{re.escape(term)}\b", folded_text)
            ),
            None,
        )
        text_term = text_term or structural
        if text_term is not None:
            raise ValueError(
                f"staged text contains prohibited term '{text_term}': {relative}"
            )
        if path.suffix.casefold() == ".json":
            structured: object = json.loads(text)
            found = _find_term(structured, terms)
            if found is not None:
                raise ValueError(
                    f"staged JSON value contains prohibited term '{found}': {relative}"
                )
        elif path.suffix.casefold() in {".yaml", ".yml"}:
            structured = yaml.safe_load(text)
            found = _find_term(structured, terms)
            if found is not None:
                raise ValueError(
                    f"staged YAML value contains prohibited term '{found}': {relative}"
                )
        elif relative == "README.md":
            _scan_model_card_front_matter(text, terms)


def _write_file_manifest(stage: Path) -> dict[str, Any]:
    files = {
        relative: sha256_file(stage / relative)
        for relative in sorted(_all_regular_files(stage))
        if relative != FILE_MANIFEST
    }
    allowlist = sorted((*files, FILE_MANIFEST))
    payload = {
        "schema_version": _FILE_MANIFEST_SCHEMA,
        "allowlist": allowlist,
        "files": files,
        "self_hash_rule": (
            "file_manifest_sha256 is canonical JSON SHA-256 with that field omitted"
        ),
    }
    result = _manifest_with_hash(payload, "file_manifest_sha256")
    _write_json(stage / FILE_MANIFEST, result)
    return result


def verify_staged_repository(
    stage_path: Union[Path, str],  # noqa: UP007
    *,
    config_path: ConfigSource = DEFAULT_RELEASE_GATE_CONFIG,
) -> StagedRepository:
    """Fail closed on any missing, extra, changed, executable, or prohibited byte."""
    root = Path(stage_path)
    settings = load_release_gate_settings(config_path)
    file_manifest = _read_json(root / FILE_MANIFEST, "file manifest")
    _require_equal(
        file_manifest.get("schema_version"),
        _FILE_MANIFEST_SCHEMA,
        "file manifest schema is invalid",
    )
    file_manifest_sha256 = _verify_self_hash(
        file_manifest, "file_manifest_sha256", "file manifest"
    )
    allowlist = file_manifest.get("allowlist")
    hashes = file_manifest.get("files")
    if not isinstance(allowlist, list) or not all(
        isinstance(item, str) for item in allowlist
    ):
        raise ValueError("file manifest allowlist must contain only paths")
    if not isinstance(hashes, dict) or not all(
        isinstance(key, str) and isinstance(value, str)
        for key, value in hashes.items()
    ):
        raise ValueError("file manifest hashes must map paths to SHA-256 values")
    actual_files = _all_regular_files(root)
    if not _equal(actual_files, set(allowlist)):
        missing = sorted(set(allowlist) - actual_files)
        extra = sorted(actual_files - set(allowlist))
        raise ValueError(
            f"staged file allowlist mismatch; missing={missing}, extra={extra}"
        )
    expected_hashed = set(allowlist) - {FILE_MANIFEST}
    _require_equal(
        set(hashes),
        expected_hashed,
        "file manifest hash keys do not exactly cover staged payload files",
    )
    for relative, expected in hashes.items():
        actual = sha256_file(root / relative)
        _require_equal(actual, expected, f"staged file SHA-256 mismatch: {relative}")
    manifest = _read_json(root / STAGING_MANIFEST, "staging manifest")
    _require_equal(
        manifest.get("schema_version"),
        _STAGING_SCHEMA,
        "staging manifest schema is invalid",
    )
    manifest_sha256 = _verify_self_hash(
        manifest, "manifest_sha256", "staging manifest"
    )
    router = manifest.get("router")
    if not isinstance(router, dict):
        raise ValueError("staging manifest router must be an object")
    router_bindings = (
        (router.get("weights_path"), router.get("weights_sha256")),
        (router.get("config_path"), router.get("config_sha256")),
    )
    for relative, expected in router_bindings:
        if not isinstance(relative, str) or not isinstance(expected, str):
            raise ValueError("staging manifest router file binding is invalid")
        _require_equal(
            sha256_file(root / relative),
            expected,
            f"staging manifest router SHA-256 mismatch: {relative}",
        )
    adapters = manifest.get("adapters")
    if not isinstance(adapters, list) or not adapters:
        raise ValueError("staging manifest adapters must be a non-empty list")
    for adapter in adapters:
        if not isinstance(adapter, dict):
            raise ValueError("staging manifest adapter entry must be an object")
        adapter_path = adapter.get("path")
        adapter_files = adapter.get("files")
        if not isinstance(adapter_path, str) or not isinstance(adapter_files, dict):
            raise ValueError("staging manifest adapter file binding is invalid")
        for filename, expected in adapter_files.items():
            if not isinstance(filename, str) or not isinstance(expected, str):
                raise ValueError("staging manifest adapter hash is invalid")
            relative = f"{adapter_path}/{filename}"
            _require_equal(
                sha256_file(root / relative),
                expected,
                f"staging manifest adapter SHA-256 mismatch: {relative}",
            )
    scan_staged_repository(root, settings.forbidden_references)
    return StagedRepository(
        root,
        manifest_sha256,
        file_manifest_sha256,
        tuple(sorted(actual_files)),
    )


def stage_huggingface_hub_artifacts(
    output_dir: Union[Path, str],  # noqa: UP007
    adapters: tuple[PublishableAdapter, ...],
    router: SequenceTop1Router,
    *,
    base_settings: Optional[StandardBaseSettings] = None,  # noqa: UP045
    expert_settings: Optional[StandardExpertSettings] = None,  # noqa: UP045
    identity_manifest_path: Optional[Union[Path, str]] = None,  # noqa: UP007, UP045
    config_path: ConfigSource = DEFAULT_RELEASE_GATE_CONFIG,
) -> StagedRepository:
    """Build one atomic standard PEFT bundle without requiring a repository name."""
    settings = load_release_gate_settings(config_path)
    if not settings.feature.enabled:
        raise RuntimeError("ADR-0011 standard publication staging is disabled by configuration")
    if not adapters:
        raise ValueError("at least one publishable adapter is required")
    base = load_standard_base_settings() if base_settings is None else base_settings
    experts = (
        load_standard_expert_settings() if expert_settings is None else expert_settings
    )
    identity = (
        load_identity_manifest(base_settings=base)
        if identity_manifest_path is None
        else load_identity_manifest(identity_manifest_path, base_settings=base)
    )
    if not _equal(version("peft"), experts.peft.version):
        raise RuntimeError("installed PEFT version does not match the reviewed exact pin")
    validated: dict[str, tuple[PublishableAdapter, ExpertManifest, Path]] = {}
    for adapter in adapters:
        if adapter.adapter_id in validated:
            raise ValueError(f"duplicate publishable adapter ID: {adapter.adapter_id}")
        manifest, adapter_dir = _validate_adapter_source(adapter, base, settings)
        validated[adapter.adapter_id] = (adapter, manifest, adapter_dir)
    _require_equal(
        set(validated),
        set(router.expert_ids),
        "router and publishable adapter IDs must match exactly",
    )
    ordered = tuple(validated[expert_id][0] for expert_id in router.expert_ids)
    router_payload, router_tensors = _router_payload(
        router,
        ordered,
        standard_base_fingerprint(base),
    )
    output = Path(output_dir)
    output.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent))
    try:
        adapter_entries: list[dict[str, Any]] = []
        for adapter in ordered:
            _, expert_manifest, source_dir = validated[adapter.adapter_id]
            destination = stage / "adapters" / adapter.adapter_id
            destination.mkdir(parents=True)
            for filename in _ADAPTER_FILES:
                shutil.copy2(source_dir / filename, destination / filename)
            adapter_entries.append(
                _adapter_manifest_entry(adapter, expert_manifest, destination)
            )
        router_weights = stage / ROUTER_WEIGHTS
        router_weights.parent.mkdir(parents=True)
        # Safetensors serializes metadata through an unordered map, so repeated
        # writes can produce different bytes. All router metadata is bound in
        # the hashed JSON sidecar; keeping the tensor header empty makes the
        # staged repository byte-reproducible and therefore idempotent.
        save_file(router_tensors, router_weights)
        _write_json(stage / ROUTER_CONFIG, router_payload)
        (stage / CHAT_TEMPLATE).write_text(_chat_template_text(), encoding="utf-8")
        (stage / "LICENSE").write_text(_license_text(), encoding="utf-8")
        (stage / "NOTICE").write_text(
            _notice_text(identity, ordered), encoding="utf-8"
        )
        manifest_payload = _staging_manifest_payload(
            base,
            experts,
            identity,
            adapter_entries,
            sha256_file(router_weights),
            sha256_file(stage / ROUTER_CONFIG),
        )
        package_manifest = _manifest_with_hash(manifest_payload, "manifest_sha256")
        _write_json(stage / STAGING_MANIFEST, package_manifest)
        (stage / "README.md").write_text(
            generate_model_card_readme(package_manifest), encoding="utf-8"
        )
        _write_file_manifest(stage)
        staged = verify_staged_repository(stage, config_path=config_path)
        if output.exists():
            existing = verify_staged_repository(output, config_path=config_path)
            if not _equal(
                existing.file_manifest_sha256, staged.file_manifest_sha256
            ):
                raise FileExistsError(
                    "refusing to replace a different staged PEFT repository"
                )
            return existing
        os.replace(stage, output)
        return verify_staged_repository(output, config_path=config_path)
    finally:
        if stage.exists():
            shutil.rmtree(stage)

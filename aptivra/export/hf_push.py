"""Publication that freshly evaluates the exact Aptivra Hub layout being uploaded."""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

from huggingface_hub import HfApi

from aptivra.eval.model_card_eval import evaluate_export_dir, sha256_file
from aptivra.eval.model_card_schema import EvaluationRecord, validate_evaluation_record
from aptivra.export.hf_hub_artifacts import HUB_LAYOUT_FILES
from aptivra.export.model_card import (
    ReleaseGateSettings,
    generate_model_card_readme,
    load_release_gate_settings,
    load_results_record,
)
from aptivra.inference.hub_config import model_config_from_export_dir

HUB_PUBLICATION_FILES = (*HUB_LAYOUT_FILES, "README.md")

# These are the only obsolete remote paths a publication may remove. Keep this list
# explicit: broad cleanup globs could delete the stock-Transformers layout.
LEGACY_HUB_DELETE_PATTERNS = (
    "dense/*",
    "experts/*",
    "gguf/*",
    "*.gguf",
    "multitier_manifest.json",
    ".expert_usage",
)


@dataclass
class HfPublicationPlan:
    """The complete, inspectable mutation sent to the Hub in one commit."""

    export_dir: Path
    upload_files: tuple[str, ...]
    delete_patterns: tuple[str, ...]
    file_sha256: dict[str, str]
    release_gate_settings: ReleaseGateSettings
    _temporary_directory: tempfile.TemporaryDirectory[str] = field(repr=False)

    def verify_staged_files(self) -> None:
        """Refuse if any staged upload byte changed or contains forbidden text."""
        for filename in self.upload_files:
            path = self.export_dir / filename
            if not path.is_file():
                raise ValueError(f"staged publication file disappeared: {filename}")
            actual = sha256_file(path)
            expected = self.file_sha256.get(filename)
            if expected is None or actual != expected:
                raise ValueError(
                    f"staged publication file changed after staging: {filename}"
                )
        _assert_no_forbidden_hub_references(
            self.export_dir,
            self.upload_files,
            self.release_gate_settings.forbidden_hub_references,
        )

    def cleanup(self) -> None:
        """Remove the private staging copy and all generated publication files."""
        self._temporary_directory.cleanup()

    def __enter__(self) -> HfPublicationPlan:
        return self

    def __exit__(self, *_exc_info: object) -> None:
        self.cleanup()


def _stage_hub_layout(export_path: Path) -> tuple[
    tempfile.TemporaryDirectory[str], Path, dict[str, str]
]:
    """Copy the allowlisted immutable inputs into a private temporary directory."""
    temporary_directory = tempfile.TemporaryDirectory(prefix="aptivra-hf-publication-")
    stage_path = Path(temporary_directory.name)
    os.chmod(stage_path, 0o700)
    try:
        for filename in HUB_LAYOUT_FILES:
            shutil.copy2(export_path / filename, stage_path / filename)
        hashes = {
            filename: sha256_file(stage_path / filename)
            for filename in HUB_LAYOUT_FILES
        }
    except BaseException:
        temporary_directory.cleanup()
        raise
    return temporary_directory, stage_path, hashes


def _num_experts(export_path: Path) -> int:
    config_path = export_path / "config.json"
    try:
        payload = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read Hub config: {config_path}") from exc
    value = payload.get("num_experts") if isinstance(payload, dict) else None
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError("config.json num_experts must be a non-negative integer")
    return value


def prepare_hf_publication(
    repo_id: str,
    export_dir: Path | str,
    results_record: Path | str | None = None,
) -> HfPublicationPlan:
    """Freshly evaluate the export and render its card without contacting the Hub.

    A supplied record is only a reproducibility cross-check. The freshly measured record always
    controls the card and release notice. Publication never rewrites the evaluated weights.
    """
    export_path = Path(export_dir)
    missing = [name for name in HUB_LAYOUT_FILES if not (export_path / name).is_file()]
    if missing:
        raise FileNotFoundError(
            "Hub layout is incomplete; evaluate a complete layout before publishing. "
            f"Missing: {', '.join(missing)}"
        )

    temporary_directory, stage_path, staged_hashes = _stage_hub_layout(export_path)
    try:
        settings = load_release_gate_settings()
        fresh_record = evaluate_export_dir(stage_path)
        fresh_validated = validate_evaluation_record(
            fresh_record,
            minimum_trained_samples=settings.minimum_trained_prompt_samples,
            minimum_heldout_samples=settings.minimum_heldout_prompt_samples,
            minimum_heldout_text_bytes=settings.minimum_heldout_text_bytes,
            export_dir=stage_path,
        )
        if results_record is not None:
            results_path = Path(results_record)
            if not results_path.is_file():
                raise FileNotFoundError(f"model-card results record not found: {results_path}")
            supplied_record = load_results_record(results_path)
            supplied_validated = validate_evaluation_record(
                supplied_record,
                minimum_trained_samples=settings.minimum_trained_prompt_samples,
                minimum_heldout_samples=settings.minimum_heldout_prompt_samples,
                minimum_heldout_text_bytes=settings.minimum_heldout_text_bytes,
                export_dir=stage_path,
            )
            _require_matching_records(
                supplied_validated,
                fresh_validated,
                tolerance=settings.supplied_record_metric_tolerance,
            )

        model_config = model_config_from_export_dir(stage_path, device="cpu")
        readme = generate_model_card_readme(
            repo_id,
            model_config,
            _num_experts(stage_path),
            results_record=fresh_record,
            fresh_evaluation=fresh_validated,
        )
        readme_path = stage_path / "README.md"
        readme_path.write_text(readme, encoding="utf-8")
        staged_hashes["README.md"] = sha256_file(readme_path)

        plan = HfPublicationPlan(
            export_dir=stage_path,
            upload_files=HUB_PUBLICATION_FILES,
            delete_patterns=LEGACY_HUB_DELETE_PATTERNS,
            file_sha256=staged_hashes,
            release_gate_settings=settings,
            _temporary_directory=temporary_directory,
        )
        plan.verify_staged_files()
        return plan
    except BaseException:
        temporary_directory.cleanup()
        raise


def _require_matching_records(
    supplied: EvaluationRecord,
    fresh: EvaluationRecord,
    *,
    tolerance: float,
) -> None:
    """Require supplied evidence to reproduce every freshly measured field."""
    _compare_record_values(
        supplied.model_dump(mode="json", by_alias=True),
        fresh.model_dump(mode="json", by_alias=True),
        path=(),
        tolerance=tolerance,
    )


def _compare_record_values(
    supplied: object,
    fresh: object,
    *,
    path: tuple[str, ...],
    tolerance: float,
) -> None:
    if path == ("timestamp_utc",):
        return
    location = ".".join(path) or "record"
    if isinstance(supplied, dict) and isinstance(fresh, dict):
        if supplied.keys() != fresh.keys():
            raise ValueError(f"supplied results record differs from fresh evaluation at {location}")
        for key in supplied:
            _compare_record_values(
                supplied[key],
                fresh[key],
                path=(*path, str(key)),
                tolerance=tolerance,
            )
        return
    if isinstance(supplied, list) and isinstance(fresh, list):
        if len(supplied) != len(fresh):
            raise ValueError(f"supplied results record differs from fresh evaluation at {location}")
        for index, (supplied_item, fresh_item) in enumerate(zip(supplied, fresh, strict=True)):
            _compare_record_values(
                supplied_item,
                fresh_item,
                path=(*path, str(index)),
                tolerance=tolerance,
            )
        return
    supplied_is_number = isinstance(supplied, int | float) and not isinstance(supplied, bool)
    fresh_is_number = isinstance(fresh, int | float) and not isinstance(fresh, bool)
    if supplied_is_number and fresh_is_number:
        supplied_number = cast(int | float, supplied)
        fresh_number = cast(int | float, fresh)
        if abs(float(supplied_number) - float(fresh_number)) <= tolerance:
            return
    elif supplied == fresh:
        return
    raise ValueError(f"supplied results record differs from fresh evaluation at {location}")


def _assert_no_forbidden_hub_references(
    stage_path: Path,
    upload_files: tuple[str, ...],
    forbidden_references: tuple[str, ...],
) -> None:
    findings: list[str] = []
    normalized_terms = tuple(term.casefold() for term in forbidden_references)
    for filename in upload_files:
        path = stage_path / filename
        if path.suffix == ".safetensors":
            continue
        try:
            with path.open(encoding="utf-8") as handle:
                for line_number, line in enumerate(handle, start=1):
                    folded = line.casefold()
                    matched = sorted(
                        term
                        for term, normalized in zip(
                            forbidden_references, normalized_terms, strict=True
                        )
                        if normalized in folded
                    )
                    if matched:
                        findings.append(
                            f"{filename}:{line_number}: forbidden reference "
                            f"{', '.join(matched)}"
                        )
        except UnicodeDecodeError as exc:
            raise ValueError(
                f"staged Hub text file is not valid UTF-8: {filename}"
            ) from exc
    if findings:
        details = "\n".join(findings)
        raise ValueError(f"Hub publication content contains forbidden references:\n{details}")


def push_aptivra_model_to_hf(
    repo_id: str,
    export_dir: Path | str,
    results_record: Path | str | None = None,
    *,
    private: bool = False,
    api: Any | None = None,
) -> str:
    """Publish one validated layout and remove only enumerated legacy artifacts."""
    with prepare_hf_publication(repo_id, export_dir, results_record) as plan:
        plan.verify_staged_files()
        client = api or HfApi()
        print(f"Creating Hugging Face repository '{repo_id}'...")
        client.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True, private=private)

        print(f"Uploading Transformers layout to 'https://huggingface.co/{repo_id}'...")
        plan.verify_staged_files()
        client.upload_folder(
            folder_path=str(plan.export_dir),
            repo_id=repo_id,
            repo_type="model",
            allow_patterns=list(plan.upload_files),
            delete_patterns=list(plan.delete_patterns),
            commit_message="Publish evaluated single-safetensors Transformers layout",
        )

    url = f"https://huggingface.co/{repo_id}"
    print(f"Upload complete! Model available at: {url}")
    return url

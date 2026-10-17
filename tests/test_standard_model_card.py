"""Evidence and attribution tests for the ADR-0011 standard model card."""

from __future__ import annotations

import json
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from aptivra.export.model_card import (
    _card_data,
    generate_model_card_readme,
    load_release_gate_settings,
)
from tests.test_standard_packaging import PackageFixture, package_fixture

__all__ = ["package_fixture"]


def test_model_card_uses_only_bound_manifest_facts(
    package_fixture: PackageFixture,
) -> None:
    manifest = json.loads(
        (package_fixture.stage / "aptivra-manifest.json").read_text(encoding="utf-8")
    )
    card = (package_fixture.stage / "README.md").read_text(encoding="utf-8")
    base = manifest["base"]
    creator = manifest["identity"]["creator"]

    assert f'"{base["repository"]}"' in card
    assert f'revision="{base["revision"]}"' in card
    assert "AutoModelForCausalLM.from_pretrained" in card
    assert "PeftModel.from_pretrained" in card
    assert 'subfolder="adapters/identity"' in card
    assert 'model.set_adapter("identity")' in card
    assert "Automatic routing is separate" in card
    assert "Automatic routing requires the\ninstalled Aptivra orchestrator" in card
    assert creator["name"] in card
    assert creator["approval"] in card


def test_model_card_is_research_only_and_claims_no_unbound_capability(
    package_fixture: PackageFixture,
) -> None:
    card = (package_fixture.stage / "README.md").read_text(encoding="utf-8")
    assert "Research model" in card
    assert "No capability is claimed" in card
    assert "no bound release-evaluation record" in card
    assert "Results outside a named, bound release-evaluation record are unmeasured" in card
    assert "ordinary conversation changes only ephemeral context" in card
    assert "Aptivra may be wrong" in card


def test_model_card_records_licence_notice_modifications_and_derivatives(
    package_fixture: PackageFixture,
) -> None:
    card = (package_fixture.stage / "README.md").read_text(encoding="utf-8")
    notice = (package_fixture.stage / "NOTICE").read_text(encoding="utf-8")
    licence = (package_fixture.stage / "LICENSE").read_text(encoding="utf-8")

    assert "license: apache-2.0" in card
    assert "SmolLM2 is licensed under Apache-2.0" in card
    assert "copyright" in card
    assert "NOTICE" in card
    assert "Aptivra modifications" in card
    assert "Derivative adapters" in notice
    assert "identity" in notice
    assert "Legal\napproval for publication remains with the repository owner" in card
    assert "Apache License" in licence


def test_model_card_limits_capability_to_a_bound_evaluation_record(
    package_fixture: PackageFixture,
) -> None:
    manifest = json.loads(
        (package_fixture.stage / "aptivra-manifest.json").read_text(encoding="utf-8")
    )
    manifest["evaluation"] = {
        "absence": None,
        "record": {
            "metric": {"name": "exact_match", "value": 0.75, "unit": "fraction"},
            "stream_version": "v2",
            "seed_set": [7, 11],
            "denominator": 16,
            "date": "2026-10-17",
        },
    }

    card = generate_model_card_readme(manifest)

    assert "`exact_match` = `0.75` `fraction`" in card
    assert "stream `v2`" in card
    assert "seeds `[7, 11]`" in card
    assert "denominator `16`" in card
    assert "No broader capability follows from it" in card


def test_release_config_rejects_extra_root_key(tmp_path: Path) -> None:
    path = tmp_path / "release.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "model_card_release_gate": {},
                "unexpected": {},
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="root must contain only"):
        load_release_gate_settings(path)


def test_release_config_is_immutable() -> None:
    settings = load_release_gate_settings()
    with pytest.raises(ValidationError, match="frozen"):
        settings.feature.enabled = True


def _write_release_payload(path: Path, payload: object) -> Path:
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return path


@pytest.mark.parametrize(
    "payload",
    [
        {},
        [],
        {"wrong": {}},
    ],
)
def test_release_config_rejects_missing_or_wrong_root(
    tmp_path: Path, payload: object
) -> None:
    path = _write_release_payload(tmp_path / "release.yaml", payload)
    with pytest.raises(ValueError, match="root must contain only"):
        load_release_gate_settings(str(path))


@pytest.mark.parametrize(
    ("update", "message"),
    [
        (("feature", "enabled", "true"), "boolean"),
        (("repository", "repository_id", "invalid"), "owner/name"),
        (("publication", "allowed_adapter_kinds", ["identity", "synthetic"]), "exactly"),
        (("publication", "allowed_adapter_kinds", ["synthetic", "synthetic"]), "exactly"),
        (("publication", "allowed_adapter_kinds", ["synthetic"]), "exactly"),
        (("forbidden_references", None, []), "non-empty"),
        (("forbidden_references", None, [" "]), "non-empty"),
        (("forbidden_references", None, ["github", "GitHub"]), "unique"),
    ],
)
def test_release_config_rejects_invalid_field_values(
    tmp_path: Path,
    update: tuple[str, str | None, object],
    message: str,
) -> None:
    raw = yaml.safe_load(Path("configs/model_card_release_gate.yaml").read_text(encoding="utf-8"))
    first, second, value = update
    if second is None:
        raw["model_card_release_gate"][first] = value
    else:
        raw["model_card_release_gate"][first][second] = value
    path = _write_release_payload(tmp_path / "release.yaml", raw)
    with pytest.raises((ValueError, ValidationError), match=message):
        load_release_gate_settings(path)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("adapters", "not-a-list", "adapters must be a non-empty list"),
        ("evaluation.record.seed_set", "not-a-list", "seed_set must be a non-empty list"),
    ],
)
def test_model_card_rejects_wrong_collection_types(
    package_fixture: PackageFixture,
    field: str,
    value: str,
    message: str,
) -> None:
    manifest = json.loads(
        (package_fixture.stage / "aptivra-manifest.json").read_text(encoding="utf-8")
    )
    if field == "adapters":
        manifest["adapters"] = value
    else:
        manifest["evaluation"] = {
            "absence": None,
            "record": {
                "metric": {"name": "exact_match", "value": 0.75, "unit": "fraction"},
                "stream_version": "v2",
                "seed_set": value,
                "denominator": 16,
                "date": "2026-10-17",
            },
        }
    with pytest.raises(ValueError, match=message):
        generate_model_card_readme(manifest)


def test_model_card_data_is_immutable_and_preserves_adapter_order(
    package_fixture: PackageFixture,
) -> None:
    manifest = json.loads(
        (package_fixture.stage / "aptivra-manifest.json").read_text(encoding="utf-8")
    )
    second = dict(manifest["adapters"][0])
    second["adapter_id"] = "synthetic-second"
    second["kind"] = "synthetic"
    manifest["adapters"].append(second)

    data = _card_data(manifest)
    with pytest.raises(FrozenInstanceError):
        data.model_name = "changed"
    card = generate_model_card_readme(manifest)
    assert 'subfolder="adapters/identity"' in card
    assert "`identity`" in card
    assert "`synthetic-second`" in card


def test_release_config_accepts_a_large_unique_forbidden_reference_set(
    tmp_path: Path,
) -> None:
    raw = yaml.safe_load(Path("configs/model_card_release_gate.yaml").read_text(encoding="utf-8"))
    expected = [f"term-{index}" for index in range(300)]
    raw["model_card_release_gate"]["forbidden_references"] = expected
    path = _write_release_payload(tmp_path / "release.yaml", raw)
    settings = load_release_gate_settings(path)
    assert settings.forbidden_references == tuple(expected)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("model-name", "model_name must be non-empty text"),
        ("limitations-type", "limitations must be a non-empty list"),
        ("limitations-empty", "limitations must be a non-empty list"),
    ],
)
def test_model_card_rejects_invalid_identity_text_and_limitations(
    package_fixture: PackageFixture,
    mutation: str,
    message: str,
) -> None:
    manifest = json.loads(
        (package_fixture.stage / "aptivra-manifest.json").read_text(encoding="utf-8")
    )
    if mutation == "model-name":
        manifest["identity"]["model_name"] = 7
    elif mutation == "limitations-type":
        manifest["identity"]["limitations"] = "not-a-list"
    else:
        manifest["identity"]["limitations"] = []
    with pytest.raises(ValueError, match=message):
        generate_model_card_readme(manifest)

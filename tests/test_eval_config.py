"""Tests for task-stream experiment configuration and generator cardinality."""

import random
import re
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, call, sentinel

import numpy as np
import pytest
import torch
import yaml

from aptivra.eval import config as config_module
from aptivra.eval.config import (
    EvaluationSettings,
    ExperimentSettings,
    TaskStreamSettings,
    _bounded_int,
    _family_role_settings,
    _require_mapping,
    _string_set,
    load_experiment_settings,
    seed_experiment,
)
from aptivra.tasks.algo_synth import AlgoSynthTaskFamily
from aptivra.tasks.base import TaskFamily
from aptivra.tasks.bug_fixer import BugFixerTaskFamily
from aptivra.tasks.docstring_code import DocstringToCodeTaskFamily
from aptivra.tasks.python_microsynth import PythonMicroSynthTaskFamily
from aptivra.tasks.stream import ContinualTaskStream
from aptivra.tasks.structure import structural_hash
from aptivra.tasks.type_inferrer import TypeInferrerTaskFamily


def _write_yaml(path: Path, document: object) -> Path:
    path.write_text(yaml.safe_dump(document), encoding="utf-8")
    return path


def _v1_document(**task_stream_overrides: object) -> dict[str, object]:
    task_stream: dict[str, object] = {
        "num_train_per_task": 10,
        "num_val_per_task": 5,
        "num_test_per_task": 5,
        "seed": 42,
        "max_generation_attempts": 100,
        **task_stream_overrides,
    }
    return {"task_stream": task_stream, "evaluation": {"max_new_tokens": 32}}


def _v2_document(**task_stream_overrides: object) -> dict[str, object]:
    document = yaml.safe_load(
        Path("configs/task_stream_v2.yaml").read_text(encoding="utf-8")
    )
    assert isinstance(document, dict)
    task_stream = document["task_stream"]
    assert isinstance(task_stream, dict)
    task_stream.update(task_stream_overrides)
    return document


def _experiment_settings(
    family_roles: dict[str, str], headline_roles: frozenset[str]
) -> ExperimentSettings:
    return ExperimentSettings(
        task_stream=TaskStreamSettings(
            num_train_per_task=17,
            num_val_per_task=7,
            num_test_per_task=11,
            num_template_ood_test_per_code_family=13,
            seed=23,
            max_generation_attempts=29,
            version="v2",
            family_roles=family_roles,
            headline_roles=headline_roles,
        ),
        evaluation=EvaluationSettings(max_new_tokens=31),
    )


def test_make_stream_passes_every_setting_and_honours_seed_zero(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    factory = MagicMock(
        side_effect=[sentinel.default_stream, sentinel.zero_seed_stream]
    )
    monkeypatch.setattr(config_module, "ContinualTaskStream", factory)
    populated_roles = {"T1": "skill", "T2": "diagnostic"}
    headline_roles = frozenset({"skill"})

    default_stream = _experiment_settings(populated_roles, headline_roles).make_stream()
    zero_seed_stream = _experiment_settings({}, headline_roles).make_stream(seed=0)

    assert default_stream is sentinel.default_stream
    assert zero_seed_stream is sentinel.zero_seed_stream
    assert factory.call_args_list == [
        call(
            num_train_per_task=17,
            num_val_per_task=7,
            num_test_per_task=11,
            num_template_ood_test_per_code_family=13,
            seed=23,
            max_generation_attempts=29,
            version="v2",
            family_roles=populated_roles,
            headline_roles=headline_roles,
        ),
        call(
            num_train_per_task=17,
            num_val_per_task=7,
            num_test_per_task=11,
            num_template_ood_test_per_code_family=13,
            seed=0,
            max_generation_attempts=29,
            version="v2",
            family_roles=None,
            headline_roles=headline_roles,
        ),
    ]


def test_require_mapping_preserves_a_valid_mapping() -> None:
    mapping = {"answer": 42}
    assert _require_mapping(mapping, "section") is mapping


@pytest.mark.parametrize("value", [None, [], (), "mapping", 1])
def test_require_mapping_rejects_non_mappings(value: object) -> None:
    with pytest.raises(ValueError, match="^'section' must be a YAML mapping\\.$"):
        _require_mapping(value, "section")


@pytest.mark.parametrize(
    ("document", "message"),
    [
        ([], "'root' must be a YAML mapping."),
        ({"task_stream": [], "evaluation": {}}, "'task_stream' must be a YAML mapping."),
        ({"task_stream": {}, "evaluation": []}, "'evaluation' must be a YAML mapping."),
    ],
)
def test_load_experiment_settings_rejects_non_mapping_schema_sections(
    tmp_path: Path, document: object, message: str
) -> None:
    path = _write_yaml(tmp_path / "invalid-schema.yaml", document)
    with pytest.raises(ValueError, match=f"^{re.escape(message)}$"):
        load_experiment_settings(path)


@pytest.mark.parametrize("version", [None, 1, "v3"])
def test_load_experiment_settings_rejects_unknown_versions(
    tmp_path: Path, version: object
) -> None:
    path = _write_yaml(tmp_path / "invalid-version.yaml", _v1_document(version=version))
    with pytest.raises(
        ValueError, match="^'version' must be either 'v1' or 'v2'\\.$"
    ):
        load_experiment_settings(path)


@pytest.mark.parametrize("value", [3, 7])
def test_bounded_int_accepts_inclusive_boundaries(value: int) -> None:
    assert _bounded_int({"workers": value}, "workers", minimum=3, maximum=7) == value


@pytest.mark.parametrize("value", [2, 8])
def test_bounded_int_rejects_values_outside_boundaries(value: int) -> None:
    with pytest.raises(
        ValueError,
        match=f"^'workers' must be between 3 and 7, got {value}\\.$",
    ):
        _bounded_int({"workers": value}, "workers", minimum=3, maximum=7)


@pytest.mark.parametrize("value", [None, True, False, 3.0, "3", [3]])
def test_bounded_int_rejects_missing_booleans_and_non_integers(value: object) -> None:
    section = {} if value is None else {"workers": value}
    with pytest.raises(ValueError, match="^'workers' must be an integer\\.$"):
        _bounded_int(section, "workers", minimum=3, maximum=7)


def test_string_set_returns_unique_values_without_losing_members() -> None:
    assert _string_set(["skill", "diagnostic"], "roles") == frozenset(
        {"skill", "diagnostic"}
    )


@pytest.mark.parametrize("value", [None, [], (), "skill"])
def test_string_set_requires_a_non_empty_yaml_list(value: object) -> None:
    with pytest.raises(
        ValueError, match="^'roles' must be a non-empty YAML list\\.$"
    ):
        _string_set(value, "roles")


@pytest.mark.parametrize("value", [[""], [1], ["skill", ""]])
def test_string_set_requires_non_empty_strings(value: object) -> None:
    with pytest.raises(
        ValueError, match="^'roles' must contain only non-empty strings\\.$"
    ):
        _string_set(value, "roles")


def test_string_set_rejects_duplicate_roles() -> None:
    with pytest.raises(
        ValueError, match="^'roles' must not contain duplicate values\\.$"
    ):
        _string_set(["skill", "skill"], "roles")


def _patch_two_v2_families(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "aptivra.tasks.v2.v2_families",
        lambda: [
            SimpleNamespace(family_id="alpha"),
            SimpleNamespace(family_id="beta"),
        ],
    )


def _valid_role_settings() -> dict[str, object]:
    return {
        "roles": ["skill", "diagnostic"],
        "headline_roles": ["skill"],
        "family_roles": {"alpha": "skill", "beta": "diagnostic"},
    }


def test_family_roles_use_v1_defaults_only_when_mapping_is_absent() -> None:
    assert _family_role_settings({}, "v1") == ({}, frozenset({"skill"}))
    with pytest.raises(
        ValueError, match="^'family_roles' is required for task stream v2\\.$"
    ):
        _family_role_settings({}, "v2")


def test_family_roles_return_the_validated_mapping_and_headline_set(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_two_v2_families(monkeypatch)
    family_roles, headline_roles = _family_role_settings(_valid_role_settings(), "v2")
    assert family_roles == {"alpha": "skill", "beta": "diagnostic"}
    assert headline_roles == frozenset({"skill"})


def test_family_roles_reject_sorted_unknown_headline_roles(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_two_v2_families(monkeypatch)
    settings = _valid_role_settings()
    settings["headline_roles"] = ["zeta", "alpha"]
    with pytest.raises(
        ValueError,
        match="^'headline_roles' contains unknown roles: alpha, zeta$",
    ):
        _family_role_settings(settings, "v2")


def test_family_roles_requires_a_mapping(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_two_v2_families(monkeypatch)
    settings = _valid_role_settings()
    settings["family_roles"] = []
    with pytest.raises(
        ValueError, match="^'family_roles' must be a YAML mapping\\.$"
    ):
        _family_role_settings(settings, "v2")


@pytest.mark.parametrize("invalid_family_id", ["", 1])
def test_family_roles_requires_non_empty_string_family_ids(
    monkeypatch: pytest.MonkeyPatch, invalid_family_id: object
) -> None:
    _patch_two_v2_families(monkeypatch)
    settings = _valid_role_settings()
    settings["family_roles"] = {
        invalid_family_id: "skill",
        "alpha": "skill",
        "beta": "diagnostic",
    }
    with pytest.raises(
        ValueError,
        match="^'family_roles' keys must be non-empty family IDs\\.$",
    ):
        _family_role_settings(settings, "v2")


@pytest.mark.parametrize("invalid_role", ["unknown", 1])
def test_family_roles_rejects_unknown_or_non_string_roles(
    monkeypatch: pytest.MonkeyPatch, invalid_role: object
) -> None:
    _patch_two_v2_families(monkeypatch)
    settings = _valid_role_settings()
    settings["family_roles"] = {"alpha": invalid_role, "beta": "diagnostic"}
    with pytest.raises(
        ValueError,
        match=f"^Family 'alpha' references unknown role '{invalid_role}'\\.$",
    ):
        _family_role_settings(settings, "v2")


def test_family_roles_reports_sorted_unknown_family_ids(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_two_v2_families(monkeypatch)
    settings = _valid_role_settings()
    settings["family_roles"] = {
        "alpha": "skill",
        "beta": "diagnostic",
        "zeta": "skill",
        "gamma": "skill",
    }
    with pytest.raises(
        ValueError,
        match="^'family_roles' contains unknown family IDs: gamma, zeta$",
    ):
        _family_role_settings(settings, "v2")


def test_family_roles_reports_sorted_missing_family_ids(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_two_v2_families(monkeypatch)
    settings = _valid_role_settings()
    settings["family_roles"] = {}
    with pytest.raises(
        ValueError,
        match="^'family_roles' is missing family IDs: alpha, beta$",
    ):
        _family_role_settings(settings, "v2")


def test_family_roles_requires_at_least_one_headline_family(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_two_v2_families(monkeypatch)
    settings = _valid_role_settings()
    settings["family_roles"] = {"alpha": "diagnostic", "beta": "diagnostic"}
    with pytest.raises(
        ValueError, match="^At least one task family must be headline-eligible\\.$"
    ):
        _family_role_settings(settings, "v2")


def test_template_ood_size_defaults_follow_stream_version(tmp_path: Path) -> None:
    v1_path = _write_yaml(tmp_path / "v1.yaml", _v1_document())
    v2_document = _v2_document()
    v2_task_stream = v2_document["task_stream"]
    assert isinstance(v2_task_stream, dict)
    del v2_task_stream["num_template_ood_test_per_code_family"]
    v2_path = _write_yaml(tmp_path / "v2.yaml", v2_document)

    assert (
        load_experiment_settings(v1_path).task_stream.num_template_ood_test_per_code_family
        == 0
    )
    assert (
        load_experiment_settings(v2_path).task_stream.num_template_ood_test_per_code_family
        == 300
    )


@pytest.mark.parametrize(("version", "value"), [("v1", 0), ("v1", 2_000), ("v2", 5), ("v2", 2_000)])
def test_template_ood_size_accepts_inclusive_version_bounds(
    tmp_path: Path, version: str, value: int
) -> None:
    document = (
        _v2_document(num_template_ood_test_per_code_family=value)
        if version == "v2"
        else _v1_document(num_template_ood_test_per_code_family=value)
    )
    path = _write_yaml(tmp_path / "boundary.yaml", document)
    settings = load_experiment_settings(path)
    assert settings.task_stream.num_template_ood_test_per_code_family == value


@pytest.mark.parametrize("value", [True, 5.0, "5", None])
def test_template_ood_size_rejects_non_integer_values(
    tmp_path: Path, value: object
) -> None:
    path = _write_yaml(
        tmp_path / "non-integer.yaml",
        _v1_document(num_template_ood_test_per_code_family=value),
    )
    with pytest.raises(
        ValueError,
        match="^'num_template_ood_test_per_code_family' must be an integer\\.$",
    ):
        load_experiment_settings(path)


@pytest.mark.parametrize("value", [-1, 2_001])
def test_template_ood_size_rejects_global_out_of_range_values(
    tmp_path: Path, value: int
) -> None:
    path = _write_yaml(
        tmp_path / "out-of-range.yaml",
        _v1_document(num_template_ood_test_per_code_family=value),
    )
    with pytest.raises(
        ValueError,
        match=(
            "^'num_template_ood_test_per_code_family' must be between 0 and 2000, "
            f"got {value}\\.$"
        ),
    ):
        load_experiment_settings(path)


@pytest.mark.parametrize("value", [0, 4])
def test_template_ood_size_enforces_v2_minimum(tmp_path: Path, value: int) -> None:
    path = _write_yaml(
        tmp_path / "v2-too-small.yaml",
        _v2_document(num_template_ood_test_per_code_family=value),
    )
    with pytest.raises(
        ValueError,
        match=(
            "^'num_template_ood_test_per_code_family' must be between 5 and 2000 "
            "for task stream v2\\.$"
        ),
    ):
        load_experiment_settings(path)


def test_seed_experiment_reproduces_python_numpy_and_torch_rngs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(config_module.torch.cuda, "is_available", lambda: False)

    seed_experiment(314_159)
    first = (
        random.getrandbits(64),
        np.random.randint(0, 2**31, size=4),
        torch.rand(4),
    )
    seed_experiment(314_159)
    repeated = (
        random.getrandbits(64),
        np.random.randint(0, 2**31, size=4),
        torch.rand(4),
    )

    assert first[0] == repeated[0]
    np.testing.assert_array_equal(first[1], repeated[1])
    assert torch.equal(first[2], repeated[2])


@pytest.mark.parametrize(("cuda_available", "expected_cuda_calls"), [(False, 0), (True, 1)])
def test_seed_experiment_seeds_cuda_only_when_available(
    monkeypatch: pytest.MonkeyPatch,
    cuda_available: bool,
    expected_cuda_calls: int,
) -> None:
    python_seed = MagicMock()
    numpy_seed = MagicMock()
    torch_seed = MagicMock()
    cuda_is_available = MagicMock(return_value=cuda_available)
    cuda_seed = MagicMock()
    monkeypatch.setattr(config_module.random, "seed", python_seed)
    monkeypatch.setattr(config_module.np.random, "seed", numpy_seed)
    monkeypatch.setattr(config_module.torch, "manual_seed", torch_seed)
    monkeypatch.setattr(config_module.torch.cuda, "is_available", cuda_is_available)
    monkeypatch.setattr(config_module.torch.cuda, "manual_seed_all", cuda_seed)

    seed_experiment(73)

    python_seed.assert_called_once_with(73)
    numpy_seed.assert_called_once_with(73)
    torch_seed.assert_called_once_with(73)
    cuda_is_available.assert_called_once_with()
    assert cuda_seed.call_args_list == [call(73)] * expected_cuda_calls


def test_repository_task_stream_config_has_statistically_useful_test_split() -> None:
    settings = load_experiment_settings(Path("configs/task_stream.yaml"))
    assert settings.task_stream.num_test_per_task >= 300
    assert settings.task_stream.num_template_ood_test_per_code_family == 0
    assert settings.task_stream.seed == 42
    assert settings.task_stream.max_generation_attempts == 100
    assert settings.task_stream.version == "v1"
    assert settings.evaluation.max_new_tokens == 128


def test_task_stream_config_rejects_invalid_generation_limit(tmp_path: Path) -> None:
    config = tmp_path / "invalid.yaml"
    config.write_text(
        """task_stream:
  num_train_per_task: 10
  num_val_per_task: 5
  num_test_per_task: 300
  seed: 42
  max_generation_attempts: 100
evaluation:
  max_new_tokens: 0
""",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="max_new_tokens"):
        load_experiment_settings(config)


def test_default_generation_bound_covers_every_current_code_target() -> None:
    settings = load_experiment_settings(Path("configs/task_stream.yaml"))
    families: tuple[TaskFamily, ...] = (
        PythonMicroSynthTaskFamily(),
        AlgoSynthTaskFamily(),
        DocstringToCodeTaskFamily(),
        BugFixerTaskFamily(),
        TypeInferrerTaskFamily(),
    )

    for family in families:
        for instance in family.generate_instances(100, seed=42):
            assert len(instance.training_target().encode("utf-8")) <= (
                settings.evaluation.max_new_tokens
            )


@pytest.mark.parametrize(
    ("yaml_text", "message"),
    [
        ("[]\n", "root"),
        (
            """task_stream:
  num_train_per_task: true
  num_val_per_task: 5
  num_test_per_task: 300
  seed: 42
  max_generation_attempts: 100
evaluation:
  max_new_tokens: 32
""",
            "num_train_per_task",
        ),
        (
            """task_stream:
  num_train_per_task: 10
  num_val_per_task: 5
  num_test_per_task: 300
  seed: -1
  max_generation_attempts: 100
evaluation:
  max_new_tokens: 32
""",
            "seed",
        ),
    ],
)
def test_task_stream_config_rejects_malformed_values(
    tmp_path: Path, yaml_text: str, message: str
) -> None:
    config = tmp_path / "malformed.yaml"
    config.write_text(yaml_text, encoding="utf-8")
    with pytest.raises(ValueError, match=message):
        load_experiment_settings(config)


def test_v2_code_tasks_have_structurally_unique_configured_splits() -> None:
    stream = ContinualTaskStream(500, 100, 300, seed=42, version="v2")

    for family in stream.families[7:]:
        split = stream.get_split(family.family_id)
        train = {structural_hash(item.target_solution) for item in split.train}
        val = {structural_hash(item.target_solution) for item in split.val}
        test = {structural_hash(item.target_solution) for item in split.test}

        assert None not in train | val | test
        assert (len(train), len(val), len(test)) == (500, 100, 300)
        assert train.isdisjoint(val)
        assert train.isdisjoint(test)
        assert val.isdisjoint(test)

        template_ood = stream.get_template_ood_split(family.family_id)
        template_ood_structures = {
            structural_hash(item.target_solution) for item in template_ood
        }
        assert len(template_ood_structures) == 300
        assert None not in template_ood_structures
        assert train.isdisjoint(template_ood_structures)
        assert val.isdisjoint(template_ood_structures)
        assert test.isdisjoint(template_ood_structures)

    for family in stream.families:
        split = stream.get_split(family.family_id)
        for instance in split.train + split.val + split.test:
            assert len(instance.training_target().encode("utf-8")) <= 128
            assert len(
                (instance.prompt + instance.training_target()).encode("utf-8")
            ) <= 256

        for instance in stream.get_template_ood_split(family.family_id):
            assert len(instance.training_target().encode("utf-8")) <= 128
            assert len(
                (instance.prompt + instance.training_target()).encode("utf-8")
            ) <= 256

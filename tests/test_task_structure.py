"""Behavioral tests for structural Python fingerprints."""

import ast
import hashlib

import pytest

from aptivra.tasks.structure import normalized_python_ast, structural_hash


def test_sync_function_renaming_includes_nested_recursion_and_closures() -> None:
    original = """
def walk(limit: int) -> int:
    def step(current: int) -> int:
        if current <= 0:
            return limit
        return step(current - 1)
    return step(limit)
"""
    renamed = """
def traverse(bound: int) -> int:
    def descend(remaining: int) -> int:
        if remaining <= 0:
            return bound
        return descend(remaining - 1)
    return descend(bound)
"""

    assert normalized_python_ast(original) == normalized_python_ast(renamed)
    assert structural_hash(original) == structural_hash(renamed)


def test_async_function_renaming_preserves_async_and_await_structure() -> None:
    original = """
async def collect(client, token):
    response = await client.fetch(token)
    return response
"""
    renamed = """
async def gather(transport, key):
    result = await transport.fetch(key)
    return result
"""
    synchronous = """
def gather(transport, key):
    result = transport.fetch(key)
    return result
"""
    changed_attribute = renamed.replace(".fetch", ".send")

    assert structural_hash(original) == structural_hash(renamed)
    assert structural_hash(original) != structural_hash(synchronous)
    assert structural_hash(original) != structural_hash(changed_attribute)


def test_recursive_async_function_uses_the_stable_canonical_name() -> None:
    original = """
async def countdown(value):
    if value <= 0:
        return value
    return await countdown(value - 1)
"""
    renamed = """
async def descend(remaining):
    if remaining <= 0:
        return remaining
    return await descend(remaining - 1)
"""

    normalized = normalized_python_ast(original)

    assert normalized is not None
    assert "AsyncFunctionDef(name='function'" in normalized
    assert "countdown" not in normalized
    assert normalized == normalized_python_ast(renamed)


def test_bound_locals_are_normalized_but_global_references_are_preserved() -> None:
    original = """
def calculate(values):
    subtotal = sum(values)
    return subtotal + OFFSET
"""
    locals_renamed = """
def compute(items):
    result = sum(items)
    return result + OFFSET
"""
    global_renamed = locals_renamed.replace("OFFSET", "LIMIT")
    builtin_changed = locals_renamed.replace("sum(items)", "max(items)")

    assert structural_hash(original) == structural_hash(locals_renamed)
    assert structural_hash(original) != structural_hash(global_renamed)
    assert structural_hash(original) != structural_hash(builtin_changed)


def test_annotations_retain_types_while_bound_aliases_and_arguments_are_renamed() -> None:
    original = """
Vector = list[int]

def convert(value: Vector) -> Vector:
    return value
"""
    renamed = """
Sequence = list[int]

def transform(item: Sequence) -> Sequence:
    return item
"""
    argument_type_changed = renamed.replace("item: Sequence", "item: tuple[int]")
    return_type_changed = renamed.replace("-> Sequence", "-> tuple[int]")

    assert structural_hash(original) == structural_hash(renamed)
    assert structural_hash(original) != structural_hash(argument_type_changed)
    assert structural_hash(original) != structural_hash(return_type_changed)


def test_bindings_inside_annotation_expressions_are_normalized() -> None:
    original = """
def apply(value: (lambda annotation_value: annotation_value)):
    return value
"""
    renamed = """
def transform(item: (lambda annotation_item: annotation_item)):
    return item
"""

    assert normalized_python_ast(original) == normalized_python_ast(renamed)


def test_formatting_and_comments_do_not_change_the_fingerprint() -> None:
    compact = "def increment(value):\n    return value + 1\n"
    formatted = """
# File-level commentary is not executable structure.
def increment( value ):
    # Nor is an implementation comment.
    return (
        value
        + 1
    )
"""

    assert normalized_python_ast(compact) == normalized_python_ast(formatted)
    assert structural_hash(compact) == structural_hash(formatted)


@pytest.mark.parametrize(
    "changed",
    [
        "def evaluate(value):\n    return value - 1",
        "def evaluate(value):\n    return value + 2",
        'def evaluate(value):\n    return value + "1"',
        "def evaluate(value):\n    return value <= 1",
    ],
)
def test_operators_and_literals_remain_part_of_the_fingerprint(changed: str) -> None:
    baseline = "def evaluate(value):\n    return value + 1"

    assert structural_hash(baseline) != structural_hash(changed)


@pytest.mark.parametrize(
    "invalid_source",
    [
        "def unfinished(",
        "if True print('missing colon')",
        "value = 1\x00",
    ],
)
def test_syntax_errors_have_no_structural_representation(invalid_source: str) -> None:
    assert normalized_python_ast(invalid_source) is None
    assert structural_hash(invalid_source) is None


def test_parser_value_errors_have_no_structural_representation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def reject_source(_: str) -> ast.AST:
        raise ValueError("parser rejected source")

    monkeypatch.setattr(ast, "parse", reject_source)

    assert normalized_python_ast("value = 1") is None
    assert structural_hash("value = 1") is None


def test_simple_fingerprint_representation_is_stable_and_sha256_derived() -> None:
    source = "value = 1"
    expected_normalized = (
        "Module(body=[Assign(targets=[Name(id='identifier', ctx=Store())], "
        "value=Constant(value=1))], type_ignores=[])"
    )
    expected_digest = "27160c7cac7fdc5da2c6c2ed45ca004265fddb1a39cd8395e267932033ecc009"

    normalized = normalized_python_ast(source)

    assert normalized == expected_normalized
    assert structural_hash(source) == expected_digest
    assert expected_digest == hashlib.sha256(expected_normalized.encode("utf-8")).hexdigest()


def test_function_and_argument_canonical_names_are_stable() -> None:
    normalized = normalized_python_ast("def named(item):\n    return item")

    assert normalized is not None
    assert "FunctionDef(name='function'" in normalized
    assert "arg(arg='argument'" in normalized
    assert "Name(id='identifier', ctx=Load())" in normalized

"""Structural fingerprints for generated Python task targets."""

import ast
import hashlib


class _BoundNameCollector(ast.NodeVisitor):
    """Collect identifiers introduced by the program rather than Python builtins."""

    def __init__(self) -> None:
        self.names: set[str] = set()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:  # noqa: N802
        self.names.add(node.name)
        self.generic_visit(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:  # noqa: N802
        self.names.add(node.name)
        self.generic_visit(node)

    def visit_arg(self, node: ast.arg) -> None:  # noqa: N802
        self.names.add(node.arg)
        self.generic_visit(node)

    def visit_Name(self, node: ast.Name) -> None:  # noqa: N802
        if isinstance(node.ctx, ast.Store):
            self.names.add(node.id)


class _IdentifierNormalizer(ast.NodeTransformer):
    """Erase identifier spelling while retaining executable structure and literals."""

    def __init__(self, bound_names: set[str]) -> None:
        self.bound_names = bound_names

    def visit_Name(self, node: ast.Name) -> ast.AST:  # noqa: N802
        if node.id not in self.bound_names:
            return node
        return ast.copy_location(ast.Name(id="identifier", ctx=node.ctx), node)

    def visit_arg(self, node: ast.arg) -> ast.AST:  # noqa: N802
        annotation = self.visit(node.annotation) if node.annotation is not None else None
        return ast.copy_location(
            ast.arg(arg="argument", annotation=annotation, type_comment=node.type_comment),
            node,
        )

    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.AST:  # noqa: N802
        normalized = self.generic_visit(node)
        assert isinstance(normalized, ast.FunctionDef)
        normalized.name = "function"
        return normalized

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> ast.AST:  # noqa: N802
        normalized = self.generic_visit(node)
        assert isinstance(normalized, ast.AsyncFunctionDef)
        normalized.name = "function"
        return normalized


def normalized_python_ast(source: str) -> str | None:
    """Return a name-normalized AST dump, or ``None`` when source is not Python."""
    try:
        tree = ast.parse(source)
    except (SyntaxError, ValueError):
        return None
    collector = _BoundNameCollector()
    collector.visit(tree)
    normalized = _IdentifierNormalizer(collector.names).visit(tree)
    ast.fix_missing_locations(normalized)
    return ast.dump(normalized, annotate_fields=True, include_attributes=False)


def structural_hash(source: str) -> str | None:
    """Hash normalized Python structure while preserving operators and literals."""
    normalized = normalized_python_ast(source)
    if normalized is None:
        return None
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()

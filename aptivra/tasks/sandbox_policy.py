"""Single builtin policy for generated Python executed by verifier workers."""

from collections.abc import Mapping
from types import MappingProxyType

ALLOWED_BUILTINS: Mapping[str, object] = MappingProxyType(
    {
        "abs": abs,
        "bool": bool,
        "enumerate": enumerate,
        "float": float,
        "int": int,
        "len": len,
        "list": list,
        "max": max,
        "min": min,
        "range": range,
        "reversed": reversed,
        "round": round,
        "set": set,
        "sorted": sorted,
        "str": str,
        "sum": sum,
        "tuple": tuple,
        "zip": zip,
    }
)


def allowed_builtins() -> dict[str, object]:
    """Return a fresh namespace containing every approved generated-code builtin."""
    return dict(ALLOWED_BUILTINS)

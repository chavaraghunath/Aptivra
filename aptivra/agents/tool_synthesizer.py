"""Milestone Genesis Phase 3: Dynamic Tool Synthesizer & Sandboxed Execution Engine."""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any


@dataclass
class SynthesizedTool:
    """A self-compiled, verified executable tool primitive."""

    tool_name: str
    signature: str
    code_body: str
    is_sandboxed: bool
    execution_fn: Callable[..., Any]


class ToolSynthesizerEngine:
    """Enables Aptivra to write, verify, and register its own executable primitives on the fly."""

    def __init__(self) -> None:
        self.registry: dict[str, SynthesizedTool] = {}

    def synthesize_tool(self, tool_name: str, signature: str, python_code: str) -> SynthesizedTool:
        """Compiles a Python function in an isolated dictionary and registers it."""
        sandbox_env: dict[str, Any] = {
            "__builtins__": {
                "range": range,
                "len": len,
                "min": min,
                "max": max,
                "sum": sum,
                "abs": abs,
                "int": int,
                "float": float,
                "str": str,
            }
        }

        # Compile and execute within restricted globals
        compiled = compile(python_code, f"<synthesized_{tool_name}>", "exec")
        exec(compiled, sandbox_env)

        fn = sandbox_env.get(tool_name)
        if not callable(fn):
            raise ValueError(f"Synthesized code did not define callable '{tool_name}'")

        tool = SynthesizedTool(
            tool_name=tool_name,
            signature=signature,
            code_body=python_code,
            is_sandboxed=True,
            execution_fn=fn,
        )
        self.registry[tool_name] = tool
        return tool

    def execute_tool(self, tool_name: str, *args: Any, **kwargs: Any) -> Any:
        """Executes a previously synthesized and verified tool."""
        tool = self.registry.get(tool_name)
        if not tool:
            raise KeyError(f"Tool '{tool_name}' not registered in synthesizer.")
        return tool.execution_fn(*args, **kwargs)

    def __len__(self) -> int:
        return len(self.registry)

"""Dynamic tool synthesis with child-process containment for generated code.

Generated tools execute through the task verifier's resource-limited child runner so
hangs and accidental failures do not take down the caller. This is not a security
boundary against hostile code.
"""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from aptivra.tasks.code_sandbox import (
    SandboxConfig,
    execute_function_source,
    load_sandbox_config,
    validate_function_source,
)


@dataclass
class SynthesizedTool:
    """A generated tool whose callable routes execution through a child process."""

    tool_name: str
    signature: str
    code_body: str
    is_sandboxed: bool
    execution_fn: Callable[..., Any]


class ToolSynthesizerEngine:
    """Validate and register generated tools with operational child containment."""

    def __init__(self, config: SandboxConfig | None = None) -> None:
        self.registry: dict[str, SynthesizedTool] = {}
        self._config = config or load_sandbox_config()

    def synthesize_tool(self, tool_name: str, signature: str, python_code: str) -> SynthesizedTool:
        """Validate a function in a child process and register its source."""
        validation = validate_function_source(
            python_code,
            tool_name,
            config=self._config,
        )
        if not validation.passed:
            detail = validation.error_message or "unknown validation failure"
            raise ValueError(f"Synthesized tool validation failed: {detail}")

        def execution_fn(*args: Any, **kwargs: Any) -> Any:
            result = execute_function_source(
                python_code,
                tool_name,
                args,
                kwargs,
                config=self._config,
            )
            if not result.passed:
                detail = result.error_message or "unknown execution failure"
                raise RuntimeError(f"Synthesized tool execution failed: {detail}")
            return result.value

        tool = SynthesizedTool(
            tool_name=tool_name,
            signature=signature,
            code_body=python_code,
            is_sandboxed=validation.ran_in_child_process,
            execution_fn=execution_fn,
        )
        self.registry[tool_name] = tool
        return tool

    def execute_tool(self, tool_name: str, *args: Any, **kwargs: Any) -> Any:
        """Execute a registered tool in a fresh resource-limited child process."""
        tool = self.registry.get(tool_name)
        if not tool:
            raise KeyError(f"Tool '{tool_name}' not registered in synthesizer.")
        return tool.execution_fn(*args, **kwargs)

    def __len__(self) -> int:
        return len(self.registry)

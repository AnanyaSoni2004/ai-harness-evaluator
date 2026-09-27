"""ToolSpec, ToolRegistry, dispatch, finish-tool factory, build_registry()."""
from __future__ import annotations

import functools
from dataclasses import dataclass
from typing import Any, Callable

from harness.shell import truncate
from harness.textproto import INVALID_TOOL, format_example
from harness.types import ToolCall, ToolResult

ALLOWED_SCHEMA_KEYS = frozenset({"type", "description", "properties", "required", "items", "enum"})
TOOL_KINDS = frozenset({"read", "write", "exec", "control"})
FINISH = "finish"


@dataclass
class ToolSpec:
    """One tool: its model-facing schema plus the bound Python function."""

    name: str
    description: str
    parameters: dict
    fn: Callable[..., ToolResult]
    kind: str

    def schema(self) -> dict:
        """OpenAI-style function schema."""
        return {"type": "function",
                "function": {"name": self.name, "description": self.description, "parameters": self.parameters}}


def validate_schema(schema: dict, where: str = "parameters") -> None:
    """Raise ValueError if a schema uses keys some providers reject (default, anyOf, ...)."""
    bad = set(schema) - ALLOWED_SCHEMA_KEYS
    if bad:
        raise ValueError(f"{where}: forbidden schema keys {sorted(bad)}")
    for name, prop in (schema.get("properties") or {}).items():
        validate_schema(prop, f"{where}.{name}")
    if isinstance(schema.get("items"), dict):
        validate_schema(schema["items"], f"{where}.items")


def summarize_parameters(parameters: dict) -> str:
    """One-line parameter summary, e.g. 'path (string, required), start_line (integer)'."""
    required = set(parameters.get("required") or [])
    parts = []
    for name, prop in (parameters.get("properties") or {}).items():
        kind = prop.get("type", "any")
        if kind == "array":
            kind = f"array of {(prop.get('items') or {}).get('type', 'any')}"
        parts.append(f"{name} ({kind}{', required' if name in required else ''})")
    return ", ".join(parts) or "none"


class ToolRegistry:
    """Named tools with schema export and a dispatch that never raises."""

    def __init__(self, max_output_chars: int | None = None) -> None:
        """max_output_chars caps every tool result (None = no cap)."""
        self._tools: dict[str, ToolSpec] = {}
        self.max_output_chars = max_output_chars

    def register(self, spec: ToolSpec) -> None:
        """Add a tool; raises ValueError for a duplicate name, unknown kind, or forbidden schema keys."""
        if spec.name in self._tools:
            raise ValueError(f"tool '{spec.name}' is already registered")
        if spec.kind not in TOOL_KINDS:
            raise ValueError(f"tool '{spec.name}': unknown kind '{spec.kind}'")
        validate_schema(spec.parameters, spec.name)
        self._tools[spec.name] = spec

    def subset(self, names: list[str]) -> "ToolRegistry":
        """A new registry with only the named tools (names that are not registered are skipped)."""
        sub = ToolRegistry(self.max_output_chars)
        for name in names:
            if name in self._tools:
                sub._tools[name] = self._tools[name]
        return sub

    def names(self) -> list[str]:
        """Registered tool names, in registration order."""
        return list(self._tools)

    def get(self, name: str) -> ToolSpec | None:
        """The ToolSpec for a name, or None."""
        return self._tools.get(name)

    def __contains__(self, name: object) -> bool:
        return name in self._tools

    def openai_schemas(self) -> list[dict]:
        """Function schemas for every tool, in registration order."""
        return [spec.schema() for spec in self._tools.values()]

    def _cap(self, result: ToolResult) -> ToolResult:
        """Apply the global output cap."""
        if self.max_output_chars and len(result.output) > self.max_output_chars:
            result.output = truncate(result.output, self.max_output_chars)
        return result

    def dispatch(self, call: ToolCall) -> ToolResult:
        """Validate a call and run its tool. Never raises; every failure is a helpful ToolResult."""
        if call.name == INVALID_TOOL:
            return ToolResult(False, f"Your tool call could not be parsed ({call.parse_error}). Reply with exactly "
                                     'one ```tool block containing {"name": "<tool>", "arguments": {...}}.')
        spec = self._tools.get(call.name)
        if spec is None:
            return ToolResult(False, f"Unknown tool '{call.name}'. Available: {', '.join(self.names())}")
        example = format_example(spec.schema())
        if call.parse_error or not isinstance(call.arguments, dict):
            err = call.parse_error or "arguments must be a JSON object"
            return ToolResult(False, f"Your tool arguments were not valid JSON ({err}). Expected parameters: "
                                     f"{summarize_parameters(spec.parameters)}. {example}")
        props = spec.parameters.get("properties") or {}
        args = {k: v for k, v in call.arguments.items() if k in props}
        missing = [r for r in spec.parameters.get("required") or [] if args.get(r) is None]
        if missing:
            return ToolResult(False, f"Missing required parameter(s) for {spec.name}: {', '.join(missing)}. "
                                     f"Expected parameters: {summarize_parameters(spec.parameters)}. {example}")
        try:
            result = spec.fn(**args)
        except Exception as e:  # noqa: BLE001 - dispatch must never raise
            return ToolResult(False, f"Tool crashed: {type(e).__name__}: {e}")
        if not isinstance(result, ToolResult):
            return ToolResult(False, f"Tool crashed: {spec.name} returned {type(result).__name__}, not ToolResult")
        return self._cap(result)


def make_finish_tool(properties: dict, required: list[str], description: str) -> ToolSpec:
    """The phase-specific 'finish' tool. The agent loop intercepts it; its fn is never called."""
    params = {"type": "object", "properties": properties, "required": list(required)}
    return ToolSpec(FINISH, description, params, lambda **_: ToolResult(True, "Phase finished."), "control")


# ---------------------------------------------------------------------- built-in tool schemas
def _obj(properties: dict, required: list[str] | None = None) -> dict:
    return {"type": "object", "properties": properties, "required": required or []}


def _s(description: str) -> dict:
    return {"type": "string", "description": description}


def _i(description: str) -> dict:
    return {"type": "integer", "description": description}


TOOL_SCHEMAS: dict[str, tuple[str, str, dict]] = {
    # name: (kind, description, parameters)
    "list_dir": ("read", "List files and folders as a tree (skips build/vendor dirs).", _obj({
        "path": _s("Directory relative to the repository root (default '.')."),
        "depth": _i("Levels to show, 1-4 (default 2)."),
    })),
    "find_files": ("read", "Find files whose path or name matches a glob, e.g. **/*config*.py.", _obj({
        "pattern": _s("Glob pattern, e.g. '*.py' or 'src/**/test_*.py'."),
    }, ["pattern"])),
    "search_code": ("read", "Search file contents (literal by default; set regex=true for a regex). "
                            "Returns up to 25 matches with 3 lines of context.", _obj({
        "query": _s("Text (or regex) to search for."),
        "regex": {"type": "boolean", "description": "Treat query as a regular expression (default false)."},
        "path": _s("File or directory to search (default '.')."),
        "file_glob": _s("Only search files matching this glob, e.g. '*.py'."),
    }, ["query"])),
    "view_file": ("read", "Show a file with line numbers (80 lines by default, max 200). Without start_line, "
                          "long files show an outline plus the first 40 lines.",
                  _obj({
                      "path": _s("File path relative to the repository root, or @scratch/<name>."),
                      "start_line": _i("First line to show (default 1)."),
                      "end_line": _i("Last line to show (default: start_line + 79)."),
                  }, ["path"])),
    "repo_map": ("read", "Outline of classes and functions (with line numbers) for a file or directory.", _obj({
        "path": _s("File or directory (default '.')."),
    })),
    "str_replace": ("write", "Replace one exact, unique occurrence of old_str with new_str in a file. "
                             "Include enough context to be unique.", _obj({
        "path": _s("File to edit."),
        "old_str": _s("Exact text to replace, including indentation; no view_file line numbers."),
        "new_str": _s("Replacement text."),
    }, ["path", "old_str", "new_str"])),
    "create_file": ("write", "Create a new file. Use @scratch/<name> for throwaway scripts.", _obj({
        "path": _s("Path of the new file."),
        "content": _s("Full file content."),
    }, ["path", "content"])),
    "run_command": ("exec", "Run a shell command in the repository root (non-interactive, time-limited). "
                            "Do not edit files with it.", _obj({
        "command": _s("Shell command to run."),
        "timeout_s": _i("Timeout in seconds (default 120, max 600)."),
    }, ["command"])),
    "run_tests": ("exec", "Run the project's tests (all, or the given files/test IDs) and summarise the results.",
                  _obj({"targets": _s("Space-separated test files or test IDs; empty = full suite.")})),
}


def build_registry(ws: Any, cfg: Any, test_runner: Any = None) -> ToolRegistry:
    """Register every built-in tool whose implementation exists, plus run_tests when a test runner is given."""
    from harness import repomap
    from harness.tools import edit_tools, exec_tools, read_tools

    impls: dict[str, Callable[..., ToolResult] | None] = {
        "list_dir": read_tools.list_dir,
        "find_files": read_tools.find_files,
        "search_code": read_tools.search_code,
        "view_file": read_tools.view_file,
        "repo_map": getattr(repomap, "repo_map", None),
        "str_replace": edit_tools.str_replace,
        "create_file": edit_tools.create_file,
        "run_command": exec_tools.run_command,
    }
    max_chars = getattr(getattr(cfg, "context", None), "max_tool_output_chars", None)
    registry = ToolRegistry(max_chars)
    for name, fn in impls.items():
        if fn is not None:
            kind, description, params = TOOL_SCHEMAS[name]
            registry.register(ToolSpec(name, description, params, functools.partial(fn, ws, cfg), kind))
    run_tests = getattr(test_runner, "run_tests_tool", None) if test_runner is not None else None
    if run_tests is not None:
        kind, description, params = TOOL_SCHEMAS["run_tests"]
        registry.register(ToolSpec("run_tests", description, params, run_tests, kind))
    return registry

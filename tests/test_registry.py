"""Tests for ToolRegistry, dispatch failure paths, the finish tool and build_registry."""
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from harness.textproto import INVALID_TOOL
from harness.tools.registry import (
    ALLOWED_SCHEMA_KEYS,
    ToolRegistry,
    ToolSpec,
    build_registry,
    make_finish_tool,
    validate_schema,
)
from harness.types import ToolCall, ToolResult
from harness.workspace import Workspace

ECHO_PARAMS = {"type": "object", "properties": {
    "text": {"type": "string", "description": "Text to echo."},
    "times": {"type": "integer", "description": "Repeat count (default 1)."},
}, "required": ["text"]}


def echo(text: str, times: int = 1) -> ToolResult:
    return ToolResult(True, text * int(times))


def boom(**_: object) -> ToolResult:
    raise RuntimeError("disk on fire")


@pytest.fixture()
def reg() -> ToolRegistry:
    r = ToolRegistry(max_output_chars=200)
    r.register(ToolSpec("echo", "Echo text.", ECHO_PARAMS, echo, "read"))
    r.register(ToolSpec("boom", "Always crashes.", {"type": "object", "properties": {}, "required": []}, boom, "exec"))
    return r


def call(name: str, args: object = None, parse_error: str | None = None) -> ToolCall:
    return ToolCall("c1", name, args if args is not None else {}, parse_error)


# ---------------------------------------------------------------- dispatch
def test_dispatch_success_and_unknown_params_dropped(reg: ToolRegistry) -> None:
    res = reg.dispatch(call("echo", {"text": "ab", "times": 2, "colour": "red"}))
    assert res.ok and res.output == "abab"


def test_unknown_tool(reg: ToolRegistry) -> None:
    res = reg.dispatch(call("teleport"))
    assert not res.ok and res.output == "Unknown tool 'teleport'. Available: echo, boom"


def test_invalid_text_protocol_call(reg: ToolRegistry) -> None:
    res = reg.dispatch(call(INVALID_TOOL, parse_error="tool block is not valid JSON"))
    assert not res.ok and "could not be parsed" in res.output and "```tool" in res.output


def test_parse_error_echoes_schema_and_example(reg: ToolRegistry) -> None:
    res = reg.dispatch(call("echo", parse_error="Expecting ',' delimiter"))
    assert not res.ok
    assert "not valid JSON (Expecting ',' delimiter)" in res.output
    assert "Expected parameters: text (string, required), times (integer)" in res.output
    example = res.output.split("Example: ", 1)[1]
    assert json.loads(example) == {"name": "echo", "arguments": {"text": "<text>"}}  # M4 schema echo


def test_non_dict_arguments(reg: ToolRegistry) -> None:
    res = reg.dispatch(call("echo", ["x"]))
    assert not res.ok and "must be a JSON object" in res.output


def test_missing_required(reg: ToolRegistry) -> None:
    res = reg.dispatch(call("echo", {"times": 3}))
    assert not res.ok and "Missing required parameter(s) for echo: text" in res.output
    assert "Example: " in res.output  # M4 schema echo
    assert not reg.dispatch(call("echo", {"text": None})).ok


def test_empty_string_counts_as_present(reg: ToolRegistry) -> None:
    assert reg.dispatch(call("echo", {"text": ""})).ok


def test_tool_crash(reg: ToolRegistry) -> None:
    res = reg.dispatch(call("boom"))
    assert not res.ok and res.output == "Tool crashed: RuntimeError: disk on fire"


def test_non_toolresult_return() -> None:
    r = ToolRegistry()
    r.register(ToolSpec("bad", "d", {"type": "object", "properties": {}, "required": []}, lambda: "oops", "read"))
    assert "returned str" in r.dispatch(call("bad")).output


def test_output_capped(reg: ToolRegistry) -> None:
    res = reg.dispatch(call("echo", {"text": "x", "times": 1000}))
    assert res.ok and "chars truncated" in res.output and len(res.output) < 260


# ---------------------------------------------------------------- registry behaviour
def test_register_rejects_bad_specs(reg: ToolRegistry) -> None:
    with pytest.raises(ValueError, match="already registered"):
        reg.register(ToolSpec("echo", "d", ECHO_PARAMS, echo, "read"))
    with pytest.raises(ValueError, match="unknown kind"):
        reg.register(ToolSpec("x", "d", ECHO_PARAMS, echo, "magic"))
    bad = {"type": "object", "properties": {"n": {"type": "integer", "default": 2}}, "required": []}
    with pytest.raises(ValueError, match="forbidden schema keys"):
        reg.register(ToolSpec("y", "d", bad, echo, "read"))
    with pytest.raises(ValueError):
        validate_schema({"type": "object", "additionalProperties": False})
    with pytest.raises(ValueError):
        validate_schema({"type": "array", "items": {"anyOf": []}})


def test_subset_and_schemas(reg: ToolRegistry) -> None:
    sub = reg.subset(["boom", "missing"])
    assert sub.names() == ["boom"] and "echo" not in sub
    assert sub.max_output_chars == 200
    schema = reg.openai_schemas()[0]
    assert schema == {"type": "function", "function": {"name": "echo", "description": "Echo text.",
                                                        "parameters": ECHO_PARAMS}}


def test_finish_tool() -> None:
    spec = make_finish_tool({"files": {"type": "array", "items": {"type": "string"}},
                             "confidence": {"type": "string", "enum": ["high", "medium", "low"]}},
                            ["files"], "End LOCALIZE.")
    assert spec.name == "finish" and spec.kind == "control"
    assert spec.parameters["required"] == ["files"]
    validate_schema(spec.parameters)
    r = ToolRegistry()
    r.register(spec)
    assert "Missing required parameter(s) for finish: files" in r.dispatch(call("finish", {})).output


# ---------------------------------------------------------------- build_registry
def _walk_keys(schema: dict) -> set:
    keys = set(schema)
    for prop in (schema.get("properties") or {}).values():
        keys |= _walk_keys(prop)
    if isinstance(schema.get("items"), dict):
        keys |= _walk_keys(schema["items"])
    return keys


def test_build_registry(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "a.py").write_text("x = 1\n")
    ws = Workspace(repo, tmp_path / "scratch")
    ws.write_scope = "repo"
    cfg = SimpleNamespace(context=SimpleNamespace(max_tool_output_chars=8000),
                          safety=SimpleNamespace(command_timeout_s=30, allow_edit_existing_tests=False))
    reg = build_registry(ws, cfg)
    assert reg.names() == ["list_dir", "find_files", "search_code", "view_file",
                           "str_replace", "create_file", "run_command"]
    for schema in reg.openai_schemas():
        params = schema["function"]["parameters"]
        assert _walk_keys(params) <= ALLOWED_SCHEMA_KEYS, schema["function"]["name"]
        assert "default" not in json.dumps(params).replace("(default", "")
    assert "x = 1" in reg.dispatch(call("view_file", {"path": "a.py"})).output
    assert reg.dispatch(call("str_replace", {"path": "a.py", "old_str": "x = 1", "new_str": "x = 2"})).ok
    assert reg.dispatch(call("run_command", {"command": "cat a.py"})).output.endswith("x = 2")

    runner = SimpleNamespace(run_tests_tool=lambda targets="": ToolResult(True, f"ran [{targets}]"))
    with_tests = build_registry(ws, cfg, test_runner=runner)
    assert with_tests.names()[-1] == "run_tests"
    assert with_tests.dispatch(call("run_tests", {"targets": "tests/x.py"})).output == "ran [tests/x.py]"

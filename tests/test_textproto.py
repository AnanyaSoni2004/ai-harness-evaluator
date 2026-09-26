"""Tests for the text-mode tool protocol."""
from harness.textproto import (
    INVALID_TOOL,
    format_tool_result,
    parse_tool_calls,
    render_tool_instructions,
)
from harness.types import ToolResult

SCHEMAS = [
    {"type": "function", "function": {
        "name": "view_file",
        "description": "Show a file with line numbers.",
        "parameters": {"type": "object", "properties": {
            "path": {"type": "string", "description": "File path."},
            "start_line": {"type": "integer", "description": "First line (default 1)."},
        }, "required": ["path"]},
    }},
    {"type": "function", "function": {
        "name": "finish",
        "description": "End this phase.",
        "parameters": {"type": "object", "properties": {
            "files": {"type": "array", "items": {"type": "string"}},
            "confidence": {"type": "string", "enum": ["high", "low"]},
        }, "required": []},
    }},
]


def _one(text: str):
    calls = parse_tool_calls(text, "c1")
    assert len(calls) == 1
    return calls[0]


def test_render_instructions() -> None:
    text = render_tool_instructions(SCHEMAS)
    assert "```tool" in text and '"arguments"' in text
    assert "- view_file: Show a file with line numbers." in text
    assert "path (string, required): File path." in text
    assert "start_line (integer, optional)" in text
    assert "files (array of string, optional)" in text
    assert "one of high|low" in text


def test_clean_tool_block() -> None:
    call = _one('```tool\n{"name": "view_file", "arguments": {"path": "a.py", "start_line": 3}}\n```')
    assert call.name == "view_file" and call.arguments == {"path": "a.py", "start_line": 3}
    assert call.parse_error is None and call.id == "c1_0"


def test_prose_around_tool_block() -> None:
    text = ('I will look at the file first.\n\n```tool\n{"name": "view_file", "arguments": {"path": "x.py"}}\n```\n'
            "Then I will decide.")
    assert _one(text).arguments == {"path": "x.py"}


def test_json_fence() -> None:
    call = _one('Here:\n```json\n{"name": "finish", "arguments": {"files": ["a.py"]}}\n```')
    assert call.name == "finish" and call.arguments["files"] == ["a.py"]


def test_json_fence_without_name_is_skipped() -> None:
    text = '```json\n{"files": []}\n```\n```json\n{"name": "finish", "arguments": {}}\n```'
    assert _one(text).name == "finish"


def test_trailing_comma_repaired() -> None:
    call = _one('```tool\n{"name": "view_file", "arguments": {"path": "a.py",},}\n```')
    assert call.arguments == {"path": "a.py"}


def test_args_and_parameters_keys() -> None:
    assert _one('```tool\n{"name": "view_file", "args": {"path": "b.py"}}\n```').arguments == {"path": "b.py"}
    assert _one('```tool\n{"name": "view_file", "parameters": {"path": "c.py"}}\n```').arguments == {"path": "c.py"}


def test_arguments_as_json_string() -> None:
    call = _one('```tool\n{"name": "view_file", "arguments": "{\\"path\\": \\"d.py\\"}"}\n```')
    assert call.arguments == {"path": "d.py"}


def test_bare_json_in_prose() -> None:
    text = 'Let me call {"name": "view_file", "arguments": {"path": "e.py", "note": "a } in a string"}} now.'
    call = _one(text)
    assert call.arguments == {"path": "e.py", "note": "a } in a string"}


def test_bare_json_skips_objects_without_name() -> None:
    text = 'Config is {"a": 1}. Call: {"name": "finish", "arguments": {}}'
    assert _one(text).name == "finish"


def test_garbage_returns_nothing() -> None:
    assert parse_tool_calls("I think the bug is in parse(). {not json at all", "c1") == []
    assert parse_tool_calls("", "c1") == []
    assert parse_tool_calls('{"name": }', "c1") == []


def test_broken_tool_block_is_invalid() -> None:
    call = _one('```tool\n{"name": "view_file", "arguments": {"path": "a.py"\n```')
    assert call.name == INVALID_TOOL and call.arguments == {}
    assert "not valid JSON" in call.parse_error


def test_tool_block_without_name_is_invalid() -> None:
    call = _one('```tool\n{"arguments": {"path": "a.py"}}\n```')
    assert call.name == INVALID_TOOL and "name" in call.parse_error


def test_unclosed_tool_block_still_parses() -> None:
    call = _one('```tool\n{"name": "view_file", "arguments": {"path": "a.py"}}')
    assert call.name == "view_file"


def test_non_object_arguments_flagged() -> None:
    call = _one('```tool\n{"name": "view_file", "arguments": ["a.py"]}\n```')
    assert call.name == "view_file" and call.parse_error


def test_at_most_one_call() -> None:
    text = ('```tool\n{"name": "view_file", "arguments": {"path": "1.py"}}\n```\n'
            '```tool\n{"name": "view_file", "arguments": {"path": "2.py"}}\n```')
    assert _one(text).arguments == {"path": "1.py"}


def test_format_tool_result() -> None:
    assert format_tool_result("view_file", ToolResult(True, "ok")) == "[tool_result name=view_file ok=true]\nok"
    assert format_tool_result("x", ToolResult(False, "bad")).startswith("[tool_result name=x ok=false]")

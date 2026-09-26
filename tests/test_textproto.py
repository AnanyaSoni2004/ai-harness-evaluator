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


# ---------------------------------------------------------------- Phase M2: reasoning output
from types import SimpleNamespace  # noqa: E402

import litellm  # noqa: E402

from harness.textproto import split_reasoning, strip_reasoning  # noqa: E402


def test_think_before_tool_block() -> None:
    text = ('<think>The user wants a.py. Maybe I should call '
            '```tool\n{"name": "list_dir", "arguments": {}}\n``` first? No.</think>\n'
            '```tool\n{"name": "view_file", "arguments": {"path": "a.py"}}\n```')
    call = _one(text)
    assert call.name == "view_file" and call.arguments == {"path": "a.py"}


def test_unclosed_think_is_removed() -> None:
    text = 'Answer first.\n<think>I am still reasoning about {"name": "view_file", "arguments": {}} and'
    assert strip_reasoning(text) == "Answer first."
    assert parse_tool_calls(text, "c1") == []


def test_other_reasoning_tags() -> None:
    text = ("<thinking>hmm</thinking>A <|begin_of_thought|>deep<|end_of_thought|>B "
            "<THINK>caps</THINK>C")
    clean, reasoning = split_reasoning(text)
    assert clean == "A B C"
    assert "hmm" in reasoning and "deep" in reasoning and "caps" in reasoning


def test_orphan_closing_tag() -> None:
    clean, reasoning = split_reasoning("the template opened the block... so x</think>\nFinal answer")
    assert clean == "Final answer" and "template opened" in reasoning


def test_no_reasoning_is_unchanged() -> None:
    assert split_reasoning("plain text") == ("plain text", "")
    assert split_reasoning("") == ("", "")


def _client(monkeypatch, message, finish_reason="stop", tool_mode="text"):
    from harness import llm as llm_mod
    from harness.config import load_config
    from harness.events import Trajectory
    from harness.llm import LLMClient
    from harness.types import Metrics

    monkeypatch.setenv("AI_API_KEY", "fake-key-for-tests-only")
    raw = SimpleNamespace(choices=[SimpleNamespace(message=message, finish_reason=finish_reason)],
                          usage=SimpleNamespace(prompt_tokens=5, completion_tokens=2))
    monkeypatch.setattr(litellm, "completion", lambda **kw: raw)
    llm_mod._TOOL_MODE_CACHE.clear()
    cfg = load_config()
    cfg.model.tool_mode = tool_mode
    warnings: list[str] = []
    ui = SimpleNamespace(thinking=lambda label: __import__("contextlib").nullcontext(),
                         warn=warnings.append, llm_call=lambda phase, usage: None)
    return LLMClient(cfg, Metrics(), Trajectory(None), ui), warnings


def test_separate_reasoning_content_field(monkeypatch) -> None:
    message = SimpleNamespace(
        content='<think>inline</think>```tool\n{"name": "view_file", "arguments": {"path": "b.py"}}\n```',
        reasoning_content="Long chain of thought.", tool_calls=None)
    client, warnings = _client(monkeypatch, message)
    resp = client.complete([], None, "fix")
    assert resp.tool_calls[0].arguments == {"path": "b.py"}
    assert "Long chain of thought." in resp.reasoning and "inline" in resp.reasoning
    assert "chain of thought" not in resp.text and "<think>" not in resp.text
    assert warnings == []


def test_native_mode_strips_think_from_text(monkeypatch) -> None:
    call = SimpleNamespace(id="c1", function=SimpleNamespace(name="view_file", arguments='{"path": "x.py"}'))
    message = SimpleNamespace(content="<think>plan</think>Looking at x.py", tool_calls=[call])
    client, _ = _client(monkeypatch, message, tool_mode="native")
    resp = client.complete([], None, "fix")
    assert resp.text == "Looking at x.py" and resp.reasoning == "plan"
    assert resp.tool_calls[0].name == "view_file"


def test_length_finish_warns(monkeypatch) -> None:
    message = SimpleNamespace(content="<think>overthinking forever", tool_calls=None)
    client, warnings = _client(monkeypatch, message, finish_reason="length")
    resp = client.complete([], None, "localize")
    assert resp.text == "" and resp.tool_calls == []
    assert len(warnings) == 1 and "cut off" in warnings[0]


# ---------------------------------------------------------------- Phase M4: robustness for smaller models
import json as _json  # noqa: E402
import re as _re  # noqa: E402

from harness.textproto import example_call, format_example  # noqa: E402


def test_example_call_uses_required_params() -> None:
    assert example_call(SCHEMAS[0]) == {"name": "view_file", "arguments": {"path": "<path>"}}


def test_example_call_types_and_enums() -> None:
    schema = {"name": "t", "parameters": {"type": "object", "properties": {
        "n": {"type": "integer"}, "flag": {"type": "boolean"}, "tags": {"type": "array", "items": {"type": "string"}},
        "level": {"type": "string", "enum": ["high", "low"]}, "opts": {"type": "object"},
    }, "required": ["n", "flag", "tags", "level", "opts"]}}
    assert example_call(schema)["arguments"] == {"n": 1, "flag": False, "tags": ["<tags>"], "level": "high", "opts": {}}


def test_example_call_without_required_uses_first_param() -> None:
    assert example_call(SCHEMAS[1])["arguments"] == {"files": ["<files>"]}
    assert example_call({"name": "noop", "parameters": {}}) == {"name": "noop", "arguments": {}}


def test_format_example_is_valid_json() -> None:
    text = format_example(SCHEMAS[0])
    assert text.startswith("Example: ")
    assert _json.loads(text[len("Example: "):])["name"] == "view_file"


def test_instructions_contain_parseable_one_shot_example() -> None:
    text = render_tool_instructions(SCHEMAS)
    block = _re.search(r"Example of a correct reply:\n(```tool\n.*?\n```)", text, _re.DOTALL).group(1)
    call = _one(block)
    assert call.name == "view_file" and call.arguments == {"path": "<path>"}
    assert len(block) / 3.5 < 80  # cheap: well under ~80 tokens
    assert "Example of a correct reply" not in render_tool_instructions([])


def test_first_json_object() -> None:
    from harness.textproto import first_json_object
    assert first_json_object('Sure!\n```json\n{"verdict": "approve", "problems": [],}\n```') == {
        "verdict": "approve", "problems": []}
    assert first_json_object("<think>{\"verdict\": \"revise\"}</think>{\"verdict\": \"approve\"}") == {
        "verdict": "approve"}
    assert first_json_object("no json here") is None and first_json_object("") is None

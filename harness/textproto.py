"""Text-mode tool protocol: render tool instructions and parse tool calls."""
from __future__ import annotations

import json
import re
from typing import Any

from harness.types import ToolCall, ToolResult

INVALID_TOOL = "__invalid__"
ARG_KEYS = ("arguments", "args", "parameters")

_TOOL_FENCE = re.compile(r"```tool[ \t]*\r?\n(.*?)(?:```|\Z)", re.DOTALL)
_JSON_FENCE = re.compile(r"```json[ \t]*\r?\n(.*?)```", re.DOTALL)
_TRAILING_COMMA = re.compile(r",\s*([}\]])")

_REASONING_TAGS = [("<think>", "</think>"), ("<thinking>", "</thinking>"),
                   ("<|begin_of_thought|>", "<|end_of_thought|>")]
_REASONING_BLOCKS = [re.compile(re.escape(o) + r"(.*?)(?:" + re.escape(c) + r"|\Z)", re.DOTALL | re.IGNORECASE)
                     for o, c in _REASONING_TAGS]
_ORPHAN_CLOSE = re.compile("|".join(re.escape(c) for _, c in _REASONING_TAGS), re.IGNORECASE)


def split_reasoning(text: str) -> tuple[str, str]:
    """Split model output into (visible text, reasoning) by removing thinking blocks.

    Handles <think>, <thinking> and <|begin_of_thought|> blocks, an unclosed opening tag (output cut
    off mid-thought: everything after it is reasoning), and a closing tag whose opening tag was part
    of the prompt template (everything before it is reasoning).
    """
    if not text:
        return "", ""
    thoughts: list[str] = []

    def _take(match: re.Match) -> str:
        thoughts.append(match.group(1).strip())
        return ""

    for pattern in _REASONING_BLOCKS:
        text = pattern.sub(_take, text)
    orphans = list(_ORPHAN_CLOSE.finditer(text))
    if orphans:
        last = orphans[-1]
        thoughts.insert(0, _ORPHAN_CLOSE.sub("", text[:last.start()]).strip())
        text = text[last.end():]
    return text.strip(), "\n\n".join(t for t in thoughts if t)


def strip_reasoning(text: str) -> str:
    """Remove thinking blocks (closed, unclosed, or orphan-closed) before tool-call parsing."""
    return split_reasoning(text)[0]


def _type_label(prop: dict) -> str:
    """Compact type for one parameter: '' for strings, 'int', 'bool', 'string[]', 'high|low' for enums."""
    if prop.get("enum"):
        return "|".join(str(v) for v in prop["enum"])
    kind = prop.get("type", "string")
    if kind == "array":
        return f"{(prop.get('items') or {}).get('type', 'string')}[]"
    return {"string": "", "integer": "int", "number": "number", "boolean": "bool", "object": "object"}.get(kind, kind)


def _signature(fn: dict) -> str:
    """'view_file(path, start_line?: int, end_line?: int)'; '?' marks optional parameters."""
    params = fn.get("parameters") or {}
    required = set(params.get("required") or [])
    parts = []
    for pname, prop in (params.get("properties") or {}).items():
        label = _type_label(prop)
        parts.append(f"{pname}{'' if pname in required else '?'}{': ' + label if label else ''}")
    return f"{fn.get('name', '?')}({', '.join(parts)})"


def _example_value(pname: str, prop: dict) -> Any:
    """A plausible placeholder value for one parameter, based on its schema."""
    if prop.get("enum"):
        return prop["enum"][0]
    kind = prop.get("type", "string")
    if kind in ("integer", "number"):
        return 1
    if kind == "boolean":
        return False
    if kind == "array":
        return [_example_value(pname, prop.get("items") or {})]
    if kind == "object":
        return {}
    return f"<{pname}>"


def example_call(schema: dict) -> dict:
    """A compact, valid example call for a tool: its required parameters (or the first one if none)."""
    fn = schema.get("function", schema)
    params = fn.get("parameters") or {}
    props = params.get("properties") or {}
    names = [n for n in (params.get("required") or []) if n in props] or list(props)[:1]
    return {"name": fn.get("name", "?"), "arguments": {n: _example_value(n, props[n]) for n in names}}


def format_example(schema: dict) -> str:
    """'Example: {"name": ..., "arguments": {...}}' for appending to argument errors."""
    return "Example: " + json.dumps(example_call(schema))


def render_tool_instructions(schemas: list[dict]) -> str:
    """Explain the ```tool block protocol and list each tool on one compact line (sent on every call)."""
    lines = ["TOOLS: reply with exactly one tool call per message, as one block:",
             "```tool", '{"name": "<tool>", "arguments": {...}}', "```"]
    if schemas:
        lines += ["Example:", "```tool", json.dumps(example_call(schemas[0])), "```"]
    lines.append("Available tools (? = optional):")
    for schema in schemas:
        fn = schema.get("function", schema)
        lines.append(f"- {_signature(fn)} — {fn.get('description', '').strip()}")
    return "\n".join(lines)


def _loads_lenient(raw: str) -> Any:
    """json.loads, retrying once with trailing commas removed. Raises ValueError on failure."""
    raw = raw.strip()
    try:
        return json.loads(raw)
    except ValueError:
        return json.loads(_TRAILING_COMMA.sub(r"\1", raw))


def _balanced_objects(text: str) -> list[str]:
    """Return every top-level balanced {...} substring, respecting JSON string quoting."""
    found: list[str] = []
    depth, start, in_str, escaped = 0, -1, False, False
    for i, ch in enumerate(text):
        if in_str:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"' and depth > 0:
            in_str = True
        elif ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}" and depth > 0:
            depth -= 1
            if depth == 0:
                found.append(text[start:i + 1])
    return found


def _to_call(obj: Any, call_id: str) -> ToolCall | None:
    """Turn a decoded object into a ToolCall, or None if it does not look like one."""
    if not isinstance(obj, dict) or not isinstance(obj.get("name"), str) or not obj["name"].strip():
        return None
    args: Any = {}
    for key in ARG_KEYS:
        if key in obj:
            args = obj[key]
            break
    if args is None:
        args = {}
    if isinstance(args, str):
        try:
            args = _loads_lenient(args) if args.strip() else {}
        except ValueError as e:
            return ToolCall(call_id, obj["name"].strip(), {}, parse_error=f"arguments string is not JSON: {e}")
    if not isinstance(args, dict):
        return ToolCall(call_id, obj["name"].strip(), {}, parse_error="arguments must be a JSON object")
    return ToolCall(call_id, obj["name"].strip(), args)


def parse_tool_calls(text: str, id_prefix: str) -> list[ToolCall]:
    """Extract at most one tool call from model text (```tool block, ```json block, then bare JSON)."""
    text = strip_reasoning(text)  # a tool block drafted inside <think> is not a real call
    if not text:
        return []
    call_id = f"{id_prefix}_0"

    tool_block = _TOOL_FENCE.search(text)
    if tool_block:
        body = tool_block.group(1)
        try:
            obj = _loads_lenient(body)
        except ValueError as e:
            return [ToolCall(call_id, INVALID_TOOL, {}, parse_error=f"tool block is not valid JSON: {e}")]
        call = _to_call(obj, call_id)
        if call is None:
            return [ToolCall(call_id, INVALID_TOOL, {},
                             parse_error='tool block must be an object with a "name" and "arguments"')]
        return [call]

    for block in _JSON_FENCE.finditer(text):
        try:
            call = _to_call(_loads_lenient(block.group(1)), call_id)
        except ValueError:
            continue
        if call is not None:
            return [call]

    for candidate in _balanced_objects(text):
        if '"name"' not in candidate:
            continue
        try:
            call = _to_call(_loads_lenient(candidate), call_id)
        except ValueError:
            continue
        if call is not None:
            return [call]
    return []


def first_json_object(text: str) -> dict | None:
    """The first balanced {...} in text (reasoning stripped, trailing commas repaired) that parses as a dict."""
    for candidate in _balanced_objects(strip_reasoning(text or "")):
        try:
            obj = _loads_lenient(candidate)
        except ValueError:
            continue
        if isinstance(obj, dict):
            return obj
    return None


def format_tool_result(name: str, result: ToolResult) -> str:
    """Render a tool result as the user message sent back in text mode."""
    return f"[tool_result name={name} ok={'true' if result.ok else 'false'}]\n{result.output}"

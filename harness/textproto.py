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


def _type_label(prop: dict) -> str:
    """Human-readable type for one JSON-schema property."""
    kind = prop.get("type", "any")
    if kind == "array":
        item_type = (prop.get("items") or {}).get("type", "any")
        kind = f"array of {item_type}"
    if prop.get("enum"):
        kind += " one of " + "|".join(str(v) for v in prop["enum"])
    return kind


def render_tool_instructions(schemas: list[dict]) -> str:
    """Explain the ```tool block protocol and list every tool with its parameters."""
    lines = [
        "TOOLS",
        "To use a tool, reply with exactly one block (one tool call per reply):",
        "```tool",
        '{"name": "<tool>", "arguments": {...}}',
        "```",
        "The block must contain valid JSON. You will receive the result in the next message.",
        "",
        "Available tools:",
    ]
    for schema in schemas:
        fn = schema.get("function", schema)
        lines.append(f"- {fn.get('name', '?')}: {fn.get('description', '').strip()}")
        params = fn.get("parameters") or {}
        required = set(params.get("required") or [])
        for pname, prop in (params.get("properties") or {}).items():
            flag = "required" if pname in required else "optional"
            desc = (prop.get("description") or "").strip()
            lines.append(f"    - {pname} ({_type_label(prop)}, {flag})" + (f": {desc}" if desc else ""))
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


def format_tool_result(name: str, result: ToolResult) -> str:
    """Render a tool result as the user message sent back in text mode."""
    return f"[tool_result name={name} ok={'true' if result.ok else 'false'}]\n{result.output}"

"""Token estimation and history compaction."""
from __future__ import annotations

import json
import math

CHARS_PER_TOKEN = 3.5
TOOL_KEEP_CHARS = 300
ASSISTANT_KEEP_CHARS = 500
PROTECTED_HEAD = 2  # messages[0] = system, messages[1] = task
ELIDED_MARKER = "\n...[older output elided to save context; re-run the tool if you need it]"
TRIMMED_MARKER = "\n...[trimmed]"


def estimate_tokens(messages: list[dict]) -> int:
    """Deterministic token estimate: serialized length / 3.5, rounded up (no tokenizer needed)."""
    return math.ceil(len(json.dumps(messages, default=str)) / CHARS_PER_TOKEN)


def _is_tool_result(msg: dict) -> bool:
    """A native tool message, or a text-mode tool result sent as a user message."""
    content = msg.get("content")
    return msg.get("role") == "tool" or (
        msg.get("role") == "user" and isinstance(content, str) and content.startswith("[tool_result"))


def _shorten(msg: dict, keep: int, marker: str) -> dict:
    """Copy of msg with its string content cut to `keep` chars plus a marker (no-op if already short)."""
    content = msg.get("content")
    if not isinstance(content, str) or content.endswith(marker) or len(content) <= keep + len(marker):
        return msg
    return {**msg, "content": content[:keep] + marker}


def compact(messages: list[dict], keep_recent: int, budget_tokens: int | None = None) -> list[dict]:
    """Shrink old history without deleting messages, so tool_call/tool-result pairs stay intact.

    Pass 1: tool results older than the last `keep_recent` messages keep their first 300 chars.
    Pass 2 (only if `budget_tokens` is given and still exceeded): older assistant text keeps 500 chars.
    messages[0] (system) and messages[1] (task) are never touched. The input list is not modified.
    """
    result = list(messages)
    cutoff = max(PROTECTED_HEAD, len(result) - max(keep_recent, 0))
    for i in range(PROTECTED_HEAD, cutoff):
        if _is_tool_result(result[i]):
            result[i] = _shorten(result[i], TOOL_KEEP_CHARS, ELIDED_MARKER)
    if budget_tokens is not None and estimate_tokens(result) > budget_tokens:
        for i in range(PROTECTED_HEAD, cutoff):
            if result[i].get("role") == "assistant":
                result[i] = _shorten(result[i], ASSISTANT_KEEP_CHARS, TRIMMED_MARKER)
    return result

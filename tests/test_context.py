"""Tests for token estimation and history compaction."""
import copy
import json

from harness.context import ELIDED_MARKER, TRIMMED_MARKER, compact, estimate_tokens

BIG = "x" * 5000


def native_history(rounds: int = 6) -> list[dict]:
    msgs = [{"role": "system", "content": "SYSTEM " + BIG}, {"role": "user", "content": "TASK " + BIG}]
    for i in range(rounds):
        msgs.append({"role": "assistant", "content": f"thinking {i} " + BIG,
                     "tool_calls": [{"id": f"call_{i}", "type": "function",
                                     "function": {"name": "view_file", "arguments": json.dumps({"path": f"f{i}.py"})}}]})
        msgs.append({"role": "tool", "tool_call_id": f"call_{i}", "name": "view_file", "content": f"out {i} " + BIG})
    return msgs


def text_history(rounds: int = 6) -> list[dict]:
    msgs = [{"role": "system", "content": "SYSTEM"}, {"role": "user", "content": "TASK"}]
    for i in range(rounds):
        msgs.append({"role": "assistant", "content": f"```tool\n{{\"name\": \"view_file\"}}\n```"})
        msgs.append({"role": "user", "content": f"[tool_result name=view_file ok=true]\n{i} " + BIG})
    return msgs


def test_estimate_tokens() -> None:
    msgs = [{"role": "user", "content": "a" * 350}]
    expected = -(-len(json.dumps(msgs)) // 3.5)
    assert estimate_tokens(msgs) == int(expected)
    assert estimate_tokens([]) == 1  # "[]" is 2 chars
    assert estimate_tokens(msgs) == estimate_tokens(copy.deepcopy(msgs))  # deterministic


def test_compact_shrinks_old_tool_results_only() -> None:
    msgs = native_history()
    out = compact(msgs, keep_recent=4)
    assert estimate_tokens(out) < estimate_tokens(msgs)
    old_tool = out[3]
    assert old_tool["content"].startswith("out 0 ") and old_tool["content"].endswith(ELIDED_MARKER)
    assert len(old_tool["content"]) == 300 + len(ELIDED_MARKER)
    assert out[-1] == msgs[-1] and out[-3] == msgs[-3]  # last keep_recent messages untouched
    assert out[2]["content"] == msgs[2]["content"]  # assistant text untouched without a budget


def test_first_two_messages_intact() -> None:
    msgs = native_history()
    out = compact(msgs, keep_recent=0, budget_tokens=1)
    assert out[0] == msgs[0] and out[1] == msgs[1]


def test_tool_call_pairing_preserved() -> None:
    msgs = native_history()
    out = compact(msgs, keep_recent=2, budget_tokens=1)
    assert len(out) == len(msgs)
    for before, after in zip(msgs, out):
        assert before["role"] == after["role"]
        assert before.get("tool_calls") == after.get("tool_calls")
        assert before.get("tool_call_id") == after.get("tool_call_id")
    call_ids = [c["id"] for m in out if m["role"] == "assistant" for c in m["tool_calls"]]
    result_ids = [m["tool_call_id"] for m in out if m["role"] == "tool"]
    assert call_ids == result_ids


def test_pass_two_only_when_over_budget() -> None:
    msgs = native_history()
    roomy = compact(msgs, keep_recent=4, budget_tokens=10**9)
    assert roomy[2]["content"] == msgs[2]["content"]
    tight = compact(msgs, keep_recent=4, budget_tokens=1)
    assert tight[2]["content"].endswith(TRIMMED_MARKER) and len(tight[2]["content"]) == 500 + len(TRIMMED_MARKER)
    assert estimate_tokens(tight) < estimate_tokens(roomy)


def test_text_mode_tool_results() -> None:
    msgs = text_history()
    out = compact(msgs, keep_recent=4)
    assert out[3]["content"].startswith("[tool_result name=view_file ok=true]")
    assert out[3]["content"].endswith(ELIDED_MARKER)
    assert out[-1] == msgs[-1]
    assert out[1] == {"role": "user", "content": "TASK"}  # plain user messages are not tool results


def test_input_not_mutated_and_idempotent() -> None:
    msgs = native_history()
    snapshot = copy.deepcopy(msgs)
    once = compact(msgs, keep_recent=4, budget_tokens=1)
    assert msgs == snapshot
    assert compact(once, keep_recent=4, budget_tokens=1) == once


def test_short_and_none_content_left_alone() -> None:
    msgs = [{"role": "system", "content": "s"}, {"role": "user", "content": "t"},
            {"role": "assistant", "content": None, "tool_calls": [{"id": "a", "type": "function",
                                                                    "function": {"name": "x", "arguments": "{}"}}]},
            {"role": "tool", "tool_call_id": "a", "name": "x", "content": "short"},
            {"role": "user", "content": "recent"}]
    assert compact(msgs, keep_recent=1, budget_tokens=1) == msgs

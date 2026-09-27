"""Tests for AgentLoop with FakeLLM (native and text tool modes)."""
import json
import time
from pathlib import Path

import pytest

from harness.agent import BLOCKED_REPEAT, FORCE_FINISH, NUDGE, WRAP_UP, AgentLoop
from harness.config import load_config
from harness.context import ELIDED_MARKER
from harness.events import NullUI, Trajectory
from harness.llm_fake import FakeLLM
from harness.textproto import parse_tool_calls
from harness.tools.registry import build_registry, make_finish_tool
from harness.types import BudgetExceeded, ContextOverflow, LLMResponse, Metrics, ToolCall, Usage
from harness.workspace import Workspace

_ids = iter(range(10**6))


def call(name: str, parse_error: str | None = None, **args) -> ToolCall:
    return ToolCall(f"call_{next(_ids)}", name, args, parse_error)


def resp(*calls: ToolCall, text: str = "", reasoning: str = "") -> LLMResponse:
    return LLMResponse(text, list(calls), Usage(10, 5), "tool_calls" if calls else "stop", reasoning)


def text_resp(text: str) -> LLMResponse:
    """What the real client returns in text mode: the raw text plus the parsed call."""
    return LLMResponse(text, parse_tool_calls(text, "t"), Usage(10, 5), "stop")


def tool_block(name: str, **args) -> str:
    return "```tool\n" + json.dumps({"name": name, "arguments": args}) + "\n```"


@pytest.fixture()
def cfg():
    c = load_config()
    c.model.max_consecutive_parse_failures = 3
    return c


@pytest.fixture()
def ws(tmp_path: Path) -> Workspace:
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "a.py").write_text("def f():\n    return 1\n")
    (repo / "big.py").write_text("".join(f"line_{i} = {i}  # {'pad' * 20}\n" for i in range(240)))
    w = Workspace(repo, tmp_path / "scratch")
    w.write_scope = "repo"
    return w


def make_loop(ws, cfg, script, mode="native", tmp_path=None, **kw):
    registry = build_registry(ws, cfg)
    registry.register(make_finish_tool({"summary": {"type": "string", "description": "What was found."}},
                                       ["summary"], "End this phase and report its result."))
    llm = FakeLLM(script, tool_mode=mode, cfg=kw.pop("llm_cfg", None))
    traj = Trajectory(tmp_path / "t.jsonl" if tmp_path else None)
    loop = AgentLoop(llm, registry, "You are a test agent.", "Find the bug.", "localize", cfg, llm.metrics,
                     traj, **kw)
    return loop, llm


def contents(messages: list) -> str:
    return "\n".join(str(m.get("content")) for m in messages)


# ---------------------------------------------------------------- happy paths
def test_native_view_then_finish(ws, cfg, tmp_path) -> None:
    loop, llm = make_loop(ws, cfg, [resp(call("view_file", path="a.py")), resp(call("finish", summary="f() in a.py"))],
                          tmp_path=tmp_path)
    result = loop.run()
    assert result.finished and result.result == {"summary": "f() in a.py"} and result.steps == 2
    second = llm.calls[1]
    assert second[0] == {"role": "system", "content": "You are a test agent."}
    assert second[2]["tool_calls"][0]["function"]["name"] == "view_file"
    assert second[3]["role"] == "tool" and "return 1" in second[3]["content"]
    assert second[3]["tool_call_id"] == second[2]["tool_calls"][0]["id"]
    assert llm.tools_seen[0] and {t["function"]["name"] for t in llm.tools_seen[0]} >= {"view_file", "finish"}
    assert llm.metrics.tool_calls == 2 and llm.metrics.llm_calls == 2
    events = [json.loads(l)["event"] for l in (tmp_path / "t.jsonl").read_text().splitlines()]
    assert events == ["phase_start", "tool_call", "phase_finish"]


def test_text_mode_view_then_finish(ws, cfg) -> None:
    script = [text_resp("Let me look.\n" + tool_block("view_file", path="a.py")),
              text_resp(tool_block("finish", summary="done"))]
    loop, llm = make_loop(ws, cfg, script, mode="text")
    result = loop.run()
    assert result.finished and result.result == {"summary": "done"}
    system = llm.calls[0][0]["content"]
    assert system.startswith("You are a test agent.") and "```tool" in system and "- view_file(" in system
    assert llm.tools_seen == [None, None]  # no native tools in text mode
    second = llm.calls[1]
    assert second[2] == {"role": "assistant", "content": script[0].text}
    assert second[3]["role"] == "user" and second[3]["content"].startswith("[tool_result name=view_file ok=true]")


def test_native_finish_with_other_call_runs_others_first(ws, cfg) -> None:
    loop, llm = make_loop(ws, cfg, [resp(call("view_file", path="a.py"), call("finish", summary="x"))])
    result = loop.run()
    assert result.finished and llm.metrics.tool_calls == 2 and result.steps == 1


# ---------------------------------------------------------------- recovery
def test_bad_json_feedback_then_recovery(ws, cfg) -> None:
    script = [resp(call("view_file", parse_error="Expecting ',' delimiter")), resp(call("finish", summary="ok"))]
    loop, llm = make_loop(ws, cfg, script)
    assert loop.run().finished
    feedback = llm.calls[1][3]["content"]
    assert "not valid JSON" in feedback and "Example: " in feedback


def test_finish_missing_required_field(ws, cfg) -> None:
    loop, llm = make_loop(ws, cfg, [resp(call("finish")), resp(call("finish", summary="now complete"))])
    result = loop.run()
    assert result.finished and result.result == {"summary": "now complete"}
    assert "missing required field(s): summary" in llm.calls[1][3]["content"]


def test_no_tool_calls_nudges_then_gives_up(ws, cfg) -> None:
    loop, llm = make_loop(ws, cfg, [resp(text="hmm"), resp(text="thinking"), resp(text="I believe it is f()")])
    result = loop.run()
    assert not result.finished and result.reason == "no_tool_calls"
    assert result.result == {"text": "I believe it is f()"}
    assert llm.calls[1][-1] == {"role": "user", "content": NUDGE}


def test_protocol_failure_after_three_malformed_replies(ws, cfg) -> None:
    broken = text_resp('```tool\n{"name": "view_file", "arguments": {"path": "a.py"\n```')
    loop, llm = make_loop(ws, cfg, [broken, broken, broken], mode="text")
    result = loop.run()
    assert not result.finished and result.reason == "protocol_failure" and result.steps == 3
    assert "could not be parsed" in llm.calls[1][-1]["content"]  # the model got feedback each time


def test_parse_failure_counter_resets_on_valid_call(ws, cfg) -> None:
    broken = text_resp('```tool\n{"name": "view_file",\n```')
    script = [broken, broken, text_resp(tool_block("view_file", path="a.py")), broken, broken,
              text_resp(tool_block("finish", summary="ok"))]
    loop, _ = make_loop(ws, cfg, script, mode="text")
    assert loop.run().finished


def test_context_overflow_compacts_and_retries(ws, cfg) -> None:
    state = {"n": 0}

    def overflow_once(messages):
        state["n"] += 1
        raise ContextOverflow("too long")

    loop, llm = make_loop(ws, cfg, [overflow_once, resp(call("finish", summary="ok"))])
    assert loop.run().finished and state["n"] == 1

    loop2, _ = make_loop(ws, cfg, [overflow_once, overflow_once])
    result = loop2.run()
    assert not result.finished and result.reason == "context_overflow"


# ---------------------------------------------------------------- loop detection
def test_third_identical_call_is_blocked(ws, cfg) -> None:
    same = lambda: call("view_file", path="a.py")  # noqa: E731
    loop, llm = make_loop(ws, cfg, [resp(same()), resp(same()), resp(same()), resp(call("finish", summary="x"))])
    assert loop.run().finished
    assert llm.calls[3][-1]["content"] == BLOCKED_REPEAT
    assert "return 1" in llm.calls[2][-1]["content"]  # the second identical call still ran


def test_successful_edit_resets_loop_detection(ws, cfg) -> None:
    run = lambda: call("run_command", command="cat a.py")  # noqa: E731
    edit = call("str_replace", path="a.py", old_str="return 1", new_str="return 2")
    script = [resp(run()), resp(run()), resp(edit), resp(run()), resp(run()), resp(run()),
              resp(call("finish", summary="x"))]
    loop, llm = make_loop(ws, cfg, script)
    assert loop.run().finished
    assert "return 2" in llm.calls[4][-1]["content"]  # 4th identical run after the edit is allowed
    assert "return 2" in llm.calls[5][-1]["content"]
    assert llm.calls[6][-1]["content"] == BLOCKED_REPEAT  # third identical run since the last change


# ---------------------------------------------------------------- budgets
def test_forced_finish_at_max_steps(ws, cfg) -> None:
    script = [resp(call("view_file", path="a.py", start_line=i)) for i in (1, 2)]
    script += [resp(call("list_dir", depth=i)) for i in (1, 2)]
    script += [resp(call("finish", summary="best guess"))]
    loop, llm = make_loop(ws, cfg, script, max_steps=4)
    result = loop.run()
    assert result.finished and result.reason == "max_steps" and result.result == {"summary": "best guess"}
    assert result.steps == 5
    assert [t["function"]["name"] for t in llm.tools_seen[-1]] == ["finish"]
    assert llm.calls[-1][-1] == {"role": "user", "content": FORCE_FINISH}
    assert {"role": "user", "content": WRAP_UP} in llm.calls[1]  # 3 calls left after step 1 of 4


def test_forced_finish_not_obeyed(ws, cfg) -> None:
    script = [resp(call("list_dir", depth=1)), resp(call("list_dir", depth=2)), resp(text="no")]
    loop, _ = make_loop(ws, cfg, script, max_steps=2)
    result = loop.run()
    assert not result.finished and result.reason == "max_steps"


def test_forced_finish_text_mode_lists_only_finish(ws, cfg) -> None:
    script = [text_resp(tool_block("list_dir")), text_resp(tool_block("finish", summary="s"))]
    loop, llm = make_loop(ws, cfg, script, mode="text", max_steps=1)
    assert loop.run().finished
    final_system = llm.calls[-1][0]["content"]
    assert "- finish(" in final_system and "- view_file(" not in final_system


def test_deadline_passed_forces_finish(ws, cfg) -> None:
    loop, llm = make_loop(ws, cfg, [resp(call("finish", summary="late"))], deadline=time.monotonic() - 1)
    result = loop.run()
    assert result.finished and result.reason == "deadline" and len(llm.calls) == 1
    assert [t["function"]["name"] for t in llm.tools_seen[0]] == ["finish"]


def test_budget_exceeded_propagates(ws, cfg) -> None:
    cfg.budgets.max_llm_calls = 1
    loop, _ = make_loop(ws, cfg, [resp(call("list_dir")), resp(call("finish", summary="x"))], llm_cfg=cfg)
    with pytest.raises(BudgetExceeded):
        loop.run()


# ---------------------------------------------------------------- context
def test_compaction_triggered(ws, cfg, tmp_path) -> None:
    cfg.context.working_budget_tokens = 3000
    cfg.context.keep_recent_messages = 2
    script = [resp(call("view_file", path="big.py", start_line=s)) for s in (1, 60, 120)]
    script += [resp(call("finish", summary="x"))]
    loop, llm = make_loop(ws, cfg, script, tmp_path=tmp_path)
    assert loop.run().finished
    last = llm.calls[-1]
    assert ELIDED_MARKER in contents(last)
    assert last[-1]["content"] and ELIDED_MARKER not in last[-1]["content"]  # most recent output kept
    assert "context_compacted" in (tmp_path / "t.jsonl").read_text()


def test_reasoning_never_enters_history(ws, cfg) -> None:
    script = [resp(call("view_file", path="a.py"), text="Looking at a.py", reasoning="SECRET_CHAIN_OF_THOUGHT"),
              resp(call("finish", summary="x"))]
    loop, llm = make_loop(ws, cfg, script)
    loop.run()
    assert "SECRET_CHAIN_OF_THOUGHT" not in contents(llm.calls[1])
    assert "Looking at a.py" in contents(llm.calls[1])


# ---------------------------------------------------------------- provider-rejected tool calls
def rejected(name: str, **args) -> LLMResponse:
    generation = json.dumps({"name": name, "arguments": args})
    return LLMResponse(generation, [call(name, **args)], Usage(0, 0), "tool_use_failed")


def read_only_loop(ws, cfg, script):
    registry = build_registry(ws, cfg).subset(["view_file", "list_dir"])
    registry.register(make_finish_tool({"summary": {"type": "string"}}, ["summary"], "End."))
    llm = FakeLLM(script)
    loop = AgentLoop(llm, registry, "sys", "task", "localize", cfg, llm.metrics, Trajectory(None))
    return loop, llm


def test_rejected_call_gets_feedback_not_a_native_tool_call(ws, cfg) -> None:
    loop, llm = read_only_loop(ws, cfg, [rejected("str_replace", path="a.py", old_str="1", new_str="2"),
                                         resp(call("finish", summary="found it"))])
    result = loop.run()
    assert result.finished and result.result == {"summary": "found it"}
    history = llm.calls[1][2:]
    assert all("tool_calls" not in m for m in history)  # the rejected call is never echoed as a native call
    assert history[-1]["role"] == "user" and "rejected by the API" in history[-1]["content"]
    assert "Unknown tool 'str_replace'. Available: view_file, list_dir, finish" in history[-1]["content"]
    assert (ws.repo_root / "a.py").read_text() == "def f():\n    return 1\n"  # nothing was edited


def test_rejected_but_valid_call_is_answered(ws, cfg) -> None:
    loop, llm = read_only_loop(ws, cfg, [rejected("view_file", path="a.py"), resp(call("finish", summary="x"))])
    assert loop.run().finished
    assert "return 1" in llm.calls[1][-1]["content"]


def test_rejected_finish_with_valid_fields_finishes(ws, cfg) -> None:
    loop, _ = read_only_loop(ws, cfg, [rejected("finish", summary="done anyway")])
    result = loop.run()
    assert result.finished and result.result == {"summary": "done anyway"}


def test_repeated_rejections_end_with_protocol_failure(ws, cfg) -> None:
    loop, _ = read_only_loop(ws, cfg, [rejected("str_replace", path="a.py") for _ in range(3)])
    result = loop.run()
    assert not result.finished and result.reason == "protocol_failure" and result.steps == 3


def test_unparseable_rejected_reply_gets_native_friendly_feedback(ws, cfg) -> None:
    garbled = LLMResponse("Search tests for remove error.",
                          [ToolCall("x", "__invalid__", {}, "the API could not use your reply as a tool call")],
                          Usage(0, 0), "tool_use_failed")
    loop, llm = read_only_loop(ws, cfg, [garbled, resp(call("finish", summary="ok"))])
    assert loop.run().finished
    feedback = llm.calls[1][-1]["content"]
    assert "could not use your last reply" in feedback and "```tool" not in feedback


def test_tool_result_preview_skips_echoed_command(ws, cfg) -> None:
    class Rec(NullUI):
        def __init__(self) -> None:
            self.results: list[str] = []

        def tool_result(self, ok: bool, summary: str) -> None:
            self.results.append(summary)
    ui = Rec()
    loop, _ = make_loop(ws, cfg, [resp(call("run_command", command="echo hello")), resp(call("finish", summary="x"))],
                        ui=ui)
    loop.run()
    assert ui.results[0] and not ui.results[0].startswith("$ ")  # the output, not the echoed command line

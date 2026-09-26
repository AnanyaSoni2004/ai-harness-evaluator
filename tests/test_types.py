"""Tests for the shared contracts in harness.types."""
import json
from pathlib import Path

import pytest

from harness.types import (
    BudgetExceeded,
    ContextOverflow,
    FatalLLMError,
    HarnessError,
    IssueSpec,
    LLMResponse,
    Metrics,
    PhaseMetrics,
    PhaseResult,
    ProcessResult,
    RunState,
    TestRun,
    ToolCall,
    ToolResult,
    Usage,
)


def _test_run() -> TestRun:
    return TestRun(
        command="python -m pytest -q", exit_code=1, passed=3, failed=1, errors=0,
        failing_ids=["tests/test_a.py::test_x"], duration_s=0.5, timed_out=False,
        env_problem=False, output_tail="1 failed, 3 passed",
    )


def test_construct_model_io() -> None:
    call = ToolCall(id="c1", name="view_file", arguments={"path": "a.py"})
    assert call.parse_error is None
    resp = LLMResponse(text="", tool_calls=[call], usage=Usage(5, 2))
    assert resp.finish_reason is None and resp.usage.completion_tokens == 2
    assert Usage().prompt_tokens == 0


def test_construct_tool_and_process() -> None:
    a, b = ToolResult(True, "ok"), ToolResult(False, "no")
    a.data["x"] = 1
    assert b.data == {}  # mutable default is not shared
    pr = ProcessResult(exit_code=None, output="", timed_out=True, duration_s=1.0)
    assert pr.timed_out
    assert _test_run().failing_ids == ["tests/test_a.py::test_x"]


def test_issue_and_phase_defaults() -> None:
    issue = IssueSpec(raw_text="Bug: crash")
    assert issue.kind == "bug" and issue.error_messages == []
    other = IssueSpec(raw_text="x")
    issue.mentioned_paths.append("a.py")
    assert other.mentioned_paths == []
    pr = PhaseResult(phase="localize", finished=True, result={"files": []}, steps=3)
    assert pr.reason == ""


def test_metrics_accounting() -> None:
    m = Metrics()
    m.add_llm("localize", Usage(100, 20))
    m.add_llm("fix", Usage(50, 10))
    m.add_tool("fix")
    m.add_tool("fix")
    m.add_time("fix", 1.5)
    assert m.total_tokens == 180
    assert m.llm_calls == 2 and m.tool_calls == 2
    assert isinstance(m.per_phase["fix"], PhaseMetrics)
    d = m.to_dict()
    assert d["total"]["total_tokens"] == 180
    assert d["per_phase"]["fix"]["seconds"] == 1.5
    json.dumps(d)


def test_run_state_to_dict_is_json() -> None:
    state = RunState(run_id="r1", repo=Path("/tmp/repo"), issue=IssueSpec(raw_text="x"))
    state.baseline_full = _test_run()
    state.attempts.append({"n": 1, "passed": False, "files": [Path("a.py")]})
    d = state.to_dict()
    text = json.dumps(d)
    assert d["status"] == "running"
    assert d["repo"] == "/tmp/repo"
    assert d["baseline_full"]["failed"] == 1
    assert d["baseline_targeted"] is None
    assert "tests/test_a.py::test_x" in text


def test_exception_hierarchy() -> None:
    for exc in (FatalLLMError, ContextOverflow, BudgetExceeded):
        assert issubclass(exc, HarnessError)
    with pytest.raises(HarnessError):
        raise BudgetExceeded("tokens")

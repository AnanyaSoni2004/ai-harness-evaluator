"""Per-issue caps and early abandonment: runaway runs stop as a NORMAL outcome, never an error."""
import pytest

from harness.llm_fake import FakeLLM
from harness.orchestrator import Orchestrator
from harness.types import LLMResponse, ToolCall, Usage
from test_failure_paths import CALC, PY, cfg, r, repo, responder  # noqa: F401 (fixtures)

FOREVER = 1000  # the fake model never runs out of replies


def run(cfg, repo, **overrides):  # noqa: F811
    llm = FakeLLM([responder(repo, **overrides)] * FOREVER, cfg=cfg)  # cfg: enforce the call/token caps
    state, run_dir = Orchestrator(cfg, llm, python_exe=PY).solve(repo, "calc.py: add(1, 2) returns -1")
    return state, run_dir, llm


def add_test(repo) -> None:  # noqa: F811
    """A test that passes at baseline and fails on the wrong edits below, so those attempts fail VERIFY."""
    (repo / "tests" / "test_calc.py").write_text("from calc import add\n\n\ndef test_add_zero():\n    assert add(5, 0) == 5\n")


def assert_abandoned_normally(state, run_dir, llm, reason: str) -> None:
    fix_attempts = [a for a in state.attempts if a["kind"] in ("fix", "rescue")]
    assert len(fix_attempts) <= 3, [a["kind"] for a in fix_attempts]
    assert llm.metrics.total_tokens < 45000, llm.metrics.total_tokens  # under the soft budget
    assert state.status in ("unverified", "no_fix"), state.status  # a normal outcome, not an error
    assert reason in state.stop_reason
    report = (run_dir / "report.md").read_text()
    assert f"**Stopped early:** {state.stop_reason}" in report and "ERROR" not in report.splitlines()[0]


def test_noop_fix_forever_is_abandoned(repo, cfg) -> None:  # noqa: F811
    """The requested case: a model whose FIX phase finishes without changing anything, forever."""
    cfg.phases.enable_rescue = True
    state, run_dir, llm = run(cfg, repo, fix=lambda m: r("finish", summary="done (changed nothing)"))
    assert_abandoned_normally(state, run_dir, llm, "two consecutive fix attempts produced no changes")
    assert state.status == "no_fix" and len(state.attempts) == 2
    assert (repo / "calc.py").read_text() == CALC


def test_same_diff_twice_is_abandoned(repo, cfg) -> None:  # noqa: F811
    cfg.phases.enable_rescue = True
    add_test(repo)

    def same_wrong_edit(messages):  # makes one wrong edit, then keeps "finishing" without changing it
        if "# checked" not in (repo / "calc.py").read_text():
            return r("str_replace", path="calc.py", old_str="return a - b", new_str="return a * b  # checked")
        return r("finish", summary="nothing more to do")

    state, run_dir, llm = run(cfg, repo, fix=same_wrong_edit)
    assert_abandoned_normally(state, run_dir, llm, "produced the same diff")
    assert state.attempts[0]["diff_hash"] == state.attempts[1]["diff_hash"] != ""
    assert (repo / "calc.py").read_text() == CALC  # the failing change was not kept


def test_no_tool_call_rule_on_its_own(repo, cfg) -> None:  # noqa: F811
    """Each attempt makes a different wrong edit, then stops calling tools: the no-tool-call rule stops it."""
    cfg.phases.enable_rescue = True
    add_test(repo)
    wrong = iter(["return a * b", "return a * b + 0", "return b - a"])
    pending = {"edit": True}

    def edit_then_silence(messages):
        current = [l.strip() for l in (repo / "calc.py").read_text().splitlines() if l.strip().startswith("return")][0]
        if pending["edit"]:
            pending["edit"] = False
            return r("str_replace", path="calc.py", old_str=current, new_str=next(wrong))
        return r(text="thinking...")

    def fix(messages):
        if messages[-1].get("content", "").startswith("PHASE: FIX"):
            pending["edit"] = True  # a new FIX phase just started
        return edit_then_silence(messages)

    state, run_dir, llm = run(cfg, repo, fix=fix)
    assert_abandoned_normally(state, run_dir, llm, "two fix attempts ended without any tool call")
    assert [a["reason"] for a in state.attempts] == ["no_tool_calls", "no_tool_calls"]
    assert state.attempts[0]["diff_hash"] != state.attempts[1]["diff_hash"]


def test_soft_budget_stops_new_attempts(repo, cfg) -> None:  # noqa: F811
    cfg.phases.enable_rescue = True
    add_test(repo)

    def expensive_wrong_fix(messages):  # each FIX reply costs 15.5k tokens
        text = (repo / "calc.py").read_text()
        resp = (r("str_replace", path="calc.py", old_str="return a - b", new_str="return a * b")
                if "a - b" in text else r("finish", summary="done"))
        return LLMResponse(resp.text, resp.tool_calls, Usage(15000, 500))

    state, run_dir, llm = run(cfg, repo, fix=expensive_wrong_fix)
    assert "soft token budget reached" in state.stop_reason
    assert len(state.attempts) == 2 and state.status in ("unverified", "no_fix")
    assert 45000 <= llm.metrics.total_tokens < cfg.budgets.max_total_tokens  # soft cap, not the hard cap
    assert f"**Stopped early:** {state.stop_reason}" in (run_dir / "report.md").read_text()


def test_hard_caps_end_as_budget_exhausted(repo, cfg) -> None:  # noqa: F811
    cfg.budgets.max_llm_calls = 4
    llm = FakeLLM([responder(repo, fix=lambda m: r("view_file", path="calc.py"))] * FOREVER, cfg=cfg)
    state, run_dir = Orchestrator(cfg, llm, python_exe=PY).solve(repo, "calc.py: add(1, 2) returns -1")
    assert state.status == "budget_exhausted" and llm.metrics.llm_calls == 4
    assert "(budget exhausted)" in (run_dir / "report.md").read_text().splitlines()[0]


def test_no_tool_calls_forever_is_abandoned(repo, cfg) -> None:  # noqa: F811
    """A model that never calls a tool in FIX: stops after two attempts (they are also empty diffs)."""
    cfg.phases.enable_rescue = True
    state, run_dir, llm = run(cfg, repo, fix=lambda m: r(text="I would change add() to use +."))
    assert_abandoned_normally(state, run_dir, llm, "no changes")
    assert [a["reason"] for a in state.attempts] == ["no_tool_calls", "no_tool_calls"]

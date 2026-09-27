"""Never crash, always report: every failure path ends with a written report and a clear status."""
import json
import sys
from pathlib import Path

import pytest

from harness.config import load_config
from harness.llm_fake import FakeLLM
from harness.orchestrator import Orchestrator
from harness.testing import TestRunner
from harness.types import ContextOverflow, FatalLLMError, LLMResponse, ToolCall, Usage
from harness.workspace import Workspace

PY = sys.executable
CALC = "def add(a, b):\n    return a - b\n"
_ids = iter(range(10**6))


def r(name: str | None = None, text: str = "", **args) -> LLMResponse:
    calls = [ToolCall(f"c{next(_ids)}", name, args)] if name else []
    return LLMResponse(text, calls, Usage(100, 20))


def phase_of(messages: list) -> str:
    first = messages[0]["content"]
    if first.startswith("Extract a structured summary"):
        return "intake"
    if first.startswith("You are a strict but fair code reviewer"):
        return "review"
    return first.rsplit("Current phase: ", 1)[-1].rstrip(".").lower()


def responder(repo: Path, **overrides):
    """A model that solves the calc bug, unless a phase is overridden to misbehave."""
    def reply(messages: list) -> LLMResponse:
        phase = phase_of(messages)
        if phase in overrides:
            return overrides[phase](messages)
        if phase == "intake":
            return r(text=json.dumps({"title": "add subtracts", "summary": "add(1, 2) returns -1."}))
        if phase == "localize":
            return r("finish", files=["calc.py"], root_cause="add subtracts")
        if phase == "reproduce":
            return r("finish", reproduced=False, command="", observed="not scripted")
        if phase == "fix":
            if "a - b" in (repo / "calc.py").read_text():
                return r("str_replace", path="calc.py", old_str="return a - b", new_str="return a + b")
            return r("finish", summary="add adds")
        return r(text='{"verdict": "approve", "problems": []}')
    return reply


@pytest.fixture()
def repo(tmp_path: Path) -> Path:
    root = tmp_path / "repo"
    (root / "tests").mkdir(parents=True)
    (root / "pytest.ini").write_text("[pytest]\ntestpaths = tests\n")
    (root / "calc.py").write_text(CALC)
    (root / "tests" / "test_other.py").write_text("def test_ok():\n    assert True\n")
    return root


@pytest.fixture()
def cfg(tmp_path: Path):
    c = load_config()
    c.output.runs_dir = str(tmp_path / "runs")
    c.phases.enable_rescue = False
    return c


def solve(cfg, repo: Path, reply, issue: str = "add(1, 2) returns -1") -> tuple:
    llm = FakeLLM([reply] * 200)
    return Orchestrator(cfg, llm, python_exe=PY).solve(repo, issue)


def assert_reported(state, run_dir: Path, status: str) -> str:
    assert state.status == status, state.notes
    for name in ("report.md", "state.json", "patch.diff", "metrics.json", "trajectory.jsonl"):
        assert (run_dir / name).exists(), name
    assert json.loads((run_dir / "state.json").read_text())["status"] == status
    return (run_dir / "report.md").read_text()


# ---------------------------------------------------------------- the ten listed failure paths
def test_api_error_or_timeout(repo, cfg) -> None:
    def api_down(messages):
        raise FatalLLMError("Model endpoint failed after 5 retries: Timeout")
    state, run_dir = solve(cfg, repo, responder(repo, localize=api_down))
    assert "⛔ ERROR" in assert_reported(state, run_dir, "error") and "Timeout" in state.notes[-1]
    assert (repo / "calc.py").read_text() == CALC


def test_malformed_tool_calls(repo, cfg) -> None:
    garbage = lambda m: LLMResponse("", [ToolCall("x", "__invalid__", {}, "not JSON")], Usage(10, 5))  # noqa: E731
    state, run_dir = solve(cfg, repo, responder(repo, localize=garbage), issue="calc.py: add(1, 2) returns -1")
    assert_reported(state, run_dir, "verified")  # LOCALIZE ended with protocol_failure; the run carried on
    assert state.localization["files"] == ["calc.py"]  # fell back to the pre-ranked candidates


def test_model_returns_no_tool_call(repo, cfg) -> None:
    state, run_dir = solve(cfg, repo, responder(repo, localize=lambda m: r(text="I think it's calc.py")))
    assert_reported(state, run_dir, "verified")
    assert any("no_tool_calls" in n for n in state.notes)


def test_context_overflow(repo, cfg) -> None:
    def too_big(messages):
        raise ContextOverflow("prompt too long")
    state, run_dir = solve(cfg, repo, responder(repo, intake=too_big))
    assert_reported(state, run_dir, "verified")
    assert any("prompt too large" in n for n in state.notes)


def test_budget_exhausted(repo, cfg) -> None:
    cfg.budgets.max_llm_calls = 3
    llm = FakeLLM([responder(repo)] * 50, cfg=cfg)
    state, run_dir = Orchestrator(cfg, llm, python_exe=PY).solve(repo, "add(1, 2) returns -1")
    assert "(budget exhausted)" in assert_reported(state, run_dir, "budget_exhausted")


def test_tests_cannot_be_detected(tmp_path, cfg) -> None:
    bare = tmp_path / "bare"
    bare.mkdir()
    (bare / "calc.py").write_text(CALC)
    state, run_dir = solve(cfg, bare, responder(bare))
    assert TestRunner(Workspace(bare, tmp_path / "s"), cfg).detect()["kind"] == "unknown"
    assert_reported(state, run_dir, "verified")


def test_test_run_times_out(repo, cfg) -> None:
    (repo / "tests" / "test_slow.py").write_text("import time\n\n\ndef test_hangs():\n    time.sleep(60)\n")
    cfg.tests.baseline_timeout_s = 2
    cfg.tests.targeted_timeout_s = 2
    state, run_dir = solve(cfg, repo, responder(repo))
    assert state.baseline_full.timed_out
    assert_reported(state, run_dir, "verified")


def test_repo_has_no_tests(repo, cfg) -> None:
    (repo / "tests" / "test_other.py").unlink()
    state, run_dir = solve(cfg, repo, responder(repo))
    assert_reported(state, run_dir, "verified")


def test_edit_that_breaks_syntax(repo, cfg) -> None:
    attempts = iter([r("str_replace", path="calc.py", old_str="def add(a, b):", new_str="def add(a, b)")])

    def fix(messages):
        return next(attempts, None) or responder(repo)(messages)

    llm = FakeLLM([responder(repo, fix=fix)] * 200)
    state, run_dir = Orchestrator(cfg, llm, python_exe=PY).solve(repo, "add(1, 2) returns -1")
    assert_reported(state, run_dir, "verified")
    assert any("Edit NOT applied" in (m.get("content") or "") for msgs in llm.calls for m in msgs)


def test_ctrl_c_mid_run(repo, cfg) -> None:
    def interrupt_after_edit(messages):
        if "a - b" in (repo / "calc.py").read_text():
            return r("str_replace", path="calc.py", old_str="return a - b", new_str="return a * b")
        raise KeyboardInterrupt

    orch = Orchestrator(cfg, FakeLLM([responder(repo, fix=interrupt_after_edit)] * 50), python_exe=PY)
    with pytest.raises(KeyboardInterrupt):
        orch.solve(repo, "add(1, 2) returns -1")
    assert "⏹ INTERRUPTED" in assert_reported(orch.state, orch.run_dir, "interrupted")
    assert (repo / "calc.py").read_text() == CALC  # the unverified half-edit was reverted


# ---------------------------------------------------------------- gaps found in the audit
def test_setup_failure_still_reports(repo, cfg, monkeypatch) -> None:
    monkeypatch.setattr(TestRunner, "detect", lambda self: (_ for _ in ()).throw(PermissionError("denied")))
    state, run_dir = solve(cfg, repo, responder(repo))
    assert_reported(state, run_dir, "error") and any("PermissionError" in n for n in state.notes)


def test_finalize_failure_still_reports(repo, cfg, monkeypatch) -> None:
    (repo / "tests" / "test_calc.py").write_text("from calc import add\n\n\ndef test_zero():\n    assert add(0, 0) == 0\n")
    breaks = r("str_replace", path="calc.py", old_str="return a - b", new_str="return a * b + 1")  # fails test_zero
    fix = iter([breaks, r("finish", summary="tweak")])
    cfg.phases.max_fix_attempts = 1
    monkeypatch.setattr(Workspace, "restore", lambda self, snap: (_ for _ in ()).throw(OSError("disk full")))
    state, run_dir = solve(cfg, repo, responder(repo, fix=lambda m: next(fix)))
    assert_reported(state, run_dir, "error") and any("finalizing failed" in n for n in state.notes)


def test_baseline_thread_crash_is_survived(repo, cfg, monkeypatch) -> None:
    real_run = TestRunner.run

    def run(self, targets=None, timeout_s=None):
        if targets is None:
            raise RuntimeError("runner exploded")
        return real_run(self, targets, timeout_s)

    monkeypatch.setattr(TestRunner, "run", run)
    state, run_dir = solve(cfg, repo, responder(repo))
    assert state.status != "error" and any("baseline: test run failed" in n for n in state.notes)
    assert (run_dir / "report.md").exists()


def test_cli_never_shows_a_traceback(repo, cfg, monkeypatch, capsys, tmp_path) -> None:
    from harness import cli
    from harness.orchestrator import Orchestrator as Orch

    monkeypatch.setenv("AI_API_KEY", "fake-key-for-cli")
    config = tmp_path / "c.yaml"
    config.write_text(f"output:\n  runs_dir: {tmp_path / 'runs'}\n")
    monkeypatch.setattr(cli, "make_llm", lambda c, ui: FakeLLM([]))
    monkeypatch.setattr(Orch, "solve", lambda self, repo, issue: (_ for _ in ()).throw(RuntimeError("boom")))
    code = cli.main(["--config", str(config), "--repo", str(repo), "--issue", "x", "--non-interactive"])
    out = capsys.readouterr().out
    assert code == 1 and "Unexpected error: RuntimeError: boom" in out and "Traceback" not in out

    monkeypatch.setattr(cli, "make_llm", lambda c, ui: (_ for _ in ()).throw(ValueError("bad endpoint")))
    assert cli.main(["--config", str(config), "--repo", str(repo), "--issue", "x", "--non-interactive"]) == 1
    assert "Unexpected error: ValueError: bad endpoint" in capsys.readouterr().out


# ---------------------------------------------------------------- presentation: no raw provider/JSON noise
def test_provider_message_drops_account_ids_and_upsell() -> None:
    from harness.llm import _provider_message
    raw = ('{"error": {"message": "Rate limit reached for model `m` in organization `org_123` service tier '
           '`on_demand` on tokens per day (TPD): Limit 200000, Used 199000. Please try again in 9m6s. '
           'Need more tokens? Upgrade to Dev Tier today at https://example.com"}}')
    msg = _provider_message(Exception(raw))
    assert "org_123" not in msg and "on_demand" not in msg and "Upgrade" not in msg
    assert msg.endswith("Please try again in 9m6s.")


def test_unfinished_localize_json_becomes_plain_root_cause() -> None:
    from harness.orchestrator import _salvage_localization
    text = '{"finish": {"files": ["a.py"], "root_cause": "regex matches one unit only"}}'
    assert _salvage_localization(text)["root_cause"] == "regex matches one unit only"
    assert _salvage_localization("just prose")["root_cause"] == "just prose"


def test_ctrl_d_at_a_prompt_is_no_answer(monkeypatch) -> None:
    from harness.ui import RichUI
    ui = RichUI()

    def eof(*_a, **_k):
        raise EOFError
    monkeypatch.setattr(ui.console, "input", eof)
    assert ui.ask("Solve another issue in the same repository? [y/N]") == ""

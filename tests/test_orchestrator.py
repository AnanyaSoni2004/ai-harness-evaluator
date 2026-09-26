"""Smoke tests for the Orchestrator with FakeLLM (the full sample-repo e2e test comes in Step 23)."""
import json
import sys
from pathlib import Path

import pytest

from harness.config import load_config
from harness.llm_fake import FakeLLM
from harness.orchestrator import Orchestrator
from harness.types import FatalLLMError, LLMResponse, ToolCall, Usage

PY = sys.executable
_ids = iter(range(10**6))
CALC = "def add(a, b):\n    return a - b\n\n\ndef mul(a, b):\n    return a * b\n"
REPRO = "from calc import add\nimport sys\nif add(1, 2) != 3:\n    print('BUG PRESENT')\n    sys.exit(1)\nprint('BUG FIXED')\n"
INTAKE = {"title": "add is wrong", "kind": "bug", "summary": "add(1, 2) returns -1.", "expected": "3", "actual": "-1",
          "error_messages": [], "mentioned_paths": ["calc.py"], "mentioned_symbols": ["add"], "repro_hints": ""}


def call(name: str, **args) -> ToolCall:
    return ToolCall(f"c{next(_ids)}", name, args)


def r(*calls: ToolCall, text: str = "") -> LLMResponse:
    return LLMResponse(text, list(calls), Usage(100, 20))


def js(obj: dict) -> LLMResponse:
    return r(text="```json\n" + json.dumps(obj) + "\n```")


@pytest.fixture()
def repo(tmp_path: Path) -> Path:
    root = tmp_path / "repo"
    (root / "tests").mkdir(parents=True)
    (root / "pytest.ini").write_text("[pytest]\ntestpaths = tests\n")
    (root / "calc.py").write_text(CALC)
    (root / "tests" / "test_calc.py").write_text("from calc import mul\n\n\ndef test_mul():\n    assert mul(2, 3) == 6\n")
    (root / "tests" / "test_legacy.py").write_text("def test_known_broken():\n    assert False\n")
    return root


@pytest.fixture()
def cfg(tmp_path: Path):
    c = load_config()
    c.output.runs_dir = str(tmp_path / "runs")
    c.phases.enable_rescue = False
    return c


def localize_and_reproduce(reproduced: bool = True) -> list:
    steps = [js(INTAKE), r(call("view_file", path="calc.py")),
             r(call("finish", files=["calc.py"], symbols=["add"], root_cause="add subtracts", confidence="high"))]
    if reproduced:
        cmd = f'"{PY}" @scratch/repro.py'
        steps += [r(call("create_file", path="@scratch/repro.py", content=REPRO)), r(call("run_command", command=cmd)),
                  r(call("finish", reproduced=True, command=cmd, observed="BUG PRESENT"))]
    else:
        steps += [r(call("finish", reproduced=False, command="", observed="could not reproduce"))]
    return steps


def test_happy_path_verified(repo: Path, cfg, tmp_path: Path) -> None:
    script = localize_and_reproduce() + [
        r(call("str_replace", path="calc.py", old_str="return a - b", new_str="return a + b")),
        r(call("finish", summary="add now adds"))]  # unambiguous verification: no REVIEW call
    llm = FakeLLM(script)
    state, run_dir = Orchestrator(cfg, llm, python_exe=PY).solve(repo, "add(1, 2) returns -1 instead of 3")
    assert state.status == "verified", state.notes
    assert (repo / "calc.py").read_text().startswith("def add(a, b):\n    return a + b")
    assert state.repro["reproduced"] and state.repro["before"]["exit_code"] == 1
    attempt = state.attempts[0]
    assert attempt["passed"] and attempt["verification"]["repro_after_exit"] == 0
    assert attempt["verification"]["new_failures"] == []  # the pre-existing failure is not "new"
    assert "tests/test_legacy.py::test_known_broken" in state.baseline_full.failing_ids
    assert state.targeted_tests == ["tests/test_calc.py"] and state.review["verdict"] == "skipped"
    assert "skipped: verification was unambiguous" in (run_dir / "report.md").read_text()
    assert run_dir.parent == tmp_path / "runs" and run_dir.name.endswith("add-1-2-returns-1")
    assert "+    return a + b" in (run_dir / "patch.diff").read_text()
    saved = json.loads((run_dir / "state.json").read_text())
    assert saved["status"] == "verified" and "snapshot" not in saved["attempts"][0]
    events = [json.loads(l)["event"] for l in (run_dir / "trajectory.jsonl").read_text().splitlines()]
    assert events[0] == "run_start" and events[-1] == "run_end"
    assert {"intake", "localize", "reproduce", "fix", "verify"} <= set(llm.metrics.per_phase)
    assert "review" not in llm.metrics.per_phase
    assert llm.metrics.per_phase["verify"].llm_calls == 0  # VERIFY never calls the model
    assert llm.script == []  # every scripted reply was used


def test_failed_attempts_revert_and_feed_back_evidence(repo: Path, cfg) -> None:
    cfg.phases.max_fix_attempts = 2
    break_mul = call("str_replace", path="calc.py", old_str="return a * b", new_str="return a + b")
    script = localize_and_reproduce(reproduced=False) + [
        r(break_mul), r(call("finish", summary="changed mul")),
        r(call("finish", summary="gave up"))]
    llm = FakeLLM(script)
    state, _ = Orchestrator(cfg, llm, python_exe=PY).solve(repo, "add is wrong")
    assert [a["passed"] for a in state.attempts] == [False, False]
    assert "tests/test_calc.py::test_mul" in state.attempts[0]["verification"]["new_failures"]
    second_fix_task = llm.calls[-1][1]["content"]
    assert "PREVIOUS ATTEMPT FAILED VERIFICATION" in second_fix_task and "tests/test_calc.py::test_mul" in second_fix_task
    assert state.status == "no_fix"
    assert (repo / "calc.py").read_text() == CALC  # never leave the repo worse than we found it


def test_fatal_llm_error_still_reports(repo: Path, cfg) -> None:
    def auth_fail(messages):
        raise FatalLLMError("Authentication/model error")

    state, run_dir = Orchestrator(cfg, FakeLLM([auth_fail]), python_exe=PY).solve(repo, "add is wrong")
    assert state.status == "error" and any("Authentication" in n for n in state.notes)
    assert (run_dir / "state.json").exists() and (run_dir / "patch.diff").read_text() == ""


def test_budget_exhausted_still_reports(repo: Path, cfg) -> None:
    cfg.budgets.max_llm_calls = 2
    llm = FakeLLM(localize_and_reproduce(), cfg=cfg)
    state, run_dir = Orchestrator(cfg, llm, python_exe=PY).solve(repo, "add is wrong")
    assert state.status == "budget_exhausted" and (run_dir / "state.json").exists()
    assert (repo / "calc.py").read_text() == CALC


def test_missing_repo_raises(cfg, tmp_path: Path) -> None:
    from harness.types import HarnessError
    with pytest.raises(HarnessError, match="Repository not found"):
        Orchestrator(cfg, FakeLLM([])).solve(tmp_path / "nope", "x")


def test_key_echoed_in_errors_never_reaches_artefacts_or_screen(repo: Path, cfg, monkeypatch, capsys) -> None:
    """A provider error that echoes the key must not leak into report.md, state.json or the terminal."""
    from harness.ui import RichUI

    key = "test-leaky-FAKE-key-9f8e7d6c5b4a"
    monkeypatch.setenv("AI_API_KEY", key)

    def auth_fail(messages):
        raise FatalLLMError(f"Incorrect API key provided: {key}")

    ui = RichUI()
    state, run_dir = Orchestrator(cfg, FakeLLM([auth_fail]), ui=ui, python_exe=PY).solve(repo, "add is wrong")
    ui.error(state.notes[-1])
    assert state.status == "error"
    for name in ("report.md", "state.json", "trajectory.jsonl", "metrics.json", "patch.diff"):
        text = (run_dir / name).read_text()
        assert key not in text, name
    assert "***" in (run_dir / "state.json").read_text()
    out = capsys.readouterr().out
    assert key not in out and "Incorrect API key provided: ***" in out


# ---------------------------------------------------------------- (b) REVIEW only when verification is ambiguous
@pytest.mark.parametrize("change, reason", [
    (lambda o, v: o.state.repro.update(reproduced=False), "the bug was not reproduced"),
    (lambda o, v: v.pop("full_after"), "the full test suite did not run"),
    (lambda o, v: v.update(new_failures=["t::x"]), "new test failures appeared"),
    (lambda o, v: None, "the patch touches more than 3 files"),
])
def test_review_needed_triggers(change, reason) -> None:
    from harness.types import IssueSpec, RunState
    orch = Orchestrator(load_config(), FakeLLM([]))
    orch.state = RunState(run_id="r", repo="/x", issue=IssueSpec(raw_text="x"))
    orch.state.repro = {"reproduced": True}
    v = {"full_after": object(), "new_failures": []}
    files = ["a.py"] * (4 if reason.startswith("the patch") else 1)
    assert orch._review_needed(dict(v), ["a.py"]) is None  # unambiguous -> no REVIEW call
    change(orch, v)
    assert orch._review_needed(v, files) == reason

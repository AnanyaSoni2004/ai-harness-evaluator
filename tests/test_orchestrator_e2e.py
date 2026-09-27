"""End-to-end: the full pipeline on a copy of the sample repo, scripted with FakeLLM (bug 4: Inventory.remove)."""
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from harness.config import load_config
from harness.llm_fake import FakeLLM
from harness.orchestrator import Orchestrator
from harness.types import LLMResponse, ToolCall, Usage

PY = sys.executable
FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"
ISSUE = (FIXTURES / "issues" / "04_inventory.md").read_text()
PRE_EXISTING = "tests/test_paging.py::test_first_page"
REPRO = ("from toolkit.inventory import Inventory\nimport sys\ninv = Inventory()\ninv.add('widget', 2)\n"
         "try:\n    inv.remove('widget', 5)\nexcept ValueError:\n    print('BUG FIXED')\n    sys.exit(0)\n"
         "print('BUG PRESENT: count =', inv.count('widget'))\nsys.exit(1)\n")
BUGGY = "        current = self._stock.get(item, 0)\n        self._stock[item] = current - qty"
FIXED = ("        current = self._stock.get(item, 0)\n        if qty > current:\n"
         "            raise ValueError(\"insufficient stock\")\n        self._stock[item] = current - qty")
REGRESSION_TEST = ("import pytest\n\nfrom toolkit.inventory import Inventory\n\n\n"
                   "def test_remove_more_than_stock_raises():\n    inv = Inventory()\n    inv.add('a', 1)\n"
                   "    with pytest.raises(ValueError, match='insufficient stock'):\n        inv.remove('a', 2)\n")
_ids = iter(range(10**6))


def call(name: str, **args) -> ToolCall:
    return ToolCall(f"call_{next(_ids)}", name, args)


def r(*calls: ToolCall, text: str = "") -> LLMResponse:
    return LLMResponse(text, list(calls), Usage(400, 60))


def js(obj: dict) -> LLMResponse:
    return r(text=json.dumps(obj))


def up_to_fix() -> list:
    repro_cmd = f'"{PY}" @scratch/repro.py'
    return [
        js({"title": "Inventory.remove lets stock go negative", "kind": "bug",
            "summary": "remove() allows removing more units than are in stock.",
            "expected": "ValueError('insufficient stock'), stock unchanged", "actual": "count becomes -3",
            "error_messages": [], "mentioned_paths": [], "mentioned_symbols": ["Inventory.remove"],
            "repro_hints": "add 2 then remove 5"}),
        r(call("view_file", path="toolkit/inventory.py")),
        r(call("finish", files=["toolkit/inventory.py"], symbols=["Inventory.remove"],
               root_cause="remove() subtracts without checking the current stock.", confidence="high")),
        r(call("create_file", path="@scratch/repro.py", content=REPRO)),
        r(call("run_command", command=repro_cmd)),
        r(call("finish", reproduced=True, command=repro_cmd, observed="BUG PRESENT: count = -3")),
    ]


@pytest.fixture()
def repo(tmp_path: Path) -> Path:
    dest = tmp_path / "sample_repo"
    shutil.copytree(FIXTURES / "sample_repo", dest)
    return dest


@pytest.fixture()
def cfg(tmp_path: Path):
    c = load_config()
    c.output.runs_dir = str(tmp_path / "runs")
    return c


def snapshot_tree(root: Path) -> dict:
    return {p.relative_to(root).as_posix(): p.read_bytes() for p in sorted(root.rglob("*")) if p.is_file()}


def fix_tasks(llm: FakeLLM) -> list[str]:
    """The task message of every FIX phase the model saw (first call of each phase)."""
    return [m[1]["content"] for m in llm.calls if len(m) == 2 and m[1]["content"].startswith("PHASE: FIX")]


def run_hidden_test(repo: Path) -> int:
    shutil.copy(FIXTURES / "hidden_tests" / "test_issue_04.py", repo / "tests" / "test_issue_04.py")
    return subprocess.run([PY, "-m", "pytest", "-q", "-p", "no:cacheprovider", "tests/test_issue_04.py"],
                          cwd=repo, capture_output=True, timeout=120).returncode


def test_e2e_verified_fix(repo: Path, cfg) -> None:
    script = up_to_fix() + [
        r(call("view_file", path="toolkit/inventory.py", start_line=15, end_line=25)),
        r(call("str_replace", path="toolkit/inventory.py", old_str=BUGGY, new_str=FIXED)),
        r(call("create_file", path="tests/test_inventory_regression.py", content=REGRESSION_TEST)),
        r(call("finish", summary="remove() raises ValueError('insufficient stock') instead of going negative.",
               files_changed=["toolkit/inventory.py"], tests_added=["tests/test_inventory_regression.py"])),
    ]  # reproduced + fixed, full suite ran, no new failures, 2 files: REVIEW is skipped
    llm = FakeLLM(script)
    state, run_dir = Orchestrator(cfg, llm, python_exe=PY).solve(repo, ISSUE)

    assert state.status == "verified", state.notes
    assert llm.script == []
    assert state.candidates[0]["path"] == "toolkit/inventory.py"
    assert state.targeted_tests == ["tests/test_inventory.py"]
    assert state.repro["before"]["exit_code"] == 1
    v = state.attempts[0]["verification"]
    assert v["repro_after_exit"] == 0 and v["new_failures"] == []
    assert PRE_EXISTING not in v["new_failures"]
    assert PRE_EXISTING in state.baseline_full.failing_ids
    assert "tests/test_inventory_regression.py" in v["targeted_after"].command  # added test was run

    patch = (run_dir / "patch.diff").read_text()
    assert patch and 'raise ValueError("insufficient stock")' in patch
    assert "+++ b/tests/test_inventory_regression.py" in patch
    report = (run_dir / "report.md").read_text()
    assert "VERIFIED" in report and PRE_EXISTING not in report.split("New failures", 1)[1].split("\n", 1)[0]
    assert state.review["verdict"] == "skipped" and "| Review verdict | — | skipped: verification was unambiguous" in report
    assert {"metrics.json", "state.json", "trajectory.jsonl"} <= {p.name for p in run_dir.iterdir()}
    assert run_hidden_test(repo) == 0  # the grader's hidden test passes on the result


def test_e2e_failure_path_keeps_repo_clean(repo: Path, cfg) -> None:
    cfg.phases.max_fix_attempts = 2
    original = snapshot_tree(repo)
    breaks_count = call("str_replace", path="toolkit/inventory.py", old_str="return self._stock.get(item, 0)",
                        new_str="return self._stock.get(item, 1)")
    breaks_more = call("str_replace", path="toolkit/inventory.py", old_str="return self._stock.get(item, 1)",
                       new_str="return self._stock.get(item, 2)")
    script = up_to_fix() + [
        r(breaks_count), r(call("finish", summary="changed count")),        # attempt 1: breaks another test
        r(breaks_more), r(call("finish", summary="tried again")),           # attempt 2: a different wrong diff
        r(call("finish", summary="still not sure")),                          # rescue: clean slate, no change
    ]
    llm = FakeLLM(script)
    state, run_dir = Orchestrator(cfg, llm, python_exe=PY).solve(repo, ISSUE)

    first = state.attempts[0]
    assert not first["passed"]
    assert "tests/test_inventory.py::test_add_and_count" in first["verification"]["new_failures"]
    assert PRE_EXISTING not in first["verification"]["new_failures"]
    tasks = fix_tasks(llm)
    assert len(tasks) == 3
    assert "PREVIOUS ATTEMPT FAILED VERIFICATION" in tasks[1] and "test_add_and_count" in tasks[1]
    assert "Start from a clean slate" in tasks[2] and [a["kind"] for a in state.attempts] == ["fix", "fix", "rescue"]
    assert state.status == "no_fix"
    assert snapshot_tree(repo) == original  # the repo is not worse than the baseline: it is untouched
    assert (run_dir / "patch.diff").read_text() == ""
    assert "NO FIX" in (run_dir / "report.md").read_text()


def test_e2e_reviewer_revise_then_approve(repo: Path, cfg) -> None:
    cfg.tests.run_full_suite_after_fix = False  # ambiguous verification (no full-suite run), so REVIEW runs
    partial = FIXED.replace('raise ValueError("insufficient stock")', "raise ValueError('nope')")
    script = up_to_fix() + [
        r(call("str_replace", path="toolkit/inventory.py", old_str=BUGGY, new_str=partial)),
        r(call("finish", summary="guard added")),
        js({"verdict": "revise", "problems": ["The error message must be 'insufficient stock' as the issue asks."],
            "confidence": "high"}),
        r(call("str_replace", path="toolkit/inventory.py", old_str="raise ValueError('nope')",
               new_str='raise ValueError("insufficient stock")')),
        r(call("finish", summary="message fixed")),
        js({"verdict": "approve", "problems": [], "confidence": "medium"}),
    ]
    llm = FakeLLM(script)
    state, _ = Orchestrator(cfg, llm, python_exe=PY).solve(repo, ISSUE)
    assert state.status == "verified" and len(state.attempts) == 2
    assert state.attempts[0]["passed"] and state.attempts[0]["review"]["verdict"] == "revise"
    assert "Reviewer requested changes" in fix_tasks(llm)[1] and "insufficient stock" in fix_tasks(llm)[1]
    assert run_hidden_test(repo) == 0


def test_budget_hit_mid_attempt_still_verifies_current_changes(repo: Path, cfg) -> None:
    """The budget runs out during attempt 2, before it is recorded: its changes must still be verified."""
    cfg.phases.max_fix_attempts = 2
    breaks_count = call("str_replace", path="toolkit/inventory.py", old_str="return self._stock.get(item, 0)",
                        new_str="return self._stock.get(item, 1)")
    undo = call("str_replace", path="toolkit/inventory.py", old_str="return self._stock.get(item, 1)",
                new_str="return self._stock.get(item, 0)")
    script = up_to_fix() + [r(breaks_count), r(call("finish", summary="changed count")),   # attempt 1: fails
                            r(undo), r(call("str_replace", path="toolkit/inventory.py", old_str=BUGGY, new_str=FIXED))]
    cfg.budgets.max_llm_calls = len(script)  # the next call (attempt 2's finish) exceeds the budget
    llm = FakeLLM(script, cfg=cfg)
    state, run_dir = Orchestrator(cfg, llm, python_exe=PY).solve(repo, ISSUE)
    assert state.status == "verified"  # the kept patch passed VERIFY; only REVIEW was cut short
    assert any("review did not run" in n for n in state.notes)
    last = state.attempts[-1]
    assert last["kind"] == "budget" and last["passed"] and last["verification"]["repro_after_exit"] == 0
    assert 'raise ValueError("insufficient stock")' in (run_dir / "patch.diff").read_text()
    assert run_hidden_test(repo) == 0


def test_fix_phase_knows_pre_existing_failures(repo: Path, cfg) -> None:
    cfg.phases.max_fix_attempts, cfg.phases.enable_rescue = 1, False
    script = up_to_fix() + [r(call("run_tests", targets="")), r(call("finish", summary="looked around"))]
    llm = FakeLLM(script)
    state, _ = Orchestrator(cfg, llm, python_exe=PY).solve(repo, ISSUE)
    assert state.status == "no_fix" and llm.script == []  # ends cleanly, no scripted replies left over
    run_tests_output = next(m["content"] for msgs in llm.calls for m in msgs
                            if m.get("role") == "tool" and m.get("name") == "run_tests")
    assert f"{PRE_EXISTING} (already failing before your change" in run_tests_output


def test_traceback_issue_skips_localize(repo: Path, cfg) -> None:
    """Issue 03 carries a traceback naming toolkit/durations.py: LOCALIZE costs zero model calls."""
    issue = (FIXTURES / "issues" / "03_durations.md").read_text()
    script = [js({"title": "combined durations", "summary": "parse_duration('1h30m') raises."}),
              r(call("finish", reproduced=False, command="", observed="skipped")),  # REPRODUCE
              r(call("finish", summary="looked"))]                                    # FIX (no change)
    cfg.phases.max_fix_attempts, cfg.phases.enable_rescue = 1, False
    llm = FakeLLM(script)
    state, _ = Orchestrator(cfg, llm, python_exe=PY).solve(repo, issue)
    assert state.localization["files"] == ["toolkit/durations.py"] and state.localization["source"] == "traceback"
    assert llm.metrics.per_phase["localize"].llm_calls == 0 and llm.script == []
    assert state.targeted_tests == ["tests/test_durations.py"]
    assert any("localize: skipped; the traceback names toolkit/durations.py" in n for n in state.notes)

"""Tests for the run report."""
import json
from pathlib import Path

import pytest

from harness.report import final_attempt, render_report, status_line, write_report
from harness.types import IssueSpec, Metrics, RunState, TestRun, Usage
from harness.workspace import Workspace


def make_run(passed: int, failing: list) -> TestRun:
    return TestRun("pytest", 1 if failing else 0, passed, len(failing), 0, failing, 1.0, False, False, "")


@pytest.fixture()
def ws(tmp_path: Path) -> Workspace:
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "inventory.py").write_text("def remove(n):\n    return n\n")
    w = Workspace(repo, tmp_path / "scratch")
    w.write_text("inventory.py", "def remove(n):\n    if n < 0:\n        raise ValueError('insufficient stock')\n"
                                 "    return n\n")
    return w


def fake_state(ws: Workspace, status: str = "verified", new_failures: list | None = None) -> RunState:
    state = RunState(run_id="20260927-120000-inventory", repo=str(ws.repo_root),
                     issue=IssueSpec(raw_text="Inventory goes negative\nbody", title="Inventory goes negative",
                                     summary="remove() lets stock go negative."))
    state.localization = {"files": ["inventory.py"], "root_cause": "remove() never checks stock."}
    state.repro = {"reproduced": True, "command": "python3 @scratch/repro.py", "observed": "BUG PRESENT",
                   "before": {"exit_code": 1, "output_tail": "setup\n" * 10 + "BUG PRESENT\n"}}
    state.targeted_tests = ["tests/test_inventory.py"]
    base = make_run(20, ["tests/test_paging.py::test_first_page"])
    state.baseline_full = base
    verification = {"repro_before_exit": 1, "repro_after_exit": 0, "repro_passed": True,
                    "repro_after_tail": "BUG FIXED\n", "targeted_before": make_run(3, []),
                    "targeted_after": make_run(4, []), "full_before": base,
                    "full_after": make_run(21, ["tests/test_paging.py::test_first_page"]),
                    "new_failures": new_failures or [], "fixed": ["tests/test_inventory.py::test_negative"],
                    "notes": []}
    state.attempts = [{"attempt": 1, "kind": "fix", "passed": not new_failures, "score": 3, "files": ["inventory.py"],
                       "diff": ws.diff(), "summary": "remove() now raises on insufficient stock.",
                       "verification": verification, "snapshot": ws.snapshot(),
                       "review": {"verdict": "approve", "problems": [], "confidence": "high"}}]
    state.status = status
    state.notes = ["reproduce: example note"]
    return state


def fake_metrics() -> Metrics:
    m = Metrics(started_at=100.0, ended_at=160.5)
    m.add_llm("intake", Usage(900, 120))
    m.add_llm("localize", Usage(3000, 300))
    m.add_tool("localize")
    m.add_time("localize", 12.25)
    m.add_time("verify", 4.0)
    return m


def test_write_report_files_and_tables(ws: Workspace, tmp_path: Path) -> None:
    paths = write_report(fake_state(ws), ws, fake_metrics(), tmp_path / "run")
    assert set(paths) == {"patch", "metrics", "state", "report"} and all(p.exists() for p in paths.values())
    assert paths["patch"].read_text() == ws.diff()
    assert json.loads(paths["metrics"].read_text())["total"]["total_tokens"] == 4320
    saved = json.loads(paths["state"].read_text())
    assert "snapshot" not in saved["attempts"][0] and saved["status"] == "verified"
    md = paths["report"].read_text()
    assert md.startswith("# ✅ VERIFIED FIX")
    assert "## Evidence" in md and "| Check | Before | After |" in md
    assert "## Efficiency" in md and "| Phase | LLM calls | Prompt tokens |" in md
    assert "| Reproduction (`python3 @scratch/repro.py`) | exit 1 | exit 0 |" in md
    assert "| Targeted tests (1 file(s)) | 3 passed / 0 failed | 4 passed / 0 failed |" in md
    assert "| Full suite | 20 passed / 1 failed | 21 passed / 1 failed |" in md
    assert "| New failures (must be empty) | — | none ✅ |" in md
    assert "tests/test_inventory.py::test_negative" in md and "| Review verdict | — | approve (high) |" in md
    assert "| localize | 1 | 3000 | 300 | 1 | 12.2 |" in md
    assert "| **Total** | **2** | **3900** | **420** | **1** | **60.5** |" in md
    assert "```diff\n--- a/inventory.py" in md and "- **Files changed:** `inventory.py`" in md
    assert "**Root cause:** remove() never checks stock." in md and "- reproduce: example note" in md


def test_repro_output_tails_limited_to_five_lines(ws: Workspace) -> None:
    md = render_report(fake_state(ws), ws.diff(), fake_metrics())
    before_block = md.split("before the fix (last 5 lines):\n```\n", 1)[1].split("```", 1)[0]
    assert before_block.strip().splitlines() == ["setup"] * 4 + ["BUG PRESENT"]
    assert "BUG FIXED" in md


def test_new_failures_are_listed(ws: Workspace) -> None:
    md = render_report(fake_state(ws, "unverified", ["tests/test_a.py::test_x"]), ws.diff(), fake_metrics())
    assert md.startswith("# ⚠️ UNVERIFIED CHANGE")
    assert "| New failures (must be empty) | — | ❌ tests/test_a.py::test_x |" in md


@pytest.mark.parametrize("status, has_diff, expected", [
    ("verified", True, "✅ VERIFIED FIX"), ("unverified", True, "⚠️ UNVERIFIED CHANGE"),
    ("no_fix", False, "❌ NO FIX"), ("error", False, "⛔ ERROR"),
    ("budget_exhausted", True, "⚠️ UNVERIFIED CHANGE (budget exhausted)"),
    ("budget_exhausted", False, "❌ NO FIX (budget exhausted)"),
])
def test_status_lines(ws: Workspace, status: str, has_diff: bool, expected: str) -> None:
    assert status_line(fake_state(ws, status), has_diff) == expected


def test_reverted_attempt_is_labelled(ws: Workspace) -> None:
    state = fake_state(ws, "no_fix", ["tests/test_a.py::test_x"])
    ws.revert_all()
    attempt, kept = final_attempt(state, ws.diff())
    assert attempt is state.attempts[0] and not kept
    md = render_report(state, ws.diff(), fake_metrics())
    assert "which was **not kept**" in md and "- **Files changed:** none" in md and "(no changes)" in md


def test_no_attempt_and_table_escaping(ws: Workspace) -> None:
    state = fake_state(ws, "error")
    state.attempts = []
    state.repro = {"reproduced": False, "observed": "needs | network"}
    md = render_report(state, "", Metrics())
    assert "No fix attempt reached verification." in md and "Baseline full suite: 20 passed / 1 failed" in md
    assert md.startswith("# ⛔ ERROR")


def test_orchestrator_falls_back_when_writer_crashes(tmp_path: Path, monkeypatch) -> None:
    from harness import report
    from harness.config import load_config
    from harness.llm_fake import FakeLLM
    from harness.orchestrator import Orchestrator
    from harness.types import FatalLLMError

    def crash(*args, **kwargs):
        raise RuntimeError("disk full")

    def fail(messages):
        raise FatalLLMError("auth")

    monkeypatch.setattr(report, "write_report", crash)
    (tmp_path / "repo").mkdir()
    cfg = load_config()
    cfg.output.runs_dir = str(tmp_path / "runs")
    state, run_dir = Orchestrator(cfg, FakeLLM([fail])).solve(tmp_path / "repo", "x")
    assert (run_dir / "state.json").exists() and (run_dir / "patch.diff").exists()
    assert any("writer failed (RuntimeError: disk full)" in n for n in state.notes)

"""End-to-end: the Tracer inside the full pipeline (bug 4 on a copy of the sample repo, FakeLLM)."""
from pathlib import Path

import test_orchestrator_e2e as e2e
from harness.llm_fake import FakeLLM
from harness.orchestrator import Orchestrator
from harness.spectrum import SpectrumAnalyzer

repo = e2e.repo  # reuse the fixtures: a fresh copy of the sample repo and a config writing runs to tmp_path
cfg = e2e.cfg


def fixing_script(reproduced: bool = True) -> list:
    steps = e2e.up_to_fix()
    if not reproduced:
        steps = steps[:3] + [e2e.r(e2e.call("finish", reproduced=False, command="", observed="could not script it"))]
    return steps + [
        e2e.r(e2e.call("str_replace", path="toolkit/inventory.py", old_str=e2e.BUGGY, new_str=e2e.FIXED)),
        e2e.r(e2e.call("finish", summary="guard added")),
        e2e.js({"verdict": "approve", "problems": [], "confidence": "high"})]


def test_tracer_evidence_reaches_fix_and_report(repo: Path, cfg) -> None:
    llm = FakeLLM(fixing_script())
    state, run_dir = Orchestrator(cfg, llm, python_exe=e2e.PY).solve(repo, e2e.ISSUE)
    assert state.status == "verified", state.notes
    assert state.spectrum["ok"], state.spectrum.get("reason")
    assert state.spectrum["functions"][0]["name"] == "Inventory.remove"
    fix_tasks = e2e.fix_tasks(llm)
    assert fix_tasks and "EXECUTION EVIDENCE" in fix_tasks[0] and "Inventory.remove" in fix_tasks[0]
    report = (run_dir / "report.md").read_text()
    assert "### Fault localization" in report and "Patch touches suspicious rank(s): #1" in report
    assert "| trace | 0 | 0 | 0 | 0 |" in report  # zero tokens spent on TRACE
    assert llm.metrics.per_phase["trace"].llm_calls == 0


def test_no_reproduction_skips_tracer_only(repo: Path, cfg) -> None:
    llm = FakeLLM(fixing_script(reproduced=False))
    state, run_dir = Orchestrator(cfg, llm, python_exe=e2e.PY).solve(repo, e2e.ISSUE)
    assert state.spectrum == {"ok": False, "reason": "no reproduction", "formula": "ochiai", "lines": [],
                              "functions": [], "failing_runs": 0, "passing_runs": 0,
                              "seconds": state.spectrum["seconds"], "low_confidence": False}
    assert state.status == "verified" and "EXECUTION EVIDENCE" not in e2e.fix_tasks(llm)[0]
    assert "Tracer skipped: no reproduction" in (run_dir / "report.md").read_text()


def test_tracer_crash_never_breaks_the_run(repo: Path, cfg, monkeypatch) -> None:
    def boom(self, state):
        raise RuntimeError("tracer exploded")

    monkeypatch.setattr(SpectrumAnalyzer, "analyze", boom)
    state, _ = Orchestrator(cfg, FakeLLM(fixing_script()), python_exe=e2e.PY).solve(repo, e2e.ISSUE)
    assert state.status == "verified"
    assert state.spectrum["reason"].startswith("crashed:") and "tracer exploded" in state.spectrum["reason"]


def test_low_time_budget_skips_tracer(repo: Path, cfg) -> None:
    cfg.budgets.max_wall_clock_s = 200  # < spectrum.timeout_s (120) + 300
    state, _ = Orchestrator(cfg, FakeLLM(fixing_script()), python_exe=e2e.PY).solve(repo, e2e.ISSUE)
    assert state.spectrum == {"ok": False, "reason": "low time budget"} and state.status == "verified"


def test_disabled_by_env(repo: Path, cfg) -> None:
    cfg.spectrum.enabled = False  # what HARNESS_SPECTRUM=0 does at load time
    state, _ = Orchestrator(cfg, FakeLLM(fixing_script()), python_exe=e2e.PY).solve(repo, e2e.ISSUE)
    assert state.spectrum["reason"] == "disabled" and state.status == "verified"

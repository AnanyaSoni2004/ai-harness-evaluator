"""Offline tests for scripts/eval.py (the live benchmark itself needs AI_API_KEY)."""
import importlib.util
import json
import shutil
import sys
from pathlib import Path

import pytest

from test_fixtures import REFERENCE_FIXES

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("harness_eval", ROOT / "scripts" / "eval.py")
ev = importlib.util.module_from_spec(spec)
sys.modules["harness_eval"] = ev  # dataclasses need the module registered
spec.loader.exec_module(ev)


def test_issue_selection_and_hidden_test_mapping() -> None:
    names = [p.name for p in ev.list_issues()]
    assert names == ["01_slugify.md", "02_paginate.md", "03_durations.md", "04_inventory.md", "05_median.md"]
    assert [p.name for p in ev.list_issues(quick=True)] == names[:2]
    assert [p.name for p in ev.list_issues(["4", "01"])] == ["01_slugify.md", "04_inventory.md"]
    assert ev.hidden_test_for(ev.list_issues(["04"])[0]).name == "test_issue_04.py"


def test_hidden_test_grading(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    shutil.copytree(ROOT / "fixtures" / "sample_repo", repo)
    hidden = ROOT / "fixtures" / "hidden_tests" / "test_issue_04.py"
    assert ev.run_hidden_test(repo, hidden) is False
    rel, old, new = REFERENCE_FIXES[4]
    (repo / rel).write_text((repo / rel).read_text().replace(old, new))
    assert ev.run_hidden_test(repo, hidden) is True


def fake_run_dir(runs: Path, name: str, spectrum: dict, patch: str) -> Path:
    d = runs / name
    d.mkdir(parents=True)
    (d / "state.json").write_text(json.dumps({"status": "verified", "spectrum": spectrum}))
    (d / "metrics.json").write_text(json.dumps({"total": {"total_tokens": 1234, "llm_calls": 9, "tool_calls": 7,
                                                          "seconds": 42.0}}))
    (d / "patch.diff").write_text(patch)
    return d


PATCH = ("--- a/toolkit/inventory.py\n+++ b/toolkit/inventory.py\n@@ -20,1 +20,3 @@\n"
         "         current = self._stock.get(item, 0)\n+        if qty > current:\n+            raise ValueError\n")
SPECTRUM = {"ok": True, "functions": [  # the patch touches rank 1 of 2 -> EXAM 0.5
    {"path": "toolkit/inventory.py", "name": "Inventory.remove", "start": 16, "end": 21},
    {"path": "toolkit/inventory.py", "name": "Inventory.add", "start": 10, "end": 14}]}


def test_read_run_computes_exam(tmp_path: Path) -> None:
    info = ev.read_run(fake_run_dir(tmp_path, "r1", SPECTRUM, PATCH))
    assert info == {"status": "verified", "tokens": 1234, "llm_calls": 9, "tool_calls": 7, "seconds": 42.0,
                    "exam": 0.5}
    assert ev.read_run(fake_run_dir(tmp_path, "r2", {"ok": False, "reason": "x"}, PATCH))["exam"] is None
    assert ev.read_run(None) == {} and ev.read_run(tmp_path / "missing") == {}


def test_run_issue_plumbing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A stand-in harness: it must get the key and the Tracer switch, and its run dir must be found."""
    runs = tmp_path / "runs"
    runs.mkdir()
    seen = {}
    real = ev.run_process

    def fake(cmd, cwd, timeout, env_extra=None):
        if " -m harness " not in cmd:
            return real(cmd, cwd, timeout, env_extra)  # the hidden test really runs
        seen.update(env_extra, cmd=cmd)
        repo = Path(cmd.split("--repo ")[1].split(" ")[0])
        rel, old, new = REFERENCE_FIXES[4]
        (repo / rel).write_text((repo / rel).read_text().replace(old, new))
        fake_run_dir(runs, "20260101-000000-inventory", SPECTRUM, PATCH)
        return type("R", (), {"exit_code": 0, "output": "", "timed_out": False})()

    monkeypatch.setattr(ev, "run_process", fake)
    monkeypatch.setenv("AI_API_KEY", "fake-key-for-eval-test")
    args = ev.argparse.Namespace(config=None, model="groq/test-model")
    result = ev.run_issue(ev.list_issues(["04"])[0], "no-tracer", runs, args, 60)
    assert seen["AI_API_KEY"] == "fake-key-for-eval-test" and seen["HARNESS_SPECTRUM"] == "0"
    assert "--model groq/test-model" in seen["cmd"] and "--strict-exit" in seen["cmd"]
    assert result.solved and result.status == "verified" and result.tokens == 1234 and result.exam == 0.5
    assert result.run_dir.endswith("20260101-000000-inventory")


def test_summary_markdown_with_comparison() -> None:
    results = [ev.IssueResult("04_inventory", "tracer", True, "verified", 20000, 15, 12, 100.0, 0.25),
               ev.IssueResult("05_median", "tracer", False, "no_fix", 30000, 20, 15, 200.0, None),
               ev.IssueResult("04_inventory", "no-tracer", True, "verified", 26000, 18, 16, 120.0, None),
               ev.IssueResult("05_median", "no-tracer", True, "verified", 28000, 19, 14, 130.0, None)]
    s = ev.summarise(results[:2])
    assert s["solved"] == 1 and s["solve_rate"] == 0.5 and s["avg_tokens"] == 25000 and s["mean_exam"] == 0.25
    md = ev.markdown(results, "groq/qwen/qwen3.8-27b")
    assert "| 04_inventory | tracer | ✅ | verified | 20000 | 15 | 12 | 100.0 | 0.25 |" in md
    assert "| tracer | 1/2 | 50% | 25000 |" in md and "| no-tracer | 2/2 | 100% | 27000 |" in md
    assert "with vs without the Tracer" in md


def test_main_requires_key(monkeypatch: pytest.MonkeyPatch, capsys) -> None:
    monkeypatch.delenv("AI_API_KEY", raising=False)
    assert ev.main([]) == 1 and "AI_API_KEY is not set" in capsys.readouterr().out

"""Tests for prompt templates: every template renders, no placeholders remain, tasks stay small."""
import re

import pytest

from harness import prompts
from harness.context import estimate_tokens
from harness.types import IssueSpec, RunState, TestRun

PLACEHOLDER = re.compile(r"\{[a-z_]+\}")


def tokens(text: str) -> int:
    return estimate_tokens([{"role": "user", "content": text}])


def sample_state(scale: int = 1) -> RunState:
    issue = IssueSpec(
        raw_text="Inventory.remove lets stock go negative\n" * scale,
        title="Inventory goes negative", summary="Inventory.remove allows negative stock. " * scale,
        expected="ValueError('insufficient stock')", actual="stock becomes -3",
        error_messages=["AssertionError: -3 != 0"] * scale, mentioned_paths=["toolkit/inventory.py"],
        mentioned_symbols=["Inventory.remove"], repro_hints="inv.add('x', 2); inv.remove('x', 5)")
    state = RunState(run_id="r1", repo="/tmp/repo", issue=issue)
    state.candidates = [{"path": f"toolkit/mod{i}.py", "score": 10 - i, "terms": ["Inventory", "remove"]}
                        for i in range(12)]
    state.localization = {"files": ["toolkit/inventory.py"], "symbols": ["Inventory.remove"],
                          "root_cause": "remove() never checks the current stock. " * scale, "confidence": "high"}
    state.repro = {"reproduced": True, "command": "python3 @scratch/repro.py", "observed": "BUG PRESENT",
                   "before": {"exit_code": 1, "output_tail": "BUG PRESENT\n" * scale}}
    state.baseline_full = TestRun("pytest", 1, 20, 1, 0, ["tests/test_paging.py::test_first_page"], 1.0,
                                  False, False, "")
    state.targeted_tests = ["tests/test_inventory.py"]
    return state


def verification() -> dict:
    after = TestRun("pytest", 1, 21, 1, 0, ["tests/test_paging.py::test_first_page"], 1.0, False, False, "")
    return {"repro_before_exit": 1, "repro_after_exit": 0, "repro_passed": True,
            "full_before": sample_state().baseline_full, "full_after": after,
            "targeted_before": {"passed": 3, "failed": 0, "errors": 0}, "targeted_after": None,
            "new_failures": [], "fixed": []}


def all_renders(state: RunState, big: str = "") -> dict:
    return {
        "system": prompts.system_prompt("fix", "/tmp/repo"),
        "intake": prompts.intake_prompt(state.issue.raw_text + big),
        "localize": prompts.localize_task(state, "toolkit/inventory.py:\n  class Inventory (L1)\n" + big, max_steps=15),
        "reproduce": prompts.reproduce_task(state),
        "fix": prompts.fix_task(state, 2, "tests/test_x.py::test_y failed\n" + big, max_attempts=3),
        "review": prompts.review_prompt(state, "--- a/x\n+++ b/x\n" + big, verification()),
    }


def test_every_template_renders_without_placeholders() -> None:
    for name, text in all_renders(sample_state()).items():
        assert text.strip(), name
        assert not PLACEHOLDER.search(text), (name, PLACEHOLDER.search(text).group(0))


def test_system_prompt() -> None:
    text = prompts.system_prompt("localize", "/work/repo")
    assert "repository at /work/repo." in text and text.endswith("Current phase: LOCALIZE.")
    assert "@scratch/" in text and "call finish immediately" in text
    shared = prompts.system_prompt("fix", "/work/repo").rsplit("\n", 1)[0]
    assert text.rsplit("\n", 1)[0] == shared  # identical prefix across phases (prompt caching)


def test_task_contents() -> None:
    r = all_renders(sample_state())
    assert r["intake"].startswith("Extract a structured summary") and '"mentioned_symbols": [str]' in r["intake"]
    assert "Candidate files" in r["localize"] and "1. toolkit/mod0.py (score 10; terms: Inventory, remove)" in r["localize"]
    assert "10. toolkit/mod9.py" in r["localize"] and "11. " not in r["localize"]  # top 10 only
    assert "You have 15 tool calls" in r["localize"]
    assert "Expected: ValueError('insufficient stock')" in r["reproduce"]  # the issue is included
    assert "Reproduction hints:" in r["reproduce"] and 'print "BUG PRESENT"' in r["reproduce"]
    assert r["fix"].startswith("PHASE: FIX (attempt 2 of 3).")
    assert "`python3 @scratch/repro.py` currently outputs:" in r["fix"]
    assert "tests/test_paging.py::test_first_page" in r["fix"]
    assert "PREVIOUS ATTEMPT FAILED VERIFICATION:\ntests/test_x.py::test_y failed" in r["fix"]
    assert "```diff\n--- a/x" in r["review"] and "Reproduction exit code: before 1 -> after 0" in r["review"]
    assert "Targeted tests: before 3 passed / 0 failed -> after not run" in r["review"]
    assert "Full suite: before 20 passed / 1 failed -> after 21 passed / 1 failed" in r["review"]


def test_fix_task_variants() -> None:
    state = sample_state()
    assert "PREVIOUS ATTEMPT" not in prompts.fix_task(state, 1, "")
    state.repro = {"reproduced": False, "observed": "needs network"}
    state.baseline_full = None
    text = prompts.fix_task(state, 1, "")
    assert "Reproduction: none" in text and "needs network" in text and "(baseline not available)" in text


def test_intake_failure_falls_back_to_raw_text() -> None:
    state = sample_state()
    state.issue.summary = ""
    assert "Inventory.remove lets stock go negative" in prompts.localize_task(state, "")


@pytest.mark.parametrize("scale", [1, 400])
def test_tasks_stay_under_budget(scale: int) -> None:
    big = "y" * 60000 if scale > 1 else ""
    for name, text in all_renders(sample_state(scale), big).items():
        assert tokens(text) <= 2600, (name, tokens(text))  # ~2,500 tokens incl. JSON overhead


def test_huge_fields_keep_closing_instructions() -> None:
    r = all_renders(sample_state(400), "z" * 60000)
    assert r["fix"].rstrip().endswith("code you change.")
    assert r["review"].rstrip().endswith('concrete, fixable problems.')
    assert r["localize"].rstrip().endswith("confidence.")

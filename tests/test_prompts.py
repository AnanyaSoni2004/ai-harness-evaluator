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


# ---------------------------------------------------------------- T5: Tracer evidence in prompts
EVIDENCE = ("EXECUTION EVIDENCE (spectrum-based fault localization, Ochiai; 1 failing run, 3 passing runs):\n"
            "#1 toolkit/inventory.py  Inventory.remove (L16-L21)  score 0.71\n   >>  20 |         current = 0")
SPECTRUM = {"ok": True, "evidence": EVIDENCE, "low_confidence": False,
            "functions": [{"path": "toolkit/inventory.py", "name": "Inventory.remove", "start": 16, "end": 21,
                           "score": 0.71, "top_line": 20, "ef": 1, "ep": 1}]}


def with_spectrum(spectrum: dict, files: list | None = None) -> RunState:
    state = sample_state()
    state.spectrum = dict(spectrum)
    if files is not None:
        state.localization["files"] = files
    return state


def test_fix_task_includes_evidence_after_root_cause() -> None:
    state = with_spectrum(SPECTRUM, files=["toolkit/inventory.py"])
    text = prompts.fix_task(state, 1, "")
    assert "EXECUTION EVIDENCE" in text
    assert text.index("Root cause analysis:") < text.index("EXECUTION EVIDENCE") < text.index("Reproduction:")
    assert "NOTE: execution evidence points to" not in text
    assert not PLACEHOLDER.search(text)


def test_fix_task_without_tracer_has_no_block() -> None:
    for spectrum in ({}, {"ok": False, "reason": "no reproduction"}, {"ok": True, "evidence": ""}):
        assert "EXECUTION EVIDENCE" not in prompts.fix_task(with_spectrum(spectrum), 1, "")


def test_mismatch_note_only_on_mismatch() -> None:
    text = prompts.fix_task(with_spectrum(SPECTRUM, files=["toolkit/stock_report.py"]), 1, "")
    assert ("NOTE: execution evidence points to toolkit/inventory.py::Inventory.remove, but localization chose "
            "toolkit/stock_report.py. Check both before editing.") in text


def test_low_confidence_wording_comes_from_the_block(tmp_path) -> None:
    from harness.spectrum import LOW_CONFIDENCE_CLOSING
    spectrum = dict(SPECTRUM, evidence=EVIDENCE + "\n" + LOW_CONFIDENCE_CLOSING, low_confidence=True)
    assert LOW_CONFIDENCE_CLOSING in prompts.fix_task(with_spectrum(spectrum, ["toolkit/inventory.py"]), 1, "")


def test_fix_task_with_evidence_stays_within_budgets() -> None:
    huge = dict(SPECTRUM, evidence=EVIDENCE + "\n" + "   >>  99 | x = 1   [score 0.50]\n" * 400)
    state = with_spectrum(huge, files=["elsewhere.py"])
    text = prompts.fix_task(state, 2, "tests/test_x.py::test_y failed\n" * 200)
    assert tokens(text) <= 2600
    assert tokens(text) < 0.40 * 32768
    assert text.rstrip().endswith("code you change.")


def test_review_prompt_names_top_suspect() -> None:
    diff = ("--- a/toolkit/inventory.py\n+++ b/toolkit/inventory.py\n@@ -19,3 +19,5 @@\n"
            "         current = self._stock.get(item, 0)\n+        if qty > current:\n+            raise ValueError\n"
            "         self._stock[item] = current - qty\n")
    touched = prompts.review_prompt(with_spectrum(SPECTRUM), diff, verification())
    assert "Top suspicious function: Inventory.remove (patch touches it: yes)." in touched
    other = prompts.review_prompt(with_spectrum(SPECTRUM), "--- a/x.py\n+++ b/x.py\n@@ -1 +1 @@\n-a\n+b\n", {})
    assert "(patch touches it: no)." in other
    assert "Top suspicious function" not in prompts.review_prompt(sample_state(), diff, {})


def test_phase_tasks_say_what_is_editable() -> None:
    state = sample_state()
    assert "editing tools become available in the FIX phase" in prompts.localize_task(state, "")
    assert "only @scratch/ is writable here" in prompts.reproduce_task(state)

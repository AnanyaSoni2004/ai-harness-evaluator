"""Every prompt template used by the harness (Appendix A)."""
from __future__ import annotations

from typing import Any

from harness.shell import truncate
from harness.types import RunState

MAX_TASK_TOKENS = 2500
MAX_TASK_CHARS = int(MAX_TASK_TOKENS * 3.5)

SHARED_RULES = """You are an autonomous senior software engineer resolving a GitHub issue in the repository at {repo_root}.
You act only through the provided tools.
Rules:
- Never assume file contents: view them before editing. Line numbers shown by view_file are for
  reference only; never put them inside old_str.
- Edit files only with str_replace/create_file (never shell redirects, sed or echo) so changes are tracked.
- Make the smallest change that fixes the root cause. Keep the existing style. No unrelated refactors.
- Never modify existing tests to make them pass. You may add new tests.
- Paths are relative to the repository root. Throwaway scripts go under @scratch/ (never inside the repo).
- Be economical: prefer search_code and small view_file ranges over reading whole files; do not repeat calls.
- When the phase goal is met, call finish immediately."""


def _t(text: Any, limit: int) -> str:
    """Stringify and cap a field (head and tail kept)."""
    return truncate(str(text or "").strip(), limit, head_chars=limit // 2)


def _get(obj: Any, key: str, default: Any = None) -> Any:
    """Read a key from a dict or an attribute from an object (TestRun or dict)."""
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _cap_task(text: str) -> str:
    """Final safety net: keep every task under ~2,500 tokens (head and tail kept)."""
    return truncate(text, MAX_TASK_CHARS, head_chars=MAX_TASK_CHARS // 2)


def _issue_line(state: RunState) -> str:
    """Summary, or the raw issue text when intake produced no summary."""
    return _t(state.issue.summary or state.issue.raw_text, 1200)


def _baseline_failures(state: RunState) -> str:
    """Test IDs already failing before any change."""
    ids: list[str] = []
    for run in (state.baseline_full, state.baseline_targeted):
        ids += [i for i in (_get(run, "failing_ids") or []) if i not in ids]
    if state.baseline_full is None and state.baseline_targeted is None:
        return "(baseline not available)"
    if not ids:
        return "none"
    shown = ", ".join(ids[:15])
    return shown + (f" (+{len(ids) - 15} more)" if len(ids) > 15 else "")


def system_prompt(phase: str, repo_root: str) -> str:
    """Shared rules (identical across phases, so provider prompt caching works) plus the phase name."""
    return SHARED_RULES.replace("{repo_root}", str(repo_root)) + f"\nCurrent phase: {phase.upper()}."


def intake_prompt(issue_text: str) -> str:
    """INTAKE: one call, no tools, JSON-only reply describing the issue."""
    return _cap_task(
        "Extract a structured summary of this GitHub issue. Respond with ONLY a JSON object:\n"
        '{"title": str, "kind": "bug"|"feature"|"other", "summary": str (max 3 sentences),\n'
        ' "expected": str, "actual": str, "error_messages": [str], "mentioned_paths": [str],\n'
        ' "mentioned_symbols": [str], "repro_hints": str}\n'
        "ISSUE:\n<<<\n" + _t(issue_text, 7000) + "\n>>>")


def localize_task(state: RunState, repo_map_text: str, max_steps: int = 15) -> str:
    """LOCALIZE: read-only search for the code responsible."""
    issue = state.issue
    errors = "; ".join(issue.error_messages) or "none given"
    cands = [f"{i}. {c.get('path')} (score {c.get('score')}; terms: {', '.join(c.get('terms') or [])})"
             for i, c in enumerate(state.candidates[:10], 1)] or ["(no keyword matches; explore with search_code)"]
    return _cap_task(
        "PHASE: LOCALIZE. Goal: find the exact code responsible for this issue. Do NOT edit anything.\n"
        f"Issue: {_issue_line(state)}\n"
        f"Expected: {_t(issue.expected, 500) or 'not stated'}\n"
        f"Actual: {_t(issue.actual, 500) or 'not stated'}\n"
        f"Errors: {_t(errors, 800)}\n"
        "Candidate files (pre-ranked by keyword match; may be wrong):\n" + "\n".join(cands) + "\n"
        "Outline of the top candidates:\n" + (_t(repo_map_text, 3500) or "(none)") + "\n"
        "Method: confirm or reject candidates with search_code/view_file; trace from the symptom to the root cause\n"
        "(callers and callees); identify the smallest set of functions that must change.\n"
        f"You have {max_steps} tool calls. Finish with files, symbols, root_cause (2-4 sentences), confidence.")


def reproduce_task(state: RunState) -> str:
    """REPRODUCE: a scratch script that fails while the bug exists."""
    issue, loc = state.issue, state.localization
    return _cap_task(
        "PHASE: REPRODUCE. Goal: a minimal script that demonstrates the problem.\n"
        f"Issue: {_issue_line(state)}\n"
        f"Expected: {_t(issue.expected, 500) or 'not stated'}  |  Actual: {_t(issue.actual, 500) or 'not stated'}\n"
        + (f"Reproduction hints: {_t(issue.repro_hints, 600)}\n" if issue.repro_hints else "")
        + "Create @scratch/repro.py (or the repository's language) that exercises the behaviour described in the "
        "issue.\n"
        'It must print "BUG PRESENT" and exit with code 1 while the problem exists, and print "BUG FIXED" and exit 0\n'
        "once it is fixed. For a feature request, check the requested behaviour the same way.\n"
        "Run it with run_command (cwd = repository root; the repository root is on PYTHONPATH).\n"
        "Do not modify repository files. If a script cannot reproduce it (needs network/UI/credentials), finish with\n"
        "reproduced=false and explain in observed.\n"
        f"Localization: {_t(', '.join(loc.get('files') or []), 400) or 'unknown'} / "
        f"{_t(', '.join(loc.get('symbols') or []), 400) or 'unknown'}\n"
        f"Root cause hypothesis: {_t(loc.get('root_cause'), 1000) or 'unknown'}\n"
        "Finish with reproduced, command, observed.")


def fix_task(state: RunState, attempt: int, feedback: str = "", max_attempts: int = 3) -> str:
    """FIX: minimal edit of the root cause, verified by the repro and relevant tests."""
    issue, loc, repro = state.issue, state.localization, state.repro
    before = repro.get("before") or {}
    if repro.get("reproduced") and repro.get("command"):
        repro_line = (f"Reproduction: `{_t(repro.get('command'), 300)}` currently outputs:\n"
                      f"{_t(before.get('output_tail') or repro.get('observed'), 1200) or '(no output)'}")
    else:
        repro_line = ("Reproduction: none (the issue could not be reproduced by a script). Observed: "
                      f"{_t(repro.get('observed'), 600) or 'n/a'}")
    targeted = ", ".join(state.targeted_tests[:15]) or "none found"
    feedback_block = f"PREVIOUS ATTEMPT FAILED VERIFICATION:\n{_t(feedback, 2500)}\n" if feedback else ""
    return _cap_task(
        f"PHASE: FIX (attempt {attempt} of {max_attempts}). Goal: a correct, minimal fix of the root cause.\n"
        f"Issue: {_issue_line(state)}  |  Expected: {_t(issue.expected, 400) or 'not stated'}  |  "
        f"Actual: {_t(issue.actual, 400) or 'not stated'}\n"
        f"Root cause analysis: {_t(loc.get('root_cause'), 1000) or 'unknown'}   "
        f"Files: {_t(', '.join(loc.get('files') or []), 400) or 'unknown'}\n"
        f"{repro_line}\n"
        f"Relevant tests: {targeted}\n"
        f"Tests already failing BEFORE your change (not your concern unless related to this issue): "
        f"{_baseline_failures(state)}\n"
        f"{feedback_block}"
        "Process: 1) view the exact code to change  2) apply minimal str_replace edit(s)  3) run the reproduction\n"
        "command  4) run run_tests on the relevant tests  5) if the repository has a test suite, add one focused\n"
        "regression test in its existing style (a new test function or file)  6) call finish.\n"
        "Think about edge cases the issue implies (empty/None input, boundaries, types) and other callers of the\n"
        "code you change.")


def _counts(run: Any) -> str:
    """'3 passed / 1 failed' for a TestRun or dict, or 'not run'."""
    if run is None:
        return "not run"
    return f"{_get(run, 'passed', 0)} passed / {_get(run, 'failed', 0) + _get(run, 'errors', 0)} failed"


def format_verification(v: dict) -> str:
    """Render the orchestrator's verification record.

    Expected keys (all optional): repro_before_exit, repro_after_exit, repro_passed, targeted_before,
    targeted_after, full_before, full_after (TestRun or dict), new_failures, fixed (lists of test IDs).
    """
    lines = [f"Reproduction exit code: before {v.get('repro_before_exit', 'n/a')} -> after "
             f"{v.get('repro_after_exit', 'n/a')} (passed: {v.get('repro_passed', 'n/a')})",
             f"Targeted tests: before {_counts(v.get('targeted_before'))} -> after {_counts(v.get('targeted_after'))}",
             f"Full suite: before {_counts(v.get('full_before'))} -> after {_counts(v.get('full_after'))}",
             f"New failures: {', '.join(v.get('new_failures') or []) or 'none'}",
             f"Fixed tests: {', '.join(v.get('fixed') or []) or 'none'}"]
    return "\n".join(lines)


def review_prompt(state: RunState, diff: str, verification: dict) -> str:
    """REVIEW: one call, no tools, JSON verdict on the patch."""
    issue = state.issue
    return _cap_task(
        "You are a strict but fair code reviewer. Decide whether this patch correctly resolves the issue.\n"
        f"Issue: {_issue_line(state)}  |  Expected: {_t(issue.expected, 400) or 'not stated'}\n"
        "Patch:\n```diff\n" + _t(diff, 4500) + "\n```\n"
        "Verification evidence:\n" + _t(format_verification(verification or {}), 1200) + "\n"
        "Check: fixes the root cause (not just the symptom)? breaks other callers? edge cases handled? minimal?\n"
        "leftover debug code? Respond with ONLY JSON:\n"
        '{"verdict": "approve"|"revise", "problems": [str], "confidence": "high"|"medium"|"low"}\n'
        'Use "revise" only for concrete, fixable problems.')

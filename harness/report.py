"""Writes the per-run artefacts (report.md, patch.diff, metrics.json, state.json)."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import harness
from harness.events import redact
from harness.spectrum import touched_ranks
from harness.types import Metrics, RunState

STATUS_LINES = {
    "verified": "✅ VERIFIED FIX",
    "unverified": "⚠️ UNVERIFIED CHANGE",
    "no_fix": "❌ NO FIX",
    "error": "⛔ ERROR",
}
TAIL_LINES = 5


def _get(obj: Any, key: str, default: Any = None) -> Any:
    """Read a key from a dict or an attribute from an object (TestRun or dict)."""
    if obj is None:
        return default
    return obj.get(key, default) if isinstance(obj, dict) else getattr(obj, key, default)


def _cell(text: Any) -> str:
    """Make text safe for a single Markdown table cell."""
    return str(text).replace("|", "\\|").replace("\n", " ").strip() or "—"


def status_line(state: RunState, has_diff: bool) -> str:
    """Headline for the report; budget_exhausted maps to UNVERIFIED CHANGE or NO FIX."""
    if state.status == "budget_exhausted":
        return (STATUS_LINES["unverified"] if has_diff else STATUS_LINES["no_fix"]) + " (budget exhausted)"
    return STATUS_LINES.get(state.status, f"❔ {state.status.upper()}")


def final_attempt(state: RunState, diff: str) -> tuple[dict | None, bool]:
    """The attempt that produced the final patch, else the last verified one (flagged as not kept)."""
    verified = [a for a in state.attempts if a.get("verification")]
    for attempt in reversed(verified):
        if diff and attempt.get("diff") == diff:
            return attempt, True
    return (verified[-1], False) if verified else (None, False)


def _counts(run: Any) -> str:
    if run is None:
        return "not run"
    failed = (_get(run, "failed", 0) or 0) + (_get(run, "errors", 0) or 0)
    extra = " (timed out)" if _get(run, "timed_out") else ""
    return f"{_get(run, 'passed', 0)} passed / {failed} failed{extra}"


def _tail(text: Any, n: int = TAIL_LINES) -> str:
    lines = [line for line in str(text or "").splitlines() if line.strip()]
    return "\n".join(lines[-n:]) or "(no output)"


def _exit(code: Any) -> str:
    return "n/a" if code is None else f"exit {code}"


def evidence_rows(state: RunState, attempt: dict) -> list[tuple[str, str, str]]:
    """(check, before, after) rows shared by report.md and the terminal UI."""
    v = attempt.get("verification") or {}
    repro = state.repro
    new = v.get("new_failures") or []
    review = attempt.get("review") or {}
    if repro.get("reproduced"):
        repro_row = (f"Reproduction ({repro.get('command')})", _exit(v.get("repro_before_exit")),
                     _exit(v.get("repro_after_exit")))
    else:
        repro_row = ("Reproduction", "not reproduced", str(repro.get("observed") or "n/a"))
    verdict = str(review.get("verdict", "not run")) + (f" ({review['confidence']})" if review.get("confidence") else "")
    return [repro_row,
            (f"Targeted tests ({len(state.targeted_tests)} file(s))", _counts(v.get("targeted_before")),
             _counts(v.get("targeted_after"))),
            ("Full suite", _counts(v.get("full_before")), _counts(v.get("full_after"))),
            ("New failures (must be empty)", "—", "none ✅" if not new else "❌ " + ", ".join(new)),
            ("Fixed tests", "—", ", ".join(v.get("fixed") or []) or "none"),
            ("Review verdict", "—", verdict)]


def evidence_section(state: RunState, attempt: dict | None, kept: bool) -> list[str]:
    """The Evidence table plus reproduction output tails."""
    out = ["## Evidence", ""]
    if attempt is None:
        base = state.baseline_full
        out += ["No fix attempt reached verification.", "",
                f"Baseline full suite: {_counts(base) if base is not None else 'not available'}", ""]
        return out
    v = attempt["verification"]
    if not kept:
        out += [f"_Showing attempt {attempt['attempt']}, which was **not kept** (its changes were reverted)._", ""]
    repro = state.repro
    review = attempt.get("review") or {}
    out += ["| Check | Before | After |", "| --- | --- | --- |"]
    out += [f"| {_cell(check)} | {_cell(before)} | {_cell(after)} |" for check, before, after in
            evidence_rows(state, attempt)]
    out.append("")
    for note in v.get("notes") or []:
        out.append(f"- {note}")
    if repro.get("reproduced"):
        before = (repro.get("before") or {}).get("output_tail")
        out += [f"Reproduction output before the fix (last {TAIL_LINES} lines):", "```",
                _tail(before), "```", f"Reproduction output after the fix (last {TAIL_LINES} lines):", "```",
                _tail(v.get("repro_after_tail")), "```"]
    if review.get("problems"):
        out += ["", "Reviewer problems:"] + [f"- {p}" for p in review["problems"]]
    return out + [""]


def fault_localization_section(state: RunState, diff: str) -> list[str]:
    """Tracer results: top suspicious functions and which of them the patch touches."""
    spec = state.spectrum or {}
    out = ["### Fault localization", ""]
    if not spec:
        return out + ["Tracer: not run.", ""]
    if not spec.get("ok"):
        return out + [f"Tracer skipped: {spec.get('reason') or 'unknown reason'}", ""]
    functions = spec.get("functions") or []
    nf, np_ = spec.get("failing_runs", 0), spec.get("passing_runs", 0)
    formula = "Tarantula" if spec.get("formula") == "tarantula" else "Ochiai"
    out += [f"{formula} over {nf} failing and {np_} passing run(s)"
            f"{' (low confidence)' if spec.get('low_confidence') else ''}:", "",
            "| Rank | Location | Function | Score | Fail/pass hits |", "| ---: | --- | --- | ---: | --- |"]
    for i, f in enumerate(functions[:5], 1):
        out.append(f"| {i} | {_cell(f['path'])}:{f['start']}-{f['end']} | {_cell(f['name'])} | {f['score']:.2f} | "
                   f"fail {f.get('ef', 0)}/{nf}, pass {f.get('ep', 0)}/{np_} |")
    ranks = touched_ranks(diff, functions) if diff else []
    touched = ", ".join(f"#{r}" for r in ranks)
    out += ["", f"Patch touches suspicious rank(s): {touched}" if ranks else
            ("Patch touches no ranked function." if diff else "No patch to compare."), ""]
    return out


def efficiency_section(metrics: Metrics) -> list[str]:
    """Per-phase and total LLM calls, tokens, tool calls and seconds."""
    out = ["## Efficiency", "", "| Phase | LLM calls | Prompt tokens | Completion tokens | Tool calls | Seconds |",
           "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for name, pm in metrics.per_phase.items():
        out.append(f"| {name} | {pm.llm_calls} | {pm.prompt_tokens} | {pm.completion_tokens} | {pm.tool_calls} | "
                   f"{pm.seconds:.1f} |")
    total = metrics.to_dict()["total"]
    out.append(f"| **Total** | **{total['llm_calls']}** | **{total['prompt_tokens']}** | "
               f"**{total['completion_tokens']}** | **{total['tool_calls']}** | **{total['seconds']:.1f}** |")
    return out + ["", f"Total tokens: {total['total_tokens']}", ""]


def render_report(state: RunState, diff: str, metrics: Metrics) -> str:
    """The full report.md text."""
    issue, loc = state.issue, state.localization
    attempt, kept = final_attempt(state, diff)
    title = issue.title or (issue.raw_text.strip().splitlines() or ["(untitled issue)"])[0]
    lines = [f"# {status_line(state, bool(diff))}", "",
             f"**Issue:** {title}  ", f"**Repository:** `{state.repo}`  ", f"**Run:** `{state.run_id}`", "",
             "## Summary", "", f"- **Issue summary:** {issue.summary or '(none)'}",
             f"- **Root cause:** {loc.get('root_cause') or '(not determined)'}",
             f"- **Files changed:** {', '.join(f'`{f}`' for f in (attempt or {}).get('files', [])) if kept else 'none'}",
             f"- **Fix summary:** {(attempt or {}).get('summary') or '(none)'}" if kept else "- **Fix summary:** none",
             f"- **Attempts:** {len(state.attempts)}", ""]
    lines += evidence_section(state, attempt, kept)
    lines += fault_localization_section(state, diff)
    lines += efficiency_section(metrics)
    lines += ["## Patch", "", "```diff", diff.rstrip("\n") or "(no changes)", "```", ""]
    if state.notes:
        lines += ["## Notes", ""] + [f"- {n}" for n in state.notes] + [""]
    lines.append(f"_Generated by AI Coding Harness v{harness.__version__}._")
    return "\n".join(lines) + "\n"


def state_for_json(state: RunState) -> dict:
    """state.to_dict() without attempt snapshots (full file contents)."""
    data = state.to_dict()
    for attempt in data.get("attempts", []):
        attempt.pop("snapshot", None)
    return data


def write_report(state: RunState, ws: Any, metrics: Metrics, run_dir: Path) -> dict[str, Path]:
    """Write patch.diff, metrics.json, state.json and report.md into run_dir; return their paths."""
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    diff = ws.diff()
    paths = {"patch": run_dir / "patch.diff", "metrics": run_dir / "metrics.json",
             "state": run_dir / "state.json", "report": run_dir / "report.md"}
    # The key can only appear via echoed error messages; redact every artefact anyway.
    paths["patch"].write_text(redact(diff), encoding="utf-8")
    paths["metrics"].write_text(redact(json.dumps(metrics.to_dict(), indent=2)), encoding="utf-8")
    paths["state"].write_text(redact(json.dumps(state_for_json(state), indent=2, default=str)), encoding="utf-8")
    paths["report"].write_text(redact(render_report(state, diff, metrics)), encoding="utf-8")
    return paths

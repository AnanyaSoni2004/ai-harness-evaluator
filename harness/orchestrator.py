"""Phase state machine, budgets, fix attempts, and rescue."""
from __future__ import annotations

import concurrent.futures
import contextlib
import hashlib
import json
import re
import tempfile
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Iterator

from harness import prompts, report
from harness.agent import AgentLoop
from harness.events import NullUI, Trajectory, redact
from harness.localize import extract_terms, rank_files, related_tests
from harness.repomap import build_repo_map
from harness.spectrum import SpectrumAnalyzer, format_suspicious
from harness.testing import TestRunner, compare
from harness.textproto import first_json_object
from harness.tools.exec_tools import run_command
from harness.tools.registry import build_registry, make_finish_tool
from harness.types import BudgetExceeded, ContextOverflow, HarnessError, IssueSpec, Metrics, RunState, TestRun
from harness.workspace import Workspace

PROJECT_ROOT = Path(__file__).resolve().parents[1]
READ_TOOLS = ["list_dir", "find_files", "search_code", "view_file", "repo_map"]
REPRO_TOOLS = READ_TOOLS + ["create_file", "str_replace", "run_command"]
STR, STRS = {"type": "string"}, {"type": "array", "items": {"type": "string"}}
LOCALIZE_FINISH = ({"files": STRS, "symbols": STRS, "root_cause": STR,
                    "confidence": {"type": "string", "enum": ["high", "medium", "low"]}}, ["files", "root_cause"])
REPRO_FINISH = ({"reproduced": {"type": "boolean"}, "command": STR, "observed": STR}, ["reproduced", "observed"])
FIX_FINISH = ({"summary": STR, "files_changed": STRS, "tests_added": STRS}, ["summary"])
TAIL = 1500
_NO_WORKSPACE = type("NoWorkspace", (), {"diff": staticmethod(lambda: "")})()  # when setup failed


def _truthy(value: Any) -> bool:
    return value is True or str(value).strip().lower() in ("true", "yes", "1")


def _strs(value: Any) -> list[str]:
    """Coerce a model-supplied list (or single string) into a list of non-empty strings."""
    if isinstance(value, str):
        value = [value]
    return [str(v).strip() for v in (value or []) if str(v).strip()] if isinstance(value, list) else []


def _slug(text: str) -> str:
    words = re.sub(r"[^a-z0-9]+", " ", text.lower()).split()[:5]
    return "-".join(words)[:40].strip("-") or "issue"


class Orchestrator:
    """Runs INTAKE -> LOCALIZE -> REPRODUCE -> FIX <-> VERIFY -> REVIEW -> REPORT for one issue."""

    def __init__(self, cfg: Any, llm: Any, ui: Any = None, python_exe: str | None = None) -> None:
        """python_exe is passed to TestRunner (tests use sys.executable)."""
        self.cfg, self.llm = cfg, llm
        self.ui = ui if ui is not None else NullUI()
        self.python_exe = python_exe

    # ------------------------------------------------------------------ setup helpers
    def _run_dir(self, issue_text: str) -> Path:
        runs = Path(self.cfg.output.runs_dir)
        runs = runs if runs.is_absolute() else PROJECT_ROOT / runs
        base = runs / f"{datetime.now().strftime('%Y%m%d-%H%M%S')}-{_slug(issue_text)}"
        run_dir, n = base, 2
        while run_dir.exists():
            run_dir, n = base.with_name(f"{base.name}-{n}"), n + 1
        run_dir.mkdir(parents=True)
        return run_dir

    @contextlib.contextmanager
    def _timed(self, phase: str, detail: str = "") -> Iterator[None]:
        self.ui.phase(phase.upper(), detail)
        self.trajectory.log("phase", phase=phase, detail=detail)
        start = time.monotonic()
        try:
            yield
        finally:
            self.metrics.add_time(phase, time.monotonic() - start)

    def _remaining(self) -> float:
        return self.deadline - time.monotonic()

    def _note(self, text: str) -> None:
        self.state.notes.append(text)
        self.trajectory.log("note", text=text)
        self.ui.info(text)

    def _agent(self, phase: str, tools: list[str] | None, finish: tuple, task: str, max_steps: int) -> Any:
        registry = self.registry.subset(tools) if tools is not None else self.registry.subset(self.registry.names())
        description = f"End the {phase.upper()} phase and report its result."
        registry.register(make_finish_tool(finish[0], finish[1], description))
        loop = AgentLoop(self.llm, registry, prompts.system_prompt(phase, str(self.ws.repo_root)), task, phase,
                         self.cfg, self.metrics, self.trajectory, self.ui, max_steps, self.deadline)
        return loop.run()

    def _ask_json(self, phase: str, prompt: str) -> dict | None:
        resp = self.llm.complete([{"role": "user", "content": prompt}], None, phase)
        return first_json_object(resp.text)

    # ------------------------------------------------------------------ phases
    def _intake(self, issue_text: str) -> None:
        with self._timed("intake", "understanding the issue"):
            try:
                obj = self._ask_json("intake", prompts.intake_prompt(issue_text))
            except ContextOverflow:
                obj = None
                self._note("intake: prompt too large for the model; using the raw issue text")
            if not obj:
                self._note("intake: no JSON in reply; using the raw issue text")
                self.state.issue = IssueSpec(raw_text=issue_text, summary=issue_text[:500])
                return
            kind = str(obj.get("kind", "bug")).lower()
            kind = kind if kind in ("bug", "feature", "other") else "bug"
            self.state.issue = IssueSpec(
                raw_text=issue_text, title=str(obj.get("title") or ""), kind=kind,
                summary=str(obj.get("summary") or issue_text[:500]), expected=str(obj.get("expected") or ""),
                actual=str(obj.get("actual") or ""), error_messages=_strs(obj.get("error_messages")),
                mentioned_paths=_strs(obj.get("mentioned_paths")),
                mentioned_symbols=_strs(obj.get("mentioned_symbols")),
                repro_hints=str(obj.get("repro_hints") or ""))

    def _prelocalize(self) -> str:
        with self._timed("prelocalize", "ranking files (no LLM)"):
            terms = dict(extract_terms(self.state.issue.raw_text))
            for term in self.state.issue.mentioned_symbols + self.state.issue.mentioned_paths:
                if len(term) >= 3:
                    terms[term] = max(terms.get(term, 0.0), 4.0)
            self.state.candidates = rank_files(self.ws, sorted(terms.items(), key=lambda kv: -kv[1]), top_k=10)
            return build_repo_map(self.ws, [c["path"] for c in self.state.candidates[:5]])

    def _localize(self, repo_map_text: str) -> None:
        with self._timed("localize", "finding the root cause"):
            self.ws.write_scope = "none"
            steps = self.cfg.phases.localize_max_steps
            res = self._agent("localize", READ_TOOLS, LOCALIZE_FINISH,
                              prompts.localize_task(self.state, repo_map_text, steps), steps)
            loc = dict(res.result) if res.finished else {"root_cause": str(res.result.get("text", ""))[:1000]}
            files = [f for f in _strs(loc.get("files")) if (self.ws.repo_root / f).is_file()]
            if not files:
                files = [c["path"] for c in self.state.candidates[:3]]
                self._note(f"localize: {'no usable files' if res.finished else res.reason}; using top candidates")
            loc.update(files=files, symbols=_strs(loc.get("symbols")), finished=res.finished)
            self.state.localization = loc
            self.state.targeted_tests = related_tests(self.ws, files)
            if self.state.targeted_tests:
                self.state.baseline_targeted = self.runner.run(self.state.targeted_tests)

    def _run_repro(self) -> dict:
        res = run_command(self.ws, self.cfg, command=self.state.repro["command"])
        return {"exit_code": res.data.get("exit_code"), "output_tail": res.output[-TAIL:]}

    def _reproduce(self) -> None:
        with self._timed("reproduce", "demonstrating the bug"):
            self.ws.write_scope = "scratch"
            res = self._agent("reproduce", REPRO_TOOLS, REPRO_FINISH, prompts.reproduce_task(self.state),
                              self.cfg.phases.reproduce_max_steps)
            r = res.result if res.finished else {}
            command = str(r.get("command") or "").strip()
            self.state.repro = {"reproduced": _truthy(r.get("reproduced")) and bool(command),
                                "command": command, "observed": str(r.get("observed") or "")}
            if self.state.repro["reproduced"]:
                before = self._run_repro()
                self.state.repro["before"] = before
                if before["exit_code"] in (0, None):
                    self.state.repro["reproduced"] = False
                    self._note(f"reproduce: repro exited {before['exit_code']} before any fix; not used as a gate")

    def _trace(self) -> None:
        """TRACE (no LLM): rank suspicious code from the failing repro vs passing tests. Purely additive."""
        with self._timed("trace", "execution-based fault localization (no LLM)"):
            scfg = self.cfg.spectrum
            if scfg.enabled and self._remaining() < float(scfg.timeout_s) + 300:
                spectrum: dict = {"ok": False, "reason": "low time budget"}
            else:
                if self._baseline_future is not None and not self._baseline_future.done():
                    self.trajectory.log("trace_overlaps_baseline")  # both read-only; they only share CPU
                try:
                    result = SpectrumAnalyzer(self.ws, self.cfg, self.runner, self.ui, self.trajectory).analyze(self.state)
                    spectrum = result.to_dict()
                    if result.ok:
                        spectrum["evidence"] = format_suspicious(result, self.ws, int(scfg.top_lines),
                                                                 int(scfg.max_evidence_chars))
                except Exception as e:  # noqa: BLE001 - the Tracer must never break a run
                    spectrum = {"ok": False, "reason": f"crashed: {type(e).__name__}: {e}"}
            self.state.spectrum = spectrum
            self.ui.show_spectrum(spectrum)

    def _join_baseline(self) -> None:
        if self._baseline_future is None:
            return
        future, self._baseline_future = self._baseline_future, None
        try:
            self.state.baseline_full = future.result(timeout=max(1.0, self._remaining()))
        except concurrent.futures.TimeoutError:
            self._note("baseline: full test run did not finish within the time budget")
        except Exception as e:  # noqa: BLE001 - a broken test runner must not end the run
            self._note(f"baseline: test run failed ({type(e).__name__}: {e}); continuing without a baseline")

    def _verify(self, tests_added: list[str]) -> dict:
        with self._timed("verify", "re-running reproduction and tests (no LLM)"):
            self._join_baseline()
            v: dict = {"new_failures": [], "fixed": [], "notes": []}
            repro = self.state.repro
            if repro.get("reproduced"):
                after = self._run_repro()
                v.update(repro_before_exit=repro["before"]["exit_code"], repro_after_exit=after["exit_code"],
                         repro_passed=after["exit_code"] == 0, repro_after_tail=after["output_tail"])
            targets = list(dict.fromkeys(self.state.targeted_tests +
                                         [t for t in tests_added if (self.ws.repo_root / t).is_file()]))
            if targets:
                v["targeted_before"], v["targeted_after"] = self.state.baseline_targeted, self.runner.run(targets)
                cmp = compare(self.state.baseline_targeted, v["targeted_after"])
                v["new_failures"] += cmp["new_failures"]
                v["fixed"] += cmp["fixed"]
            base = self.state.baseline_full
            if self.cfg.tests.run_full_suite_after_fix and base is not None:
                if self._remaining() > base.duration_s * 1.5 + 10:
                    v["full_before"], v["full_after"] = base, self.runner.run(None)
                    cmp = compare(base, v["full_after"])
                    v["new_failures"] += [f for f in cmp["new_failures"] if f not in v["new_failures"]]
                    v["fixed"] += [f for f in cmp["fixed"] if f not in v["fixed"]]
                else:
                    v["notes"].append("full suite skipped: not enough time left")
            v["passed"] = (v.get("repro_passed", True) is True) and not v["new_failures"]
            v["score"] = 2 * int(bool(v.get("repro_passed"))) + len(v["fixed"]) - 3 * len(v["new_failures"])
            return v

    def _feedback(self, v: dict) -> str:
        parts = []
        if v.get("repro_passed") is False:
            parts.append(f"The reproduction still fails (exit {v.get('repro_after_exit')}):\n"
                         f"{v.get('repro_after_tail', '')}")
        if v["new_failures"]:
            tails = [getattr(v.get(k), "output_tail", "") for k in ("targeted_after", "full_after")]
            parts.append("New failing tests (they passed before your change): " + ", ".join(v["new_failures"]) +
                         "\n" + next((t for t in tails if t), "")[-TAIL:])
        return "\n\n".join(parts) or "Verification failed."

    def _review(self, v: dict) -> dict:
        with self._timed("review", "critic checks the patch"):
            try:
                obj = self._ask_json("review", prompts.review_prompt(self.state, self.ws.diff(), v)) or {}
            except ContextOverflow:
                obj = {}
                self._note("review: prompt too large for the model; the reviewer is advisory, so approving")
            verdict = str(obj.get("verdict", "")).lower()
            problems = _strs(obj.get("problems"))
            if verdict != "revise" or not problems:
                if verdict not in ("approve", "revise"):
                    self._note("review: no usable verdict; treating as approve (the reviewer is advisory)")
                verdict = "approve"
            return {"verdict": verdict, "problems": problems, "confidence": str(obj.get("confidence", ""))}

    def _attempt(self, n: int, kind: str, feedback: str) -> tuple[bool, str]:
        """One FIX + VERIFY (+ REVIEW). Returns (success, feedback for the next attempt)."""
        with self._timed("fix", f"attempt {n} ({kind})"):
            self.ws.write_scope = "repo"
            self.runner.known_failures = {t for run in (self.state.baseline_full, self.state.baseline_targeted)
                                          for t in (getattr(run, "failing_ids", None) or [])}
            max_attempts = self.cfg.phases.max_fix_attempts if kind == "fix" else n
            res = self._agent("fix", None, FIX_FINISH, prompts.fix_task(self.state, n, feedback, max_attempts),
                              self.cfg.phases.fix_max_steps)
        diff = self.ws.diff()
        record: dict = {"attempt": n, "kind": kind, "finished": res.finished, "reason": res.reason,
                        "summary": str(res.result.get("summary") or res.result.get("text") or "")[:1000],
                        "files": self.ws.edited_files(), "diff": diff, "passed": False,
                        "diff_hash": hashlib.sha256(diff.encode()).hexdigest()[:12] if diff else ""}
        self.state.attempts.append(record)
        if not record["diff"]:
            record["reason"] = record["reason"] or "no_changes"
            return False, "No changes were made to repository files."
        v = self._verify(_strs(res.result.get("tests_added")))
        record.update(verification=v, passed=v["passed"], score=v["score"], snapshot=self.ws.snapshot())
        if not v["passed"]:
            return False, self._feedback(v)
        if not self.cfg.phases.enable_review:
            return True, ""
        record["review"] = self.state.review = self._review(v)
        if record["review"]["verdict"] == "approve":
            return True, ""
        return False, "Reviewer requested changes:\n- " + "\n- ".join(record["review"]["problems"])

    def _abandon_reason(self) -> str | None:
        """Why no further fix attempt should start (checked before attempt 2+ and the rescue)."""
        soft = int(getattr(self.cfg.budgets, "soft_total_tokens", 0) or 0)
        if soft and self.metrics.total_tokens >= soft:
            return f"soft token budget reached ({self.metrics.total_tokens} >= {soft} tokens)"
        if self._remaining() <= 0:
            return "wall-clock budget exhausted"
        tried = self.state.attempts
        if len(tried) >= 2:
            a, b = tried[-2], tried[-1]
            if not a.get("diff") and not b.get("diff"):
                return "two consecutive fix attempts produced no changes"
            if a.get("diff_hash") and a.get("diff_hash") == b.get("diff_hash"):
                return "two consecutive fix attempts produced the same diff"
        if sum(1 for a in tried if a.get("reason") == "no_tool_calls") >= 2:
            return "two fix attempts ended without any tool call"
        return None

    def _stop_early(self, reason: str) -> None:
        self.state.stop_reason = reason
        self._note(f"stopped early: {reason}")

    def _fix_loop(self) -> bool:
        feedback = ""
        for n in range(1, self.cfg.phases.max_fix_attempts + 1):
            reason = self._abandon_reason() if n > 1 else (
                "wall-clock budget exhausted" if self._remaining() <= 0 else None)
            if reason:
                self._stop_early(reason)
                return False
            ok, feedback = self._attempt(n, "fix", feedback)
            if ok:
                return True
        reason = self._abandon_reason()
        if reason and self.cfg.phases.enable_rescue:
            self._stop_early(reason)
            return False
        if self.cfg.phases.enable_rescue:
            self.ws.revert_all()
            lines = []
            for a in self.state.attempts:
                outcome = "passed tests but not approved" if a["passed"] else a.get("reason") or "failed verification"
                lines.append(f"attempt {a['attempt']}: {outcome} ({', '.join(a['files']) or 'no files'})")
            feedback = ("Previous attempts failed: " + "; ".join(lines) +
                        ". Start from a clean slate and reconsider the root cause.")
            edited = sorted({f for a in self.state.attempts for f in a.get("files", [])})
            if self.state.spectrum.get("ok") and edited:
                feedback += (f"\nThe previous attempts edited {', '.join(edited)}; if those are not in the top "
                             "suspicious functions, reconsider.")
            ok, _ = self._attempt(len(self.state.attempts) + 1, "rescue", feedback)
            return ok
        return False

    # ------------------------------------------------------------------ finish
    def _finalize(self, success: bool, budget_hit: bool, error: bool, interrupted: bool = False) -> None:
        if self.ws is None:  # setup itself failed
            self.state.status = "error"
            return
        if interrupted and not any(a.get("passed") for a in self.state.attempts):
            if self.ws.diff():
                self.ws.revert_all()
                self._note("interrupted: unverified changes were reverted")
            success = False
        elif not success:
            tried = [a for a in self.state.attempts if a.get("snapshot") is not None]
            best = max(tried, key=lambda a: (a["passed"], a["score"]), default=None)
            if best is not None:
                self.ws.restore(best["snapshot"])
                if best["score"] <= 0 and best["verification"]["new_failures"]:
                    self.ws.revert_all()
                    self._note("final: best attempt made things worse; all changes reverted")
        self.ws.write_scope = "none"
        if error:
            self.state.status = "error"
        elif interrupted:
            self.state.status = "interrupted"
        elif success:
            self.state.status = "verified"
        elif budget_hit:
            self.state.status = "budget_exhausted"
        else:
            self.state.status = "unverified" if self.ws.diff() else "no_fix"

    def _write_report(self) -> None:
        self.metrics.ended_at = time.time()
        writer = getattr(report, "write_report", None)
        ws = self.ws if self.ws is not None else _NO_WORKSPACE
        try:
            if writer is not None:
                writer(self.state, ws, self.metrics, self.run_dir)
                return
        except Exception as e:  # noqa: BLE001 - never crash without a report
            self._note(f"report: writer failed ({type(e).__name__}: {e}); wrote the minimal report")
        state = self.state.to_dict()
        for attempt in state.get("attempts", []):
            attempt.pop("snapshot", None)
        (self.run_dir / "patch.diff").write_text(redact(ws.diff()), encoding="utf-8")
        (self.run_dir / "state.json").write_text(redact(json.dumps(state, indent=2, default=str)), encoding="utf-8")

    def _verify_after_budget(self) -> None:
        """After a budget stop, verify the current changes unless an attempt already verified exactly them."""
        diff = self.ws.diff()
        # The budget may run out mid-attempt, before that attempt is recorded.
        if diff and not any(a.get("verification") and a.get("diff") == diff for a in self.state.attempts):
            v = self._verify([])
            self.state.attempts.append({
                "attempt": len(self.state.attempts) + 1, "kind": "budget", "passed": v["passed"],
                "score": v["score"], "verification": v, "files": self.ws.edited_files(),
                "diff": self.ws.diff(), "snapshot": self.ws.snapshot(), "summary": ""})

    def _setup(self, repo: Path, issue_text: str) -> None:
        """Workspace, test runner, tools and the background baseline (inside solve's protected block)."""
        scratch = self.run_dir / "scratch"
        if re.search(r"\s", str(scratch)):  # @scratch/ is expanded textually into shell commands
            scratch = Path(tempfile.mkdtemp(prefix="harness-scratch-"))
        self.ws = Workspace(repo, scratch, self.cfg)
        self.runner = TestRunner(self.ws, self.cfg, self.python_exe)
        self.registry = build_registry(self.ws, self.cfg, self.runner)
        self.runner.detect()
        self.ws.python_exe = self.runner.python()  # `python` in run_command = the test interpreter
        self._pool = concurrent.futures.ThreadPoolExecutor(1)
        self._baseline_future = self._pool.submit(self.runner.run, None, self.cfg.tests.baseline_timeout_s)

    def solve(self, repo: Path, issue_text: str) -> tuple[RunState, Path]:
        """Resolve one issue in repo. Always returns the final state and the run directory.

        Every failure ends as a status plus a written report. Only KeyboardInterrupt propagates, after the
        report is written, so the CLI can exit 130.
        """
        repo = Path(repo).resolve()
        if not repo.is_dir():
            raise HarnessError(f"Repository not found: {repo}")
        self.run_dir = self._run_dir(issue_text)
        self.trajectory = Trajectory(self.run_dir / "trajectory.jsonl")
        self.metrics = Metrics()
        self.llm.metrics = self.metrics
        self.llm.trajectory = self.trajectory
        self.deadline = time.monotonic() + self.cfg.budgets.max_wall_clock_s
        self.state = RunState(run_id=self.run_dir.name, repo=str(repo), issue=IssueSpec(raw_text=issue_text,
                                                                                         summary=issue_text[:500]))
        self.ws = None
        self._pool = None
        self._baseline_future = None
        self.trajectory.log("run_start", repo=str(repo), model=getattr(self.cfg.model, "name", ""),
                            tool_mode=getattr(self.llm, "tool_mode", ""))
        success = budget_hit = error = interrupted = False
        try:
            self._setup(repo, issue_text)
            self._intake(issue_text)
            self._localize(self._prelocalize())
            self._reproduce()
            self._trace()
            self._join_baseline()
            success = self._fix_loop()
        except BudgetExceeded as e:
            budget_hit = True
            self._note(f"budget exhausted: {e}")
            try:
                self._verify_after_budget()
            except Exception as verify_error:  # noqa: BLE001
                self._note(f"verification after the budget stop failed: {type(verify_error).__name__}: {verify_error}")
        except KeyboardInterrupt:
            interrupted = True
            self._note("interrupted by the user (Ctrl-C)")
            raise
        except Exception as e:  # noqa: BLE001 - FatalLLMError or a bug: report it, never crash silently
            error = True
            self._note(f"error: {type(e).__name__}: {e}")
            self.trajectory.log("error", traceback=traceback.format_exc())
        finally:
            if self._pool is not None:
                self._pool.shutdown(wait=False)
            try:
                self._finalize(success, budget_hit, error, interrupted)
            except Exception as e:  # noqa: BLE001 - finalizing must not hide the report
                self.state.status = "error"
                self._note(f"finalizing failed: {type(e).__name__}: {e}")
                self.trajectory.log("error", traceback=traceback.format_exc())
            finally:
                self._write_report()
                self.trajectory.log("run_end", status=self.state.status, tokens=self.metrics.total_tokens)
        return self.state, self.run_dir

"""Tracer: spectrum-based fault localization (scoring, aggregation, collection, prompt rendering)."""
from __future__ import annotations

import ast
import json
import math
import re
import shlex
import time
from collections import Counter
from pathlib import Path
from typing import Any

from harness.events import NullUI, Trajectory
from harness.shell import run_process, truncate
from harness.types import SpectrumResult

FORMULAS = ("ochiai", "tarantula")


# ---------------------------------------------------------------------- scoring (pure functions)
def _suspiciousness(ef: int, ep: int, nf: int, np_: int, formula: str) -> float:
    """Ochiai ef/sqrt(nf*(ef+ep)) or Tarantula (ef/nf)/(ef/nf+ep/np); callers guarantee nf, np > 0."""
    if formula == "tarantula":
        fail, passed = ef / nf, ep / np_
        return fail / (fail + passed) if fail + passed else 0.0
    return ef / math.sqrt(nf * (ef + ep)) if ef else 0.0


def _symbol_match(function: str | None, symbols: list[str]) -> bool:
    """True if a function name matches a hinted symbol ('Inventory.remove' or just 'remove')."""
    if not function:
        return False
    last = function.split(".")[-1]
    return any(function == s or last == str(s).split(".")[-1] for s in symbols)


def rank(items: list[dict], hints: dict | None = None) -> list[dict]:
    """Deterministic order: score desc, ef desc, ep asc, hinted symbol, hinted file, path, line."""
    symbols = list((hints or {}).get("symbols") or [])
    files = set((hints or {}).get("files") or [])
    return sorted(items, key=lambda d: (-d["score"], -d["ef"], d["ep"],
                                        0 if _symbol_match(d.get("function"), symbols) else 1,
                                        0 if d["path"] in files else 1, d["path"], d["line"]))


def score_lines(runs: list[dict], formula: str = "ochiai", hints: dict | None = None) -> list[dict]:
    """Score every (path, line) executed by at least one failing run. [] without failing or passing runs."""
    failed = [r for r in runs if r.get("outcome") == "failed"]
    passed = [r for r in runs if r.get("outcome") == "passed"]
    nf, np_ = len(failed), len(passed)
    if nf == 0 or np_ == 0:
        return []  # no failure to explain, or no contrast (every line would tie)
    ef: Counter = Counter()
    ep: Counter = Counter()
    for runs_of, counter in ((failed, ef), (passed, ep)):
        for run in runs_of:
            for path, lines in (run.get("lines") or {}).items():
                for line in set(lines):
                    counter[(path, int(line))] += 1
    scored = [{"path": path, "line": line, "score": round(_suspiciousness(ef[key], ep[key], nf, np_, formula), 4),
               "ef": ef[key], "ep": ep[key]} for key in ef for path, line in [key]]
    return rank(scored, hints)


def enclosing_functions(path: str, text: str) -> list[tuple[str, int, int]]:
    """(qualified name, def line, end line) for every function/method, nested ones as 'Outer.inner'."""
    try:
        tree = ast.parse(text)
    except (SyntaxError, ValueError):
        return []
    found: list[tuple[str, int, int]] = []

    def visit(node: ast.AST, prefix: str) -> None:
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                name = prefix + child.name
                found.append((name, child.lineno, getattr(child, "end_lineno", child.lineno) or child.lineno))
                visit(child, name + ".")
            elif isinstance(child, ast.ClassDef):
                visit(child, prefix + child.name + ".")
            else:
                visit(child, prefix)

    visit(tree, "")
    return found


def attach_functions(scored: list[dict], ws: Any, hints: dict | None = None) -> list[dict]:
    """Map lines to their innermost function; drop module-level lines, test files and scratch files."""
    functions: dict[str, list[tuple[str, int, int]]] = {}
    attached: list[dict] = []
    for item in scored:
        path = item["path"]
        if path.startswith("@scratch") or ws.is_test_file(path):
            continue
        if path not in functions:
            try:
                functions[path] = enclosing_functions(path, ws.read_text(path))
            except (ValueError, OSError):
                functions[path] = []
        best = None
        for name, start, end in functions[path]:
            # The def line itself runs at import time in every run, so it carries no signal.
            if start < item["line"] <= end and (best is None or end - start < best[2] - best[1]):
                best = (name, start, end)
        if best is not None:
            attached.append({**item, "function": best[0], "start": best[1], "end": best[2]})
    return rank(attached, hints)


def aggregate_functions(lines: list[dict], top_k: int | None = None) -> list[dict]:
    """Function score = max of its line scores (lines must already be ranked); top_line = that line."""
    best: dict[tuple[str, str], dict] = {}
    for item in lines:
        key = (item["path"], item["function"])
        if key not in best:
            best[key] = {"path": item["path"], "name": item["function"], "start": item["start"],
                         "end": item["end"], "score": item["score"], "top_line": item["line"],
                         "ef": item["ef"], "ep": item["ep"]}
    ranked = list(best.values())
    return ranked[:top_k] if top_k else ranked


# ---------------------------------------------------------------------- collection

TRACE_RUNNER = Path(__file__).with_name("_trace_runner.py")
_SHELL_OPERATOR = re.compile(r"\s*(?:;|&&|\|\||\|)\s*")
_PYTHON = re.compile(r"(?:.*/)?python3?(?:\.\d+)?")


def parse_repro_command(command: str, scratch_dir: Any) -> tuple[str, list[str]] | None:
    """('script', [path, *args]) or ('pytest', [pytest args]) for the first shell segment, else None."""
    text = str(command or "").replace("@scratch/", str(scratch_dir).rstrip("/") + "/")
    first = _SHELL_OPERATOR.split(text, maxsplit=1)[0].strip()
    try:
        tokens = shlex.split(first)
    except ValueError:
        return None
    for i, token in enumerate(tokens):
        if token == "pytest" or token.endswith("/pytest"):
            return "pytest", tokens[i + 1:]
    if len(tokens) >= 2 and _PYTHON.fullmatch(tokens[0]) and tokens[1].endswith(".py"):
        return "script", tokens[1:]
    return None


class SpectrumAnalyzer:
    """Runs the reproduction and related passing tests under the tracer and ranks suspicious code."""

    def __init__(self, ws: Any, cfg: Any, test_runner: Any, ui: Any = None,
                 trajectory: Trajectory | None = None) -> None:
        self.ws, self.cfg, self.runner = ws, cfg, test_runner
        self.scfg = cfg.spectrum
        self.ui = ui if ui is not None else NullUI()
        self.trajectory = trajectory if trajectory is not None else Trajectory(None)

    def analyze(self, state: Any) -> SpectrumResult:
        """Never raises: any failure becomes ok=False with a reason."""
        start = time.monotonic()
        try:
            result = self._analyze(state, start)
        except Exception as e:  # noqa: BLE001 - the Tracer is purely additive
            result = SpectrumResult(ok=False, reason=f"crashed: {type(e).__name__}: {e}")
        result.formula = self.scfg.formula
        result.seconds = round(time.monotonic() - start, 3)
        self.trajectory.log("spectrum_result", ok=result.ok, reason=result.reason, seconds=result.seconds,
                            top_lines=result.lines[:5])
        return result

    # ------------------------------------------------------------------ helpers
    def _trace(self, py: str, mode: str, args: list[str], name: str, timeout: float) -> tuple[dict | None, Any]:
        out = self.ws.scratch_dir / ".spectrum" / f"{name}.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        partial = Path(str(out) + ".partial.jsonl")
        for stale in (out, partial):
            if stale.exists():
                stale.unlink()
        cmd = " ".join(shlex.quote(p) for p in [py, str(TRACE_RUNNER), "--root", str(self.ws.repo_root),
                                                "--out", str(out), "--", mode, *args])
        res = run_process(cmd, self.ws.repo_root, max(1.0, timeout), env_extra={"PYTHONPATH": self.ws.pythonpath()})
        data = None
        if out.exists():
            data = json.loads(out.read_text(encoding="utf-8"))
        elif partial.exists():  # killed on timeout: keep the tests that finished
            runs = [json.loads(line) for line in partial.read_text(encoding="utf-8").splitlines() if line.strip()]
            data = {"runs": runs, "error": None, "exit_code": None, "partial": True}
        self.trajectory.log("spectrum_run", mode=mode, exit_code=res.exit_code, seconds=round(res.duration_s, 2),
                            timed_out=res.timed_out, runs=len((data or {}).get("runs") or []))
        return data, res

    def _passing(self, py: str, targets: list[str], exclude: set, deadline: float, name: str) -> tuple[list, bool]:
        """Passed runs (minus pre-existing failures) from one traced pytest process over targets."""
        if not targets or deadline - time.monotonic() < 1:
            return [], False
        data, res = self._trace(py, "pytest", targets, name, deadline - time.monotonic())
        runs = [r for r in (data or {}).get("runs") or [] if r.get("outcome") == "passed" and r.get("id") not in exclude]
        return runs, bool(res.timed_out)

    def _top_up_targets(self, state: Any, already: list[str]) -> list[str]:
        """Test files next to the localized code or the targeted tests; else the whole test suite."""
        dirs = {str(Path(p).parent) for p in list(state.localization.get("files") or []) + list(already)}
        tests = [self.ws.rel(p) for p in self.ws.iter_source_files()
                 if p.suffix == ".py" and self.ws.is_test_file(self.ws.rel(p)) and self.ws.rel(p) not in already]
        near = [t for t in tests if str(Path(t).parent) in dirs]
        return near or tests

    # ------------------------------------------------------------------ main
    def _analyze(self, state: Any, start: float) -> SpectrumResult:
        scfg = self.scfg
        if not scfg.enabled:
            return SpectrumResult(ok=False, reason="disabled")
        kind = self.runner.detect().get("kind")
        if kind not in ("pytest", "unittest") and not any(p.suffix == ".py" for p in self.ws.iter_source_files()):
            return SpectrumResult(ok=False, reason="not a Python repo")
        if not state.repro.get("reproduced"):
            return SpectrumResult(ok=False, reason="no reproduction")
        parsed = parse_repro_command(state.repro.get("command", ""), self.ws.scratch_dir)
        if parsed is None:
            return SpectrumResult(ok=False, reason="unsupported repro command")
        if self.ws.edited_files():
            return SpectrumResult(ok=False, reason="repo already modified")
        py = self.runner.python()
        deadline = start + float(scfg.timeout_s)

        data, res = self._trace(py, parsed[0], parsed[1], "repro",
                                min(float(scfg.repro_timeout_s), deadline - time.monotonic()))
        if res.timed_out:
            return SpectrumResult(ok=False, reason="repro timed out under tracer")
        if not data or data.get("error"):
            return SpectrumResult(ok=False, reason=f"tracer error: {(data or {}).get('error') or 'no output'}")
        failing = [r for r in data.get("runs") or [] if r.get("outcome") == "failed"]
        if not failing or data.get("exit_code") == 0:
            return SpectrumResult(ok=False, reason="repro did not fail under tracer")

        exclude = set()
        for baseline in (state.baseline_full, state.baseline_targeted):
            exclude |= set(getattr(baseline, "failing_ids", None) or [])
        targets = list(state.targeted_tests)
        passing, timed_out = self._passing(py, targets, exclude, deadline, "passing")
        if len(passing) < scfg.min_passing_runs:
            extra = self._top_up_targets(state, targets)
            more, more_timed_out = self._passing(py, extra, exclude, deadline, "passing_topup")
            passing, timed_out = passing + more, timed_out or more_timed_out
        passing = passing[:int(scfg.max_passing_tests)]
        if len(passing) < scfg.min_passing_runs:
            return SpectrumResult(ok=False, reason=f"only {len(passing)} passing runs; no contrast",
                                  failing_runs=len(failing), passing_runs=len(passing))

        hints = state.localization or {}
        lines = attach_functions(score_lines(failing + passing, scfg.formula, hints), self.ws, hints)
        if not lines:
            return SpectrumResult(ok=False, reason="no source lines covered", failing_runs=len(failing),
                                  passing_runs=len(passing))
        functions = aggregate_functions(lines)
        top = [f["score"] for f in functions[:5]]
        low = len(passing) < 3 or timed_out or (len(top) >= 2 and max(top) - min(top) < 0.05)
        return SpectrumResult(ok=True, lines=lines[:max(int(scfg.top_lines) * 3, 30)], functions=functions[:200],
                              failing_runs=len(failing), passing_runs=len(passing), low_confidence=low)


# ---------------------------------------------------------------------- evidence for prompts and reports
_HUNK = re.compile(r"^@@ -(\d+)(?:,\d+)? \+\d+(?:,\d+)? @@")
CONFIDENT_CLOSING = ("Treat this as strong evidence, not proof: the defect may be a MISSING line (a guard that was "
                     "never\nwritten) or a caller of this code. Confirm with view_file before editing.")
LOW_CONFIDENCE_CLOSING = "Scores are close together, so this narrows the search only slightly — rely on your own analysis."


def as_result(result: Any) -> SpectrumResult:
    """Accept a SpectrumResult or its dict form (state.spectrum)."""
    if isinstance(result, SpectrumResult):
        return result
    data = dict(result or {})
    known = SpectrumResult.__dataclass_fields__
    return SpectrumResult(**{k: v for k, v in data.items() if k in known}) if "ok" in data else SpectrumResult(ok=False)


def changed_lines(diff: str) -> dict[str, set[int]]:
    """Original-file line numbers each patch touches (removed lines, and the line above each insertion)."""
    touched: dict[str, set[int]] = {}
    path, old_line = "", 0
    for line in diff.splitlines():
        if line.startswith("--- "):
            path = line[4:].strip()
            path = path[2:] if path.startswith("a/") else path
        elif line.startswith("+++ "):
            new = line[4:].strip()
            if path == "/dev/null":
                path = new[2:] if new.startswith("b/") else new
        elif (m := _HUNK.match(line)):
            old_line = int(m.group(1))
        elif line.startswith("-"):
            touched.setdefault(path, set()).add(old_line)
            old_line += 1
        elif line.startswith("+"):
            touched.setdefault(path, set()).add(max(old_line - 1, 1))
        elif line.startswith(" "):
            old_line += 1
    return touched


def touched_ranks(diff: str, functions: list[dict]) -> list[int]:
    """1-based ranks of the ranked functions whose original lines the patch changes."""
    touched = changed_lines(diff)
    return [i for i, f in enumerate(functions, 1)
            if any(f["start"] <= n <= f["end"] for n in touched.get(f["path"], ()))]


def format_suspicious(result: Any, ws: Any, top_lines: int = 10, max_chars: int = 5000) -> str:
    """The EXECUTION EVIDENCE block for the FIX prompt ('' when the Tracer has no result)."""
    r = as_result(result)
    if not r.ok or not r.lines:
        return ""
    formula = "Tarantula" if r.formula == "tarantula" else "Ochiai"
    runs = "run" if r.failing_runs == 1 else "runs"
    out = [f"EXECUTION EVIDENCE (spectrum-based fault localization, {formula}; {r.failing_runs} failing {runs}, "
           f"{r.passing_runs} passing runs):",
           "Lines executed by the failing reproduction but rarely by passing tests are more suspicious."]
    rank_of = {(f["path"], f["name"]): i for i, f in enumerate(r.functions)}
    groups: dict[tuple[str, str], list[dict]] = {}
    for item in r.lines[:top_lines]:
        groups.setdefault((item["path"], item["function"]), []).append(item)
    sources: dict[str, list[str]] = {}
    for n, key in enumerate(sorted(groups, key=lambda k: rank_of.get(k, len(rank_of))), 1):
        path, name = key
        func = r.functions[rank_of[key]] if key in rank_of else groups[key][0]
        out.append(f"#{n} {path}  {name} (L{func['start']}-L{func['end']})  score {func['score']:.2f}")
        if path not in sources:
            try:
                sources[path] = ws.read_text(path).splitlines()
            except (ValueError, OSError):
                sources[path] = []
        source, marked = sources[path], {i["line"]: i for i in groups[key]}
        shown = sorted({n for ln in marked for n in (ln - 1, ln, ln + 1)
                        if func["start"] <= n <= func["end"] and 1 <= n <= len(source)})
        previous = None
        for ln in shown:
            if previous is not None and ln != previous + 1:
                out.append("         ...")
            item = marked.get(ln)
            note = (f"        [score {item['score']:.2f}, fail {item['ef']}/{r.failing_runs}, "
                    f"pass {item['ep']}/{r.passing_runs}]") if item else ""
            out.append(f"   {'>>' if item else '  '} {ln:>3} | {source[ln - 1]}{note}")
            previous = ln
    out.append(LOW_CONFIDENCE_CLOSING if r.low_confidence else CONFIDENT_CLOSING)
    return truncate("\n".join(out), max_chars, head_chars=max_chars - 400)


def exam_score(spectrum: dict, diff: str) -> float | None:
    """EXAM (lower is better): rank of the first ranked function the patch touches / number ranked.

    None when the Tracer did not produce a ranking or there is no patch; 1.0 when the patch touches no
    ranked function (the fault was never ranked). Rankings are capped at 200 functions in state.spectrum.
    """
    functions = (spectrum or {}).get("functions") or []
    if not (spectrum or {}).get("ok") or not functions or not diff:
        return None
    ranks = touched_ranks(diff, functions)
    return round(min(ranks) / len(functions), 4) if ranks else 1.0

"""Benchmark: run the harness on every bundled issue, grade it with the hidden tests, and summarise.

Usage (via `make eval EVAL_ARGS="..."` or directly):
  .venv/bin/python scripts/eval.py                 # all issues, Tracer as configured
  .venv/bin/python scripts/eval.py --quick         # first 2 issues
  .venv/bin/python scripts/eval.py --issues 01,04  # a subset
  .venv/bin/python scripts/eval.py --no-spectrum   # Tracer disabled (HARNESS_SPECTRUM=0)
  .venv/bin/python scripts/eval.py --ablation      # every issue with AND without the Tracer
  .venv/bin/python scripts/eval.py --model gemini    # a preset from config.yaml, or any LiteLLM model

Writes runs/eval_summary.md (with a with/without comparison table after --ablation) and
runs/eval-<timestamp>.json. A solved issue means the grader's hidden test passes on the harness's result.
"""
from __future__ import annotations

import argparse
import json
import os
import shlex
import shutil
import sys
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from harness.config import load_config  # noqa: E402
from harness.providers import KEY_ENV_VARS  # noqa: E402
from harness.shell import run_process  # noqa: E402
from harness.spectrum import exam_score  # noqa: E402
from harness.types import HarnessError  # noqa: E402

FIXTURES = ROOT / "fixtures"
HIDDEN_TIMEOUT_S = 120


@dataclass
class IssueResult:
    """One harness run on one issue, graded by its hidden test."""

    issue: str
    mode: str                 # "tracer" or "no-tracer"
    solved: bool
    status: str
    tokens: int = 0
    llm_calls: int = 0
    tool_calls: int = 0
    seconds: float = 0.0
    exam: float | None = None
    run_dir: str = ""
    note: str = ""


def list_issues(selected: list[str] | None = None, quick: bool = False) -> list[Path]:
    """Issue files, optionally filtered by numeric prefix ('01', '4') or limited to the first two."""
    issues = sorted((FIXTURES / "issues").glob("*.md"))
    if selected:
        wanted = {s.strip().zfill(2) for s in selected if s.strip()}
        issues = [p for p in issues if p.name[:2] in wanted]
    return issues[:2] if quick else issues


def hidden_test_for(issue: Path) -> Path:
    """fixtures/issues/04_inventory.md -> fixtures/hidden_tests/test_issue_04.py"""
    return FIXTURES / "hidden_tests" / f"test_issue_{issue.name[:2]}.py"


def run_hidden_test(repo: Path, hidden: Path) -> bool:
    """Copy the grader's test into the repo and run only that file."""
    dest = repo / "tests" / hidden.name
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(hidden, dest)
    cmd = f"{shlex.quote(sys.executable)} -m pytest -q -p no:cacheprovider {shlex.quote('tests/' + hidden.name)}"
    return run_process(cmd, repo, HIDDEN_TIMEOUT_S).exit_code == 0


def read_run(run_dir: Path | None) -> dict:
    """status, metrics and EXAM from a run directory written by the harness."""
    if run_dir is None or not (run_dir / "state.json").exists():
        return {}
    state = json.loads((run_dir / "state.json").read_text(encoding="utf-8"))
    total = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))["total"]
    patch = (run_dir / "patch.diff").read_text(encoding="utf-8") if (run_dir / "patch.diff").exists() else ""
    return {"status": state.get("status", "?"), "tokens": total.get("total_tokens", 0),
            "llm_calls": total.get("llm_calls", 0), "tool_calls": total.get("tool_calls", 0),
            "seconds": total.get("seconds", 0.0), "exam": exam_score(state.get("spectrum") or {}, patch)}


def run_issue(issue: Path, mode: str, runs_dir: Path, args: argparse.Namespace, timeout_s: float) -> IssueResult:
    """Fresh copy of the sample repo -> harness -> hidden test."""
    work = Path(tempfile.mkdtemp(prefix=f"eval-{issue.name[:2]}-")) / "sample_repo"
    shutil.copytree(FIXTURES / "sample_repo", work)
    parts = [sys.executable, "-m", "harness", "--repo", str(work), "--issue-file", str(issue),
             "--non-interactive", "--quiet", "--strict-exit"]
    if args.config:
        parts += ["--config", args.config]
    if args.model:
        parts += ["--model", args.model]
    # run_process removes model keys from child processes; the harness itself needs them back.
    env = {name: os.environ[name] for name in KEY_ENV_VARS if os.environ.get(name)}
    env["HARNESS_SPECTRUM"] = "0" if mode == "no-tracer" else "1"
    before = set(runs_dir.iterdir()) if runs_dir.exists() else set()
    started = time.monotonic()
    res = run_process(" ".join(shlex.quote(p) for p in parts), ROOT, timeout_s, env_extra=env)
    new_dirs = sorted((set(runs_dir.iterdir()) if runs_dir.exists() else set()) - before, key=lambda p: p.name)
    run_dir = new_dirs[-1] if new_dirs else None
    info = read_run(run_dir)
    solved = run_hidden_test(work, hidden_test_for(issue))
    note = "harness timed out" if res.timed_out else ("" if info else res.output.strip()[-300:])
    result = IssueResult(issue=issue.stem, mode=mode, solved=solved, status=info.get("status", "no run"),
                         tokens=info.get("tokens", 0), llm_calls=info.get("llm_calls", 0),
                         tool_calls=info.get("tool_calls", 0),
                         seconds=round(info.get("seconds") or (time.monotonic() - started), 1),
                         exam=info.get("exam"), run_dir=str(run_dir or ""), note=note)
    shutil.rmtree(work.parent, ignore_errors=True)
    return result


def summarise(results: list[IssueResult]) -> dict:
    """Solve rate and averages for one mode."""
    n = len(results) or 1
    exams = [r.exam for r in results if r.exam is not None]
    return {"issues": len(results), "solved": sum(r.solved for r in results),
            "solve_rate": round(sum(r.solved for r in results) / n, 3),
            "avg_tokens": round(sum(r.tokens for r in results) / n), "avg_llm_calls": round(sum(r.llm_calls for r in results) / n, 1),
            "avg_tool_calls": round(sum(r.tool_calls for r in results) / n, 1),
            "avg_seconds": round(sum(r.seconds for r in results) / n, 1),
            "mean_exam": round(sum(exams) / len(exams), 3) if exams else None}


def _fmt(value: object) -> str:
    return "-" if value is None else str(value)


def markdown(results: list[IssueResult], model: str) -> str:
    """runs/eval_summary.md: per-issue table, totals, and a comparison when both modes ran."""
    lines = ["# Evaluation summary", "", f"Model: `{model}`", "",
             "| Issue | Mode | Solved | Status | Tokens | LLM calls | Tool calls | Seconds | EXAM |",
             "| --- | --- | :---: | --- | ---: | ---: | ---: | ---: | ---: |"]
    for r in results:
        lines.append(f"| {r.issue} | {r.mode} | {'✅' if r.solved else '❌'} | {r.status} | {r.tokens} | {r.llm_calls} | "
                     f"{r.tool_calls} | {r.seconds} | {_fmt(r.exam)} |")
    modes = sorted({r.mode for r in results}, key=lambda m: m != "tracer")
    lines += ["", "| Mode | Solved | Solve rate | Avg tokens | Avg LLM calls | Avg tool calls | Avg seconds | Mean EXAM |",
              "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for mode in modes:
        s = summarise([r for r in results if r.mode == mode])
        lines.append(f"| {mode} | {s['solved']}/{s['issues']} | {s['solve_rate']:.0%} | {s['avg_tokens']} | "
                     f"{s['avg_llm_calls']} | {s['avg_tool_calls']} | {s['avg_seconds']} | {_fmt(s['mean_exam'])} |")
    if len(modes) == 2:
        lines += ["", "Comparison above: with vs without the Tracer (EXAM = rank of the first patched function / "
                      "ranked functions; lower is better)."]
    return "\n".join(lines) + "\n"


def print_table(results: list[IssueResult]) -> None:
    """Rich table on the terminal."""
    from rich.console import Console
    from rich.table import Table

    table = Table(title="Evaluation")
    for column in ("Issue", "Mode", "Solved", "Status", "Tokens", "Calls", "Seconds", "EXAM"):
        table.add_column(column)
    for r in results:
        table.add_row(r.issue, r.mode, "✅" if r.solved else "❌", r.status, str(r.tokens), str(r.llm_calls),
                      str(r.seconds), _fmt(r.exam))
    Console().print(table)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--quick", action="store_true", help="only the first two issues")
    parser.add_argument("--issues", help="comma-separated issue numbers, e.g. 01,04")
    parser.add_argument("--no-spectrum", action="store_true", help="disable the Tracer (HARNESS_SPECTRUM=0)")
    parser.add_argument("--ablation", action="store_true", help="run every issue with and without the Tracer")
    parser.add_argument("--config", help="config.yaml to pass to the harness")
    parser.add_argument("--model", help="model override to pass to the harness")
    args = parser.parse_args(argv)
    try:
        cfg = load_config(args.config, model=args.model)
        cfg.api_key()
    except HarnessError as e:
        print(f"Error: {e}")
        return 1
    runs_dir = Path(cfg.output.runs_dir)
    runs_dir = runs_dir if runs_dir.is_absolute() else ROOT / runs_dir
    runs_dir.mkdir(parents=True, exist_ok=True)
    modes = ["tracer", "no-tracer"] if args.ablation else (["no-tracer"] if args.no_spectrum else ["tracer"])
    issues = list_issues(args.issues.split(",") if args.issues else None, args.quick)
    if not issues:
        print("No issues selected.")
        return 1
    timeout_s = float(cfg.budgets.max_wall_clock_s) + 180
    results: list[IssueResult] = []
    for mode in modes:
        for issue in issues:
            print(f"[{mode}] {issue.stem} ...", flush=True)
            result = run_issue(issue, mode, runs_dir, args, timeout_s)
            print(f"    {'SOLVED' if result.solved else 'not solved'} ({result.status}, {result.tokens} tokens, "
                  f"{result.seconds}s){' - ' + result.note if result.note else ''}", flush=True)
            results.append(result)
    print_table(results)
    model = cfg.model.name
    (runs_dir / "eval_summary.md").write_text(markdown(results, model), encoding="utf-8")
    stamp = time.strftime("%Y%m%d-%H%M%S")
    (runs_dir / f"eval-{stamp}.json").write_text(json.dumps(
        {"model": model, "modes": modes, "results": [asdict(r) for r in results],
         "summary": {m: summarise([r for r in results if r.mode == m]) for m in modes}}, indent=2), encoding="utf-8")
    print(f"Summary: {runs_dir / 'eval_summary.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

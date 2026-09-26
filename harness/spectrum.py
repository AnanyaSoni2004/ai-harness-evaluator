"""Tracer: spectrum-based fault localization (scoring, aggregation, collection, prompt rendering)."""
from __future__ import annotations

import ast
import math
from collections import Counter
from typing import Any

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

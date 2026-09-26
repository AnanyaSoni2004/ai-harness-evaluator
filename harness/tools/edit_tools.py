"""Edit tools: str_replace, create_file, check_syntax."""
from __future__ import annotations

import ast
import difflib
import json
import re
import shutil
import tempfile
from pathlib import Path
from typing import Any

import yaml

from harness.shell import run_process
from harness.tools.read_tools import never_raises
from harness.types import ToolResult
from harness.workspace import Workspace

CONTEXT_LINES = 4
SIMILAR_REGIONS = 3
MIN_SIMILARITY = 0.3
NODE_CHECK_TIMEOUT_S = 20.0
_LINE_PREFIX = re.compile(r"^\s*\d+\s*\|")


def _numbered(lines: list[str], start: int, end: int) -> str:
    """Lines start..end (1-based, inclusive, clamped) formatted like view_file."""
    start, end = max(1, start), min(len(lines), end)
    return "\n".join(f"{n:>5} | {lines[n - 1]}" for n in range(start, end + 1))


def _error_context(content: str, lineno: int | None, radius: int = 3) -> str:
    """Numbered lines around a syntax error position."""
    if not lineno:
        return ""
    return "\n" + _numbered(content.splitlines(), lineno - radius, lineno + radius)


def _node_check(path: Path, content: str) -> tuple[bool, str]:
    """Syntax-check JavaScript with `node --check` (retrying as ES module); OK if node is absent."""
    if not shutil.which("node"):
        return True, ""
    with tempfile.TemporaryDirectory() as tmp:
        result = None
        for suffix in (path.suffix, ".mjs"):
            probe = Path(tmp) / f"check{suffix}"
            probe.write_text(content, encoding="utf-8")
            result = run_process(f"node --check {probe.name}", Path(tmp), NODE_CHECK_TIMEOUT_S)
            if result.exit_code == 0 or result.timed_out:
                return True, ""
            if suffix == ".mjs" or not re.search(r"import|export|module", result.output):
                break
    output = result.output.replace(str(Path(tmp)), "").strip() if result else ""
    m = re.search(r"check\.\w+:(\d+)", output)
    lineno = int(m.group(1)) if m else None
    lines = [l for l in output.splitlines() if l.strip()]
    msg = next((l for l in lines if "Error" in l), lines[-1] if lines else "invalid JavaScript")
    if not msg.startswith("SyntaxError"):
        msg = f"SyntaxError: {msg}"
    return False, msg + (f" (line {lineno})" if lineno else "") + _error_context(content, lineno)


def check_syntax(path: Path, content: str) -> tuple[bool, str]:
    """Parse content by file type (.py, .json, .yml/.yaml, .js/.mjs/.cjs); other types are always OK."""
    suffix = Path(path).suffix.lower()
    try:
        if suffix == ".py":
            ast.parse(content)
        elif suffix == ".json":
            json.loads(content)
        elif suffix in (".yml", ".yaml"):
            for _ in yaml.safe_load_all(content):
                pass
        elif suffix in (".js", ".mjs", ".cjs"):
            return _node_check(Path(path), content)
    except SyntaxError as e:
        return False, f"SyntaxError: {e.msg} (line {e.lineno})" + _error_context(content, e.lineno)
    except json.JSONDecodeError as e:
        return False, f"JSON error: {e.msg} (line {e.lineno})" + _error_context(content, e.lineno)
    except yaml.YAMLError as e:
        mark = getattr(e, "problem_mark", None)
        lineno = mark.line + 1 if mark is not None else None
        problem = getattr(e, "problem", None) or str(e)
        return False, f"YAML error: {problem}" + (f" (line {lineno})" if lineno else "") + _error_context(content, lineno)
    return True, ""


def _scope_error(ws: Workspace, target: Path) -> str | None:
    """Why the current phase may not write this path, or None if it may."""
    if ws.write_scope == "repo":
        return None
    if ws.write_scope == "scratch":
        if ws.is_scratch(target):
            return None
        return "In this phase you may only write under @scratch/ (e.g. @scratch/repro.py); repository files are read-only."
    return "Editing files is not allowed in this phase."


def _similar_regions(content: str, old_str: str) -> str:
    """The 3 regions of the file most similar to old_str (same line count), with line numbers."""
    lines = content.splitlines()
    k = max(1, len(old_str.splitlines()))
    if not lines:
        return ""
    target = "\n".join(line.strip() for line in old_str.splitlines())
    scored = []
    for i in range(0, max(1, len(lines) - k + 1)):
        window = "\n".join(line.strip() for line in lines[i:i + k])
        matcher = difflib.SequenceMatcher(None, target, window, autojunk=False)
        scored.append((matcher.quick_ratio(), i, matcher))
    scored.sort(key=lambda t: -t[0])
    refined = sorted(((m.ratio(), i) for _, i, m in scored[:25]), key=lambda t: (-t[0], t[1]))
    picked: list[tuple[float, int]] = []
    for ratio, i in refined:
        if picked and ratio < MIN_SIMILARITY:
            break  # weak matches only cost tokens; always keep the best one
        if all(abs(i - j) >= k for _, j in picked):
            picked.append((ratio, i))
        if len(picked) == SIMILAR_REGIONS:
            break
    blocks = [f"--- similar region (lines {i + 1}-{i + k}, {ratio:.0%} similar):\n{_numbered(lines, i + 1, i + k)}"
              for ratio, i in picked]
    return "\n".join(blocks)


def _line_of(content: str, index: int) -> int:
    """1-based line number of a character offset."""
    return content.count("\n", 0, index) + 1


@never_raises
def str_replace(ws: Workspace, cfg: Any, path: str = "", old_str: str = "", new_str: str = "") -> ToolResult:
    """Replace one exact, unique occurrence of old_str with new_str, with safety checks."""
    if not str(path or "").strip():
        return ToolResult(False, "path is required.")
    target = ws.resolve(path)
    scope_err = _scope_error(ws, target)
    if scope_err:
        return ToolResult(False, scope_err)
    if not target.is_file():
        return ToolResult(False, f"File not found: {path}; use create_file to create new files.")
    old_str, new_str = str(old_str or ""), str(new_str if new_str is not None else "")
    if not old_str:
        return ToolResult(False, "old_str must not be empty. Copy the exact lines to replace from view_file.")
    if old_str == new_str:
        return ToolResult(False, "old_str and new_str are identical; nothing to change.")
    rel = ws.rel(target)
    content = ws.read_text(target)

    count = content.count(old_str)
    if count == 0:
        hint = ""
        if all(_LINE_PREFIX.match(l) for l in old_str.splitlines() if l.strip()):
            hint = "\nIt looks like old_str contains view_file line-number prefixes ('  12 | '); remove them."
        regions = _similar_regions(content, old_str)
        return ToolResult(False, f"old_str was not found in {rel}.{hint}\n{regions}\n"
                                 "old_str must match exactly, including indentation; do not include the "
                                 "line-number prefixes from view_file.")
    if count > 1:
        starts, pos = [], content.find(old_str)
        while pos != -1:
            starts.append(_line_of(content, pos))
            pos = content.find(old_str, pos + 1)
        return ToolResult(False, f"old_str matches {count} times in {rel} (lines {', '.join(map(str, starts))}). "
                                 "Include more surrounding lines to make old_str unique.")

    allow_tests = bool(getattr(getattr(cfg, "safety", None), "allow_edit_existing_tests", False))
    if (not ws.is_scratch(target) and ws.is_test_file(rel) and ws.existed_at_start(target)
            and not allow_tests and old_str not in new_str):
        return ToolResult(False, "Modifying existing test code is not allowed; fix the source. "
                                 "(Adding new tests is fine: keep old_str inside new_str.)")

    index = content.index(old_str)
    updated = content[:index] + new_str + content[index + len(old_str):]
    ok_before, _ = check_syntax(target, content)
    ok_after, err = check_syntax(target, updated)
    if ok_before and not ok_after:
        return ToolResult(False, f"Edit NOT applied: it would break {rel}.\n{err}\nFix the edit and try again.")

    ws.write_text(target, updated)
    first = _line_of(updated, index)
    last = first + new_str.count("\n")
    snippet = _numbered(updated.splitlines(), first - CONTEXT_LINES, last + CONTEXT_LINES)
    return ToolResult(True, f"Edit applied to {rel}.\n{snippet}", {"path": rel, "line": first})


@never_raises
def create_file(ws: Workspace, cfg: Any, path: str = "", content: str = "") -> ToolResult:
    """Create a new file (repo files must not exist yet; @scratch/ files may be overwritten)."""
    if not str(path or "").strip():
        return ToolResult(False, "path is required.")
    target = ws.resolve(path)
    scope_err = _scope_error(ws, target)
    if scope_err:
        return ToolResult(False, scope_err)
    rel = ws.rel(target)
    if target.is_dir():
        return ToolResult(False, f"{rel} is a directory.")
    if target.exists() and not ws.is_scratch(target):
        return ToolResult(False, f"{rel} exists; use str_replace to edit it.")
    content = str(content if content is not None else "")
    ok, err = check_syntax(target, content)
    if not ok:
        return ToolResult(False, f"File NOT created: the content is invalid.\n{err}")
    ws.write_text(target, content)
    n_lines = len(content.splitlines())
    return ToolResult(True, f"Created {rel} ({n_lines} lines).", {"path": rel, "lines": n_lines})

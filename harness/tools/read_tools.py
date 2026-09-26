"""Read-only tools: list_dir, find_files, search_code, view_file."""
from __future__ import annotations

import fnmatch
import functools
import os
import re
import shlex
import shutil
from pathlib import Path
from typing import Any, Callable

from harness.shell import run_process
from harness.types import ToolResult
from harness.workspace import IGNORED_DIRS, Workspace

MAX_LIST_ENTRIES = 300
MAX_LIST_DEPTH = 4
MAX_FIND_RESULTS = 100
MAX_SEARCH_MATCHES = 50
MAX_MATCH_TEXT = 200
MAX_VIEW_LINES = 250
MAX_LINE_CHARS = 400
SEARCH_TIMEOUT_S = 30.0

_RG_LINE = re.compile(r"^(.*?):(\d+):(.*)$")


def never_raises(fn: Callable[..., ToolResult]) -> Callable[..., ToolResult]:
    """Turn ValueError into a plain error result and any other exception into a crash report."""
    @functools.wraps(fn)
    def wrapper(*args: Any, **kwargs: Any) -> ToolResult:
        try:
            return fn(*args, **kwargs)
        except ValueError as e:
            return ToolResult(False, str(e))
        except Exception as e:  # noqa: BLE001 - tools must never raise into the agent loop
            return ToolResult(False, f"{fn.__name__} failed: {type(e).__name__}: {e}")
    return wrapper


def as_int(value: Any, name: str, default: int | None) -> int | None:
    """Coerce a model-supplied argument to int (models often send "10"); None/"" gives the default."""
    if value is None or value == "":
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        raise ValueError(f"{name} must be an integer, got {value!r}.") from None


def _skip_dir(name: str) -> bool:
    """True for vendor/build/cache directories the tools never descend into."""
    return name in IGNORED_DIRS or name.endswith(".egg-info")


# ---------------------------------------------------------------------- list_dir
@never_raises
def list_dir(ws: Workspace, cfg: Any, path: str = ".", depth: Any = 2) -> ToolResult:
    """Indented directory tree (directories end with '/'), depth <= 4, at most 300 entries."""
    root = ws.resolve(path)
    if not root.exists():
        return ToolResult(False, f"Directory not found: {path}")
    if not root.is_dir():
        return ToolResult(False, f"{ws.rel(root)} is a file; use view_file to read it.")
    max_depth = max(1, min(as_int(depth, "depth", 2), MAX_LIST_DEPTH))
    lines = [ws.rel(root).rstrip("/") + "/" if ws.rel(root) != "." else "./"]
    count = 0
    truncated = False

    def walk(directory: Path, level: int) -> None:
        nonlocal count, truncated
        try:
            entries = sorted(directory.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower()))
        except OSError:
            return
        for entry in entries:
            if truncated:
                return
            is_dir = entry.is_dir() and not entry.is_symlink()
            if is_dir and _skip_dir(entry.name):
                continue
            if count >= MAX_LIST_ENTRIES:
                truncated = True
                return
            count += 1
            lines.append("  " * level + entry.name + ("/" if is_dir else ""))
            if is_dir and level < max_depth:
                walk(entry, level + 1)

    walk(root, 1)
    if count == 0:
        lines.append("  (empty)")
    if truncated:
        lines.append(f"... [stopped at {MAX_LIST_ENTRIES} entries; list a subdirectory or use a smaller depth]")
    return ToolResult(True, "\n".join(lines), {"entries": count, "truncated": truncated})


# ---------------------------------------------------------------------- find_files
def _walk_files(ws: Workspace, root: Path) -> list[Path]:
    """All files under root (sorted), skipping ignored directories."""
    found: list[Path] = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(d for d in dirnames if not _skip_dir(d))
        found.extend(Path(dirpath) / f for f in sorted(filenames))
    return found


@never_raises
def find_files(ws: Workspace, cfg: Any, pattern: str = "") -> ToolResult:
    """Files whose repo-relative path or file name matches a glob, at most 100 results."""
    pattern = str(pattern or "").strip()
    if not pattern:
        return ToolResult(False, "pattern is required, e.g. '*.py' or '**/*config*'.")
    variants = {pattern, pattern[3:] if pattern.startswith("**/") else pattern}
    matches: list[str] = []
    for file in _walk_files(ws, ws.repo_root):
        rel = ws.rel(file)
        if any(fnmatch.fnmatch(rel, v) or fnmatch.fnmatch(file.name, v) for v in variants):
            matches.append(rel)
    if not matches:
        return ToolResult(True, f"No files match '{pattern}'.", {"count": 0})
    shown = matches[:MAX_FIND_RESULTS]
    out = "\n".join(shown)
    if len(matches) > len(shown):
        out += f"\n... {len(matches) - len(shown)} more files; use a more specific pattern"
    return ToolResult(True, out, {"count": len(matches)})


# ---------------------------------------------------------------------- search_code
def _format_matches(matches: list[tuple[str, int, str]], total: int, query: str) -> ToolResult:
    """Render search hits as 'path:line: text' with the 50-match cap."""
    if total == 0:
        return ToolResult(True, f"No matches for '{query}'. Try a shorter or different term.", {"count": 0})
    lines = []
    for rel, lineno, text in matches[:MAX_SEARCH_MATCHES]:
        text = text.strip()
        if len(text) > MAX_MATCH_TEXT:
            text = text[:MAX_MATCH_TEXT] + "..."
        lines.append(f"{rel}:{lineno}: {text}")
    if total > MAX_SEARCH_MATCHES:
        lines.append(f"... {total - MAX_SEARCH_MATCHES} more matches; narrow the query")
    return ToolResult(True, "\n".join(lines), {"count": total})


def _search_rg(ws: Workspace, rg: str, query: str, regex: bool, root: Path, file_glob: str | None) -> ToolResult:
    """Search with ripgrep; exit code 1 means no matches, 2 means an error."""
    parts = [rg, "--line-number", "--no-heading", "--color", "never", "--max-columns", "300", "--hidden"]
    if not regex:
        parts.append("-F")
    for name in sorted(IGNORED_DIRS):
        parts += ["-g", f"!{name}/"]
    if file_glob:
        parts += ["-g", file_glob]
    parts += ["--", query, str(root)]
    res = run_process(" ".join(shlex.quote(p) for p in parts), ws.repo_root, SEARCH_TIMEOUT_S)
    if res.timed_out:
        return ToolResult(False, f"Search timed out after {SEARCH_TIMEOUT_S:.0f}s; narrow the path or query.")
    if res.exit_code not in (0, 1):
        return ToolResult(False, f"Search failed: {res.output.strip()[-500:]}")
    matches: list[tuple[str, int, str]] = []
    for line in res.output.splitlines():
        m = _RG_LINE.match(line)
        if not m:
            continue
        path = Path(m.group(1))
        path = path if path.is_absolute() else root / path if root.is_dir() else root
        try:
            rel = ws.rel(path)
        except ValueError:
            rel = m.group(1)
        matches.append((rel, int(m.group(2)), m.group(3)))
    matches.sort(key=lambda t: (t[0], t[1]))
    return _format_matches(matches, len(matches), query)


def _search_python(ws: Workspace, query: str, regex: bool, root: Path, file_glob: str | None) -> ToolResult:
    """Pure-Python search over text files (used when ripgrep is not installed)."""
    try:
        pattern = re.compile(query) if regex else None
    except re.error as e:
        return ToolResult(False, f"Invalid regex: {e}. Set regex=false for a literal search.")
    files = [root] if root.is_file() else [p for p in _walk_files(ws, root)]
    matches: list[tuple[str, int, str]] = []
    total = 0
    for file in files:
        rel = ws.rel(file)
        if file_glob and not (fnmatch.fnmatch(rel, file_glob) or fnmatch.fnmatch(file.name, file_glob)):
            continue
        try:
            if file.stat().st_size > 1_000_000:
                continue
            with file.open("rb") as fh:
                if b"\0" in fh.read(8192):
                    continue
            text = file.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        for lineno, line in enumerate(text.splitlines(), 1):
            if (pattern.search(line) if pattern else query in line):
                total += 1
                if len(matches) < MAX_SEARCH_MATCHES:
                    matches.append((rel, lineno, line))
    return _format_matches(matches, total, query)


@never_raises
def search_code(ws: Workspace, cfg: Any, query: str = "", regex: Any = False, path: str = ".",
                file_glob: str | None = None) -> ToolResult:
    """Search file contents (literal by default), returning 'path:line: text', at most 50 matches."""
    query = str(query or "")
    if not query.strip():
        return ToolResult(False, "query is required.")
    use_regex = regex is True or str(regex).lower() == "true"
    root = ws.resolve(path or ".")
    if not root.exists():
        return ToolResult(False, f"Path not found: {path}")
    rg = shutil.which("rg")
    if rg:
        return _search_rg(ws, rg, query, use_regex, root, file_glob or None)
    return _search_python(ws, query, use_regex, root, file_glob or None)


# ---------------------------------------------------------------------- view_file
@never_raises
def view_file(ws: Workspace, cfg: Any, path: str = "", start_line: Any = 1, end_line: Any = None) -> ToolResult:
    """Show a file with line numbers, at most 250 lines per call, with a paging hint."""
    if not str(path or "").strip():
        return ToolResult(False, "path is required.")
    target = ws.resolve(path)
    if target.is_dir():
        return ToolResult(False, f"{ws.rel(target)} is a directory; use list_dir.")
    text = ws.read_text(target)
    lines = text.splitlines()
    total = len(lines)
    rel = ws.rel(target)
    if total == 0:
        return ToolResult(True, f"File: {rel} (0 lines)\n(empty file)", {"total": 0})
    start = max(1, as_int(start_line, "start_line", 1))
    if start > total:
        return ToolResult(False, f"start_line {start} is past the end of {rel} ({total} lines).")
    end = as_int(end_line, "end_line", None)
    end = total if end is None else min(end, total)
    if end < start:
        return ToolResult(False, f"end_line ({end}) must be >= start_line ({start}).")
    end = min(end, start + MAX_VIEW_LINES - 1)
    body = []
    for n in range(start, end + 1):
        line = lines[n - 1]
        if len(line) > MAX_LINE_CHARS:
            line = line[:MAX_LINE_CHARS] + f"...[+{len(line) - MAX_LINE_CHARS} chars]"
        body.append(f"{n:>5} | {line}")
    out = [f"File: {rel} ({total} lines) — showing {start}-{end}", *body]
    if end < total:
        out.append(f"[{total - end} more lines — call view_file with start_line={end + 1}]")
    return ToolResult(True, "\n".join(out), {"total": total, "start": start, "end": end})

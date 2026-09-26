"""Symbol outlines per file and the repo_map tool."""
from __future__ import annotations

import ast
import re
from pathlib import Path
from typing import Any

from harness.types import ToolResult

DEFAULT_MAP_CHARS = 6000

PY_EXTS = {".py"}
JS_EXTS = {".js", ".jsx", ".ts", ".tsx", ".mjs", ".cjs"}
GO_EXTS = {".go"}
JVM_EXTS = {".java", ".kt", ".kts", ".cs"}
RUST_EXTS = {".rs"}
RUBY_EXTS = {".rb"}
OUTLINE_EXTS = PY_EXTS | JS_EXTS | GO_EXTS | JVM_EXTS | RUST_EXTS | RUBY_EXTS

_NOT_METHODS = {"if", "for", "while", "switch", "catch", "function", "return", "with", "else", "do", "try",
                "new", "typeof", "super", "constructor"}

# (compiled regex, formatter(match) -> label). Nesting comes from the line's indentation.
_Rule = tuple  # (re.Pattern, callable)

_JS_RULES: list[_Rule] = [
    (re.compile(r"^\s*(?:export\s+)?(?:default\s+)?(?:async\s+)?function\s*\*?\s*(\w+)"),
     lambda m: f"function {m.group(1)}"),
    (re.compile(r"^\s*(?:export\s+)?(?:default\s+)?(?:abstract\s+)?class\s+(\w+)"), lambda m: f"class {m.group(1)}"),
    (re.compile(r"^\s*(?:export\s+)?(?:const|let|var)\s+(\w+)\s*=\s*(?:async\s+)?(?:\([^)]*\)|\w+)\s*(?::\s*[^=]+)?=>"),
     lambda m: f"const {m.group(1)}"),
    (re.compile(r"^\s*(?:export\s+)?(?:interface|type|enum)\s+(\w+)"), lambda m: f"type {m.group(1)}"),
    (re.compile(r"^\s+(?:(?:public|private|protected|static|async|readonly|override)\s+)*(?:get\s+|set\s+)?"
                r"(\w+)\s*\([^)]*\)\s*(?::\s*[^{]+)?\{"),
     lambda m: None if m.group(1) in _NOT_METHODS else f"{m.group(1)}()"),
]
_GO_RULES: list[_Rule] = [
    (re.compile(r"^func\s+(\([^)]*\)\s*)?(\w+)"),
     lambda m: f"func {m.group(1).strip() + ' ' if m.group(1) else ''}{m.group(2)}"),
    (re.compile(r"^type\s+(\w+)\s+(struct|interface)"), lambda m: f"type {m.group(1)} {m.group(2)}"),
]
_JVM_RULES: list[_Rule] = [
    (re.compile(r"^\s*(?:(?:public|private|protected|internal|abstract|final|static|sealed|data|open|partial|"
                r"inner|enum|annotation)\s+)*(class|interface|enum|record|object)\s+(\w+)"),
     lambda m: f"{m.group(1)} {m.group(2)}"),
    (re.compile(r"^\s*(?:(?:override|private|public|internal|protected|suspend|open|inline|operator)\s+)*fun\s+"
                r"(?:<[^>]*>\s*)?(?:[\w.]+\.)?(\w+)\s*\("), lambda m: f"fun {m.group(1)}"),
    (re.compile(r"^\s+(?:(?:public|private|protected|internal|static|final|abstract|synchronized|override|"
                r"virtual|async)\s+)+[\w<>\[\],.?]+\s+(\w+)\s*\("),
     lambda m: None if m.group(1) in _NOT_METHODS else f"{m.group(1)}()"),
]
_RUST_RULES: list[_Rule] = [
    (re.compile(r"^\s*(?:pub(?:\([^)]*\))?\s+)?(?:const\s+)?(?:async\s+)?(?:unsafe\s+)?fn\s+(\w+)"),
     lambda m: f"fn {m.group(1)}"),
    (re.compile(r"^\s*(?:pub(?:\([^)]*\))?\s+)?(struct|enum|trait)\s+(\w+)"), lambda m: f"{m.group(1)} {m.group(2)}"),
    (re.compile(r"^\s*impl(?:<[^>]*>)?\s+([\w:<>, ]+?)(?:\s+for\s+([\w:<>]+))?\s*(?:where\b.*)?\{"),
     lambda m: f"impl {m.group(1)}" + (f" for {m.group(2)}" if m.group(2) else "")),
]
_RUBY_RULES: list[_Rule] = [
    (re.compile(r"^\s*(class|module)\s+([\w:]+)"), lambda m: f"{m.group(1)} {m.group(2)}"),
    (re.compile(r"^\s*def\s+(self\.)?([\w?!=]+)"), lambda m: f"def {m.group(1) or ''}{m.group(2)}"),
]
_PY_RULES: list[_Rule] = [
    (re.compile(r"^\s*class\s+(\w+)"), lambda m: f"class {m.group(1)}"),
    (re.compile(r"^\s*(async\s+)?def\s+(\w+)"), lambda m: f"{'async def' if m.group(1) else 'def'} {m.group(2)}"),
]


def _rules_for(suffix: str) -> list[_Rule]:
    """Regex outline rules for a file extension."""
    if suffix in JS_EXTS:
        return _JS_RULES
    if suffix in GO_EXTS:
        return _GO_RULES
    if suffix in JVM_EXTS:
        return _JVM_RULES
    if suffix in RUST_EXTS:
        return _RUST_RULES
    if suffix in RUBY_EXTS:
        return _RUBY_RULES
    if suffix in PY_EXTS:
        return _PY_RULES
    return []


def _regex_outline(text: str, rules: list[_Rule]) -> list[str]:
    """Line-by-line regex outline; indented symbols are shown one level deeper."""
    out: list[str] = []
    for lineno, line in enumerate(text.splitlines(), 1):
        for pattern, fmt in rules:
            m = pattern.match(line)
            if not m:
                continue
            label = fmt(m)
            if label:
                indent = "  " if line[:1] in (" ", "\t") else ""
                out.append(f"{indent}{label} (L{lineno})")
            break
    return out


def _python_outline(text: str) -> list[str]:
    """Classes (with their methods) and top-level functions via ast."""
    tree = ast.parse(text)
    out: list[str] = []

    def label(node: ast.AST) -> str:
        prefix = "async def" if isinstance(node, ast.AsyncFunctionDef) else "def"
        return f"{prefix} {node.name} (L{node.lineno})"

    def visit_class(node: ast.ClassDef, depth: int) -> None:
        out.append(f"{'  ' * depth}class {node.name} (L{node.lineno})")
        for child in node.body:
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                out.append(f"{'  ' * (depth + 1)}{label(child)}")
            elif isinstance(child, ast.ClassDef):
                visit_class(child, depth + 1)

    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            visit_class(node, 0)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            out.append(label(node))
    return out


def outline_file(path: str | Path, text: str) -> list[str]:
    """Symbol outline lines for one file ('class Foo (L10)', '  def bar (L12)', ...)."""
    suffix = Path(path).suffix.lower()
    if suffix in PY_EXTS:
        try:
            return _python_outline(text)
        except (SyntaxError, ValueError):
            return _regex_outline(text, _PY_RULES)
    return _regex_outline(text, _rules_for(suffix))


def build_repo_map(ws: Any, paths: list[str] | None = None, max_chars: int = DEFAULT_MAP_CHARS) -> str:
    """Outline of the given files (in priority order), or of every outlinable file; cut at max_chars."""
    if paths is None:
        paths = [ws.rel(p) for p in ws.iter_source_files() if p.suffix.lower() in OUTLINE_EXTS]
    chunks: list[str] = []
    used = 0
    for i, rel in enumerate(paths):
        try:
            text = ws.read_text(rel)
        except ValueError:
            continue
        block = "\n".join([f"{rel}:"] + [f"  {line}" for line in outline_file(rel, text)])
        if used + len(block) + 1 > max_chars:
            remaining = len(paths) - i
            chunks.append(f"... [repo map truncated; {remaining} more file(s) not shown — call repo_map on a "
                          f"subdirectory or file]")
            break
        chunks.append(block)
        used += len(block) + 1
    return "\n".join(chunks)


def repo_map(ws: Any, cfg: Any, path: str = ".") -> ToolResult:
    """Tool: outline of classes and functions (with line numbers) for a file or directory."""
    try:
        target = ws.resolve(path or ".")
        if not target.exists():
            return ToolResult(False, f"Path not found: {path}")
        limit = DEFAULT_MAP_CHARS
        cap = getattr(getattr(cfg, "context", None), "max_tool_output_chars", None)
        if cap:
            limit = min(limit, int(cap))
        if target.is_file():
            rel = ws.rel(target)
            lines = outline_file(rel, ws.read_text(target))
            body = "\n".join([f"{rel}:"] + [f"  {line}" for line in lines])
            if not lines:
                body += "\n  (no classes or functions found)"
            return ToolResult(True, body[:limit])
        files = [ws.rel(p) for p in ws.iter_source_files()
                 if p.suffix.lower() in OUTLINE_EXTS and (p == target or target in p.parents)]
        if not files:
            return ToolResult(True, f"No source files with outlines under {ws.rel(target)}.")
        return ToolResult(True, build_repo_map(ws, files, limit), {"files": len(files)})
    except ValueError as e:
        return ToolResult(False, str(e))
    except Exception as e:  # noqa: BLE001 - tools must never raise
        return ToolResult(False, f"repo_map failed: {type(e).__name__}: {e}")
